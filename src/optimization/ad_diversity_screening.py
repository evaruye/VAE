# -*- coding: utf-8 -*-
"""
VAE乳液 Round3 多目标优化：Practical AD / OOD 后筛选 + 配方多样性重排 V4
================================================================================

V4 相比 V3 的关键修改：
1) OOD / AD 特征空间改为“四个冻结代理模型真实使用特征的并集”，并显式剔除未参与训练的 _isna 列。
   - VC / SC / EC 通常为 39 个特征
   - HCHO 通常为 40 个特征
   - OOD 使用 union 后的模型输入特征，避免把中间处理列误纳入 AD 距离计算。
2) 新增配方多样性重排：
   - 构造 formula_family = 氧化剂组合 + 还原剂组合 + 第三单体组合 + 乳化剂组合 + 保护胶体组合
   - 计算候选之间的 mixed/Gower-like distance（类别原料 + 数值用量/工艺）
   - 从 Practical AD = Inside 且机制规则全通过的候选池中进行 greedy diversity selection
3) 输出新的 Diverse-AD-C1 ~ Diverse-AD-C8，并保留与“分数优先 Top-8”的对照。

注意：
- 本脚本不是重新运行 NSGA-II；它是在 NSGA-II 已生成候选池后进行 AD-aware + diversity-aware post-screening。
- Mahalanobis 距离在高维、小样本、强相关特征中较保守，本脚本保留其为 strict AD 风险指标，Practical AD 不将其作为唯一硬排除项。

运行：
python vae_ood_ad_diverse_candidate_selector_v4.py

输出：
round3_AD_constrained_diverse_candidate_selection_v4.xlsx
round3_AD_constrained_diverse_selected_candidates_v4.csv
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import xlsxwriter  # noqa: F401
    EXCEL_ENGINE = "xlsxwriter"
except Exception:
    EXCEL_ENGINE = "openpyxl"

# =============================================================================
# 0. 路径配置
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models"
BASE_DIR = REPO_ROOT / "results" / "nsga2" / "round3_frontier"
OUTPUT_DIR = REPO_ROOT / "results" / "ad_gower"

# 候选池文件
CAND_X_MASTER_FULL = "round3_frontier_recommended_X_master_full_v1.xlsx"
CAND_X_MASTER_FULL_ALT = "round3_frontier_nsga2_recommended_X_master_full.xlsx"
CAND_RECOMMENDED = "round3_frontier_nsga2_recommended_candidates.csv"
CAND_RECOMMENDED_ALT = "round3_frontier_recommended_candidates_v1.csv"
CAND_RAW_FORMULA_FULL = "round3_frontier_recommended_raw_formula_full_v1.xlsx"
CAND_RAW_FORMULA_FULL_ALT = "round3_frontier_nsga2_recommended_raw_formula_full.xlsx"

# 参考训练/历史分布
REFERENCE_WORKBOOK = DATA_DIR / "X_reduced_final.xlsx"
REFERENCE_SHEET = None
X_REDUCED_FINAL = DATA_DIR / "X_reduced_final.xlsx"

# 可选模型文件：用于读取真实 feature_names
MODEL_PATHS = {
    "VC": MODEL_DIR / "VC_CatBoost.joblib",
    "SC": MODEL_DIR / "SC_XGBoost.joblib",
    "EC": MODEL_DIR / "EC_CatBoost.joblib",
    "HCHO": MODEL_DIR / "HCHO_MLP.joblib",
}

# 机制规则
RULE_FILE = "candidate_mechanism_rules.xlsx"
RULE_SHEET = "rules"


# 内置 19 条机理一致性规则：当 candidate_mechanism_rules.xlsx 不在脚本目录时自动使用。
# 这样 V4 可以独立运行，不再因为缺少规则文件中断。
DEFAULT_MECH_RULES = [
    {"rule_id": "R01_HCHO_A_grade", "feature": "HCHO_pred_ppm", "low": 0.0, "high": 1.5, "target": "HCHO", "rule_name": "A-grade predicted free HCHO", "mechanism_note": "Final candidates should remain within the A-grade/below-LOD-equivalent HCHO window used as the Round-3 hard gate.", "rule_type": "hard_gate", "basis": "Round-3 HCHO engineering gate"},
    {"rule_id": "R02_HCHO_unclipped", "feature": "HCHO_pred_clipped_zero", "low": 0.0, "high": 0.0, "target": "HCHO", "rule_name": "Unclipped HCHO prediction", "mechanism_note": "Rejects candidates whose HCHO value is artificially obtained by clipping a negative raw prediction to zero; this avoids over-interpreting numerical artifacts.", "rule_type": "diagnostic_gate", "basis": "HCHO model diagnostic"},
    {"rule_id": "R03_low_reducer_HCHO_risk", "feature": "red_hcho_potential_share", "low": 0.0, "high": 0.0, "target": "HCHO", "rule_name": "Zero HCHO-risk reducer share", "mechanism_note": "Consistent with the HCHO source-control conclusion: the reducer family should avoid formaldehyde-generating risk contribution.", "rule_type": "mechanism_gate", "basis": "HCHO SHAP/PDP + Round-3 structure rule"},
    {"rule_id": "R04_low_third_monomer_HCHO_risk", "feature": "ter_hcho_potential_share", "low": 0.0, "high": 0.0, "target": "HCHO", "rule_name": "Zero HCHO-risk third-monomer share", "mechanism_note": "Consistent with the source-level HCHO suppression strategy by avoiding HCHO-risk third-monomer contribution.", "rule_type": "mechanism_gate", "basis": "HCHO SHAP/PDP + Round-3 structure rule"},
    {"rule_id": "R05_third_monomer_dosage_gate", "feature": "ter_total_pct_raw", "low": 0.0, "high": 6.0, "target": "HCHO/SC/EC", "rule_name": "Third-monomer total dosage within Round-3 window", "mechanism_note": "Keeps third-monomer dosage in the constrained region used to suppress HCHO risk while maintaining product-property trade-offs.", "rule_type": "feasibility_gate", "basis": "Round-3 input/structure constraint"},
    {"rule_id": "R06_oxidizer_electrophilicity_gate", "feature": "oxid_mean_f+", "low": 0.0, "high": 0.15, "target": "HCHO", "rule_name": "Oxidizer mean f+ below HCHO-risk threshold", "mechanism_note": "Consistent with the HCHO interpretability result that overly high oxidizer electrophilicity-related descriptor can increase unfavorable HCHO-risk response.", "rule_type": "mechanism_gate", "basis": "HCHO SHAP/PDP + Round-3 structure rule"},
    {"rule_id": "R07_protective_colloid_dosage_gate", "feature": "pc_total_pct_raw", "low": 0.0, "high": 8.0, "target": "Feasibility/VC/SC", "rule_name": "Protective-colloid total dosage within engineering limit", "mechanism_note": "Avoids unrealistic protective-colloid dosage and supports viscosity/emulsion-stability feasibility.", "rule_type": "feasibility_gate", "basis": "Round-3 engineering constraint"},
    {"rule_id": "R08_emulsifier_dosage_window", "feature": "emul_total_pct_raw", "low": 3.0, "high": 17.5, "target": "SC/HCHO/Feasibility", "rule_name": "Emulsifier total dosage within Round-3 feasible window", "mechanism_note": "Keeps emulsifier dosage within the explored feasible region while supporting solids content and low-HCHO trade-off.", "rule_type": "feasibility_gate", "basis": "Round-3 input bound"},
    {"rule_id": "R09_pc_DP_supported_region", "feature": "pc_mean_DP", "low": 1200.0, "high": 2000.0, "target": "VC/SC/HCHO", "rule_name": "Protective-colloid DP in supported PDP/mechanistic region", "mechanism_note": "Places the weighted protective-colloid polymerization degree in a supported region that is consistent with viscosity control and emulsion stabilization.", "rule_type": "PDP_supported_region", "basis": "VC/HCHO PDP soft preference"},
    {"rule_id": "R10_pc_hydrolysis_real_grade", "feature": "pc_mean_degree_of_hydrolysis", "low": 80.0, "high": 98.0, "target": "VC/EC/SC", "rule_name": "Protective-colloid hydrolysis degree in real-grade region", "mechanism_note": "Avoids pseudo-continuous extrapolation of protective-colloid grades; values should correspond to realistic PVA/HEC-grade combinations.", "rule_type": "real_material_grade", "basis": "Feature-type interpretation + PDP caution"},
    {"rule_id": "R11_poly_pressure_round3_window", "feature": "poly_pressure", "low": 5.8, "high": 8.5, "target": "EC", "rule_name": "Polymerization pressure in Round-3 EC-supported window", "mechanism_note": "Consistent with EC-oriented design: pressure is kept within the high-end window used for ethylene incorporation.", "rule_type": "process_window", "basis": "EC PDP + Round-3 input bound"},
    {"rule_id": "R12_poly_temp_round3_window", "feature": "poly_temp", "low": 60.0, "high": 88.0, "target": "SC/VC", "rule_name": "Polymerization temperature in feasible Round-3 window", "mechanism_note": "Maintains polymerization temperature within the historical/mechanistically feasible region for solids content and viscosity balance.", "rule_type": "process_window", "basis": "SC PDP + Round-3 input bound"},
    {"rule_id": "R13_time_round3_window", "feature": "time", "low": 240.0, "high": 330.0, "target": "EC/SC", "rule_name": "Reaction time in Round-3 high-end window", "mechanism_note": "Consistent with the EC/SC high-end design window while avoiding low-time under-polymerization risk.", "rule_type": "process_window", "basis": "EC/SC PDP + Round-3 input bound"},
    {"rule_id": "R14_coinitiator_supported_region", "feature": "coinitiator_pct", "low": 0.0015, "high": 0.0035, "target": "SC/VC", "rule_name": "Coinitiator dosage in supported SC/VC window", "mechanism_note": "Uses a moderate coinitiator dosage range that was previously selected as a PDP-supported region for balancing solid content and viscosity.", "rule_type": "PDP_supported_region", "basis": "SC/VC PDP soft preference"},
    {"rule_id": "R15_APEO_free", "feature": "APEO_free", "low": 1.0, "high": 1.0, "target": "Sustainability", "rule_name": "APEO-free formulation", "mechanism_note": "Ensures the selected formulation satisfies the sustainability-oriented APEO-free requirement.", "rule_type": "hard_gate", "basis": "Round-3 formulation constraint"},
    {"rule_id": "R16_SC_window_pass", "feature": "pass_SC", "low": 1.0, "high": 1.0, "target": "SC", "rule_name": "Solid-content prediction passes target window", "mechanism_note": "Confirms that the candidate satisfies the Round-3 solid-content target window.", "rule_type": "target_pass", "basis": "Round-3 target window"},
    {"rule_id": "R17_VC_window_pass", "feature": "pass_VC", "low": 1.0, "high": 1.0, "target": "VC", "rule_name": "Viscosity prediction passes target window", "mechanism_note": "Confirms that the candidate does not sacrifice viscosity while reducing HCHO.", "rule_type": "target_pass", "basis": "Round-3 target window"},
    {"rule_id": "R18_EC_window_pass", "feature": "pass_EC", "low": 1.0, "high": 1.0, "target": "EC", "rule_name": "Ethylene-content prediction passes target window", "mechanism_note": "Confirms that the candidate maintains ethylene content within the targeted performance range.", "rule_type": "target_pass", "basis": "Round-3 target window"},
    {"rule_id": "R19_feasible_candidate", "feature": "is_feasible", "low": 1.0, "high": 1.0, "target": "Feasibility", "rule_name": "All Round-3 feasibility constraints passed", "mechanism_note": "Ensures the final candidate is not selected from an infeasible region of the optimization search.", "rule_type": "hard_gate", "basis": "NSGA-II feasibility flag"},
]

# 输出
OUT_XLSX = "round3_AD_constrained_diverse_candidate_selection_v4.xlsx"
OUT_CSV = "round3_AD_constrained_diverse_selected_candidates_v4.csv"
OUT_PY_COPY = "vae_ood_ad_diverse_candidate_selector_v4.py"

TOP_N = 8
RANDOM_STATE = 42

# 显式排除：未进入模型训练的中间缺失指示列
EXCLUDE_OOD_COLS = [
    "oxidizer_pct2_isna",
    "time_isna",
    "pH_buffer_pct_isna",
]

Q_LOW = 0.01
Q_HIGH = 0.99

# Strict AD 阈值：非常保守
STRICT_NN5_INSIDE = 95.0
STRICT_MAHAL_INSIDE = 95.0
STRICT_PCA_INSIDE = 95.0
STRICT_NN5_BORDER = 99.0
STRICT_MAHAL_BORDER = 99.0
STRICT_PCA_BORDER = 99.0

# Practical AD 阈值：用于多目标优化后筛选
PRACTICAL_Q_OUTSIDE_MAX_INSIDE = 2
PRACTICAL_Q_OUTSIDE_MAX_BORDER = 3
PRACTICAL_NN5_INSIDE = 99.1
PRACTICAL_NN5_BORDER = 99.6
PRACTICAL_PCA_INSIDE = 95.0
PRACTICAL_PCA_BORDER = 99.0

# 多样性选择参数
DIVERSITY_DISTANCE_THRESHOLDS = [0.32, 0.28, 0.24, 0.20, 0.16, 0.12, 0.08, 0.00]
MAX_PER_FORMULA_FAMILY_SEQUENCE = [1, 2, 3, 999]
CATEGORY_DISTANCE_WEIGHT = 0.55
NUMERIC_DISTANCE_WEIGHT = 0.45

# 原料类别列
RAW_CAT_COLS = (
    [f"oxidizer_type{i}" for i in range(1, 3)]
    + [f"reducer_type{i}" for i in range(1, 4)]
    + [f"third_monomer_type{i}" for i in range(1, 8)]
    + [f"emulsifier_type{i}" for i in range(1, 5)]
    + [f"protective_colloid_type{i}" for i in range(1, 4)]
    + ["initiator_feed_mode"]
)
RAW_NUM_COLS = (
    [f"oxidizer_pct{i}" for i in range(1, 3)]
    + [f"reducer_pct{i}" for i in range(1, 4)]
    + [f"third_monomer_pct{i}" for i in range(1, 8)]
    + [f"emulsifier_pct{i}" for i in range(1, 5)]
    + [f"protective_colloid_pct{i}" for i in range(1, 4)]
    + ["coinitiator_pct", "init_pressure", "init_temp", "poly_pressure", "poly_temp", "time", "dispersant_pct", "pH_buffer_pct"]
)

FAMILY_GROUPS = {
    "oxidizer_sig": [f"oxidizer_type{i}" for i in range(1, 3)],
    "reducer_sig": [f"reducer_type{i}" for i in range(1, 4)],
    "third_monomer_sig": [f"third_monomer_type{i}" for i in range(1, 8)],
    "emulsifier_sig": [f"emulsifier_type{i}" for i in range(1, 5)],
    "protective_colloid_sig": [f"protective_colloid_type{i}" for i in range(1, 4)],
}

# 额外展示列
PRED_COLS = ["VC_pred_raw", "SC_pred", "EC_pred", "HCHO_pred_ppm"]
DISPLAY_COLS = [
    "round_score", "nsga2_rank", "nsga2_crowding", "is_feasible",
    "HCHO_grade", "HCHO_pred_ppm", "VC_pred_raw", "SC_pred", "EC_pred",
    "ter_total_pct_raw", "emul_total_pct_raw", "pc_total_pct_raw", "coinitiator_pct",
    "poly_pressure", "poly_temp", "time", "APEO_free",
]

# =============================================================================
# 1. 基础工具
# =============================================================================

def find_existing(*names: str) -> Path:
    for name in names:
        p = BASE_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find any of: " + ", ".join(names))


def read_table(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name if sheet_name else 0)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "0", "0.0"}:
        return ""
    return s


def group_signature(row: pd.Series, cols: List[str]) -> str:
    vals = []
    for c in cols:
        v = norm_text(row.get(c, ""))
        if v:
            vals.append(v)
    vals = sorted(vals)
    return "+".join(vals) if vals else "none"


def classify_hcho_grade(x: float) -> str:
    if pd.isna(x):
        return "NA"
    if x <= 1.5:
        return "A"
    if x <= 5:
        return "B"
    if x <= 30:
        return "C"
    return "Reject"


def safe_sheet_name(name: str) -> str:
    for ch in ["\\", "/", "*", "?", ":", "[", "]"]:
        name = name.replace(ch, "_")
    return name[:31]

# =============================================================================
# 2. 读取模型真实特征名
# =============================================================================

def unwrap_predictor(model):
    if isinstance(model, dict):
        for key in ["reg", "model", "best_model", "final_model", "pipeline"]:
            if key in model and hasattr(model[key], "predict"):
                return model[key]
        for value in model.values():
            if hasattr(value, "predict"):
                return value
    return model


def get_feature_names_from_model(path: Path) -> Optional[List[str]]:
    try:
        import joblib
        model = joblib.load(path)
        m = unwrap_predictor(model)
        if hasattr(m, "feature_names_in_"):
            return [str(c) for c in list(m.feature_names_in_)]
        if hasattr(m, "feature_names_"):
            return [str(c) for c in list(m.feature_names_)]
        if hasattr(m, "named_steps"):
            for _, step in reversed(list(m.named_steps.items())):
                if hasattr(step, "feature_names_in_"):
                    return [str(c) for c in list(step.feature_names_in_)]
            for _, step in m.named_steps.items():
                if hasattr(step, "feature_names_in_"):
                    return [str(c) for c in list(step.feature_names_in_)]
    except Exception as e:
        print(f"WARNING: cannot read feature names from {path.name}: {e}")
    return None


def determine_ood_feature_cols(candidate_x: pd.DataFrame, reference: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    x_reduced_path = Path(X_REDUCED_FINAL)
    x_reduced_cols = pd.read_excel(x_reduced_path, nrows=1).columns.astype(str).tolist()

    model_rows = []
    union = []
    for target, p in MODEL_PATHS.items():
        p = Path(p)
        cols = None
        if p.exists():
            cols = get_feature_names_from_model(p)
        if cols is None:
            cols = [c for c in x_reduced_cols if c not in EXCLUDE_OOD_COLS]
            source = "fallback_X_reduced_excluding_isna"
        else:
            source = "model_file_feature_names"
        cols = [str(c) for c in cols if str(c) not in EXCLUDE_OOD_COLS]
        for c in cols:
            if c not in union:
                union.append(c)
        model_rows.append({
            "target": target,
            "source": source,
            "n_features_raw": len(cols),
            "features": "; ".join(cols),
        })

    # 按 X_reduced_final 原顺序排序，其后补模型中特有列
    preferred = [c for c in x_reduced_cols if c in union and c not in EXCLUDE_OOD_COLS]
    extras = [c for c in union if c not in preferred and c not in EXCLUDE_OOD_COLS]
    feature_cols = preferred + extras

    # 必须同时存在于候选和参考中，并且参考分布有效
    final_cols = []
    for c in feature_cols:
        if c in candidate_x.columns and c in reference.columns:
            s = to_num(reference[c])
            if s.notna().sum() >= 5 and s.nunique(dropna=True) > 1:
                final_cols.append(c)
    feat_info = pd.DataFrame(model_rows)
    return final_cols, feat_info

# =============================================================================
# 3. OOD / AD 计算
# =============================================================================

def compute_reference_stats(reference: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        s = to_num(reference[c]).dropna()
        rows.append({
            "feature": c,
            "n_ref": len(s),
            "ref_min": s.min(),
            "ref_q01": s.quantile(Q_LOW),
            "ref_q05": s.quantile(0.05),
            "ref_median": s.median(),
            "ref_q95": s.quantile(0.95),
            "ref_q99": s.quantile(Q_HIGH),
            "ref_max": s.max(),
            "ref_mean": s.mean(),
            "ref_std": s.std(ddof=1),
        })
    return pd.DataFrame(rows)


def feature_range_detail(candidate_x: pd.DataFrame, ref_stats: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    stats = ref_stats.set_index("feature")
    rows = []
    for i, row in candidate_x.iterrows():
        cand = row.get("Candidate", f"K{i+1}")
        cid = row.get("_candidate_id", "")
        for c in feature_cols:
            val = pd.to_numeric(pd.Series([row.get(c, np.nan)]), errors="coerce").iloc[0]
            st = stats.loc[c]
            if pd.isna(val):
                status = "missing"
            elif val < st["ref_min"] or val > st["ref_max"]:
                status = "outside_minmax"
            elif val < st["ref_q01"] or val > st["ref_q99"]:
                status = "inside_minmax_but_outside_qrange"
            else:
                status = "inside_qrange"
            denom = st["ref_max"] - st["ref_min"]
            margin = np.nan
            if pd.notna(val) and pd.notna(denom) and denom != 0:
                margin = min((val - st["ref_min"]) / denom, (st["ref_max"] - val) / denom)
            rows.append({
                "Candidate": cand,
                "_candidate_id": cid,
                "feature": c,
                "candidate_value": val,
                "ref_min": st["ref_min"],
                "ref_q01": st["ref_q01"],
                "ref_median": st["ref_median"],
                "ref_q99": st["ref_q99"],
                "ref_max": st["ref_max"],
                "range_status": status,
                "outside_minmax": int(status == "outside_minmax"),
                "outside_qrange": int(status in ["outside_minmax", "inside_minmax_but_outside_qrange"]),
                "minmax_margin_ratio": margin,
            })
    return pd.DataFrame(rows)


def percentile_rank(x: float, ref_arr: np.ndarray) -> float:
    return float(100.0 * np.mean(ref_arr <= x))


def compute_multivariate_ood(reference: pd.DataFrame, candidate_x: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_num = reference[feature_cols].apply(pd.to_numeric, errors="coerce")
    cand_num = candidate_x[feature_cols].apply(pd.to_numeric, errors="coerce")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_ref = pipe.fit_transform(ref_num)
    X_cand = pipe.transform(cand_num)
    n_ref, n_feat = X_ref.shape

    # Nearest-neighbor reference distribution based on leave-one-out distances
    nn = NearestNeighbors(n_neighbors=min(6, n_ref), metric="euclidean")
    nn.fit(X_ref)
    dist_ref, _ = nn.kneighbors(X_ref)
    ref_nn1 = dist_ref[:, 1] if dist_ref.shape[1] > 1 else dist_ref[:, 0]
    ref_nn5 = dist_ref[:, 1:min(6, dist_ref.shape[1])].mean(axis=1) if dist_ref.shape[1] > 1 else dist_ref[:, 0]
    dist_cand, idx_cand = nn.kneighbors(X_cand)
    cand_nn1 = dist_cand[:, 0]
    cand_nn5 = dist_cand[:, :min(5, dist_cand.shape[1])].mean(axis=1)

    # Ledoit-Wolf Mahalanobis
    lw = LedoitWolf().fit(X_ref)
    ref_mahal = lw.mahalanobis(X_ref)
    cand_mahal = lw.mahalanobis(X_cand)

    # PCA score distance
    n_comp = min(10, n_feat, max(1, n_ref - 1))
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_ref_pca = pca.fit_transform(X_ref)
    X_cand_pca = pca.transform(X_cand)
    center = X_ref_pca.mean(axis=0)
    ref_pca_dist = np.linalg.norm(X_ref_pca - center, axis=1)
    cand_pca_dist = np.linalg.norm(X_cand_pca - center, axis=1)

    # Isolation Forest
    iso = IsolationForest(n_estimators=500, contamination=0.05, random_state=RANDOM_STATE)
    iso.fit(X_ref)
    iso_pred = iso.predict(X_cand)
    iso_score = iso.decision_function(X_cand)

    rows = []
    nn_rows = []
    for i in range(len(candidate_x)):
        cand = candidate_x.iloc[i].get("Candidate", f"K{i+1}")
        cid = candidate_x.iloc[i].get("_candidate_id", "")
        nn1_p = percentile_rank(cand_nn1[i], ref_nn1)
        nn5_p = percentile_rank(cand_nn5[i], ref_nn5)
        mahal_p = percentile_rank(cand_mahal[i], ref_mahal)
        pca_p = percentile_rank(cand_pca_dist[i], ref_pca_dist)

        rows.append({
            "Candidate": cand,
            "_candidate_id": cid,
            "n_reference_samples": n_ref,
            "n_ood_features": n_feat,
            "nn1_distance": cand_nn1[i],
            "nn1_percentile_vs_ref": nn1_p,
            "nn5_mean_distance": cand_nn5[i],
            "nn5_percentile_vs_ref": nn5_p,
            "mahalanobis_sq": cand_mahal[i],
            "mahalanobis_percentile_vs_ref": mahal_p,
            "pca_score_distance": cand_pca_dist[i],
            "pca_distance_percentile_vs_ref": pca_p,
            "isolation_forest_label": "inlier" if iso_pred[i] == 1 else "outlier",
            "isolation_forest_score": iso_score[i],
        })
        for rank_j in range(min(5, idx_cand.shape[1])):
            ref_idx = int(idx_cand[i, rank_j])
            ref_row = reference.iloc[ref_idx]
            nn_rows.append({
                "Candidate": cand,
                "_candidate_id": cid,
                "neighbor_rank": rank_j + 1,
                "neighbor_ref_index": ref_idx,
                "neighbor_distance_scaled": float(dist_cand[i, rank_j]),
                "neighbor_row_id": ref_row.get("_row_id", ref_idx),
                "neighbor_VC_pred_raw": ref_row.get("VC_pred_raw", np.nan),
                "neighbor_SC_pred": ref_row.get("SC_pred", np.nan),
                "neighbor_EC_pred": ref_row.get("EC_pred", np.nan),
                "neighbor_HCHO_pred_ppm": ref_row.get("HCHO_pred_ppm", np.nan),
            })
    return pd.DataFrame(rows), pd.DataFrame(nn_rows)


def build_ad_summary(range_df: pd.DataFrame, multi_df: pd.DataFrame) -> pd.DataFrame:
    agg = range_df.groupby(["Candidate", "_candidate_id"], dropna=False).agg(
        n_features_checked=("feature", "count"),
        n_outside_minmax=("outside_minmax", "sum"),
        n_outside_q01_q99=("outside_qrange", "sum"),
        worst_minmax_margin_ratio=("minmax_margin_ratio", "min"),
    ).reset_index()
    out = agg.merge(multi_df, on=["Candidate", "_candidate_id"], how="left")

    def strict_label(r):
        if r["n_outside_minmax"] > 0:
            return "Outside"
        if r["n_outside_q01_q99"] > 0:
            return "Outside"
        if (r["nn5_percentile_vs_ref"] <= STRICT_NN5_INSIDE and
            r["mahalanobis_percentile_vs_ref"] <= STRICT_MAHAL_INSIDE and
            r["pca_distance_percentile_vs_ref"] <= STRICT_PCA_INSIDE and
            r["isolation_forest_label"] == "inlier"):
            return "Inside"
        if (r["nn5_percentile_vs_ref"] <= STRICT_NN5_BORDER and
            r["mahalanobis_percentile_vs_ref"] <= STRICT_MAHAL_BORDER and
            r["pca_distance_percentile_vs_ref"] <= STRICT_PCA_BORDER and
            r["isolation_forest_label"] == "inlier"):
            return "Borderline"
        return "Outside"

    def practical_label(r):
        if r["n_outside_minmax"] > 0:
            return "Outside"
        if (r["n_outside_q01_q99"] <= PRACTICAL_Q_OUTSIDE_MAX_INSIDE and
            r["nn5_percentile_vs_ref"] <= PRACTICAL_NN5_INSIDE and
            r["pca_distance_percentile_vs_ref"] <= PRACTICAL_PCA_INSIDE and
            r["isolation_forest_label"] == "inlier"):
            return "Inside"
        if (r["n_outside_q01_q99"] <= PRACTICAL_Q_OUTSIDE_MAX_BORDER and
            r["nn5_percentile_vs_ref"] <= PRACTICAL_NN5_BORDER and
            r["pca_distance_percentile_vs_ref"] <= PRACTICAL_PCA_BORDER and
            r["isolation_forest_label"] == "inlier"):
            return "Borderline"
        return "Outside"

    out["AD_label_strict"] = out.apply(strict_label, axis=1)
    out["AD_label_practical"] = out.apply(practical_label, axis=1)

    # 分数：仅用于排序，不作为硬证据
    penalty = (
        np.maximum(0, out["nn5_percentile_vs_ref"] - 90) * 1.2
        + np.maximum(0, out["pca_distance_percentile_vs_ref"] - 90) * 0.8
        + np.maximum(0, out["n_outside_q01_q99"] - 1) * 3.0
        + (out["isolation_forest_label"].ne("inlier").astype(float) * 15.0)
    )
    out["AD_score_practical"] = np.maximum(0, 100 - penalty)
    out["AD_note"] = np.where(
        out["AD_label_practical"].eq("Inside"),
        "Practical AD inside: no min-max extrapolation with acceptable NN/PCA/IF indicators; Mahalanobis retained as conservative risk indicator.",
        np.where(out["AD_label_practical"].eq("Borderline"),
                 "Practical AD borderline: usable as secondary candidate or SI candidate.",
                 "Practical AD outside: high extrapolation risk or obvious outlier."),
    )
    return out

# =============================================================================
# 4. 机制规则闭环
# =============================================================================

def load_rules() -> pd.DataFrame:
    """读取机理规则。

    优先使用脚本目录/上级目录中的 candidate_mechanism_rules.xlsx。
    如果找不到该文件，自动回退到脚本内置的 19 条规则，避免程序在
    [6/9] Mechanistic-rule closure 阶段中断。
    """
    try:
        p = find_existing(RULE_FILE)
        try:
            rules = pd.read_excel(p, sheet_name=RULE_SHEET)
        except ValueError:
            # 如果没有 rules 这个 sheet，就读取第一个 sheet。
            rules = pd.read_excel(p, sheet_name=0)
        print(f"[INFO] Mechanistic rules loaded from: {p}")
    except FileNotFoundError:
        rules = pd.DataFrame(DEFAULT_MECH_RULES)
        print("[WARN] candidate_mechanism_rules.xlsx not found; using embedded DEFAULT_MECH_RULES.")

    required = ["rule_id", "feature", "low", "high", "target", "rule_name", "mechanism_note"]
    missing = [c for c in required if c not in rules.columns]
    if missing:
        raise ValueError(f"Mechanistic rule table is missing required columns: {missing}")

    # 兼容 Excel 空值，并保证 low/high 是数值或 NaN。
    rules = rules.copy()
    rules["low"] = pd.to_numeric(rules["low"], errors="coerce")
    rules["high"] = pd.to_numeric(rules["high"], errors="coerce")
    return rules


def value_passes(val, low, high) -> Optional[bool]:
    if pd.isna(val):
        return None
    if pd.notna(low) and val < float(low):
        return False
    if pd.notna(high) and val > float(high):
        return False
    return True


def compute_mechanism(candidate_meta: pd.DataFrame, rules: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, cand in candidate_meta.iterrows():
        for _, rule in rules.iterrows():
            feat = rule["feature"]
            val = pd.to_numeric(pd.Series([cand.get(feat, np.nan)]), errors="coerce").iloc[0]
            ok = value_passes(val, rule.get("low", np.nan), rule.get("high", np.nan))
            rows.append({
                "Candidate": cand.get("Candidate", ""),
                "_candidate_id": cand.get("_candidate_id", ""),
                "rule_id": rule.get("rule_id", ""),
                "target": rule.get("target", ""),
                "feature": feat,
                "candidate_value": val,
                "low": rule.get("low", np.nan),
                "high": rule.get("high", np.nan),
                "pass_rule": np.nan if ok is None else int(ok),
                "rule_status": "pass" if ok is True else ("fail" if ok is False else "not_evaluable"),
                "rule_name": rule.get("rule_name", ""),
                "mechanism_note": rule.get("mechanism_note", ""),
                "rule_type": rule.get("rule_type", ""),
                "basis": rule.get("basis", ""),
            })
    detail = pd.DataFrame(rows)
    valid = detail[detail["rule_status"].isin(["pass", "fail"])].copy()
    summary = valid.groupby(["Candidate", "_candidate_id"], dropna=False).agg(
        n_mechanism_rules=("rule_id", "count"),
        n_rules_passed=("pass_rule", "sum"),
    ).reset_index()
    summary["mechanism_consistency_rate"] = summary["n_rules_passed"] / summary["n_mechanism_rules"].replace(0, np.nan)
    summary["mechanism_all_pass"] = summary["n_rules_passed"].eq(summary["n_mechanism_rules"])
    return summary, detail

# =============================================================================
# 5. 配方家族与 Gower-like 距离
# =============================================================================

def add_formula_family(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    # 避免 merge 后出现重复列名导致 Pandas 返回多列 DataFrame
    out = out.loc[:, ~out.columns.duplicated()].copy()
    # 若已有旧的 signature / formula_family 列，先删除后重算
    drop_cols = [c for c in list(FAMILY_GROUPS.keys()) + ["formula_family"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    for name, cols in FAMILY_GROUPS.items():
        out[name] = out.apply(lambda r: group_signature(r, cols), axis=1)
    out["formula_family"] = [" | ".join(str(out.at[idx, name]) for name in FAMILY_GROUPS.keys()) for idx in out.index]
    return out


def compute_mixed_gower_matrix(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    cat_cols = [c for c in RAW_CAT_COLS if c in df.columns]
    num_cols = [c for c in RAW_NUM_COLS if c in df.columns]

    cat_arr = df[cat_cols].applymap(norm_text).to_numpy(dtype=object) if cat_cols else np.empty((n, 0))
    num_df = df[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else pd.DataFrame(index=df.index)
    # 缺失视为 0 用量/0 条件；若实际缺失较多，建议回原始表修正
    num_df = num_df.fillna(0.0)
    if num_cols:
        ranges = num_df.max(axis=0) - num_df.min(axis=0)
        ranges = ranges.replace(0, np.nan)
        num_scaled = ((num_df - num_df.min(axis=0)) / ranges).fillna(0.0).to_numpy(float)
    else:
        num_scaled = np.empty((n, 0))

    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        if cat_cols:
            cat_dist = (cat_arr[i, :] != cat_arr).mean(axis=1)
        else:
            cat_dist = np.zeros(n)
        if num_cols:
            num_dist = np.abs(num_scaled[i, :] - num_scaled).mean(axis=1)
        else:
            num_dist = np.zeros(n)
        if cat_cols and num_cols:
            dist = CATEGORY_DISTANCE_WEIGHT * cat_dist + NUMERIC_DISTANCE_WEIGHT * num_dist
        elif cat_cols:
            dist = cat_dist
        else:
            dist = num_dist
        D[i, :] = dist
    labels = df["Candidate"].astype(str).tolist() if "Candidate" in df.columns else [f"K{i+1}" for i in range(n)]
    return pd.DataFrame(D, index=labels, columns=labels)


def normalize_for_score(s: pd.Series, higher_better=True) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=s.index)
    z = (s - mn) / (mx - mn)
    return z if higher_better else 1 - z


def add_selection_score(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    round_norm = normalize_for_score(out.get("round_score", pd.Series(0, index=out.index)), True)
    ad_norm = normalize_for_score(out.get("AD_score_practical", pd.Series(0, index=out.index)), True)
    hcho_norm = normalize_for_score(out.get("HCHO_pred_ppm", pd.Series(0, index=out.index)), False)
    nn_norm = normalize_for_score(out.get("nn5_percentile_vs_ref", pd.Series(0, index=out.index)), False)
    out["selection_score"] = 0.55 * round_norm + 0.20 * ad_norm + 0.15 * hcho_norm + 0.10 * nn_norm
    return out


def greedy_diverse_selection(pool: pd.DataFrame, distance_matrix: pd.DataFrame, top_n: int = TOP_N) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pool.empty:
        return pool.copy(), pd.DataFrame()

    pool = pool.sort_values(
        by=["selection_score", "round_score", "AD_score_practical", "HCHO_pred_ppm", "nn5_percentile_vs_ref"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    attempts = []
    best_selected = []
    best_params = None

    for max_family in MAX_PER_FORMULA_FAMILY_SEQUENCE:
        for threshold in DIVERSITY_DISTANCE_THRESHOLDS:
            selected_idx = []
            family_counts: Dict[str, int] = {}
            for idx, row in pool.iterrows():
                fam = row.get("formula_family", "")
                if family_counts.get(fam, 0) >= max_family:
                    continue
                cand = str(row["Candidate"])
                if selected_idx:
                    selected_labels = [str(pool.loc[j, "Candidate"]) for j in selected_idx]
                    min_d = float(distance_matrix.loc[cand, selected_labels].min())
                else:
                    min_d = np.inf
                if (not selected_idx) or min_d >= threshold:
                    selected_idx.append(idx)
                    family_counts[fam] = family_counts.get(fam, 0) + 1
                if len(selected_idx) >= top_n:
                    break
            attempts.append({
                "max_per_formula_family": max_family,
                "min_gower_threshold": threshold,
                "n_selected": len(selected_idx),
            })
            if len(selected_idx) > len(best_selected):
                best_selected = selected_idx
                best_params = (max_family, threshold)
            if len(selected_idx) >= top_n:
                sel = pool.loc[selected_idx].copy().reset_index(drop=True)
                sel["diversity_max_per_formula_family"] = max_family
                sel["diversity_min_gower_threshold_used"] = threshold
                sel["New_Candidate"] = [f"Diverse-AD-C{i+1}" for i in range(len(sel))]
                return sel, pd.DataFrame(attempts)

    # 如果怎么都不足 top_n，则用最佳结果并继续按分数补齐
    selected_idx = list(best_selected)
    remaining = [i for i in pool.index if i not in selected_idx]
    for i in remaining:
        selected_idx.append(i)
        if len(selected_idx) >= top_n:
            break
    sel = pool.loc[selected_idx].copy().reset_index(drop=True)
    if best_params is None:
        best_params = (999, 0.0)
    sel["diversity_max_per_formula_family"] = best_params[0]
    sel["diversity_min_gower_threshold_used"] = best_params[1]
    sel["New_Candidate"] = [f"Diverse-AD-C{i+1}" for i in range(len(sel))]
    return sel, pd.DataFrame(attempts)


def pairwise_summary(selected: pd.DataFrame, D: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = selected["Candidate"].astype(str).tolist()
    if not labels:
        return pd.DataFrame(), pd.DataFrame()
    mat = D.loc[labels, labels].copy()
    rows = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j <= i:
                continue
            rows.append({
                "Candidate_A": a,
                "Candidate_B": b,
                "Gower_like_distance": mat.loc[a, b],
                "New_Candidate_A": selected.loc[selected["Candidate"].astype(str).eq(a), "New_Candidate"].iloc[0] if "New_Candidate" in selected else a,
                "New_Candidate_B": selected.loc[selected["Candidate"].astype(str).eq(b), "New_Candidate"].iloc[0] if "New_Candidate" in selected else b,
            })
    return mat.reset_index().rename(columns={"index": "Candidate"}), pd.DataFrame(rows)

# =============================================================================
# 6. 主流程
# =============================================================================

def main():
    print("[1/9] Loading candidate/reference files...")
    cand_x_path = find_existing(CAND_X_MASTER_FULL, CAND_X_MASTER_FULL_ALT)
    cand_meta_path = find_existing(CAND_RECOMMENDED, CAND_RECOMMENDED_ALT)
    cand_raw_path = find_existing(CAND_RAW_FORMULA_FULL, CAND_RAW_FORMULA_FULL_ALT)
    ref_path = Path(REFERENCE_WORKBOOK)

    cand_x = read_table(cand_x_path)
    cand_meta = read_table(cand_meta_path)
    cand_raw = read_table(cand_raw_path)
    ref = read_table(ref_path, sheet_name=REFERENCE_SHEET)

    # 统一 Candidate 标签
    for df in [cand_x, cand_meta, cand_raw]:
        if "Candidate" not in df.columns:
            df["Candidate"] = [f"K{i+1}" for i in range(len(df))]

    # 确保 candidate_id 为字符串
    for df in [cand_x, cand_meta, cand_raw]:
        if "_candidate_id" in df.columns:
            df["_candidate_id"] = df["_candidate_id"].astype(str)

    print("[2/9] Determining OOD feature columns from model feature names...")
    feature_cols, model_feature_info = determine_ood_feature_cols(cand_x, ref)
    print(f"  OOD feature columns used: {len(feature_cols)}")
    print(f"  Excluded columns: {EXCLUDE_OOD_COLS}")

    print("[3/9] Computing feature-range OOD...")
    ref_stats = compute_reference_stats(ref, feature_cols)
    range_df = feature_range_detail(cand_x, ref_stats, feature_cols)

    print("[4/9] Computing multivariate OOD...")
    multi_df, nearest_df = compute_multivariate_ood(ref, cand_x, feature_cols)
    ad_summary = build_ad_summary(range_df, multi_df)

    # 合并元信息
    print("[5/9] Merging candidate metadata and raw formula...")
    # 先用 candidate meta 补充目标/约束/分数列
    candidate_all = ad_summary.merge(cand_meta, on=["Candidate", "_candidate_id"], how="left", suffixes=("", "_meta"))
    raw_family = add_formula_family(cand_raw)
    family_cols = ["Candidate", "_candidate_id", *FAMILY_GROUPS.keys(), "formula_family"]
    candidate_all = candidate_all.merge(raw_family[family_cols], on=["Candidate", "_candidate_id"], how="left")

    # HCHO grade 兜底
    if "HCHO_grade" not in candidate_all.columns and "HCHO_pred_ppm" in candidate_all.columns:
        candidate_all["HCHO_grade"] = candidate_all["HCHO_pred_ppm"].apply(classify_hcho_grade)

    print("[6/9] Mechanistic-rule closure...")
    rules = load_rules()
    mech_summary, mech_detail = compute_mechanism(candidate_all, rules)
    candidate_all = candidate_all.merge(mech_summary, on=["Candidate", "_candidate_id"], how="left")
    candidate_all["mechanism_all_pass"] = candidate_all["mechanism_all_pass"].fillna(False)

    print("[7/9] Building Practical-Inside pool and diversity-aware selection...")
    pool = candidate_all[
        candidate_all["AD_label_practical"].eq("Inside")
        & candidate_all["mechanism_all_pass"].eq(True)
    ].copy()
    # 可选硬条件：必须为可行候选
    if "is_feasible" in pool.columns:
        pool = pool[pool["is_feasible"].astype(bool)].copy()
    pool = add_selection_score(pool)

    # 配方距离只在 pool 内算
    pool_raw = raw_family[raw_family["_candidate_id"].isin(pool["_candidate_id"])]
    # raw 与 pool 同步排序/合并
    pool_for_dist = pool[["Candidate", "_candidate_id"]].merge(pool_raw, on=["Candidate", "_candidate_id"], how="left", suffixes=("", "_raw"))
    # 补回 formula_family
    if "formula_family" not in pool_for_dist.columns or pool_for_dist["formula_family"].isna().all():
        pool_for_dist = add_formula_family(pool_for_dist)
    D_pool = compute_mixed_gower_matrix(pool_for_dist)

    top_ranked = pool.sort_values(
        by=["selection_score", "round_score", "AD_score_practical", "HCHO_pred_ppm", "nn5_percentile_vs_ref"],
        ascending=[False, False, False, True, True],
    ).head(TOP_N).copy().reset_index(drop=True)
    top_ranked["New_Candidate"] = [f"TopRanked-AD-C{i+1}" for i in range(len(top_ranked))]
    diverse_selected, diversity_attempts = greedy_diverse_selection(pool, D_pool, TOP_N)

    # 最终采用 diverse 版本
    final_selected = diverse_selected.copy()
    final_selected["selection_mode"] = "Practical AD Inside + mechanism all-pass + diversity-aware greedy re-ranking"

    # 原始配方输出
    final_raw = final_selected[["Candidate", "_candidate_id", "New_Candidate", "AD_label_practical", "AD_label_strict"]].merge(
        cand_raw, on=["Candidate", "_candidate_id"], how="left", suffixes=("", "_raw")
    )
    top_raw = top_ranked[["Candidate", "_candidate_id", "New_Candidate", "AD_label_practical", "AD_label_strict"]].merge(
        cand_raw, on=["Candidate", "_candidate_id"], how="left", suffixes=("", "_raw")
    )

    print("[8/9] Preparing summaries and Origin tables...")
    # AD label counts
    strict_counts = candidate_all["AD_label_strict"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    strict_counts.insert(0, "AD_label_type", "AD_label_strict")
    practical_counts = candidate_all["AD_label_practical"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    practical_counts.insert(0, "AD_label_type", "AD_label_practical")
    label_counts = pd.concat([strict_counts, practical_counts], ignore_index=True)

    # family summaries
    family_summary = pool.groupby("formula_family", dropna=False).agg(
        n_candidates=("Candidate", "count"),
        best_selection_score=("selection_score", "max"),
        best_round_score=("round_score", "max"),
        min_HCHO_pred_ppm=("HCHO_pred_ppm", "min"),
        mean_nn5_percentile=("nn5_percentile_vs_ref", "mean"),
        mean_mahalanobis_percentile=("mahalanobis_percentile_vs_ref", "mean"),
    ).reset_index().sort_values(["n_candidates", "best_selection_score"], ascending=[False, False])

    matrix_selected, pairwise_selected = pairwise_summary(final_selected, D_pool)
    matrix_top, pairwise_top = pairwise_summary(top_ranked, D_pool)

    # 对照表
    comparison = pd.DataFrame({
        "candidate_set": ["Top-ranked practical AD", "Diversity-aware practical AD"],
        "n_candidates": [len(top_ranked), len(final_selected)],
        "n_unique_formula_family": [top_ranked["formula_family"].nunique(), final_selected["formula_family"].nunique()],
        "mean_pairwise_gower": [pairwise_top["Gower_like_distance"].mean() if not pairwise_top.empty else np.nan,
                                 pairwise_selected["Gower_like_distance"].mean() if not pairwise_selected.empty else np.nan],
        "min_pairwise_gower": [pairwise_top["Gower_like_distance"].min() if not pairwise_top.empty else np.nan,
                                pairwise_selected["Gower_like_distance"].min() if not pairwise_selected.empty else np.nan],
        "mean_round_score": [top_ranked["round_score"].mean(), final_selected["round_score"].mean()],
        "mean_HCHO_pred_ppm": [top_ranked["HCHO_pred_ppm"].mean(), final_selected["HCHO_pred_ppm"].mean()],
        "mean_nn5_percentile": [top_ranked["nn5_percentile_vs_ref"].mean(), final_selected["nn5_percentile_vs_ref"].mean()],
        "mean_mahalanobis_percentile": [top_ranked["mahalanobis_percentile_vs_ref"].mean(), final_selected["mahalanobis_percentile_vs_ref"].mean()],
    })

    # 机制规则 heatmap
    selected_ids = final_selected["_candidate_id"].astype(str).tolist()
    heat = mech_detail[mech_detail["_candidate_id"].astype(str).isin(selected_ids)].copy()
    heat["value_for_heatmap"] = heat["rule_status"].map({"pass": 1, "fail": 0, "not_evaluable": np.nan})
    rule_heat = heat.pivot_table(index="rule_id", columns="Candidate", values="value_for_heatmap", aggfunc="first").reset_index()

    # Origin long tables
    origin_selected_bar_cols = [
        "New_Candidate", "Candidate", "AD_label_practical", "AD_label_strict",
        "selection_score", "round_score", "AD_score_practical", "HCHO_pred_ppm", "VC_pred_raw", "SC_pred", "EC_pred",
        "nn5_percentile_vs_ref", "mahalanobis_percentile_vs_ref", "pca_distance_percentile_vs_ref",
        "n_outside_minmax", "n_outside_q01_q99", "mechanism_consistency_rate", "formula_family"
    ]
    origin_selected_bar = final_selected[[c for c in origin_selected_bar_cols if c in final_selected.columns]].copy()

    formula_long_rows = []
    raw_display_cols = [c for c in RAW_CAT_COLS + RAW_NUM_COLS if c in final_raw.columns]
    for _, row in final_raw.iterrows():
        for c in raw_display_cols:
            formula_long_rows.append({
                "New_Candidate": row.get("New_Candidate", ""),
                "Candidate": row.get("Candidate", ""),
                "feature": c,
                "value": row.get(c, np.nan),
            })
    origin_formula_long = pd.DataFrame(formula_long_rows)

    notes = pd.DataFrame([
        {"Item": "Script version", "Value": "V4 Practical AD + diversity-aware selection"},
        {"Item": "Candidate pool", "Value": f"{len(cand_x)} candidates from {cand_x_path.name}"},
        {"Item": "Reference distribution", "Value": f"{len(ref)} rows from {ref_path.name} / {REFERENCE_SHEET}"},
        {"Item": "OOD features used", "Value": len(feature_cols)},
        {"Item": "Excluded OOD columns", "Value": "; ".join(EXCLUDE_OOD_COLS)},
        {"Item": "Strict AD meaning", "Value": "Conservative multivariate AD using NN + Mahalanobis + PCA + Isolation Forest + q01-q99."},
        {"Item": "Practical AD meaning", "Value": "No min-max extrapolation, limited q01-q99 boundary features, acceptable NN/PCA/IF; Mahalanobis retained as conservative risk indicator."},
        {"Item": "Diversity selection", "Value": "Greedy selection using formula-family limits and mixed/Gower-like distance on raw formula categories and numeric dosages/process conditions."},
        {"Item": "Final candidate set", "Value": "Diverse-AD-C1 to Diverse-AD-C8 from practical AD inside + mechanism all-pass pool."},
        {"Item": "Important wording", "Value": "Do not claim strict AD inside if AD_label_strict is Outside. Claim practical AD-constrained and diversity-aware post-screening."},
    ])

    print("[9/9] Writing outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_xlsx = OUTPUT_DIR / OUT_XLSX
    out_csv = OUTPUT_DIR / OUT_CSV
    with pd.ExcelWriter(out_xlsx, engine=EXCEL_ENGINE) as writer:
        notes.to_excel(writer, sheet_name="README", index=False)
        pd.DataFrame({"OOD_feature": feature_cols}).to_excel(writer, sheet_name="OOD_features_used", index=False)
        model_feature_info.to_excel(writer, sheet_name="model_feature_sources", index=False)
        label_counts.to_excel(writer, sheet_name="AD_label_counts", index=False)
        comparison.to_excel(writer, sheet_name="compare_top_vs_diverse", index=False)
        top_ranked.to_excel(writer, sheet_name="top_ranked_selected_C1C8", index=False)
        final_selected.to_excel(writer, sheet_name="diverse_selected_C1C8", index=False)
        top_raw.to_excel(writer, sheet_name="top_ranked_raw_formula", index=False)
        final_raw.to_excel(writer, sheet_name="diverse_raw_formula", index=False)
        family_summary.to_excel(writer, sheet_name="formula_family_summary", index=False)
        matrix_selected.to_excel(writer, sheet_name="selected_gower_matrix", index=False)
        pairwise_selected.to_excel(writer, sheet_name="selected_pairwise_gower", index=False)
        diversity_attempts.to_excel(writer, sheet_name="diversity_selection_attempts", index=False)
        candidate_all.to_excel(writer, sheet_name="all_candidate_AD_summary", index=False)
        range_df.to_excel(writer, sheet_name="all_feature_range_detail", index=False)
        multi_df.to_excel(writer, sheet_name="all_multivariate_OOD", index=False)
        nearest_df.to_excel(writer, sheet_name="nearest_neighbors", index=False)
        mech_summary.to_excel(writer, sheet_name="mechanism_summary", index=False)
        mech_detail.to_excel(writer, sheet_name="mechanism_rule_detail", index=False)
        rules.to_excel(writer, sheet_name="rules_used", index=False)
        ref_stats.to_excel(writer, sheet_name="reference_feature_stats", index=False)
        origin_selected_bar.to_excel(writer, sheet_name="Origin_selected_bar", index=False)
        rule_heat.to_excel(writer, sheet_name="Origin_rule_heatmap", index=False)
        origin_formula_long.to_excel(writer, sheet_name="Origin_formula_long", index=False)
        matrix_selected.to_excel(writer, sheet_name="Origin_gower_matrix", index=False)
        family_summary.to_excel(writer, sheet_name="Origin_family_counts", index=False)

    final_selected.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Excel: {out_xlsx}")
    print(f"CSV:   {out_csv}")
    print("Summary:")
    print(label_counts.to_string(index=False))
    print("\nCompare top-ranked vs diverse:")
    print(comparison.to_string(index=False))
    print("\nFinal selected candidates:")
    cols = [c for c in ["New_Candidate", "Candidate", "_candidate_id", "AD_label_practical", "AD_label_strict", "selection_score", "round_score", "HCHO_pred_ppm", "nn5_percentile_vs_ref", "mahalanobis_percentile_vs_ref", "formula_family"] if c in final_selected.columns]
    print(final_selected[cols].to_string(index=False))


if __name__ == "__main__":
    main()
