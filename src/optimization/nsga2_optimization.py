# -*- coding: utf-8 -*-
"""
VAE 乳液多目标优化 V2：NSGA-II 三轮优化版本
=========================================================
版本：PC8 + corrected HCHO MLP + HCHO A/B/C 工程分级版

V2 = V1-PC8-HCHO_MLP_corrected 约束体系 + NSGA-II 进化搜索 + Pareto 前沿输出

核心特点：
1) 不依赖 pymoo，内置简化 NSGA-II；
2) 直接复用 v0.py 中已验证的 FrozenFeatureBuilder；
3) 在原始配方空间搜索，保持槽位顺序，不引入训练集外原料；
4) 四模型预测：
   - VC: log1p(viscosity) -> raw mPa·s
   - SC: raw %
   - EC: raw %
   - HCHO: corrected MLP, log1p(ppm) -> raw ppm
5) HCHO 工程分级：
   - A: HCHO_pred_ppm <= 1.5 ppm, below-LOD-equivalent / 未检出等效水平
   - B: 1.5 < HCHO_pred_ppm <= 5 ppm, ultra-low / 极低甲醛
   - C: 5 < HCHO_pred_ppm <= 30 ppm, acceptable trade-off / 可权衡候选
   - D: HCHO_pred_ppm > 30 ppm, high-HCHO risk / 高甲醛风险
6) 三轮 HCHO 窗口：
   - Round 1: HCHO <= 30 ppm
   - Round 2: HCHO <= 5 ppm
   - Round 3: HCHO <= 1.5 ppm
7) A级内部不继续比较 0、0.001、0.01 ppm 等低值差异；
   NSGA-II 目标函数中 HCHO <= 1.5 ppm 统一按 1.5 ppm 处理。
8) 保留 HCHO_pred_ppm_unclipped 和 HCHO_pred_clipped_zero，用于诊断负值截断。
9) 保留 corrected HCHO SHAP / 1D PDP / 2D PDP 支持的约束：
   - Round 2/3: red_hcho_potential_share = 0
   - Round 3: ter_hcho_potential_share = 0
   - Round 3: ter_total_pct <= 6
   - Round 3: oxid_mean_f+ <= 0.15
   - pc_total_pct <= 8
10) 输出：
   - final_population
   - feasible_candidates
   - pareto_candidates
   - recommended_top
   - raw_formula_top
   - X_master_top
   - final_population_X_master_all（完整 800 个最终种群的 X_master）
   - recommended_raw_formula_full（完整 recommended/feasible 候选原始配方）
   - recommended_X_master_full（完整 recommended/feasible 候选 X_master）
   - NSGA-II generation history
"""

from __future__ import annotations

import re
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 0. 复用 V0 冻结版特征生成器
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.feature_engineering.frozen_feature_pipeline import CONFIG as V0_CONFIG
    from src.feature_engineering.frozen_feature_pipeline import FrozenFeatureBuilder
    from src.feature_engineering.frozen_feature_pipeline import TARGET_COLS_RAW
except Exception as e:
    raise ImportError(
        "Cannot import the frozen feature pipeline from src.feature_engineering. "
        f"Original error: {repr(e)}"
    )


# ============================================================
# 1. 用户配置区
# ============================================================

PROJECT_ROOT = REPO_ROOT
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models"
APEO_TABLE_PATH = DATA_DIR / "ML1_descriptors_final_APEO.xlsx"
OUTPUT_DIR = REPO_ROOT / "results" / "nsga2"

RANDOM_SEED = 42

# NSGA-II 参数
NSGA2_POP_SIZE = 800
NSGA2_INIT_RANDOM_SIZE = 5000
NSGA2_N_GEN = 80
NSGA2_OFFSPRING_SIZE = 800

# 交叉/变异参数
CROSSOVER_PROB = 0.90
PROCESS_MUTATION_PROB = 0.15
FAMILY_TOTAL_MUTATION_PROB = 0.25
FAMILY_PATTERN_MUTATION_PROB = 0.10
GAUSSIAN_MUTATION_SCALE = 0.08

# 输出数量
MAX_EXPORT_POPULATION = 10000
MAX_EXPORT_FEASIBLE = 5000
MAX_EXPORT_PARETO = 2000
MAX_EXPORT_RECOMMENDED = 200

ENABLE_APEO_FILTER = True
CLIP_NEGATIVE_RAW_TARGETS = True

# 保护胶体总量工程上限
PC_TOTAL_PCT_MAX = 8.0


# ============================================================
# 2. HCHO 工程分级阈值
# ============================================================

HCHO_GRADE_A_MAX = 1.5
HCHO_GRADE_B_MAX = 5.0
HCHO_GRADE_C_MAX = 30.0

# 项目要求参考值；当前优化中 >30 ppm 已作为高风险剔除。
HCHO_PROJECT_SPEC_MAX = 50.0


# ============================================================
# 3. 模型路径配置
# ============================================================

MODEL_CONFIG = {
    "VC": {
        "path": MODEL_DIR / "VC_CatBoost.joblib",
        "target_col": "Viscosity",
        "target_dataset": DATA_DIR / "X_VC_true.xlsx",
        "output_is_log1p": True,
    },
    "SC": {
        "path": MODEL_DIR / "SC_XGBoost.joblib",
        "target_col": "Solid Content",
        "target_dataset": DATA_DIR / "X_SC_true.xlsx",
        "output_is_log1p": False,
    },
    "EC": {
        "path": MODEL_DIR / "EC_CatBoost.joblib",
        "target_col": "Ethylene Content",
        "target_dataset": DATA_DIR / "X_EC_true.xlsx",
        "output_is_log1p": False,
    },
    "HCHO": {
        "path": MODEL_DIR / "HCHO_MLP.joblib",
        "target_col": "Free HCHO Content",
        "target_dataset": DATA_DIR / "X_HCHO_true.xlsx",
        "output_is_log1p": True,
    },
}

MODEL_BASENAME_FALLBACK = {
    "VC": ["VC_CatBoost.joblib"],
    "SC": ["SC_XGBoost.joblib"],
    "EC": ["EC_CatBoost.joblib"],
    "HCHO": [
        "HCHO_MLP.joblib",
    ],
}


# ============================================================
# 4. 三轮目标窗口
# ============================================================

ROUND_TARGET_WINDOWS = {
    "round1_stable": {
        "round_label": "Round 1 - stable feasible window",
        "HCHO_ppm_max": HCHO_GRADE_C_MAX,
        "HCHO_target_grade": "C_or_better",
        "SC_window": (54.5, 56.5),
        "VC_raw_window": (1000.0, 4500.0),
        "EC_window": (16.0, 19.0),
        "APEO_free_required": False,
    },

    "round2_high_end": {
        "round_label": "Round 2 - high-end contraction window",
        "HCHO_ppm_max": HCHO_GRADE_B_MAX,
        "HCHO_target_grade": "B_or_better",
        "SC_window": (56.0, 59.0),
        "VC_raw_window": (1500.0, 5500.0),
        "EC_window": (18.0, 21.0),
        "APEO_free_required": True,
    },

    "round3_frontier": {
        "round_label": "Round 3 - frontier exploration window",
        "HCHO_ppm_max": HCHO_GRADE_A_MAX,
        "HCHO_target_grade": "A",
        "SC_window": (59.0, 61.5),
        "VC_raw_window": (2000.0, 6000.0),
        "EC_window": (19.0, 22.0),
        "APEO_free_required": True,
    },
}


# ============================================================
# 5. 输入搜索空间约束
# ============================================================

ROUND_INPUT_BOUNDS = {
    "round1_stable": {
        "poly_pressure": (4.5, 8.5),
        "poly_temp": (50.0, 88.0),
        "time": (100.0, 330.0),
        "coinitiator_pct": (0.0, 0.0078),
        "red_total_pct": (0.13, 0.76),
        "ter_total_pct": (0.0, 23.35),
        "emul_total_pct": (0.0, 17.5),
        "pc_total_pct": (0.0, PC_TOTAL_PCT_MAX),
        "oxid_total_pct": "Q5_Q95",
        "pH_buffer_pct": "Q5_Q95",
        "dispersant_pct": "Q5_Q95",
        "init_temp": "Q5_Q95",
    },

    "round2_high_end": {
        "poly_pressure": (4.5, 8.5),
        "poly_temp": (60.0, 88.0),
        "time": (170.0, 330.0),
        "coinitiator_pct": (0.0, 0.0078),
        "red_total_pct": (0.13, 0.76),
        "ter_total_pct": (0.0, 12.0),
        "emul_total_pct": (1.0, 17.5),
        "pc_total_pct": (0.0, PC_TOTAL_PCT_MAX),
        "oxid_total_pct": "Q5_Q95",
        "pH_buffer_pct": "Q5_Q95",
        "dispersant_pct": "Q5_Q95",
        "init_temp": "Q5_Q95",
    },

    "round3_frontier": {
        "poly_pressure": (5.8, 8.5),
        "poly_temp": (60.0, 88.0),
        "time": (240.0, 330.0),
        "coinitiator_pct": (0.0, 0.0078),
        "red_total_pct": (0.13, 0.76),
        "ter_total_pct": (0.0, 6.0),
        "emul_total_pct": (3.0, 17.5),
        "pc_total_pct": (0.0, PC_TOTAL_PCT_MAX),
        "oxid_total_pct": "Q5_Q95",
        "pH_buffer_pct": "Q5_Q95",
        "dispersant_pct": "Q5_Q95",
        "init_temp": "Q5_Q95",
    },
}


# ============================================================
# 6. 结构硬约束与 PDP 软偏好
# ============================================================

ROUND_STRUCTURE_RULES = {
    "round1_stable": {
        "red_hcho_potential_share_max": 1.0,
        "ter_hcho_potential_share_max": 1.0,
        "oxid_mean_f+_max": None,
        "ter_total_pct_max": 23.35,
        "emul_total_pct_min": 0.0,
        "pc_total_pct_max": PC_TOTAL_PCT_MAX,
        "red_mean_E_HOMO_soft_max": None,
    },

    "round2_high_end": {
        "red_hcho_potential_share_max": 0.0,
        "ter_hcho_potential_share_max": 0.1,
        "oxid_mean_f+_max": 0.228,
        "ter_total_pct_max": 12.0,
        "emul_total_pct_min": 1.0,
        "pc_total_pct_max": PC_TOTAL_PCT_MAX,
        "red_mean_E_HOMO_soft_max": -0.31,
    },

    "round3_frontier": {
        "red_hcho_potential_share_max": 0.0,
        "ter_hcho_potential_share_max": 0.0,
        "oxid_mean_f+_max": 0.15,
        "ter_total_pct_max": 6.0,
        "emul_total_pct_min": 3.0,
        "pc_total_pct_max": PC_TOTAL_PCT_MAX,
        "red_mean_E_HOMO_soft_max": -0.32,
    },
}

PDP_SOFT_PREFERENCES = {
    "EC": {
        "poly_pressure": (5.8, 8.5),
        "time": (282.0, 330.0),
        "coinitiator_pct": (0.006, 0.0078),
    },

    "SC": {
        "poly_temp": (60.0, 88.0),
        "emul_total_pct": (6.0, 13.5),
        "emul_mean_EO_number": (23.0, 30.0),
        "coinitiator_pct": (0.0015, 0.0035),
    },

    "VC_medium": {
        "red_total_pct": (0.33, 0.52),
        "coinitiator_pct": (0.002, 0.004),
        "pc_total_pct": (0.0, PC_TOTAL_PCT_MAX),
        "pc_mean_DP": (1200.0, 1700.0),
    },

    "HCHO": {
        "ter_total_pct": (0.0, 3.0),
        "ter_hcho_potential_share": (0.0, 0.0),

        "oxid_mean_f+": (0.0, 0.10),
        "red_mean_E_HOMO_max_round2": -0.31,
        "red_mean_E_HOMO_max_round3": -0.32,

        "emul_total_pct": (5.7, 17.5),

        "pc_total_pct": (3.0, PC_TOTAL_PCT_MAX),
        "pc_mean_DP": (1200.0, 2000.0),

        "red_hcho_potential_share": (0.0, 0.0),
    },
}


# ============================================================
# 7. 原始配方槽位配置
# ============================================================

FAMILY_SLOT_CONFIG = {
    "oxid": {
        "type_prefix": "oxidizer_type",
        "pct_prefix": "oxidizer_pct",
        "n_slots": 2,
        "total_feature_raw": "oxid_total_pct",
    },
    "red": {
        "type_prefix": "reducer_type",
        "pct_prefix": "reducer_pct",
        "n_slots": 3,
        "total_feature_raw": "red_total_pct",
    },
    "ter": {
        "type_prefix": "third_monomer_type",
        "pct_prefix": "third_monomer_pct",
        "n_slots": 7,
        "total_feature_raw": "ter_total_pct",
    },
    "emul": {
        "type_prefix": "emulsifier_type",
        "pct_prefix": "emulsifier_pct",
        "n_slots": 4,
        "total_feature_raw": "emul_total_pct",
    },
    "pc": {
        "type_prefix": "protective_colloid_type",
        "pct_prefix": "protective_colloid_pct",
        "n_slots": 3,
        "total_feature_raw": "pc_total_pct",
    },
}


DROP_COLS_FOR_MODEL = [
    "oxidizer_pct2_isna",
    "init_pressure_isna",
    "time_isna",
    "pH_buffer_pct_isna",
]


# ============================================================
# 8. 基础工具函数
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm_text(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace("\u00A0", " ").replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def safe_expm1(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    out = np.expm1(arr)
    if CLIP_NEGATIVE_RAW_TARGETS:
        out = np.clip(out, 0.0, None)
    return out


def random_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0
    if hi <= lo:
        return float(lo)
    return float(rng.uniform(lo, hi))


def clamp(v: float, lo: float, hi: float) -> float:
    if not np.isfinite(v):
        return float(lo)
    return float(min(max(v, lo), hi))


def find_file_by_names(root: Path, names: List[str]) -> Optional[Path]:
    root = Path(root)
    if not root.exists():
        return None

    for name in names:
        p = root / name
        if p.exists():
            return p

    lower_names = set([n.lower() for n in names])
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() in lower_names:
            return p

    return None


def resolve_model_path(target_key: str, configured_path: Path) -> Path:
    configured_path = Path(configured_path)
    if configured_path.exists():
        return configured_path

    fallback_names = MODEL_BASENAME_FALLBACK.get(target_key, [])
    found = find_file_by_names(PROJECT_ROOT, fallback_names)

    if found is not None:
        print(f"[INFO] {target_key}: 配置路径不存在，自动找到模型文件: {found}")
        return found

    raise FileNotFoundError(
        f"{target_key} 模型文件不存在: {configured_path}\n"
        f"也未能在 {PROJECT_ROOT} 下按 {fallback_names} 自动找到。\n"
        "请修改 MODEL_CONFIG 中对应路径。"
    )


def load_table_if_exists(path: Path) -> Optional[pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return None


def is_model_like(obj) -> bool:
    if obj is None:
        return False
    cls = obj.__class__
    mod = getattr(cls, "__module__", "").lower()
    name = getattr(cls, "__name__", "")
    if "sklearn" in mod or "catboost" in mod or "xgboost" in mod:
        return True
    return any(k in name for k in [
        "CatBoost", "XGB", "LGBM", "Pipeline", "Regressor",
        "RandomForest", "ExtraTrees", "SVR", "MLP"
    ])


def extract_model_from_joblib(obj):
    if not isinstance(obj, dict):
        return obj, None

    keys = ["model", "best_model", "final_model", "estimator", "regressor", "pipeline"]
    for k in keys:
        if k in obj and is_model_like(obj[k]):
            return obj[k], obj

    for _, v in obj.items():
        if is_model_like(v):
            return v, obj

    raise ValueError(f"joblib 是 dict，但没有找到模型对象。keys={list(obj.keys())}")


def try_get_feature_cols_from_bundle(bundle: Optional[dict]) -> Optional[List[str]]:
    if bundle is None:
        return None

    keys = [
        "feature_cols", "features", "selected_features",
        "x_cols", "X_cols", "columns", "feature_names"
    ]
    for k in keys:
        if k in bundle:
            v = bundle[k]
            if isinstance(v, (list, tuple, np.ndarray, pd.Index)):
                return list(v)
    return None


def try_get_feature_cols_from_model(model) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "feature_names_"):
        try:
            return list(model.feature_names_)
        except Exception:
            pass
    return None


def infer_feature_cols_for_target(
    target_key: str,
    model,
    bundle,
    target_dataset: Path,
    target_col: str
) -> List[str]:
    cols = try_get_feature_cols_from_bundle(bundle)
    if cols is not None:
        print(f"[INFO] {target_key}: 使用 joblib bundle 中的 feature_cols, n={len(cols)}")
        return cols

    cols = try_get_feature_cols_from_model(model)
    if cols is not None:
        print(f"[INFO] {target_key}: 使用模型自身 feature_names, n={len(cols)}")
        return cols

    df_target = load_table_if_exists(target_dataset)

    if df_target is None:
        model_path = Path(MODEL_CONFIG[target_key]["path"])
        candidate_names = [
            target_dataset.name,
            "X_HCHO_true.xlsx",
            "X_reduced_final_HCHO_true.xlsx",
            "X_HCHO_corrected_true.xlsx",
        ]
        found = find_file_by_names(model_path.parent, candidate_names)
        if found is not None:
            print(f"[INFO] {target_key}: target_dataset 不存在，自动找到: {found}")
            df_target = load_table_if_exists(found)

    if df_target is not None:
        cols = [
            c for c in df_target.columns
            if c != target_col and c not in DROP_COLS_FOR_MODEL
        ]
        print(f"[INFO] {target_key}: 从目标数据集推断 feature_cols, n={len(cols)}")
        return cols

    raise ValueError(
        f"{target_key}: 无法获得模型输入特征列。\n"
        f"请确认模型保存了 feature_cols，或 target_dataset 存在: {target_dataset}"
    )


def align_X_for_model(X_master: pd.DataFrame, feature_cols: List[str], target_key: str) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in X_master.columns]
    if missing:
        raise ValueError(
            f"{target_key}: X_master 缺少模型输入特征:\n{missing}\n"
            "说明冻结版特征生成器或目标模型特征列不一致。"
        )
    return X_master[feature_cols].copy()


def read_raw_data_from_v0_config() -> pd.DataFrame:
    raw_path = Path(V0_CONFIG["raw_data"])
    if not raw_path.exists():
        raise FileNotFoundError(f"V0_CONFIG['raw_data'] 不存在: {raw_path}")

    if raw_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(raw_path)
    if raw_path.suffix.lower() == ".csv":
        return pd.read_csv(raw_path)

    raise ValueError(f"不支持的 raw_data 文件类型: {raw_path}")


# ============================================================
# 9. HCHO 分级
# ============================================================

def make_hcho_grade_array(hcho_ppm: np.ndarray) -> np.ndarray:
    grade = np.full(len(hcho_ppm), "D_high_risk", dtype=object)
    grade[hcho_ppm <= HCHO_GRADE_C_MAX] = "C_acceptable_tradeoff"
    grade[hcho_ppm <= HCHO_GRADE_B_MAX] = "B_ultra_low"
    grade[hcho_ppm <= HCHO_GRADE_A_MAX] = "A_below_LOD_equivalent"
    return grade


def make_hcho_grade_label_from_values(grade: str, clipped: bool) -> str:
    if grade == "A_below_LOD_equivalent":
        if clipped:
            return "≤1.5 ppm (A, below-LOD-equivalent; clipped)"
        return "≤1.5 ppm (A, below-LOD-equivalent)"
    if grade == "B_ultra_low":
        return "1.5–5 ppm (B, ultra-low)"
    if grade == "C_acceptable_tradeoff":
        return "5–30 ppm (C, trade-off acceptable)"
    return ">30 ppm (high-HCHO risk)"


def hcho_grade_score_from_series(grade_series: pd.Series) -> pd.Series:
    score_map = {
        "A_below_LOD_equivalent": 1.00,
        "B_ultra_low": 0.75,
        "C_acceptable_tradeoff": 0.50,
        "D_high_risk": 0.00,
    }
    return grade_series.map(score_map).fillna(0.0)


def hcho_grade_priority_series(grade_series: pd.Series) -> pd.Series:
    priority_map = {
        "A_below_LOD_equivalent": 3,
        "B_ultra_low": 2,
        "C_acceptable_tradeoff": 1,
        "D_high_risk": 0,
    }
    return grade_series.map(priority_map).fillna(0).astype(int)


# ============================================================
# 10. APEO 表读取与匹配
# ============================================================

def detect_name_columns(df: pd.DataFrame) -> List[str]:
    keys = [
        "raw_name", "standard_name", "std_name", "name", "名称",
        "别名", "商品名", "英文", "中文", "synonym", "alias",
        "chemical", "material", "原料"
    ]
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if any(k.lower() in cl for k in keys):
            cols.append(c)
    if not cols:
        cols = [c for c in df.columns if df[c].dtype == "object"]
    return cols


def load_apeo_allowed_names(path: Path) -> Tuple[set, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(
            f"APEO 标注表不存在: {path}\n"
            "请先生成 ML1_descriptors_final_APEO.xlsx。"
        )

    sheets = pd.read_excel(path, sheet_name=None)

    df = None
    for _, sdf in sheets.items():
        if "round3_APEO_free_allowed" in sdf.columns:
            df = sdf.copy()
            break

    if df is None:
        raise ValueError("APEO 表中没有找到 round3_APEO_free_allowed 列。")

    name_cols = detect_name_columns(df)

    allowed_names = set()
    for _, row in df.iterrows():
        allowed = row.get("round3_APEO_free_allowed", 0)
        try:
            allowed = int(allowed)
        except Exception:
            allowed = 0

        if allowed == 1:
            for c in name_cols:
                val = norm_text(row.get(c, ""))
                if val:
                    allowed_names.add(val)

    return allowed_names, df


def candidate_is_apeo_free(raw_row: pd.Series, allowed_names: set) -> bool:
    cfg = FAMILY_SLOT_CONFIG["emul"]
    for i in range(1, cfg["n_slots"] + 1):
        type_col = f"{cfg['type_prefix']}{i}"
        pct_col = f"{cfg['pct_prefix']}{i}"

        name = norm_text(raw_row.get(type_col, ""))
        pct = raw_row.get(pct_col, 0.0)

        try:
            pct = float(pct)
        except Exception:
            pct = 0.0

        if pct <= 0 or name == "":
            continue

        if name not in allowed_names:
            return False

    return True


# ============================================================
# 11. 原始配方操作函数
# ============================================================

def slot_cols(family: str) -> Tuple[List[str], List[str]]:
    cfg = FAMILY_SLOT_CONFIG[family]
    type_cols = [f"{cfg['type_prefix']}{i}" for i in range(1, cfg["n_slots"] + 1)]
    pct_cols = [f"{cfg['pct_prefix']}{i}" for i in range(1, cfg["n_slots"] + 1)]
    return type_cols, pct_cols


def family_total_from_raw_row(row: pd.Series, family: str) -> float:
    _, pct_cols = slot_cols(family)
    total = 0.0
    for c in pct_cols:
        if c in row.index:
            try:
                v = float(row.get(c, 0.0))
                if np.isfinite(v):
                    total += max(v, 0.0)
            except Exception:
                pass
    return total


def compute_raw_family_total_series(df: pd.DataFrame, family: str) -> pd.Series:
    _, pct_cols = slot_cols(family)
    existing = [c for c in pct_cols if c in df.columns]
    if not existing:
        return pd.Series(np.zeros(len(df)), index=df.index)
    tmp = df[existing].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    tmp = tmp.clip(lower=0.0)
    return tmp.sum(axis=1)


def build_donor_indices(raw_x: pd.DataFrame, allowed_names: set) -> Dict[str, List[int]]:
    donors = {}

    for fam in FAMILY_SLOT_CONFIG:
        total = compute_raw_family_total_series(raw_x, fam)
        idx = list(raw_x.index[total > 0].astype(int))
        donors[fam] = idx

    emul_total = compute_raw_family_total_series(raw_x, "emul")
    emul_free_idx = []
    for idx, row in raw_x.iterrows():
        if emul_total.loc[idx] > 0 and candidate_is_apeo_free(row, allowed_names):
            emul_free_idx.append(int(idx))
    donors["emul_apeo_free"] = emul_free_idx

    return donors


def copy_family_pattern_from_donor(
    row: pd.Series,
    donor: pd.Series,
    family: str,
    target_total: float
) -> pd.Series:
    type_cols, pct_cols = slot_cols(family)

    donor_pcts = []
    for c in pct_cols:
        try:
            v = float(donor.get(c, 0.0))
            if not np.isfinite(v):
                v = 0.0
        except Exception:
            v = 0.0
        donor_pcts.append(max(v, 0.0))

    donor_total = sum(donor_pcts)

    if donor_total <= 0 or target_total <= 0:
        for tc, pc in zip(type_cols, pct_cols):
            if tc in row.index:
                row[tc] = ""
            if pc in row.index:
                row[pc] = 0.0
        return row

    shares = [v / donor_total for v in donor_pcts]

    for tc, pc, sh in zip(type_cols, pct_cols, shares):
        if tc in row.index and tc in donor.index:
            row[tc] = donor.get(tc, "")
        if pc in row.index:
            row[pc] = target_total * sh

    return row


def scale_family_total(
    row: pd.Series,
    family: str,
    target_total: float,
    raw_x: pd.DataFrame,
    donors: Dict[str, List[int]],
    rng: np.random.Generator,
    require_apeo_free_emul: bool = False,
) -> pd.Series:
    _, pct_cols = slot_cols(family)

    if not any(c in row.index for c in pct_cols):
        return row

    target_total = max(float(target_total), 0.0)

    current_pcts = []
    for c in pct_cols:
        try:
            v = float(row.get(c, 0.0))
            if not np.isfinite(v):
                v = 0.0
        except Exception:
            v = 0.0
        current_pcts.append(max(v, 0.0))

    current_total = sum(current_pcts)

    if target_total <= 0:
        for pc in pct_cols:
            if pc in row.index:
                row[pc] = 0.0
        return row

    if current_total > 0:
        shares = [v / current_total for v in current_pcts]
        for pc, sh in zip(pct_cols, shares):
            if pc in row.index:
                row[pc] = target_total * sh
        return row

    donor_key = "emul_apeo_free" if (family == "emul" and require_apeo_free_emul) else family
    donor_list = donors.get(donor_key, [])

    if not donor_list:
        return row

    donor_idx = int(rng.choice(donor_list))
    donor = raw_x.loc[donor_idx]
    row = copy_family_pattern_from_donor(row, donor, family, target_total)
    return row


def get_empirical_bound_from_raw(raw_x: pd.DataFrame, name: str) -> Tuple[float, float]:
    family_by_total = {
        cfg["total_feature_raw"]: fam
        for fam, cfg in FAMILY_SLOT_CONFIG.items()
    }

    if name in family_by_total:
        s = compute_raw_family_total_series(raw_x, family_by_total[name])
    elif name in raw_x.columns:
        s = pd.to_numeric(raw_x[name], errors="coerce")
    else:
        return (0.0, 0.0)

    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return (0.0, 0.0)

    lo = float(np.quantile(s, 0.05))
    hi = float(np.quantile(s, 0.95))

    if not np.isfinite(lo):
        lo = float(s.min())
    if not np.isfinite(hi):
        hi = float(s.max())

    if hi < lo:
        lo, hi = hi, lo

    return lo, hi


def resolve_bound(bound_spec, raw_x: pd.DataFrame, name: str) -> Tuple[float, float]:
    if isinstance(bound_spec, str) and bound_spec == "Q5_Q95":
        return get_empirical_bound_from_raw(raw_x, name)
    if isinstance(bound_spec, (tuple, list)) and len(bound_spec) == 2:
        return float(bound_spec[0]), float(bound_spec[1])
    raise ValueError(f"无法解析变量 {name} 的 bound_spec={bound_spec}")


def get_round_numeric_bounds(raw_x: pd.DataFrame, round_key: str) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for k, spec in ROUND_INPUT_BOUNDS[round_key].items():
        bounds[k] = resolve_bound(spec, raw_x, k)
    return bounds


def perturb_process_and_dosage(
    row: pd.Series,
    round_key: str,
    raw_x: pd.DataFrame,
    donors: Dict[str, List[int]],
    allowed_names: set,
    rng: np.random.Generator,
) -> pd.Series:
    bounds = get_round_numeric_bounds(raw_x, round_key)
    apeo_required = ROUND_TARGET_WINDOWS[round_key]["APEO_free_required"] and ENABLE_APEO_FILTER

    for col in ["poly_pressure", "poly_temp", "time", "init_temp", "pH_buffer_pct", "dispersant_pct"]:
        if col in row.index and col in bounds:
            lo, hi = bounds[col]
            row[col] = random_uniform(rng, lo, hi)

    if "coinitiator_pct" in row.index and "coinitiator_pct" in bounds:
        lo, hi = bounds["coinitiator_pct"]
        row["coinitiator_pct"] = random_uniform(rng, lo, hi)

    total_to_family = {
        "oxid_total_pct": "oxid",
        "red_total_pct": "red",
        "ter_total_pct": "ter",
        "emul_total_pct": "emul",
        "pc_total_pct": "pc",
    }

    for total_name, fam in total_to_family.items():
        if total_name not in bounds:
            continue

        lo, hi = bounds[total_name]
        target_total = random_uniform(rng, lo, hi)

        row = scale_family_total(
            row=row,
            family=fam,
            target_total=target_total,
            raw_x=raw_x,
            donors=donors,
            rng=rng,
            require_apeo_free_emul=apeo_required,
        )

    return row


def generate_random_candidates(
    raw_x: pd.DataFrame,
    round_key: str,
    n_candidates: int,
    allowed_names: set,
    rng: np.random.Generator,
) -> pd.DataFrame:
    donors = build_donor_indices(raw_x, allowed_names)

    rows = []
    source_indices = raw_x.index.values

    for i in range(n_candidates):
        base_idx = int(rng.choice(source_indices))
        row = raw_x.loc[base_idx].copy()

        row = perturb_process_and_dosage(
            row=row,
            round_key=round_key,
            raw_x=raw_x,
            donors=donors,
            allowed_names=allowed_names,
            rng=rng,
        )

        row["_candidate_id"] = f"{round_key}_init_{i:07d}"
        row["_source_row_index"] = base_idx
        row["_generated_for_round"] = round_key
        row["_generation"] = 0

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# ============================================================
# 12. NSGA-II 交叉与变异
# ============================================================

def get_col_value(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        v = float(row.get(col, default))
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def blend_value(
    a: float,
    b: float,
    lo: float,
    hi: float,
    rng: np.random.Generator,
    mutation_prob: float,
    mutation_scale: float,
) -> float:
    if rng.random() < CROSSOVER_PROB:
        alpha = rng.uniform(0.0, 1.0)
        v = alpha * a + (1.0 - alpha) * b
    else:
        v = a if rng.random() < 0.5 else b

    if rng.random() < mutation_prob:
        width = max(hi - lo, 1e-12)
        v = v + rng.normal(0.0, mutation_scale * width)

    if rng.random() < mutation_prob * 0.25:
        v = random_uniform(rng, lo, hi)

    return clamp(v, lo, hi)


def ensure_emulsifier_apeo_free_if_needed(
    row: pd.Series,
    round_key: str,
    raw_x: pd.DataFrame,
    donors: Dict[str, List[int]],
    allowed_names: set,
    rng: np.random.Generator,
) -> pd.Series:
    apeo_required = ROUND_TARGET_WINDOWS[round_key]["APEO_free_required"] and ENABLE_APEO_FILTER

    if not apeo_required:
        return row

    if candidate_is_apeo_free(row, allowed_names):
        return row

    current_total = family_total_from_raw_row(row, "emul")
    if current_total <= 0:
        lo, hi = get_round_numeric_bounds(raw_x, round_key).get("emul_total_pct", (3.0, 17.5))
        current_total = random_uniform(rng, lo, hi)

    donor_list = donors.get("emul_apeo_free", [])
    if not donor_list:
        return row

    donor_idx = int(rng.choice(donor_list))
    donor = raw_x.loc[donor_idx]
    row = copy_family_pattern_from_donor(row, donor, "emul", current_total)

    return row


def make_child_from_parents(
    parent_a: pd.Series,
    parent_b: pd.Series,
    round_key: str,
    raw_x: pd.DataFrame,
    donors: Dict[str, List[int]],
    allowed_names: set,
    rng: np.random.Generator,
    child_id: str,
    generation: int,
) -> pd.Series:
    bounds = get_round_numeric_bounds(raw_x, round_key)
    child = parent_a.copy()

    process_cols = [
        "poly_pressure", "poly_temp", "time", "init_temp",
        "pH_buffer_pct", "dispersant_pct", "coinitiator_pct"
    ]

    for col in process_cols:
        if col in child.index and col in bounds:
            lo, hi = bounds[col]
            a = get_col_value(parent_a, col, lo)
            b = get_col_value(parent_b, col, lo)

            child[col] = blend_value(
                a=a,
                b=b,
                lo=lo,
                hi=hi,
                rng=rng,
                mutation_prob=PROCESS_MUTATION_PROB,
                mutation_scale=GAUSSIAN_MUTATION_SCALE,
            )

    total_to_family = {
        "oxid_total_pct": "oxid",
        "red_total_pct": "red",
        "ter_total_pct": "ter",
        "emul_total_pct": "emul",
        "pc_total_pct": "pc",
    }

    for total_name, fam in total_to_family.items():
        if total_name not in bounds:
            continue

        lo, hi = bounds[total_name]
        total_a = family_total_from_raw_row(parent_a, fam)
        total_b = family_total_from_raw_row(parent_b, fam)

        target_total = blend_value(
            a=total_a,
            b=total_b,
            lo=lo,
            hi=hi,
            rng=rng,
            mutation_prob=FAMILY_TOTAL_MUTATION_PROB,
            mutation_scale=GAUSSIAN_MUTATION_SCALE,
        )

        donor_parent = parent_a if rng.random() < 0.5 else parent_b
        child = copy_family_pattern_from_donor(child, donor_parent, fam, target_total)

        if rng.random() < FAMILY_PATTERN_MUTATION_PROB:
            apeo_required = (
                fam == "emul"
                and ROUND_TARGET_WINDOWS[round_key]["APEO_free_required"]
                and ENABLE_APEO_FILTER
            )
            donor_key = "emul_apeo_free" if apeo_required else fam
            donor_list = donors.get(donor_key, [])

            if donor_list:
                donor_idx = int(rng.choice(donor_list))
                donor = raw_x.loc[donor_idx]
                child = copy_family_pattern_from_donor(child, donor, fam, target_total)

    child = ensure_emulsifier_apeo_free_if_needed(
        row=child,
        round_key=round_key,
        raw_x=raw_x,
        donors=donors,
        allowed_names=allowed_names,
        rng=rng,
    )

    child["_candidate_id"] = child_id
    child["_source_row_index"] = parent_a.get("_source_row_index", np.nan)
    child["_generated_for_round"] = round_key
    child["_generation"] = generation

    return child


# ============================================================
# 13. 模型加载与预测
# ============================================================

def load_all_models() -> Dict[str, dict]:
    models = {}
    for key, cfg in MODEL_CONFIG.items():
        path = resolve_model_path(key, Path(cfg["path"]))
        obj = joblib.load(path)
        model, bundle = extract_model_from_joblib(obj)

        feature_cols = infer_feature_cols_for_target(
            target_key=key,
            model=model,
            bundle=bundle,
            target_dataset=Path(cfg["target_dataset"]),
            target_col=cfg["target_col"],
        )

        models[key] = {
            "model": model,
            "bundle": bundle,
            "path": path,
            "feature_cols": feature_cols,
            "output_is_log1p": bool(cfg.get("output_is_log1p", False)),
            "target_col": cfg["target_col"],
        }

        print(f"[MODEL] {key}: {path}")
        print(f"        feature n = {len(feature_cols)}")
        print(f"        output_is_log1p = {models[key]['output_is_log1p']}")

    return models


def predict_all_targets(
    X_master: pd.DataFrame,
    models: Dict[str, dict]
) -> pd.DataFrame:
    out = pd.DataFrame(index=X_master.index)

    # VC
    X_vc = align_X_for_model(X_master, models["VC"]["feature_cols"], "VC")
    vc_pred_model = np.asarray(models["VC"]["model"].predict(X_vc)).reshape(-1)
    vc_raw = safe_expm1(vc_pred_model) if models["VC"]["output_is_log1p"] else vc_pred_model
    if CLIP_NEGATIVE_RAW_TARGETS:
        vc_raw = np.clip(vc_raw, 0.0, None)

    out["VC_pred_model_scale"] = vc_pred_model
    out["VC_pred_raw"] = vc_raw

    # SC
    X_sc = align_X_for_model(X_master, models["SC"]["feature_cols"], "SC")
    sc_pred = np.asarray(models["SC"]["model"].predict(X_sc)).reshape(-1)
    out["SC_pred"] = sc_pred

    # EC
    X_ec = align_X_for_model(X_master, models["EC"]["feature_cols"], "EC")
    ec_pred = np.asarray(models["EC"]["model"].predict(X_ec)).reshape(-1)
    out["EC_pred"] = ec_pred

    # HCHO corrected MLP
    X_hcho = align_X_for_model(X_master, models["HCHO"]["feature_cols"], "HCHO")
    hcho_pred_model = np.asarray(models["HCHO"]["model"].predict(X_hcho)).reshape(-1)

    if models["HCHO"]["output_is_log1p"]:
        hcho_ppm_unclipped = np.expm1(hcho_pred_model)
    else:
        hcho_ppm_unclipped = hcho_pred_model.copy()

    hcho_clipped_zero = hcho_ppm_unclipped < 0.0
    hcho_ppm = np.clip(hcho_ppm_unclipped, 0.0, None)

    hcho_grade = make_hcho_grade_array(hcho_ppm)

    # A级内部不继续比较极低值。
    hcho_ppm_for_objective = np.where(
        hcho_ppm <= HCHO_GRADE_A_MAX,
        HCHO_GRADE_A_MAX,
        hcho_ppm
    )

    hcho_report_label = [
        make_hcho_grade_label_from_values(g, bool(c))
        for g, c in zip(hcho_grade, hcho_clipped_zero)
    ]

    out["HCHO_pred_log1p_or_model_scale"] = hcho_pred_model
    out["HCHO_pred_ppm_unclipped"] = hcho_ppm_unclipped
    out["HCHO_pred_clipped_zero"] = hcho_clipped_zero
    out["HCHO_pred_ppm"] = hcho_ppm
    out["HCHO_pred_ppm_for_objective"] = hcho_ppm_for_objective
    out["HCHO_grade"] = hcho_grade
    out["HCHO_report_label"] = hcho_report_label
    out["HCHO_is_A_grade"] = hcho_ppm <= HCHO_GRADE_A_MAX
    out["HCHO_is_B_or_better"] = hcho_ppm <= HCHO_GRADE_B_MAX
    out["HCHO_is_C_or_better"] = hcho_ppm <= HCHO_GRADE_C_MAX

    return out


# ============================================================
# 14. 约束特征、评分与目标函数
# ============================================================

def raw_feature_from_master(X: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in X.columns:
        return X[col]
    return pd.Series(default, index=X.index)


def expm1_feature_from_master(X: pd.DataFrame, log_col: str, default=0.0) -> pd.Series:
    if log_col in X.columns:
        return pd.Series(np.expm1(pd.to_numeric(X[log_col], errors="coerce")), index=X.index)
    return pd.Series(default, index=X.index)


def window_violation(y: pd.Series, low: float, high: float) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    below = np.maximum(low - y, 0.0)
    above = np.maximum(y - high, 0.0)
    return pd.Series(below + above, index=y.index)


def window_score(y: pd.Series, low: float, high: float, margin_ratio: float = 0.25) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    width = max(high - low, 1e-12)
    margin = width * margin_ratio

    score = pd.Series(1.0, index=y.index)

    low0 = low - margin
    high0 = high + margin

    below = y < low
    above = y > high

    score.loc[below] = (y.loc[below] - low0) / max(low - low0, 1e-12)
    score.loc[above] = (high0 - y.loc[above]) / max(high0 - high, 1e-12)

    return score.clip(lower=0.0, upper=1.0)


def in_range_score(y: pd.Series, low: float, high: float) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    return ((y >= low) & (y <= high)).astype(float)


def compute_pdp_soft_score(df: pd.DataFrame, round_key: str) -> pd.Series:
    score_parts = []

    # EC
    if "poly_pressure" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["EC"]["poly_pressure"]
        score_parts.append(in_range_score(df["poly_pressure"], lo, hi))

    if "time" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["EC"]["time"]
        score_parts.append(in_range_score(df["time"], lo, hi))

    if "coinitiator_pct" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["EC"]["coinitiator_pct"]
        score_parts.append(in_range_score(df["coinitiator_pct"], lo, hi))

    # SC
    if "poly_temp" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["SC"]["poly_temp"]
        score_parts.append(in_range_score(df["poly_temp"], lo, hi))

    if "emul_total_pct_raw" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["SC"]["emul_total_pct"]
        score_parts.append(in_range_score(df["emul_total_pct_raw"], lo, hi))

    if "emul_mean_EO_number" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["SC"]["emul_mean_EO_number"]
        score_parts.append(in_range_score(df["emul_mean_EO_number"], lo, hi))

    if "coinitiator_pct" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["SC"]["coinitiator_pct"]
        score_parts.append(in_range_score(df["coinitiator_pct"], lo, hi))

    # VC
    if "red_total_pct_raw" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["VC_medium"]["red_total_pct"]
        score_parts.append(in_range_score(df["red_total_pct_raw"], lo, hi))

    if "pc_mean_DP" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["VC_medium"]["pc_mean_DP"]
        score_parts.append(in_range_score(df["pc_mean_DP"], lo, hi))

    # HCHO corrected soft preferences
    if "ter_total_pct_raw" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["HCHO"]["ter_total_pct"]
        score_parts.append(in_range_score(df["ter_total_pct_raw"], lo, hi))

    if "ter_hcho_potential_share" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["HCHO"]["ter_hcho_potential_share"]
        score_parts.append(in_range_score(df["ter_hcho_potential_share"], lo, hi))

    if "oxid_mean_f+" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["HCHO"]["oxid_mean_f+"]
        score_parts.append(in_range_score(df["oxid_mean_f+"], lo, hi))

    if "red_mean_E_HOMO" in df.columns:
        if round_key == "round3_frontier":
            max_val = PDP_SOFT_PREFERENCES["HCHO"]["red_mean_E_HOMO_max_round3"]
        else:
            max_val = PDP_SOFT_PREFERENCES["HCHO"]["red_mean_E_HOMO_max_round2"]
        score_parts.append((df["red_mean_E_HOMO"] <= max_val).astype(float))

    if "emul_total_pct_raw" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["HCHO"]["emul_total_pct"]
        score_parts.append(in_range_score(df["emul_total_pct_raw"], lo, hi))

    if "pc_total_pct_raw" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["HCHO"]["pc_total_pct"]
        score_parts.append(in_range_score(df["pc_total_pct_raw"], lo, hi))

    if "pc_mean_DP" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["HCHO"]["pc_mean_DP"]
        score_parts.append(in_range_score(df["pc_mean_DP"], lo, hi))

    if "red_hcho_potential_share" in df.columns:
        lo, hi = PDP_SOFT_PREFERENCES["HCHO"]["red_hcho_potential_share"]
        score_parts.append(in_range_score(df["red_hcho_potential_share"], lo, hi))

    if not score_parts:
        return pd.Series(0.0, index=df.index)

    mat = pd.concat(score_parts, axis=1)
    return mat.mean(axis=1).fillna(0.0)


def add_master_constraint_features(
    raw_candidates: pd.DataFrame,
    X_master: pd.DataFrame,
    allowed_names: set
) -> pd.DataFrame:
    f = pd.DataFrame(index=X_master.index)

    f["ter_total_pct_raw"] = expm1_feature_from_master(X_master, "ter_total_pct_log1p", 0.0)
    f["emul_total_pct_raw"] = expm1_feature_from_master(X_master, "emul_total_pct_log1p", 0.0)
    f["red_total_pct_raw"] = expm1_feature_from_master(X_master, "red_total_pct_log1p", 0.0)
    f["oxid_total_pct_raw"] = expm1_feature_from_master(X_master, "oxid_total_pct_log1p", 0.0)
    f["pc_total_pct_raw"] = expm1_feature_from_master(X_master, "pc_total_pct_log1p", 0.0)

    for col in [
        "red_hcho_potential_share",
        "ter_hcho_potential_share",
        "oxid_mean_f+",
        "red_mean_E_HOMO",
        "emul_mean_EO_number",
        "pc_mean_DP",
        "pc_mean_degree_of_hydrolysis",
        "oxid_mean_E_LUMO",
        "ter_mean_V(vdw)",
        "red_mean_TPSA",
    ]:
        f[col] = raw_feature_from_master(X_master, col, np.nan)

    for col in ["poly_pressure", "poly_temp", "time", "init_temp", "coinitiator_pct", "pH_buffer_pct", "dispersant_pct"]:
        if col in raw_candidates.columns:
            f[col] = pd.to_numeric(raw_candidates[col], errors="coerce").values

    apeo_list = []
    for _, row in raw_candidates.iterrows():
        apeo_list.append(candidate_is_apeo_free(row, allowed_names))
    f["APEO_free"] = apeo_list

    return f


def evaluate_round_constraints(
    round_key: str,
    pred_df: pd.DataFrame,
    feat_df: pd.DataFrame
) -> pd.DataFrame:
    cfg = ROUND_TARGET_WINDOWS[round_key]
    rules = ROUND_STRUCTURE_RULES[round_key]

    out = pd.DataFrame(index=pred_df.index)

    sc_low, sc_high = cfg["SC_window"]
    vc_low, vc_high = cfg["VC_raw_window"]
    ec_low, ec_high = cfg["EC_window"]
    hcho_max = cfg["HCHO_ppm_max"]

    out["pass_HCHO"] = pred_df["HCHO_pred_ppm"] <= hcho_max
    out["pass_SC"] = (pred_df["SC_pred"] >= sc_low) & (pred_df["SC_pred"] <= sc_high)
    out["pass_VC"] = (pred_df["VC_pred_raw"] >= vc_low) & (pred_df["VC_pred_raw"] <= vc_high)
    out["pass_EC"] = (pred_df["EC_pred"] >= ec_low) & (pred_df["EC_pred"] <= ec_high)

    if cfg["APEO_free_required"] and ENABLE_APEO_FILTER:
        out["pass_APEO"] = feat_df["APEO_free"].astype(bool)
    else:
        out["pass_APEO"] = True

    out["pass_red_hcho_potential"] = feat_df["red_hcho_potential_share"].fillna(0.0) <= rules["red_hcho_potential_share_max"]
    out["pass_ter_hcho_potential"] = feat_df["ter_hcho_potential_share"].fillna(0.0) <= rules["ter_hcho_potential_share_max"]

    out["pass_ter_total_pct"] = feat_df["ter_total_pct_raw"].fillna(0.0) <= rules["ter_total_pct_max"]
    out["pass_emul_total_pct"] = feat_df["emul_total_pct_raw"].fillna(0.0) >= rules["emul_total_pct_min"]
    out["pass_pc_total_pct"] = feat_df["pc_total_pct_raw"].fillna(0.0) <= rules["pc_total_pct_max"]

    if rules["oxid_mean_f+_max"] is not None and "oxid_mean_f+" in feat_df.columns:
        out["pass_oxid_mean_f+"] = feat_df["oxid_mean_f+"].fillna(0.0) <= rules["oxid_mean_f+_max"]
    else:
        out["pass_oxid_mean_f+"] = True

    pass_cols = [c for c in out.columns if c.startswith("pass_")]
    out["is_feasible"] = out[pass_cols].all(axis=1)

    out["score_HCHO_grade"] = hcho_grade_score_from_series(pred_df["HCHO_grade"])
    out["score_SC_window"] = window_score(pred_df["SC_pred"], sc_low, sc_high)
    out["score_VC_window"] = window_score(pred_df["VC_pred_raw"], vc_low, vc_high)
    out["score_EC_window"] = window_score(pred_df["EC_pred"], ec_low, ec_high)

    combined_for_soft = pd.concat([pred_df, feat_df], axis=1)
    out["score_PDP_soft"] = compute_pdp_soft_score(combined_for_soft, round_key)

    soft_homo_max = rules.get("red_mean_E_HOMO_soft_max", None)
    if soft_homo_max is not None and "red_mean_E_HOMO" in feat_df.columns:
        out["score_red_E_HOMO_low_hcho"] = (feat_df["red_mean_E_HOMO"].fillna(999) <= soft_homo_max).astype(float)
    else:
        out["score_red_E_HOMO_low_hcho"] = 0.5

    out["round_score"] = (
        0.30 * out["score_HCHO_grade"]
        + 0.20 * out["score_SC_window"]
        + 0.20 * out["score_VC_window"]
        + 0.20 * out["score_EC_window"]
        + 0.07 * out["score_PDP_soft"]
        + 0.03 * out["score_red_E_HOMO_low_hcho"]
    )

    out["SC_window_violation"] = window_violation(pred_df["SC_pred"], sc_low, sc_high)
    out["VC_window_violation"] = window_violation(pred_df["VC_pred_raw"], vc_low, vc_high)
    out["EC_window_violation"] = window_violation(pred_df["EC_pred"], ec_low, ec_high)
    out["HCHO_violation"] = np.maximum(pred_df["HCHO_pred_ppm"] - hcho_max, 0.0)
    out["pc_total_pct_violation"] = np.maximum(feat_df["pc_total_pct_raw"] - rules["pc_total_pct_max"], 0.0)

    return out


def compute_objectives_and_cv(round_key: str, result_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cfg = ROUND_TARGET_WINDOWS[round_key]
    sc_low, sc_high = cfg["SC_window"]
    vc_low, vc_high = cfg["VC_raw_window"]
    ec_low, ec_high = cfg["EC_window"]
    hcho_max = cfg["HCHO_ppm_max"]

    sc_center = 0.5 * (sc_low + sc_high)
    vc_center = 0.5 * (vc_low + vc_high)
    ec_center = 0.5 * (ec_low + ec_high)

    sc_half = max((sc_high - sc_low) / 2.0, 1e-12)
    vc_half = max((vc_high - vc_low) / 2.0, 1e-12)
    ec_half = max((ec_high - ec_low) / 2.0, 1e-12)

    if "HCHO_pred_ppm_for_objective" in result_df.columns:
        hcho_obj = result_df["HCHO_pred_ppm_for_objective"].values
    else:
        hcho_raw = result_df["HCHO_pred_ppm"].values
        hcho_obj = np.where(hcho_raw <= HCHO_GRADE_A_MAX, HCHO_GRADE_A_MAX, hcho_raw)

    f1 = hcho_obj / max(hcho_max, 1e-12)
    f2 = np.abs(result_df["SC_pred"].values - sc_center) / sc_half
    f3 = np.abs(result_df["VC_pred_raw"].values - vc_center) / vc_half
    f4 = np.abs(result_df["EC_pred"].values - ec_center) / ec_half

    F = np.column_stack([f1, f2, f3, f4]).astype(float)

    cv = np.zeros(len(result_df), dtype=float)
    cv += result_df["HCHO_violation"].values / max(hcho_max, 1e-12)
    cv += result_df["SC_window_violation"].values / max(sc_high - sc_low, 1e-12)
    cv += result_df["VC_window_violation"].values / max(vc_high - vc_low, 1e-12)
    cv += result_df["EC_window_violation"].values / max(ec_high - ec_low, 1e-12)
    cv += result_df["pc_total_pct_violation"].values / max(PC_TOTAL_PCT_MAX, 1e-12)

    pass_cols = [c for c in result_df.columns if c.startswith("pass_")]
    for c in pass_cols:
        cv += (~result_df[c].astype(bool)).astype(float).values

    return F, cv


# ============================================================
# 15. NSGA-II 排序与选择
# ============================================================

def dominates_constrained(i: int, j: int, F: np.ndarray, CV: np.ndarray) -> bool:
    fi = F[i]
    fj = F[j]
    cvi = CV[i]
    cvj = CV[j]

    i_feas = cvi <= 1e-12
    j_feas = cvj <= 1e-12

    if i_feas and not j_feas:
        return True
    if not i_feas and j_feas:
        return False
    if not i_feas and not j_feas:
        return cvi < cvj

    return np.all(fi <= fj) and np.any(fi < fj)


def fast_non_dominated_sort_constrained(F: np.ndarray, CV: np.ndarray) -> List[List[int]]:
    n = F.shape[0]
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates_constrained(p, q, F, CV):
                S[p].append(q)
            elif dominates_constrained(q, p, F, CV):
                n_dom[p] += 1

        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if len(fronts[-1]) == 0:
        fronts.pop()

    return fronts


def crowding_distance(F: np.ndarray, front: List[int]) -> np.ndarray:
    if len(front) == 0:
        return np.array([])

    m = F.shape[1]
    dist = np.zeros(len(front), dtype=float)

    if len(front) <= 2:
        dist[:] = np.inf
        return dist

    F_front = F[front, :]

    for k in range(m):
        order = np.argsort(F_front[:, k])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf

        f_min = F_front[order[0], k]
        f_max = F_front[order[-1], k]

        if abs(f_max - f_min) < 1e-12:
            continue

        for r in range(1, len(front) - 1):
            dist[order[r]] += (
                F_front[order[r + 1], k] - F_front[order[r - 1], k]
            ) / (f_max - f_min)

    return dist


def nsga2_rank_and_crowding(F: np.ndarray, CV: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    fronts = fast_non_dominated_sort_constrained(F, CV)
    rank = np.full(F.shape[0], 10**9, dtype=int)
    crowd = np.zeros(F.shape[0], dtype=float)

    for i, front in enumerate(fronts):
        cd = crowding_distance(F, front)
        for local_idx, global_idx in enumerate(front):
            rank[global_idx] = i
            crowd[global_idx] = cd[local_idx]

    return rank, crowd, fronts


def nsga2_select_indices(F: np.ndarray, CV: np.ndarray, n_select: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank, crowd, fronts = nsga2_rank_and_crowding(F, CV)

    selected = []

    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
        else:
            cd = crowding_distance(F, front)
            order = np.argsort(-cd)
            remaining = n_select - len(selected)
            selected.extend([front[i] for i in order[:remaining]])
            break

    selected = np.array(selected, dtype=int)
    return selected, rank, crowd


def tournament_select(
    rank: np.ndarray,
    crowd: np.ndarray,
    CV: np.ndarray,
    rng: np.random.Generator
) -> int:
    i = int(rng.integers(0, len(rank)))
    j = int(rng.integers(0, len(rank)))

    if CV[i] <= 1e-12 and CV[j] > 1e-12:
        return i
    if CV[j] <= 1e-12 and CV[i] > 1e-12:
        return j

    if CV[i] > 1e-12 and CV[j] > 1e-12:
        return i if CV[i] < CV[j] else j

    if rank[i] < rank[j]:
        return i
    if rank[j] < rank[i]:
        return j

    return i if crowd[i] > crowd[j] else j


def non_dominated_mask_unconstrained(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not mask[i]:
            continue
        dominated_by_any = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominated_by_any[i] = False
        if np.any(dominated_by_any):
            mask[i] = False

    return mask


# ============================================================
# 16. 评估种群
# ============================================================

def evaluate_population(
    raw_pop: pd.DataFrame,
    round_key: str,
    builder,
    models: Dict[str, dict],
    allowed_names: set,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_pop = raw_pop.reset_index(drop=True)

    internal_cols = ["_candidate_id", "_source_row_index", "_generated_for_round", "_generation"]
    cand_for_builder = raw_pop.drop(columns=[c for c in internal_cols if c in raw_pop.columns], errors="ignore")

    X_master = builder.transform_raw_to_master(cand_for_builder)
    X_master = X_master.reset_index(drop=True)

    pred_df = predict_all_targets(X_master, models)
    pred_df = pred_df.reset_index(drop=True)

    feat_df = add_master_constraint_features(raw_pop, X_master, allowed_names)
    feat_df = feat_df.reset_index(drop=True)

    eval_df = evaluate_round_constraints(round_key, pred_df, feat_df)
    eval_df = eval_df.reset_index(drop=True)

    meta_cols = [c for c in internal_cols if c in raw_pop.columns]
    result_df = pd.concat(
        [
            raw_pop[meta_cols].reset_index(drop=True),
            pred_df,
            feat_df,
            eval_df,
        ],
        axis=1,
    )

    result_df.insert(0, "_row_id", np.arange(len(result_df)))

    return result_df, X_master


def create_offspring_population(
    raw_pop: pd.DataFrame,
    result_df: pd.DataFrame,
    round_key: str,
    raw_x: pd.DataFrame,
    donors: Dict[str, List[int]],
    allowed_names: set,
    rng: np.random.Generator,
    generation: int,
    n_offspring: int,
) -> pd.DataFrame:
    F, CV = compute_objectives_and_cv(round_key, result_df)
    rank, crowd, _ = nsga2_rank_and_crowding(F, CV)

    rows = []

    for i in range(n_offspring):
        idx_a = tournament_select(rank, crowd, CV, rng)
        idx_b = tournament_select(rank, crowd, CV, rng)

        parent_a = raw_pop.iloc[idx_a]
        parent_b = raw_pop.iloc[idx_b]

        child = make_child_from_parents(
            parent_a=parent_a,
            parent_b=parent_b,
            round_key=round_key,
            raw_x=raw_x,
            donors=donors,
            allowed_names=allowed_names,
            rng=rng,
            child_id=f"{round_key}_g{generation:03d}_child_{i:07d}",
            generation=generation,
        )

        rows.append(child)

    return pd.DataFrame(rows).reset_index(drop=True)


# ============================================================
# 17. 输出与推荐
# ============================================================

def get_pareto_feasible(round_key: str, feasible_df: pd.DataFrame) -> pd.DataFrame:
    if len(feasible_df) == 0:
        return feasible_df.copy()

    F, _ = compute_objectives_and_cv(round_key, feasible_df)
    mask = non_dominated_mask_unconstrained(F)
    pareto = feasible_df.loc[mask].copy()

    pareto["HCHO_grade_priority"] = hcho_grade_priority_series(pareto["HCHO_grade"])
    pareto["HCHO_unclipped_priority"] = (~pareto["HCHO_pred_clipped_zero"].astype(bool)).astype(int)

    pareto = pareto.sort_values(
        ["HCHO_grade_priority", "round_score", "HCHO_unclipped_priority"],
        ascending=[False, False, False]
    )

    return pareto


def build_recommended_top(round_key: str, feasible_df: pd.DataFrame) -> pd.DataFrame:
    if len(feasible_df) == 0:
        return feasible_df.copy()

    df = feasible_df.copy()

    cfg = ROUND_TARGET_WINDOWS[round_key]
    sc_low, sc_high = cfg["SC_window"]
    vc_low, vc_high = cfg["VC_raw_window"]
    ec_low, ec_high = cfg["EC_window"]

    df["SC_center_dist"] = np.abs(df["SC_pred"] - 0.5 * (sc_low + sc_high))
    df["VC_center_dist"] = np.abs(df["VC_pred_raw"] - 0.5 * (vc_low + vc_high))
    df["EC_center_dist"] = np.abs(df["EC_pred"] - 0.5 * (ec_low + ec_high))

    df["HCHO_grade_priority"] = hcho_grade_priority_series(df["HCHO_grade"])
    df["HCHO_unclipped_priority"] = (~df["HCHO_pred_clipped_zero"].astype(bool)).astype(int)

    # 推荐排序：
    # 1) HCHO 等级优先；
    # 2) 综合得分；
    # 3) 同等级内优先未截断；
    # 4) 再看 SC/VC/EC 是否靠近窗口中心。
    df = df.sort_values(
        [
            "HCHO_grade_priority",
            "round_score",
            "HCHO_unclipped_priority",
            "SC_center_dist",
            "VC_center_dist",
            "EC_center_dist",
            "HCHO_pred_ppm_for_objective",
        ],
        ascending=[
            False,
            False,
            False,
            True,
            True,
            True,
            True,
        ],
    )

    return df


def safe_to_excel(df: pd.DataFrame, path: Path, index=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="data", index=index)



def make_X_master_export_table(
    ref_df: pd.DataFrame,
    X_part: pd.DataFrame,
    extra_meta_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    将 X_master 与候选元信息拼接后导出。

    目的：
    1) 让后续 OOD / AD / 候选闭环脚本可以直接用 _candidate_id 匹配 C1-C8；
    2) 保留 _row_id，使 X_master 能准确回溯 final_raw/final_result；
    3) 同时保留预测值、HCHO 等级、round_score、可行性和 NSGA-II 排名等关键信息，
       便于检查某个候选为什么进入/没有进入最终推荐集合。
    """
    if extra_meta_cols is None:
        extra_meta_cols = []

    default_meta_cols = [
        "_row_id",
        "_candidate_id",
        "_source_row_index",
        "_generated_for_round",
        "_generation",
        "is_feasible",
        "round_score",
        "nsga2_rank",
        "nsga2_crowding",
        "nsga2_constraint_violation",
        "HCHO_grade",
        "HCHO_report_label",
        "HCHO_pred_ppm",
        "HCHO_pred_ppm_unclipped",
        "HCHO_pred_clipped_zero",
        "HCHO_pred_ppm_for_objective",
        "VC_pred_raw",
        "SC_pred",
        "EC_pred",
        "HCHO_is_A_grade",
        "HCHO_is_B_or_better",
        "HCHO_is_C_or_better",
    ]

    meta_cols = []
    for c in default_meta_cols + list(extra_meta_cols):
        if c in ref_df.columns and c not in meta_cols:
            meta_cols.append(c)

    meta = ref_df[meta_cols].reset_index(drop=True).copy()
    X_part = X_part.reset_index(drop=True).copy()

    # 防止 X_master 中偶然已有同名元信息列造成重复列名。
    X_part = X_part.drop(columns=[c for c in meta.columns if c in X_part.columns], errors="ignore")
    return pd.concat([meta, X_part], axis=1)


def make_raw_formula_export_table(
    ref_df: pd.DataFrame,
    raw_part: pd.DataFrame,
    extra_meta_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    将原始配方与候选元信息拼接后导出，便于从 C1-C8 回溯真实原料槽位。
    """
    if extra_meta_cols is None:
        extra_meta_cols = []

    default_meta_cols = [
        "_row_id",
        "_candidate_id",
        "_source_row_index",
        "_generated_for_round",
        "_generation",
        "is_feasible",
        "round_score",
        "nsga2_rank",
        "nsga2_crowding",
        "nsga2_constraint_violation",
        "HCHO_grade",
        "HCHO_report_label",
        "HCHO_pred_ppm",
        "VC_pred_raw",
        "SC_pred",
        "EC_pred",
    ]

    meta_cols = []
    for c in default_meta_cols + list(extra_meta_cols):
        if c in ref_df.columns and c not in meta_cols:
            meta_cols.append(c)

    meta = ref_df[meta_cols].reset_index(drop=True).copy()
    raw_part = raw_part.reset_index(drop=True).copy()
    raw_part = raw_part.drop(columns=[c for c in meta.columns if c in raw_part.columns], errors="ignore")
    return pd.concat([meta, raw_part], axis=1)

def export_round_outputs(
    out_dir: Path,
    round_key: str,
    final_raw: pd.DataFrame,
    final_result: pd.DataFrame,
    final_X: pd.DataFrame,
    history_df: pd.DataFrame,
):
    round_dir = ensure_dir(out_dir / round_key)

    feasible = final_result[final_result["is_feasible"]].copy()
    feasible["HCHO_grade_priority"] = hcho_grade_priority_series(feasible["HCHO_grade"])
    feasible["HCHO_unclipped_priority"] = (~feasible["HCHO_pred_clipped_zero"].astype(bool)).astype(int)

    feasible = feasible.sort_values(
        ["HCHO_grade_priority", "round_score", "HCHO_unclipped_priority", "HCHO_pred_ppm_for_objective"],
        ascending=[False, False, False, True]
    )

    pareto = get_pareto_feasible(round_key, feasible)
    recommended = build_recommended_top(round_key, feasible)

    # --------------------------------------------------------
    # 1) 常规结果输出
    # --------------------------------------------------------
    final_result.to_csv(round_dir / f"{round_key}_nsga2_final_population.csv", index=False, encoding="utf-8-sig")
    feasible.to_csv(round_dir / f"{round_key}_nsga2_feasible_candidates.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(round_dir / f"{round_key}_nsga2_pareto_candidates.csv", index=False, encoding="utf-8-sig")
    recommended.to_csv(round_dir / f"{round_key}_nsga2_recommended_candidates.csv", index=False, encoding="utf-8-sig")
    history_df.to_csv(round_dir / f"{round_key}_nsga2_generation_history.csv", index=False, encoding="utf-8-sig")

    safe_to_excel(final_result.head(MAX_EXPORT_POPULATION), round_dir / f"{round_key}_nsga2_final_population_top.xlsx", index=False)
    safe_to_excel(feasible.head(MAX_EXPORT_FEASIBLE), round_dir / f"{round_key}_nsga2_feasible_top.xlsx", index=False)
    safe_to_excel(pareto.head(MAX_EXPORT_PARETO), round_dir / f"{round_key}_nsga2_pareto_top.xlsx", index=False)
    safe_to_excel(recommended.head(MAX_EXPORT_RECOMMENDED), round_dir / f"{round_key}_nsga2_recommended_top.xlsx", index=False)

    # --------------------------------------------------------
    # 2) 新增：导出完整 800 个最终种群的 X_master
    #    用途：当某个最终 C1-C8 候选没有进入 recommended_top_200 时，
    #    仍然可以根据 _candidate_id / _row_id 找回完整模型输入特征。
    # --------------------------------------------------------
    final_X_all_export = make_X_master_export_table(final_result, final_X)
    final_X_all_export.to_csv(
        round_dir / f"{round_key}_nsga2_final_population_X_master_all.csv",
        index=False,
        encoding="utf-8-sig"
    )
    final_X_all_export.to_excel(
        round_dir / f"{round_key}_nsga2_final_population_X_master_all.xlsx",
        index=False
    )

    # --------------------------------------------------------
    # 3) 新增：导出完整 recommended/feasible 候选的 raw formula 和 X_master
    #    注意：recommended 是对 feasible candidates 排序后的全集，不再只导出 top 200。
    # --------------------------------------------------------
    if len(recommended) > 0:
        idx_full = recommended["_row_id"].astype(int).values
        raw_full = final_raw.iloc[idx_full].copy()
        X_full = final_X.iloc[idx_full].copy()

        raw_full_export = make_raw_formula_export_table(recommended, raw_full)
        X_full_export = make_X_master_export_table(recommended, X_full)

        raw_full_export.to_csv(
            round_dir / f"{round_key}_nsga2_recommended_raw_formula_full.csv",
            index=False,
            encoding="utf-8-sig"
        )
        X_full_export.to_csv(
            round_dir / f"{round_key}_nsga2_recommended_X_master_full.csv",
            index=False,
            encoding="utf-8-sig"
        )
        raw_full_export.to_excel(
            round_dir / f"{round_key}_nsga2_recommended_raw_formula_full.xlsx",
            index=False
        )
        X_full_export.to_excel(
            round_dir / f"{round_key}_nsga2_recommended_X_master_full.xlsx",
            index=False
        )

        # 兼容你之前 OOD 脚本可能优先查找的非 nsga2 命名。
        # 文件内容与 nsga2_recommended_X_master_full 完全一致。
        X_full_export.to_excel(
            round_dir / f"{round_key}_recommended_X_master_full_v1.xlsx",
            index=False
        )
        raw_full_export.to_excel(
            round_dir / f"{round_key}_recommended_raw_formula_full_v1.xlsx",
            index=False
        )

        # 保留原来的 top 文件，但现在 top 文件也带上 _candidate_id / _row_id / 预测值等元信息。
        idx_top = recommended.head(MAX_EXPORT_RECOMMENDED)["_row_id"].astype(int).values
        raw_top = final_raw.iloc[idx_top].copy()
        X_top = final_X.iloc[idx_top].copy()

        raw_top_export = make_raw_formula_export_table(recommended.head(MAX_EXPORT_RECOMMENDED), raw_top)
        X_top_export = make_X_master_export_table(recommended.head(MAX_EXPORT_RECOMMENDED), X_top)

        raw_top_export.to_excel(round_dir / f"{round_key}_nsga2_recommended_raw_formula_top.xlsx", index=False)
        X_top_export.to_excel(round_dir / f"{round_key}_nsga2_recommended_X_master_top.xlsx", index=False)

    return {
        "round": round_key,
        "n_final_population": len(final_result),
        "n_feasible": len(feasible),
        "n_pareto": len(pareto),
        "n_recommended": len(recommended),
        "feasible_ratio": len(feasible) / max(len(final_result), 1),

        "HCHO_min_ppm": float(final_result["HCHO_pred_ppm"].min()) if len(final_result) else np.nan,
        "HCHO_median_ppm": float(final_result["HCHO_pred_ppm"].median()) if len(final_result) else np.nan,
        "HCHO_A_ratio": float(final_result["HCHO_is_A_grade"].mean()) if len(final_result) else np.nan,
        "HCHO_B_or_better_ratio": float(final_result["HCHO_is_B_or_better"].mean()) if len(final_result) else np.nan,
        "HCHO_C_or_better_ratio": float(final_result["HCHO_is_C_or_better"].mean()) if len(final_result) else np.nan,
        "HCHO_clipped_zero_ratio": float(final_result["HCHO_pred_clipped_zero"].mean()) if len(final_result) else np.nan,

        "SC_median": float(final_result["SC_pred"].median()) if len(final_result) else np.nan,
        "VC_median_raw": float(final_result["VC_pred_raw"].median()) if len(final_result) else np.nan,
        "EC_median": float(final_result["EC_pred"].median()) if len(final_result) else np.nan,

        "pc_total_pct_raw_max": float(final_result["pc_total_pct_raw"].max()) if len(final_result) else np.nan,
    }


# ============================================================
# 18. 单轮 NSGA-II
# ============================================================

def run_nsga2_for_round(
    round_key: str,
    raw_x: pd.DataFrame,
    builder,
    models: Dict[str, dict],
    allowed_names: set,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 100)
    print(f"NSGA-II 开始：{round_key} | {ROUND_TARGET_WINDOWS[round_key]['round_label']}")
    print("=" * 100)

    donors = build_donor_indices(raw_x, allowed_names)

    print(f"[{round_key}] 初始化随机候选：{NSGA2_INIT_RANDOM_SIZE}")
    raw_init = generate_random_candidates(
        raw_x=raw_x,
        round_key=round_key,
        n_candidates=NSGA2_INIT_RANDOM_SIZE,
        allowed_names=allowed_names,
        rng=rng,
    )

    result_init, _ = evaluate_population(
        raw_pop=raw_init,
        round_key=round_key,
        builder=builder,
        models=models,
        allowed_names=allowed_names,
    )

    F_init, CV_init = compute_objectives_and_cv(round_key, result_init)
    selected, _, _ = nsga2_select_indices(F_init, CV_init, NSGA2_POP_SIZE)

    raw_pop = raw_init.iloc[selected].copy().reset_index(drop=True)

    history_rows = []

    for gen in range(1, NSGA2_N_GEN + 1):
        result_pop, _ = evaluate_population(
            raw_pop=raw_pop,
            round_key=round_key,
            builder=builder,
            models=models,
            allowed_names=allowed_names,
        )

        F_pop, CV_pop = compute_objectives_and_cv(round_key, result_pop)

        n_feasible = int(result_pop["is_feasible"].sum())
        best_hcho = float(result_pop["HCHO_pred_ppm"].min())
        med_hcho = float(result_pop["HCHO_pred_ppm"].median())
        best_score = float(result_pop["round_score"].max())

        history_rows.append({
            "round": round_key,
            "generation": gen,
            "population_size": len(result_pop),
            "n_feasible": n_feasible,
            "feasible_ratio": n_feasible / max(len(result_pop), 1),

            "HCHO_min_ppm": best_hcho,
            "HCHO_median_ppm": med_hcho,
            "HCHO_A_ratio": float(result_pop["HCHO_is_A_grade"].mean()),
            "HCHO_B_or_better_ratio": float(result_pop["HCHO_is_B_or_better"].mean()),
            "HCHO_C_or_better_ratio": float(result_pop["HCHO_is_C_or_better"].mean()),
            "HCHO_clipped_zero_ratio": float(result_pop["HCHO_pred_clipped_zero"].mean()),

            "round_score_max": best_score,
            "SC_median": float(result_pop["SC_pred"].median()),
            "VC_median_raw": float(result_pop["VC_pred_raw"].median()),
            "EC_median": float(result_pop["EC_pred"].median()),
            "pc_total_pct_raw_max": float(result_pop["pc_total_pct_raw"].max()),
        })

        if gen == 1 or gen % 10 == 0 or gen == NSGA2_N_GEN:
            print(
                f"[{round_key}] Gen {gen:03d}/{NSGA2_N_GEN} | "
                f"feasible={n_feasible}/{len(result_pop)} | "
                f"HCHO_min={best_hcho:.4f} | "
                f"HCHO_A={result_pop['HCHO_is_A_grade'].mean():.3f} | "
                f"score_max={best_score:.4f}"
            )

        offspring = create_offspring_population(
            raw_pop=raw_pop,
            result_df=result_pop,
            round_key=round_key,
            raw_x=raw_x,
            donors=donors,
            allowed_names=allowed_names,
            rng=rng,
            generation=gen,
            n_offspring=NSGA2_OFFSPRING_SIZE,
        )

        combined_raw = pd.concat([raw_pop, offspring], axis=0, ignore_index=True)

        combined_result, _ = evaluate_population(
            raw_pop=combined_raw,
            round_key=round_key,
            builder=builder,
            models=models,
            allowed_names=allowed_names,
        )

        F_combined, CV_combined = compute_objectives_and_cv(round_key, combined_result)
        selected, _, _ = nsga2_select_indices(F_combined, CV_combined, NSGA2_POP_SIZE)

        raw_pop = combined_raw.iloc[selected].copy().reset_index(drop=True)

    final_result, final_X = evaluate_population(
        raw_pop=raw_pop,
        round_key=round_key,
        builder=builder,
        models=models,
        allowed_names=allowed_names,
    )

    final_F, final_CV = compute_objectives_and_cv(round_key, final_result)
    rank, crowd, _ = nsga2_rank_and_crowding(final_F, final_CV)

    final_result["nsga2_rank"] = rank
    final_result["nsga2_crowding"] = crowd
    final_result["nsga2_constraint_violation"] = final_CV

    history_df = pd.DataFrame(history_rows)

    return raw_pop.reset_index(drop=True), final_result.reset_index(drop=True), final_X.reset_index(drop=True), history_df


# ============================================================
# 19. 主程序
# ============================================================

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    out_dir = ensure_dir(OUTPUT_DIR)

    print("=" * 100)
    print("V2 NSGA-II 三轮多目标优化启动：PC8 + corrected HCHO MLP + HCHO A/B/C 分级")
    print("=" * 100)
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"OUTPUT_DIR = {out_dir}")
    print(f"POP_SIZE = {NSGA2_POP_SIZE}")
    print(f"INIT_RANDOM_SIZE = {NSGA2_INIT_RANDOM_SIZE}")
    print(f"N_GEN = {NSGA2_N_GEN}")
    print(f"OFFSPRING_SIZE = {NSGA2_OFFSPRING_SIZE}")
    print(f"PC_TOTAL_PCT_MAX = {PC_TOTAL_PCT_MAX}")
    print(f"HCHO A <= {HCHO_GRADE_A_MAX} ppm")
    print(f"HCHO B <= {HCHO_GRADE_B_MAX} ppm")
    print(f"HCHO C <= {HCHO_GRADE_C_MAX} ppm")

    print("\n[1/6] 读取原始数据")
    raw_df = read_raw_data_from_v0_config()
    print("raw_df shape =", raw_df.shape)

    raw_x = raw_df.drop(
        columns=[c for c in TARGET_COLS_RAW if c in raw_df.columns],
        errors="ignore"
    ).copy()
    print("raw_x shape =", raw_x.shape)

    print("\n[2/6] 读取 APEO 标注表")
    allowed_names, _ = load_apeo_allowed_names(APEO_TABLE_PATH)
    print("APEO allowed names =", len(allowed_names))

    print("\n[3/6] 初始化冻结版特征生成器")
    builder = FrozenFeatureBuilder(V0_CONFIG)

    print("\n[4/6] 加载四个代理模型")
    models = load_all_models()

    print("\n[5/6] 逐轮运行 NSGA-II")
    round_summaries = []
    all_history = []

    for round_key in ROUND_TARGET_WINDOWS.keys():
        final_raw, final_result, final_X, history_df = run_nsga2_for_round(
            round_key=round_key,
            raw_x=raw_x,
            builder=builder,
            models=models,
            allowed_names=allowed_names,
            rng=rng,
        )

        summary = export_round_outputs(
            out_dir=out_dir,
            round_key=round_key,
            final_raw=final_raw,
            final_result=final_result,
            final_X=final_X,
            history_df=history_df,
        )

        round_summaries.append(summary)
        all_history.append(history_df)

    print("\n[6/6] 输出总汇总")
    summary_df = pd.DataFrame(round_summaries)
    summary_path = out_dir / "V2_NSGAII_round123_PC8_HCHO_MLP_corrected_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)

    if all_history:
        all_history_df = pd.concat(all_history, axis=0, ignore_index=True)
        all_history_df.to_csv(
            out_dir / "V2_NSGAII_round123_PC8_HCHO_MLP_corrected_all_generation_history.csv",
            index=False,
            encoding="utf-8-sig"
        )

    config_record = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_DIR": str(DATA_DIR),
        "APEO_TABLE_PATH": str(APEO_TABLE_PATH),
        "OUTPUT_DIR": str(OUTPUT_DIR),

        "RANDOM_SEED": RANDOM_SEED,
        "NSGA2_POP_SIZE": NSGA2_POP_SIZE,
        "NSGA2_INIT_RANDOM_SIZE": NSGA2_INIT_RANDOM_SIZE,
        "NSGA2_N_GEN": NSGA2_N_GEN,
        "NSGA2_OFFSPRING_SIZE": NSGA2_OFFSPRING_SIZE,

        "PC_TOTAL_PCT_MAX": PC_TOTAL_PCT_MAX,
        "HCHO_GRADE_A_MAX": HCHO_GRADE_A_MAX,
        "HCHO_GRADE_B_MAX": HCHO_GRADE_B_MAX,
        "HCHO_GRADE_C_MAX": HCHO_GRADE_C_MAX,
        "HCHO_PROJECT_SPEC_MAX": HCHO_PROJECT_SPEC_MAX,

        "ROUND_TARGET_WINDOWS": ROUND_TARGET_WINDOWS,
        "ROUND_INPUT_BOUNDS": ROUND_INPUT_BOUNDS,
        "ROUND_STRUCTURE_RULES": ROUND_STRUCTURE_RULES,
        "PDP_SOFT_PREFERENCES": PDP_SOFT_PREFERENCES,

        "MODEL_CONFIG": {
            k: {
                "path": str(v["path"]),
                "target_col": v["target_col"],
                "target_dataset": str(v["target_dataset"]),
                "output_is_log1p": v["output_is_log1p"],
            }
            for k, v in MODEL_CONFIG.items()
        },

        "objective_definition": [
            "f1 = HCHO_pred_ppm_for_objective / HCHO_ppm_max, minimized",
            "HCHO_pred_ppm_for_objective treats A-grade HCHO values <=1.5 ppm as 1.5 ppm to avoid over-ranking low-end numerical noise.",
            "f2 = abs(SC_pred - SC_window_center) / SC_window_half_width, minimized",
            "f3 = abs(VC_pred_raw - VC_window_center) / VC_window_half_width, minimized",
            "f4 = abs(EC_pred - EC_window_center) / EC_window_half_width, minimized",
        ],

        "constraint_notes": [
            "Corrected HCHO model is MLP direct regression.",
            "HCHO target column is Free HCHO Content.",
            "HCHO model output is log1p(ppm); HCHO_pred_ppm is raw ppm.",
            "HCHO engineering grades: A<=1.5 ppm, B<=5 ppm, C<=30 ppm, D>30 ppm.",
            "Round 1 uses HCHO<=30 ppm, Round 2 uses HCHO<=5 ppm, Round 3 uses HCHO<=1.5 ppm.",
            "HCHO_pred_ppm_unclipped and HCHO_pred_clipped_zero are exported for diagnostics.",
            "pc_total_pct_raw <= 8.0 is enforced.",
            "Round 2/3 require APEO-free.",
            "Round 2/3 require red_hcho_potential_share=0.",
            "Round 3 requires ter_hcho_potential_share=0, ter_total_pct<=6, oxid_mean_f+<=0.15.",
            "PDP soft preferences are derived from corrected HCHO SHAP/1D PDP/2D PDP analysis.",
        ],
    }

    config_path = out_dir / "V2_NSGAII_round123_PC8_HCHO_MLP_corrected_run_config.json"
    config_path.write_text(json.dumps(config_record, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n全部完成。")
    print("总汇总文件：", summary_path)
    print("配置文件：", config_path)
    print("输出目录：", out_dir)
    print("\n重点检查：")
    print("1) V2_NSGAII_round123_PC8_HCHO_MLP_corrected_summary.xlsx")
    print("2) round2_high_end/round2_high_end_nsga2_recommended_top.xlsx")
    print("3) round2_high_end/round2_high_end_nsga2_pareto_top.xlsx")
    print("4) round3_frontier/round3_frontier_nsga2_recommended_top.xlsx")
    print("5) round3_frontier/round3_frontier_nsga2_pareto_top.xlsx")
    print("6) round3_frontier/round3_frontier_nsga2_recommended_raw_formula_top.xlsx")
    print("7) round3_frontier/round3_frontier_nsga2_recommended_X_master_full.xlsx  # 新增：完整 recommended X_master")
    print("8) round3_frontier/round3_frontier_nsga2_final_population_X_master_all.xlsx  # 新增：完整 800 个最终种群 X_master")
    print("9) HCHO_grade、HCHO_report_label、HCHO_pred_ppm_unclipped、HCHO_pred_clipped_zero")
    print("10) pc_total_pct_raw 是否全部 <= 8")


if __name__ == "__main__":
    main()