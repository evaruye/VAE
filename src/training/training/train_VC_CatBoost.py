# -*- coding: utf-8 -*-
r"""
CatBoost 最终版 V2（viscosity_log1p / 全链路 log 标度）
仅保留：
1) NO_OUTLIER_DELETE（不删除异常值）
2) objective_best（仅保留 Optuna + 10-fold CV 选出的最优参数版本）

V2 核心策略：
- 从 V1 的“超窄局部精搜”改为“controlled escape 受控跳出局部最优”
- 保留 CatBoost 当前有效结构的大方向：
    bootstrap_type = MVS
    grow_policy = Depthwise
    boosting_type = Plain
- 重新放开关键泛化参数：
    iterations              = 700 ~ 2200
    learning_rate           = 0.025 ~ 0.095
    depth                   = [4, 5, 6, 7, 8]
    l2_leaf_reg             = 0.7 ~ 12.0
    random_strength         = 6.0 ~ 25.0
    rsm                     = 0.45 ~ 0.90
    subsample               = 0.65 ~ 0.95
    min_data_in_leaf         = [1, 5, 10, 20, 30, 45, 60]
    border_count             = [32, 64, 128, 254]
    leaf_estimation_iterations = [3, 5, 8]

泛化增强：
- objective 不再只最小化 OOF RMSE
- 使用轻微 gap penalty：
    score = oof_rmse + GAP_PENALTY_WEIGHT * max(0, gap_rmse_cv - GAP_SAFE_THR)

说明：
- 当前脚本假设 TARGET_COL 已经是 log1p(viscosity)
- 输出 train_train / train_valid / train_oof / train_full / test 全部指标
- 输出 OOF / train_full / test 的预测表与散点图，方便 Origin 作图
"""

import os
from pathlib import Path
import json
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import optuna
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# ====================== 0. 基本配置 ======================

MODEL_TAG = "catboost"

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_XLSX = str(REPO_ROOT / "data" / "Data_S2" / "VC_Modeling_Data.xlsx")

# 如果你的目标列真实名称是 viscosity_log1p，请改这里
TARGET_COL = "Viscosity"

EXCLUDE_ISNA_COLS = [
    "oxidizer_pct2_isna",
    "init_pressure_isna",
    "time_isna",
    "pH_buffer_pct_isna",
]

N_SPLITS_LIST = [10]
RANDOM_SEED = 23

# 建议先跑 500 次；如果结果接近 0.90，可再加到 800–1000 次
N_TRIALS_SPLIT = 1000

TEST_SIZE = 0.1
N_JOBS = 1

# 为避免 Windows 路径过长，使用短输出根目录
SHORT_OUTPUT_ROOT = str(REPO_ROOT / "results" / "training" / "VC")
os.makedirs(SHORT_OUTPUT_ROOT, exist_ok=True)

OUTPUT_DIR_BASE = os.path.join(SHORT_OUTPUT_ROOT, f"{MODEL_TAG}_VFINAL_V7")
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

# ======================
CB_BOOSTING_TYPE = "Plain"
CB_BOOTSTRAP_TYPE_CHOICES = ["Bernoulli"]
CB_GROW_POLICY_CHOICES = ["SymmetricTree"]
CB_DEPTH_CHOICES = [3]                             # 保持最浅深度
CB_MIN_DATA_IN_LEAF_CHOICES = [120, 150]            # 进一步提高叶子最小样本数（原100-120）
CB_BORDER_COUNT_CHOICES = [24, 32]                  # 进一步减少分箱数（原32-48）
CB_LEAF_ESTIMATION_ITERATIONS_CHOICES = [5, 8]      # 减少迭代次数（去掉10）
CB_ITERATIONS = (100, 1200)                         # 降低最大迭代数，让早停更早生效
CB_LEARNING_RATE = (0.02, 0.06)                    # 进一步降低学习率（原0.03-0.08）
CB_L2_LEAF_REG = (300, 500)                         # 加强 L2 正则化（原250-400）
CB_RANDOM_STRENGTH = (55, 70)                      # 增加随机噪声（原50-60）
CB_RSM = (0.3, 0.45)                               # 降低特征采样比例（原0.4-0.5）
CB_SUBSAMPLE = (0.35, 0.5)                         # 降低样本采样比例下限（原0.4-0.5）

# Gap Penalty（加大惩罚力度）
USE_GAP_PENALTY = True
GAP_SAFE_THR = 0.18                                # 进一步降低安全阈值（原0.20）
GAP_PENALTY_WEIGHT = 0.15                          # 提高惩罚权重（原0.12）

# Early Stopping（更严格）
CB_USE_EARLY_STOPPING = True
CB_EARLY_STOPPING_ROUNDS = 250                     # 增加耐心（原200）
CB_FINAL_ES_VALID_SIZE = 0.30                      # 验证集比例提高到30%（原25%）

# 连续值分层切分：若失败自动回退随机切分
USE_REGRESSION_STRATIFY = True
REG_STRATIFY_MAX_BINS = 8

print(f"[{MODEL_TAG.upper()}] 输出目录: {OUTPUT_DIR_BASE}")

# ====================== 1. 读取数据（仅不删异常值） ======================

df_raw = pd.read_excel(INPUT_XLSX)
if TARGET_COL not in df_raw.columns:
    raise ValueError(f"缺少目标列 {TARGET_COL}")

ROW_ID_COL = "_row_id"
if ROW_ID_COL not in df_raw.columns:
    df_raw[ROW_ID_COL] = np.arange(len(df_raw), dtype=int)

df_base = df_raw.reset_index(drop=True).copy()
base_table_path = os.path.join(OUTPUT_DIR_BASE, f"{MODEL_TAG}_v5_{TARGET_COL}.csv")
df_base.to_csv(base_table_path, index=False, encoding="utf-8-sig")
del df_raw

# ====================== 2. 工具函数 ======================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return 0.0
    if np.allclose(y_true, y_true[0]):
        return 0.0
    return float(r2_score(y_true, y_pred))


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(safe_r2(y_true, y_pred)),
    }


def _float_to_str_full(x):
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    return str(x)


def save_best_params_precise(best_params: dict, out_dir: str, prefix: str):
    ensure_dir(out_dir)

    best_params_num = {}
    best_params_str = {}

    for k, v in best_params.items():
        if isinstance(v, (float, np.floating)):
            best_params_num[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            best_params_num[k] = int(v)
        else:
            best_params_num[k] = v
        best_params_str[k] = _float_to_str_full(v)

    path_json = os.path.join(out_dir, f"{prefix}_params.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(
            {"numeric": best_params_num, "string_repr": best_params_str},
            f, ensure_ascii=False, indent=2
        )

    path_txt = os.path.join(out_dir, f"{prefix}_params.txt")
    with open(path_txt, "w", encoding="utf-8") as f:
        for k in sorted(best_params.keys()):
            f.write(f"{k}: {best_params_str[k]}\n")


def make_regression_stratify_labels(y, max_bins=8):
    if not USE_REGRESSION_STRATIFY:
        return None

    y = np.asarray(y, dtype=float)
    if len(y) < 20:
        return None

    n_unique = len(np.unique(y))
    if n_unique < 6:
        return None

    max_bins = int(min(max_bins, n_unique, max(2, len(y) // 20)))
    for n_bins in range(max_bins, 1, -1):
        try:
            bins = pd.qcut(y, q=n_bins, duplicates="drop")
            vc = pd.Series(bins).value_counts()
            if len(vc) >= 2 and vc.min() >= 2:
                return bins.astype(str)
        except Exception:
            continue
    return None


def robust_train_test_split_indices(n_samples, y, test_size, random_state):
    indices = np.arange(n_samples)
    stratify_labels = make_regression_stratify_labels(y, max_bins=REG_STRATIFY_MAX_BINS)

    if stratify_labels is not None:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_labels
        )
        split_mode = "regression_stratified"
    else:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        split_mode = "random"
    return train_idx, test_idx, split_mode


def build_xy_dropisna(df_in: pd.DataFrame):
    y_log = df_in[TARGET_COL].astype(float).values

    drop_cols = [TARGET_COL, ROW_ID_COL] + [c for c in EXCLUDE_ISNA_COLS if c in df_in.columns]
    X = df_in.drop(columns=[c for c in drop_cols if c in df_in.columns]).copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    print(f"[dropISNA] removed isna cols: {[c for c in EXCLUDE_ISNA_COLS if c in df_in.columns]}")
    print(f"[dropISNA] removed row id col: {ROW_ID_COL}")
    print(f"[dropISNA] X={X.shape}, y={y_log.shape}")

    return X, y_log


def plot_scatter(y_true, y_pred, title, out_path,
                 xlab="True log1p(viscosity)", ylab="Pred log1p(viscosity)"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_prediction_artifacts(
    out_dir: str,
    stem: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    row_ids: np.ndarray,
    df_ref_table: pd.DataFrame,
    split_tag: str,
    scatter_title: str,
):
    """
    保存：
    1) 简表 csv
    2) 带原始字段 merged 表
    3) 散点图
    """
    base_df = pd.DataFrame({
        ROW_ID_COL: row_ids,
        "y_true_log": y_true,
        "y_pred_log": y_pred,
        "residual_log": y_true - y_pred,
        "abs_residual_log": np.abs(y_true - y_pred),
    })

    csv_path = os.path.join(out_dir, f"{stem}.csv")
    base_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    merge_ref = df_ref_table.drop(columns=["__split"], errors="ignore").copy()
    table_df = base_df.copy()
    table_df["__split"] = split_tag
    table_df = table_df.merge(merge_ref, on=ROW_ID_COL, how="left")

    table_path = os.path.join(out_dir, f"{stem}_table.csv")
    table_df.to_csv(table_path, index=False, encoding="utf-8-sig")

    scatter_path = os.path.join(out_dir, f"{stem}.png")
    plot_scatter(y_true, y_pred, scatter_title, scatter_path)

    return {
        f"{stem}_csv": os.path.abspath(csv_path),
        f"{stem}_table": os.path.abspath(table_path),
        f"{stem}_scatter": os.path.abspath(scatter_path),
    }


# ====================== 3. CatBoost 参数与模型 ======================

def normalize_cb_params(params: dict) -> dict:
    p = dict(params)

    p["iterations"] = int(p.get("iterations", CB_ITERATIONS[0]))
    p["learning_rate"] = float(p.get("learning_rate", CB_LEARNING_RATE[0]))

    p["depth"] = int(p.get("depth", 8))
    p["l2_leaf_reg"] = float(p.get("l2_leaf_reg", CB_L2_LEAF_REG[0]))
    p["random_strength"] = float(p.get("random_strength", CB_RANDOM_STRENGTH[0]))
    p["rsm"] = float(p.get("rsm", CB_RSM[0]))
    p["subsample"] = float(p.get("subsample", CB_SUBSAMPLE[0]))

    p["border_count"] = int(p.get("border_count", 64))
    p["grow_policy"] = str(p.get("grow_policy", CB_GROW_POLICY_CHOICES[0]))
    p["leaf_estimation_iterations"] = int(p.get("leaf_estimation_iterations", 5))
    p["bootstrap_type"] = str(p.get("bootstrap_type", CB_BOOTSTRAP_TYPE_CHOICES[0]))
    p["min_data_in_leaf"] = int(p.get("min_data_in_leaf", 45))
    p["boosting_type"] = str(p.get("boosting_type", CB_BOOSTING_TYPE))

    return p


def create_params_from_trial(trial):
    return {
        "iterations": trial.suggest_int(
            "iterations",
            CB_ITERATIONS[0],
            CB_ITERATIONS[1]
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            CB_LEARNING_RATE[0],
            CB_LEARNING_RATE[1],
            log=True
        ),

        "depth": trial.suggest_categorical(
            "depth",
            CB_DEPTH_CHOICES
        ),

        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            CB_L2_LEAF_REG[0],
            CB_L2_LEAF_REG[1],
            log=True
        ),

        "random_strength": trial.suggest_float(
            "random_strength",
            CB_RANDOM_STRENGTH[0],
            CB_RANDOM_STRENGTH[1]
        ),

        "rsm": trial.suggest_float(
            "rsm",
            CB_RSM[0],
            CB_RSM[1]
        ),

        "subsample": trial.suggest_float(
            "subsample",
            CB_SUBSAMPLE[0],
            CB_SUBSAMPLE[1]
        ),

        "border_count": trial.suggest_categorical(
            "border_count",
            CB_BORDER_COUNT_CHOICES
        ),

        "grow_policy": trial.suggest_categorical(
            "grow_policy",
            CB_GROW_POLICY_CHOICES
        ),

        "leaf_estimation_iterations": trial.suggest_categorical(
            "leaf_estimation_iterations",
            CB_LEAF_ESTIMATION_ITERATIONS_CHOICES
        ),

        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type",
            CB_BOOTSTRAP_TYPE_CHOICES
        ),

        "min_data_in_leaf": trial.suggest_categorical(
            "min_data_in_leaf",
            CB_MIN_DATA_IN_LEAF_CHOICES
        ),
    }


def build_model(params: dict, iterations_override=None, random_seed_override=None):
    p = normalize_cb_params(params)

    model_kwargs = dict(
        iterations=int(iterations_override if iterations_override is not None else p["iterations"]),
        learning_rate=p["learning_rate"],
        depth=p["depth"],
        l2_leaf_reg=p["l2_leaf_reg"],
        random_strength=p["random_strength"],
        rsm=p["rsm"],
        border_count=p["border_count"],
        grow_policy=p["grow_policy"],
        leaf_estimation_iterations=p["leaf_estimation_iterations"],
        bootstrap_type=p["bootstrap_type"],
        min_data_in_leaf=p["min_data_in_leaf"],
        subsample=p["subsample"],
        loss_function="RMSE",
        eval_metric="RMSE",
        boosting_type=p["boosting_type"],
        random_seed=int(RANDOM_SEED if random_seed_override is None else random_seed_override),
        thread_count=N_JOBS,
        verbose=False,
        allow_writing_files=False,
        task_type="CPU",
    )

    return CatBoostRegressor(**model_kwargs)


def fit_catboost_with_eval(params, X_fit, y_fit, X_eval, y_eval, tag="", random_seed_override=None):
    p = normalize_cb_params(params)
    requested_iters = int(p["iterations"])

    model = build_model(
        p,
        iterations_override=requested_iters,
        random_seed_override=random_seed_override
    )

    if CB_USE_EARLY_STOPPING:
        model.fit(
            X_fit, y_fit,
            eval_set=(X_eval, y_eval),
            use_best_model=True,
            early_stopping_rounds=CB_EARLY_STOPPING_ROUNDS,
            verbose=False
        )
    else:
        model.fit(X_fit, y_fit, verbose=False)

    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter < 0:
        effective_iters = requested_iters
    else:
        effective_iters = int(best_iter) + 1

    best_score = model.get_best_score()
    best_rmse = np.nan
    try:
        best_rmse = float(best_score["validation"]["RMSE"])
    except Exception:
        pass

    es_info = {
        "requested_iterations": requested_iters,
        "best_iterations_es": int(effective_iters),
        "best_es_rmse": float(best_rmse) if not np.isnan(best_rmse) else np.nan,
        "es_stopped_early": bool(effective_iters < requested_iters),
    }

    print(
        f"[CB-ES {tag}] req={requested_iters}, best_iter={effective_iters}, best_rmse={best_rmse}"
    )
    return model, es_info


def fit_catboost_final_refit(params, X_tr, y_tr, tag=""):
    p = normalize_cb_params(params)
    requested_iters = int(p["iterations"])

    if (not CB_USE_EARLY_STOPPING) or (len(y_tr) < 10):
        final_model = build_model(p, iterations_override=requested_iters)
        final_model.fit(X_tr, y_tr, verbose=False)
        es_info = {
            "requested_iterations": requested_iters,
            "best_iterations_es": requested_iters,
            "best_es_rmse": np.nan,
            "es_stopped_early": False,
            "final_es_split_mode": "no_early_stopping",
        }
        return final_model, es_info

    es_stratify = make_regression_stratify_labels(y_tr, max_bins=REG_STRATIFY_MAX_BINS)
    if es_stratify is not None:
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_tr, y_tr,
            test_size=CB_FINAL_ES_VALID_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=es_stratify
        )
        final_es_split_mode = "regression_stratified"
    else:
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_tr, y_tr,
            test_size=CB_FINAL_ES_VALID_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        final_es_split_mode = "random"

    probe_model = build_model(p, iterations_override=requested_iters)
    probe_model.fit(
        X_fit, y_fit,
        eval_set=(X_es, y_es),
        use_best_model=True,
        early_stopping_rounds=CB_EARLY_STOPPING_ROUNDS,
        verbose=False
    )

    best_iter = probe_model.get_best_iteration()
    if best_iter is None or best_iter < 0:
        effective_iters = requested_iters
    else:
        effective_iters = int(best_iter) + 1

    best_score = probe_model.get_best_score()
    best_rmse = np.nan
    try:
        best_rmse = float(best_score["validation"]["RMSE"])
    except Exception:
        pass

    final_model = build_model(p, iterations_override=effective_iters)
    final_model.fit(X_tr, y_tr, verbose=False)

    es_info = {
        "requested_iterations": requested_iters,
        "best_iterations_es": int(effective_iters),
        "best_es_rmse": float(best_rmse) if not np.isnan(best_rmse) else np.nan,
        "es_stopped_early": bool(effective_iters < requested_iters),
        "final_es_split_mode": final_es_split_mode,
    }

    print(
        f"[CB-FINAL-ES {tag}] req={requested_iters}, best_iter={effective_iters}, best_rmse={best_rmse}, split={final_es_split_mode}"
    )
    return final_model, es_info


def evaluate_params_kfold_log(params, X, y_log, n_splits, tag=""):
    p = normalize_cb_params(params)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    n = len(y_log)
    oof = np.zeros(n, dtype=float)

    tr_metrics = []
    val_metrics = []
    es_best_iters = []
    es_best_rmses = []
    es_stop_flags = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_log[tr_idx], y_log[val_idx]

        model, es_info = fit_catboost_with_eval(
            params=p,
            X_fit=X_tr, y_fit=y_tr,
            X_eval=X_val, y_eval=y_val,
            tag=f"{tag}_F{fold}",
            random_seed_override=RANDOM_SEED
        )

        y_val_pred = model.predict(X_val)
        oof[val_idx] = y_val_pred

        val_rmse = float(mean_squared_error(y_val, y_val_pred) ** 0.5)
        val_r2 = float(safe_r2(y_val, y_val_pred))
        val_mae = float(mean_absolute_error(y_val, y_val_pred))

        y_tr_pred = model.predict(X_tr)
        tr_rmse = float(mean_squared_error(y_tr, y_tr_pred) ** 0.5)
        tr_r2 = float(safe_r2(y_tr, y_tr_pred))
        tr_mae = float(mean_absolute_error(y_tr, y_tr_pred))

        tr_metrics.append((tr_mae, tr_rmse, tr_r2))
        val_metrics.append((val_mae, val_rmse, val_r2))

        es_best_iters.append(es_info["best_iterations_es"])
        es_best_rmses.append(es_info["best_es_rmse"])
        es_stop_flags.append(int(es_info["es_stopped_early"]))

    tr_metrics = np.array(tr_metrics)
    val_metrics = np.array(val_metrics)

    oof_rmse = float(mean_squared_error(y_log, oof) ** 0.5)
    oof_r2 = float(safe_r2(y_log, oof))
    oof_mae = float(mean_absolute_error(y_log, oof))

    gap_rmse_cv = float(val_metrics[:, 1].mean() - tr_metrics[:, 1].mean())

    metrics = {
        "train_mae_log_cv": float(tr_metrics[:, 0].mean()),
        "train_rmse_log_cv": float(tr_metrics[:, 1].mean()),
        "train_r2_log_cv": float(tr_metrics[:, 2].mean()),

        "valid_mae_log_cv": float(val_metrics[:, 0].mean()),
        "valid_rmse_log_cv": float(val_metrics[:, 1].mean()),
        "valid_r2_log_cv": float(val_metrics[:, 2].mean()),

        "valid_mae_log": float(oof_mae),
        "valid_rmse_log": float(oof_rmse),
        "valid_r2_log": float(oof_r2),

        "gap_rmse_cv": gap_rmse_cv,

        "es_best_iterations_mean": float(np.mean(es_best_iters)),
        "es_best_iterations_std": float(np.std(es_best_iters)),
        "es_best_rmse_mean": float(np.nanmean(es_best_rmses)),
        "es_stopped_ratio": float(np.mean(es_stop_flags)),
    }

    return float(oof_rmse), metrics, oof, es_best_iters


# ====================== 4. 评估与保存（仅 objective_best） ======================

def evaluate_and_save_objective(
    params: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    df_train_table: pd.DataFrame,
    df_test_table: pd.DataFrame,
    out_dir: str,
    n_splits: int,
):
    candidate_name = "objective_best"
    candidate_dir = os.path.join(out_dir, f"cand_{candidate_name}")
    ensure_dir(candidate_dir)

    save_best_params_precise(
        params,
        out_dir=candidate_dir,
        prefix=f"{MODEL_TAG}_{candidate_name}_K{n_splits}"
    )

    # 1) 正式 OOF
    _, best_metrics_train, oof_pred_train, _ = evaluate_params_kfold_log(
        params=params,
        X=X_train,
        y_log=y_train,
        n_splits=n_splits,
        tag=f"{candidate_name}_OOF_K{n_splits}"
    )

    train_row_ids = df_train_table[ROW_ID_COL].values

    oof_artifacts = save_prediction_artifacts(
        out_dir=candidate_dir,
        stem=f"oof_{candidate_name}",
        y_true=y_train,
        y_pred=oof_pred_train,
        row_ids=train_row_ids,
        df_ref_table=df_train_table,
        split_tag="train_oof",
        scatter_title=f"[{candidate_name}] OOF RMSE={best_metrics_train['valid_rmse_log']:.4f}, R2={best_metrics_train['valid_r2_log']:.4f}",
    )

    # 2) train 全量 refit
    model_final, final_es_info = fit_catboost_final_refit(
        params=params,
        X_tr=X_train,
        y_tr=y_train,
        tag=f"{candidate_name}_FINAL_K{n_splits}"
    )

    model_path = os.path.join(candidate_dir, "model.joblib")
    joblib.dump(model_final, model_path)

    native_model_path = os.path.join(candidate_dir, "model.cbm")
    model_final.save_model(native_model_path)

    # 3) train_full / test
    y_train_full_pred = model_final.predict(X_train)
    train_full = compute_metrics(y_train, y_train_full_pred)

    y_test_pred = model_final.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)

    train_full_artifacts = save_prediction_artifacts(
        out_dir=candidate_dir,
        stem=f"train_full_pred_{candidate_name}",
        y_true=y_train,
        y_pred=y_train_full_pred,
        row_ids=train_row_ids,
        df_ref_table=df_train_table,
        split_tag="train",
        scatter_title=f"[{candidate_name}] TRAIN_FULL RMSE={train_full['rmse']:.4f}, R2={train_full['r2']:.4f}",
    )

    test_row_ids = df_test_table[ROW_ID_COL].values
    test_artifacts = save_prediction_artifacts(
        out_dir=candidate_dir,
        stem=f"test_pred_{candidate_name}",
        y_true=y_test,
        y_pred=y_test_pred,
        row_ids=test_row_ids,
        df_ref_table=df_test_table,
        split_tag="test",
        scatter_title=f"[{candidate_name}] TEST RMSE={test_metrics['rmse']:.4f}, R2={test_metrics['r2']:.4f}",
    )

    result = {
        "candidate_name": candidate_name,
        "params": normalize_cb_params(params),

        "train_train_mae_log_cv": float(best_metrics_train["train_mae_log_cv"]),
        "train_train_rmse_log_cv": float(best_metrics_train["train_rmse_log_cv"]),
        "train_train_r2_log_cv": float(best_metrics_train["train_r2_log_cv"]),

        "train_valid_mae_log_cv": float(best_metrics_train["valid_mae_log_cv"]),
        "train_valid_rmse_log_cv": float(best_metrics_train["valid_rmse_log_cv"]),
        "train_valid_r2_log_cv": float(best_metrics_train["valid_r2_log_cv"]),

        "train_oof_mae_log": float(best_metrics_train["valid_mae_log"]),
        "train_oof_rmse_log": float(best_metrics_train["valid_rmse_log"]),
        "train_oof_r2_log": float(best_metrics_train["valid_r2_log"]),
        "train_gap_rmse_cv": float(best_metrics_train["gap_rmse_cv"]),

        "train_full_mae_log": float(train_full["mae"]),
        "train_full_rmse_log": float(train_full["rmse"]),
        "train_full_r2_log": float(train_full["r2"]),

        "test_mae_log": float(test_metrics["mae"]),
        "test_rmse_log": float(test_metrics["rmse"]),
        "test_r2_log": float(test_metrics["r2"]),

        "cv_es_best_iterations_mean": float(best_metrics_train["es_best_iterations_mean"]),
        "cv_es_best_iterations_std": float(best_metrics_train["es_best_iterations_std"]),
        "cv_es_best_rmse_mean": float(best_metrics_train["es_best_rmse_mean"]),
        "cv_es_stopped_ratio": float(best_metrics_train["es_stopped_ratio"]),

        "final_requested_iterations": int(final_es_info["requested_iterations"]),
        "final_effective_iterations": int(final_es_info["best_iterations_es"]),
        "final_es_rmse": float(final_es_info["best_es_rmse"]) if not np.isnan(final_es_info["best_es_rmse"]) else np.nan,
        "final_es_stopped_early": bool(final_es_info["es_stopped_early"]),
        "final_es_split_mode": final_es_info["final_es_split_mode"],

        "model_dir": os.path.abspath(candidate_dir),
        "final_model_path": os.path.abspath(model_path),
        "final_model_native_path": os.path.abspath(native_model_path),
    }

    result.update(oof_artifacts)
    result.update(train_full_artifacts)
    result.update(test_artifacts)

    pd.DataFrame([result]).to_csv(
        os.path.join(candidate_dir, f"summary_metrics_{candidate_name}.csv"),
        index=False, encoding="utf-8-sig"
    )

    return result


# ====================== 5. 主流程 ======================

def run_final_catboost(n_splits: int, df_base_in: pd.DataFrame):
    print("\n" + "=" * 100)
    print(f"开始场景：dropISNA + split(9:1) + K={n_splits} —— NO_OUTLIER_DELETE —— TARGET={TARGET_COL}")
    print("=" * 100)

    out_dir = os.path.join(OUTPUT_DIR_BASE, f"K{n_splits}_NO_OUTLIER")
    ensure_dir(out_dir)

    X_all, y_all = build_xy_dropisna(df_base_in)
    df_all_table = df_base_in.copy().reset_index(drop=True)

    train_idx, test_idx, split_mode = robust_train_test_split_indices(
        n_samples=len(y_all),
        y=y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    np.save(os.path.join(out_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(out_dir, "test_idx.npy"), test_idx)

    X_train = X_all.iloc[train_idx].reset_index(drop=True)
    X_test = X_all.iloc[test_idx].reset_index(drop=True)
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    df_train_table = df_all_table.iloc[train_idx].reset_index(drop=True).copy()
    df_test_table = df_all_table.iloc[test_idx].reset_index(drop=True).copy()
    df_train_table["__split"] = "train"
    df_test_table["__split"] = "test"

    df_train_table.to_csv(os.path.join(out_dir, "train_table.csv"), index=False, encoding="utf-8-sig")
    df_test_table.to_csv(os.path.join(out_dir, "test_table.csv"), index=False, encoding="utf-8-sig")

    def objective(trial):
        params = normalize_cb_params(create_params_from_trial(trial))

        oof_rmse, metrics, _, es_best_iters = evaluate_params_kfold_log(
            params=params,
            X=X_train,
            y_log=y_train,
            n_splits=n_splits,
            tag=f"CV_K{n_splits}"
        )

        gap_rmse_cv = float(metrics["gap_rmse_cv"])

        if USE_GAP_PENALTY:
            gap_penalty = GAP_PENALTY_WEIGHT * max(0.0, gap_rmse_cv - GAP_SAFE_THR)
            objective_score = float(oof_rmse + gap_penalty)
        else:
            gap_penalty = 0.0
            objective_score = float(oof_rmse)

        trial.set_user_attr("requested_iterations", int(params["iterations"]))
        trial.set_user_attr("raw_oof_rmse_log", float(oof_rmse))
        trial.set_user_attr("gap_rmse_cv", float(gap_rmse_cv))
        trial.set_user_attr("gap_penalty", float(gap_penalty))
        trial.set_user_attr("objective_score", float(objective_score))
        trial.set_user_attr("gap_safe_thr", float(GAP_SAFE_THR))
        trial.set_user_attr("gap_penalty_weight", float(GAP_PENALTY_WEIGHT))
        trial.set_user_attr("es_best_iterations_mean", float(np.mean(es_best_iters)))
        trial.set_user_attr("es_best_iterations_list", json.dumps([int(x) for x in es_best_iters], ensure_ascii=False))

        for k, v in metrics.items():
            trial.set_user_attr("train_" + k, float(v))

        return objective_score

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_SEED,
        n_startup_trials=50,
        multivariate=True
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{MODEL_TAG}_{TARGET_COL}_K{n_splits}_V2_controlled_escape",
        sampler=sampler
    )

    study.optimize(objective, n_trials=N_TRIALS_SPLIT)

    study_best_trial = study.best_trial
    objective_best_trial_number = int(study_best_trial.number)
    objective_best_params = normalize_cb_params(study_best_trial.params)

    # 保存 trials
    rows = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "objective_score": t.value,
        }
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)

    df_trials = pd.DataFrame(rows)
    df_trials["is_objective_best"] = df_trials["trial"].eq(objective_best_trial_number)

    # 方便排序查看
    if "raw_oof_rmse_log" in df_trials.columns:
        df_trials = df_trials.sort_values(
            by=["objective_score", "raw_oof_rmse_log"],
            ascending=[True, True]
        ).reset_index(drop=True)

    trials_csv = os.path.join(out_dir, "trials.csv")
    df_trials.to_csv(trials_csv, index=False, encoding="utf-8-sig")

    save_best_params_precise(objective_best_params, out_dir, "objective_best")

    # 只评估 objective_best
    objective_result = evaluate_and_save_objective(
        params=objective_best_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        df_train_table=df_train_table,
        df_test_table=df_test_table,
        out_dir=out_dir,
        n_splits=n_splits,
    )

    best_raw_oof_rmse = study_best_trial.user_attrs.get("raw_oof_rmse_log", np.nan)
    best_gap_penalty = study_best_trial.user_attrs.get("gap_penalty", np.nan)
    best_gap_rmse_cv = study_best_trial.user_attrs.get("gap_rmse_cv", np.nan)

    # 把所有指标都写进 summary
    summary = {
        "target": TARGET_COL,
        "mode": "NO_OUTLIER_DELETE",
        "dropISNA": True,
        "use_outlier_delete": False,
        "n_base_samples": int(len(df_all_table)),
        "n_splits": int(n_splits),
        "test_size": float(TEST_SIZE),
        "split_mode": split_mode,

        "objective_best_trial": int(objective_best_trial_number),
        "objective_best_score_from_study": float(study_best_trial.value),
        "objective_best_raw_oof_rmse_log_from_study": float(best_raw_oof_rmse) if not pd.isna(best_raw_oof_rmse) else np.nan,
        "objective_best_gap_rmse_cv_from_study": float(best_gap_rmse_cv) if not pd.isna(best_gap_rmse_cv) else np.nan,
        "objective_best_gap_penalty_from_study": float(best_gap_penalty) if not pd.isna(best_gap_penalty) else np.nan,
        "use_gap_penalty": bool(USE_GAP_PENALTY),
        "gap_safe_thr": float(GAP_SAFE_THR),
        "gap_penalty_weight": float(GAP_PENALTY_WEIGHT),

        # ===== train_train_cv =====
        "train_train_mae_log_cv": float(objective_result["train_train_mae_log_cv"]),
        "train_train_rmse_log_cv": float(objective_result["train_train_rmse_log_cv"]),
        "train_train_r2_log_cv": float(objective_result["train_train_r2_log_cv"]),

        # ===== train_valid_cv =====
        "train_valid_mae_log_cv": float(objective_result["train_valid_mae_log_cv"]),
        "train_valid_rmse_log_cv": float(objective_result["train_valid_rmse_log_cv"]),
        "train_valid_r2_log_cv": float(objective_result["train_valid_r2_log_cv"]),

        # ===== train_oof =====
        "train_oof_mae_log": float(objective_result["train_oof_mae_log"]),
        "train_oof_rmse_log": float(objective_result["train_oof_rmse_log"]),
        "train_oof_r2_log": float(objective_result["train_oof_r2_log"]),
        "train_gap_rmse_cv": float(objective_result["train_gap_rmse_cv"]),

        # ===== train_full =====
        "train_full_mae_log": float(objective_result["train_full_mae_log"]),
        "train_full_rmse_log": float(objective_result["train_full_rmse_log"]),
        "train_full_r2_log": float(objective_result["train_full_r2_log"]),

        # ===== test =====
        "test_mae_log": float(objective_result["test_mae_log"]),
        "test_rmse_log": float(objective_result["test_rmse_log"]),
        "test_r2_log": float(objective_result["test_r2_log"]),

        # ===== CV early stopping summary =====
        "cv_es_best_iterations_mean": float(objective_result["cv_es_best_iterations_mean"]),
        "cv_es_best_iterations_std": float(objective_result["cv_es_best_iterations_std"]),
        "cv_es_best_rmse_mean": float(objective_result["cv_es_best_rmse_mean"]),
        "cv_es_stopped_ratio": float(objective_result["cv_es_stopped_ratio"]),

        # ===== final refit early stopping summary =====
        "final_requested_iterations": int(objective_result["final_requested_iterations"]),
        "final_effective_iterations": int(objective_result["final_effective_iterations"]),
        "final_es_rmse": float(objective_result["final_es_rmse"]) if not np.isnan(objective_result["final_es_rmse"]) else np.nan,
        "final_es_stopped_early": bool(objective_result["final_es_stopped_early"]),
        "final_es_split_mode": objective_result["final_es_split_mode"],

        # ===== 路径 =====
        "base_table_path": os.path.abspath(base_table_path),
        "output_dir": os.path.abspath(out_dir),
        "trials_csv": os.path.abspath(trials_csv),

        "model_dir": objective_result["model_dir"],
        "final_model_path": objective_result["final_model_path"],
        "final_model_native_path": objective_result["final_model_native_path"],

        "oof_csv": objective_result["oof_objective_best_csv"],
        "oof_table_csv": objective_result["oof_objective_best_table"],
        "oof_scatter_png": objective_result["oof_objective_best_scatter"],

        "train_full_csv": objective_result["train_full_pred_objective_best_csv"],
        "train_full_table_csv": objective_result["train_full_pred_objective_best_table"],
        "train_full_scatter_png": objective_result["train_full_pred_objective_best_scatter"],

        "test_csv": objective_result["test_pred_objective_best_csv"],
        "test_table_csv": objective_result["test_pred_objective_best_table"],
        "test_scatter_png": objective_result["test_pred_objective_best_scatter"],
    }

    summary_path = os.path.join(out_dir, "summary_metrics.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[SUMMARY] saved: {summary_path}")

    return pd.DataFrame([summary])


# ====================== 6. 主程序 ======================

if __name__ == "__main__":
    all_summary = []

    for k in N_SPLITS_LIST:
        one_summary = run_final_catboost(
            n_splits=k,
            df_base_in=df_base,
        )
        all_summary.append(one_summary)

    if len(all_summary) > 0:
        df_all_summary = pd.concat(all_summary, axis=0, ignore_index=True)
        all_summary_path = os.path.join(OUTPUT_DIR_BASE, "summary_metrics_ALL.csv")
        df_all_summary.to_csv(all_summary_path, index=False, encoding="utf-8-sig")
        print(f"[SUMMARY ALL] 已保存: {all_summary_path}")

    print(
        f"\n=== 全部 {MODEL_TAG.upper()} 场景完成："
        f"{TARGET_COL} | NO_OUTLIER_DELETE | objective_best only | V2 controlled escape ==="
    )