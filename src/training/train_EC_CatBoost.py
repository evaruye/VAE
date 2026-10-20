# -*- coding: utf-8 -*-
r"""
CatBoost + Optuna（原始标度 y，全链路不做 log 变换）——仅保留【NO_OUTLIER_DELETE】+ CatBoost 原生早停版

当前版本流程：
1) 不删除异常值（NO_OUTLIER_DELETE）
2) 固定外层 train/test 划分（9:1）
3) 仅在 train 上做 10-fold CV + Optuna
4) 以 train 内部 OOF RMSE 作为最优参数选择标准
5) 用最优参数在 train 全量 refit
6) 在 test 上评估并导出结果

重要说明：
- CatBoost 原生支持 early stopping。
- 本脚本使用的是“正确的 CatBoost 早停方案”：
  在每个 fold 内直接使用验证集 eval_set + early_stopping_rounds + use_best_model

保留：
- 保存最终 train/test 索引（确保后续解释用同一批样本）
- 保存 final 模型，后续直接 load，不重新 fit
- 输出 train_full_pred 的数据表格（含 row_id、一一对应 y_true/y_pred/residual）
- 输出不删异常值全量表
- 输出可一一对应的训练集/测试集表格（含 row_id、split 标记，方便查找）

训练指标口径：
- train_train_*_cv：K 折内每折训练子集指标取平均（更“乐观”）
- train_valid_*_cv：K 折内每折验证子集指标取平均
- train_oof_*：OOF（out-of-fold，全训练集每个样本由未见过它的折模型预测一次）指标
- train_full_*：用最优超参数在 Train 全量 refit 后，在 Train 全量上计算的拟合指标
"""

import os
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, CatBoostError
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")

# ====================== 0. 基本配置 ======================

MODEL_TAG = "catboost"

# ---------- 固含量数据集 ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_XLSX = str(REPO_ROOT / "data" / "Data_S2" / "EC_Modeling_Data.xlsx")
TARGET_COL = "Ethylene Content"

# dropISNA：不作为输入特征的缺失指示列（必须删掉）
EXCLUDE_ISNA_COLS = [
    "oxidizer_pct2_isna",
    "init_pressure_isna",
    "time_isna",
    "pH_buffer_pct_isna",
]

# CV / 随机种子 / trials
N_SPLITS_LIST = [10]
RANDOM_SEED = 23
N_TRIALS_SPLIT = 500

# 9:1 split
TEST_SIZE = 0.1

# 线程数
N_JOBS = 1

# ======================
# CatBoost 搜索空间（放大泛化波动版；仅调参，不改目标函数）
# ======================
CB_ITERATIONS = (100, 600)
CB_LEARNING_RATE = (1e-2, 3e-1)              # log
CB_DEPTH = (3, 5)
CB_L2_LEAF_REG = (20, 300.0)               # log
CB_RANDOM_STRENGTH = (5, 50.0)            # log
CB_RSM = (0.1, 1.0)
CB_BORDER_COUNT_CHOICES = [16, 32, 64]
CB_GROW_POLICY_CHOICES = ["SymmetricTree", "Depthwise"]
CB_MIN_DATA_IN_LEAF = (16, 64)                # Depthwise / Lossguide 时生效
CB_LEAF_ESTIMATION_ITERATIONS = (1, 10)
CB_BOOTSTRAP_TYPE_CHOICES = ["Bayesian", "Bernoulli"]
CB_BAGGING_TEMPERATURE = (0.0, 2.0)         # 仅 Bayesian
CB_SUBSAMPLE = (0.2, 0.8)                    # 仅 Bernoulli / MVS

# ======================
# CatBoost 原生早停配置
# ======================
CB_USE_EARLY_STOPPING = True
CB_EARLY_STOPPING_ROUNDS = 200
CB_FINAL_ES_VALID_SIZE = 0.15

# ====================== 0.1 路径稳健工具（长路径 + 自动短根目录） ======================

IS_WIN = (os.name == "nt")
SHORT_OUTPUT_ROOT = str(REPO_ROOT / "results" / "training" / "EC")
WIN_PATH_SOFT_LIMIT = 240


def safe_path_str(p: str) -> str:
    if p is None:
        return p
    if not isinstance(p, str):
        p = str(p)
    p2 = p.strip()
    while len(p2) > 0 and p2[-1] in ["'", '"']:
        p2 = p2[:-1].rstrip()
    return p2


def win_long_path(path: str) -> str:
    p = safe_path_str(path)
    p = os.path.normpath(p)

    if not IS_WIN:
        return p

    if p.startswith("\\\\?\\"):
        return p

    p_abs = os.path.abspath(p)
    p_abs = os.path.normpath(p_abs)

    if p_abs.startswith("\\\\"):
        p_unc = "\\\\?\\UNC\\" + p_abs.lstrip("\\")
        return p_unc

    return "\\\\?\\" + p_abs


def ensure_parent_dir(filepath: str) -> str:
    fp = safe_path_str(filepath)
    parent = os.path.dirname(fp)
    if parent:
        parent_norm = os.path.normpath(parent)
        if not os.path.exists(parent_norm):
            os.makedirs(parent_norm, exist_ok=True)

    return win_long_path(fp)


def get_out_dir(preferred_out_dir: str) -> str:
    p = safe_path_str(preferred_out_dir)
    p_abs = os.path.abspath(os.path.normpath(p))

    if (not IS_WIN) or (len(p_abs) < WIN_PATH_SOFT_LIMIT):
        os.makedirs(p_abs, exist_ok=True)
        return p_abs

    os.makedirs(SHORT_OUTPUT_ROOT, exist_ok=True)

    base_name = os.path.basename(p_abs)
    parent_name = os.path.basename(os.path.dirname(p_abs))
    new_out = os.path.join(SHORT_OUTPUT_ROOT, parent_name, base_name)

    new_out_abs = os.path.abspath(os.path.normpath(new_out))
    os.makedirs(new_out_abs, exist_ok=True)

    print(f"[PATH WARN] 输出路径过长，已自动切换到更短目录：")
    print(f"  原路径: {p_abs}")
    print(f"  新路径: {new_out_abs}")

    return new_out_abs


# ====================== 0.2 输出目录（可能自动迁移） ======================

DATA_DIR = os.path.dirname(INPUT_XLSX) or "."
OUTPUT_DIR_BASE_PREFERRED = SHORT_OUTPUT_ROOT
OUTPUT_DIR_BASE = get_out_dir(OUTPUT_DIR_BASE_PREFERRED)
print(f"[{MODEL_TAG.upper()}] 总输出目录: {OUTPUT_DIR_BASE}")

# ====================== 1. 读取数据（仅 NO_OUTLIER_DELETE） ======================

df_raw = pd.read_excel(INPUT_XLSX)
if TARGET_COL not in df_raw.columns:
    raise ValueError(f"找不到目标列 {TARGET_COL} ，请检查列名是否正确。")

ROW_ID_COL = "_row_id"
if ROW_ID_COL not in df_raw.columns:
    df_raw[ROW_ID_COL] = np.arange(len(df_raw), dtype=int)

df_no_outlier_delete = df_raw.reset_index(drop=True).copy()

# 输出不删异常值全量表
base_tables_dir = get_out_dir(os.path.join(OUTPUT_DIR_BASE, "base_tables"))
no_outlier_delete_path = ensure_parent_dir(
    os.path.join(base_tables_dir, f"{MODEL_TAG}_no_outlier_delete_full_table_{TARGET_COL}.csv")
)
df_no_outlier_delete.to_csv(no_outlier_delete_path, index=False, encoding="utf-8-sig")
print(f"[Base Table] 不删异常值全量表已保存: {no_outlier_delete_path}")

del df_raw

# ====================== 2. 工具函数 ======================

def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return 0.0
    if np.allclose(y_true, y_true[0]):
        return 0.0
    return float(r2_score(y_true, y_pred))


def _float_to_str_full(x):
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    return str(x)


def save_best_params_precise(best_params: dict, out_dir: str, prefix: str):
    out_dir = get_out_dir(out_dir)

    best_params_num = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in best_params.items()}
    best_params_str = {k: _float_to_str_full(v) for k, v in best_params.items()}

    path_json = ensure_parent_dir(os.path.join(out_dir, f"{prefix}_best_params_precise.json"))
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({"numeric": best_params_num, "string_repr": best_params_str}, f, ensure_ascii=False, indent=2)

    path_txt = ensure_parent_dir(os.path.join(out_dir, f"{prefix}_best_params_precise.txt"))
    with open(path_txt, "w", encoding="utf-8") as f:
        for k in sorted(best_params.keys()):
            f.write(f"{k}: {best_params_str[k]}\n")

    print(f"[BEST PARAMS] 已保存（高精度）: {path_json}")
    print(f"[BEST PARAMS] 已保存（高精度）: {path_txt}")


def save_trials_with_highlight(df_trials, best_trial_number, csv_path, sheet_name="trials"):
    csv_path = safe_path_str(csv_path)
    xlsx_path = ensure_parent_dir(csv_path.replace(".csv", ".xlsx"))
    try:
        import xlsxwriter
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df_trials.to_excel(writer, index=False, sheet_name=sheet_name)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            red_fmt = workbook.add_format({"font_color": "red"})
            idx_list = df_trials.index[df_trials["trial"] == best_trial_number]
            if len(idx_list) > 0:
                worksheet.set_row(int(idx_list[0]) + 1, None, red_fmt)
        print(f"[Trials] 带高亮的 Excel 已保存到: {xlsx_path}")
    except Exception as e:
        print(f"[Trials] 生成高亮 Excel 失败: {e}")


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = float(safe_r2(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ====================== 3. CatBoost 模型与原生早停 ======================

def infer_boosting_type_from_grow_policy(grow_policy: str) -> str:
    if str(grow_policy) == "SymmetricTree":
        return "Ordered"
    return "Plain"


def normalize_cb_params(params: dict) -> dict:
    p = dict(params)

    p["iterations"] = int(p["iterations"])
    p["learning_rate"] = float(p["learning_rate"])
    p["depth"] = int(p["depth"])
    p["l2_leaf_reg"] = float(p["l2_leaf_reg"])
    p["random_strength"] = float(p["random_strength"])
    p["rsm"] = float(p["rsm"])
    p["border_count"] = int(p["border_count"])
    p["grow_policy"] = str(p["grow_policy"])
    p["leaf_estimation_iterations"] = int(p["leaf_estimation_iterations"])
    p["bootstrap_type"] = str(p["bootstrap_type"])

    if p["grow_policy"] in ["Depthwise", "Lossguide"]:
        p["min_data_in_leaf"] = int(p.get("min_data_in_leaf", 1))
    else:
        p["min_data_in_leaf"] = None

    if p["bootstrap_type"] == "Bayesian":
        p["bagging_temperature"] = float(p.get("bagging_temperature", 0.0))
        p["subsample"] = None
    elif p["bootstrap_type"] in ["Bernoulli", "MVS"]:
        p["subsample"] = float(p.get("subsample", 1.0))
        p["bagging_temperature"] = None
    else:
        p["bagging_temperature"] = None
        p["subsample"] = None

    p["boosting_type"] = infer_boosting_type_from_grow_policy(p["grow_policy"])
    return p


def build_model(params: dict, iterations_override=None):
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
        loss_function="RMSE",
        eval_metric="RMSE",
        boosting_type=p["boosting_type"],
        random_seed=RANDOM_SEED,
        thread_count=N_JOBS,
        verbose=False,
        allow_writing_files=False,
        task_type="CPU",
    )

    if p["min_data_in_leaf"] is not None:
        model_kwargs["min_data_in_leaf"] = p["min_data_in_leaf"]

    if p["bagging_temperature"] is not None:
        model_kwargs["bagging_temperature"] = p["bagging_temperature"]

    if p["subsample"] is not None:
        model_kwargs["subsample"] = p["subsample"]

    reg = CatBoostRegressor(**model_kwargs)
    return reg


def create_params_from_trial(trial: optuna.Trial) -> dict:
    grow_policy = trial.suggest_categorical("grow_policy", CB_GROW_POLICY_CHOICES)
    bootstrap_type = trial.suggest_categorical("bootstrap_type", CB_BOOTSTRAP_TYPE_CHOICES)

    params = {
        "iterations": trial.suggest_int("iterations", CB_ITERATIONS[0], CB_ITERATIONS[1]),
        "learning_rate": trial.suggest_float("learning_rate", CB_LEARNING_RATE[0], CB_LEARNING_RATE[1], log=True),
        "depth": trial.suggest_int("depth", CB_DEPTH[0], CB_DEPTH[1]),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", CB_L2_LEAF_REG[0], CB_L2_LEAF_REG[1], log=True),
        "random_strength": trial.suggest_float(
            "random_strength", CB_RANDOM_STRENGTH[0], CB_RANDOM_STRENGTH[1], log=True
        ),
        "rsm": trial.suggest_float("rsm", CB_RSM[0], CB_RSM[1]),
        "border_count": trial.suggest_categorical("border_count", CB_BORDER_COUNT_CHOICES),
        "grow_policy": grow_policy,
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations",
            CB_LEAF_ESTIMATION_ITERATIONS[0],
            CB_LEAF_ESTIMATION_ITERATIONS[1]
        ),
        "bootstrap_type": bootstrap_type,
    }

    if grow_policy in ["Depthwise", "Lossguide"]:
        params["min_data_in_leaf"] = trial.suggest_int(
            "min_data_in_leaf",
            CB_MIN_DATA_IN_LEAF[0],
            CB_MIN_DATA_IN_LEAF[1]
        )
    else:
        params["min_data_in_leaf"] = None

    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature",
            CB_BAGGING_TEMPERATURE[0],
            CB_BAGGING_TEMPERATURE[1]
        )
        params["subsample"] = None
    elif bootstrap_type in ["Bernoulli", "MVS"]:
        params["subsample"] = trial.suggest_float(
            "subsample",
            CB_SUBSAMPLE[0],
            CB_SUBSAMPLE[1]
        )
        params["bagging_temperature"] = None
    else:
        params["subsample"] = None
        params["bagging_temperature"] = None

    return params


def fit_catboost_cv_fold(params, X_tr, y_tr, X_val, y_val, tag=""):
    p = normalize_cb_params(params)
    requested_iters = int(p["iterations"])

    model = build_model(p)

    if CB_USE_EARLY_STOPPING:
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=CB_EARLY_STOPPING_ROUNDS,
            verbose=False
        )
    else:
        model.fit(X_tr, y_tr, verbose=False)

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
        "boosting_type": p["boosting_type"],
        "grow_policy": p["grow_policy"],
    }

    print(
        f"[CB-ES {tag}] grow_policy={p['grow_policy']}, boosting_type={p['boosting_type']}, "
        f"requested_iters={requested_iters}, best_iters={effective_iters}, best_es_rmse={best_rmse}"
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
            "boosting_type": p["boosting_type"],
            "grow_policy": p["grow_policy"],
        }
        return final_model, es_info

    X_fit, X_es, y_fit, y_es = train_test_split(
        X_tr, y_tr,
        test_size=CB_FINAL_ES_VALID_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

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
        "boosting_type": p["boosting_type"],
        "grow_policy": p["grow_policy"],
    }

    print(
        f"[CB-FINAL-ES {tag}] grow_policy={p['grow_policy']}, boosting_type={p['boosting_type']}, "
        f"requested_iters={requested_iters}, best_iters={effective_iters}, best_es_rmse={best_rmse}"
    )
    return final_model, es_info


def evaluate_params_kfold(params, X, y, n_splits, tag=""):
    n_samples = len(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof_pred = np.zeros(n_samples, dtype=float)

    train_metrics = []
    valid_metrics = []

    es_best_iters = []
    es_best_rmses = []
    es_stop_flags = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]

        model, es_info = fit_catboost_cv_fold(
            params, X_tr, y_tr, X_val, y_val,
            tag=f"{tag}_Fold{fold}"
        )

        y_val_pred = model.predict(X_val)
        oof_pred[valid_idx] = y_val_pred

        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
        val_r2 = safe_r2(y_val, y_val_pred)

        y_tr_pred = model.predict(X_tr)
        tr_mae = mean_absolute_error(y_tr, y_tr_pred)
        tr_rmse = mean_squared_error(y_tr, y_tr_pred) ** 0.5
        tr_r2 = safe_r2(y_tr, y_tr_pred)

        train_metrics.append((tr_mae, tr_rmse, tr_r2))
        valid_metrics.append((val_mae, val_rmse, val_r2))

        es_best_iters.append(es_info["best_iterations_es"])
        es_best_rmses.append(es_info["best_es_rmse"])
        es_stop_flags.append(int(es_info["es_stopped_early"]))

        print(
            f"[{MODEL_TAG.upper()} {tag} Fold {fold}/{n_splits}] "
            f"TRAIN_R2={tr_r2:.3f}, TRAIN_RMSE={tr_rmse:.3f} | "
            f"VALID_R2={val_r2:.3f}, VALID_RMSE={val_rmse:.3f} | "
            f"ES_BEST_ITERS={es_info['best_iterations_es']} | "
            f"GROW={es_info['grow_policy']} | BOOST={es_info['boosting_type']}"
        )

    train_metrics = np.array(train_metrics)
    valid_metrics = np.array(valid_metrics)

    mean_tr_mae, mean_tr_rmse, mean_tr_r2 = train_metrics.mean(axis=0)
    mean_v_mae, mean_v_rmse, mean_v_r2 = valid_metrics.mean(axis=0)

    valid_mae_oof = mean_absolute_error(y, oof_pred)
    valid_rmse_oof = mean_squared_error(y, oof_pred) ** 0.5
    valid_r2_oof = safe_r2(y, oof_pred)

    metrics = {
        "train_mae_cv": float(mean_tr_mae),
        "train_rmse_cv": float(mean_tr_rmse),
        "train_r2_cv": float(mean_tr_r2),

        "valid_mae_cv": float(mean_v_mae),
        "valid_rmse_cv": float(mean_v_rmse),
        "valid_r2_cv": float(mean_v_r2),

        "valid_mae": float(valid_mae_oof),
        "valid_rmse": float(valid_rmse_oof),
        "valid_r2": float(valid_r2_oof),

        "es_best_iterations_mean": float(np.mean(es_best_iters)),
        "es_best_iterations_std": float(np.std(es_best_iters)),
        "es_best_rmse_mean": float(np.nanmean(es_best_rmses)),
        "es_stopped_ratio": float(np.mean(es_stop_flags)),
    }
    return float(valid_rmse_oof), metrics, oof_pred


def build_xy_dropisna(df_in):
    y_all = df_in[TARGET_COL].astype(float).values

    drop_cols = [TARGET_COL, ROW_ID_COL] + [c for c in EXCLUDE_ISNA_COLS if c in df_in.columns]
    X_all = df_in.drop(columns=[c for c in drop_cols if c in df_in.columns])

    removed_isna = [c for c in EXCLUDE_ISNA_COLS if c in df_in.columns]
    print(f"[dropISNA] 已从特征中移除的缺失指示列: {removed_isna}")
    if ROW_ID_COL in df_in.columns:
        print(f"[dropISNA] 已从特征中移除追溯列: {ROW_ID_COL}")
    print(f"[dropISNA] 数据形状: X = {X_all.shape}, y = {y_all.shape}")

    return X_all, y_all


# ====================== 4. 主流程（仅 NO_OUTLIER_DELETE） ======================

def run_dropisna_split_catboost_no_outlier(n_splits: int, df_base_in: pd.DataFrame):
    mode = "NO_OUTLIER_DELETE"

    print("\n" + "=" * 100)
    print(f"开始场景：dropISNA + split(9:1) + K={n_splits} —— {mode} —— TARGET={TARGET_COL}")
    print("=" * 100)

    out_dir_preferred = os.path.join(OUTPUT_DIR_BASE, f"dropISNA_split9to1_K{n_splits}_{mode}")
    out_dir = get_out_dir(out_dir_preferred)
    print(f"[{MODEL_TAG.upper()}] 输出目录: {out_dir}")

    # 1) dropISNA（建模用 X/y）
    X_all, y_all = build_xy_dropisna(df_base_in)

    # 用于查找/对应的全量表
    df_all_table = df_base_in.copy().reset_index(drop=True)

    print(f"[dropISNA-K{n_splits}] {mode}: X={X_all.shape}, y={y_all.shape}")

    # 输出当前模式对应的全量表
    base_table_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_no_outlier_delete_table_K{n_splits}_{TARGET_COL}.csv")
    )
    df_all_table.to_csv(base_table_path, index=False, encoding="utf-8-sig")
    print(f"[Table] 当前模式全量表已保存: {base_table_path}")

    # 2) 9:1 split —— 保存索引
    indices = np.arange(len(y_all))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
    )
    np.save(ensure_parent_dir(os.path.join(out_dir, "train_indices.npy")), train_idx)
    np.save(ensure_parent_dir(os.path.join(out_dir, "test_indices.npy")), test_idx)
    print(f"[SPLIT] 已保存 train/test 索引: {out_dir}")

    X_train = X_all.iloc[train_idx].reset_index(drop=True)
    X_test = X_all.iloc[test_idx].reset_index(drop=True)
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    print(f"[split] Train={len(y_train)}, Test={len(y_test)} (9:1)")

    # 输出可一一对应的 train/test 原始表格（含 row_id）
    df_train_table = df_all_table.iloc[train_idx].reset_index(drop=True).copy()
    df_test_table = df_all_table.iloc[test_idx].reset_index(drop=True).copy()

    df_train_table["__split"] = "train"
    df_test_table["__split"] = "test"
    df_train_table["__mode"] = mode
    df_test_table["__mode"] = mode

    train_table_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_train_table_K{n_splits}_dropISNA_{mode}_{TARGET_COL}.csv")
    )
    test_table_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_test_table_K{n_splits}_dropISNA_{mode}_{TARGET_COL}.csv")
    )
    df_train_table.to_csv(train_table_path, index=False, encoding="utf-8-sig")
    df_test_table.to_csv(test_table_path, index=False, encoding="utf-8-sig")
    print(f"[Split Table] train 已保存: {train_table_path}")
    print(f"[Split Table] test  已保存: {test_table_path}")

    split_map_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_train_test_mapping_table_K{n_splits}_dropISNA_{mode}_{TARGET_COL}.csv")
    )
    pd.concat([df_train_table, df_test_table], axis=0, ignore_index=True).to_csv(
        split_map_path, index=False, encoding="utf-8-sig"
    )
    print(f"[Split Table] train+test mapping 已保存: {split_map_path}")

    # 3) Optuna：train 上 K 折
    def objective(trial):
        params = create_params_from_trial(trial)
        valid_rmse, metrics, _ = evaluate_params_kfold(
            params, X_train, y_train, n_splits=n_splits,
            tag=f"SPLIT_TRAIN_{mode}_K{n_splits}"
        )
        for k, v in metrics.items():
            trial.set_user_attr("train_" + k, float(v))
        return float(valid_rmse)

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{MODEL_TAG}_{TARGET_COL}_dropISNA_{mode}_trainK{n_splits}"
    )
    study.optimize(
        objective,
        n_trials=N_TRIALS_SPLIT,
        catch=(CatBoostError,)
    )

    best_trial = study.best_trial
    best_params = normalize_cb_params(best_trial.params)
    print(f"\n[BEST] trial={best_trial.number}, best_train_oof_RMSE={best_trial.value}")
    print("[BEST] params:", best_params)

    save_best_params_precise(
        best_params,
        out_dir=out_dir,
        prefix=f"{MODEL_TAG}_{TARGET_COL}_dropISNA_{mode}_K{n_splits}"
    )

    # trials 导出
    rows = []
    for t in study.trials:
        row = {"trial": t.number, "valid_rmse": t.value}
        row.update(t.params)
        row.update(t.user_attrs)
        for pn, pv in t.params.items():
            row[f"param_repr__{pn}"] = _float_to_str_full(pv)
        rows.append(row)

    df_trials = pd.DataFrame(rows)
    trials_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_bayes_trials_train_K{n_splits}_dropISNA_{mode}_{TARGET_COL}.csv")
    )
    df_trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    print(f"[TRIALS] saved: {trials_path}")
    save_trials_with_highlight(df_trials, best_trial.number, trials_path, sheet_name=f"K{n_splits}_{mode}")

    # 4) 固定 best_params：train OOF
    _, best_metrics_train, oof_pred_train = evaluate_params_kfold(
        best_params, X_train, y_train, n_splits=n_splits,
        tag=f"SPLIT_TRAIN_BEST_{mode}_K{n_splits}"
    )

    # OOF 导出加 row_id
    train_row_ids = df_train_table[ROW_ID_COL].values
    oof_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_oof_predictions_best_train_K{n_splits}_{mode}_{TARGET_COL}.csv")
    )
    pd.DataFrame({
        ROW_ID_COL: train_row_ids,
        "y_true": y_train,
        "y_pred_oof": oof_pred_train,
        "residual_oof": (y_train - oof_pred_train),
        "abs_residual_oof": np.abs(y_train - oof_pred_train),
    }).to_csv(oof_path, index=False, encoding="utf-8-sig")
    print(f"[OUT] train OOF（含row_id）已保存: {oof_path}")

    # 5) train 全量拟合 -> train_full_* 与 test_*
    model_final, final_es_info = fit_catboost_final_refit(
        best_params, X_train, y_train,
        tag=f"FINAL_FIT_{mode}_K{n_splits}"
    )

    model_path = ensure_parent_dir(os.path.join(out_dir, "final_model.joblib"))
    joblib.dump(model_final, model_path)
    print(f"[MODEL] 已保存 final 模型: {model_path}")

    native_model_path = ensure_parent_dir(os.path.join(out_dir, "final_model.cbm"))
    model_final.save_model(native_model_path)
    print(f"[MODEL] 已保存 CatBoost 原生模型: {native_model_path}")
    print(f"[MODEL] final effective iterations = {final_es_info['best_iterations_es']}")

    # 5.1 Train 全量拟合后的训练集拟合指标
    y_train_full_pred = model_final.predict(X_train)
    train_full = compute_metrics(y_train, y_train_full_pred)
    print(
        f"[TRAIN_FULL {mode}] R2={train_full['r2']:.3f}, RMSE={train_full['rmse']:.3f}, "
        f"MAE={train_full['mae']:.3f}"
    )

    train_full_pred_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_train_full_predictions_best_split_K{n_splits}_dropISNA_{mode}_{TARGET_COL}.csv")
    )
    pd.DataFrame({"y_true": y_train, "y_pred": y_train_full_pred}).to_csv(
        train_full_pred_path, index=False, encoding="utf-8-sig"
    )

    df_train_full_pred = pd.DataFrame({
        ROW_ID_COL: train_row_ids,
        "y_true": y_train,
        "y_pred": y_train_full_pred,
        "residual": (y_train - y_train_full_pred),
        "abs_residual": np.abs(y_train - y_train_full_pred),
        "__split": "train",
        "__mode": mode,
    })
    df_train_full_pred = df_train_full_pred.merge(
        df_train_table.drop(columns=["__split", "__mode"], errors="ignore"),
        on=ROW_ID_COL, how="left"
    )

    train_full_pred_table_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_train_full_pred_table_best_K{n_splits}_dropISNA_{mode}_{TARGET_COL}.csv")
    )
    df_train_full_pred.to_csv(train_full_pred_table_path, index=False, encoding="utf-8-sig")
    print(f"[OUT] train_full_pred 表格（含row_id/原始字段）已保存: {train_full_pred_table_path}")

    # 5.2 Test 指标
    y_test_pred = model_final.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_rmse = float(mean_squared_error(y_test, y_test_pred) ** 0.5)
    test_r2 = float(safe_r2(y_test, y_test_pred))
    print(f"[TEST {mode}] MAE={test_mae:.3f}, RMSE={test_rmse:.3f}, R2={test_r2:.3f}")

    test_pred_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_test_predictions_best_split_K{n_splits}_{mode}_{TARGET_COL}.csv")
    )
    pd.DataFrame({"y_true": y_test, "y_pred": y_test_pred}).to_csv(
        test_pred_path, index=False, encoding="utf-8-sig"
    )

    test_row_ids = df_test_table[ROW_ID_COL].values
    df_test_pred = pd.DataFrame({
        ROW_ID_COL: test_row_ids,
        "y_true": y_test,
        "y_pred": y_test_pred,
        "residual": (y_test - y_test_pred),
        "abs_residual": np.abs(y_test - y_test_pred),
        "__split": "test",
        "__mode": mode,
    }).merge(
        df_test_table.drop(columns=["__split", "__mode"], errors="ignore"),
        on=ROW_ID_COL, how="left"
    )
    test_pred_table_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_test_pred_table_best_K{n_splits}_dropISNA_{mode}_{TARGET_COL}.csv")
    )
    df_test_pred.to_csv(test_pred_table_path, index=False, encoding="utf-8-sig")
    print(f"[OUT] test_pred 表格（含row_id/原始字段）已保存: {test_pred_table_path}")

    # scatter（test）
    plt.figure(figsize=(7.8, 7.8))
    plt.scatter(y_test, y_test_pred, alpha=0.65, edgecolor="none")
    mn = float(min(y_test.min(), y_test_pred.min()))
    mx = float(max(y_test.max(), y_test_pred.max()))
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1.5)
    plt.xlabel(f"True {TARGET_COL}")
    plt.ylabel(f"Pred {TARGET_COL}")
    plt.title(f"[dropISNA {mode} K{n_splits}] TEST RMSE={test_rmse:.3f}, R2={test_r2:.3f}")
    plt.tight_layout()
    fig_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_test_true_vs_pred_K{n_splits}_{mode}_{TARGET_COL}.png")
    )
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # ======================
    # 汇总指标
    # ======================
    summary = {
        "target": TARGET_COL,
        "mode": mode,
        "dropISNA": True,
        "use_outlier_delete": False,
        "n_base_samples": int(len(df_all_table)),
        "n_splits": int(n_splits),
        "test_size": float(TEST_SIZE),
        "best_trial": int(best_trial.number),

        "best_train_oof_rmse": float(best_trial.value),

        "train_train_mae_cv": float(best_metrics_train["train_mae_cv"]),
        "train_train_rmse_cv": float(best_metrics_train["train_rmse_cv"]),
        "train_train_r2_cv": float(best_metrics_train["train_r2_cv"]),

        "train_valid_mae_cv": float(best_metrics_train["valid_mae_cv"]),
        "train_valid_rmse_cv": float(best_metrics_train["valid_rmse_cv"]),
        "train_valid_r2_cv": float(best_metrics_train["valid_r2_cv"]),

        "train_oof_mae": float(best_metrics_train["valid_mae"]),
        "train_oof_rmse": float(best_metrics_train["valid_rmse"]),
        "train_oof_r2": float(best_metrics_train["valid_r2"]),

        "train_full_mae": float(train_full["mae"]),
        "train_full_rmse": float(train_full["rmse"]),
        "train_full_r2": float(train_full["r2"]),

        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),

        "cv_es_best_iterations_mean": float(best_metrics_train["es_best_iterations_mean"]),
        "cv_es_best_iterations_std": float(best_metrics_train["es_best_iterations_std"]),
        "cv_es_best_rmse_mean": float(best_metrics_train["es_best_rmse_mean"]),
        "cv_es_stopped_ratio": float(best_metrics_train["es_stopped_ratio"]),
        "final_requested_iterations": int(final_es_info["requested_iterations"]),
        "final_effective_iterations": int(final_es_info["best_iterations_es"]),
        "final_es_rmse": float(final_es_info["best_es_rmse"]) if not np.isnan(final_es_info["best_es_rmse"]) else np.nan,
        "final_es_stopped_early": bool(final_es_info["es_stopped_early"]),
        "final_grow_policy": final_es_info["grow_policy"],
        "final_boosting_type": final_es_info["boosting_type"],

        "final_model_path": model_path,
        "final_model_native_path": native_model_path,
        "n_jobs": int(N_JOBS),

        "no_outlier_delete_table": os.path.abspath(no_outlier_delete_path),
        "base_table_path": os.path.abspath(safe_path_str(base_table_path)),
        "train_table": os.path.abspath(safe_path_str(train_table_path)),
        "test_table": os.path.abspath(safe_path_str(test_table_path)),
        "train_full_pred_table": os.path.abspath(safe_path_str(train_full_pred_table_path)),
        "test_pred_table": os.path.abspath(safe_path_str(test_pred_table_path)),
        "split_mapping_table": os.path.abspath(safe_path_str(split_map_path)),
    }

    summary_path = ensure_parent_dir(os.path.join(out_dir, "summary_metrics.csv"))
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[SUMMARY] saved: {summary_path}")
    print(f"[DONE] {TARGET_COL} / {mode} finished.\n")

    return pd.DataFrame([summary])


if __name__ == "__main__":
    all_summary = []

    df_base_mode = df_no_outlier_delete.copy()

    for k in N_SPLITS_LIST:
        one_summary = run_dropisna_split_catboost_no_outlier(
            n_splits=k,
            df_base_in=df_base_mode
        )
        all_summary.append(one_summary)

    if len(all_summary) > 0:
        df_all_summary = pd.concat(all_summary, axis=0, ignore_index=True)
        all_summary_path = ensure_parent_dir(os.path.join(OUTPUT_DIR_BASE, "summary_metrics_ALL_MODES.csv"))
        df_all_summary.to_csv(all_summary_path, index=False, encoding="utf-8-sig")
        print(f"[SUMMARY ALL] 已保存: {all_summary_path}")

    print("\n=== ALL DONE: ec (raw) | dropISNA | split(9:1) | NO_OUTLIER_DELETE | CatBoost + native early stop ===")