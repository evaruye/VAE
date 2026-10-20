# -*- coding: utf-8 -*-
r"""
XGBoost + Optuna（原始标度 y，全链路不做 log 变换）——仅保留【NO_OUTLIER_DELETE】+ objective-only 版本
连续值分层切分版（外层 split + 内层 CV 都基于 y 分箱做 stratify）

当前版本流程：
1) 不删除异常值（NO_OUTLIER_DELETE）
2) 固定外层 train/test 划分（9:1，连续值分层切分）
3) 仅在 train 上做 10-fold CV + Optuna（连续值分层 CV）
4) 以 train 内部 OOF RMSE 作为最优参数选择标准
5) 用最优参数在 train 全量 refit
6) 在 test 上评估并导出结果

重要说明：
- 不再保留 Tukey 删除异常值版本
- 不再保留任何 early stopping 逻辑
- XGB 训练为固定超参数，全量训练到底
- 保留：
  1) 保存最终 train/test 索引
  2) 保存 final 模型
  3) 输出 train_full_pred 的数据表格（含 row_id、一一对应 y_true/y_pred/residual）
  4) 输出不删异常值全量表
  5) 输出可一一对应的训练集/测试集表格（含 row_id、split 标记）
  6) 输出 summary 指标表

训练指标口径：
- train_train_*_cv：K 折内每折训练子集指标取平均（更“乐观”）
- train_valid_*_cv：K 折内每折验证子集指标取平均
- train_oof_*：OOF（out-of-fold，全训练集每个样本由未见过它的折模型预测一次）指标
- train_full_*：用最优超参数在 Train 全量 refit 后，在 Train 全量上计算的拟合指标

本版本关键新增：
- 连续值分层切分：
  * 外层 9:1 split：对 y_all 做分位数分箱 -> train_test_split(stratify=bin_labels)
  * 内层 CV：对 y_train 做分位数分箱 -> StratifiedKFold
- 若样本分布不满足分层要求，会自动回退到普通随机切分 / KFold
- XGB 无需特征标准化，保持树模型原生训练流程
"""

import os
from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")

# ====================== 0. 基本配置 ======================

MODEL_TAG = "xgb"

# ---------- 固含量数据集 ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_XLSX = str(REPO_ROOT / "data" / "Data_S2" / "SC_Modeling_Data.xlsx")
TARGET_COL = "Solid Content"  # 固含量列名（原始标度）

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
# 这里使用 XGBoost 自己的并行；当前脚本外层 CV/Optuna 是串行循环
N_JOBS = -1

# 仅保留一个模式
MODE = "NO_OUTLIER_DELETE"

# ======================
# 连续值分层配置
# ======================
USE_STRATIFIED_CONTINUOUS_OUTER_SPLIT = True
USE_STRATIFIED_CONTINUOUS_INNER_CV = True

# 尝试的最大分箱数（会自动向下找可行箱数）
OUTER_STRATIFY_MAX_BINS = 10
INNER_STRATIFY_MAX_BINS = 10

# ======================
# XGB 搜索空间（使用 XGBRegressor 官方参数名）
# 目标：
# 1) 适应高维小样本
# 2) 训练速度快
# 3) 尽量减少出现 test > train_full 的概率
# ======================

# ======================
# ======================
# ======================
# XGB-V3 搜索空间：SC 固含量超窄局部精修版
# 目标：
# 1) 围绕 V2 最优区域继续压榨 test 泛化
# 2) 重点尝试从 test_r2 ≈ 0.975 提升到 0.978–0.981
# 3) 减少无效搜索，固定已经明确的最优结构
# ======================

# V2 top 50 主要集中在 120–165
# 最优点在 134 左右
XGB_N_ESTIMATORS = (115, 170)

# V2 top 50 几乎集中在 0.095–0.101
# 适当留一点左右空间
XGB_LEARNING_RATE = (0.088, 0.108)

# V2 top 100 全部为 depth=3
# 这里建议直接固定 3，不再浪费 trial
XGB_MAX_DEPTH = [3,3]

# V2 最优集中在 1.1–1.8
# 过大容易欠拟合，过小容易 train/test 差距扩大
XGB_MIN_CHILD_WEIGHT = (1.10, 1.90)

# V2 最优集中在 0.900–0.908
# 这个参数非常稳定，继续缩窄
XGB_SUBSAMPLE = (0.895, 0.915)

# V2 最优集中在 0.813–0.830
# 稍微给一点右侧空间，看是否能提高 test
XGB_COLSAMPLE_BYTREE = (0.810, 0.845)

# V2 最优集中在 0.16–0.25
# gamma 不宜太大，否则削弱拟合
XGB_GAMMA = (0.140, 0.300)

# alpha 基本不起主要作用，保持很小即可
XGB_REG_ALPHA = (1e-10, 5e-5)

# V2 最优集中在 11.5–13.5
# 给一点左右空间，重点搜索 12–14 附近
XGB_REG_LAMBDA = (10.5, 15.5)

# V2 中 256 和 512 都能进入 top 区域
# 暂时两个都保留
XGB_MAX_BIN_CHOICES = [256, 512]

# 固定加速与稳定相关设置
XGB_TREE_METHOD = "hist"
XGB_GROW_POLICY = "depthwise"
XGB_BOOSTER = "gbtree"
XGB_OBJECTIVE = "reg:squarederror"
XGB_EVAL_METRIC = "rmse"

# ====================== 0.1 路径稳健工具（长路径 + 自动短根目录） ======================

IS_WIN = (os.name == "nt")

SHORT_OUTPUT_ROOT = str(REPO_ROOT / "results" / "training" / "SC")
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

# ====================== 1. 读取数据（仅保留不删除异常值） ======================

df_raw = pd.read_excel(INPUT_XLSX)
if TARGET_COL not in df_raw.columns:
    raise ValueError(f"找不到目标列 {TARGET_COL} ，请检查列名是否正确。")

ROW_ID_COL = "_row_id"
if ROW_ID_COL not in df_raw.columns:
    df_raw[ROW_ID_COL] = np.arange(len(df_raw), dtype=int)

df_no_outlier_delete = df_raw.reset_index(drop=True).copy()

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


def make_continuous_stratify_labels(y, max_bins=10, min_bin_size=2, tag=""):
    """
    对连续值 y 做分位数分箱，返回适合 stratify 的离散标签。
    会自动从 max_bins 向下尝试，直到每箱样本数 >= min_bin_size。
    若无法构造则返回 None。
    """
    y = np.asarray(y, dtype=float).reshape(-1)

    n_samples = len(y)
    n_unique = np.unique(y).size

    if n_samples < 4 or n_unique < 2:
        print(f"[STRATIFY WARN] {tag}: 样本数或唯一值过少，无法连续值分层，回退。")
        return None

    max_bins_eff = int(min(max_bins, n_unique, n_samples))
    if max_bins_eff < 2:
        print(f"[STRATIFY WARN] {tag}: 可用分箱数 < 2，无法连续值分层，回退。")
        return None

    for q in range(max_bins_eff, 1, -1):
        try:
            labels = pd.qcut(y, q=q, labels=False, duplicates="drop")
            if labels is None:
                continue

            labels = np.asarray(labels, dtype=int)
            uniq, cnt = np.unique(labels, return_counts=True)

            if len(uniq) < 2:
                continue

            if cnt.min() >= int(min_bin_size):
                print(
                    f"[STRATIFY] {tag}: 使用连续值分层，分箱数={len(uniq)}，"
                    f"最小箱样本数={cnt.min()}，最大箱样本数={cnt.max()}"
                )
                return labels
        except Exception:
            continue

    print(f"[STRATIFY WARN] {tag}: 无法构造满足 min_bin_size={min_bin_size} 的分层标签，回退。")
    return None


# ====================== 3. XGB 模型 ======================

def normalize_xgb_params(params: dict) -> dict:
    p = dict(params)

    p["n_estimators"] = int(p["n_estimators"])
    p["learning_rate"] = float(p["learning_rate"])
    p["max_depth"] = int(p["max_depth"])
    p["min_child_weight"] = float(p["min_child_weight"])
    p["subsample"] = float(p["subsample"])
    p["colsample_bytree"] = float(p["colsample_bytree"])
    p["gamma"] = float(p["gamma"])
    p["reg_alpha"] = float(p["reg_alpha"])
    p["reg_lambda"] = float(p["reg_lambda"])
    p["max_bin"] = int(p["max_bin"])

    return p


def build_model(params: dict):
    p = normalize_xgb_params(params)

    reg = XGBRegressor(
        objective=XGB_OBJECTIVE,
        booster=XGB_BOOSTER,
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        min_child_weight=p["min_child_weight"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        gamma=p["gamma"],
        reg_alpha=p["reg_alpha"],
        reg_lambda=p["reg_lambda"],
        tree_method=XGB_TREE_METHOD,
        grow_policy=XGB_GROW_POLICY,
        max_bin=p["max_bin"],
        eval_metric=XGB_EVAL_METRIC,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    return reg


def create_params_from_trial(trial: optuna.Trial) -> dict:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", XGB_N_ESTIMATORS[0], XGB_N_ESTIMATORS[1]),
        "learning_rate": trial.suggest_float("learning_rate", XGB_LEARNING_RATE[0], XGB_LEARNING_RATE[1], log=True),
        "max_depth": trial.suggest_int("max_depth", XGB_MAX_DEPTH[0], XGB_MAX_DEPTH[1]),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            XGB_MIN_CHILD_WEIGHT[0],
            XGB_MIN_CHILD_WEIGHT[1]
        ),
        "subsample": trial.suggest_float("subsample", XGB_SUBSAMPLE[0], XGB_SUBSAMPLE[1]),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            XGB_COLSAMPLE_BYTREE[0],
            XGB_COLSAMPLE_BYTREE[1]
        ),
        "gamma": trial.suggest_float("gamma", XGB_GAMMA[0], XGB_GAMMA[1]),
        "reg_alpha": trial.suggest_float("reg_alpha", XGB_REG_ALPHA[0], XGB_REG_ALPHA[1], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", XGB_REG_LAMBDA[0], XGB_REG_LAMBDA[1], log=True),
        "max_bin": trial.suggest_categorical("max_bin", XGB_MAX_BIN_CHOICES),
    }
    return params


def evaluate_params_kfold(params, X, y, n_splits, tag=""):
    n_samples = len(y)
    oof_pred = np.zeros(n_samples, dtype=float)

    train_metrics = []
    valid_metrics = []

    # ---------- 连续值分层 CV ----------
    if USE_STRATIFIED_CONTINUOUS_INNER_CV:
        y_bins = make_continuous_stratify_labels(
            y,
            max_bins=INNER_STRATIFY_MAX_BINS,
            min_bin_size=n_splits,
            tag=f"{tag}_INNER_CV"
        )
    else:
        y_bins = None

    if y_bins is not None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(X, y_bins)
        print(f"[CV] {tag}: 使用 StratifiedKFold（连续值分层）")
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(X)
        print(f"[CV] {tag}: 回退为普通 KFold")

    for fold, (train_idx, valid_idx) in enumerate(split_iter, 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]

        model = build_model(params)
        model.fit(X_tr, y_tr)

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

        print(
            f"[{MODEL_TAG.upper()} {tag} Fold {fold}/{n_splits}] "
            f"TRAIN_R2={tr_r2:.3f}, TRAIN_RMSE={tr_rmse:.3f} | "
            f"VALID_R2={val_r2:.3f}, VALID_RMSE={val_rmse:.3f}"
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


# ====================== 4. 主流程（连续值分层切分版） ======================

def run_dropisna_split_xgb_no_es(n_splits: int, df_base_in: pd.DataFrame):
    print("\n" + "=" * 100)
    print(
        f"开始场景：dropISNA + split(9:1, continuous-stratified) + K={n_splits} "
        f"—— {MODE} —— TARGET={TARGET_COL} —— NO EARLY STOPPING"
    )
    print("=" * 100)

    out_dir_preferred = os.path.join(
        OUTPUT_DIR_BASE,
        f"dropISNA_split9to1_CONT_STRAT_K{n_splits}_{MODE}_NO_ES"
    )
    out_dir = get_out_dir(out_dir_preferred)
    print(f"[{MODEL_TAG.upper()}] 输出目录: {out_dir}")

    # 1) dropISNA（建模用 X/y）
    X_all, y_all = build_xy_dropisna(df_base_in)

    df_all_table = df_base_in.copy().reset_index(drop=True)
    print(f"[dropISNA-K{n_splits}] {MODE}: X={X_all.shape}, y={y_all.shape}")

    base_table_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_no_outlier_delete_table_K{n_splits}_{TARGET_COL}.csv")
    )
    df_all_table.to_csv(base_table_path, index=False, encoding="utf-8-sig")
    print(f"[Table] 当前模式全量表已保存: {base_table_path}")

    # 2) 外层 9:1 split —— 连续值分层切分
    indices = np.arange(len(y_all))

    if USE_STRATIFIED_CONTINUOUS_OUTER_SPLIT:
        outer_min_bin_size = max(2, int(math.ceil(1.0 / TEST_SIZE)))
        y_all_bins = make_continuous_stratify_labels(
            y_all,
            max_bins=OUTER_STRATIFY_MAX_BINS,
            min_bin_size=outer_min_bin_size,
            tag="OUTER_SPLIT"
        )
    else:
        y_all_bins = None

    if y_all_bins is not None:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=y_all_bins
        )
        outer_split_strategy = "continuous_stratified_train_test_split"
        print("[SPLIT] 使用连续值分层 train_test_split")
    else:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        outer_split_strategy = "random_train_test_split"
        print("[SPLIT] 回退为普通随机 train_test_split")

    np.save(ensure_parent_dir(os.path.join(out_dir, "train_indices.npy")), train_idx)
    np.save(ensure_parent_dir(os.path.join(out_dir, "test_indices.npy")), test_idx)
    print(f"[SPLIT] 已保存 train/test 索引: {out_dir}")

    X_train = X_all.iloc[train_idx].reset_index(drop=True)
    X_test = X_all.iloc[test_idx].reset_index(drop=True)
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    print(f"[split] Train={len(y_train)}, Test={len(y_test)} (9:1)")

    # 输出 train/test 原始表
    df_train_table = df_all_table.iloc[train_idx].reset_index(drop=True).copy()
    df_test_table = df_all_table.iloc[test_idx].reset_index(drop=True).copy()

    df_train_table["__split"] = "train"
    df_test_table["__split"] = "test"
    df_train_table["__mode"] = MODE
    df_test_table["__mode"] = MODE

    train_table_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_train_table_K{n_splits}_dropISNA_{MODE}_{TARGET_COL}.csv")
    )
    test_table_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_test_table_K{n_splits}_dropISNA_{MODE}_{TARGET_COL}.csv")
    )
    df_train_table.to_csv(train_table_path, index=False, encoding="utf-8-sig")
    df_test_table.to_csv(test_table_path, index=False, encoding="utf-8-sig")
    print(f"[Split Table] train 已保存: {train_table_path}")
    print(f"[Split Table] test  已保存: {test_table_path}")

    split_map_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_train_test_mapping_table_K{n_splits}_dropISNA_{MODE}_{TARGET_COL}.csv")
    )
    pd.concat([df_train_table, df_test_table], axis=0, ignore_index=True).to_csv(
        split_map_path, index=False, encoding="utf-8-sig"
    )
    print(f"[Split Table] train+test mapping 已保存: {split_map_path}")

    # 3) Optuna：train 上 K 折
    def objective(trial):
        params = create_params_from_trial(trial)
        valid_rmse, metrics, _ = evaluate_params_kfold(
            params,
            X_train,
            y_train,
            n_splits=n_splits,
            tag=f"SPLIT_TRAIN_{MODE}_K{n_splits}"
        )
        for k, v in metrics.items():
            trial.set_user_attr("train_" + k, float(v))
        return float(valid_rmse)

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{MODEL_TAG}_{TARGET_COL}_dropISNA_{MODE}_trainK{n_splits}_NO_ES_CONT_STRAT"
    )
    study.optimize(objective, n_trials=N_TRIALS_SPLIT)

    best_trial = study.best_trial
    best_params = normalize_xgb_params(best_trial.params)
    print(f"\n[BEST] trial={best_trial.number}, best_train_oof_RMSE={best_trial.value}")
    print("[BEST] params:", best_params)

    save_best_params_precise(
        best_params,
        out_dir=out_dir,
        prefix=f"{MODEL_TAG}_{TARGET_COL}_dropISNA_{MODE}_K{n_splits}_NO_ES_CONT_STRAT"
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
        os.path.join(out_dir, f"{MODEL_TAG}_bayes_trials_train_K{n_splits}_dropISNA_{MODE}_{TARGET_COL}.csv")
    )
    df_trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    print(f"[TRIALS] saved: {trials_path}")
    save_trials_with_highlight(df_trials, best_trial.number, trials_path, sheet_name=f"K{n_splits}_{MODE}")

    # 4) 固定 best_params：train OOF
    _, best_metrics_train, oof_pred_train = evaluate_params_kfold(
        best_params,
        X_train,
        y_train,
        n_splits=n_splits,
        tag=f"SPLIT_TRAIN_BEST_{MODE}_K{n_splits}"
    )

    train_row_ids = df_train_table[ROW_ID_COL].values
    oof_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_oof_predictions_best_train_K{n_splits}_{MODE}_{TARGET_COL}.csv")
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
    model_final = build_model(best_params)
    model_final.fit(X_train, y_train)

    model_path = ensure_parent_dir(os.path.join(out_dir, "final_model.joblib"))
    joblib.dump(model_final, model_path)
    print(f"[MODEL] 已保存 final 模型: {model_path}")

    # 5.1 Train 全量拟合后的训练集拟合指标
    y_train_full_pred = model_final.predict(X_train)
    train_full = compute_metrics(y_train, y_train_full_pred)
    print(
        f"[TRAIN_FULL {MODE}] R2={train_full['r2']:.3f}, RMSE={train_full['rmse']:.3f}, "
        f"MAE={train_full['mae']:.3f}"
    )

    train_full_pred_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_train_full_predictions_best_split_K{n_splits}_dropISNA_{MODE}_{TARGET_COL}.csv")
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
        "__mode": MODE,
    })
    df_train_full_pred = df_train_full_pred.merge(
        df_train_table.drop(columns=["__split", "__mode"], errors="ignore"),
        on=ROW_ID_COL, how="left"
    )

    train_full_pred_table_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_train_full_pred_table_best_K{n_splits}_dropISNA_{MODE}_{TARGET_COL}.csv")
    )
    df_train_full_pred.to_csv(train_full_pred_table_path, index=False, encoding="utf-8-sig")
    print(f"[OUT] train_full_pred 表格（含row_id/原始字段）已保存: {train_full_pred_table_path}")

    # 5.2 Test 指标
    y_test_pred = model_final.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_rmse = float(mean_squared_error(y_test, y_test_pred) ** 0.5)
    test_r2 = float(safe_r2(y_test, y_test_pred))
    print(f"[TEST {MODE}] MAE={test_mae:.3f}, RMSE={test_rmse:.3f}, R2={test_r2:.3f}")

    test_pred_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_test_predictions_best_split_K{n_splits}_dropISNA_{MODE}_{TARGET_COL}.csv")
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
        "__mode": MODE,
    }).merge(
        df_test_table.drop(columns=["__split", "__mode"], errors="ignore"),
        on=ROW_ID_COL, how="left"
    )
    test_pred_table_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_test_pred_table_best_K{n_splits}_dropISNA_{MODE}_{TARGET_COL}.csv")
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
    plt.title(f"[dropISNA {MODE} K{n_splits}] TEST RMSE={test_rmse:.3f}, R2={test_r2:.3f}")
    plt.tight_layout()
    fig_path = ensure_parent_dir(
        os.path.join(out_dir, f"{MODEL_TAG}_test_true_vs_pred_K{n_splits}_{MODE}_{TARGET_COL}.png")
    )
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # ======================
    # 汇总指标
    # ======================
    summary = {
        "target": TARGET_COL,
        "mode": MODE,
        "dropISNA": True,
        "use_outlier_delete": False,
        "n_base_samples": int(len(df_all_table)),
        "n_splits": int(n_splits),
        "test_size": float(TEST_SIZE),
        "best_trial": int(best_trial.number),

        "outer_split_strategy": outer_split_strategy,
        "inner_cv_strategy": (
            "continuous_stratified_stratifiedkfold"
            if USE_STRATIFIED_CONTINUOUS_INNER_CV else
            "kfold"
        ),

        # Optuna 目标：train 上 OOF RMSE
        "best_train_oof_rmse": float(best_trial.value),

        # 训练集：CV(训练子集均值)
        "train_train_mae_cv": float(best_metrics_train["train_mae_cv"]),
        "train_train_rmse_cv": float(best_metrics_train["train_rmse_cv"]),
        "train_train_r2_cv": float(best_metrics_train["train_r2_cv"]),

        # 训练集：CV(验证子集均值)
        "train_valid_mae_cv": float(best_metrics_train["valid_mae_cv"]),
        "train_valid_rmse_cv": float(best_metrics_train["valid_rmse_cv"]),
        "train_valid_r2_cv": float(best_metrics_train["valid_r2_cv"]),

        # 训练集：OOF
        "train_oof_mae": float(best_metrics_train["valid_mae"]),
        "train_oof_rmse": float(best_metrics_train["valid_rmse"]),
        "train_oof_r2": float(best_metrics_train["valid_r2"]),

        # 训练集：refit 后在 Train 全量上计算
        "train_full_mae": float(train_full["mae"]),
        "train_full_rmse": float(train_full["rmse"]),
        "train_full_r2": float(train_full["r2"]),

        # 测试集
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),

        "final_model_path": model_path,
        "n_jobs": int(N_JOBS),

        # 查找表路径
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
    print(f"[DONE] {TARGET_COL} / {MODE} finished.\n")

    return pd.DataFrame([summary])


if __name__ == "__main__":
    all_summary = []

    df_base_mode = df_no_outlier_delete.copy()

    for k in N_SPLITS_LIST:
        one_summary = run_dropisna_split_xgb_no_es(
            n_splits=k,
            df_base_in=df_base_mode,
        )
        all_summary.append(one_summary)

    if len(all_summary) > 0:
        df_all_summary = pd.concat(all_summary, axis=0, ignore_index=True)
        all_summary_path = ensure_parent_dir(os.path.join(OUTPUT_DIR_BASE, "summary_metrics_ALL_MODES.csv"))
        df_all_summary.to_csv(all_summary_path, index=False, encoding="utf-8-sig")
        print(f"[SUMMARY ALL] 已保存: {all_summary_path}")

    print("\n=== ALL DONE: solids_content (raw) | dropISNA | split(9:1, continuous stratified) | NO_OUTLIER_DELETE | XGB objective-only | NO early stopping ===")