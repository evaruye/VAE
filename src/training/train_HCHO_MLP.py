# -*- coding: utf-8 -*-
"""
MLP 直接回归对比模型（仅 Direct Regression，hcho 预测，V4：只修改超参数搜索空间）

本版本功能：
1) 只保留一个目标列 TARGET_COL（已是 log1p 目标）
2) 仅保留 Direct Regression：
   - 使用 MLPRegressor
   - Optuna 调参
   - 输出 train / valid(OOF) / test 的 MAE / RMSE / R2
   - 输出 train / OOF / test 散点图和表格
3) 导出每一次 Optuna trial 的完整参数表（csv + xlsx）
   - xlsx 中最优 trial 用红色高亮
4) 打印 Direct Regression 最优参数
5) 固定关闭 early_stopping，避免 fold-train 内部再切验证集
6) 修复数据泄露风险：
   - 不在 train/test 划分前对全数据做 fillna(median)
   - 缺失值统一放到 Pipeline 中，由 SimpleImputer(strategy="median") 在每个 fold 的训练子集上拟合
7) V4 修改重点：
   - 不修改 Optuna 目标函数，仍然 return OOF-RMSE
   - 仅根据上一轮 500 次结果收缩 MLP 超参数搜索空间
   - 固定 activation="tanh"，固定 batch_size=32
   - 只搜索中等规模 3 层网络、alpha、learning_rate_init、max_iter、tol、n_iter_no_change
"""

import os
from pathlib import Path
import re
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# ====================== 0. 基本配置 ======================

MODEL_TAG = "mlp"

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_XLSX = str(REPO_ROOT / "data" / "Data_S2" / "HCHO_Modeling_Data.xlsx")
TARGET_COL = "Free HCHO Content"   # 这里假定该列已经是 log1p 目标

EXCLUDE_ISNA_COLS = [
    "oxidizer_pct2_isna",
    "init_pressure_isna",
    "time_isna",
    "pH_buffer_pct_isna",
]

RANDOM_SEED = 23
N_SPLITS = 10
TEST_SIZE = 0.1
N_JOBS = 1   # MLPRegressor 本身不使用 n_jobs，这里仅保留配置
N_TRIALS = 500

# Optuna
import optuna
from optuna.samplers import TPESampler

DATA_DIR = os.path.dirname(INPUT_XLSX) or "."
OUTPUT_DIR_BASE = str(REPO_ROOT / "results" / "training" / "HCHO")
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
print(f"[{MODEL_TAG}] 输出目录: {OUTPUT_DIR_BASE}")


# ====================== 1. 工具函数 ======================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return 0.0
    if np.allclose(y_true, y_true[0]):
        return 0.0
    return float(r2_score(y_true, y_pred))


def metrics_pack(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "r2": float(safe_r2(y_true, y_pred)),
    }


def clip_nonneg(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, None)


def _excel_writer(path: str):
    try:
        return pd.ExcelWriter(path, engine="xlsxwriter")
    except Exception:
        return pd.ExcelWriter(path, engine="openpyxl")


def _detect_and_drop_index_cols(df: pd.DataFrame):
    patterns = [
        r"^row_index$", r"^rowindex$",
        r"^_?row_?id$", r"^_?rowid$",
        r"^id$", r"^_?id$",
        r"^index$", r"^_?index$",
        r"^idx$", r"^_?idx$",
    ]
    drop = []
    for c in df.columns:
        lc = str(c).strip().lower()
        for p in patterns:
            if re.match(p, lc):
                drop.append(c)
                break
    return drop


def scatter_plot(y_true, y_pred, title, out_png, xlabel="True", ylabel="Pred"):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(7.8, 7.2))
    plt.scatter(y_true, y_pred, alpha=0.65)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def study_to_trials_dataframe(study: optuna.Study, preferred_param_cols=None):
    """
    导出 Optuna 每个 trial 的完整信息表
    """
    preferred_param_cols = preferred_param_cols or []
    best_number = study.best_trial.number

    rows = []
    for t in study.trials:
        row = {
            "trial_number": t.number,
            "is_best": 1 if t.number == best_number else 0,
            "state": str(t.state),
            "objective_value": np.nan if t.value is None else float(t.value),
        }

        for k, v in t.params.items():
            row[k] = v

        if hasattr(t, "user_attrs") and t.user_attrs:
            for k, v in t.user_attrs.items():
                row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    ordered_front = ["trial_number", "is_best", "state", "objective_value"]
    ordered_front += [c for c in preferred_param_cols if c in df.columns]
    remain = [c for c in df.columns if c not in ordered_front]
    df = df[ordered_front + remain]
    return df


def export_trials_table_with_best_highlight(df_trials: pd.DataFrame, xlsx_path: str, sheet_name="optuna_trials"):
    """
    导出 trial 表并把最优 trial 整行标红
    """
    try:
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df_trials.to_excel(writer, index=False, sheet_name=sheet_name)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            header_fmt = workbook.add_format({
                "bold": True,
                "bg_color": "#D9EAF7",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            })
            best_fmt = workbook.add_format({
                "font_color": "red",
                "bold": True,
                "bg_color": "#FFF2F2",
            })

            for col_idx, col_name in enumerate(df_trials.columns):
                worksheet.write(0, col_idx, col_name, header_fmt)
                width = max(
                    12,
                    min(
                        30,
                        int(max(df_trials[col_name].astype(str).map(len).max() if len(df_trials) > 0 else 0, len(str(col_name))) + 2)
                    )
                )
                worksheet.set_column(col_idx, col_idx, width)

            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, len(df_trials), len(df_trials.columns) - 1)

            best_rows = df_trials.index[df_trials["is_best"] == 1].tolist()
            for r in best_rows:
                worksheet.set_row(r + 1, None, best_fmt)

    except Exception:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_trials.to_excel(writer, index=False, sheet_name=sheet_name)


# ====================== 2. 数据构建（只保留 log1p 目标；不做全数据填补） ======================

def build_xy(df_in: pd.DataFrame, drop_isna_cols: bool):
    """
    仅保留一个目标：TARGET_COL（已是 log1p）
    ynz 仅用于分层 CV，不参与回归建模

    修复点：
    - 不再在这里对全数据做 fillna(median)
    - 缺失值留给 Pipeline 中的 SimpleImputer 处理
    """
    if TARGET_COL not in df_in.columns:
        raise ValueError(f"缺少目标列 {TARGET_COL}")

    y = pd.to_numeric(df_in[TARGET_COL], errors="coerce").astype(float).values
    if np.isnan(y).any():
        raise ValueError(
            f"目标列 {TARGET_COL} 中存在缺失值或非数值，"
            f"请先清洗目标列。NaN个数={int(np.isnan(y).sum())}"
        )

    ynz = (y > 0.0).astype(int)

    drop_cols = [TARGET_COL]
    if drop_isna_cols:
        drop_cols += [c for c in EXCLUDE_ISNA_COLS if c in df_in.columns]
    drop_cols += _detect_and_drop_index_cols(df_in)

    X = df_in.drop(columns=[c for c in drop_cols if c in df_in.columns]).copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y, ynz


# ====================== 3. MLP 模型与 Optuna 搜索空间 ======================

def _sample_hidden_layers(trial: optuna.Trial) -> Tuple[int, ...]:
    """
    V4：只修改超参数搜索空间，不修改目标函数。

    上一轮 500 次结果显示最优区域集中在：
    - activation = tanh
    - batch_size = 32
    - hidden_layer_sizes 接近 (128, 128, 64)

    因此这里不再搜索 1 层/2 层网络，也不再搜索 16/256 等过小或过大的宽度，
    而是只在更可能提升泛化的中等 3 层结构附近搜索。
    """
    arch_map = {
        "128_128_64": (128, 128, 64),   # 当前最优区域，保留
        "128_96_64":  (128, 96, 64),
        "96_128_64":  (96, 128, 64),
        "96_96_64":   (96, 96, 64),
        "96_96_48":   (96, 96, 48),
        "64_128_64":  (64, 128, 64),
        "64_96_64":   (64, 96, 64),
        "128_64_32":  (128, 64, 32),
        "96_64_32":   (96, 64, 32),
        "64_64_32":   (64, 64, 32),
    }

    arch_key = trial.suggest_categorical("hidden_arch", list(arch_map.keys()))
    layers = tuple(int(x) for x in arch_map[arch_key])

    # 作为 user_attr 额外保存，方便导出的 trial 表阅读。
    trial.set_user_attr("n_layers", len(layers))
    for i, units in enumerate(layers, 1):
        trial.set_user_attr(f"n_units_l{i}", int(units))

    return layers


def _normalize_best_params(best_params: dict) -> dict:
    """
    统一确保 build_model 一定能拿到 hidden_layer_sizes。

    兼容三种情况：
    1) 旧版 trial：n_layers / n_units_l* 形式
    2) 新版 V4 trial：hidden_arch 形式
    3) 手动传入：hidden_layer_sizes 形式
    """
    p = {} if best_params is None else dict(best_params)

    arch_map = {
        "128_128_64": (128, 128, 64),
        "128_96_64":  (128, 96, 64),
        "96_128_64":  (96, 128, 64),
        "96_96_64":   (96, 96, 64),
        "96_96_48":   (96, 96, 48),
        "64_128_64":  (64, 128, 64),
        "64_96_64":   (64, 96, 64),
        "128_64_32":  (128, 64, 32),
        "96_64_32":   (96, 64, 32),
        "64_64_32":   (64, 64, 32),
    }

    if "hidden_layer_sizes" in p:
        h = p["hidden_layer_sizes"]
        if isinstance(h, tuple):
            return p
        if isinstance(h, list):
            p["hidden_layer_sizes"] = tuple(int(x) for x in h)
            return p
        if isinstance(h, str):
            nums = re.findall(r"\d+", h)
            if len(nums) > 0:
                p["hidden_layer_sizes"] = tuple(int(x) for x in nums)
                return p
        p["hidden_layer_sizes"] = (128, 128, 64)
        return p

    if "hidden_arch" in p and str(p["hidden_arch"]) in arch_map:
        p["hidden_layer_sizes"] = arch_map[str(p["hidden_arch"])]
        return p

    if "n_layers" in p:
        n_layers = int(p.get("n_layers", 3))
        layers = []
        for i in range(n_layers):
            key = f"n_units_l{i+1}"
            if key in p:
                layers.append(int(p[key]))
        if len(layers) == 0:
            layers = [128, 128, 64]
        p["hidden_layer_sizes"] = tuple(layers)
    else:
        p["hidden_layer_sizes"] = (128, 128, 64)

    return p


def build_model(params: dict) -> Pipeline:
    p = {} if params is None else dict(params)
    p = _normalize_best_params(p)

    reg = MLPRegressor(
        hidden_layer_sizes=tuple(p.get("hidden_layer_sizes", (64, 64))),
        activation=str(p.get("activation", "tanh")),
        solver="adam",
        alpha=float(p.get("alpha", 5e-3)),
        learning_rate_init=float(p.get("learning_rate_init", 1.3e-2)),
        batch_size=int(p.get("batch_size", 32)),
        max_iter=int(p.get("max_iter", 2000)),
        tol=float(p.get("tol", 1e-4)),
        n_iter_no_change=int(p.get("n_iter_no_change", 50)),
        early_stopping=False,   # 固定关闭早停，不引入 fold-train 内部验证集
        random_state=RANDOM_SEED,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", reg)
    ])


def create_params_from_trial(trial: optuna.Trial) -> dict:
    """
    V4：目标函数不变，只修改 MLP 超参数搜索空间。

    修改依据：上一轮 500 次结果的高排名 trial 集中在：
    - tanh
    - batch_size=32
    - 三层网络，尤其接近 (128, 128, 64)
    - learning_rate_init 约 0.013–0.0145
    - alpha 约 1e-3–1e-2 量级

    因此这里收缩搜索空间，避免继续浪费 trial 在明显较差的 relu、1/2 层、过大/过小 batch 上。
    """
    hidden_layer_sizes = _sample_hidden_layers(trial)

    # 由上一轮结果固定，不再搜索。
    activation = "tanh"
    batch_size = 32

    # 只在上一轮最优区域附近继续细搜。
    alpha = trial.suggest_float("alpha", 5e-4, 5e-2, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 0.0095, 0.0165)
    max_iter = trial.suggest_int("max_iter", 1000, 3000)

    # 训练停止相关超参数：仍然不启用 validation early_stopping，
    # 只控制训练 loss 收敛容忍度，属于 MLP 自身超参数。
    tol = trial.suggest_float("tol", 1e-5, 2e-4, log=True)
    n_iter_no_change = trial.suggest_int("n_iter_no_change", 30, 100)

    return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "batch_size": batch_size,
        "max_iter": max_iter,
        "tol": tol,
        "n_iter_no_change": n_iter_no_change,
    }


# ====================== 4. Direct Regression：CV / Optuna / 训练 ======================

def compute_oof_direct_regression(X_train, y_train, ynz_train, reg_params):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    n = len(y_train)
    oof_pred = np.zeros(n, dtype=float)

    reg_params = _normalize_best_params(reg_params)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, ynz_train), 1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr = y_train[tr_idx]

        reg = build_model(reg_params)
        reg.fit(X_tr, y_tr)

        pred_va = clip_nonneg(reg.predict(X_va))
        oof_pred[va_idx] = pred_va

        m_fold = metrics_pack(y_train[va_idx], pred_va)
        print(f"[MLP][Fold {fold}/{N_SPLITS}] RMSE={m_fold['rmse']:.4f} | MAE={m_fold['mae']:.4f} | R2={m_fold['r2']:.4f}")

    return oof_pred


def optuna_search_direct_regression(X_train, y_train, ynz_train):
    def objective(trial: optuna.Trial):
        params = create_params_from_trial(trial)

        # 为了 trial 表更可读，把隐藏层结构也记到 user_attrs 里
        trial.set_user_attr("hidden_layer_sizes", str(params["hidden_layer_sizes"]))

        oof_pred = compute_oof_direct_regression(X_train, y_train, ynz_train, params)
        m = metrics_pack(y_train, oof_pred)

        trial.set_user_attr("oof_mae", float(m["mae"]))
        trial.set_user_attr("oof_rmse", float(m["rmse"]))
        trial.set_user_attr("oof_r2", float(m["r2"]))
        return m["rmse"]

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    return study.best_params, float(study.best_value), study


def fit_full_direct_regression(X_train, y_train, reg_params):
    reg_params = _normalize_best_params(reg_params)
    reg = build_model(reg_params)
    reg.fit(X_train, y_train)
    return reg


def predict_direct_regression(reg, X):
    return clip_nonneg(reg.predict(X))


# ====================== 5. Direct Regression 主流程 ======================

def run_direct_regression(
    X_train, X_test,
    y_train, y_test,
    ynz_train, ynz_test,
    train_row_index, test_row_index
):
    out_dir = os.path.join(OUTPUT_DIR_BASE, "direct_regression")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 160)
    print("[RUN] MLP Direct Regression")
    print("=" * 160)

    # ---------- Optuna ----------
    print("\n[Optuna][Direct] Objective = OOF-RMSE")
    best_params, best_oof_rmse, study = optuna_search_direct_regression(
        X_train, y_train, ynz_train
    )

    # 关键修复：best_params 里只有 n_layers / n_units_l* 时，补回 hidden_layer_sizes
    best_params = _normalize_best_params(best_params)

    # ---------- 导出 trial 表 ----------
    preferred_cols = [
        "hidden_arch",
        "n_layers",
        "n_units_l1",
        "n_units_l2",
        "n_units_l3",
        "activation",
        "alpha",
        "learning_rate_init",
        "batch_size",
        "max_iter",
        "tol",
        "n_iter_no_change",
        "hidden_layer_sizes",
    ]
    df_trials = study_to_trials_dataframe(study, preferred_param_cols=preferred_cols)
    df_trials.to_csv(
        os.path.join(out_dir, "optuna_trials_direct.csv"),
        index=False, encoding="utf-8-sig"
    )
    export_trials_table_with_best_highlight(
        df_trials,
        os.path.join(out_dir, "optuna_trials_direct.xlsx"),
        sheet_name="optuna_trials_direct"
    )

    # ---------- 打印最优参数 ----------
    print("\n================== BEST HYPERPARAMS (MLP Direct Regression) ==================")
    for k in sorted(best_params.keys()):
        print(f"{k}: {best_params[k]}")
    print("================================================================================")

    best_params_to_save = dict(best_params)
    best_params_to_save["hidden_layer_sizes"] = str(best_params_to_save["hidden_layer_sizes"])
    pd.DataFrame([best_params_to_save]).to_csv(
        os.path.join(out_dir, "best_direct_params.csv"),
        index=False, encoding="utf-8-sig"
    )

    # ---------- OOF ----------
    print("\n[CV][Direct] Computing OOF with best params ...")
    oof_pred = compute_oof_direct_regression(
        X_train, y_train, ynz_train, best_params
    )
    m_oof = metrics_pack(y_train, oof_pred)

    # ---------- Full fit ----------
    reg = fit_full_direct_regression(X_train, y_train, best_params)

    pred_train = predict_direct_regression(reg, X_train)
    pred_test = predict_direct_regression(reg, X_test)

    m_train = metrics_pack(y_train, pred_train)
    m_test = metrics_pack(y_test, pred_test)

    print("\n================== MLP Direct Regression Metrics ==================")
    print(f"VALID(OOF)  | MAE={m_oof['mae']:.4f} | RMSE={m_oof['rmse']:.4f} | R2={m_oof['r2']:.4f}")
    print(f"TRAIN_FULL  | MAE={m_train['mae']:.4f} | RMSE={m_train['rmse']:.4f} | R2={m_train['r2']:.4f}")
    print(f"TEST        | MAE={m_test['mae']:.4f} | RMSE={m_test['rmse']:.4f} | R2={m_test['r2']:.4f}")
    print("===================================================================")

    # ---------- Summary ----------
    summary = {
        "model_type": "mlp_direct_regression",
        "optuna_best_oof_rmse": float(best_oof_rmse),

        "valid_oof_mae": m_oof["mae"],
        "valid_oof_rmse": m_oof["rmse"],
        "valid_oof_r2": m_oof["r2"],

        "train_full_mae": m_train["mae"],
        "train_full_rmse": m_train["rmse"],
        "train_full_r2": m_train["r2"],

        "test_mae": m_test["mae"],
        "test_rmse": m_test["rmse"],
        "test_r2": m_test["r2"],
    }
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False, encoding="utf-8-sig")

    # ---------- 表格 ----------
    df_oof = pd.DataFrame({
        "row_index": train_row_index,
        "y_true": y_train,
        "is_nonzero_true": ynz_train,
        "pred_oof": oof_pred,
    })

    df_train_full = pd.DataFrame({
        "row_index": train_row_index,
        "y_true": y_train,
        "is_nonzero_true": ynz_train,
        "pred_fit": pred_train,
    })

    df_test = pd.DataFrame({
        "row_index": test_row_index,
        "y_true": y_test,
        "is_nonzero_true": ynz_test,
        "pred_hat": pred_test,
    })

    df_oof.to_csv(os.path.join(out_dir, "direct_train_oof_table.csv"), index=False, encoding="utf-8-sig")
    df_train_full.to_csv(os.path.join(out_dir, "direct_train_full_table.csv"), index=False, encoding="utf-8-sig")
    df_test.to_csv(os.path.join(out_dir, "direct_test_table.csv"), index=False, encoding="utf-8-sig")

    # ---------- 图 ----------
    scatter_plot(
        y_train, oof_pred,
        title=f"[MLP Direct OOF] MAE={m_oof['mae']:.3f}, RMSE={m_oof['rmse']:.3f}, R2={m_oof['r2']:.3f}",
        out_png=os.path.join(out_dir, "direct_scatter_oof.png"),
        xlabel="True (log1p)", ylabel="Pred (log1p)"
    )
    scatter_plot(
        y_train, pred_train,
        title=f"[MLP Direct TRAIN_FULL] MAE={m_train['mae']:.3f}, RMSE={m_train['rmse']:.3f}, R2={m_train['r2']:.3f}",
        out_png=os.path.join(out_dir, "direct_scatter_train_full.png"),
        xlabel="True (log1p)", ylabel="Pred (log1p)"
    )
    scatter_plot(
        y_test, pred_test,
        title=f"[MLP Direct TEST] MAE={m_test['mae']:.3f}, RMSE={m_test['rmse']:.3f}, R2={m_test['r2']:.3f}",
        out_png=os.path.join(out_dir, "direct_scatter_test.png"),
        xlabel="True (log1p)", ylabel="Pred (log1p)"
    )

    # ---------- XLSX 汇总 ----------
    out_xlsx = os.path.join(out_dir, "reports_tables.xlsx")
    with _excel_writer(out_xlsx) as writer:
        df_summary.to_excel(writer, index=False, sheet_name="summary_metrics")
        df_oof.to_excel(writer, index=False, sheet_name="train_oof")
        df_train_full.to_excel(writer, index=False, sheet_name="train_full")
        df_test.to_excel(writer, index=False, sheet_name="test")
        df_trials.to_excel(writer, index=False, sheet_name="optuna_trials")

    # ---------- 保存模型 ----------
    joblib.dump(
        {
            "model_type": "mlp_direct_regression",
            "reg": reg,
            "best_params": best_params,
            "random_seed": RANDOM_SEED,
        },
        os.path.join(out_dir, "direct_regression_final_model.joblib")
    )

    print(f"\n[saved] {out_dir}")
    return df_summary


# ====================== 6. 主程序 ======================

if __name__ == "__main__":
    print("\n[LOAD] reading data ...")
    df_all = pd.read_excel(INPUT_XLSX).copy().reset_index(drop=True)
    df_all.insert(0, "row_index", np.arange(len(df_all), dtype=int))

    X_all, y_all, ynz_all = build_xy(df_all, drop_isna_cols=True)

    idx_all = np.arange(len(y_all), dtype=int)
    tr_idx, te_idx = train_test_split(
        idx_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=ynz_all
    )

    X_train = X_all.iloc[tr_idx].reset_index(drop=True)
    X_test = X_all.iloc[te_idx].reset_index(drop=True)

    y_train = y_all[tr_idx]
    y_test = y_all[te_idx]

    ynz_train = ynz_all[tr_idx]
    ynz_test = ynz_all[te_idx]

    train_row_index = df_all["row_index"].values[tr_idx]
    test_row_index = df_all["row_index"].values[te_idx]

    print("\n[DATA SUMMARY]")
    print(f"Total samples   : {len(y_all)}")
    print(f"Train samples   : {len(y_train)}")
    print(f"Test samples    : {len(y_test)}")
    print(f"Total zero      : {int(np.sum(ynz_all == 0))}")
    print(f"Total non-zero  : {int(np.sum(ynz_all == 1))}")
    print(f"Train zero      : {int(np.sum(ynz_train == 0))}")
    print(f"Train non-zero  : {int(np.sum(ynz_train == 1))}")
    print(f"Test zero       : {int(np.sum(ynz_test == 0))}")
    print(f"Test non-zero   : {int(np.sum(ynz_test == 1))}")

    s_direct = run_direct_regression(
        X_train, X_test,
        y_train, y_test,
        ynz_train, ynz_test,
        train_row_index, test_row_index
    )

    df_all_summary = pd.concat([s_direct], axis=0, ignore_index=True)
    df_all_summary.to_csv(
        os.path.join(OUTPUT_DIR_BASE, "summary_metrics_ALL.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with _excel_writer(os.path.join(OUTPUT_DIR_BASE, "summary_metrics_ALL.xlsx")) as writer:
        df_all_summary.to_excel(writer, index=False, sheet_name="summary_all")

    print("\n=== ALL DONE ===")
    print(df_all_summary.to_string(index=False))