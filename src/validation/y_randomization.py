# -*- coding: utf-8 -*-
"""Y-randomization only, extracted from the author-supplied groupkfold.py.

This is a packaging/extraction revision, not a newly developed validation method.
The original train/test reconstruction, fixed CV folds, estimator cloning,
permutation RNG and p-value calculations are retained. No GroupKFold runs.
Default: 100 permutations, all four targets. Use --check-only to inspect inputs
and save split indices without fitting any model or running any permutation.
Use only trusted author-supplied joblib files and the original training environment.
See README.md for file layout, provenance and scope of verification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

RANDOM_SEED = 23
YR_N_SPLITS = 10
TEST_SIZE = 0.10
N_PERMUTATIONS = 100
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
YR_DIR = REPO_ROOT / "results" / "Y_randomization"
ORIGIN_DIR = YR_DIR / "Origin_data"

# Canonical modeling-workbook filenames for the four target datasets.
CONFIG = {
    "EC": {"data_file": "data/Data_S2/EC_Modeling_Data.xlsx", "model_file": "models/EC_CatBoost.joblib", "target_col": "Ethylene Content", "patent_col": "文献", "scale": "raw"},
    "SC": {"data_file": "data/Data_S2/SC_Modeling_Data.xlsx", "model_file": "models/SC_XGBoost.joblib", "target_col": "Solid Content", "patent_col": "文献", "scale": "raw"},
    "VC": {"data_file": "data/Data_S2/VC_Modeling_Data.xlsx", "model_file": "models/VC_CatBoost.joblib", "target_col": "Viscosity", "patent_col": "文献", "scale": "log1p"},
    "HCHO": {"data_file": "data/Data_S2/HCHO_Modeling_Data.xlsx", "model_file": "models/HCHO_MLP.joblib", "target_col": "Free HCHO Content", "patent_col": "文献", "scale": "log1p"},
}

def safe_r2(y_true, y_pred):
    """常量/极小验证集时 R² 无定义，返回 NaN，不人为置 0。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]):
        return np.nan
    return float(r2_score(y_true, y_pred))


def metric_pack(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "r2": safe_r2(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def load_final_estimator(model_path: Path):
    """
    EC/SC/VC 的 joblib 直接是 estimator。
    HCHO 的 joblib 为 dict，其中 reg 是最终 sklearn Pipeline。
    """
    obj = joblib.load(model_path)

    if isinstance(obj, dict):
        if "reg" not in obj:
            raise ValueError(
                f"{model_path.name} 是 dict，但找不到键 'reg'。实际 keys={list(obj.keys())}"
            )
        estimator = obj["reg"]
    else:
        estimator = obj

    # A compatibility guard for inspecting a scikit-learn 1.7 saved
    # SimpleImputer with the current scikit-learn 1.8 runtime. The project
    # requirements still pin 1.7.0 for strict environment consistency.
    if isinstance(obj, dict) and "reg" in obj:
        try:
            imputer = estimator.named_steps.get("imputer")
            if imputer is not None and hasattr(imputer, "statistics_") and not hasattr(imputer, "_fill_dtype"):
                imputer._fill_dtype = imputer.statistics_.dtype
        except (AttributeError, KeyError):
            pass

    return estimator


def get_feature_names(estimator):
    """
    优先读取 sklearn feature_names_in_；
    CatBoost 使用 feature_names_。
    """
    if hasattr(estimator, "feature_names_in_"):
        names = list(estimator.feature_names_in_)
    elif hasattr(estimator, "feature_names_"):
        names = list(estimator.feature_names_)
    else:
        raise ValueError(
            f"无法从最终模型 {type(estimator)} 中读取特征名。"
            "请不要手工猜特征列表。"
        )

    return [str(x) for x in names]


def postprocess_prediction(target_key, pred):
    pred = np.asarray(pred, dtype=float)
    if target_key == "HCHO":
        # 与原 HCHO 最终代码保持一致
        pred = np.clip(pred, 0.0, None)
    return pred


def fit_predict_frozen(estimator_template, X_train, y_train, X_valid, target_key):
    """
    clone 最终冻结 estimator：
    - 不重新 Optuna；
    - 不使用当前验证 fold 做二次超参数搜索；
    - 直接使用最终模型保存的实际参数重新 fit。
    """
    model = clone(estimator_template)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    return postprocess_prediction(target_key, pred)


def make_continuous_stratify_labels(
    y,
    max_bins=10,
    min_bin_size=2,
):
    """
    与 SC 原代码逻辑一致：
    从 max_bins 向下尝试 qcut，直至每箱样本数 >= min_bin_size。
    """
    y = np.asarray(y, dtype=float).reshape(-1)

    n_samples = len(y)
    n_unique = np.unique(y).size

    if n_samples < 4 or n_unique < 2:
        return None

    max_bins_eff = int(min(max_bins, n_unique, n_samples))
    if max_bins_eff < 2:
        return None

    for q in range(max_bins_eff, 1, -1):
        try:
            labels = pd.qcut(y, q=q, labels=False, duplicates="drop")
            labels = np.asarray(labels, dtype=int)

            uniq, cnt = np.unique(labels, return_counts=True)
            if len(uniq) >= 2 and cnt.min() >= int(min_bin_size):
                return labels
        except Exception:
            continue

    return None


def make_vc_regression_stratify_labels(y, max_bins=8):
    """
    与 VC CatBoost 最终训练代码中的 make_regression_stratify_labels 对齐。
    """
    y = np.asarray(y, dtype=float)

    if len(y) < 20:
        return None

    n_unique = len(np.unique(y))
    if n_unique < 6:
        return None

    max_bins_eff = int(min(max_bins, n_unique, max(2, len(y) // 20)))

    for n_bins in range(max_bins_eff, 1, -1):
        try:
            bins = pd.qcut(y, q=n_bins, duplicates="drop")
            vc = pd.Series(bins).value_counts()
            if len(vc) >= 2 and vc.min() >= 2:
                return bins.astype(str)
        except Exception:
            continue

    return None


def reconstruct_original_train_indices(target_key, y):
    """
    根据用户四个最终训练脚本复现原 9:1 外层 split。
    这样 Y-randomization 在原模型开发中的 train 部分执行，
    不把留出 test 混入随机化验证。
    """
    indices = np.arange(len(y), dtype=int)

    if target_key == "EC":
        tr_idx, te_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
        )

    elif target_key == "SC":
        min_bin_size = max(2, int(math.ceil(1.0 / TEST_SIZE)))
        labels = make_continuous_stratify_labels(
            y,
            max_bins=10,
            min_bin_size=min_bin_size,
        )
        if labels is None:
            raise RuntimeError("SC: 无法复现连续目标分层的 9:1 split。")

        tr_idx, te_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=labels,
        )

    elif target_key == "VC":
        labels = make_vc_regression_stratify_labels(y, max_bins=8)
        if labels is not None:
            tr_idx, te_idx = train_test_split(
                indices,
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED,
                shuffle=True,
                stratify=labels,
            )
        else:
            tr_idx, te_idx = train_test_split(
                indices,
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED,
                shuffle=True,
            )

    elif target_key == "HCHO":
        ynz = (np.asarray(y, dtype=float) > 0.0).astype(int)
        tr_idx, te_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=ynz,
        )

    else:
        raise ValueError(target_key)

    return np.asarray(tr_idx, dtype=int), np.asarray(te_idx, dtype=int)


def make_fixed_original_cv_splits(target_key, y_train):
    """
    用真实 y 生成一次原始 10-fold folds，后续所有 Y permutation 均固定这些 folds。
    这样随机化只改变 y 与 X 的对应关系，不混入 CV 划分波动。
    """
    dummy_X = np.zeros((len(y_train), 1))

    if target_key in ["EC", "VC"]:
        splitter = KFold(
            n_splits=YR_N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        splits = list(splitter.split(dummy_X))

    elif target_key == "SC":
        labels = make_continuous_stratify_labels(
            y_train,
            max_bins=10,
            min_bin_size=YR_N_SPLITS,
        )
        if labels is None:
            raise RuntimeError("SC: 无法复现连续目标分层的 10-fold CV。")

        splitter = StratifiedKFold(
            n_splits=YR_N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        splits = list(splitter.split(dummy_X, labels))

    elif target_key == "HCHO":
        ynz = (np.asarray(y_train, dtype=float) > 0.0).astype(int)
        splitter = StratifiedKFold(
            n_splits=YR_N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        splits = list(splitter.split(dummy_X, ynz))

    else:
        raise ValueError(target_key)

    return [(np.asarray(a), np.asarray(b)) for a, b in splits]


def fixed_cv_oof(estimator, X_train, y_for_fit_and_score, cv_splits, target_key):
    """
    对给定 y 做固定 folds 的 OOF。
    Y-randomization 时 y_for_fit_and_score 就是打乱后的 y。
    """
    y_local = np.asarray(y_for_fit_and_score, dtype=float)
    oof = np.full(len(y_local), np.nan, dtype=float)

    for tr_idx, va_idx in cv_splits:
        pred = fit_predict_frozen(
            estimator_template=estimator,
            X_train=X_train.iloc[tr_idx],
            y_train=y_local[tr_idx],
            X_valid=X_train.iloc[va_idx],
            target_key=target_key,
        )
        oof[va_idx] = pred

    if np.isnan(oof).any():
        raise RuntimeError(f"{target_key}: Y-randomization OOF 未覆盖全部训练样本。")

    return oof


def validate_dataset_and_model(target_key, cfg):
    """Validate explicit inputs. Patent metadata are optional for Y-randomization.

    No label imputation, label transformation, row filtering or feature engineering
    is done here. The model's stored feature names determine the input columns.
    """
    data_path = Path(cfg["data_file"])
    model_path = Path(cfg["model_file"])
    for path in (data_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path).copy().reset_index(drop=True)
    else:
        df = pd.read_excel(data_path).copy().reset_index(drop=True)
    if cfg["target_col"] not in df.columns:
        raise ValueError(f"{target_key}: missing target column {cfg['target_col']}")
    if df.columns.duplicated().any():
        raise ValueError(f"{target_key}: duplicate column names are not supported.")
    y_num = pd.to_numeric(df[cfg["target_col"]], errors="coerce")
    if not np.isfinite(y_num.to_numpy(dtype=float)).all():
        raise ValueError(f"{target_key}: invalid target values; no target imputation is allowed.")
    df[cfg["target_col"]] = y_num.astype(float)
    estimator = load_final_estimator(model_path)
    features = get_feature_names(estimator)
    # Modeling workbooks use the descriptive alias ter_mean_E_gap for the
    # same pooled energy-gap feature saved in the frozen estimators as
    # ter_mean_△G. Preserve the frozen model's expected input name and order.
    # Do not alter source workbooks or silently apply any other schema changes.
    aliases = {"ter_mean_△G": "ter_mean_E_gap"}
    for model_col, workbook_col in aliases.items():
        if model_col in features and model_col not in df.columns and workbook_col in df.columns:
            df[model_col] = df[workbook_col]
            print(f"[ALIAS] {target_key}: {workbook_col} -> {model_col} (model schema)")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"{target_key}: missing model features: {missing}")
    X = df[features].apply(pd.to_numeric, errors="coerce")
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(f"{target_key}: infinite feature values are not supported.")
    if X.isna().any().any() and target_key != "HCHO":
        raise ValueError(f"{target_key}: missing model inputs; do not silently change preprocessing.")
    if X.isna().any().any():
        print("[WARN] HCHO has missing features; its original pipeline contains a median imputer.")
    patent_col = cfg.get("patent_col", "文献")
    if patent_col in df.columns:
        groups = df[patent_col].to_numpy()
    else:
        groups = np.full(len(df), None, dtype=object)
    print(f"[CHECK] {target_key}: n={len(df)}, features={len(features)}, scale={cfg['scale']}")
    return df, X, y_num.to_numpy(dtype=float), groups, estimator, features


def run_y_randomization_one_target(target_key, cfg):
    print("\n" + "=" * 110)
    print(f"[Y-RANDOMIZATION] {target_key} | permutations={N_PERMUTATIONS}")
    print("=" * 110)

    df, X, y, groups, estimator, features = validate_dataset_and_model(target_key, cfg)

    # 复现原始 9:1，仅使用原 train 做 Y-randomization
    train_idx, test_idx = reconstruct_original_train_indices(target_key, y)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx].copy()
    df_train = df.iloc[train_idx].reset_index(drop=True)

    cv_splits = make_fixed_original_cv_splits(target_key, y_train)

    # 真实标签 OOF：作为同一 frozen-parameter protocol 下的参考
    true_oof = fixed_cv_oof(
        estimator=estimator,
        X_train=X_train,
        y_for_fit_and_score=y_train,
        cv_splits=cv_splits,
        target_key=target_key,
    )
    true_metrics = metric_pack(y_train, true_oof)

    print(
        f"[{target_key} TRUE frozen OOF] "
        f"R2={true_metrics['r2']:.4f}, "
        f"RMSE={true_metrics['rmse']:.4f}, "
        f"MAE={true_metrics['mae']:.4f}"
    )

    # 随机化
    rng = np.random.default_rng(RANDOM_SEED + 1000)
    perm_rows = []

    for i in range(1, N_PERMUTATIONS + 1):
        y_perm = rng.permutation(y_train)

        perm_oof = fixed_cv_oof(
            estimator=estimator,
            X_train=X_train,
            y_for_fit_and_score=y_perm,
            cv_splits=cv_splits,
            target_key=target_key,
        )

        m = metric_pack(y_perm, perm_oof)

        perm_rows.append({
            "target": target_key,
            "permutation": i,
            "r2": m["r2"],
            "rmse": m["rmse"],
            "mae": m["mae"],
        })

        if i == 1 or i % 10 == 0 or i == N_PERMUTATIONS:
            print(
                f"[{target_key} YR {i}/{N_PERMUTATIONS}] "
                f"R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}"
            )

    perm_df = pd.DataFrame(perm_rows)

    # 经验置换 p：
    # R² 越高越好 -> 统计 permutation >= true
    # RMSE/MAE 越低越好 -> 统计 permutation <= true
    p_r2 = (
        1 + np.sum(perm_df["r2"].to_numpy() >= true_metrics["r2"])
    ) / (N_PERMUTATIONS + 1)

    p_rmse = (
        1 + np.sum(perm_df["rmse"].to_numpy() <= true_metrics["rmse"])
    ) / (N_PERMUTATIONS + 1)

    p_mae = (
        1 + np.sum(perm_df["mae"].to_numpy() <= true_metrics["mae"])
    ) / (N_PERMUTATIONS + 1)

    summary = {
        "target": target_key,
        "target_column": cfg["target_col"],
        "target_scale": cfg["scale"],
        "n_total_samples": len(y),
        "n_original_train": len(train_idx),
        "n_original_test": len(test_idx),
        "n_features": len(features),
        "cv_n_splits": YR_N_SPLITS,
        "n_permutations": N_PERMUTATIONS,

        "true_frozen_oof_r2": true_metrics["r2"],
        "true_frozen_oof_rmse": true_metrics["rmse"],
        "true_frozen_oof_mae": true_metrics["mae"],

        "perm_r2_mean": float(np.nanmean(perm_df["r2"])),
        "perm_r2_std": float(np.nanstd(perm_df["r2"], ddof=1)),
        "perm_r2_max": float(np.nanmax(perm_df["r2"])),

        "perm_rmse_mean": float(perm_df["rmse"].mean()),
        "perm_rmse_std": float(perm_df["rmse"].std(ddof=1)),
        "perm_rmse_min": float(perm_df["rmse"].min()),

        "empirical_p_r2": float(p_r2),
        "empirical_p_rmse": float(p_rmse),
        "empirical_p_mae": float(p_mae),
    }

    # 真实 OOF 表
    true_pred_df = pd.DataFrame({
        "target": target_key,
        "row_index_in_single_target_file": train_idx,
        "patent": (df_train[cfg["patent_col"]] if cfg["patent_col"] in df_train.columns else [None] * len(df_train)),
        "y_true": y_train,
        "y_pred_true_oof": true_oof,
        "residual": y_train - true_oof,
        "abs_residual": np.abs(y_train - true_oof),
    })

    # Origin 数据
    origin_df = perm_df.copy()
    origin_df["true_r2_reference"] = true_metrics["r2"]
    origin_df["true_rmse_reference"] = true_metrics["rmse"]
    origin_df["true_mae_reference"] = true_metrics["mae"]
    origin_df.to_csv(
        ORIGIN_DIR / f"{target_key}_Y_randomization_Origin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # R² 随机化分布快速检查图
    plt.figure(figsize=(7.2, 5.6))
    plt.hist(perm_df["r2"].dropna().to_numpy(), bins=20, alpha=0.75)
    plt.axvline(true_metrics["r2"], linestyle="--", linewidth=1.5)
    plt.xlabel("OOF R² after Y-randomization")
    plt.ylabel("Count")
    plt.title(
        f"{target_key} Y-randomization\n"
        f"True R²={true_metrics['r2']:.3f}, empirical p={p_r2:.4f}"
    )
    plt.tight_layout()
    plt.savefig(YR_DIR / f"{target_key}_Yrandom_R2_distribution.png", dpi=300)
    plt.close()

    print(
        f"[{target_key} YR SUMMARY] "
        f"true_R2={true_metrics['r2']:.4f}, "
        f"perm_R2_mean={perm_df['r2'].mean():.4f}, "
        f"p_R2={p_r2:.4f}"
    )

    return (
        pd.DataFrame([summary]),
        perm_df,
        true_pred_df,
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_split_check(target, cfg, out_dir):
    df, X, y, _, estimator, features = validate_dataset_and_model(target, cfg)
    train_idx, test_idx = reconstruct_original_train_indices(target, y)
    folds = make_fixed_original_cv_splits(target, y[train_idx])
    rows = []
    fold_for_train = np.zeros(len(train_idx), dtype=int)
    for fold_id, (_, valid) in enumerate(folds, 1):
        fold_for_train[valid] = fold_id
    for local_index, data_index in enumerate(train_idx):
        rows.append({"row_index_in_single_target_file": int(data_index), "split": "train", "fixed_cv_fold": int(fold_for_train[local_index])})
    for data_index in test_idx:
        rows.append({"row_index_in_single_target_file": int(data_index), "split": "test", "fixed_cv_fold": None})
    pd.DataFrame(rows).sort_values("row_index_in_single_target_file").to_csv(
        out_dir / f"{target}_split_reconstructed_from_code.csv", index=False, encoding="utf-8-sig")
    return {"target": target, "n_samples": len(df), "n_train": len(train_idx), "n_test": len(test_idx),
            "n_features": len(features), "scale": cfg["scale"], "features": features,
            "data_file": str(cfg["data_file"]), "model_file": str(cfg["model_file"]),
            "data_sha256": _sha256(cfg["data_file"]), "model_sha256": _sha256(cfg["model_file"]),
            "split_origin": "reconstructed from supplied code; not original saved split artifacts"}


def main():
    global N_PERMUTATIONS, YR_DIR, ORIGIN_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root; relative input paths are resolved here.")
    parser.add_argument("--config", type=Path, help="Optional JSON mapping of EC/SC/VC/HCHO to data_file and model_file overrides.")
    parser.add_argument("--target", type=str.upper, choices=["ALL", "EC", "SC", "VC", "HCHO"], default=os.environ.get("VAE_TARGET", "ALL").upper())
    parser.add_argument("--permutations", type=int, default=int(os.environ.get("VAE_N_PERM", "100")))
    parser.add_argument("--output-dir", type=Path, help="New or empty output directory. Existing output is not overwritten.")
    parser.add_argument("--check-only", action="store_true", help="Load inputs, check columns, and export reconstructed splits; no fit, no permutation.")
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be at least 1.")
    N_PERMUTATIONS = args.permutations
    root = args.root.resolve()
    cfg_all = {k: dict(v) for k, v in CONFIG.items()}
    if args.config:
        overrides = json.loads(args.config.read_text(encoding="utf-8-sig"))
        for target, values in overrides.items():
            if target not in cfg_all or not isinstance(values, dict):
                raise ValueError(f"Invalid config target: {target}")
            invalid = set(values) - {"data_file", "model_file", "patent_col"}
            if invalid:
                raise ValueError(f"Only input paths and patent_col can be overridden; invalid fields: {invalid}")
            cfg_all[target].update(values)
    for cfg in cfg_all.values():
        for key in ("data_file", "model_file"):
            path = Path(cfg[key])
            cfg[key] = path if path.is_absolute() else root / path
    selected = {k: v for k, v in cfg_all.items() if args.target == "ALL" or k == args.target}
    out_dir = (args.output_dir or root / "results" / ("Y_randomization_input_check" if args.check_only else "Y_randomization")).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty. Choose a new --output-dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    YR_DIR = out_dir
    ORIGIN_DIR = out_dir / "Origin_data"
    checks = [_save_split_check(t, cfg, out_dir) for t, cfg in selected.items()]
    (out_dir / "input_manifest.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.check_only:
        print("Input/split checks complete. No model was fitted and no permutation was run.")
        return
    ORIGIN_DIR.mkdir(parents=True, exist_ok=True)
    if N_PERMUTATIONS != 100:
        print("[NOTICE] This run is not the paper's 100-permutation protocol. Do not report it as such.")
    summaries, permutation_tables, predictions = [], [], []
    for target, cfg in selected.items():
        summary, permutation, pred = run_y_randomization_one_target(target, cfg)
        summaries.append(summary); permutation_tables.append(permutation); predictions.append(pred)
        # Save completed targets immediately; model files are never rewritten.
        pd.concat(summaries, ignore_index=True).to_csv(out_dir / "Y_randomization_summary.csv", index=False, encoding="utf-8-sig")
        pd.concat(permutation_tables, ignore_index=True).to_csv(out_dir / "Y_randomization_all_permutations.csv", index=False, encoding="utf-8-sig")
        pd.concat(predictions, ignore_index=True).to_csv(out_dir / "Y_randomization_true_OOF_predictions.csv", index=False, encoding="utf-8-sig")
    print(pd.concat(summaries, ignore_index=True).to_string(index=False))
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
