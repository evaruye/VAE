# -*- coding: utf-8 -*-
"""
方案B V0：冻结版候选配方特征生成器 + 四代理模型预测 + 随机候选 Pareto 筛选
================================================================================

定位：
1) 这是“能否正常运行/是否存在列错位或模型预测错误”的 V0 检查版。
2) 先不加入 PDP 表格约束、商业产品窗口，也暂不使用 NSGA-II。
3) 本版使用“原始训练配方行随机抽样 + 连续变量经验范围扰动”的方式生成候选原始配方，
   然后按冻结版流程转成模型输入特征，并调用四个代理模型。
4) V0 通过后，再升级为：
   - V1：加入 PDP 推荐区间 + 商业性能窗口 + APEO-free 约束；
   - V2：替换随机搜索为 NSGA-II 混合变量优化。

请确认 BASE_DIR 目录下包含：
- VAE_final_V1_test.xlsx
- ML1_descriptors_final.xlsx
- X_reduced_final.csv
- X_VC_true.xlsx
- X_SC_true.xlsx
- X_EC_true.xlsx
- X_HCHO_true.xlsx
- model.joblib
- final_model.joblib
- EC_final_model.joblib
- direct_regression_final_model.joblib
"""

from __future__ import annotations

import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib


# =========================
# 0. Repository-relative paths
# =========================
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models"

CONFIG = {
    "raw_data": DATA_DIR / "VAE_formulation_dataset.xlsx",
    "descriptor_map": DATA_DIR / "ML1_descriptors_final.xlsx",
    "x_master_ref": DATA_DIR / "X_reduced_final.xlsx",
    "x_vc": DATA_DIR / "X_VC_true.xlsx",
    "x_sc": DATA_DIR / "X_SC_true.xlsx",
    "x_ec": DATA_DIR / "X_EC_true.xlsx",
    "x_hcho": DATA_DIR / "X_HCHO_true.xlsx",
    "model_vc": MODEL_DIR / "VC_CatBoost.joblib",
    "model_sc": MODEL_DIR / "SC_XGBoost.joblib",
    "model_ec": MODEL_DIR / "EC_CatBoost.joblib",
    "model_hcho": MODEL_DIR / "HCHO_MLP.joblib",
    "out_dir": REPO_ROOT / "results" / "feature_pipeline",
    "n_candidates": 2000,
    "random_seed": 42,
    "mutation_rate": 0.35,
    "q_low": 0.05,
    "q_high": 0.95,
    "vc_target_is_log1p": True,
    "clip_hcho_to_nonnegative": True,
}

# =========================
# 1. 常量定义：原始表槽位
# =========================
OXI_TYPES = ["oxidizer_type1", "oxidizer_type2"]
OXI_PCTS = ["oxidizer_pct1", "oxidizer_pct2"]

RED_TYPES = ["reducer_type1", "reducer_type2", "reducer_type3"]
RED_PCTS = ["reducer_pct1", "reducer_pct2", "reducer_pct3"]

TER_TYPES = [
    "third_monomer_type1", "third_monomer_type2", "third_monomer_type3",
    "third_monomer_type4", "third_monomer_type5", "third_monomer_type6",
    "third_monomer_type7"
]
TER_PCTS = [
    "third_monomer_pct1", "third_monomer_pct2", "third_monomer_pct3",
    "third_monomer_pct4", "third_monomer_pct5", "third_monomer_pct6",
    "third_monomer_pct7"
]

EML_TYPES = ["emulsifier_type1", "emulsifier_type2", "emulsifier_type3", "emulsifier_type4"]
EML_PCTS = ["emulsifier_pct1", "emulsifier_pct2", "emulsifier_pct3", "emulsifier_pct4"]

PC_TYPES = ["protective_colloid_type1", "protective_colloid_type2", "protective_colloid_type3"]
PC_PCTS = ["protective_colloid_pct1", "protective_colloid_pct2", "protective_colloid_pct3"]

FAMILIES = {
    "oxid": (OXI_TYPES, OXI_PCTS),
    "red": (RED_TYPES, RED_PCTS),
    "ter": (TER_TYPES, TER_PCTS),
    "emul": (EML_TYPES, EML_PCTS),
    "pc": (PC_TYPES, PC_PCTS),
}

TARGET_COLS_RAW = ["solids_content", "viscosity", "ethylene_content", "residual_hcho"]

NUM_COLS_TO_IMPUTE = [
    "init_pressure", "init_temp", "poly_pressure", "poly_temp", "time",
    "coinitiator_pct", "dispersant_pct", "pH_buffer_pct"
]

MAP_ALIAS = "raw_name"
MAP_STDNAME = "standard_name"
MAP_TYPE = "type"

MAP_DESC_COLS_ORDERED = [
    "SMILES", "MW", "HeavyAtomCount", "S_count", "O_count", "N_count",
    "HBA", "HBD", "TPSA", "RotatableBonds", "E_HOMO", "E_LUMO",
    "△G", "αiso", "V(vdw)", "f+", "f-", "f0", "HLB",
    "EO_number", "degree_of_hydrolysis", "DP"
]

# 训练时保留在母表、但模型训练时剔除的列
DROP_UNUSED_FOR_ALL_MODELS = [
    "oxidizer_pct2_isna",
    "time_isna",
    "pH_buffer_pct_isna",
]

# VC/SC/EC 不使用 semi_continuous；HCHO 使用
DROP_FOR_VC_SC_EC = [
    "initiator_feed_semi_continuous",
]


# =========================
# 2. 基础工具函数
# =========================
def clean_headers_and_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = (
        out.columns.astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.replace("\u3000", " ", regex=False)
        .str.strip()
        .str.replace(r"[,;，；]+$", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
    )
    out.columns = cols
    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    obj_cols = [c for c in out.columns if out[c].dtype == "object"]
    for c in obj_cols:
        s = (
            out[c].astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.replace("\u3000", " ", regex=False)
            .str.strip()
        )
        out[c] = s.replace({
            "nan": np.nan,
            "None": np.nan,
            "NA": np.nan,
            "": np.nan,
            "NAN": np.nan,
            "NaN": np.nan,
            "-": np.nan,
            "—": np.nan,
        })
    return out


def parse_numeric(x: Any) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    s = str(x).strip().lower()
    if s in {"", "na", "nan", "none", "-", "—"}:
        return np.nan

    # 去掉常见单位；不做单位换算
    s = re.sub(r"\s*(ppm|mg/?l|mpa|bar|kpa|wt%|%|phr)\s*$", "", s)

    # 区间取均值，例如 10-20、10~20
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*[-~–—]\s*([+-]?\d+(?:\.\d+)?)\s*$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0

    # 不等号取边界值
    m = re.match(r"^\s*(?:<=|≤|>=|≥|<|>)\s*([+-]?\d+(?:\.\d+)?)\s*$", s)
    if m:
        return float(m.group(1))

    m = re.match(r"^[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", s, flags=re.I)
    if m:
        return float(s)

    m = re.search(r"([+-]?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan


def _norm_text(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).strip().lower()


def is_valid_name(x: Any) -> bool:
    s = _norm_text(x)
    return s not in {"", "none", "nan", "na", "n/a", "null", "-", "—", "无", "0"}


def arcsin_sqrt(x: pd.Series) -> pd.Series:
    return np.arcsin(np.sqrt(pd.to_numeric(x, errors="coerce").clip(0, 1)))


def normalize_mode_value_en(x: Any) -> str:
    if pd.isna(x):
        return "batch"
    s = str(x).strip().lower()
    s = s.replace("－", "-").replace("—", "-").replace("–", "-").replace(" ", "").replace("　", "")
    if "semi" in s or "半连续" in s:
        return "semi_continuous"
    if "conti" in s or "连续" in s:
        return "continuous"
    if "batch" in s or "间歇" in s or "批" in s:
        return "batch"
    return "batch"


# =========================
# 3. HCHO 潜力规则
# =========================
TER_NON_HCHO_WHITELIST = {
    "vinyltrimethoxysilane", "vinyl tri methoxy silane", "vinyl tri-methoxy silane", "vtms",
    "vinyltriethoxysilane", "triethoxyvinylsilane", "vtes",
    "vinyltris(2-methoxyethoxy)silane", "vtmoeo",
    "vinyltriacetoxysilane", "vtas",
    "n-methylacrylamide", "n-甲基丙烯酰胺",
    "glycidyl methacrylate", "glycidol methacrylate", "gma", "2,3-epoxypropyl methacrylate",
    "n-(3,3-diethoxybutyl)acrylamide", "n-(3,3-diethoxybutyl) methacrylamide",
    "n-(3,3-diethoxybutyl)prop-2-enamide",
    "n-acryloylglycine methyl ester", "n-丙烯酰氧基甘氨酸甲酯", "n-丙烯酰氧基甘氨酸"
}

TER_HCHO_NAME_KEYS = [
    "n-methylolacrylamide", "n-methylol acrylamide", "n-(hydroxymethyl)acrylamide",
    "n-methylolmethacrylamide", "n-(hydroxymethyl)methacrylamide",
    "n-(methoxymethyl)acrylamide", "n-(ethoxymethyl)acrylamide",
    "n-(isobutoxymethyl)acrylamide", "isobutoxymethylacrylamide",
    "n-(butoxymethyl)acrylamide", "n-(n-butoxymethyl)acrylamide", "butoxymethylacrylamide",
    "n-(2,2-dimethoxy-1-hydroxyethyl)-acrylamide", "n-(2,2-二甲氧基-1-羟乙基)丙烯酰胺",
    "hexamethoxymethylmelamine", "hmmm", "melamine formaldehyde"
]

TER_HCHO_SMILES_PATTERNS = [
    r"C\(OC\)\(OC\)",
    r"NC\(=O\).*OCH3",
    r"NC\(=O\).*OCC"
]

RED_HCHO_NAME_KEYS = [
    "sodium formaldehyde sulfoxylate", "zinc formaldehyde sulfoxylate",
    "potassium formaldehyde sulfoxylate", "ammonium formaldehyde sulfoxylate",
    "hydroxymethanesulfinate", "rongalite", "zfs", "zfs·2h2o",
    "formaldehyde sodium bisulfite", "formaldehyde bisulfite",
    "hydroxymethanesulfonate", "sodium hydroxymethanesulfonate",
    "zinc bisulfite formaldehyde"
]

RED_HCHO_SMILES_PATTERNS = [
    r"OCH2S\(=O\)",
    r"OCH2SO2",
    r"OCH2SO3",
    r"SO2CH2O",
    r"SO3CH2O"
]

RED_GRAY_NAMES = {
    "sodium hydrogen sulfite", "sodium bisulfite", "sodium metabisulfite",
    "sodium thiosulfate", "sodium sulfite", "sodium pyrosulfite",
    "disodium 2-hydroxy-2-sulfinatoacetate", "zinc 2-hydroxy-2-sulfinatoacetate",
    "glyoxal sodium bisulfite", "sodium acetone bisulfite",
    "l-ascorbic acid", "d-sodium erythorbate"
}


def ter_hcho_flag_reason(std_name: Any, smiles: Any) -> Tuple[bool, str]:
    n, smi = _norm_text(std_name), _norm_text(smiles)

    for key in TER_NON_HCHO_WHITELIST:
        if key in n:
            return False, "whitelist_non_formal_ter"

    for key in TER_HCHO_NAME_KEYS:
        if key in n:
            return True, "ter_name_hit"

    if smi:
        for p in TER_HCHO_SMILES_PATTERNS:
            if re.search(p, smi, flags=re.I):
                return True, "ter_smiles_hit"

    return False, "none"


def red_hcho_flag_reason(std_name: Any, smiles: Any) -> Tuple[bool, str]:
    n, smi = _norm_text(std_name), _norm_text(smiles)

    for key in RED_GRAY_NAMES:
        if key in n:
            return False, "reducer_graylist_non_formal"

    for key in RED_HCHO_NAME_KEYS:
        if key in n:
            if "sulfoxylate" in key or "sulfinate" in key or "rongalite" in key:
                return True, "reducer_sulfoxylate_family"
            return True, "reducer_bisulfite_formal_adduct"

    if smi:
        for p in RED_HCHO_SMILES_PATTERNS:
            if re.search(p, smi, flags=re.I):
                return True, "reducer_smiles_hit"

    return False, "none"


# =========================
# 4. 冻结版特征生成器
# =========================
class FrozenFeatureBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.raw_train = clean_headers_and_strings(pd.read_excel(config["raw_data"]))
        self.map_df = clean_headers_and_strings(pd.read_excel(config["descriptor_map"]))

        if not config["x_master_ref"].exists():
            raise FileNotFoundError(f"Cannot find frozen master reference: {config['x_master_ref']}")

        self.x_master_ref = pd.read_excel(config["x_master_ref"])
        self.master_columns = list(self.x_master_ref.columns)
        self.master_medians = self.x_master_ref.apply(pd.to_numeric, errors="coerce").median(numeric_only=True)

        self.alias_map = self._build_alias_map(self.map_df)
        self.pct_slot_medians = self._fit_pct_slot_medians(self.raw_train)
        self.num_col_medians = self._fit_num_col_medians(self.raw_train)

        # 目标特异性列，从用户给出的最终训练表中读取，避免手动列顺序错误
        self.feature_cols = self._load_target_feature_columns(config)

    def _build_alias_map(self, dic: pd.DataFrame) -> Dict[str, dict]:
        rename_map = {}
        lower_to_real = {str(c).lower(): c for c in dic.columns}

        for want, cands in {
            MAP_ALIAS: ["raw_name", "rawname", "alias", "别名", "原名", "原始名称"],
            MAP_STDNAME: ["standard_name", "std_name", "standard", "标准名", "标准名称"],
            MAP_TYPE: ["type", "类别", "class", "kind"],
        }.items():
            for cand in cands:
                if cand.lower() in lower_to_real:
                    rename_map[lower_to_real[cand.lower()]] = want
                    break

        dic = dic.rename(columns=rename_map)

        if MAP_ALIAS not in dic.columns or MAP_STDNAME not in dic.columns:
            raise ValueError("映射表缺少 raw_name / standard_name。")

        amap = {}
        for _, r in dic.iterrows():
            key = _norm_text(r.get(MAP_ALIAS, ""))
            if key:
                amap[key] = r.to_dict()

        return amap

    def _name_to_desc(self, name: Any) -> dict:
        out = {"std_name": np.nan, "type": np.nan}
        for c in MAP_DESC_COLS_ORDERED:
            out[c] = np.nan

        if not is_valid_name(name):
            return out

        row = self.alias_map.get(_norm_text(name), None)
        if row is None:
            return out

        out["std_name"] = row.get(MAP_STDNAME, np.nan)
        out["type"] = row.get(MAP_TYPE, np.nan)

        for c in MAP_DESC_COLS_ORDERED:
            val = row.get(c, np.nan)
            if c == "SMILES":
                out[c] = val if is_valid_name(val) else np.nan
            else:
                try:
                    out[c] = parse_numeric(val)
                except Exception:
                    out[c] = np.nan

        return out

    def _fit_pct_slot_medians(self, df: pd.DataFrame) -> Dict[str, float]:
        medians = {}

        for _, (type_cols, pct_cols) in FAMILIES.items():
            for tcol, pcol in zip(type_cols, pct_cols):
                if tcol not in df.columns or pcol not in df.columns:
                    medians[pcol] = 0.0
                    continue

                ser = df[pcol].apply(parse_numeric)
                has_name = df[tcol].apply(is_valid_name)
                med = ser[has_name].median(skipna=True)
                medians[pcol] = float(med) if pd.notna(med) else 0.0

        return medians

    def _fit_num_col_medians(self, df: pd.DataFrame) -> Dict[str, float]:
        medians = {}

        for col in NUM_COLS_TO_IMPUTE:
            if col in df.columns:
                ser = df[col].apply(parse_numeric)
                med = ser.median(skipna=True)
                medians[col] = float(med) if pd.notna(med) else 0.0
            else:
                medians[col] = 0.0

        return medians

    def _load_target_feature_columns(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        target_files = {
            "VC": (config["x_vc"], "Viscosity"),
            "SC": (config["x_sc"], "Solid Content"),
            "EC": (config["x_ec"], "Ethylene Content"),
            "HCHO": (config["x_hcho"], "Free HCHO Content"),
        }

        cols = {}

        for key, (path, target_col) in target_files.items():
            if not path.exists():
                raise FileNotFoundError(f"找不到 {key} 训练表: {path}")

            df = pd.read_excel(path)

            if target_col in df.columns:
                c = [x for x in df.columns if x != target_col]
            else:
                # 容错：若目标列名称不同，则先用 master columns 裁剪
                c = [x for x in df.columns if x in self.master_columns]

            # 训练时明确不使用的缺失指示列
            c = [x for x in c if x not in DROP_UNUSED_FOR_ALL_MODELS]

            # VC/SC/EC 不使用 semi_continuous
            if key in {"VC", "SC", "EC"}:
                c = [x for x in c if x not in DROP_FOR_VC_SC_EC]

            cols[key] = c

        return cols

    def _impute_pct_slots(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for _, (type_cols, pct_cols) in FAMILIES.items():
            for tcol, pcol in zip(type_cols, pct_cols):
                if pcol not in out.columns:
                    out[pcol] = np.nan
                if tcol not in out.columns:
                    out[tcol] = np.nan

                ser_raw = out[pcol].apply(parse_numeric)
                out[f"{pcol}_isna"] = ser_raw.isna().astype("int8")

                has_name = out[tcol].apply(is_valid_name)
                med = self.pct_slot_medians.get(pcol, 0.0)

                ser_imp = ser_raw.copy()
                ser_imp.loc[has_name & ser_raw.isna()] = med
                ser_imp.loc[~has_name & ser_raw.isna()] = 0.0
                out[pcol] = ser_imp.fillna(0.0)

        return out

    def _pool_category(self, row: pd.Series, name_cols: List[str], pct_cols: List[str], prefix: str) -> dict:
        names, weights = [], []

        for i, col in enumerate(name_cols):
            nm = row.get(col, np.nan)
            wt = row.get(pct_cols[i], np.nan) if pct_cols and i < len(pct_cols) else np.nan
            names.append(nm)
            weights.append(0.0 if pd.isna(wt) else float(wt))

        valid_name_flags = [is_valid_name(nm) for nm in names]

        if not any(valid_name_flags):
            out = {
                f"{prefix}_none_used": 1.0,
                f"{prefix}_hcho_potential_share": 0.0,
                f"{prefix}_hcho_reasons": "",
            }
            for c in MAP_DESC_COLS_ORDERED:
                if c != "SMILES":
                    out[f"{prefix}_mean_{c}"] = np.nan
                    out[f"{prefix}_coverage_{c}"] = 0.0
            return out

        w = np.array([0.0 if (v is None or np.isnan(v)) else float(v) for v in weights], dtype=float)
        total_w = w.sum()

        if total_w <= 0:
            w = np.zeros_like(w)
            idx = [i for i, flag in enumerate(valid_name_flags) if flag]
            for i in idx:
                w[i] = 1.0 / len(idx)
        else:
            w = w / total_w

        descs = [self._name_to_desc(nm) for nm in names]

        out = {f"{prefix}_none_used": 0.0}

        # type 占比
        type_weights: Dict[str, float] = {}
        for wi, d in zip(w, descs):
            t = d.get("type", np.nan)
            key = _norm_text(t) if pd.notna(t) else "unknown"
            if not key:
                key = "unknown"
            type_weights[key] = type_weights.get(key, 0.0) + float(wi)

        for key, wi in type_weights.items():
            out[f"{prefix}_type_{key}"] = wi

        # HCHO 潜力
        hcho_share, reasons = 0.0, []
        for wi, d in zip(w, descs):
            stdn = d.get("std_name", np.nan)
            smi = d.get("SMILES", np.nan)

            if prefix == "ter":
                flag, why = ter_hcho_flag_reason(stdn, smi)
            elif prefix == "red":
                flag, why = red_hcho_flag_reason(stdn, smi)
            else:
                flag, why = False, "none"

            if flag:
                hcho_share += float(wi)
                if why != "none":
                    reasons.append(why)

        out[f"{prefix}_hcho_potential_share"] = float(min(1.0, max(0.0, hcho_share)))
        out[f"{prefix}_hcho_reasons"] = ";".join(sorted(set(reasons)))

        # 数值描述符加权均值 + coverage
        for c in MAP_DESC_COLS_ORDERED:
            if c == "SMILES":
                continue

            vals, mask = [], []
            for d in descs:
                v = d.get(c, np.nan)
                if isinstance(v, (int, float, np.integer, np.floating)) and not np.isnan(v):
                    vals.append(float(v))
                    mask.append(True)
                else:
                    vals.append(np.nan)
                    mask.append(False)

            vals = np.array(vals, dtype=float)
            mask = np.array(mask, dtype=bool)

            if mask.sum() == 0:
                out[f"{prefix}_mean_{c}"] = np.nan
                out[f"{prefix}_coverage_{c}"] = 0.0
            else:
                ww = w[mask] / w[mask].sum()
                out[f"{prefix}_mean_{c}"] = float(np.sum(vals[mask] * ww))
                out[f"{prefix}_coverage_{c}"] = float(np.sum(w[mask]))

        return out

    def _add_mechanistic_dose_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for prefix, (_, pct_cols) in FAMILIES.items():
            cols = [c for c in pct_cols if c in out.columns]
            if not cols:
                continue

            num = out[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            tcol = f"{prefix}_total_pct"
            out[tcol] = num.sum(axis=1)
            denom = out[tcol].replace(0, np.nan)

            for i, c in enumerate(cols, start=1):
                scol = f"{prefix}_share_of_additives{i}"
                out[scol] = (num[c] / denom).fillna(0.0)

        return out

    def _post_pool_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # ter_used_any
        ter_pct_cols = [c for c in TER_PCTS if c in out.columns]
        if ter_pct_cols:
            ter_sum = out[ter_pct_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
            out["ter_used_any"] = (ter_sum > 0).astype("int8")
        else:
            out["ter_used_any"] = 0

        # 未使用原料族时，mean_/coverage_/type_/hcho_share 置 0
        for prefix in ["oxid", "red", "emul", "pc", "ter"]:
            if prefix == "ter":
                mask_unused = out["ter_used_any"].eq(0)
            else:
                none_col = f"{prefix}_none_used"
                if none_col in out.columns:
                    mask_unused = pd.to_numeric(out[none_col], errors="coerce").fillna(0) > 0.5
                else:
                    mask_unused = pd.Series(False, index=out.index)

            mean_cols = [c for c in out.columns if c.startswith(f"{prefix}_mean_")]
            cov_cols = [c for c in out.columns if c.startswith(f"{prefix}_coverage_")]
            type_cols = [c for c in out.columns if c.startswith(f"{prefix}_type_")]

            for cols in (mean_cols, cov_cols, type_cols):
                if cols:
                    out.loc[mask_unused, cols] = 0.0

            hcho_col = f"{prefix}_hcho_potential_share"
            if hcho_col in out.columns:
                out.loc[mask_unused, hcho_col] = 0.0

        return out

    def _apply_v1_transforms_needed_for_master(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # 温度保持原始值
        for c in ["init_temp", "poly_temp"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        # 正值工艺变量 log1p
        for c in ["init_pressure", "poly_pressure", "time"]:
            if c in out.columns:
                s = pd.to_numeric(out[c], errors="coerce").clip(lower=0)
                out[c] = s
                out[f"{c}_log1p"] = np.log1p(s)

        # 小剂量助剂 log1p
        for c in ["coinitiator_pct", "pH_buffer_pct", "dispersant_pct"]:
            if c in out.columns:
                s = pd.to_numeric(out[c], errors="coerce").clip(lower=0)
                out[c] = s
                out[f"{c}_log1p"] = np.log1p(s)

        # total_pct log1p
        for prefix in ["oxid", "red", "ter", "emul", "pc"]:
            c = f"{prefix}_total_pct"
            if c in out.columns:
                s = pd.to_numeric(out[c], errors="coerce").clip(lower=0)
                out[c] = s
                out[f"{c}_log1p"] = np.log1p(s)

        # Fukui / HCHO / share clip
        for c in list(out.columns):
            if re.search(r"_mean_f[+\-0]$", c):
                out[c] = pd.to_numeric(out[c], errors="coerce").clip(0, 1)
            if c.endswith("_hcho_potential_share"):
                out[c] = pd.to_numeric(out[c], errors="coerce").clip(0, 1)
            if "_share_of_additives" in c and not c.endswith("_asin"):
                out[c] = pd.to_numeric(out[c], errors="coerce").clip(0, 1)

        # 只生成最终需要的 asin 列
        if "emul_share_of_additives3" in out.columns:
            out["emul_share_of_additives3_asin"] = arcsin_sqrt(out["emul_share_of_additives3"])

        # initiator_feed_mode one-hot，以 batch 为基准
        if "initiator_feed_mode" in out.columns:
            ser = out["initiator_feed_mode"].map(normalize_mode_value_en)
        else:
            ser = pd.Series(["batch"] * len(out), index=out.index)

        out["initiator_feed_continuous"] = (ser == "continuous").astype(float)
        out["initiator_feed_semi_continuous"] = (ser == "semi_continuous").astype(float)

        return out

    def transform_raw_to_master(self, raw_candidate_df: pd.DataFrame) -> pd.DataFrame:
        """
        原始候选配方表 → master feature table。
        """
        df = clean_headers_and_strings(raw_candidate_df.copy())

        # 保证必要原始列存在
        needed_groups = [
            OXI_TYPES, RED_TYPES, TER_TYPES, EML_TYPES, PC_TYPES,
            OXI_PCTS, RED_PCTS, TER_PCTS, EML_PCTS, PC_PCTS,
            NUM_COLS_TO_IMPUTE, ["initiator_feed_mode"]
        ]
        for cols in needed_groups:
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan

        # y 列不参与特征
        df = df.drop(columns=[c for c in TARGET_COLS_RAW if c in df.columns], errors="ignore")

        # pct 槽位缺失记录 + 中位数填补
        df = self._impute_pct_slots(df)

        # 五大类池化
        blocks = []
        for prefix, (type_cols, pct_cols) in FAMILIES.items():
            block = df.apply(
                lambda r: self._pool_category(r, type_cols, pct_cols, prefix),
                axis=1,
                result_type="expand"
            )
            blocks.append(block)

        df_model = pd.concat([df] + blocks, axis=1)

        # 机理化剂量
        df_model = self._add_mechanistic_dose_features(df_model)

        # 工艺数值列缺失掩码 + 训练集中位数填补
        for col in NUM_COLS_TO_IMPUTE:
            if col in df_model.columns:
                ser = df_model[col].apply(parse_numeric)
                df_model[f"{col}_isna"] = ser.isna().astype("int8")
                med = self.num_col_medians.get(col, 0.0)
                df_model[col] = ser.fillna(med)

        # clean_slim 关键规则
        df_model = self._post_pool_clean(df_model)

        # V1 标准化/派生中最终 master 需要的部分
        df_model = self._apply_v1_transforms_needed_for_master(df_model)

        # 对齐 master columns
        X = pd.DataFrame(index=df_model.index)
        for col in self.master_columns:
            if col in df_model.columns:
                X[col] = pd.to_numeric(df_model[col], errors="coerce")
            else:
                med = self.master_medians.get(col, 0.0)
                X[col] = med if pd.notna(med) else 0.0

        # 最终缺失兜底
        for col in X.columns:
            med = self.master_medians.get(col, 0.0)
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(med if pd.notna(med) else 0.0)

        return X[self.master_columns].copy()


# =========================
# 5. 模型加载和预测
# =========================
def patch_sklearn_compatibility(obj: Any) -> Any:
    """
    兼容 sklearn 1.7 保存、1.8 加载时 SimpleImputer 可能缺少 _fill_dtype 的问题。
    """
    def patch_one(est):
        if est.__class__.__name__ == "SimpleImputer":
            if not hasattr(est, "_fill_dtype") and hasattr(est, "_fit_dtype"):
                est._fill_dtype = est._fit_dtype

    if hasattr(obj, "steps"):
        for _, step in obj.steps:
            patch_one(step)
    elif hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            patch_one(step)

    return obj


def load_models(config: Dict[str, Any]) -> Dict[str, Any]:
    models = {}

    model_paths = {
        "VC": config["model_vc"],
        "SC": config["model_sc"],
        "EC": config["model_ec"],
        "HCHO": config["model_hcho"],
    }

    for key, path in model_paths.items():
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"找不到 {key} 模型文件: {path}")

        obj = joblib.load(path)

        # HCHO 是 dict，内部 reg 是 Pipeline
        if isinstance(obj, dict):
            if "reg" in obj:
                obj = obj["reg"]
            elif "model" in obj:
                obj = obj["model"]
            else:
                raise ValueError(f"{key} 模型是 dict，但没有 reg/model 键。keys={list(obj.keys())}")

        obj = patch_sklearn_compatibility(obj)
        models[key] = obj

    return models


def predict_four_targets(
    X_master: pd.DataFrame,
    builder: FrozenFeatureBuilder,
    models: Dict[str, Any],
    config: Dict[str, Any]
) -> pd.DataFrame:
    preds = pd.DataFrame(index=X_master.index)

    for target in ["VC", "SC", "EC", "HCHO"]:
        cols = builder.feature_cols[target]
        missing = [c for c in cols if c not in X_master.columns]
        if missing:
            raise ValueError(f"{target} 输入缺少列: {missing}")

        X_in = X_master[cols].copy()
        y_pred = models[target].predict(X_in)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

        if target == "VC":
            preds["VC_pred_model_scale"] = y_pred
            if config.get("vc_target_is_log1p", True):
                preds["VC_pred_raw"] = np.expm1(y_pred)
            else:
                preds["VC_pred_raw"] = y_pred

        elif target == "HCHO":
            preds["HCHO_pred_raw"] = y_pred
            if config.get("clip_hcho_to_nonnegative", True):
                preds["HCHO_pred_for_optim"] = np.maximum(y_pred, 0.0)
            else:
                preds["HCHO_pred_for_optim"] = y_pred

        else:
            preds[f"{target}_pred"] = y_pred

    return preds


# =========================
# 6. V0 随机候选生成
# =========================
def empirical_quantile_ranges(raw_train: pd.DataFrame, q_low: float, q_high: float) -> Dict[str, Tuple[float, float]]:
    numeric_cols = (
        NUM_COLS_TO_IMPUTE
        + OXI_PCTS + RED_PCTS + TER_PCTS + EML_PCTS + PC_PCTS
    )

    ranges = {}

    for c in numeric_cols:
        if c not in raw_train.columns:
            continue

        s = raw_train[c].apply(parse_numeric)

        if s.notna().sum() < 3:
            continue

        lo = float(s.quantile(q_low))
        hi = float(s.quantile(q_high))

        if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
            ranges[c] = (lo, hi)

    return ranges


def generate_random_raw_candidates(
    raw_train: pd.DataFrame,
    n_candidates: int,
    seed: int,
    mutation_rate: float,
    q_low: float,
    q_high: float
) -> pd.DataFrame:
    """
    V0：从真实训练配方行中抽样作为模板，对连续变量进行经验范围扰动。
    化学品种类暂时保持来自真实配方模板，因此不会生成完全脱离原始库的原料名称。
    """
    rng = np.random.default_rng(seed)
    raw = clean_headers_and_strings(raw_train.copy())

    idx = rng.integers(0, len(raw), size=n_candidates)
    cand = raw.iloc[idx].reset_index(drop=True).copy()
    cand.insert(0, "candidate_id", np.arange(1, n_candidates + 1))

    ranges = empirical_quantile_ranges(raw, q_low=q_low, q_high=q_high)
    numeric_mut_cols = list(ranges.keys())

    for c in numeric_mut_cols:
        mask = rng.random(n_candidates) < mutation_rate
        lo, hi = ranges[c]
        sampled = rng.uniform(lo, hi, size=n_candidates)
        cand.loc[mask, c] = sampled[mask]

    # 若某槽位没有原料名称，对应 pct 强制为 0，避免出现“无原料但有用量”
    for _, (type_cols, pct_cols) in FAMILIES.items():
        for tcol, pcol in zip(type_cols, pct_cols):
            if tcol in cand.columns and pcol in cand.columns:
                no_name = ~cand[tcol].apply(is_valid_name)
                cand.loc[no_name, pcol] = 0.0

    # 加料方式：从训练集已出现类别中随机重抽一部分
    if "initiator_feed_mode" in cand.columns and "initiator_feed_mode" in raw.columns:
        modes = raw["initiator_feed_mode"].dropna().unique().tolist()
        if modes:
            mask = rng.random(n_candidates) < mutation_rate
            sampled_modes = rng.choice(modes, size=n_candidates)
            cand.loc[mask, "initiator_feed_mode"] = sampled_modes[mask]

    # 删除目标列，保留 candidate_id
    cand = cand.drop(columns=[c for c in TARGET_COLS_RAW if c in cand.columns], errors="ignore")

    return cand


# =========================
# 7. Pareto 和 V0 排序
# =========================
def pareto_mask_minimize(F: np.ndarray) -> np.ndarray:
    """
    返回非支配解 mask。F 越小越好。
    简单 O(n^2) 实现，V0 候选数不大时足够。
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue

        # j 支配 i：所有目标 <= 且至少一个 <
        less_equal = np.all(F <= F[i], axis=1)
        strictly_less = np.any(F < F[i], axis=1)
        dom_by_any = less_equal & strictly_less
        dom_by_any[i] = False

        if np.any(dom_by_any):
            dominated[i] = True

    return ~dominated


def build_v0_objectives(preds: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    V0 目标定义：
    - VC：接近训练集 VC 中位数，使用模型尺度；
    - SC：越高越好 → 最小化 -SC；
    - EC：接近训练集 EC 中位数；
    - HCHO：越低越好。

    后续 V1 会替换成 PDP/商业窗口 desirability 和硬约束。
    """
    vc_df = pd.read_excel(CONFIG["x_vc"])
    ec_df = pd.read_excel(CONFIG["x_ec"])

    vc_center = float(pd.to_numeric(vc_df["Viscosity"], errors="coerce").median())
    ec_center = float(pd.to_numeric(ec_df["Ethylene Content"], errors="coerce").median())

    obj = pd.DataFrame(index=preds.index)
    obj["obj_vc_abs_to_train_median"] = np.abs(preds["VC_pred_model_scale"] - vc_center)
    obj["obj_sc_negative"] = -preds["SC_pred"]
    obj["obj_ec_abs_to_train_median"] = np.abs(preds["EC_pred"] - ec_center)
    obj["obj_hcho"] = preds["HCHO_pred_for_optim"]

    F = obj.values.astype(float)
    return obj, F


# =========================
# 8. 文件检查
# =========================
def check_required_files(config: Dict[str, Any]) -> None:
    required_keys = [
        "raw_data",
        "descriptor_map",
        "x_master_ref",
        "x_vc",
        "x_sc",
        "x_ec",
        "x_hcho",
        "model_vc",
        "model_sc",
        "model_ec",
        "model_hcho",
    ]

    missing = []
    for key in required_keys:
        p = Path(config[key])
        if not p.exists():
            missing.append((key, str(p)))

    if missing:
        msg = "\n".join([f" - {key}: {path}" for key, path in missing])
        raise FileNotFoundError(
            "以下必要文件不存在，请检查文件名或 BASE_DIR：\n" + msg
        )


# =========================
# 9. 主程序
# =========================