from __future__ import annotations
 
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
 
import numpy as np
import pandas as pd
 
OXI_TYPES = ["oxidizer_type1", "oxidizer_type2"]
OXI_PCTS = ["oxidizer_pct1", "oxidizer_pct2"]
RED_TYPES = ["reducer_type1", "reducer_type2", "reducer_type3"]
RED_PCTS = ["reducer_pct1", "reducer_pct2", "reducer_pct3"]
TER_TYPES = [f"third_monomer_type{i}" for i in range(1, 8)]
TER_PCTS = [f"third_monomer_pct{i}" for i in range(1, 8)]
EML_TYPES = [f"emulsifier_type{i}" for i in range(1, 5)]
EML_PCTS = [f"emulsifier_pct{i}" for i in range(1, 5)]
PC_TYPES = [f"protective_colloid_type{i}" for i in range(1, 4)]
PC_PCTS = [f"protective_colloid_pct{i}" for i in range(1, 4)]
 
FAMILIES = {
    "oxid": (OXI_TYPES, OXI_PCTS),
    "red": (RED_TYPES, RED_PCTS),
    "ter": (TER_TYPES, TER_PCTS),
    "emul": (EML_TYPES, EML_PCTS),
    "pc": (PC_TYPES, PC_PCTS),
}
 
NUM_COLS_TO_IMPUTE = [
    "init_pressure", "init_temp", "poly_pressure", "poly_temp", "time",
    "coinitiator_pct", "dispersant_pct", "pH_buffer_pct",
]
 
MAP_DESC_COLS_ORDERED = [
    "SMILES", "MW", "HeavyAtomCount", "S_count", "O_count", "N_count",
    "HBA", "HBD", "TPSA", "RotatableBonds", "E_HOMO", "E_LUMO",
    "△G", "αiso", "V(vdw)", "f+", "f-", "f0", "HLB", "EO_number",
    "degree_of_hydrolysis", "DP",
]
 
 
def clean_headers_and_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.replace("\u3000", " ", regex=False)
        .str.strip()
        .str.replace(r"[,;，；]+$", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
    )
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    for c in [c for c in out.columns if out[c].dtype == "object"]:
        s = out[c].astype(str).str.replace("\u00A0", " ", regex=False).str.replace("\u3000", " ", regex=False).str.strip()
        out[c] = s.mask(s.isin(["nan", "None", "NA", "", "NAN", "NaN", "-", "—"]), np.nan)
    return out
 
 
def parse_numeric(x: Any) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().lower()
    if s in {"", "na", "nan", "none", "-", "—"}:
        return np.nan
    s = re.sub(r"\s*(ppm|mg/?l|mpa|bar|kpa|wt%|%|phr)\s*$", "", s)
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*[-~–—]\s*([+-]?\d+(?:\.\d+)?)\s*$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
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
    return _norm_text(x) not in {"", "none", "nan", "na", "n/a", "null", "-", "—", "无", "0"}
 
 
def arcsin_sqrt(x: pd.Series) -> pd.Series:
    return np.arcsin(np.sqrt(pd.to_numeric(x, errors="coerce").clip(0, 1)))
 
 
def normalize_mode_value_en(x: Any) -> str:
    if pd.isna(x):
        return "batch"
    s = str(x).strip().lower().replace("－", "-").replace("—", "-").replace("–", "-").replace(" ", "").replace("　", "")
    if "semi" in s or "半连续" in s:
        return "semi_continuous"
    if "conti" in s or "连续" in s:
        return "continuous"
    return "batch"
 
 
TER_NON_HCHO_WHITELIST = {
    "vinyltrimethoxysilane", "vinyl tri methoxy silane", "vinyl tri-methoxy silane", "vtms",
    "vinyltriethoxysilane", "triethoxyvinylsilane", "vtes",
    "vinyltris(2-methoxyethoxy)silane", "vtmoeo", "vinyltriacetoxysilane", "vtas",
    "n-methylacrylamide", "n-甲基丙烯酰胺", "glycidyl methacrylate", "glycidol methacrylate", "gma",
    "2,3-epoxypropyl methacrylate", "n-(3,3-diethoxybutyl)acrylamide",
    "n-(3,3-diethoxybutyl) methacrylamide", "n-(3,3-diethoxybutyl)prop-2-enamide",
    "n-acryloylglycine methyl ester", "n-丙烯酰氧基甘氨酸甲酯", "n-丙烯酰氧基甘氨酸",
}
TER_HCHO_NAME_KEYS = [
    "n-methylolacrylamide", "n-methylol acrylamide", "n-(hydroxymethyl)acrylamide",
    "n-methylolmethacrylamide", "n-(hydroxymethyl)methacrylamide", "n-(methoxymethyl)acrylamide",
    "n-(ethoxymethyl)acrylamide", "n-(isobutoxymethyl)acrylamide", "isobutoxymethylacrylamide",
    "n-(butoxymethyl)acrylamide", "n-(n-butoxymethyl)acrylamide", "butoxymethylacrylamide",
    "n-(2,2-dimethoxy-1-hydroxyethyl)-acrylamide", "n-(2,2-二甲氧基-1-羟乙基)丙烯酰胺",
    "hexamethoxymethylmelamine", "hmmm", "melamine formaldehyde",
]
TER_HCHO_SMILES_PATTERNS = [r"C\(OC\)\(OC\)", r"NC\(=O\).*OCH3", r"NC\(=O\).*OCC"]
RED_HCHO_NAME_KEYS = [
    "sodium formaldehyde sulfoxylate", "zinc formaldehyde sulfoxylate", "potassium formaldehyde sulfoxylate",
    "ammonium formaldehyde sulfoxylate", "hydroxymethanesulfinate", "rongalite", "zfs", "zfs·2h2o",
    "formaldehyde sodium bisulfite", "formaldehyde bisulfite", "hydroxymethanesulfonate",
    "sodium hydroxymethanesulfonate", "zinc bisulfite formaldehyde",
]
RED_HCHO_SMILES_PATTERNS = [r"OCH2S\(=O\)", r"OCH2SO2", r"OCH2SO3", r"SO2CH2O", r"SO3CH2O"]
RED_GRAY_NAMES = {
    "sodium hydrogen sulfite", "sodium bisulfite", "sodium metabisulfite", "sodium thiosulfate", "sodium sulfite",
    "sodium pyrosulfite", "disodium 2-hydroxy-2-sulfinatoacetate", "zinc 2-hydroxy-2-sulfinatoacetate",
    "glyoxal sodium bisulfite", "sodium acetone bisulfite", "l-ascorbic acid", "d-sodium erythorbate",
}
 
 
def ter_hcho_flag_reason(std_name: Any, smiles: Any) -> Tuple[bool, str]:
    n, smi = _norm_text(std_name), _norm_text(smiles)
    for key in TER_NON_HCHO_WHITELIST:
        if key in n:
            return False, "whitelist_non_formal_ter"
    for key in TER_HCHO_NAME_KEYS:
        if key in n:
            return True, "ter_name_hit"
    for p in TER_HCHO_SMILES_PATTERNS:
        if smi and re.search(p, smi, flags=re.I):
            return True, "ter_smiles_hit"
    return False, "none"
 
 
def red_hcho_flag_reason(std_name: Any, smiles: Any) -> Tuple[bool, str]:
    n, smi = _norm_text(std_name), _norm_text(smiles)
    for key in RED_GRAY_NAMES:
        if key in n:
            return False, "reducer_graylist_non_formal"
    for key in RED_HCHO_NAME_KEYS:
        if key in n:
            return True, "reducer_name_hit"
    for p in RED_HCHO_SMILES_PATTERNS:
        if smi and re.search(p, smi, flags=re.I):
            return True, "reducer_smiles_hit"
    return False, "none"
 
 
class FrozenFeatureBuilder:
    """V0 冻结版原始配方 -> master feature 的部署化实现。"""
    def __init__(self, raw_data: Path, descriptor_map: Path, x_master_ref: Path):
        self.raw_train = clean_headers_and_strings(pd.read_excel(raw_data))
        self.map_df = clean_headers_and_strings(pd.read_excel(descriptor_map))
        self.x_master_ref = pd.read_excel(x_master_ref)
        self.master_columns = list(self.x_master_ref.columns)
        self.master_medians = self.x_master_ref.apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
        self.alias_map = self._build_alias_map(self.map_df)
        self.pct_slot_medians = self._fit_pct_slot_medians(self.raw_train)
        self.num_col_medians = self._fit_num_col_medians(self.raw_train)
 
    def _build_alias_map(self, dic: pd.DataFrame) -> Dict[str, dict]:
        rename_map = {}
        lower_to_real = {str(c).lower(): c for c in dic.columns}
        for want, cands in {
            "raw_name": ["raw_name", "rawname", "alias", "别名", "原名", "原始名称"],
            "standard_name": ["standard_name", "std_name", "standard", "标准名", "标准名称"],
            "type": ["type", "类别", "class", "kind"],
        }.items():
            for cand in cands:
                if cand.lower() in lower_to_real:
                    rename_map[lower_to_real[cand.lower()]] = want
                    break
        dic = dic.rename(columns=rename_map)
        if "raw_name" not in dic.columns or "standard_name" not in dic.columns:
            raise ValueError("描述符映射表缺少 raw_name / standard_name。")
        amap = {}
        for _, r in dic.iterrows():
            key = _norm_text(r.get("raw_name", ""))
            if key:
                amap[key] = r.to_dict()
        return amap
 
    def material_options(self) -> Dict[str, List[str]]:
        groups = {
            "oxidizer": OXI_TYPES, "reducer": RED_TYPES, "third_monomer": TER_TYPES,
            "emulsifier": EML_TYPES, "protective_colloid": PC_TYPES,
        }
        out: Dict[str, List[str]] = {}
        for key, cols in groups.items():
            vals = []
            for c in cols:
                if c in self.raw_train.columns:
                    vals.extend([str(x).strip() for x in self.raw_train[c].dropna().tolist() if is_valid_name(x)])
            out[key] = sorted(set(vals), key=lambda x: x.lower())
        return out
 
    def material_label_map(self) -> Dict[str, str]:
        """Map raw training material names to English standardized display names."""
        labels: Dict[str, str] = {}
        for values in self.material_options().values():
            for raw_name in values:
                row = self.alias_map.get(_norm_text(raw_name))
                if row is None:
                    labels[raw_name] = raw_name
                    continue
                std = row.get("standard_name", raw_name)
                labels[raw_name] = str(std).strip() if is_valid_name(std) else raw_name
        return labels
 
    def _name_to_desc(self, name: Any) -> dict:
        out = {"std_name": np.nan, "type": np.nan, **{c: np.nan for c in MAP_DESC_COLS_ORDERED}}
        if not is_valid_name(name):
            return out
        row = self.alias_map.get(_norm_text(name))
        if row is None:
            return out
        out["std_name"] = row.get("standard_name", np.nan)
        out["type"] = row.get("type", np.nan)
        for c in MAP_DESC_COLS_ORDERED:
            val = row.get(c, np.nan)
            out[c] = (val if is_valid_name(val) else np.nan) if c == "SMILES" else parse_numeric(val)
        return out
 
    def _fit_pct_slot_medians(self, df: pd.DataFrame) -> Dict[str, float]:
        medians = {}
        for _, (type_cols, pct_cols) in FAMILIES.items():
            for tcol, pcol in zip(type_cols, pct_cols):
                if tcol not in df.columns or pcol not in df.columns:
                    medians[pcol] = 0.0; continue
                ser = df[pcol].apply(parse_numeric); has_name = df[tcol].apply(is_valid_name)
                med = ser[has_name].median(skipna=True); medians[pcol] = float(med) if pd.notna(med) else 0.0
        return medians
 
    def _fit_num_col_medians(self, df: pd.DataFrame) -> Dict[str, float]:
        out = {}
        for col in NUM_COLS_TO_IMPUTE:
            ser = df[col].apply(parse_numeric) if col in df.columns else pd.Series(dtype=float)
            med = ser.median(skipna=True) if len(ser) else np.nan
            out[col] = float(med) if pd.notna(med) else 0.0
        return out
 
    def _impute_pct_slots(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for _, (type_cols, pct_cols) in FAMILIES.items():
            for tcol, pcol in zip(type_cols, pct_cols):
                if pcol not in out: out[pcol] = np.nan
                if tcol not in out: out[tcol] = np.nan
                ser_raw = out[pcol].apply(parse_numeric); out[f"{pcol}_isna"] = ser_raw.isna().astype("int8")
                has_name = out[tcol].apply(is_valid_name); med = self.pct_slot_medians.get(pcol, 0.0)
                ser_imp = ser_raw.copy(); ser_imp.loc[has_name & ser_raw.isna()] = med; ser_imp.loc[~has_name & ser_raw.isna()] = 0.0
                out[pcol] = ser_imp.fillna(0.0)
        return out
 
    def _pool_category(self, row: pd.Series, name_cols: List[str], pct_cols: List[str], prefix: str) -> dict:
        names = [row.get(c, np.nan) for c in name_cols]
        weights = [0.0 if pd.isna(row.get(pct_cols[i], np.nan)) else float(row.get(pct_cols[i])) for i in range(len(name_cols))]
        valid = [is_valid_name(n) for n in names]
        if not any(valid):
            out = {f"{prefix}_none_used":1.0, f"{prefix}_hcho_potential_share":0.0, f"{prefix}_hcho_reasons":""}
            for c in MAP_DESC_COLS_ORDERED:
                if c != "SMILES": out[f"{prefix}_mean_{c}"] = np.nan; out[f"{prefix}_coverage_{c}"] = 0.0
            return out
        w = np.asarray(weights, dtype=float); total = w.sum()
        if total <= 0:
            w = np.zeros_like(w); idx=[i for i,v in enumerate(valid) if v]
            for i in idx: w[i]=1/len(idx)
        else: w = w/total
        descs=[self._name_to_desc(n) for n in names]; out={f"{prefix}_none_used":0.0}
        type_weights={}
        for wi,d in zip(w,descs):
            key=_norm_text(d.get("type")) or "unknown"; type_weights[key]=type_weights.get(key,0.0)+float(wi)
        for key,wi in type_weights.items(): out[f"{prefix}_type_{key}"]=wi
        hcho_share=0.0; reasons=[]
        for wi,d in zip(w,descs):
            flag,why=(ter_hcho_flag_reason(d.get("std_name"),d.get("SMILES")) if prefix=="ter" else red_hcho_flag_reason(d.get("std_name"),d.get("SMILES")) if prefix=="red" else (False,"none"))
            if flag: hcho_share += float(wi); reasons.append(why)
        out[f"{prefix}_hcho_potential_share"] = float(np.clip(hcho_share,0,1)); out[f"{prefix}_hcho_reasons"]=";".join(sorted(set(reasons)))
        for c in MAP_DESC_COLS_ORDERED:
            if c=="SMILES": continue
            vals=np.asarray([d.get(c,np.nan) if isinstance(d.get(c,np.nan),(int,float,np.integer,np.floating)) else np.nan for d in descs],dtype=float); mask=np.isfinite(vals)
            if not mask.any(): out[f"{prefix}_mean_{c}"]=np.nan; out[f"{prefix}_coverage_{c}"]=0.0
            else:
                ww=w[mask]; sw=ww.sum(); ww=(ww/sw) if sw>0 else np.ones(mask.sum())/mask.sum()
                out[f"{prefix}_mean_{c}"]=float(np.sum(vals[mask]*ww)); out[f"{prefix}_coverage_{c}"]=float(np.sum(w[mask]))
        return out
 
    def _add_mechanistic_dose_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out=df.copy()
        for prefix,(_,pct_cols) in FAMILIES.items():
            cols=[c for c in pct_cols if c in out.columns]
            num=out[cols].apply(pd.to_numeric,errors="coerce").fillna(0.0); t=f"{prefix}_total_pct"; out[t]=num.sum(axis=1); denom=out[t].replace(0,np.nan)
            for i,c in enumerate(cols,1): out[f"{prefix}_share_of_additives{i}"]=(num[c]/denom).fillna(0.0)
        return out
 
    def _post_pool_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out=df.copy(); ter_sum=out[[c for c in TER_PCTS if c in out]].apply(pd.to_numeric,errors="coerce").fillna(0).sum(axis=1); out["ter_used_any"]=(ter_sum>0).astype("int8")
        for prefix in ["oxid","red","emul","pc","ter"]:
            mask=out["ter_used_any"].eq(0) if prefix=="ter" else (pd.to_numeric(out.get(f"{prefix}_none_used",0),errors="coerce").fillna(0)>0.5 if isinstance(out.get(f"{prefix}_none_used",0),pd.Series) else pd.Series(False,index=out.index))
            for start in [f"{prefix}_mean_",f"{prefix}_coverage_",f"{prefix}_type_"]:
                cols=[c for c in out.columns if c.startswith(start)]
                if cols: out.loc[mask,cols]=0.0
            hc=f"{prefix}_hcho_potential_share"
            if hc in out: out.loc[mask,hc]=0.0
        return out
 
    def _apply_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        out=df.copy()
        for c in ["init_temp","poly_temp"]:
            if c in out: out[c]=pd.to_numeric(out[c],errors="coerce")
        for c in ["init_pressure","poly_pressure","time","coinitiator_pct","pH_buffer_pct","dispersant_pct"]:
            if c in out:
                s=pd.to_numeric(out[c],errors="coerce").clip(lower=0); out[c]=s; out[f"{c}_log1p"]=np.log1p(s)
        for prefix in ["oxid","red","ter","emul","pc"]:
            c=f"{prefix}_total_pct"
            if c in out:
                s=pd.to_numeric(out[c],errors="coerce").clip(lower=0); out[c]=s; out[f"{c}_log1p"]=np.log1p(s)
        for c in list(out.columns):
            if re.search(r"_mean_f[+\-0]$",c): out[c]=pd.to_numeric(out[c],errors="coerce").clip(0,1)
            if c.endswith("_hcho_potential_share") or ("_share_of_additives" in c and not c.endswith("_asin")): out[c]=pd.to_numeric(out[c],errors="coerce").clip(0,1)
        if "emul_share_of_additives3" in out: out["emul_share_of_additives3_asin"]=arcsin_sqrt(out["emul_share_of_additives3"])
        ser=out["initiator_feed_mode"].map(normalize_mode_value_en) if "initiator_feed_mode" in out else pd.Series(["batch"]*len(out),index=out.index)
        out["initiator_feed_continuous"]=(ser=="continuous").astype(float); out["initiator_feed_semi_continuous"]=(ser=="semi_continuous").astype(float)
        return out
 
    def validate_material_names(self, raw: Dict[str, Any]) -> List[str]:
        bad=[]
        for type_cols,_ in FAMILIES.values():
            for c in type_cols:
                n=raw.get(c)
                if is_valid_name(n) and _norm_text(n) not in self.alias_map: bad.append(str(n))
        return sorted(set(bad))
 
    def transform_raw_to_master(self, raw_candidate_df: pd.DataFrame) -> pd.DataFrame:
        df=clean_headers_and_strings(raw_candidate_df.copy())
        for cols in [OXI_TYPES,RED_TYPES,TER_TYPES,EML_TYPES,PC_TYPES,OXI_PCTS,RED_PCTS,TER_PCTS,EML_PCTS,PC_PCTS,NUM_COLS_TO_IMPUTE,["initiator_feed_mode"]]:
            for c in cols:
                if c not in df: df[c]=np.nan
        df=self._impute_pct_slots(df)
        blocks=[df.apply(lambda r:self._pool_category(r,t,p,prefix),axis=1,result_type="expand") for prefix,(t,p) in FAMILIES.items()]
        dm=pd.concat([df]+blocks,axis=1); dm=self._add_mechanistic_dose_features(dm)
        for col in NUM_COLS_TO_IMPUTE:
            ser=dm[col].apply(parse_numeric); dm[f"{col}_isna"]=ser.isna().astype("int8"); dm[col]=ser.fillna(self.num_col_medians.get(col,0.0))
        dm=self._post_pool_clean(dm); dm=self._apply_transforms(dm)
        X=pd.DataFrame(index=dm.index)
        for col in self.master_columns:
            X[col]=pd.to_numeric(dm[col],errors="coerce") if col in dm else self.master_medians.get(col,0.0)
        for col in X.columns:
            med=self.master_medians.get(col,0.0); X[col]=pd.to_numeric(X[col],errors="coerce").fillna(med if pd.notna(med) else 0.0)
        return X[self.master_columns].copy()