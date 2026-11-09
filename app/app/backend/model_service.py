from __future__ import annotations
 
from pathlib import Path
from typing import Any, Dict, List
import json
import math
import warnings
 
import joblib
import numpy as np
import pandas as pd
 
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
CONFIG_PATH = Path(__file__).with_name("feature_config.json")
 
 
def _feature_names(model: Any) -> List[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]
    if hasattr(model, "feature_names_"):
        return [str(x) for x in model.feature_names_]
    raise RuntimeError(f"Feature names were not found in model {type(model).__name__}.")
 
 
def _patch_legacy_simple_imputer(pipeline: Any) -> None:
    """兼容 sklearn 1.7 保存、1.8 加载时 SimpleImputer 新增的 _fill_dtype 属性。"""
    try:
        imp = pipeline.named_steps.get("imputer")
        if imp is not None and hasattr(imp, "statistics_") and not hasattr(imp, "_fill_dtype"):
            imp._fill_dtype = imp.statistics_.dtype
    except Exception:
        pass
 
 
class VAEModelService:
    def __init__(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.range_map = {f["name"]: f for f in self.config["features"]}
        self.models: Dict[str, Any] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.load_warnings: List[str] = []
        self._load_models()
 
    def _safe_load(self, path: Path) -> Any:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = joblib.load(path)
            for w in caught:
                text = str(w.message)
                if "version" in text.lower() or "serialized model" in text.lower():
                    self.load_warnings.append(text)
        return obj
 
    def _load_models(self) -> None:
        ec = self._safe_load(MODEL_DIR / "EC_final_model.joblib")
        sc = self._safe_load(MODEL_DIR / "SC_final_model.joblib")
        vc = self._safe_load(MODEL_DIR / "VC_model.joblib")
        hcho_obj = self._safe_load(MODEL_DIR / "HCHO_final_model.joblib")
 
        if not isinstance(hcho_obj, dict) or "reg" not in hcho_obj:
            raise RuntimeError("Invalid HCHO_final_model.joblib structure: expected a dictionary containing 'reg'.")
        hcho = hcho_obj["reg"]
        _patch_legacy_simple_imputer(hcho)
 
        self.models = {"EC": ec, "SC": sc, "VC": vc, "HCHO": hcho}
        self.feature_names = {k: _feature_names(v) for k, v in self.models.items()}
 
        # 硬检查：三个目标 39 特征，HCHO 40 特征，且 HCHO 多 semi-continuous
        common = self.feature_names["EC"]
        for key in ("SC", "VC"):
            if self.feature_names[key] != common:
                raise RuntimeError(f"Feature names/order for {key} do not match EC.")
        if self.feature_names["HCHO"][:-1] != common or self.feature_names["HCHO"][-1] != "initiator_feed_semi_continuous":
            raise RuntimeError("HCHO feature structure does not match the expected configuration.")
 
    def model_info(self) -> Dict[str, Any]:
        return {
            "EC": {"algorithm": "CatBoost", "features": len(self.feature_names["EC"]), "output": "wt%"},
            "SC": {"algorithm": "XGBoost", "features": len(self.feature_names["SC"]), "output": "wt%"},
            "VC": {"algorithm": "CatBoost", "features": len(self.feature_names["VC"]), "output": "mPa·s", "target_scale": "log1p -> expm1"},
            "HCHO": {"algorithm": "MLP Pipeline", "features": len(self.feature_names["HCHO"]), "output": "ppm", "target_scale": "clip(log1p, >=0) -> expm1"},
            "load_warnings": self.load_warnings,
        }
 
    def _validate_and_frame(self, features: Dict[str, Any], target: str) -> pd.DataFrame:
        names = self.feature_names[target]
        missing = [name for name in names if name not in features or features[name] is None or features[name] == ""]
        if missing:
            raise ValueError("Missing model input features: " + ", ".join(missing))
 
        row = {}
        for name in names:
            try:
                value = float(features[name])
            except (TypeError, ValueError):
                raise ValueError(f"Feature {name} is not a valid number.")
            if not math.isfinite(value):
                raise ValueError(f"Feature {name} must be finite.")
            row[name] = value
 
        # 二元变量强检查
        for name in ("initiator_feed_continuous", "initiator_feed_semi_continuous"):
            if name in row and row[name] not in (0.0, 1.0):
                raise ValueError(f"{name} must be 0 or 1.")
 
        return pd.DataFrame([row], columns=names)
 
    def range_warnings(self, features: Dict[str, Any]) -> List[str]:
        messages = []
        for name, meta in self.range_map.items():
            if name not in features or features[name] in (None, ""):
                continue
            try:
                v = float(features[name])
            except Exception:
                continue
            lo, hi = meta.get("min"), meta.get("max")
            if lo is not None and hi is not None and (v < lo or v > hi):
                messages.append(
                    f"{meta['label']} ({name}) = {v:g} is outside the reference-data range [{lo:g}, {hi:g}]"
                )
        return messages
 
    def predict_master(self, X_master: pd.DataFrame) -> Dict[str, Any]:
        """Predict from one-row frozen master feature table generated from a raw formulation."""
        if len(X_master) != 1:
            raise ValueError("predict_master accepts one formulation at a time.")
        frames = {}
        flat = {}
        for target, names in self.feature_names.items():
            missing = [c for c in names if c not in X_master.columns]
            if missing:
                raise ValueError(f"Frozen master feature table is missing features required by {target}: " + ", ".join(missing))
            frames[target] = X_master[names].copy()
            flat.update({c: float(frames[target].iloc[0][c]) for c in names})
 
        ec = float(np.asarray(self.models["EC"].predict(frames["EC"])).reshape(-1)[0])
        sc = float(np.asarray(self.models["SC"].predict(frames["SC"])).reshape(-1)[0])
        vc_log = float(np.asarray(self.models["VC"].predict(frames["VC"])).reshape(-1)[0])
        vc = float(np.expm1(vc_log))
        hcho_log_raw = float(np.asarray(self.models["HCHO"].predict(frames["HCHO"])).reshape(-1)[0])
        hcho = float(np.expm1(max(0.0, hcho_log_raw)))
        return {
            "predictions": {
                "viscosity": {"value": vc, "unit": "mPa·s", "model": "CatBoost", "model_output_log1p": vc_log},
                "ethylene_content": {"value": ec, "unit": "wt%", "model": "CatBoost"},
                "solid_content": {"value": sc, "unit": "wt%", "model": "XGBoost"},
                "free_hcho": {"value": hcho, "unit": "ppm", "model": "MLP", "model_output_log1p": hcho_log_raw},
            },
            "range_warnings": self.range_warnings(flat),
        }
 
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        X_ec = self._validate_and_frame(features, "EC")
        X_sc = self._validate_and_frame(features, "SC")
        X_vc = self._validate_and_frame(features, "VC")
        X_hcho = self._validate_and_frame(features, "HCHO")
 
        ec = float(np.asarray(self.models["EC"].predict(X_ec)).reshape(-1)[0])
        sc = float(np.asarray(self.models["SC"].predict(X_sc)).reshape(-1)[0])
 
        vc_log = float(np.asarray(self.models["VC"].predict(X_vc)).reshape(-1)[0])
        vc = float(np.expm1(vc_log))
 
        hcho_log_raw = float(np.asarray(self.models["HCHO"].predict(X_hcho)).reshape(-1)[0])
        hcho_log = max(0.0, hcho_log_raw)  # 与训练代码 predict_direct_regression 的 clip_nonneg 一致
        hcho = float(np.expm1(hcho_log))
 
        return {
            "predictions": {
                "viscosity": {"value": vc, "unit": "mPa·s", "model": "CatBoost", "model_output_log1p": vc_log},
                "ethylene_content": {"value": ec, "unit": "wt%", "model": "CatBoost"},
                "solid_content": {"value": sc, "unit": "wt%", "model": "XGBoost"},
                "free_hcho": {"value": hcho, "unit": "ppm", "model": "MLP", "model_output_log1p": hcho_log_raw},
            },
            "range_warnings": self.range_warnings(features),
        }