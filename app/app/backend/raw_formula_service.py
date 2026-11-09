from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import math

import numpy as np
import pandas as pd

from .feature_builder import (
    FrozenFeatureBuilder,
    FAMILIES,
    OXI_TYPES, OXI_PCTS, RED_TYPES, RED_PCTS, TER_TYPES, TER_PCTS,
    normalize_mode_value_en,
    parse_numeric,
    is_valid_name,
    clean_headers_and_strings,
)

BASE_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = BASE_DIR / "feature_assets"


class RawFormulaService:
    """Raw formulation -> frozen feature engineering -> four-model prediction.

    V3.4 uses property-driven input for emulsifiers and protective colloids:
    - emulsifier slots: dosage + EO number
    - protective-colloid slots: dosage + degree of hydrolysis + DP
    Commercial/product names are not required for these two families.
    """

    def __init__(self, model_service):
        self.model_service = model_service
        self.paths = {
            "raw_data": ASSET_DIR / "VAE_formulation_dataset.xlsx",
            "descriptor_map": ASSET_DIR / "ML1_descriptors_final.xlsx",
            "x_master_ref": ASSET_DIR / "X_reduced_final_features_only.xlsx",
        }
        self.builder = None
        self.init_error = None
        self.raw_train = None
        self.map_df = None

        # The raw table and descriptor dictionary are useful for the UI even if
        # the binary model bundle or frozen master reference is not present.
        if self.paths["raw_data"].exists():
            try:
                self.raw_train = clean_headers_and_strings(pd.read_excel(self.paths["raw_data"]))
            except Exception as exc:
                self.init_error = f"Raw data could not be loaded: {exc}"
        if self.paths["descriptor_map"].exists():
            try:
                self.map_df = clean_headers_and_strings(pd.read_excel(self.paths["descriptor_map"]))
            except Exception as exc:
                self.init_error = f"Descriptor map could not be loaded: {exc}"

        if all(p.exists() for p in self.paths.values()):
            try:
                self.builder = FrozenFeatureBuilder(**self.paths)
            except Exception as exc:
                self.init_error = str(exc)

    @property
    def ready(self) -> bool:
        return self.builder is not None and self.model_service is not None

    @property
    def ui_ready(self) -> bool:
        return self.raw_train is not None and self.map_df is not None

    def _material_options_from_raw(self) -> Dict[str, list[str]]:
        if self.raw_train is None:
            return {"oxidizer": [], "reducer": [], "third_monomer": []}
        groups = {
            "oxidizer": OXI_TYPES,
            "reducer": RED_TYPES,
            "third_monomer": TER_TYPES,
        }
        out = {}
        for key, cols in groups.items():
            vals = []
            for col in cols:
                if col in self.raw_train.columns:
                    vals.extend(str(x).strip() for x in self.raw_train[col].dropna().tolist() if is_valid_name(x))
            out[key] = sorted(set(vals), key=lambda x: x.lower())
        return out

    def _label_map(self, materials: Dict[str, list[str]]) -> Dict[str, str]:
        labels = {}
        if self.map_df is None:
            for vals in materials.values():
                for name in vals:
                    labels[name] = name
            return labels
        lookup = {}
        for _, row in self.map_df.iterrows():
            raw = str(row.get("raw_name", "")).strip().lower()
            if raw:
                lookup[raw] = row
        for vals in materials.values():
            for name in vals:
                row = lookup.get(str(name).strip().lower())
                std = row.get("standard_name") if row is not None else None
                labels[name] = str(std).strip() if is_valid_name(std) else name
        return labels

    def schema(self) -> Dict[str, Any]:
        materials = self._material_options_from_raw()
        labels = self._label_map(materials)
        missing = [p.name for p in self.paths.values() if not p.exists()]
        return {
            "ready": self.ready,
            "ui_ready": self.ui_ready,
            "model_ready": self.model_service is not None,
            "materials": materials,
            "material_labels": labels,
            "material_counts": {k: len(v) for k, v in materials.items()},
            "missing_assets": missing,
            "init_error": self.init_error,
            "process_fields": [
                "init_temp", "poly_temp", "poly_pressure", "time",
                "initiator_feed_mode", "coinitiator_pct", "dispersant_pct",
                "pH_buffer_pct",
            ],
            "excluded_from_ui": ["init_pressure"],
            "dosage_basis": "VAc-based wt%",
            "property_driven_inputs": {
                "emulsifier": {
                    "slots": 4,
                    "fields": ["dosage", "EO_number"],
                    "note": "EO number is required for every active emulsifier slot; HLB alone is not used as a substitute.",
                },
                "protective_colloid": {
                    "slots": 3,
                    "fields": ["dosage", "degree_of_hydrolysis", "DP"],
                    "note": "Use the same encoding as model development; for HEC-type materials, degree of hydrolysis is entered as 0.",
                },
            },
            "master_feature_count": int(len(self.builder.master_columns)) if self.builder else None,
            "raw_training_rows": int(len(self.raw_train)) if self.raw_train is not None else None,
        }

    def _validate_process(self, raw: Dict[str, Any]) -> None:
        required_num = [
            "init_temp", "poly_temp", "poly_pressure", "time",
            "coinitiator_pct", "dispersant_pct", "pH_buffer_pct",
        ]
        for col in required_num:
            if raw.get(col) in (None, ""):
                raise ValueError(f"Missing process parameter: {col}")
            try:
                value = float(raw[col])
            except Exception:
                raise ValueError(f"Process parameter {col} is not a valid number.")
            if not math.isfinite(value):
                raise ValueError(f"Process parameter {col} must be finite.")
            if col in {"time", "coinitiator_pct", "dispersant_pct", "pH_buffer_pct"} and value < 0:
                raise ValueError(f"Process parameter {col} cannot be negative.")
        if raw.get("initiator_feed_mode") not in {"batch", "continuous", "semi_continuous"}:
            raise ValueError("Initiator feeding mode must be batch / continuous / semi_continuous.")

    @staticmethod
    def _active_pct(raw: Dict[str, Any], key: str) -> float:
        val = raw.get(key)
        if val in (None, ""):
            return 0.0
        try:
            out = float(val)
        except Exception:
            raise ValueError(f"{key} is not a valid number.")
        if not math.isfinite(out) or out < 0:
            raise ValueError(f"{key} must be a non-negative finite number.")
        return out

    def _validate_named_families(self, raw: Dict[str, Any]) -> None:
        family_specs = [
            ("Oxidizer", OXI_TYPES, OXI_PCTS),
            ("Reducer", RED_TYPES, RED_PCTS),
            ("Functional comonomer", TER_TYPES, TER_PCTS),
        ]
        for label, type_cols, pct_cols in family_specs:
            for i, (tcol, pcol) in enumerate(zip(type_cols, pct_cols), 1):
                name = raw.get(tcol)
                pct = self._active_pct(raw, pcol)
                has_name = is_valid_name(name)
                if has_name and pct <= 0:
                    raise ValueError(f"{label} {i}: a material is selected, but dosage is missing or not greater than 0.")
                if (not has_name) and pct > 0:
                    raise ValueError(f"{label} {i}: a dosage is entered, but no material is selected.")

    def _validate_property_families(self, raw: Dict[str, Any]) -> None:
        for i in range(1, 5):
            pct = self._active_pct(raw, f"emulsifier_pct{i}")
            eo = raw.get(f"emulsifier_eo{i}")
            if pct > 0:
                if eo in (None, ""):
                    raise ValueError(f"Emulsifier {i}: EO number is required when dosage is greater than 0.")
                try:
                    eo_val = float(eo)
                except Exception:
                    raise ValueError(f"Emulsifier {i}: EO number is not a valid number.")
                if not math.isfinite(eo_val) or eo_val < 0:
                    raise ValueError(f"Emulsifier {i}: EO number must be a non-negative finite number.")

        for i in range(1, 4):
            pct = self._active_pct(raw, f"protective_colloid_pct{i}")
            dh = raw.get(f"protective_colloid_dh{i}")
            dp = raw.get(f"protective_colloid_dp{i}")
            if pct > 0:
                if dh in (None, "") or dp in (None, ""):
                    raise ValueError(f"Protective colloid {i}: degree of hydrolysis and DP are required when dosage is greater than 0.")
                try:
                    dh_val, dp_val = float(dh), float(dp)
                except Exception:
                    raise ValueError(f"Protective colloid {i}: DH/DP contains an invalid number.")
                if not math.isfinite(dh_val) or dh_val < 0 or dh_val > 100:
                    raise ValueError(f"Protective colloid {i}: degree of hydrolysis must be within 0–100%.")
                if not math.isfinite(dp_val) or dp_val < 0:
                    raise ValueError(f"Protective colloid {i}: DP must be a non-negative finite number.")

    def _lookup_desc(self, raw_name: Any) -> Dict[str, Any]:
        if self.map_df is None or not is_valid_name(raw_name):
            return {}
        key = str(raw_name).strip().lower()
        for _, row in self.map_df.iterrows():
            if str(row.get("raw_name", "")).strip().lower() == key:
                return row.to_dict()
        return {}

    def example(self) -> Dict[str, Any]:
        if self.raw_train is None:
            raise RuntimeError("Raw training table is not available.")
        df = self.raw_train
        core = ["init_temp", "poly_temp", "poly_pressure", "time", "initiator_feed_mode", "coinitiator_pct", "dispersant_pct", "pH_buffer_pct"]
        candidates = []
        for idx, row in df.iterrows():
            if any(pd.isna(row.get(c)) for c in core):
                continue
            used = sum(1 for c in OXI_TYPES + RED_TYPES + TER_TYPES if is_valid_name(row.get(c)))
            used += sum(1 for c in [f"emulsifier_type{i}" for i in range(1,5)] + [f"protective_colloid_type{i}" for i in range(1,4)] if is_valid_name(row.get(c)))
            candidates.append((used, int(idx)))
        if not candidates:
            raise RuntimeError("No usable example formulation was found.")
        _, idx = max(candidates, key=lambda x: (x[0], -x[1]))
        row = df.loc[idx]

        out: Dict[str, Any] = {
            "init_temp": parse_numeric(row.get("init_temp")),
            "poly_temp": parse_numeric(row.get("poly_temp")),
            "poly_pressure": parse_numeric(row.get("poly_pressure")),
            "time": parse_numeric(row.get("time")),
            "initiator_feed_mode": normalize_mode_value_en(row.get("initiator_feed_mode")),
            "coinitiator_pct": parse_numeric(row.get("coinitiator_pct")),
            "dispersant_pct": parse_numeric(row.get("dispersant_pct")),
            "pH_buffer_pct": parse_numeric(row.get("pH_buffer_pct")),
        }
        for type_cols, pct_cols in [(OXI_TYPES, OXI_PCTS), (RED_TYPES, RED_PCTS), (TER_TYPES, TER_PCTS)]:
            for tcol, pcol in zip(type_cols, pct_cols):
                name = row.get(tcol)
                pct = parse_numeric(row.get(pcol))
                out[tcol] = str(name).strip() if is_valid_name(name) and np.isfinite(pct) and pct > 0 else None
                out[pcol] = float(pct) if np.isfinite(pct) and pct > 0 else None

        # Property-driven emulsifier fields.
        for i in range(1,5):
            name = row.get(f"emulsifier_type{i}")
            pct = parse_numeric(row.get(f"emulsifier_pct{i}"))
            out[f"emulsifier_type{i}"] = None
            out[f"emulsifier_pct{i}"] = float(pct) if np.isfinite(pct) and pct > 0 else None
            desc = self._lookup_desc(name)
            eo = parse_numeric(desc.get("EO_number")) if desc else np.nan
            out[f"emulsifier_eo{i}"] = float(eo) if np.isfinite(eo) and pct and pct > 0 else None

        # Property-driven protective-colloid fields.
        for i in range(1,4):
            name = row.get(f"protective_colloid_type{i}")
            pct = parse_numeric(row.get(f"protective_colloid_pct{i}"))
            out[f"protective_colloid_type{i}"] = None
            out[f"protective_colloid_pct{i}"] = float(pct) if np.isfinite(pct) and pct > 0 else None
            desc = self._lookup_desc(name)
            dh = parse_numeric(desc.get("degree_of_hydrolysis")) if desc else np.nan
            dp = parse_numeric(desc.get("DP")) if desc else np.nan
            out[f"protective_colloid_dh{i}"] = float(dh) if np.isfinite(dh) and pct and pct > 0 else None
            out[f"protective_colloid_dp{i}"] = float(dp) if np.isfinite(dp) and pct and pct > 0 else None

        return {"source_row": idx, "raw_formula": out}

    @staticmethod
    def _weighted_mean(raw: Dict[str, Any], pct_prefix: str, value_prefix: str, slots: int) -> float:
        weights, values = [], []
        for i in range(1, slots + 1):
            pct = parse_numeric(raw.get(f"{pct_prefix}{i}"))
            val = parse_numeric(raw.get(f"{value_prefix}{i}"))
            if np.isfinite(pct) and pct > 0:
                if not np.isfinite(val):
                    raise ValueError(f"Missing property value for active slot {i}: {value_prefix}{i}")
                weights.append(float(pct)); values.append(float(val))
        if not weights:
            return 0.0
        w = np.asarray(weights, dtype=float)
        v = np.asarray(values, dtype=float)
        return float(np.sum(w * v) / np.sum(w))

    def _apply_property_overrides(self, raw: Dict[str, Any], X_master: pd.DataFrame) -> pd.DataFrame:
        out = X_master.copy()
        overrides = {
            "emul_mean_EO_number": self._weighted_mean(raw, "emulsifier_pct", "emulsifier_eo", 4),
            "pc_mean_degree_of_hydrolysis": self._weighted_mean(raw, "protective_colloid_pct", "protective_colloid_dh", 3),
            "pc_mean_DP": self._weighted_mean(raw, "protective_colloid_pct", "protective_colloid_dp", 3),
        }
        for col, value in overrides.items():
            if col in out.columns:
                out.loc[:, col] = value
        return out

    def predict(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if self.model_service is None:
            raise RuntimeError("Four trained model files are not present. Copy the verified V3.3 model bundle into the models folder before prediction.")
        if self.builder is None:
            missing = ", ".join(self.schema()["missing_assets"])
            extra = f"; initialization error: {self.init_error}" if self.init_error else ""
            raise RuntimeError(f"Frozen feature engineering is not ready. Missing: {missing}{extra}")

        raw = dict(raw)
        raw.pop("init_pressure", None)
        self._validate_process(raw)
        self._validate_named_families(raw)
        self._validate_property_families(raw)

        # Emulsifier/protective-colloid names are intentionally not used in V3.4.
        # Their final retained descriptors are supplied directly by the user.
        for i in range(1,5):
            raw[f"emulsifier_type{i}"] = None
        for i in range(1,4):
            raw[f"protective_colloid_type{i}"] = None

        bad = self.builder.validate_material_names(raw)
        if bad:
            raise ValueError("The following material names were not found in the frozen descriptor mapping: " + ", ".join(bad))

        raw_df = pd.DataFrame([raw])
        X_master = self.builder.transform_raw_to_master(raw_df)
        X_master = self._apply_property_overrides(raw, X_master)
        result = self.model_service.predict_master(X_master)
        result["warnings"] = result.pop("range_warnings", [])
        result["feature_engineering"] = {
            "master_feature_count": int(X_master.shape[1]),
            "raw_formula_mode": True,
            "property_driven_emulsifier_input": True,
            "property_driven_protective_colloid_input": True,
            "pipeline": [
                "raw formulation",
                "descriptor lookup for redox/comonomer identities",
                "direct EO/DH/DP property input for stabilizer families",
                "dosage-weighted pooling",
                "dose/share construction",
                "frozen transforms",
                "target-specific feature alignment",
            ],
        }
        return result
