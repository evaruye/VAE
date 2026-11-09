"""Verify frozen surrogates on reconstructed 90/10 held-out tests.

Usage (from repository root): python tests/verify_heldout.py

No hyperparameter tuning or training occurs. Metrics are on original model
scales: EC/SC in wt%, VC/HCHO on ln(1+y). Strict matching of held-out
metrics is environment-dependent; consult requirements.txt for the listed dependencies.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('vae_yr_validation', ROOT/'src/validation/y_randomization.py')
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)

REFERENCE_METRICS = {
    'EC': (0.9798, 0.6261, 0.3573),
    'SC': (0.9751, 0.6773, 0.5106),
    'VC': (0.9031, 0.3495, 0.2683),
    'HCHO': (0.9212, 0.4134, 0.2088),
}


def main():
    all_pass = True
    for target, cfg in validation.CONFIG.items():
        resolved = dict(cfg)
        resolved['data_file'] = str(ROOT / cfg['data_file'])
        resolved['model_file'] = str(ROOT / cfg['model_file'])
        _, X, y, _, frozen, names = validation.validate_dataset_and_model(target, resolved)
        train_idx, test_idx = validation.reconstruct_original_train_indices(target,y)
        observed = validation.postprocess_prediction(target,frozen.predict(X.iloc[test_idx]))
        metrics=(float(r2_score(y[test_idx],observed)),
                 float(np.sqrt(mean_squared_error(y[test_idx],observed))),
                 float(mean_absolute_error(y[test_idx],observed)))
        expected=REFERENCE_METRICS[target]
        passed = all(abs(a-b)<0.001 for a,b in zip(metrics,expected))
        all_pass &= passed
        print(f'{target}: test_n={len(test_idx)}, R2={metrics[0]:.6f}, RMSE={metrics[1]:.6f}, MAE={metrics[2]:.6f}; reference_metrics={expected}; {"PASS" if passed else "MISMATCH"}')
    if not all_pass:
        raise SystemExit('Frozen-model held-out metrics differ from stored reference values (at least one target).')
    print('PASS: all four frozen models agree with stored held-out reference scores within 0.001')


if __name__=='__main__':
    main()
