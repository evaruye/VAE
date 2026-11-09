# Data Fusion and Explainable Machine Learning for Constrained Multi-Property Design of VAE Emulsions

This repository contains curated VAE emulsion formulation data, chemical descriptors, feature-engineering tools, property-specific regression models, model validation scripts, formulation optimization code, and a four-property forward-prediction application.

## 1. Project overview

- **Data:** 214 formulation–process examples curated from 27 public patents on redox-initiated vinyl acetate–ethylene (VAE) emulsion polymerization.
- **Properties:** ethylene content (EC), solid content (SC), viscosity (VC), and free formaldehyde (HCHO).
- **Representation:** functional-class dosage totals, within-class composition fractions, and dosage-weighted pooled molecular/empirical descriptors.
- **Workflow:** data processing → feature engineering → model training and validation → interpretation → constrained candidate screening.
- **Candidates:** C1–C8 are computationally selected formulations; experimental validation has not been performed.

| Property | Regression model | Held-out test R² | Evaluation scale |
| --- | --- | ---: | --- |
| EC | CatBoost | 0.9798 | wt% |
| SC | XGBoost | 0.9751 | wt% |
| VC | CatBoost | 0.9031 | ln(1 + VC) |
| HCHO | MLP | 0.9212 | ln(1 + HCHO) |

For VC and HCHO, R²/RMSE/MAE are evaluated on the transformed model-output scales. Physical values can be recovered by `expm1(z)`.

## 2. Project structure

```text
.
├── README.md                       # Project overview and usage
├── requirements.txt                # Research environment
├── FILE_MANIFEST.json              # File checksums
├── data/
│   ├── Data_S1.xlsx                # Plot source data (16 worksheets)
│   └── Data_S2/                   # Curated data, model matrices, ablation inputs and Gaussian inputs
├── models/                         # Four frozen regression models
├── src/
│   ├── feature_engineering/        # Feature construction
│   ├── training/                   # Regression training and ablation scripts
│   ├── validation/                 # Y-randomization tests
│   └── optimization/               # NSGA-II, applicability-domain and diversity screening
├── app/                            # FastAPI four-property prediction platform
└── tests/                          # Frozen-model held-out test checks
```

## 3. Datasets

**`data/Data_S1.xlsx`** contains 16 worksheets of numerical source data for the property distributions, prediction plots, representation comparisons, SHAP plots, candidate selection plots, and related diagnostics. Worksheet names such as `Figure 4a`, `Figure 4b`, and `Figure 9` are retained as data identifiers.

**`data/Data_S2/`** includes:

| File | Contents |
| --- | --- |
| `Curated_VAE_Formulation_Dataset.xlsx` | 214 formulation/process/property records in physical units |
| `Raw_Material_Descriptors.xlsx` | Molecular, quantum-chemical and empirical raw-material descriptors |
| `EC_Modeling_Data.xlsx` | 185 rows; 39 model input features plus EC response |
| `SC_Modeling_Data.xlsx` | 185 rows; 39 model input features plus SC response |
| `VC_Modeling_Data.xlsx` | 176 rows; 39 model input features plus log-transformed VC response |
| `HCHO_Modeling_Data.xlsx` | 214 rows; 40 model input features plus log-transformed HCHO response |
| `Process_Only.xlsx` | Four target worksheets with process inputs and observed responses |
| `Equal_Weight_Pooling.xlsx` | Four target worksheets with equal-weight pooled descriptors and observed responses |
| `Final_C1_C8_Candidates.xlsx` | Candidate process conditions, formulation details, predictions and reliability diagnostics |
| `Data_Dictionary.xlsx` | Variable names, definitions, transformations and units |
| `DFT_Inputs/` | 68 Gaussian `.gjf` inputs and mapping documentation |

See [`data/Data_S2/README.md`](data/Data_S2/README.md) for data conventions.

## 4. Feature engineering

The fixed-length feature representation combines process variables, formulation dosages, within-class ingredient fractions and pooled descriptors. For structurally characterized ingredient classes, pooled descriptors reflect relative ingredient dosages.

The process-only comparison uses five input features for EC, SC and VC: `init_temp`, `poly_pressure`, `poly_temp`, `time_log1p`, and `initiator_feed_continuous`. HCHO additionally includes `initiator_feed_semi_continuous`. The equal-weight comparison retains the 39/40-column input schema, replacing only the 11 pooled descriptor values with equal-weight aggregates.

Scripts in `src/feature_engineering/` document research-stage feature-construction steps. Some scripts refer to intermediate files that are not included in this repository.

## 5. Model training

Install the dependencies in a new Python environment:

```bash
python -m pip install -r requirements.txt
```

Train the four regressors using the target-specific pipelines:

```bash
python src/training/train_EC_CatBoost.py
python src/training/train_SC_XGBoost.py
python src/training/train_VC_CatBoost.py
python src/training/train_HCHO_MLP.py
```

Training scripts may be computationally intensive due to hyperparameter optimization. The frozen model files are provided in `models/`; retraining can produce models that differ from those saved artifacts.

## 6. Model validation

Run the frozen-model held-out test check:

```bash
python tests/verify_heldout.py
```

Run Y-randomization, or inspect the input definitions without running permutations:

```bash
python src/validation/y_randomization.py --check-only --target ALL
python src/validation/y_randomization.py --target ALL --permutations 100
```

The validation check reconstructs the target-specific 90/10 partitions and compares computed test metrics with the project reference metrics. It does not retrain the estimators.

## 7. Feature-representation comparisons

The eight process-only and equal-weight comparison scripts are in `src/training/`. For example:

```bash
python src/training/EC_process_only_originalCV.py
python src/training/EC_equal_pooling_originalCV.py
```

Each target has two such scripts. They reuse the target-specific fitting routines and fixed hyperparameters without rerunning Optuna. They generate local fold-level and OOF evaluation outputs under `results/` (ignored by Git). The source workbooks contain only input features and observed responses, not calculated validation outputs.

See [`src/training/README_ABLATION.md`](src/training/README_ABLATION.md).

## 8. Optimization and candidate screening

The `src/optimization/` directory contains research-stage code for NSGA-II search, engineering constraints, applicability-domain assessment and candidate diversity selection. Some intermediate input workbooks or generated search outputs are not bundled, so the entire optimization sequence is not directly runnable on a fresh checkout without those inputs. The selected C1–C8 formulations and reliability diagnostics are available in `data/Data_S2/Final_C1_C8_Candidates.xlsx`.

## 9. Prediction platform

The `app/` directory provides a FastAPI application for forward prediction of EC, SC, VC and HCHO from formulation/process inputs. The platform includes its own model files and feature assets; verify that its feature schema matches the selected input source.

```bash
cd app
python -m pip install -r requirements.txt
python run.py
```

The default local address is `http://127.0.0.1:8000`. From the repository root, run `python app/smoke_test.py` to check available endpoints. The web interface performs forward prediction, not NSGA-II optimization or applicability-domain filtering.

## 10. File integrity

`FILE_MANIFEST.json` lists SHA-256 checksums and byte lengths for the repository files. Local cache files, virtual environments and generated results are excluded through `.gitignore`.
