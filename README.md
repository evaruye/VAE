# Data Fusion and Explainable Machine Learning for Constrained Multi-Property Design of VAE Emulsions

**Repository:** [https://github.com/evaruye/VAE](https://github.com/evaruye/VAE)  

Research data, frozen surrogate models, and supporting implementation files accompanying the manuscript **“Data Fusion and Explainable Machine Learning for Constrained Multi-Property Design of VAE Emulsions”**.

## Study overview

The study curates 214 formulation–process examples from 27 public patents describing redox-initiated vinyl acetate–ethylene (VAE) emulsion polymerization. Functional-class-specific dosage-weighted pooling represents heterogeneous, variable-length ingredient lists using class dosages, within-class fractions, and pooled molecular or empirical descriptors. Four property-specific models support multi-property prediction and constrained candidate prioritization. The C1–C8 formulations are **computational candidates, not experimentally validated optima**.

| Property | Final model | Held-out test R² | Scale of reported test metrics |
| --- | --- | ---: | --- |
| Ethylene content (EC) | CatBoost | 0.9798 | wt% (native) |
| Solid content (SC) | XGBoost | 0.9751 | wt% (native) |
| Viscosity (VC) | CatBoost | 0.9031 | ln(1 + VC) |
| Free formaldehyde content (HCHO) | MLP | 0.9212 | ln(1 + HCHO) |

Note: The VC/HCHO R², RMSE and MAE are calculated on the transformed model-output scales. Engineering-window checks and selected response plots are reported in physical units after inverse transformation where indicated in the manuscript.

## Directory layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── FILE_MANIFEST.json
├── data/
│   ├── Data_S1.xlsx
│   └── Data_S2/
│       ├── Curated_VAE_Formulation_Dataset.xlsx
│       ├── Raw_Material_Descriptors.xlsx
│       ├── EC_Modeling_Data.xlsx
│       ├── SC_Modeling_Data.xlsx
│       ├── VC_Modeling_Data.xlsx
│       ├── HCHO_Modeling_Data.xlsx
│       ├── Final_C1_C8_Candidates.xlsx
│       ├── Data_Dictionary.xlsx
│       ├── Equal_Weight_Pooling.xlsx
│       ├── Process_Only.xlsx
│       ├── README.md
│       └── DFT_Inputs/              # 68 Gaussian .gjf inputs + mapping README
├── models/                       # Four frozen .joblib surrogates
├── src/
│   ├── training/                 # Target-specific training scripts
│   ├── validation/               # Y-randomization script
│   ├── feature_engineering/      # Research-stage feature-construction script
│   └── optimization/             # Research-stage NSGA-II and AD/diversity scripts
├── app/                          # Independent FastAPI forward-prediction interface
│   └── smoke_test.py              # Health, schema, prediction endpoint smoke tests
└── tests/                        # Frozen-model held-out performance verification
```

**There is no checked-in `results/` directory.** Some research scripts write their outputs there when run; generated files are ignored by Git. Published C1–C8 results are instead in `data/Data_S2/Final_C1_C8_Candidates.xlsx`. This repository does not include a pre-generated Pareto/results archive.


## JCIM release: feature ablations and reproducibility

This updated release aligns figure-source data and supporting materials with the JCIM manuscript numbering.

- `data/Data_S1.xlsx`: 16 independent figure-source worksheets: `Figure 2`, `Figure 3`, `Figure 4a`, `Figure 4b`, `Figure 5`–`Figure 9`, and `Figure S1`–`Figure S7`. The previous SHAP and optimization figures were renumbered to make room for the two-panel Figure 4. The final submission-aligned file corrects the Figure 9 worksheet panel-(b) label to **Figure 9b**; underlying plotted numbers are unchanged.
- `data/Data_S2/Equal_Weight_Pooling.xlsx` and `data/Data_S2/Process_Only.xlsx`: each contains four EC/SC/VC/HCHO input-plus-target worksheets. Only source data are distributed; per-fold scores, OOF predictions and run metadata are omitted.
- `src/training/*_process_only_originalCV.py` and `*_equal_pooling_originalCV.py`: eight launchers for fixed-hyperparameter, original-protocol ablations. They invoke `run_ablation_core.py` and the existing target-specific original model modules, without running Optuna.
- `src/training/README_ABLATION_JCIM.md`: instructions for running fixed-parameter ablation comparisons.

**Reproducibility.** Dependency requirements are provided in `requirements.txt`. The ablation scripts generate local evaluation outputs, which can be compared with the manuscript and Supporting Information. The released data and scripts do not imply independent re-execution of every published experiment. The final candidates are computational hypotheses and have not been experimentally validated.

This archive is a repository-upload package, **not evidence that the GitHub URL above is currently public or up to date**. The corresponding author should verify the public URL and make sure this complete version is accessible to reviewers before submission.

## Supplementary Data S1 — figure source data

`data/Data_S1.xlsx` contains 16 worksheets of numerical data underlying Figures 2–9 and Figures S1–S7. Figure 1 is a schematic and Figure S8 is a screenshot; neither has a corresponding numerical worksheet. The journal's Supplementary Data S1 archive contains this workbook. In this repository it is supplied directly as an XLSX file for convenient access.

## Supplementary Data S2 — curated and modeling datasets

`data/Data_S2/` contains the files described in Supplementary Data S2 of the Supporting Information. The corresponding journal supplementary archive is distributed as a ZIP; here its contents are unpacked for browsing. See [`data/Data_S2/README.md`](data/Data_S2/README.md) for descriptions, variable conventions, transformations, and input-file mapping.

- `Curated_VAE_Formulation_Dataset.xlsx`: 214 records; targets expressed in their physical units and missingness retained.
- `Raw_Material_Descriptors.xlsx`: standardized raw-material descriptors.
- Target-specific matrices: EC 185 × 39 inputs; SC 185 × 39; VC 176 × 39; HCHO 214 × 40, plus one target column in each file. These contain the complete eligible modeling records (both train and held-out test data), not training-only subsets.
- `Final_C1_C8_Candidates.xlsx`: shortlisted process/formulation data, surrogate predictions, desirability and applicability-domain diagnostics.
- `Data_Dictionary.xlsx`: dataset, feature and descriptor definitions.
- `Equal_Weight_Pooling.xlsx` and `Process_Only.xlsx`: four target-specific input matrices and corresponding observed response columns in each workbook; no calculated metrics.
- `DFT_Inputs/`: 34 Gaussian calculation pairs (68 `.gjf` input files). **Gaussian output logs and descriptor-extraction scripts are not included.** Multiple materials may share one ionic species calculation; see the mapping README.

For viscosity and free formaldehyde, model-ready target columns are `ln(1+y)` transformed, while the curated table contains the responses in mPa·s and ppm respectively. To return to physical units, use `y = exp(z) - 1`.

## Model files and platform

The frozen models are in `models/`: `EC_CatBoost.joblib`, `SC_XGBoost.joblib`, `VC_CatBoost.joblib`, and `HCHO_MLP.joblib`. The self-contained `app/` directory also has its own frozen model copies and feature assets. Those platform assets are not interchangeable with the English-standardized publication dataset without checking the raw-material mapping and feature schema.

To run the **forward-prediction platform** (not the optimization interface):

```bash
cd app
python -m venv .venv
# Activate the virtual environment using the command for your OS
python -m pip install -r requirements.txt
python run.py
```

Verify the API, model bundle, and sample forward prediction without starting the server:

```bash
python app/smoke_test.py
```

**Verification performed for this release:** all four frozen models loaded, and both raw-formulation and engineered-feature prediction endpoints returned successful results. The raw-example predictions were VC = 328.8125 mPa·s, SC = 49.9101 wt%, EC = 22.4477 wt%, and HCHO = 23.0649 ppm. This was tested via FastAPI's in-process TestClient. A separate graphical-browser end-to-end test was **not** completed in the restricted execution environment.

See [`app/README.md`](app/README.md). Its default interface address is `http://127.0.0.1:8000`. The platform predicts EC, SC, VC and HCHO from raw formulation/process inputs. It does **not** run NSGA-II or applicability-domain diagnostics through the web UI.

## Training, validation, and reproducibility scope

Research dependencies are listed in the repository's root `requirements.txt`. Create and activate a fresh Python environment before installation:

```bash
python -m pip install -r requirements.txt
```

The four training-script input paths and the Y-randomization script input paths reference the corresponding published workbooks in `data/Data_S2/` in this release. These source scripts document the model-specific workflows and hyperparameter searches, which can be computationally expensive:

```bash
python src/training/train_EC_CatBoost.py
python src/training/train_SC_XGBoost.py
python src/training/train_VC_CatBoost.py
python src/training/train_HCHO_MLP.py
python src/validation/y_randomization.py --target ALL --permutations 100
```

**Frozen-model held-out verification:** `python tests/verify_heldout.py` reconstructs the original train/test splits, reloads the published modeling matrices and frozen `.joblib` estimators, and checks the four published held-out R²/RMSE/MAE triples. This verification was run successfully for all four targets (EC, SC, VC, HCHO), matching the manuscript to the reported four decimal places. The validation script explicitly maps the publication column `ter_mean_E_gap` to the saved model name `ter_mean_△G` when necessary; no original workbook is modified. Run `python src/validation/y_randomization.py --check-only --target ALL` to inspect all input matrices and split definitions **without** executing 100 permutations.

**Scope limitation:** Full end-to-end model retraining, hyperparameter search, 100-permutation reruns and numerical regeneration of every optimization intermediate were **not** established as part of this release check. Executing retraining can create new model artifacts that need not be byte-identical to the supplied frozen models.

The code in `src/feature_engineering/` and `src/optimization/` is retained as research-stage implementation material. It still refers to some **legacy intermediate workbooks absent from this data release**, such as `data/X_reduced_final.xlsx`, `data/X_EC_true.xlsx`, and `data/ML1_descriptors_final_APEO.xlsx`, as well as generated `results/nsga2/` files. Therefore, the NSGA-II → AD/Gower chain **is not a turnkey reproducible command sequence from this ZIP alone**. Do not interpret the existence of these scripts as confirmation that the full optimization is immediately executable on a fresh clone. The final published candidate tables and parameter definitions are accessible in Data S2 and the manuscript/Supporting Information.

## Data provenance and interpretation

The 27 patent sources are enumerated in Table S1 of the Supporting Information. Model evaluation used held-out testing, cross-validation and 100 Y-randomization trials as described there. SHAP/PDP trends are model-response associations, not direct causal measurements. C1–C8 all satisfy the Practical applicability domain but fall outside the Strict core; no experimental validation of these proposed formulations is claimed.

## File integrity

`FILE_MANIFEST.json` records the SHA-256 digest and size of each released file (except the manifest itself). The repository excludes local Python virtual environments, cache files and generated results. Use `requirements.txt` to install dependencies instead of uploading `.venv/`.

## Independent verification summary (this release)

| Frozen model | Held-out n | Recomputed test R² | Recomputed RMSE | Recomputed MAE | Manuscript agreement |
| --- | ---: | ---: | ---: | ---: | --- |
| EC–CatBoost | 19 | 0.979806 | 0.626121 | 0.357341 | Yes |
| SC–XGBoost | 19 | 0.975105 | 0.677262 | 0.510569 | Yes |
| VC–CatBoost | 18 | 0.903102 | 0.349485 | 0.268303 | Yes |
| HCHO–MLP | 22 | 0.921247 | 0.413379 | 0.208810 | Yes |

All VC/HCHO validation metrics are on the log(1+y) scale. This frozen-model replay **does not** prove that a complete retraining or NSGA-II/AD run can be regenerated using the present release alone.

