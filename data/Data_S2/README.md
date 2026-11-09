# Supplementary Data S2 — VAE Dataset and Modeling Data

This package supports the manuscript **Data Fusion and Explainable Machine Learning for Constrained Multi-Property Design of VAE Emulsions**.

It contains the curated formulation–process–property dataset, English-standardized raw-material descriptor table, final target-specific modeling matrices, data dictionary, Gaussian calculation inputs, and final C1–C8 candidate tables used in the manuscript and Supporting Information.

## Package contents

- `Curated_VAE_Formulation_Dataset.xlsx` — 214 curated formulation/process/property records. Ingredient names are standardized to English. Property values are on their physical scales; source missingness is retained.
- `Raw_Material_Descriptors.xlsx` — molecular, quantum-chemical, and empirical descriptor table with English material identifiers.
- `EC_Modeling_Data.xlsx` — final EC model-ready matrix: 185 samples and 39 input features; target on the native wt% scale.
- `SC_Modeling_Data.xlsx` — final SC model-ready matrix: 185 samples and 39 input features; target on the native wt% scale.
- `VC_Modeling_Data.xlsx` — final VC model-ready matrix: 176 samples and 39 input features; target on the ln(1 + VC) scale.
- `HCHO_Modeling_Data.xlsx` — final HCHO model-ready matrix: 214 samples and 40 input features; target on the ln(1 + HCHO) scale.
- `Final_C1_C8_Candidates.xlsx` — final C1–C8 process conditions, formulations, predicted properties, desirability values, and applicability-domain classifications. Candidate numbering matches the manuscript and Supporting Information.
- `Data_Dictionary.xlsx` — four worksheets: `Package_Contents`, `Raw_Data_Columns`, `Model_Features`, and `Descriptor_Columns`.
- `DFT_Inputs/` — 68 Gaussian input files mapped to 39 raw-material entries through 34 pairs of calculations. `DFT_Inputs/README.md` lists the correspondence.
- `Equal_Weight_Pooling.xlsx` — four worksheets (EC, SC, VC, HCHO), containing equal-weight pooled feature matrices and observed target values; no validation results.
- `Process_Only.xlsx` — four worksheets (EC, SC, VC, HCHO), containing only process-feature matrices and observed target values; no validation results.

## Data representation

Ingredient dosages are normalized relative to vinyl acetate (VAc) mass. Variable-length ingredient lists are represented through functional-class totals, within-class composition fractions, and dosage-weighted pooled descriptors as described in the manuscript and Supporting Information.

The four target-specific modeling matrices contain all eligible records used for their respective reported workflows, including both training and test records; they are not training-only tables. They contain the retained inputs used by the final property-specific surrogate models. Missingness-indicator columns excluded from the retained feature sets are not included. EC, SC, and VC use 39 inputs; HCHO uses 40 because the semi-continuous initiator-feeding indicator is additionally retained for HCHO.

## Response eligibility and scales

For each target, only records with valid numerical response values were retained; records with missing or interval-only responses were excluded from the corresponding modeling dataset. Response eligibility was assessed separately for each target. Missing target responses were not imputed.

| Target | Physical scale | Scale in the modeling workbook |
| --- | --- | --- |
| EC | wt% | Native physical scale |
| SC | wt% | Native physical scale |
| VC | mPa·s | ln(1 + VC) |
| HCHO | ppm | ln(1 + HCHO) |

`ln` denotes the natural logarithm. For VC/HCHO model-scale response z, recover the physical-scale value as exp(z) − 1, using the units above.

## Missing inputs

Missing numerical input values were imputed using the median of the corresponding feature. Missing entries are distinct from zero dosage or an absent ingredient slot. The curated source table retains source missingness, while the target-specific modeling workbooks contain the processed inputs used for the reported models.

## Field names and transformations

The raw-data field `free_formaldehyde_content` corresponds to free formaldehyde content (HCHO) in the manuscript and is reported in ppm.

`reducer_type2` identifies the ingredient in reducer slot 2; `reducer_pct2` is its dosage relative to VAc mass; `red_share_of_additives2` is its within-reducer-class dosage fraction. These fields describe different quantities and retain distinct names, as documented in the dictionary.

`E_gap` denotes the HOMO–LUMO energy gap, E_LUMO − E_HOMO, in Hartree. The pooled functional-comonomer input is `ter_mean_E_gap`.

Inputs with the suffix `_log1p` use ln(1 + x). `emul_share_of_additives3_asin` uses asin(sqrt(s)), where s is the within-emulsifier-class dosage fraction. The latter is a transformed, dimensionless angular quantity, not the untransformed fraction. Feature-specific definitions and units are in `Model_Features`; Table S2 describes the corresponding manuscript feature definitions.

## Candidate data and naming

The C1–C8 identifiers are the final identifiers used in the manuscript and Supporting Information.

Reader-facing text is in English. Chemical/material names from non-English patent records are translated or standardized using the curated descriptor mapping. Commercial product codes are retained when they identify formulations.

## Molecular structures and calculation files

The SMILES entries describe the modeled molecular or ionic species. Counterions are omitted from the ionic calculation models where indicated by the supplied structures. Multiple material entries can therefore refer to the same calculation pair. The mapping and the charge/multiplicity of each species are listed in `DFT_Inputs/README.md`.

`NSa-6` corresponds to N-Succinimidyl acrylate, and `N4P2E-5` corresponds to N-(4,4-diethoxybutyl)prop-2-enamide. The VeoVa11 entry uses the C13H24O2 representative structure in `VeoVa11-1`. The sodium and zinc formaldehyde sulfoxylate entries share the supplied Sfs-12 ionic model and the same isotropic polarizability value, 9.8261281201.

## Relationship to the other supplementary files

Data S1 contains numerical source data for Figures 2–9 and S1–S7, including the separate `Figure 4a` and `Figure 4b` worksheets. Figure 5–8 are the four property-specific SHAP plots, and Figure 9 is the multiobjective-optimization plot. The `Figure 3` sheet contains the model training/test labels and actual/predicted responses. Data S2 supplies the underlying curated records and model-input/response matrices. The manuscript and Supporting Information describe preprocessing, model settings, validation procedures, optimization criteria, and applicability-domain definitions.

## Reduced-input datasets

`Process_Only.xlsx` contains the five process inputs `init_temp`, `poly_pressure`, `poly_temp`, `time_log1p`, and `initiator_feed_continuous` for EC, SC, and VC; HCHO additionally contains `initiator_feed_semi_continuous`.

`Equal_Weight_Pooling.xlsx` retains the original 39/40 input feature names and the non-pooled features; only the pooled descriptor values are calculated with equal ingredient weights. Each workbook has four tabs and includes the corresponding observed target column. The files contain source data only, not fold scores, predictions, or model results.

The original manuscript, Supporting Information, and figure-source data document the evaluation methods and comparisons.
