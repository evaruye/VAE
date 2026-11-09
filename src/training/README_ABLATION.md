# Feature-representation ablation experiments

This directory provides two comparison inputs for each target (EC, SC, VC and HCHO): process-only features and equal-weight descriptor pooling. The scripts retain the target-specific train/test partitions, 10-fold validation procedures and fixed model hyperparameters. Optuna is not rerun.

## Input data

- `data/Data_S2/Process_Only.xlsx`: four worksheets with 5/6 process features and observed targets.
- `data/Data_S2/Equal_Weight_Pooling.xlsx`: four worksheets with 39/40 features and observed targets; only 11 pooled descriptor columns differ from the dosage-weighted representation.
- `data/Data_S2/{EC,SC,VC,HCHO}_Modeling_Data.xlsx`: complete target-specific modeling matrices used for row alignment and splitting.

## Run from the repository root

```bash
python -m pip install -r requirements.txt
python src/training/EC_process_only_originalCV.py
python src/training/EC_equal_pooling_originalCV.py
python src/training/SC_process_only_originalCV.py
python src/training/SC_equal_pooling_originalCV.py
python src/training/VC_process_only_originalCV.py
python src/training/VC_equal_pooling_originalCV.py
python src/training/HCHO_process_only_originalCV.py
python src/training/HCHO_equal_pooling_originalCV.py
```

## Output files

The shared `run_ablation_core.py` computes fold-level R²/RMSE/MAE, OOF predictions and summary statistics. Newly generated files are stored locally under `results/aligned_ablation/` and are not part of the input data workbooks. The scripts use the corresponding target-specific model routines and any applicable fold-level early-stopping settings.
