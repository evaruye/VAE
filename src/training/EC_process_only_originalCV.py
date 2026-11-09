# -*- coding: utf-8 -*-
"""EC process_only: original model CV with fixed data splits and hyperparameters, no Optuna refit."""
from run_ablation_core import main

if __name__ == '__main__':
    main('EC', 'process_only')
