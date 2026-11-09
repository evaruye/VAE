# -*- coding: utf-8 -*-
"""SC process_only: original model CV with fixed data splits and hyperparameters, no Optuna refit."""
from run_ablation_core import main

if __name__ == '__main__':
    main('SC', 'process_only')
