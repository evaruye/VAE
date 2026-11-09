# -*- coding: utf-8 -*-
"""HCHO equal_pooling: original model CV with fixed data splits and hyperparameters, no Optuna refit."""
from run_ablation_core import main

if __name__ == '__main__':
    main('HCHO', 'equal_pooling')
