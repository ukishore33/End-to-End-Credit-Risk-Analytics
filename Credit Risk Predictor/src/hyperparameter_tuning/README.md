# Hyperparameter Tuning Module

This module contains hyperparameter optimization utilities for credit risk models.

## Overview

The hyperparameter tuning module provides:
- Grid Search
- Random Search
- Bayesian Optimization
- Cross-validation strategies
- Parameter space definitions

## Files

- `tuner.py` - Main hyperparameter tuning class
- `search_spaces.py` - Predefined search spaces for different models
- `cross_validation.py` - Cross-validation utilities

## Usage

```python
from src.hyperparameter_tuning.tuner import HyperparameterTuner
from src.hyperparameter_tuning.search_spaces import get_search_space

# Get default search space for model
search_space = get_search_space('xgboost')

# Initialize tuner
tuner = HyperparameterTuner(
    search_strategy='bayesian',
    cv=5,
    scoring='roc_auc',
    n_trials=100
)

# Run optimization
best_params = tuner.tune(model, X_train, y_train, search_space)
print(f"Best parameters: {best_params}")
print(f"Best score: {tuner.best_score_}")
```

## Search Strategies

### Grid Search
- Exhaustive search over parameter grid
- Best for small search spaces

### Random Search
- Random sampling from parameter distributions
- Efficient for large search spaces

### Bayesian Optimization
- Uses Optuna or similar libraries
- Most efficient for complex models

## TODO

- [ ] Add Optuna integration
- [ ] Implement early stopping
- [ ] Add parallelization support
- [ ] Create hyperparameter importance analysis
