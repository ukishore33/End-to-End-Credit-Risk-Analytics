# Model Evaluation Module

This module contains model evaluation utilities for credit risk models.

## Overview

The evaluation module provides:
- Classification metrics (AUC-ROC, Precision, Recall, F1)
- Regression metrics (RMSE, MAE, R²)
- Credit-specific metrics (KS statistic, Gini coefficient)
- Visualization tools for model performance
- Model comparison utilities

## Files

- `metrics.py` - Model evaluation metrics
- `visualizations.py` - Evaluation visualizations
- `comparison.py` - Model comparison utilities

## Usage

```python
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_credit_metrics,
    calculate_regression_metrics
)
from src.evaluation.visualizations import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_lift_curve
)

# Calculate metrics
clf_metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
credit_metrics = calculate_credit_metrics(y_true, y_prob)

# Plot visualizations
plot_roc_curve(y_true, y_prob, save_path='reports/figures/roc.png')
plot_confusion_matrix(y_true, y_pred, save_path='reports/figures/cm.png')
```

## Key Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR
- Confusion Matrix

### Credit-Specific
- KS Statistic
- Gini Coefficient
- Lift/Gain Charts
- Population Stability Index (PSI)

### Regression
- MSE, RMSE, MAE
- R², Adjusted R²
- MAPE

## TODO

- [ ] Add calibration plots
- [ ] Implement bootstrap confidence intervals
- [ ] Add model fairness metrics
