# Model Explainability Module

This module contains model explainability tools using SHAP and LIME.

## Overview

The explainability module provides:
- SHAP (SHapley Additive exPlanations) analysis
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis
- Individual prediction explanations
- Global model interpretability

## Files

- `shap_explainer.py` - SHAP-based explanations
- `lime_explainer.py` - LIME-based explanations
- `global_explanations.py` - Global interpretability methods
- `local_explanations.py` - Local (instance-level) explanations

## Usage

### SHAP Analysis

```python
from src.explainability.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(model, X_train)
shap_values = explainer.explain(X_test)

# Plot summary
explainer.plot_summary(X_test)

# Individual explanation
explainer.explain_instance(X_test[0])
```

### LIME Analysis

```python
from src.explainability.lime_explainer import LIMEExplainer

explainer = LIMEExplainer(model, X_train, feature_names=feature_names)

# Explain single prediction
explanation = explainer.explain_instance(X_test[0])
explanation.show_in_notebook()
```

## Key Features

### SHAP
- TreeSHAP for tree-based models
- KernelSHAP for any model
- Summary plots
- Dependence plots
- Force plots

### LIME
- Tabular explainer
- Instance-level explanations
- Feature contributions

## TODO

- [ ] Add SHAP waterfall plots
- [ ] Implement partial dependence plots
- [ ] Add counterfactual explanations
- [ ] Integrate with model monitoring
