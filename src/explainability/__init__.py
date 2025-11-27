"""
Model Explainability Module

This module provides tools for explaining model predictions
using SHAP and LIME.
"""

from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .global_explanations import (
    get_global_feature_importance,
    plot_partial_dependence,
    get_feature_interactions
)
from .local_explanations import (
    explain_prediction,
    get_influential_features,
    generate_explanation_report
)

__all__ = [
    'SHAPExplainer',
    'LIMEExplainer',
    'get_global_feature_importance',
    'plot_partial_dependence',
    'get_feature_interactions',
    'explain_prediction',
    'get_influential_features',
    'generate_explanation_report'
]
