"""
Dashboard Module

This module provides dashboard components for credit risk analytics visualization.
"""

from .components import (
    create_metric_card,
    create_performance_chart,
    create_feature_importance_plot,
    create_confusion_matrix_plot
)
from .layouts import get_layout, AVAILABLE_LAYOUTS

__all__ = [
    'create_metric_card',
    'create_performance_chart',
    'create_feature_importance_plot',
    'create_confusion_matrix_plot',
    'get_layout',
    'AVAILABLE_LAYOUTS'
]
