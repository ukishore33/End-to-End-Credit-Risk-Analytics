"""
Dashboard Layouts Module

This module provides layout configurations for different dashboard pages.
"""

from typing import Dict, List, Any, Optional


# Available layout configurations
AVAILABLE_LAYOUTS = [
    'overview',
    'performance',
    'predictions',
    'monitoring',
    'comparison',
    'explainability'
]


def get_layout(
    layout_name: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get layout configuration for a dashboard page.
    
    Parameters
    ----------
    layout_name : str
        Name of the layout
    config : Dict[str, Any], optional
        Additional configuration options
        
    Returns
    -------
    Dict[str, Any]
        Layout configuration
        
    Examples
    --------
    >>> layout = get_layout('overview')
    >>> print(layout['sections'])
    """
    layouts = {
        'overview': _get_overview_layout(),
        'performance': _get_performance_layout(),
        'predictions': _get_predictions_layout(),
        'monitoring': _get_monitoring_layout(),
        'comparison': _get_comparison_layout(),
        'explainability': _get_explainability_layout()
    }
    
    if layout_name not in layouts:
        raise ValueError(f"Unknown layout: {layout_name}. Available: {AVAILABLE_LAYOUTS}")
    
    layout = layouts[layout_name]
    
    # Apply custom configuration
    if config:
        layout.update(config)
    
    return layout


def _get_overview_layout() -> Dict[str, Any]:
    """Get overview page layout."""
    return {
        'name': 'overview',
        'title': 'Credit Risk Analytics Overview',
        'sections': [
            {
                'name': 'key_metrics',
                'type': 'metrics_row',
                'metrics': ['auc_roc', 'ks_statistic', 'default_rate', 'predictions_count'],
                'columns': 4
            },
            {
                'name': 'performance_trend',
                'type': 'chart',
                'chart_type': 'line',
                'title': 'Model Performance Trend',
                'width': 'half'
            },
            {
                'name': 'prediction_distribution',
                'type': 'chart',
                'chart_type': 'histogram',
                'title': 'Prediction Distribution',
                'width': 'half'
            },
            {
                'name': 'recent_activity',
                'type': 'table',
                'title': 'Recent Activity',
                'width': 'full'
            }
        ]
    }


def _get_performance_layout() -> Dict[str, Any]:
    """Get performance page layout."""
    return {
        'name': 'performance',
        'title': 'Model Performance',
        'sections': [
            {
                'name': 'model_selector',
                'type': 'dropdown',
                'title': 'Select Model'
            },
            {
                'name': 'metrics_summary',
                'type': 'metrics_grid',
                'metrics': [
                    'accuracy', 'precision', 'recall', 'f1_score',
                    'auc_roc', 'auc_pr', 'ks_statistic', 'gini'
                ],
                'columns': 4
            },
            {
                'name': 'roc_curve',
                'type': 'chart',
                'chart_type': 'roc',
                'title': 'ROC Curve',
                'tab': 'ROC Curve'
            },
            {
                'name': 'confusion_matrix',
                'type': 'chart',
                'chart_type': 'heatmap',
                'title': 'Confusion Matrix',
                'tab': 'Confusion Matrix'
            },
            {
                'name': 'lift_chart',
                'type': 'chart',
                'chart_type': 'bar',
                'title': 'Lift Chart',
                'tab': 'Lift Chart'
            },
            {
                'name': 'calibration',
                'type': 'chart',
                'chart_type': 'line',
                'title': 'Calibration Curve',
                'tab': 'Calibration'
            }
        ]
    }


def _get_predictions_layout() -> Dict[str, Any]:
    """Get predictions page layout."""
    return {
        'name': 'predictions',
        'title': 'Predictions',
        'sections': [
            {
                'name': 'single_prediction',
                'type': 'form',
                'title': 'Single Prediction',
                'tab': 'Single',
                'fields': [
                    {'name': 'loan_amount', 'type': 'number'},
                    {'name': 'annual_income', 'type': 'number'},
                    {'name': 'credit_score', 'type': 'slider', 'min': 300, 'max': 850},
                    {'name': 'debt_to_income', 'type': 'slider', 'min': 0, 'max': 1},
                    {'name': 'employment_length', 'type': 'slider', 'min': 0, 'max': 30},
                    {'name': 'home_ownership', 'type': 'select', 'options': ['RENT', 'OWN', 'MORTGAGE']}
                ]
            },
            {
                'name': 'prediction_result',
                'type': 'result_card',
                'title': 'Prediction Result'
            },
            {
                'name': 'explanation',
                'type': 'shap_plot',
                'title': 'Prediction Explanation'
            },
            {
                'name': 'batch_prediction',
                'type': 'file_upload',
                'title': 'Batch Predictions',
                'tab': 'Batch',
                'accept': '.csv'
            }
        ]
    }


def _get_monitoring_layout() -> Dict[str, Any]:
    """Get monitoring page layout."""
    return {
        'name': 'monitoring',
        'title': 'Model Monitoring',
        'sections': [
            {
                'name': 'alerts',
                'type': 'alerts_table',
                'title': 'Active Alerts',
                'width': 'full'
            },
            {
                'name': 'data_drift',
                'type': 'drift_indicators',
                'title': 'Data Drift',
                'width': 'half'
            },
            {
                'name': 'performance_drift',
                'type': 'metrics_trend',
                'title': 'Performance Drift',
                'width': 'half'
            },
            {
                'name': 'feature_distributions',
                'type': 'distribution_comparison',
                'title': 'Feature Distributions',
                'width': 'full'
            }
        ]
    }


def _get_comparison_layout() -> Dict[str, Any]:
    """Get model comparison page layout."""
    return {
        'name': 'comparison',
        'title': 'Model Comparison',
        'sections': [
            {
                'name': 'model_selectors',
                'type': 'multi_select',
                'title': 'Select Models to Compare',
                'columns': 2
            },
            {
                'name': 'metrics_comparison',
                'type': 'comparison_table',
                'title': 'Performance Comparison',
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'ks_statistic']
            },
            {
                'name': 'roc_comparison',
                'type': 'chart',
                'chart_type': 'multi_roc',
                'title': 'ROC Curve Comparison'
            },
            {
                'name': 'statistical_test',
                'type': 'statistical_comparison',
                'title': 'Statistical Significance'
            }
        ]
    }


def _get_explainability_layout() -> Dict[str, Any]:
    """Get explainability page layout."""
    return {
        'name': 'explainability',
        'title': 'Model Explainability',
        'sections': [
            {
                'name': 'global_importance',
                'type': 'chart',
                'chart_type': 'bar',
                'title': 'Global Feature Importance',
                'tab': 'Global'
            },
            {
                'name': 'shap_summary',
                'type': 'shap_summary',
                'title': 'SHAP Summary Plot',
                'tab': 'SHAP'
            },
            {
                'name': 'partial_dependence',
                'type': 'pdp_plot',
                'title': 'Partial Dependence Plots',
                'tab': 'PDP'
            },
            {
                'name': 'local_explanation',
                'type': 'lime_explanation',
                'title': 'Local Explanation (LIME)',
                'tab': 'LIME'
            }
        ]
    }


if __name__ == "__main__":
    print("Dashboard Layouts Module")
    print("=" * 50)
    print("Available layouts:")
    for layout in AVAILABLE_LAYOUTS:
        print(f"  - {layout}")
    print()
    print("Usage:")
    print("  from src.dashboard.layouts import get_layout")
    print("  layout = get_layout('overview')")
