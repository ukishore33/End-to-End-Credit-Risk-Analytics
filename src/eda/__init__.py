"""
Exploratory Data Analysis (EDA) Module

This module provides utilities for exploratory data analysis
of credit risk datasets.
"""

from .eda_report import generate_eda_report, get_summary_statistics
from .visualizations import plot_distributions, plot_correlations, plot_target_analysis
from .statistical_analysis import perform_statistical_tests, check_normality

__all__ = [
    'generate_eda_report',
    'get_summary_statistics',
    'plot_distributions',
    'plot_correlations',
    'plot_target_analysis',
    'perform_statistical_tests',
    'check_normality'
]
