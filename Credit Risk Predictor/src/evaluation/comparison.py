"""
Model Comparison Module

This module provides utilities for comparing multiple models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = 'classification'
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.
    
    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary mapping model names to fitted model objects
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target
    task_type : str, default='classification'
        Type of task ('classification' or 'regression')
        
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each model
        
    Examples
    --------
    >>> models = {
    ...     'Logistic Regression': lr_model,
    ...     'Random Forest': rf_model,
    ...     'XGBoost': xgb_model
    ... }
    >>> comparison = compare_models(models, X_test, y_test)
    >>> print(comparison)
    """
    from .metrics import calculate_classification_metrics, calculate_regression_metrics
    
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            # Get probabilities if available
            y_prob = None
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
            
            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
        else:
            metrics = calculate_regression_metrics(y_test, y_pred)
        
        metrics['model'] = name
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['model'] + [c for c in comparison_df.columns if c != 'model']
    comparison_df = comparison_df[cols]
    
    return comparison_df


def rank_models(
    comparison_df: pd.DataFrame,
    metric: str,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Rank models based on a specific metric.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        Model comparison DataFrame
    metric : str
        Metric to rank by
    ascending : bool, default=False
        Whether to sort in ascending order
        
    Returns
    -------
    pd.DataFrame
        Ranked comparison table
        
    Examples
    --------
    >>> ranked = rank_models(comparison_df, metric='auc_roc', ascending=False)
    >>> print(ranked)
    """
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison DataFrame")
    
    ranked_df = comparison_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    ranked_df['rank'] = range(1, len(ranked_df) + 1)
    
    # Reorder columns
    cols = ['rank', 'model'] + [c for c in ranked_df.columns if c not in ['rank', 'model']]
    ranked_df = ranked_df[cols]
    
    return ranked_df


def cross_validate_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'roc_auc'
) -> pd.DataFrame:
    """
    Cross-validate multiple models.
    
    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary mapping model names to model objects (not fitted)
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='roc_auc'
        Scoring metric
        
    Returns
    -------
    pd.DataFrame
        Cross-validation results for each model
        
    Examples
    --------
    >>> cv_results = cross_validate_models(models, X, y, cv=5, scoring='roc_auc')
    >>> print(cv_results)
    """
    from sklearn.model_selection import cross_val_score
    
    results = []
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results.append({
            'model': name,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max(),
            'scores': scores.tolist()
        })
    
    return pd.DataFrame(results).sort_values('mean_score', ascending=False)


def statistical_comparison(
    model_a_scores: np.ndarray,
    model_b_scores: np.ndarray,
    test: str = 't_test'
) -> Dict[str, float]:
    """
    Perform statistical comparison between two models.
    
    Parameters
    ----------
    model_a_scores : np.ndarray
        CV scores for model A
    model_b_scores : np.ndarray
        CV scores for model B
    test : str, default='t_test'
        Statistical test to use ('t_test' or 'wilcoxon')
        
    Returns
    -------
    Dict[str, float]
        Test statistic and p-value
        
    Examples
    --------
    >>> result = statistical_comparison(rf_scores, xgb_scores)
    >>> if result['p_value'] < 0.05:
    ...     print("Significant difference between models")
    """
    from scipy import stats
    
    if test == 't_test':
        statistic, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
    elif test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(model_a_scores, model_b_scores)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {
        'test': test,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant_at_05': p_value < 0.05,
        'significant_at_01': p_value < 0.01
    }


def generate_comparison_report(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive model comparison report.
    
    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of fitted models
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target
    output_path : str, optional
        Path to save the report
        
    Returns
    -------
    str
        Report text
    """
    comparison = compare_models(models, X_test, y_test, task_type='classification')
    
    report_lines = [
        "=" * 60,
        "MODEL COMPARISON REPORT",
        "=" * 60,
        "",
        f"Number of models compared: {len(models)}",
        f"Test set size: {len(y_test)}",
        f"Positive class rate: {y_test.mean():.2%}",
        "",
        "-" * 40,
        "PERFORMANCE METRICS",
        "-" * 40,
        ""
    ]
    
    # Add metrics table
    metrics_to_show = ['model', 'accuracy', 'precision', 'recall', 'f1_score']
    if 'auc_roc' in comparison.columns:
        metrics_to_show.append('auc_roc')
    if 'ks_statistic' in comparison.columns:
        metrics_to_show.append('ks_statistic')
    
    available_metrics = [m for m in metrics_to_show if m in comparison.columns]
    report_lines.append(comparison[available_metrics].to_string(index=False))
    
    report_lines.extend([
        "",
        "-" * 40,
        "BEST MODEL BY METRIC",
        "-" * 40,
        ""
    ])
    
    # Find best model for each metric
    for metric in available_metrics[1:]:  # Skip 'model'
        best_idx = comparison[metric].idxmax()
        best_model = comparison.loc[best_idx, 'model']
        best_value = comparison.loc[best_idx, metric]
        report_lines.append(f"  {metric}: {best_model} ({best_value:.4f})")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


if __name__ == "__main__":
    print("Model Comparison Module")
    print("=" * 50)
    print("Functions:")
    print("  - compare_models(models, X_test, y_test)")
    print("  - rank_models(comparison_df, metric)")
    print("  - cross_validate_models(models, X, y)")
    print("  - statistical_comparison(scores_a, scores_b)")
    print("  - generate_comparison_report(models, X_test, y_test)")
