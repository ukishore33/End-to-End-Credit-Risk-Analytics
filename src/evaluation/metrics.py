"""
Metrics Module

This module provides evaluation metrics for credit risk models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels
    y_prob : np.ndarray, optional
        Predicted probabilities for positive class
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and values
        
    Examples
    --------
    >>> metrics = calculate_classification_metrics(y_test, predictions, probabilities)
    >>> print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
        matthews_corrcoef,
        balanced_accuracy_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
    }
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Probability-based metrics
    if y_prob is not None:
        probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        metrics['auc_roc'] = roc_auc_score(y_true, probs)
        metrics['auc_pr'] = average_precision_score(y_true, probs)
        
        # Credit-specific metrics
        credit_metrics = calculate_credit_metrics(y_true, probs)
        metrics.update(credit_metrics)
    
    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and values
        
    Examples
    --------
    >>> metrics = calculate_regression_metrics(y_test, predictions)
    >>> print(f"RMSE: {metrics['rmse']:.4f}")
    """
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        mean_absolute_percentage_error,
        explained_variance_score
    )
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred, squared=False),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }
    
    # MAPE (handle division by zero)
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
    except Exception:
        metrics['mape'] = None
    
    # Additional metrics
    residuals = y_true - y_pred
    metrics['max_error'] = float(np.max(np.abs(residuals)))
    metrics['median_ae'] = float(np.median(np.abs(residuals)))
    
    return metrics


def calculate_credit_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Calculate credit-specific metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0=no default, 1=default)
    y_prob : np.ndarray
        Predicted probabilities for positive class
        
    Returns
    -------
    Dict[str, float]
        Dictionary of credit metrics
        
    Examples
    --------
    >>> credit_metrics = calculate_credit_metrics(y_true, probabilities)
    >>> print(f"KS Statistic: {credit_metrics['ks_statistic']:.4f}")
    >>> print(f"Gini Coefficient: {credit_metrics['gini']:.4f}")
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    metrics = {}
    
    # KS Statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_values = tpr - fpr
    ks_idx = np.argmax(ks_values)
    
    metrics['ks_statistic'] = float(ks_values[ks_idx])
    metrics['ks_threshold'] = float(thresholds[ks_idx])
    
    # Gini Coefficient
    auc = roc_auc_score(y_true, y_prob)
    metrics['gini'] = 2 * auc - 1
    
    # Capture rates at different thresholds
    metrics['capture_rate_10pct'] = _calculate_capture_rate(y_true, y_prob, 0.10)
    metrics['capture_rate_20pct'] = _calculate_capture_rate(y_true, y_prob, 0.20)
    
    return metrics


def _calculate_capture_rate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    percentile: float
) -> float:
    """
    Calculate capture rate at given percentile.
    
    The capture rate is the percentage of actual positives captured
    in the top percentile of predictions.
    """
    n_samples = int(len(y_prob) * percentile)
    top_indices = np.argsort(y_prob)[-n_samples:]
    
    captured = y_true[top_indices].sum()
    total_positives = y_true.sum()
    
    if total_positives == 0:
        return 0.0
    
    return float(captured / total_positives)


def calculate_lift(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Calculate lift at different deciles.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
    n_bins : int, default=10
        Number of bins (deciles by default)
        
    Returns
    -------
    pd.DataFrame
        Lift table with cumulative metrics
        
    Examples
    --------
    >>> lift_df = calculate_lift(y_true, probabilities)
    >>> print(lift_df)
    """
    df = pd.DataFrame({'actual': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    
    # Create bins
    df['bin'] = pd.qcut(range(len(df)), n_bins, labels=False) + 1
    
    # Aggregate by bin
    lift_table = df.groupby('bin').agg({
        'actual': ['count', 'sum', 'mean']
    }).round(4)
    
    lift_table.columns = ['count', 'positives', 'positive_rate']
    
    # Calculate cumulative metrics
    total_positives = y_true.sum()
    base_rate = y_true.mean()
    
    lift_table['cum_count'] = lift_table['count'].cumsum()
    lift_table['cum_positives'] = lift_table['positives'].cumsum()
    lift_table['cum_positive_rate'] = lift_table['cum_positives'] / lift_table['cum_count']
    lift_table['capture_rate'] = lift_table['cum_positives'] / total_positives
    lift_table['lift'] = lift_table['cum_positive_rate'] / base_rate
    
    return lift_table


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures the shift in population distributions between
    training and scoring/production data.
    
    Parameters
    ----------
    expected : np.ndarray
        Expected distribution (training data scores)
    actual : np.ndarray
        Actual distribution (scoring data scores)
    n_bins : int, default=10
        Number of bins
        
    Returns
    -------
    Tuple[float, pd.DataFrame]
        (PSI value, detailed breakdown by bin)
        
    Interpretation
    --------------
    - PSI < 0.10: No significant shift
    - 0.10 <= PSI < 0.25: Moderate shift
    - PSI >= 0.25: Significant shift
        
    Examples
    --------
    >>> psi, breakdown = calculate_psi(train_scores, test_scores)
    >>> print(f"PSI: {psi:.4f}")
    """
    # Create bins based on expected distribution
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # Calculate distributions
    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]
    
    # Calculate percentages (with small epsilon to avoid division by zero)
    epsilon = 0.0001
    expected_pct = (expected_counts + epsilon) / (expected_counts.sum() + epsilon * n_bins)
    actual_pct = (actual_counts + epsilon) / (actual_counts.sum() + epsilon * n_bins)
    
    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = float(psi_values.sum())
    
    # Create breakdown DataFrame
    breakdown = pd.DataFrame({
        'bin': range(1, n_bins + 1),
        'expected_count': expected_counts,
        'actual_count': actual_counts,
        'expected_pct': expected_pct,
        'actual_pct': actual_pct,
        'psi_bin': psi_values
    })
    
    return psi, breakdown


if __name__ == "__main__":
    print("Metrics Module")
    print("=" * 50)
    print("Functions:")
    print("  - calculate_classification_metrics(y_true, y_pred, y_prob)")
    print("  - calculate_regression_metrics(y_true, y_pred)")
    print("  - calculate_credit_metrics(y_true, y_prob)")
    print("  - calculate_lift(y_true, y_prob)")
    print("  - calculate_psi(expected, actual)")
