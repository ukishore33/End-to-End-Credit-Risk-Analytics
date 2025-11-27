"""
Model Utilities Module

This module provides utility functions for model saving, loading, and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union
from pathlib import Path


def save_model(
    model: Any,
    path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save model to disk with optional metadata.
    
    Parameters
    ----------
    model : Any
        Model to save
    path : str
        Path to save the model
    metadata : Dict, optional
        Additional metadata to save
        
    Examples
    --------
    >>> save_model(model, 'models/classifier_v1.joblib', metadata={'version': '1.0'})
    """
    import joblib
    
    save_data = {
        'model': model,
        'metadata': metadata or {}
    }
    
    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(save_data, path)
    print(f"Model saved to {path}")


def load_model(path: str) -> tuple:
    """
    Load model from disk.
    
    Parameters
    ----------
    path : str
        Path to saved model
        
    Returns
    -------
    tuple
        (model, metadata)
        
    Examples
    --------
    >>> model, metadata = load_model('models/classifier_v1.joblib')
    """
    import joblib
    
    save_data = joblib.load(path)
    
    return save_data['model'], save_data.get('metadata', {})


def get_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'classification',
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate model performance metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    task_type : str, default='classification'
        Type of task ('classification' or 'regression')
    y_prob : np.ndarray, optional
        Predicted probabilities (for classification)
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and values
        
    Examples
    --------
    >>> metrics = get_model_metrics(y_test, predictions, task_type='classification')
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    >>> print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    """
    metrics = {}
    
    if task_type == 'classification':
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
            classification_report
        )
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        if y_prob is not None:
            # Use probabilities for class 1
            probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            metrics['auc_roc'] = roc_auc_score(y_true, probs)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = tp
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
    elif task_type == 'regression':
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
            mean_absolute_percentage_error
        )
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = mean_squared_error(y_true, y_pred, squared=False)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except Exception:
            metrics['mape'] = None
        
        # Additional metrics
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['median_absolute_error'] = np.median(np.abs(y_true - y_pred))
        
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
) -> None:
    """
    Print a detailed classification report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    target_names : list, optional
        Names for target classes
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    if target_names is None:
        target_names = ['No Default', 'Default']
    
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    print("\nConfusion Matrix")
    print("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f'Actual {name}' for name in target_names],
        columns=[f'Predicted {name}' for name in target_names]
    )
    print(cm_df)


def calculate_ks_statistic(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Calculate KS (Kolmogorov-Smirnov) statistic.
    
    The KS statistic is commonly used in credit scoring to measure
    the discriminatory power of a model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
        
    Returns
    -------
    Dict[str, float]
        KS statistic and threshold
        
    Examples
    --------
    >>> ks_result = calculate_ks_statistic(y_test, probabilities[:, 1])
    >>> print(f"KS Statistic: {ks_result['ks_statistic']:.4f}")
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # KS statistic is the maximum difference between TPR and FPR
    ks_values = tpr - fpr
    ks_statistic = np.max(ks_values)
    ks_threshold = thresholds[np.argmax(ks_values)]
    
    return {
        'ks_statistic': ks_statistic,
        'ks_threshold': ks_threshold
    }


def calculate_gini_coefficient(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Calculate Gini coefficient from AUC-ROC.
    
    Gini = 2 * AUC - 1
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
        
    Returns
    -------
    float
        Gini coefficient
    """
    from sklearn.metrics import roc_auc_score
    
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    
    return gini


if __name__ == "__main__":
    # Example usage
    print("Model Utilities Module")
    print("=" * 50)
    print("Functions:")
    print("  - save_model(model, path)")
    print("  - load_model(path)")
    print("  - get_model_metrics(y_true, y_pred, task_type)")
    print("  - print_classification_report(y_true, y_pred)")
    print("  - calculate_ks_statistic(y_true, y_prob)")
    print("  - calculate_gini_coefficient(y_true, y_prob)")
