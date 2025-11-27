"""
Evaluation Visualizations Module

This module provides visualization utilities for model evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot ROC curve with AUC.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
    model_name : str, default='Model'
        Name for the legend
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
        
    Examples
    --------
    >>> plot_roc_curve(y_test, probabilities, 'XGBoost', 'reports/figures/roc.png')
    """
    # TODO: Implement with matplotlib
    # from sklearn.metrics import roc_curve, roc_auc_score
    # import matplotlib.pyplot as plt
    # 
    # fpr, tpr, _ = roc_curve(y_true, y_prob)
    # auc = roc_auc_score(y_true, y_prob)
    # 
    # plt.figure(figsize=figsize)
    # plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
    # plt.plot([0, 1], [0, 1], 'k--', label='Random')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend()
    # 
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("TODO: Implement ROC curve visualization")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : List[str], optional
        Names for classes
    normalize : bool, default=False
        Whether to normalize values
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
        
    Examples
    --------
    >>> plot_confusion_matrix(y_test, predictions, ['No Default', 'Default'])
    """
    # TODO: Implement with matplotlib/seaborn
    # from sklearn.metrics import confusion_matrix
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # 
    # cm = confusion_matrix(y_true, y_pred)
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 
    # if class_names is None:
    #     class_names = ['Class 0', 'Class 1']
    # 
    # plt.figure(figsize=figsize)
    # sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
    #             xticklabels=class_names, yticklabels=class_names,
    #             cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix')
    # 
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("TODO: Implement confusion matrix visualization")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot Precision-Recall curve with AUC-PR.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
    model_name : str, default='Model'
        Name for the legend
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
    """
    # TODO: Implement with matplotlib
    # from sklearn.metrics import precision_recall_curve, average_precision_score
    # import matplotlib.pyplot as plt
    # 
    # precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # ap = average_precision_score(y_true, y_prob)
    # 
    # plt.figure(figsize=figsize)
    # plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.4f})')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend()
    # 
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("TODO: Implement Precision-Recall curve visualization")


def plot_lift_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot lift and cumulative gains curves.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
    n_bins : int, default=10
        Number of bins (deciles)
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    """
    # TODO: Implement lift curve visualization
    # import matplotlib.pyplot as plt
    # from src.evaluation.metrics import calculate_lift
    # 
    # lift_df = calculate_lift(y_true, y_prob, n_bins)
    # 
    # fig, axes = plt.subplots(1, 2, figsize=figsize)
    # 
    # # Lift curve
    # axes[0].bar(lift_df.index, lift_df['lift'])
    # axes[0].axhline(y=1, color='r', linestyle='--')
    # axes[0].set_xlabel('Decile')
    # axes[0].set_ylabel('Lift')
    # axes[0].set_title('Lift Chart')
    # 
    # # Cumulative gains
    # axes[1].plot(lift_df.index, lift_df['capture_rate'], marker='o')
    # axes[1].plot([1, n_bins], [0.1, 1], 'r--', label='Random')
    # axes[1].set_xlabel('Decile')
    # axes[1].set_ylabel('Cumulative Capture Rate')
    # axes[1].set_title('Cumulative Gains Chart')
    # axes[1].legend()
    # 
    # plt.tight_layout()
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("TODO: Implement lift curve visualization")


def plot_ks_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot KS (Kolmogorov-Smirnov) curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
    """
    # TODO: Implement KS curve visualization
    # from sklearn.metrics import roc_curve
    # import matplotlib.pyplot as plt
    # 
    # fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # ks = tpr - fpr
    # ks_max_idx = np.argmax(ks)
    # 
    # plt.figure(figsize=figsize)
    # plt.plot(thresholds, tpr[:-1] if len(tpr) > len(thresholds) else tpr, label='TPR')
    # plt.plot(thresholds, fpr[:-1] if len(fpr) > len(thresholds) else fpr, label='FPR')
    # plt.axvline(x=thresholds[ks_max_idx], color='r', linestyle='--',
    #             label=f'KS = {ks[ks_max_idx]:.4f}')
    # plt.xlabel('Threshold')
    # plt.ylabel('Rate')
    # plt.title('KS Curve')
    # plt.legend()
    # 
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("TODO: Implement KS curve visualization")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    model_name: str = 'Model',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot calibration curve (reliability diagram).
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
    n_bins : int, default=10
        Number of bins
    model_name : str, default='Model'
        Model name for legend
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
    """
    # TODO: Implement calibration curve visualization
    # from sklearn.calibration import calibration_curve
    # import matplotlib.pyplot as plt
    # 
    # prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    # 
    # plt.figure(figsize=figsize)
    # plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    # plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    # plt.xlabel('Mean predicted probability')
    # plt.ylabel('Fraction of positives')
    # plt.title('Calibration Curve')
    # plt.legend()
    # 
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("TODO: Implement calibration curve visualization")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot feature importance.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int, default=20
        Number of top features to show
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    """
    # TODO: Implement feature importance visualization
    # import matplotlib.pyplot as plt
    # 
    # df = importance_df.head(top_n).sort_values('importance')
    # 
    # plt.figure(figsize=figsize)
    # plt.barh(df['feature'], df['importance'])
    # plt.xlabel('Importance')
    # plt.title(f'Top {top_n} Feature Importance')
    # 
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("TODO: Implement feature importance visualization")


if __name__ == "__main__":
    print("Evaluation Visualizations Module")
    print("=" * 50)
    print("Functions:")
    print("  - plot_roc_curve(y_true, y_prob)")
    print("  - plot_confusion_matrix(y_true, y_pred)")
    print("  - plot_precision_recall_curve(y_true, y_prob)")
    print("  - plot_lift_curve(y_true, y_prob)")
    print("  - plot_ks_curve(y_true, y_prob)")
    print("  - plot_calibration_curve(y_true, y_prob)")
