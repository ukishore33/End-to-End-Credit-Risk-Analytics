"""
Dashboard Components Module

This module provides reusable components for the credit risk dashboard.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


def create_metric_card(
    title: str,
    value: float,
    change: Optional[float] = None,
    format_str: str = ".2f",
    prefix: str = "",
    suffix: str = ""
) -> Dict[str, Any]:
    """
    Create a metric card configuration.
    
    Parameters
    ----------
    title : str
        Metric title
    value : float
        Current metric value
    change : float, optional
        Change from previous period
    format_str : str, default=".2f"
        Number format string
    prefix : str, default=""
        Value prefix (e.g., "$")
    suffix : str, default=""
        Value suffix (e.g., "%")
        
    Returns
    -------
    Dict[str, Any]
        Metric card configuration
    """
    formatted_value = f"{prefix}{value:{format_str}}{suffix}"
    
    card = {
        'title': title,
        'value': formatted_value,
        'raw_value': value,
        'change': change
    }
    
    if change is not None:
        card['change_formatted'] = f"{'+' if change > 0 else ''}{change:{format_str}}"
        card['change_direction'] = 'up' if change > 0 else 'down' if change < 0 else 'neutral'
    
    return card


def create_performance_chart(
    metrics_history: pd.DataFrame,
    metrics: List[str] = None,
    title: str = "Model Performance Over Time"
) -> Any:
    """
    Create a performance trend chart.
    
    Parameters
    ----------
    metrics_history : pd.DataFrame
        DataFrame with 'date' and metric columns
    metrics : List[str], optional
        Metrics to plot (default: all)
    title : str, default="Model Performance Over Time"
        Chart title
        
    Returns
    -------
    Any
        Chart object (plotly figure)
    """
    # TODO: Implement with plotly
    # import plotly.express as px
    # import plotly.graph_objects as go
    # 
    # if metrics is None:
    #     metrics = [col for col in metrics_history.columns if col != 'date']
    # 
    # fig = go.Figure()
    # 
    # for metric in metrics:
    #     fig.add_trace(go.Scatter(
    #         x=metrics_history['date'],
    #         y=metrics_history[metric],
    #         mode='lines+markers',
    #         name=metric
    #     ))
    # 
    # fig.update_layout(
    #     title=title,
    #     xaxis_title='Date',
    #     yaxis_title='Value',
    #     legend_title='Metrics'
    # )
    # 
    # return fig
    
    print(f"TODO: Create performance chart - {title}")
    return None


def create_feature_importance_plot(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importance"
) -> Any:
    """
    Create a feature importance bar chart.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int, default=15
        Number of top features to show
    title : str, default="Feature Importance"
        Chart title
        
    Returns
    -------
    Any
        Chart object (plotly figure)
    """
    # TODO: Implement with plotly
    # import plotly.express as px
    # 
    # df = importance_df.head(top_n).sort_values('importance')
    # 
    # fig = px.bar(
    #     df,
    #     x='importance',
    #     y='feature',
    #     orientation='h',
    #     title=title
    # )
    # 
    # fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    # 
    # return fig
    
    print(f"TODO: Create feature importance plot - {title}")
    return None


def create_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    title: str = "Confusion Matrix"
) -> Any:
    """
    Create a confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : List[str], optional
        Names for classes
    title : str, default="Confusion Matrix"
        Chart title
        
    Returns
    -------
    Any
        Chart object (plotly figure)
    """
    # TODO: Implement with plotly
    # import plotly.figure_factory as ff
    # from sklearn.metrics import confusion_matrix
    # 
    # if class_names is None:
    #     class_names = ['No Default', 'Default']
    # 
    # cm = confusion_matrix(y_true, y_pred)
    # 
    # fig = ff.create_annotated_heatmap(
    #     z=cm,
    #     x=class_names,
    #     y=class_names,
    #     colorscale='Blues'
    # )
    # 
    # fig.update_layout(
    #     title=title,
    #     xaxis_title='Predicted',
    #     yaxis_title='Actual'
    # )
    # 
    # return fig
    
    print(f"TODO: Create confusion matrix plot - {title}")
    return None


def create_roc_curve_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve"
) -> Any:
    """
    Create a ROC curve plot.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
    title : str, default="ROC Curve"
        Chart title
        
    Returns
    -------
    Any
        Chart object (plotly figure)
    """
    # TODO: Implement with plotly
    # import plotly.graph_objects as go
    # from sklearn.metrics import roc_curve, roc_auc_score
    # 
    # fpr, tpr, _ = roc_curve(y_true, y_prob)
    # auc = roc_auc_score(y_true, y_prob)
    # 
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={auc:.4f})'))
    # fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')))
    # 
    # fig.update_layout(
    #     title=title,
    #     xaxis_title='False Positive Rate',
    #     yaxis_title='True Positive Rate'
    # )
    # 
    # return fig
    
    print(f"TODO: Create ROC curve plot - {title}")
    return None


def create_prediction_gauge(
    probability: float,
    title: str = "Default Probability"
) -> Any:
    """
    Create a gauge chart for prediction probability.
    
    Parameters
    ----------
    probability : float
        Prediction probability (0-1)
    title : str, default="Default Probability"
        Chart title
        
    Returns
    -------
    Any
        Chart object (plotly figure)
    """
    # TODO: Implement with plotly
    # import plotly.graph_objects as go
    # 
    # fig = go.Figure(go.Indicator(
    #     mode="gauge+number",
    #     value=probability * 100,
    #     title={'text': title},
    #     gauge={
    #         'axis': {'range': [0, 100]},
    #         'bar': {'color': "darkblue"},
    #         'steps': [
    #             {'range': [0, 30], 'color': "lightgreen"},
    #             {'range': [30, 60], 'color': "yellow"},
    #             {'range': [60, 100], 'color': "red"}
    #         ],
    #         'threshold': {
    #             'line': {'color': "red", 'width': 4},
    #             'thickness': 0.75,
    #             'value': 50
    #         }
    #     }
    # ))
    # 
    # return fig
    
    print(f"TODO: Create prediction gauge - {title}")
    return None


def create_shap_force_plot(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    base_value: float
) -> Any:
    """
    Create a SHAP force plot.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values for prediction
    features : np.ndarray
        Feature values
    feature_names : List[str]
        Feature names
    base_value : float
        Base/expected value
        
    Returns
    -------
    Any
        SHAP force plot
    """
    # TODO: Implement SHAP force plot
    # import shap
    # 
    # return shap.force_plot(
    #     base_value,
    #     shap_values,
    #     features,
    #     feature_names=feature_names
    # )
    
    print("TODO: Create SHAP force plot")
    return None


def create_drift_indicator(
    feature_name: str,
    psi_value: float,
    threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Create a drift indicator component.
    
    Parameters
    ----------
    feature_name : str
        Name of the feature
    psi_value : float
        PSI value for the feature
    threshold : float, default=0.25
        Threshold for significant drift
        
    Returns
    -------
    Dict[str, Any]
        Drift indicator configuration
    """
    if psi_value < 0.10:
        status = "stable"
        color = "green"
        message = "No significant drift"
    elif psi_value < threshold:
        status = "moderate"
        color = "yellow"
        message = "Moderate drift detected"
    else:
        status = "significant"
        color = "red"
        message = "Significant drift detected"
    
    return {
        'feature': feature_name,
        'psi': psi_value,
        'status': status,
        'color': color,
        'message': message,
        'threshold': threshold
    }


if __name__ == "__main__":
    print("Dashboard Components Module")
    print("=" * 50)
    print("Available components:")
    print("  - create_metric_card(title, value)")
    print("  - create_performance_chart(metrics_history)")
    print("  - create_feature_importance_plot(importance_df)")
    print("  - create_confusion_matrix_plot(y_true, y_pred)")
    print("  - create_roc_curve_plot(y_true, y_prob)")
    print("  - create_prediction_gauge(probability)")
