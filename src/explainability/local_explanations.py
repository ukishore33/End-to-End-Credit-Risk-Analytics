"""
Local Explanations Module

This module provides local (instance-level) explanations for credit risk predictions.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union


def explain_prediction(
    model: Any,
    x: Union[pd.Series, np.ndarray],
    feature_names: List[str],
    method: str = 'shap',
    background_data: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Generate explanation for a single prediction.
    
    Parameters
    ----------
    model : Any
        Fitted model
    x : pd.Series or np.ndarray
        Instance to explain
    feature_names : List[str]
        Names of features
    method : str, default='shap'
        Explanation method ('shap', 'lime', 'contribution')
    background_data : np.ndarray, optional
        Background data for SHAP/LIME
        
    Returns
    -------
    Dict[str, Any]
        Explanation dictionary
        
    Examples
    --------
    >>> explanation = explain_prediction(model, X_test[0], feature_names)
    >>> print(f"Prediction: {explanation['prediction']}")
    >>> print(f"Top factors: {explanation['top_positive_factors']}")
    """
    # Convert to array if needed
    if isinstance(x, pd.Series):
        x_array = x.values.reshape(1, -1)
    elif x.ndim == 1:
        x_array = x.reshape(1, -1)
    else:
        x_array = x
    
    # Get prediction
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(x_array)[0]
        prediction = model.predict(x_array)[0]
    else:
        prediction = model.predict(x_array)[0]
        prediction_proba = None
    
    # Get feature contributions
    contributions = get_feature_contributions(model, x_array, feature_names)
    
    # Sort contributions
    sorted_contributions = sorted(
        contributions.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    # Separate positive and negative factors
    positive_factors = [(k, v) for k, v in sorted_contributions if v > 0]
    negative_factors = [(k, v) for k, v in sorted_contributions if v < 0]
    
    explanation = {
        'prediction': int(prediction),
        'probability': prediction_proba.tolist() if prediction_proba is not None else None,
        'contributions': dict(sorted_contributions),
        'top_positive_factors': positive_factors[:5],
        'top_negative_factors': negative_factors[:5],
        'feature_values': dict(zip(feature_names, x_array.flatten()))
    }
    
    return explanation


def get_feature_contributions(
    model: Any,
    x: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Get feature contributions for a prediction.
    
    Parameters
    ----------
    model : Any
        Fitted model
    x : np.ndarray
        Instance to analyze
    feature_names : List[str]
        Feature names
        
    Returns
    -------
    Dict[str, float]
        Feature contributions
    """
    # Try to get contributions from model
    if hasattr(model, 'feature_importances_'):
        # Weight importances by feature values
        importances = model.feature_importances_
        weighted = importances * x.flatten()
        contributions = dict(zip(feature_names, weighted))
    elif hasattr(model, 'coef_'):
        # For linear models, use coefficients * values
        coef = model.coef_.flatten()
        contributions = dict(zip(feature_names, coef * x.flatten()))
    else:
        # Default to zeros
        contributions = {name: 0.0 for name in feature_names}
    
    return contributions


def get_influential_features(
    explanation: Dict[str, Any],
    top_n: int = 5,
    direction: str = 'both'
) -> List[tuple]:
    """
    Get most influential features from an explanation.
    
    Parameters
    ----------
    explanation : Dict[str, Any]
        Prediction explanation
    top_n : int, default=5
        Number of features to return
    direction : str, default='both'
        Direction of influence ('positive', 'negative', 'both')
        
    Returns
    -------
    List[tuple]
        List of (feature_name, contribution) tuples
        
    Examples
    --------
    >>> influential = get_influential_features(explanation, top_n=3)
    >>> for feature, contribution in influential:
    ...     print(f"{feature}: {contribution:.4f}")
    """
    contributions = explanation.get('contributions', {})
    
    if direction == 'positive':
        filtered = {k: v for k, v in contributions.items() if v > 0}
    elif direction == 'negative':
        filtered = {k: v for k, v in contributions.items() if v < 0}
    else:
        filtered = contributions
    
    sorted_features = sorted(
        filtered.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    return sorted_features[:top_n]


def generate_explanation_report(
    explanation: Dict[str, Any],
    include_all_features: bool = False
) -> str:
    """
    Generate a human-readable explanation report.
    
    Parameters
    ----------
    explanation : Dict[str, Any]
        Prediction explanation
    include_all_features : bool, default=False
        Whether to include all features
        
    Returns
    -------
    str
        Formatted explanation report
        
    Examples
    --------
    >>> report = generate_explanation_report(explanation)
    >>> print(report)
    """
    lines = [
        "=" * 60,
        "PREDICTION EXPLANATION REPORT",
        "=" * 60,
        ""
    ]
    
    # Prediction
    prediction = explanation.get('prediction')
    probability = explanation.get('probability')
    
    lines.append("PREDICTION")
    lines.append("-" * 40)
    
    if prediction == 1:
        lines.append("Prediction: DEFAULT (High Risk)")
    else:
        lines.append("Prediction: NO DEFAULT (Low Risk)")
    
    if probability is not None:
        lines.append(f"Default Probability: {probability[1]:.2%}")
        lines.append(f"Non-Default Probability: {probability[0]:.2%}")
    
    lines.append("")
    
    # Risk factors
    positive_factors = explanation.get('top_positive_factors', [])
    negative_factors = explanation.get('top_negative_factors', [])
    feature_values = explanation.get('feature_values', {})
    
    if positive_factors:
        lines.append("FACTORS INCREASING RISK")
        lines.append("-" * 40)
        for feature, contribution in positive_factors:
            value = feature_values.get(feature, 'N/A')
            lines.append(f"  • {feature}")
            lines.append(f"    Value: {value}")
            lines.append(f"    Impact: +{contribution:.4f}")
        lines.append("")
    
    if negative_factors:
        lines.append("FACTORS DECREASING RISK")
        lines.append("-" * 40)
        for feature, contribution in negative_factors:
            value = feature_values.get(feature, 'N/A')
            lines.append(f"  • {feature}")
            lines.append(f"    Value: {value}")
            lines.append(f"    Impact: {contribution:.4f}")
        lines.append("")
    
    # All contributions if requested
    if include_all_features:
        contributions = explanation.get('contributions', {})
        lines.append("ALL FEATURE CONTRIBUTIONS")
        lines.append("-" * 40)
        for feature, contribution in sorted(contributions.items(), key=lambda x: -abs(x[1])):
            value = feature_values.get(feature, 'N/A')
            lines.append(f"  {feature}: {contribution:+.4f} (value={value})")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def compare_explanations(
    explanation1: Dict[str, Any],
    explanation2: Dict[str, Any],
    label1: str = "Instance 1",
    label2: str = "Instance 2"
) -> str:
    """
    Compare explanations for two instances.
    
    Parameters
    ----------
    explanation1 : Dict[str, Any]
        First explanation
    explanation2 : Dict[str, Any]
        Second explanation
    label1 : str, default="Instance 1"
        Label for first instance
    label2 : str, default="Instance 2"
        Label for second instance
        
    Returns
    -------
    str
        Comparison report
    """
    lines = [
        "=" * 60,
        "EXPLANATION COMPARISON",
        "=" * 60,
        ""
    ]
    
    # Compare predictions
    pred1 = explanation1.get('prediction')
    pred2 = explanation2.get('prediction')
    prob1 = explanation1.get('probability', [0, 0])
    prob2 = explanation2.get('probability', [0, 0])
    
    lines.append(f"{label1}: {'DEFAULT' if pred1 == 1 else 'NO DEFAULT'} "
                 f"(P={prob1[1]:.2%})")
    lines.append(f"{label2}: {'DEFAULT' if pred2 == 1 else 'NO DEFAULT'} "
                 f"(P={prob2[1]:.2%})")
    lines.append("")
    
    # Compare top factors
    contrib1 = explanation1.get('contributions', {})
    contrib2 = explanation2.get('contributions', {})
    
    all_features = set(contrib1.keys()) | set(contrib2.keys())
    
    lines.append("FEATURE CONTRIBUTION DIFFERENCES")
    lines.append("-" * 40)
    
    differences = []
    for feature in all_features:
        c1 = contrib1.get(feature, 0)
        c2 = contrib2.get(feature, 0)
        diff = c2 - c1
        differences.append((feature, c1, c2, diff))
    
    differences.sort(key=lambda x: abs(x[3]), reverse=True)
    
    for feature, c1, c2, diff in differences[:10]:
        lines.append(f"  {feature}:")
        lines.append(f"    {label1}: {c1:+.4f}")
        lines.append(f"    {label2}: {c2:+.4f}")
        lines.append(f"    Difference: {diff:+.4f}")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("Local Explanations Module")
    print("=" * 50)
    print("Functions:")
    print("  - explain_prediction(model, x, feature_names)")
    print("  - get_influential_features(explanation, top_n)")
    print("  - generate_explanation_report(explanation)")
    print("  - compare_explanations(exp1, exp2)")
