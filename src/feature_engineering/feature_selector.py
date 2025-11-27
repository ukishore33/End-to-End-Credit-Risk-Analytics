"""
Feature Selector Module

This module provides feature selection utilities
for credit risk modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def select_features(
    df: pd.DataFrame,
    target: str,
    method: str = 'mutual_info',
    k: int = 20,
    threshold: Optional[float] = None
) -> List[str]:
    """
    Select top features based on specified method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target : str
        Target column name
    method : str, default='mutual_info'
        Feature selection method
        Options: 'mutual_info', 'chi2', 'f_classif', 'f_regression', 
                 'rfe', 'lasso', 'correlation'
    k : int, default=20
        Number of features to select
    threshold : float, optional
        Threshold for feature selection (alternative to k)
        
    Returns
    -------
    List[str]
        List of selected feature names
        
    Examples
    --------
    >>> selected = select_features(df, 'default', method='mutual_info', k=15)
    >>> print(f"Selected features: {selected}")
    """
    # TODO: Implement with sklearn feature selection
    # from sklearn.feature_selection import SelectKBest, mutual_info_classif
    # from sklearn.feature_selection import chi2, f_classif, RFE
    
    feature_cols = [col for col in df.columns if col != target]
    X = df[feature_cols]
    y = df[target]
    
    # Ensure numeric features only for most methods
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    if method == 'mutual_info':
        # TODO: Implement mutual information
        # from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        # selector = SelectKBest(mutual_info_classif, k=k)
        # selector.fit(X[numeric_cols], y)
        # scores = selector.scores_
        pass
        
    elif method == 'correlation':
        # Select features with highest absolute correlation with target
        correlations = X[numeric_cols].corrwith(y).abs()
        selected = correlations.nlargest(k).index.tolist()
        return selected
        
    elif method == 'chi2':
        # TODO: Implement chi-square selection (for non-negative features)
        pass
        
    elif method == 'f_classif':
        # TODO: Implement ANOVA F-value
        pass
        
    elif method == 'f_regression':
        # TODO: Implement F-regression
        pass
        
    elif method == 'rfe':
        # TODO: Implement Recursive Feature Elimination
        pass
        
    elif method == 'lasso':
        # TODO: Implement Lasso-based selection
        pass
    
    # Default: return first k features
    return numeric_cols[:min(k, len(numeric_cols))]


def get_feature_importance(
    df: pd.DataFrame,
    target: str,
    method: str = 'random_forest'
) -> pd.DataFrame:
    """
    Get feature importance scores using specified method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target : str
        Target column name
    method : str, default='random_forest'
        Method for calculating importance
        Options: 'random_forest', 'xgboost', 'permutation', 'shap'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features and their importance scores
        
    Examples
    --------
    >>> importance_df = get_feature_importance(df, 'default', method='random_forest')
    >>> print(importance_df.head(10))
    """
    # TODO: Implement feature importance calculation
    # from sklearn.ensemble import RandomForestClassifier
    
    feature_cols = [col for col in df.columns if col != target]
    X = df[feature_cols].select_dtypes(include=['number'])
    y = df[target]
    
    if method == 'random_forest':
        # TODO: Train RF and get feature_importances_
        # model = RandomForestClassifier(n_estimators=100, random_state=42)
        # model.fit(X, y)
        # importances = model.feature_importances_
        importances = [None] * len(X.columns)
        
    elif method == 'xgboost':
        # TODO: Train XGBoost and get feature importance
        importances = [None] * len(X.columns)
        
    elif method == 'permutation':
        # TODO: Calculate permutation importance
        importances = [None] * len(X.columns)
        
    elif method == 'shap':
        # TODO: Calculate SHAP values
        importances = [None] * len(X.columns)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    })
    
    if importance_df['importance'].notna().any():
        importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.9,
    method: str = 'pearson'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    threshold : float, default=0.9
        Correlation threshold above which to remove features
    method : str, default='pearson'
        Correlation method
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (DataFrame with reduced features, list of removed features)
        
    Examples
    --------
    >>> df_reduced, removed = remove_correlated_features(df, threshold=0.85)
    >>> print(f"Removed features: {removed}")
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numeric_cols].corr(method=method).abs()
    
    # Create upper triangle mask
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_corr = corr_matrix.where(upper_tri)
    
    # Find features to remove
    to_remove = [col for col in upper_corr.columns 
                 if any(upper_corr[col] > threshold)]
    
    # Keep non-numeric columns and remaining numeric columns
    cols_to_keep = [col for col in df.columns if col not in to_remove]
    df_reduced = df[cols_to_keep].copy()
    
    return df_reduced, to_remove


def remove_low_variance_features(
    df: pd.DataFrame,
    threshold: float = 0.01
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with low variance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    threshold : float, default=0.01
        Variance threshold
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (DataFrame with reduced features, list of removed features)
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    variances = df[numeric_cols].var()
    
    low_var_cols = variances[variances < threshold].index.tolist()
    
    cols_to_keep = [col for col in df.columns if col not in low_var_cols]
    df_reduced = df[cols_to_keep].copy()
    
    return df_reduced, low_var_cols


if __name__ == "__main__":
    # Example usage
    print("Feature Selector Module")
    print("=" * 50)
    print("Usage:")
    print("  from src.feature_engineering.feature_selector import select_features")
    print("  selected = select_features(df, 'default', method='mutual_info', k=20)")
