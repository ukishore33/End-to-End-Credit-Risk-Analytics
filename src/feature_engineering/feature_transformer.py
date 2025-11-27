"""
Feature Transformer Module

This module provides feature transformation utilities
for credit risk data preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path


def transform_features(
    df: pd.DataFrame,
    config: Optional[Union[str, Dict]] = None,
    transformations: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Apply feature transformations based on configuration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    config : str or Dict, optional
        Path to configuration file or configuration dictionary
    transformations : Dict[str, str], optional
        Dictionary mapping column names to transformation types
        
    Returns
    -------
    pd.DataFrame
        Transformed DataFrame
        
    Examples
    --------
    >>> transformations = {
    ...     'loan_amount': 'log',
    ...     'income': 'standard',
    ...     'age': 'minmax'
    ... }
    >>> df_transformed = transform_features(df, transformations=transformations)
    """
    df_transformed = df.copy()
    
    if transformations is None and config is None:
        return df_transformed
    
    if config is not None:
        # TODO: Load configuration from file
        # if isinstance(config, str):
        #     import yaml
        #     with open(config) as f:
        #         transformations = yaml.safe_load(f)
        pass
    
    if transformations:
        df_transformed = apply_transformations(df_transformed, transformations)
    
    return df_transformed


def apply_transformations(
    df: pd.DataFrame,
    transformations: Dict[str, str]
) -> pd.DataFrame:
    """
    Apply specified transformations to columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    transformations : Dict[str, str]
        Mapping of column names to transformation types
        Supported types: 'log', 'log1p', 'sqrt', 'standard', 'minmax', 'robust'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with transformed columns
    """
    df_transformed = df.copy()
    
    for column, transform_type in transformations.items():
        if column not in df.columns:
            continue
            
        col_data = df[column].copy()
        
        if transform_type == 'log':
            # Handle zeros and negative values
            min_val = col_data[col_data > 0].min() if (col_data > 0).any() else 1
            df_transformed[column] = np.log(col_data.clip(lower=min_val))
            
        elif transform_type == 'log1p':
            df_transformed[column] = np.log1p(col_data.clip(lower=0))
            
        elif transform_type == 'sqrt':
            df_transformed[column] = np.sqrt(col_data.clip(lower=0))
            
        elif transform_type == 'standard':
            mean = col_data.mean()
            std = col_data.std()
            df_transformed[column] = (col_data - mean) / std if std > 0 else 0
            
        elif transform_type == 'minmax':
            min_val = col_data.min()
            max_val = col_data.max()
            range_val = max_val - min_val
            df_transformed[column] = (col_data - min_val) / range_val if range_val > 0 else 0
            
        elif transform_type == 'robust':
            median = col_data.median()
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            df_transformed[column] = (col_data - median) / iqr if iqr > 0 else 0
            
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")
    
    return df_transformed


def encode_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'onehot',
    target_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Encode categorical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : List[str], optional
        Columns to encode. If None, encodes all object/category columns
    method : str, default='onehot'
        Encoding method ('onehot', 'label', 'target', 'ordinal')
    target_column : str, optional
        Target column for target encoding
        
    Returns
    -------
    pd.DataFrame
        DataFrame with encoded columns
        
    Examples
    --------
    >>> df_encoded = encode_categorical(df, method='onehot')
    >>> df_encoded = encode_categorical(df, method='target', target_column='default')
    """
    df_encoded = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
        
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
    elif method == 'target':
        if target_column is None:
            raise ValueError("target_column required for target encoding")
        # TODO: Implement target encoding
        # For each category, replace with mean of target
        for col in columns:
            means = df.groupby(col)[target_column].mean()
            df_encoded[col] = df_encoded[col].map(means)
            
    elif method == 'ordinal':
        # TODO: Implement ordinal encoding with custom order
        pass
        
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return df_encoded


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'median',
    columns: Optional[List[str]] = None,
    fill_value: Optional[Any] = None
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str, default='median'
        Strategy for handling missing values
        Options: 'mean', 'median', 'mode', 'constant', 'drop'
    columns : List[str], optional
        Columns to process. If None, processes all columns with missing values
    fill_value : Any, optional
        Value to use for 'constant' strategy
        
    Returns
    -------
    pd.DataFrame
        DataFrame with handled missing values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        elif strategy == 'median':
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        elif strategy == 'mode':
            df_filled[col] = df_filled[col].fillna(df_filled[col].mode().iloc[0])
        elif strategy == 'constant':
            df_filled[col] = df_filled[col].fillna(fill_value)
        elif strategy == 'drop':
            df_filled = df_filled.dropna(subset=[col])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_filled


if __name__ == "__main__":
    # Example usage
    print("Feature Transformer Module")
    print("=" * 50)
    print("Usage:")
    print("  from src.feature_engineering.feature_transformer import transform_features")
    print("  df_transformed = transform_features(df, transformations={'income': 'log'})")
