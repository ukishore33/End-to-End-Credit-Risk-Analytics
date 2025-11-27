"""
Data Validators Module

This module provides functions for validating credit risk data
including schema validation and data quality checks.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple


def validate_schema(
    df: pd.DataFrame,
    expected_columns: List[str],
    expected_dtypes: Optional[Dict[str, str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame schema against expected columns and dtypes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    expected_columns : List[str]
        List of expected column names
    expected_dtypes : Dict[str, str], optional
        Dictionary mapping column names to expected dtypes
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list of validation errors)
        
    Examples
    --------
    >>> is_valid, errors = validate_schema(df, ['loan_id', 'amount', 'status'])
    >>> if not is_valid:
    ...     print(f"Validation errors: {errors}")
    """
    errors = []
    
    # Check for missing columns
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
    
    # Check for extra columns
    extra_columns = set(df.columns) - set(expected_columns)
    if extra_columns:
        errors.append(f"Unexpected columns: {extra_columns}")
    
    # Check data types
    if expected_dtypes:
        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if actual_dtype != expected_dtype:
                    errors.append(
                        f"Column '{col}' has dtype '{actual_dtype}', "
                        f"expected '{expected_dtype}'"
                    )
    
    is_valid = len(errors) == 0
    return is_valid, errors


def check_data_quality(
    df: pd.DataFrame,
    checks: Optional[Dict] = None
) -> Dict[str, any]:
    """
    Perform data quality checks on a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    checks : Dict, optional
        Custom checks to perform
        
    Returns
    -------
    Dict
        Quality report with metrics
        
    Examples
    --------
    >>> quality_report = check_data_quality(df)
    >>> print(f"Missing values: {quality_report['missing_values']}")
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'column_dtypes': df.dtypes.astype(str).to_dict()
    }
    
    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        report['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Add categorical column statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        report['categorical_unique_counts'] = {
            col: df[col].nunique() for col in categorical_cols
        }
    
    return report


def check_target_distribution(
    df: pd.DataFrame,
    target_column: str
) -> Dict[str, any]:
    """
    Check the distribution of target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing target column
    target_column : str
        Name of target column
        
    Returns
    -------
    Dict
        Target distribution statistics
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    target = df[target_column]
    
    report = {
        'value_counts': target.value_counts().to_dict(),
        'value_percentages': (target.value_counts(normalize=True) * 100).to_dict(),
        'missing_count': target.isnull().sum(),
        'unique_values': target.nunique()
    }
    
    # Calculate class imbalance ratio for binary classification
    if target.nunique() == 2:
        value_counts = target.value_counts()
        report['imbalance_ratio'] = value_counts.max() / value_counts.min()
    
    return report


if __name__ == "__main__":
    # Example usage
    print("Data Validators Module")
    print("=" * 50)
    print("Usage:")
    print("  from src.data_sourcing.validators import validate_schema, check_data_quality")
    print("  is_valid, errors = validate_schema(df, expected_columns)")
    print("  report = check_data_quality(df)")
