"""
EDA Report Module

This module provides functions for generating comprehensive
exploratory data analysis reports for credit risk data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


def generate_eda_report(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive EDA report for the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str, optional
        Name of target column for target-specific analysis
    output_path : str, optional
        Path to save the report
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing EDA results
        
    Examples
    --------
    >>> report = generate_eda_report(df, target_column='default')
    >>> print(report['summary_statistics'])
    """
    report = {}
    
    # Basic information
    report['basic_info'] = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    # Summary statistics
    report['summary_statistics'] = get_summary_statistics(df)
    
    # Missing values analysis
    report['missing_values'] = _analyze_missing_values(df)
    
    # Duplicate analysis
    report['duplicates'] = {
        'num_duplicates': df.duplicated().sum(),
        'duplicate_percentage': df.duplicated().sum() / len(df) * 100
    }
    
    # Column type breakdown
    report['column_types'] = {
        'numeric': list(df.select_dtypes(include=['number']).columns),
        'categorical': list(df.select_dtypes(include=['object', 'category']).columns),
        'datetime': list(df.select_dtypes(include=['datetime']).columns),
        'boolean': list(df.select_dtypes(include=['bool']).columns)
    }
    
    # Target analysis
    if target_column and target_column in df.columns:
        report['target_analysis'] = _analyze_target(df, target_column)
    
    # Correlation analysis for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        report['correlations'] = df[numeric_cols].corr().to_dict()
    
    # TODO: Save report if output_path provided
    if output_path:
        # Save report to file
        pass
    
    return report


def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for all columns in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    Dict[str, Any]
        Summary statistics for numeric and categorical columns
    """
    stats = {}
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        stats['numeric'] = df[numeric_cols].describe().to_dict()
        
        # Additional statistics
        for col in numeric_cols:
            if col not in stats['numeric']:
                stats['numeric'][col] = {}
            stats['numeric'][col]['skewness'] = df[col].skew()
            stats['numeric'][col]['kurtosis'] = df[col].kurtosis()
            stats['numeric'][col]['missing_count'] = df[col].isnull().sum()
    
    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        stats['categorical'] = {}
        for col in categorical_cols:
            stats['categorical'][col] = {
                'unique_count': df[col].nunique(),
                'top_values': df[col].value_counts().head(10).to_dict(),
                'missing_count': df[col].isnull().sum()
            }
    
    return stats


def _analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze missing values in the DataFrame."""
    missing = df.isnull().sum()
    missing_pct = missing / len(df) * 100
    
    return {
        'missing_counts': missing.to_dict(),
        'missing_percentages': missing_pct.to_dict(),
        'columns_with_missing': list(missing[missing > 0].index),
        'total_missing_cells': missing.sum(),
        'total_missing_percentage': missing.sum() / (len(df) * len(df.columns)) * 100
    }


def _analyze_target(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Analyze the target variable."""
    target = df[target_column]
    
    analysis = {
        'dtype': str(target.dtype),
        'unique_values': target.nunique(),
        'missing_count': target.isnull().sum(),
        'value_counts': target.value_counts().to_dict()
    }
    
    # For binary classification
    if target.nunique() == 2:
        analysis['type'] = 'binary'
        value_counts = target.value_counts()
        analysis['class_ratio'] = value_counts.max() / value_counts.min()
        analysis['positive_rate'] = (target == 1).sum() / len(target) * 100
    
    # For continuous target
    elif pd.api.types.is_numeric_dtype(target):
        analysis['type'] = 'continuous'
        analysis['mean'] = target.mean()
        analysis['median'] = target.median()
        analysis['std'] = target.std()
        analysis['min'] = target.min()
        analysis['max'] = target.max()
    
    return analysis


if __name__ == "__main__":
    # Example usage
    print("EDA Report Module")
    print("=" * 50)
    print("Usage:")
    print("  from src.eda.eda_report import generate_eda_report")
    print("  report = generate_eda_report(df, target_column='default')")
