"""
Domain Features Module

This module provides credit risk domain-specific feature engineering functions.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_credit_features(
    df: pd.DataFrame,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Create domain-specific credit risk features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with loan/credit data
    config : dict, optional
        Configuration for feature creation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features
        
    Examples
    --------
    >>> df = create_credit_features(df)
    >>> print(df[['debt_to_income', 'credit_utilization']].head())
    """
    df_features = df.copy()
    
    # Financial Ratio Features
    df_features = calculate_risk_ratios(df_features)
    
    # Credit History Features
    df_features = create_credit_history_features(df_features)
    
    # Behavioral Features
    df_features = create_behavioral_features(df_features)
    
    # Temporal Features
    df_features = create_temporal_features(df_features)
    
    return df_features


def calculate_risk_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common credit risk ratios.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        DataFrame with risk ratio features
        
    Features Created
    ----------------
    - debt_to_income: Total debt divided by income
    - credit_utilization: Credit used divided by credit limit
    - loan_to_income: Loan amount divided by income
    - installment_to_income: Monthly installment divided by monthly income
    """
    df_ratios = df.copy()
    
    # Debt-to-Income Ratio
    if 'total_debt' in df.columns and 'annual_income' in df.columns:
        df_ratios['debt_to_income'] = df['total_debt'] / df['annual_income'].replace(0, np.nan)
    
    # Credit Utilization Ratio
    if 'credit_used' in df.columns and 'credit_limit' in df.columns:
        df_ratios['credit_utilization'] = df['credit_used'] / df['credit_limit'].replace(0, np.nan)
    
    # Loan-to-Income Ratio
    if 'loan_amount' in df.columns and 'annual_income' in df.columns:
        df_ratios['loan_to_income'] = df['loan_amount'] / df['annual_income'].replace(0, np.nan)
    
    # Installment-to-Income Ratio
    if 'installment' in df.columns and 'annual_income' in df.columns:
        monthly_income = df['annual_income'] / 12
        df_ratios['installment_to_income'] = df['installment'] / monthly_income.replace(0, np.nan)
    
    # Payment-to-Income Ratio
    if 'total_payment' in df.columns and 'annual_income' in df.columns:
        df_ratios['payment_to_income'] = df['total_payment'] / df['annual_income'].replace(0, np.nan)
    
    return df_ratios


def create_credit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from credit history data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        DataFrame with credit history features
        
    Features Created
    ----------------
    - avg_account_age: Average age of credit accounts
    - delinquency_rate: Proportion of delinquent accounts
    - credit_mix_score: Diversity of credit types
    """
    df_history = df.copy()
    
    # Average account age
    if 'total_account_age' in df.columns and 'num_accounts' in df.columns:
        df_history['avg_account_age'] = df['total_account_age'] / df['num_accounts'].replace(0, 1)
    
    # Delinquency rate
    if 'num_delinquent_accounts' in df.columns and 'num_accounts' in df.columns:
        df_history['delinquency_rate'] = df['num_delinquent_accounts'] / df['num_accounts'].replace(0, 1)
    
    # Open to total accounts ratio
    if 'num_open_accounts' in df.columns and 'num_accounts' in df.columns:
        df_history['open_account_ratio'] = df['num_open_accounts'] / df['num_accounts'].replace(0, 1)
    
    # Recent inquiries impact
    if 'num_inquiries_6m' in df.columns:
        df_history['high_inquiry_flag'] = (df['num_inquiries_6m'] > 3).astype(int)
    
    return df_history


def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioral features based on payment patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        DataFrame with behavioral features
    """
    df_behavioral = df.copy()
    
    # Payment behavior
    if 'num_late_payments' in df.columns and 'num_total_payments' in df.columns:
        df_behavioral['late_payment_rate'] = (
            df['num_late_payments'] / df['num_total_payments'].replace(0, 1)
        )
    
    # Revolving balance trend
    if 'revolving_balance' in df.columns and 'revolving_limit' in df.columns:
        df_behavioral['revolving_util'] = (
            df['revolving_balance'] / df['revolving_limit'].replace(0, np.nan)
        )
    
    # Number of accounts with balance
    if 'num_accounts_with_balance' in df.columns and 'num_accounts' in df.columns:
        df_behavioral['active_account_ratio'] = (
            df['num_accounts_with_balance'] / df['num_accounts'].replace(0, 1)
        )
    
    return df_behavioral


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from date columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        DataFrame with temporal features
    """
    df_temporal = df.copy()
    
    # Loan term features
    if 'issue_date' in df.columns:
        df['issue_date'] = pd.to_datetime(df['issue_date'])
        df_temporal['issue_month'] = df['issue_date'].dt.month
        df_temporal['issue_year'] = df['issue_date'].dt.year
        df_temporal['issue_quarter'] = df['issue_date'].dt.quarter
        
        # Days since issue
        today = pd.Timestamp.now()
        df_temporal['days_since_issue'] = (today - df['issue_date']).dt.days
    
    # First credit line age
    if 'earliest_credit_line' in df.columns:
        df['earliest_credit_line'] = pd.to_datetime(df['earliest_credit_line'])
        today = pd.Timestamp.now()
        df_temporal['credit_history_months'] = (
            (today - df['earliest_credit_line']).dt.days / 30
        ).astype(int)
    
    # Last payment recency
    if 'last_payment_date' in df.columns:
        df['last_payment_date'] = pd.to_datetime(df['last_payment_date'])
        today = pd.Timestamp.now()
        df_temporal['days_since_last_payment'] = (today - df['last_payment_date']).dt.days
    
    return df_temporal


def create_binned_features(
    df: pd.DataFrame,
    columns: List[str],
    n_bins: int = 5,
    strategy: str = 'quantile'
) -> pd.DataFrame:
    """
    Create binned versions of continuous features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : List[str]
        Columns to bin
    n_bins : int, default=5
        Number of bins
    strategy : str, default='quantile'
        Binning strategy ('quantile', 'uniform', 'kmeans')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with binned features
    """
    df_binned = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'quantile':
            df_binned[f'{col}_bin'] = pd.qcut(
                df[col], q=n_bins, labels=False, duplicates='drop'
            )
        elif strategy == 'uniform':
            df_binned[f'{col}_bin'] = pd.cut(
                df[col], bins=n_bins, labels=False
            )
        # TODO: Add kmeans binning
    
    return df_binned


def create_interaction_features(
    df: pd.DataFrame,
    feature_pairs: List[tuple]
) -> pd.DataFrame:
    """
    Create interaction features from pairs of columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_pairs : List[tuple]
        List of (col1, col2) pairs for interaction
        
    Returns
    -------
    pd.DataFrame
        DataFrame with interaction features
        
    Examples
    --------
    >>> pairs = [('income', 'debt'), ('credit_score', 'utilization')]
    >>> df = create_interaction_features(df, pairs)
    """
    df_interactions = df.copy()
    
    for col1, col2 in feature_pairs:
        if col1 in df.columns and col2 in df.columns:
            # Multiplication interaction
            df_interactions[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            
            # Division interaction (with safety for zeros)
            df_interactions[f'{col1}_div_{col2}'] = df[col1] / df[col2].replace(0, np.nan)
    
    return df_interactions


if __name__ == "__main__":
    # Example usage
    print("Domain Features Module")
    print("=" * 50)
    print("Usage:")
    print("  from src.feature_engineering.domain_features import create_credit_features")
    print("  df = create_credit_features(df)")
