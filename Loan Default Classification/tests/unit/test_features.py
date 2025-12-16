"""Unit tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    create_income_features,
    create_demographic_features,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        'ApplicantIncome': [5000, 6000, 7000, 8000],
        'CoapplicantIncome': [1000, 1500, 0, 2000],
        'LoanAmount': [100, 150, 200, 250],
        'Loan_Amount_Term': [360, 360, 180, 360],
        'Married': [1, 1, 0, 1],
        'Dependents': [0, 1, 2, 0],
        'Education': [1, 0, 1, 1],
        'Self_Employed': [0, 1, 0, 1]
    })


def test_create_income_features(sample_dataframe):
    """Test income feature creation."""
    df_features = create_income_features(sample_dataframe)
    
    # Check that new features are created
    assert 'TotalIncome' in df_features.columns, "TotalIncome should be created"
    assert 'IncomeToLoanRatio' in df_features.columns, "IncomeToLoanRatio should be created"
    assert 'EMI' in df_features.columns, "EMI should be created"
    
    # Verify calculations
    assert df_features['TotalIncome'].iloc[0] == 6000, "TotalIncome calculation incorrect"
    assert df_features['EMI'].iloc[0] == pytest.approx(100/360, rel=1e-3), "EMI calculation incorrect"


def test_create_demographic_features(sample_dataframe):
    """Test demographic feature creation."""
    df_features = create_demographic_features(sample_dataframe)
    
    # Check that FamilySize is created
    assert 'FamilySize' in df_features.columns, "FamilySize should be created"
    
    # Verify FamilySize calculation (applicant + spouse if married + dependents)
    # For first row: 1 (applicant) + 1 (married) + 0 (dependents) = 2
    assert df_features['FamilySize'].iloc[0] == 2, "FamilySize calculation incorrect"


def test_income_features_with_zero_values():
    """Test income features with zero values to avoid division by zero."""
    df = pd.DataFrame({
        'ApplicantIncome': [0, 5000],
        'CoapplicantIncome': [0, 1000],
        'LoanAmount': [0, 100],
        'Loan_Amount_Term': [360, 360]
    })
    
    df_features = create_income_features(df)
    
    # Should not raise errors
    assert not df_features['IncomeToLoanRatio'].isnull().all(), "Should handle zero values"
    assert not df_features['EMI'].isnull().all(), "Should handle zero values"


def test_feature_engineering_maintains_shape():
    """Test that feature engineering doesn't drop rows."""
    df = pd.DataFrame({
        'ApplicantIncome': [5000, 6000, 7000],
        'CoapplicantIncome': [1000, 1500, 2000],
        'LoanAmount': [100, 150, 200],
        'Loan_Amount_Term': [360, 360, 180]
    })
    
    df_features = create_income_features(df)
    
    assert len(df_features) == len(df), "Number of rows should not change"
