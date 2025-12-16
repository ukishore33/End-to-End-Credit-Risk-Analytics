"""Unit tests for data preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import (
    handle_missing_values,
    encode_categorical_variables,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        'ApplicantIncome': [5000, 6000, np.nan, 8000],
        'LoanAmount': [100, 150, 200, np.nan],
        'Gender': ['Male', 'Female', 'Male', None],
        'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate'],
        'Loan_Status': ['Y', 'N', 'Y', 'Y']
    })


def test_handle_missing_values(sample_dataframe):
    """Test missing value handling."""
    df_cleaned = handle_missing_values(sample_dataframe)
    
    # Check that no missing values remain
    assert df_cleaned.isnull().sum().sum() == 0, "Missing values should be handled"
    
    # Check that the shape is unchanged
    assert df_cleaned.shape == sample_dataframe.shape, "DataFrame shape should not change"


def test_encode_categorical_variables(sample_dataframe):
    """Test categorical encoding."""
    # First handle missing values
    df_cleaned = handle_missing_values(sample_dataframe)
    
    # Then encode
    df_encoded = encode_categorical_variables(df_cleaned)
    
    # Check that categorical columns are encoded (except target)
    categorical_cols = ['Gender', 'Education']
    for col in categorical_cols:
        assert df_encoded[col].dtype in [np.int32, np.int64], f"{col} should be encoded to numeric"
    
    # Target should remain unchanged
    assert df_encoded['Loan_Status'].dtype == object, "Target column should not be encoded"


def test_handle_missing_values_empty_dataframe():
    """Test handling of empty dataframe."""
    df_empty = pd.DataFrame()
    result = handle_missing_values(df_empty)
    assert len(result) == 0, "Empty dataframe should remain empty"


def test_missing_value_strategies():
    """Test different missing value strategies."""
    df = pd.DataFrame({
        'num_col': [1, 2, np.nan, 4, 5],
        'cat_col': ['A', 'B', None, 'A', 'B']
    })
    
    df_handled = handle_missing_values(df)
    
    # Numerical column should be filled
    assert not df_handled['num_col'].isnull().any()
    
    # Categorical column should be filled
    assert not df_handled['cat_col'].isnull().any()
