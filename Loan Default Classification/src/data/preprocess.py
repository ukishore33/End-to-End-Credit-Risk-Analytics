"""Data preprocessing script for loan eligibility prediction.

This module handles loading and preprocessing of raw loan application data.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info("Handling missing values")
    df_cleaned = df.copy()
    
    # Example: Fill numerical columns with median
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            median_value = df_cleaned[col].median()
            df_cleaned[col].fillna(median_value, inplace=True)
            logger.info(f"Filled {col} missing values with median: {median_value}")
    
    # Example: Fill categorical columns with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            mode_value = df_cleaned[col].mode()[0]
            df_cleaned[col].fillna(mode_value, inplace=True)
            logger.info(f"Filled {col} missing values with mode: {mode_value}")
    
    return df_cleaned


def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with encoded categorical variables
    """
    logger.info("Encoding categorical variables")
    df_encoded = df.copy()
    
    # Example: Simple label encoding for binary categorical variables
    # In production, consider using one-hot encoding or target encoding
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != 'Loan_Status':  # Assuming this is the target variable
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
            logger.info(f"Encoded column: {col}")
    
    return df_encoded


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        test_size: Proportion of dataset to include in test split
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data with test_size={test_size}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series, output_dir: str):
    """Save processed data to CSV files.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        output_dir: Directory to save the processed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed data to {output_dir}")
    
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df.to_csv(output_path / 'train.csv', index=False)
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv(output_path / 'test.csv', index=False)
    
    logger.info("Data saved successfully")


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess loan eligibility data')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--target', type=str, default='Loan_Status', help='Name of target column')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.input)
    
    # Preprocess data
    df = handle_missing_values(df)
    df = encode_categorical_variables(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, args.target, args.test_size)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, args.output)
    
    logger.info("Preprocessing completed successfully")


if __name__ == '__main__':
    main()
