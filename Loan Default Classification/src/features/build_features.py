"""Feature engineering for loan eligibility prediction.

This module creates new features from the existing data to improve model performance.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_income_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create income-related features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new income features
    """
    logger.info("Creating income features")
    df_features = df.copy()
    
    # Example features (adjust column names based on actual data)
    if 'ApplicantIncome' in df_features.columns and 'CoapplicantIncome' in df_features.columns:
        df_features['TotalIncome'] = df_features['ApplicantIncome'] + df_features['CoapplicantIncome']
        logger.info("Created TotalIncome feature")
    
    if 'LoanAmount' in df_features.columns and 'TotalIncome' in df_features.columns:
        df_features['IncomeToLoanRatio'] = df_features['TotalIncome'] / (df_features['LoanAmount'] + 1)
        logger.info("Created IncomeToLoanRatio feature")
    
    if 'LoanAmount' in df_features.columns and 'Loan_Amount_Term' in df_features.columns:
        df_features['EMI'] = df_features['LoanAmount'] / (df_features['Loan_Amount_Term'] + 1)
        logger.info("Created EMI feature")
    
    return df_features


def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create demographic features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new demographic features
    """
    logger.info("Creating demographic features")
    df_features = df.copy()
    
    # Example: Family size
    if 'Married' in df_features.columns and 'Dependents' in df_features.columns:
        df_features['FamilySize'] = df_features['Dependents'].astype(str).replace('+', '', regex=True).astype(float)
        df_features['FamilySize'] += df_features['Married'].apply(lambda x: 1 if x == 1 else 0)
        df_features['FamilySize'] += 1  # Add applicant
        logger.info("Created FamilySize feature")
    
    return df_features


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with interaction features
    """
    logger.info("Creating interaction features")
    df_features = df.copy()
    
    # Example interactions
    if 'Education' in df_features.columns and 'Self_Employed' in df_features.columns:
        df_features['Education_SelfEmployed'] = df_features['Education'] * df_features['Self_Employed']
        logger.info("Created Education_SelfEmployed interaction")
    
    return df_features


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log transformation to skewed features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with scaled features
    """
    logger.info("Applying log transformation to skewed features")
    df_scaled = df.copy()
    
    # Apply log transformation to reduce skewness
    numeric_cols = ['LoanAmount', 'TotalIncome', 'ApplicantIncome', 'CoapplicantIncome']
    
    for col in numeric_cols:
        if col in df_scaled.columns:
            df_scaled[f'{col}_log'] = np.log1p(df_scaled[col])
            logger.info(f"Created {col}_log feature")
    
    return df_scaled


def build_features(input_path: str, output_path: str):
    """Build features from preprocessed data.
    
    Args:
        input_path: Path to preprocessed data file
        output_path: Path to save feature-engineered data
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Apply feature engineering
    df = create_income_features(df)
    df = create_demographic_features(df)
    df = create_interaction_features(df)
    df = scale_features(df)
    
    # Save features
    logger.info(f"Saving features to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Feature engineering completed. Final shape: {df.shape}")


def main():
    """Main feature engineering pipeline."""
    parser = argparse.ArgumentParser(description='Build features for loan eligibility prediction')
    parser.add_argument('--input', type=str, required=True, help='Path to input directory with train/test data')
    parser.add_argument('--output', type=str, required=True, help='Output directory for feature-engineered data')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process training data
    train_input = input_dir / 'train.csv'
    train_output = output_dir / 'train_features.csv'
    if train_input.exists():
        build_features(str(train_input), str(train_output))
    
    # Process test data
    test_input = input_dir / 'test.csv'
    test_output = output_dir / 'test_features.csv'
    if test_input.exists():
        build_features(str(test_input), str(test_output))
    
    logger.info("All feature engineering completed successfully")


if __name__ == '__main__':
    main()
