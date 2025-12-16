# Data Directory

This directory contains all datasets used in the project.

## Structure

- **raw/**: Original, immutable data dump
  - Place your raw loan application data here
  - Files in this directory should never be modified
  - Supported formats: CSV, Excel, JSON, Parquet

- **processed/**: Cleaned and preprocessed data
  - Output from preprocessing scripts
  - Ready for feature engineering and model training
  - Includes train/test splits

- **external/**: External data from third-party sources
  - Reference data
  - Economic indicators
  - Industry benchmarks

## Data Files (Example)

```
data/
├── raw/
│   ├── loan_applications.csv       # Main loan application data
│   ├── credit_history.csv          # Credit history data
│   └── applicant_demographics.csv  # Demographic information
├── processed/
│   ├── train.csv                   # Training dataset
│   ├── test.csv                    # Test dataset
│   └── train_features.csv          # Feature-engineered training data
└── external/
    ├── economic_indicators.csv     # Macro-economic data
    └── industry_benchmarks.csv     # BFSI industry standards
```

## Data Privacy

⚠️ **Important**: Do not commit sensitive or personal data to version control!

- All data files (*.csv, *.xlsx, *.json, *.parquet) in this directory are gitignored
- Only directory structure is tracked in git
- Store sensitive data securely and access through proper channels

## Data Sources

Document your data sources here:
- Source name
- Date acquired
- Data description
- Any preprocessing done at source

## Usage

To preprocess raw data:
```bash
python src/data/preprocess.py --input data/raw/loan_data.csv --output data/processed/
```
