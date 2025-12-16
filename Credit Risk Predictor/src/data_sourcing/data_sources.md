# Data Sources Documentation

## Overview

This document describes the data sources used in the Credit Risk Analytics project.

## Primary Data Sources

### 1. Loan Application Data

- **Source**: [Specify source - e.g., Internal database, Kaggle, etc.]
- **Format**: CSV/Parquet
- **Update Frequency**: [Daily/Weekly/Monthly]
- **Size**: [Approximate size]

#### Schema

| Column Name | Data Type | Description |
|------------|-----------|-------------|
| loan_id | string | Unique loan identifier |
| applicant_id | string | Unique applicant identifier |
| loan_amount | float | Requested loan amount |
| interest_rate | float | Interest rate (%) |
| loan_term | int | Loan term in months |
| employment_length | int | Years of employment |
| annual_income | float | Annual income |
| debt_to_income | float | Debt-to-income ratio |
| credit_score | int | Credit score |
| home_ownership | string | Home ownership status |
| loan_purpose | string | Purpose of loan |
| loan_status | string | Current loan status |
| default | int | Default flag (1=default, 0=paid) |

### 2. Credit Bureau Data

- **Source**: [Specify source]
- **Format**: [Format]
- **Update Frequency**: [Frequency]

#### Schema

| Column Name | Data Type | Description |
|------------|-----------|-------------|
| applicant_id | string | Unique applicant identifier |
| credit_history_length | int | Length of credit history (months) |
| num_open_accounts | int | Number of open credit accounts |
| num_closed_accounts | int | Number of closed accounts |
| total_credit_limit | float | Total credit limit |
| total_credit_used | float | Total credit utilized |
| num_delinquencies | int | Number of delinquencies |
| num_bankruptcies | int | Number of bankruptcies |

### 3. Economic Indicators (Optional)

- **Source**: Federal Reserve Economic Data (FRED)
- **Format**: API/CSV
- **Update Frequency**: Monthly

## Data Dictionary

### Target Variables

1. **default** (Binary Classification)
   - 0: Loan was fully paid
   - 1: Loan defaulted

2. **risk_score** (Regression)
   - Continuous value representing credit risk (0-1000)

3. **future_default_probability** (Time-based Prediction)
   - Probability of default in next 6-12 months

## Data Quality Notes

- [ ] Document missing value patterns
- [ ] Document known data quality issues
- [ ] Document any data transformations applied at source

## Data Access

```python
# Example: Loading primary loan data
from src.data_sourcing.data_loader import load_credit_data

df = load_credit_data(
    source='local',
    path='data/raw/loan_data.csv'
)
```

## TODO

- [ ] Add actual data source details
- [ ] Document data refresh procedures
- [ ] Add data lineage information
- [ ] Set up automated data quality checks
