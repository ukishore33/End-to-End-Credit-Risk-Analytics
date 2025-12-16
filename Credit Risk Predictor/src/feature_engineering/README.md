# Feature Engineering Module

This module contains feature engineering utilities for credit risk models.

## Overview

The feature engineering module provides:
- Feature transformation functions
- Feature selection utilities
- Domain-specific feature creation
- Encoding strategies

## Files

- `feature_transformer.py` - Feature transformation utilities
- `feature_selector.py` - Feature selection methods
- `domain_features.py` - Credit risk domain-specific features

## Usage

```python
from src.feature_engineering.feature_transformer import transform_features
from src.feature_engineering.feature_selector import select_features
from src.feature_engineering.domain_features import create_credit_features

# Transform features
df_transformed = transform_features(df, config='configs/feature_config.yaml')

# Create domain-specific features
df = create_credit_features(df)

# Select best features
selected_features = select_features(df, target='default', method='mutual_info', k=20)
```

## Feature Categories

### 1. Derived Features
- Ratio features (debt-to-income, credit utilization)
- Aggregate features (total debt, total accounts)
- Temporal features (account age, time since last payment)

### 2. Transformed Features
- Log transformations (for skewed distributions)
- Standardization / Normalization
- Polynomial features

### 3. Encoded Features
- One-hot encoding (low cardinality categorical)
- Target encoding (high cardinality categorical)
- Ordinal encoding (ordered categories)

## TODO

- [ ] Implement feature store integration
- [ ] Add automated feature selection
- [ ] Create feature importance tracking
