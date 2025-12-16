# Models Module

This module contains model implementations for credit risk analytics.

## Overview

The models module provides implementations for three main prediction tasks:

1. **Loan Default Prediction** (Binary Classification)
2. **Credit Risk Scoring Model** (Regression)
3. **Future Risk Score Prediction** (Time-based / Out-of-sample)

## Files

- `binary_classifier.py` - Binary classification for loan default prediction
- `risk_scorer.py` - Regression model for credit risk scoring
- `time_series_predictor.py` - Time-based future risk prediction
- `base_model.py` - Base model class with common functionality
- `model_utils.py` - Model utilities and helpers

## Usage

### Binary Classification (Loan Default Prediction)

```python
from src.models.binary_classifier import LoanDefaultClassifier

model = LoanDefaultClassifier(model_type='xgboost')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Regression (Credit Risk Scoring)

```python
from src.models.risk_scorer import CreditRiskScorer

model = CreditRiskScorer(model_type='gradient_boosting')
model.fit(X_train, y_train)
risk_scores = model.predict(X_test)
```

### Time-based Prediction

```python
from src.models.time_series_predictor import FutureRiskPredictor

model = FutureRiskPredictor(horizon=6)  # 6 months ahead
model.fit(X_train, y_train, dates=dates_train)
future_risks = model.predict(X_test)
```

## Supported Algorithms

### Classification
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Neural Network

### Regression
- Linear Regression
- Ridge/Lasso
- Gradient Boosting
- XGBoost
- Neural Network

### Time Series
- ARIMA-based models
- LSTM
- Prophet
- Temporal Fusion Transformer

## TODO

- [ ] Implement model ensemble capabilities
- [ ] Add model versioning and tracking
- [ ] Integrate with MLflow
