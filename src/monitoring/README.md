# Model Monitoring Module

This module contains model monitoring utilities for credit risk models in production.

## Overview

The monitoring module provides:
- Model performance tracking
- Data drift detection
- Concept drift detection
- Alert generation
- Logging and reporting

## Files

- `drift_detector.py` - Data and concept drift detection
- `performance_tracker.py` - Model performance tracking
- `alerting.py` - Alert generation and notification

## Usage

```python
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.performance_tracker import PerformanceTracker

# Initialize drift detector
drift_detector = DriftDetector(reference_data=X_train)

# Check for drift
drift_report = drift_detector.detect_drift(X_new)
if drift_report['drift_detected']:
    print("Data drift detected!")

# Track performance
tracker = PerformanceTracker(model_name='loan_default_v1')
tracker.log_prediction(features, prediction, actual_outcome)
tracker.generate_report()
```

## Monitoring Aspects

### Data Drift
- Feature distribution shifts
- Population Stability Index (PSI)
- Kolmogorov-Smirnov test

### Concept Drift
- Model performance degradation
- Prediction distribution changes
- Feature importance shifts

### Performance Metrics
- Real-time accuracy tracking
- Rolling window metrics
- Threshold-based alerts

## TODO

- [ ] Add real-time monitoring dashboard
- [ ] Implement automated retraining triggers
- [ ] Add integration with MLflow/Weights & Biases
- [ ] Create Slack/email alerting
