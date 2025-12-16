# Dashboard Module

This module contains dashboard components for visualizing credit risk analytics.

## Overview

The dashboard module provides:
- Interactive web dashboard (Streamlit/Dash)
- Key performance metrics display
- Model performance visualizations
- Prediction explanations
- Drift monitoring views

## Files

- `app.py` - Main dashboard application
- `components.py` - Reusable dashboard components
- `layouts.py` - Dashboard layouts and pages

## Usage

### Running the Dashboard

```bash
# Using Streamlit
streamlit run src/dashboard/app.py

# Or using Dash
python src/dashboard/app.py
```

### Components

```python
from src.dashboard.components import (
    create_metric_card,
    create_performance_chart,
    create_feature_importance_plot
)

# Create a metric card
card = create_metric_card('AUC-ROC', 0.85, change=0.02)
```

## Dashboard Pages

### 1. Overview
- Key metrics summary
- Model performance trends
- Recent predictions

### 2. Model Performance
- ROC curves
- Confusion matrix
- Lift charts
- Performance over time

### 3. Predictions
- Individual prediction lookup
- Batch prediction results
- Explanation visualization

### 4. Monitoring
- Data drift indicators
- Performance alerts
- Feature distribution changes

### 5. Model Comparison
- Side-by-side model comparison
- A/B test results

## TODO

- [ ] Implement Streamlit dashboard
- [ ] Add real-time data refresh
- [ ] Create embeddable components
- [ ] Add authentication
