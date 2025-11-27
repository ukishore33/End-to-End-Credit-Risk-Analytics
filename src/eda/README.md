# Exploratory Data Analysis (EDA) Module

This module contains tools and scripts for exploratory data analysis of credit risk data.

## Overview

The EDA module provides:
- Automated EDA reports
- Statistical analysis
- Visualization utilities
- Distribution analysis
- Correlation analysis
- Outlier detection

## Files

- `eda_report.py` - Automated EDA report generation
- `visualizations.py` - Visualization utilities
- `statistical_analysis.py` - Statistical tests and analysis

## Usage

```python
from src.eda.eda_report import generate_eda_report
from src.eda.visualizations import plot_distributions, plot_correlations

# Generate comprehensive EDA report
report = generate_eda_report(df, target_column='default')

# Create visualizations
plot_distributions(df, output_dir='reports/figures/')
plot_correlations(df, output_dir='reports/figures/')
```

## Key Analyses

1. **Target Variable Analysis**
   - Class distribution (for classification)
   - Target distribution (for regression)
   - Temporal patterns

2. **Feature Analysis**
   - Missing value analysis
   - Distribution analysis
   - Outlier detection

3. **Relationship Analysis**
   - Correlation with target
   - Feature correlations
   - Multicollinearity check

## TODO

- [ ] Implement automated EDA report generation
- [ ] Add interactive visualizations
- [ ] Integrate with profiling libraries (pandas-profiling, sweetviz)
