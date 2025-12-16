# Notebooks Directory

This directory contains Jupyter notebooks for exploratory data analysis, experimentation, and visualization.

## Structure

Notebooks are organized by purpose:
- `01_*` - Data exploration and understanding
- `02_*` - Data preprocessing and cleaning
- `03_*` - Feature engineering experiments
- `04_*` - Model training and comparison
- `05_*` - Model evaluation and interpretation
- `06_*` - Results visualization

## Naming Convention

Use descriptive names with prefixes:
```
{number}_{descriptive_name}.ipynb

Examples:
01_exploratory_data_analysis.ipynb
02_data_cleaning.ipynb
03_feature_engineering.ipynb
04_model_comparison.ipynb
05_model_evaluation.ipynb
06_results_visualization.ipynb
```

## Best Practices

1. **Clear Structure**: Each notebook should have a clear purpose
2. **Documentation**: Add markdown cells to explain your analysis
3. **Reproducibility**: Set random seeds for reproducible results
4. **Clean Code**: Move reusable code to src/ modules
5. **Version Control**: Clear outputs before committing (optional)

## Example Notebooks

### 01_exploratory_data_analysis.ipynb
- Data loading and overview
- Statistical summaries
- Distribution analysis
- Correlation analysis
- Missing value analysis

### 02_data_cleaning.ipynb
- Handling missing values
- Outlier detection and treatment
- Data type conversions
- Data validation

### 03_feature_engineering.ipynb
- Creating new features
- Feature transformations
- Feature selection
- Feature importance analysis

### 04_model_comparison.ipynb
- Training multiple models
- Cross-validation
- Hyperparameter tuning
- Model comparison

### 05_model_evaluation.ipynb
- Performance metrics
- Confusion matrix
- ROC curves
- Feature importance
- Model interpretation (SHAP, LIME)

### 06_results_visualization.ipynb
- Business insights
- Presentation-ready visualizations
- Dashboard prototypes

## Running Notebooks

Start Jupyter:
```bash
jupyter notebook
```

Or use JupyterLab:
```bash
jupyter lab
```

Using Docker:
```bash
docker-compose up jupyter
```

## Tips

- Use virtual environment to avoid dependency conflicts
- Regularly clear outputs to reduce notebook size
- Extract production code to Python modules
- Use nbconvert to generate reports
