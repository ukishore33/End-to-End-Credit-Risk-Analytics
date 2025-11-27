# End-to-End Credit Risk Analytics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive end-to-end credit risk analytics pipeline for building, evaluating, and deploying credit risk models.

## 🎯 Overview

This project provides a complete framework for credit risk modeling including:

- **Loan Default Prediction** (Binary Classification)
- **Credit Risk Scoring Model** (Regression)
- **Future Risk Score Prediction** (Time-based / Out-of-sample)

## 📊 End-to-End Pipeline

The pipeline covers the entire machine learning lifecycle:

1. **Data Sourcing** - Data acquisition and validation
2. **EDA** - Exploratory data analysis and visualization
3. **Feature Engineering** - Feature creation, selection, and transformation
4. **Model Building** - Classification, regression, and time-series models
5. **Hyperparameter Tuning** - Grid search, random search, and Bayesian optimization
6. **Model Evaluation** - Comprehensive metrics and visualizations
7. **Model Explainability** - SHAP and LIME explanations
8. **Model Monitoring** - Drift detection and performance tracking
9. **Dashboard** - Interactive visualization dashboard

## 📁 Project Structure

```
End-to-End-Credit-Risk-Analytics/
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data files
│   └── interim/                # Intermediate data files
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_explainability.ipynb
├── src/
│   ├── data_sourcing/          # Data loading and validation
│   │   ├── data_loader.py
│   │   ├── validators.py
│   │   └── data_sources.md
│   ├── eda/                    # Exploratory data analysis
│   │   ├── eda_report.py
│   │   ├── visualizations.py
│   │   └── statistical_analysis.py
│   ├── feature_engineering/    # Feature engineering
│   │   ├── feature_transformer.py
│   │   ├── feature_selector.py
│   │   └── domain_features.py
│   ├── models/                 # Model implementations
│   │   ├── binary_classifier.py      # Loan Default Prediction
│   │   ├── risk_scorer.py            # Credit Risk Scoring
│   │   ├── time_series_predictor.py  # Future Risk Prediction
│   │   └── model_utils.py
│   ├── hyperparameter_tuning/  # HPO utilities
│   │   ├── tuner.py
│   │   ├── search_spaces.py
│   │   └── cross_validation.py
│   ├── evaluation/             # Model evaluation
│   │   ├── metrics.py
│   │   ├── visualizations.py
│   │   └── comparison.py
│   ├── explainability/         # Model explainability (SHAP + LIME)
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   ├── global_explanations.py
│   │   └── local_explanations.py
│   ├── monitoring/             # Model monitoring
│   │   ├── drift_detector.py
│   │   ├── performance_tracker.py
│   │   └── alerting.py
│   ├── dashboard/              # Dashboard application
│   │   ├── app.py
│   │   ├── components.py
│   │   └── layouts.py
│   └── utils/                  # Utility functions
│       ├── logger.py
│       ├── config.py
│       └── helpers.py
├── configs/                    # Configuration files
│   ├── model_config.yaml
│   └── feature_config.yaml
├── tests/                      # Unit tests
│   ├── test_data_loader.py
│   ├── test_models.py
│   └── test_feature_engineering.py
├── docs/                       # Documentation
│   ├── getting_started.md
│   └── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ukishore33/End-to-End-Credit-Risk-Analytics.git
cd End-to-End-Credit-Risk-Analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Load Data

```python
from src.data_sourcing.data_loader import load_credit_data

df = load_credit_data(source='local', path='data/raw/credit_data.csv')
```

#### 2. Train a Binary Classifier (Loan Default Prediction)

```python
from src.models.binary_classifier import LoanDefaultClassifier

model = LoanDefaultClassifier(model_type='random_forest')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

#### 3. Train a Risk Scorer (Credit Risk Scoring)

```python
from src.models.risk_scorer import CreditRiskScorer

scorer = CreditRiskScorer(model_type='gradient_boosting')
scorer.fit(X_train, y_train)
risk_scores = scorer.predict(X_test)
```

#### 4. Train a Time-Series Predictor (Future Risk)

```python
from src.models.time_series_predictor import FutureRiskPredictor

predictor = FutureRiskPredictor(model_type='gradient_boosting', horizon=6)
predictor.fit(X_train, y_train, dates=dates_train)
future_risks = predictor.predict(X_test, dates=dates_test)
```

#### 5. Model Explainability

```python
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer

# SHAP Analysis
shap_explainer = SHAPExplainer(model, X_train)
shap_values = shap_explainer.explain(X_test)
shap_explainer.plot_summary(X_test)

# LIME Analysis
lime_explainer = LIMEExplainer(model, X_train, feature_names)
explanation = lime_explainer.explain_instance(X_test[0])
```

#### 6. Run Dashboard

```bash
streamlit run src/dashboard/app.py
```

## 📈 Key Metrics

The evaluation module provides comprehensive metrics:

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR
- Confusion Matrix

### Credit-Specific Metrics
- KS Statistic
- Gini Coefficient
- Lift Charts
- Population Stability Index (PSI)

### Regression Metrics
- MSE, RMSE, MAE
- R², Adjusted R²
- MAPE

## 🔧 Configuration

Configuration files are located in the `configs/` directory:

- `model_config.yaml` - Model settings, hyperparameters
- `feature_config.yaml` - Feature definitions, transformations

## 📚 Documentation

- [Getting Started Guide](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Data Sources](src/data_sourcing/data_sources.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions or suggestions, please open an issue on GitHub.

---

**TODO Items:**
- [ ] Add sample dataset
- [ ] Implement advanced model types (XGBoost, LightGBM, Neural Networks)
- [ ] Complete SHAP and LIME implementations
- [ ] Add MLflow integration
- [ ] Create comprehensive API documentation
- [ ] Add CI/CD pipeline