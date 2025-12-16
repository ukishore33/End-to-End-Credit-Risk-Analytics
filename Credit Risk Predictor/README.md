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
Further Project ideas under this project

# Loan default analysis dashboard

# Airlines-Dashboard

### Dashboard Link : https://app.powerbi.com/groups/me/reports/384d017e-e935-44dc-9e7d-1626c1a36de1/ReportSection

## Problem Statement

This dashboard helps the airlines understand their customers better. It helps the airlines know if their customers are satisfied with their services. Through different ratings, they get to know their improvement area, & thus they can improve their services by identifying these area. It also lets them know the average delay & departure time, thus since by using this dashboard they have identified this problem, they can further work on factors responsible for these unwanted delays.

Since, number of neutral/dissatisfied customers (almost 57 %) are more than satisfied customers (around 43 %), thus in all they must work on improving their services. 

Also since average delay in arrival & departure both is 15 minutes, thus they must try to reduce it.


### Steps followed 

- Step 1 : Load data into Power BI Desktop, dataset is a csv file.
- Step 2 : Open power query editor & in view tab under Data preview section, check "column distribution", "column quality" & "column profile" options.
- Step 3 : Also since by default, profile will be opened only for 1000 rows so you need to select "column profiling based on entire dataset".
- Step 4 : It was observed that in none of the columns errors & empty values were present except column named "Arrival Delay".
- Step 5 : For calculating average delay time, null values were not taken into account as only less than 1% values are null in this column(i.e column named "Arrival Delay") 
- Step 6 : In the report view, under the view tab, theme was selected.
- Step 7 : Since the data contains various ratings, thus in order to represent ratings, a new visual was added using the three ellipses in the visualizations pane in report view. 
- Step 8 : Visual filters (Slicers) were added for four fields named "Class", "Customer Type", "Gate Location" & "Type of travel".
- Step 9 : Two card visuals were added to the canvas, one representing average departure delay in minutes & other representing average arrival delay in minutes.
           Using visual level filter from the filters pane, basic filtering was used & null values were unselected for consideration into average calculation.
           
           Although, by default, while calculating average, blank values are ignored.
- Step 10 : A bar chart was also added to the report design area representing the number of satisfied & neutral/unsatisfied customers. While creating this visual, field named "Gender" was also added to the Legends bucket, thus number of customers are also seggregated according the gender. 
- Step 11 : Ratings Visual was used to represent different ratings mentioned below,

  (a) Baggage Handling

  (b) Check-in Services
  
  (c) Cleanliness
  
  (d) Ease of online booking
  
  (e) Food & Drink
  
  (f) In-flight Entertainment

  (g) In-flight Service
  
  (h) In-flight wifi service
  
  (i) Leg Room service
  
  (j) On-board service
  
  (k) Online boarding
  
  (l) Seat comfort
  
  (m) Departure & arrival time convenience
  
In our dataset, Some parameters were assigned value 0, representing those parameters are not applicable for some customers.

All these values have been ignored while calculating average rating for each of the parameters mentioned above.

- Step 12 : In the report view, under the insert tab, two text boxes were added to the canvas, in one of them name of the airlines was mentioned & in the other one company's tagline was written.
- Step 13 : In the report view, under the insert tab, using shapes option from elements group a rectangle was inserted & similarly using image option company's logo was added to the report design area. 
- Step 14 : Calculated column was created in which, customers were grouped into various age groups.

for creating new column following DAX expression was written;
       
        Age Group = 
        
        if(airline_passenger_satisfaction[Age]<=25, "0-25 (25 included)",
        
        if(airline_passenger_satisfaction[Age]<=50, "25-50 (50 included)",
        
        if(airline_passenger_satisfaction[Age]<=75, "50-75 (75 included)",
        
        "75-100 (100 included)")))
        
Snap of new calculated column ,

![Snap_1](https://user-images.githubusercontent.com/102996550/174089602-ab834a6b-62ce-4b62-8922-a1d241ec240e.jpg)

        
- Step 15 : New measure was created to find total count of customers.

Following DAX expression was written for the same,
        
        Count of Customers = COUNT(airline_passenger_satisfaction[ID])
        
A card visual was used to represent count of customers.

![Snap_Count](https://user-images.githubusercontent.com/102996550/174090154-424dc1a4-3ff7-41f8-9617-17a2fb205825.jpg)

        
 - Step 16 : New measure was created to find  % of customers,
 
 Following DAX expression was written to find % of customers,
 
         % Customers = (DIVIDE(airline_passenger_satisfaction[Count of Customers], 129880)*100)
 
 A card visual was used to represent this perecntage.
 
 Snap of % of customers who preferred business class
 
 ![Snap_Percentage](https://user-images.githubusercontent.com/102996550/174090653-da02feb4-4775-4a95-affb-a211ca985d07.jpg)

 
 - Step 17 : New measure was created to calculate total distance travelled by flights & a card visual was used to represent total distance.
 
 Following DAX expression was written to find total distance,
 
         Total Distance Travelled = SUM(airline_passenger_satisfaction[Flight Distance])
    
 A card visual was used to represent this total distance.
 
 
 ![Snap_3](https://user-images.githubusercontent.com/102996550/174091618-bf770d6c-34c6-44d4-9f5e-49583a6d5f68.jpg)
 
 - Step 18 : The report was then published to Power BI Service.
 
 
![Publish_Message](https://user-images.githubusercontent.com/102996550/174094520-3a845196-97e6-4d44-8760-34a64abc3e77.jpg)

# Snapshot of Dashboard (Power BI Service)

![dashboard_snapo](https://user-images.githubusercontent.com/102996550/174096257-11f1aae5-203d-44fc-bfca-25d37faf3237.jpg)

 
 # Report Snapshot (Power BI DESKTOP)

 
![Dashboard_upload](https://user-images.githubusercontent.com/102996550/174074051-4f08287a-0568-4fdf-8ac9-6762e0d8fa94.jpg)

# Insights

A single page report was created on Power BI Desktop & it was then published to Power BI Service.

Following inferences can be drawn from the dashboard;

### [1] Total Number of Customers = 129880

   Number of satisfied Customers (Male) = 28159 (21.68 %)

   Number of satisfied Customers (Female) = 28269 (21.76 %)

   Number of neutral/unsatisfied customers (Male) = 35822 (27.58 %)

   Number of neutral/unsatisfied customers (Female) = 37630 (28.97 %)


           thus, higher number of customers are neutral/unsatisfied.
           
### [2] Average Ratings

    a) Baggage Handling - 3.63/5
    b) Check-in Service - 3.31/5
    c) Cleanliness - 3.29/5
    d) Ease of online booking - 2.88/5
    e) Food & Drink - 3.21/5
    f) In-flight Entertainment - 3.36/5
    g) In-flight service - 3.64/5
    h) In-flight Wifi service - 2.81/5
    i) Leg room service - 3.37/5
    j) On-board service - 3.38/5
    k) Online boarding - 3.33/5
    l) Seat comfort - 3.44/5
    m) Departure & arrival convenience - 3.22/5
  
  while calculating average rating, null values have been ignored as they were not relevant for some customers. 
  
  These ratings will change if different visual filters will be applied.  
  
  ### [3] Average Delay 
  
      a) Average delay in arrival(minutes) - 15.09
      b) Average delay in departure(minutes) - 14.71
Average delay will change if different visual filters will be applied.

 ### [4] Some other insights
 
 ### Class
 
 1.1) 47.87 % customers travelled by Business class.
 
 1.2) 44.89 % customers travelled by Economy class.
 
 1.3) 7.25 % customers travelled by Economy plus class.
 
         thus, maximum customers travelled by Business class.
 
 ### Age Group
 
 2.1)  21.69 % customers belong to '0-25' age group.
 
 2.2)  52.44 % customers belong to '25-50' age group.
 
 2.3)  25.57 % customers belong to '50-75' age group.
 
 2.4)  0.31 % customers belong to '75-100' age group.
 
         thus, maximum customers belong to '25-50' age group.
         
### Customer Type

3.1) 18.31 % customers have customer type 'First time'.

3.2) 81.69 % customers have customer type 'returning'.
       
       thus, more customers have customer type 'returning'.

### Type of travel

4.1) 69.06 % customers have travel type 'Business'.

4.2) 30.94 % customers have travel type 'Personal'.

        thus, more customers have travel type 'Business'.

# Customer-Risk-Scoring-Model

# Credit-Scorecard-Excel-Python

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
