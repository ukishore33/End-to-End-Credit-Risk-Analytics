# Getting Started

This guide will help you get started with the Credit Risk Analytics project.

## Prerequisites

- Python 3.8+
- pip or conda package manager
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ukishore33/End-to-End-Credit-Risk-Analytics.git
cd End-to-End-Credit-Risk-Analytics
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Using conda
conda create -n credit-risk python=3.9
conda activate credit-risk
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import src; print('Installation successful!')"
```

## Quick Start

### 1. Prepare Your Data

Place your credit data in `data/raw/` directory.

### 2. Run EDA

```python
from src.data_sourcing.data_loader import load_credit_data
from src.eda.eda_report import generate_eda_report

# Load data
df = load_credit_data(source='local', path='data/raw/your_data.csv')

# Generate EDA report
report = generate_eda_report(df, target_column='default')
```

### 3. Train a Model

```python
from src.models.binary_classifier import LoanDefaultClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LoanDefaultClassifier(model_type='random_forest')
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### 4. Run Dashboard

```bash
streamlit run src/dashboard/app.py
```

## Next Steps

- Explore the [Jupyter notebooks](../notebooks/) for detailed examples
- Read the [User Guide](user_guide.md) for comprehensive documentation
- Check the [API Reference](api_reference.md) for function details

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure you've installed all dependencies
2. **FileNotFoundError**: Check that data files are in the correct location
3. **Memory Issues**: Try using a smaller dataset or increase system memory

### Getting Help

- Open an issue on GitHub
- Check existing documentation
