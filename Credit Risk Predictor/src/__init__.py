"""
Credit Risk Analytics Package

End-to-end credit risk analytics pipeline including:
- Data sourcing and validation
- Exploratory data analysis
- Feature engineering
- Model building (classification, regression, time-series)
- Hyperparameter tuning
- Model evaluation
- Model explainability (SHAP + LIME)
- Model monitoring
- Dashboard visualization
"""

__version__ = "0.1.0"
__author__ = "Kishore Umaprasad"

from . import data_sourcing
from . import eda
from . import feature_engineering
from . import models
from . import hyperparameter_tuning
from . import evaluation
from . import explainability
from . import monitoring
from . import dashboard
from . import utils

__all__ = [
    'data_sourcing',
    'eda',
    'feature_engineering',
    'models',
    'hyperparameter_tuning',
    'evaluation',
    'explainability',
    'monitoring',
    'dashboard',
    'utils'
]
