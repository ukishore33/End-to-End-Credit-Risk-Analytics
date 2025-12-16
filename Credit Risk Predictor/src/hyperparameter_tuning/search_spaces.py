"""
Search Spaces Module

This module provides predefined hyperparameter search spaces for different models.
"""

from typing import Dict, Any, List
from scipy.stats import uniform, randint, loguniform


# Predefined search spaces for common models
SEARCH_SPACES = {
    'logistic_regression': {
        'C': loguniform(0.001, 100),
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'max_iter': [100, 500, 1000]
    },
    
    'random_forest': {
        'n_estimators': randint(50, 500),
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    },
    
    'gradient_boosting': {
        'n_estimators': randint(50, 500),
        'learning_rate': loguniform(0.001, 0.3),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.6, 0.4),
        'max_features': ['sqrt', 'log2', None]
    },
    
    'xgboost': {
        'n_estimators': randint(50, 500),
        'learning_rate': loguniform(0.001, 0.3),
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': loguniform(0.001, 1),
        'reg_alpha': loguniform(0.001, 10),
        'reg_lambda': loguniform(0.001, 10)
    },
    
    'lightgbm': {
        'n_estimators': randint(50, 500),
        'learning_rate': loguniform(0.001, 0.3),
        'max_depth': randint(3, 10),
        'num_leaves': randint(10, 100),
        'min_child_samples': randint(5, 50),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': loguniform(0.001, 10),
        'reg_lambda': loguniform(0.001, 10)
    },
    
    'svm': {
        'C': loguniform(0.001, 100),
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto'] + list(loguniform(0.001, 1).rvs(5))
    },
    
    'ridge': {
        'alpha': loguniform(0.001, 100)
    },
    
    'lasso': {
        'alpha': loguniform(0.001, 100)
    },
    
    'elastic_net': {
        'alpha': loguniform(0.001, 100),
        'l1_ratio': uniform(0, 1)
    }
}


# Grid search versions (discrete values)
GRID_SEARCH_SPACES = {
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    'gradient_boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9]
    },
    
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    },
    
    'lightgbm': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [20, 31, 50],
        'subsample': [0.7, 0.8, 0.9]
    }
}


def get_search_space(
    model_type: str,
    search_type: str = 'random'
) -> Dict[str, Any]:
    """
    Get predefined search space for a model type.
    
    Parameters
    ----------
    model_type : str
        Type of model
    search_type : str, default='random'
        Type of search ('random' or 'grid')
        
    Returns
    -------
    Dict[str, Any]
        Parameter search space
        
    Examples
    --------
    >>> space = get_search_space('xgboost', search_type='random')
    >>> print(space.keys())
    """
    if search_type == 'grid':
        spaces = GRID_SEARCH_SPACES
    else:
        spaces = SEARCH_SPACES
    
    if model_type not in spaces:
        raise ValueError(
            f"No predefined search space for {model_type}. "
            f"Available: {list(spaces.keys())}"
        )
    
    return spaces[model_type].copy()


def create_custom_search_space(
    base_model: str,
    custom_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a custom search space based on a predefined one.
    
    Parameters
    ----------
    base_model : str
        Base model type to start from
    custom_params : Dict[str, Any]
        Custom parameter ranges to override
        
    Returns
    -------
    Dict[str, Any]
        Combined search space
        
    Examples
    --------
    >>> custom = {'n_estimators': [50, 100], 'max_depth': [5, 10, 15]}
    >>> space = create_custom_search_space('random_forest', custom)
    """
    base_space = get_search_space(base_model, 'random')
    base_space.update(custom_params)
    return base_space


def get_param_importance(
    cv_results: Dict[str, Any],
    param_names: List[str]
) -> Dict[str, float]:
    """
    Analyze parameter importance from CV results.
    
    Parameters
    ----------
    cv_results : Dict[str, Any]
        Cross-validation results from tuning
    param_names : List[str]
        Names of parameters to analyze
        
    Returns
    -------
    Dict[str, float]
        Parameter importance scores
    """
    # TODO: Implement parameter importance analysis
    # This could use ANOVA or correlation analysis
    raise NotImplementedError("Parameter importance not yet implemented")


if __name__ == "__main__":
    print("Search Spaces Module")
    print("=" * 50)
    print("Available predefined search spaces:")
    for model in SEARCH_SPACES.keys():
        print(f"  - {model}")
    print()
    print("Usage:")
    print("  from src.hyperparameter_tuning.search_spaces import get_search_space")
    print("  space = get_search_space('xgboost', search_type='random')")
