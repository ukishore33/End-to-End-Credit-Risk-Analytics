"""
Risk Scorer Module

This module provides regression models for credit risk scoring.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union, List
from .base_model import BaseModel


class CreditRiskScorer(BaseModel):
    """
    Regression model for predicting credit risk scores.
    
    This class provides a unified interface for various regression
    algorithms to predict continuous risk scores.
    
    Parameters
    ----------
    model_type : str, default='gradient_boosting'
        Type of regression algorithm
        Options: 'linear', 'ridge', 'lasso', 'elastic_net',
                 'random_forest', 'gradient_boosting', 'xgboost',
                 'lightgbm', 'neural_network'
    model_params : Dict[str, Any], optional
        Model hyperparameters
    random_state : int, default=42
        Random seed for reproducibility
    score_range : tuple, optional
        (min_score, max_score) to clip predictions
        
    Examples
    --------
    >>> from src.models.risk_scorer import CreditRiskScorer
    >>> 
    >>> # Initialize and train model
    >>> model = CreditRiskScorer(model_type='gradient_boosting')
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Predict risk scores
    >>> risk_scores = model.predict(X_test)
    >>> 
    >>> # Evaluate model
    >>> from sklearn.metrics import mean_squared_error, r2_score
    >>> print(f"RMSE: {mean_squared_error(y_test, risk_scores, squared=False):.4f}")
    >>> print(f"R2: {r2_score(y_test, risk_scores):.4f}")
    """
    
    SUPPORTED_MODELS = [
        'linear',
        'ridge',
        'lasso',
        'elastic_net',
        'random_forest',
        'gradient_boosting',
        'xgboost',
        'lightgbm',
        'neural_network',
        'svr'
    ]
    
    def __init__(
        self,
        model_type: str = 'gradient_boosting',
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        score_range: Optional[tuple] = None
    ):
        """
        Initialize the credit risk scorer.
        
        Parameters
        ----------
        model_type : str, default='gradient_boosting'
            Type of regression algorithm
        model_params : Dict[str, Any], optional
            Model hyperparameters
        random_state : int, default=42
            Random seed for reproducibility
        score_range : tuple, optional
            (min_score, max_score) for prediction clipping
        """
        super().__init__(model_type, model_params, random_state)
        self.score_range = score_range
        
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )
    
    def _create_model(self) -> Any:
        """
        Create the underlying model based on model_type.
        
        Returns
        -------
        Any
            Model instance
        """
        if self.model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**self.model_params)
            
        elif self.model_type == 'ridge':
            from sklearn.linear_model import Ridge
            default_params = {'random_state': self.random_state}
            params = {**default_params, **self.model_params}
            return Ridge(**params)
            
        elif self.model_type == 'lasso':
            from sklearn.linear_model import Lasso
            default_params = {'random_state': self.random_state}
            params = {**default_params, **self.model_params}
            return Lasso(**params)
            
        elif self.model_type == 'elastic_net':
            from sklearn.linear_model import ElasticNet
            default_params = {'random_state': self.random_state}
            params = {**default_params, **self.model_params}
            return ElasticNet(**params)
            
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            default_params = {
                'random_state': self.random_state,
                'n_estimators': 100
            }
            params = {**default_params, **self.model_params}
            return RandomForestRegressor(**params)
            
        elif self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            default_params = {
                'random_state': self.random_state,
                'n_estimators': 100
            }
            params = {**default_params, **self.model_params}
            return GradientBoostingRegressor(**params)
            
        elif self.model_type == 'xgboost':
            # TODO: Install and import xgboost
            # import xgboost as xgb
            # default_params = {
            #     'random_state': self.random_state,
            #     'n_estimators': 100
            # }
            # params = {**default_params, **self.model_params}
            # return xgb.XGBRegressor(**params)
            raise NotImplementedError("XGBoost not yet implemented")
            
        elif self.model_type == 'lightgbm':
            # TODO: Install and import lightgbm
            raise NotImplementedError("LightGBM not yet implemented")
            
        elif self.model_type == 'svr':
            from sklearn.svm import SVR
            return SVR(**self.model_params)
            
        elif self.model_type == 'neural_network':
            # TODO: Implement neural network regressor
            raise NotImplementedError("Neural Network not yet implemented")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[tuple] = None,
        **kwargs
    ) -> 'CreditRiskScorer':
        """
        Fit the regression model.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features
        y : pd.Series or np.ndarray
            Target risk scores
        validation_data : tuple, optional
            (X_val, y_val) for validation during training
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        self
            Fitted model instance
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Create and fit model
        self.model = self._create_model()
        
        X_array = self._validate_input(X)
        y_array = np.asarray(y)
        
        # Fit the model
        self.model.fit(X_array, y_array, **kwargs)
        self.is_fitted = True
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict credit risk scores.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
            
        Returns
        -------
        np.ndarray
            Predicted risk scores
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        X_array = self._validate_input(X)
        predictions = self.model.predict(X_array)
        
        # Clip predictions to score range if specified
        if self.score_range is not None:
            predictions = np.clip(
                predictions, 
                self.score_range[0], 
                self.score_range[1]
            )
        
        return predictions
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores.
        
        Returns
        -------
        pd.DataFrame or None
            Feature importance scores if available
        """
        if not self.is_fitted:
            return None
        
        # Get importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            return None
        
        feature_names = self.feature_names or [
            f'feature_{i}' for i in range(len(importances))
        ]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def score_to_rating(
        self,
        scores: np.ndarray,
        thresholds: Optional[Dict[str, tuple]] = None
    ) -> np.ndarray:
        """
        Convert risk scores to rating categories.
        
        Parameters
        ----------
        scores : np.ndarray
            Risk scores to convert
        thresholds : Dict[str, tuple], optional
            Rating thresholds mapping rating to (min, max)
            
        Returns
        -------
        np.ndarray
            Rating categories
            
        Examples
        --------
        >>> thresholds = {
        ...     'AAA': (800, 1000),
        ...     'AA': (700, 800),
        ...     'A': (600, 700),
        ...     'BBB': (500, 600),
        ...     'BB': (400, 500),
        ...     'B': (300, 400),
        ...     'CCC': (200, 300),
        ...     'D': (0, 200)
        ... }
        >>> ratings = model.score_to_rating(risk_scores, thresholds)
        """
        if thresholds is None:
            # Default credit rating thresholds
            thresholds = {
                'Excellent': (750, float('inf')),
                'Good': (700, 750),
                'Fair': (650, 700),
                'Poor': (600, 650),
                'Very Poor': (0, 600)
            }
        
        ratings = np.full(len(scores), 'Unknown', dtype=object)
        
        for rating, (min_score, max_score) in thresholds.items():
            mask = (scores >= min_score) & (scores < max_score)
            ratings[mask] = rating
        
        return ratings


if __name__ == "__main__":
    # Example usage
    print("Credit Risk Scorer Module")
    print("=" * 50)
    print("Supported models:", CreditRiskScorer.SUPPORTED_MODELS)
    print()
    print("Usage:")
    print("  from src.models.risk_scorer import CreditRiskScorer")
    print("  model = CreditRiskScorer(model_type='gradient_boosting')")
    print("  model.fit(X_train, y_train)")
    print("  risk_scores = model.predict(X_test)")
