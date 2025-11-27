"""
Binary Classifier Module

This module provides binary classification models for loan default prediction.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union, List
from .base_model import BaseModel


class LoanDefaultClassifier(BaseModel):
    """
    Binary classification model for predicting loan defaults.
    
    This class provides a unified interface for various classification
    algorithms including Logistic Regression, Random Forest, XGBoost,
    LightGBM, and Neural Networks.
    
    Parameters
    ----------
    model_type : str, default='logistic_regression'
        Type of classification algorithm
        Options: 'logistic_regression', 'random_forest', 'xgboost',
                 'lightgbm', 'neural_network', 'ensemble'
    model_params : Dict[str, Any], optional
        Model hyperparameters
    random_state : int, default=42
        Random seed for reproducibility
        
    Examples
    --------
    >>> from src.models.binary_classifier import LoanDefaultClassifier
    >>> 
    >>> # Initialize and train model
    >>> model = LoanDefaultClassifier(model_type='xgboost')
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Make predictions
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
    >>> 
    >>> # Evaluate model
    >>> from sklearn.metrics import classification_report
    >>> print(classification_report(y_test, predictions))
    """
    
    SUPPORTED_MODELS = [
        'logistic_regression',
        'random_forest',
        'xgboost',
        'lightgbm',
        'neural_network',
        'gradient_boosting',
        'svm',
        'ensemble'
    ]
    
    def __init__(
        self,
        model_type: str = 'logistic_regression',
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        class_weight: Optional[Union[str, Dict]] = 'balanced'
    ):
        """
        Initialize the loan default classifier.
        
        Parameters
        ----------
        model_type : str, default='logistic_regression'
            Type of classification algorithm
        model_params : Dict[str, Any], optional
            Model hyperparameters
        random_state : int, default=42
            Random seed for reproducibility
        class_weight : str or Dict, optional
            Class weights for handling imbalanced data
        """
        super().__init__(model_type, model_params, random_state)
        self.class_weight = class_weight
        
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
        if self.model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            default_params = {
                'class_weight': self.class_weight,
                'random_state': self.random_state,
                'max_iter': 1000
            }
            params = {**default_params, **self.model_params}
            return LogisticRegression(**params)
            
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            default_params = {
                'class_weight': self.class_weight,
                'random_state': self.random_state,
                'n_estimators': 100
            }
            params = {**default_params, **self.model_params}
            return RandomForestClassifier(**params)
            
        elif self.model_type == 'xgboost':
            # TODO: Install and import xgboost
            # import xgboost as xgb
            # default_params = {
            #     'random_state': self.random_state,
            #     'n_estimators': 100,
            #     'scale_pos_weight': 1  # Adjust for class imbalance
            # }
            # params = {**default_params, **self.model_params}
            # return xgb.XGBClassifier(**params)
            raise NotImplementedError("XGBoost not yet implemented")
            
        elif self.model_type == 'lightgbm':
            # TODO: Install and import lightgbm
            # import lightgbm as lgb
            # default_params = {
            #     'random_state': self.random_state,
            #     'n_estimators': 100,
            #     'class_weight': self.class_weight
            # }
            # params = {**default_params, **self.model_params}
            # return lgb.LGBMClassifier(**params)
            raise NotImplementedError("LightGBM not yet implemented")
            
        elif self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            default_params = {
                'random_state': self.random_state,
                'n_estimators': 100
            }
            params = {**default_params, **self.model_params}
            return GradientBoostingClassifier(**params)
            
        elif self.model_type == 'svm':
            from sklearn.svm import SVC
            default_params = {
                'class_weight': self.class_weight,
                'random_state': self.random_state,
                'probability': True
            }
            params = {**default_params, **self.model_params}
            return SVC(**params)
            
        elif self.model_type == 'neural_network':
            # TODO: Implement neural network classifier
            raise NotImplementedError("Neural Network not yet implemented")
            
        elif self.model_type == 'ensemble':
            # TODO: Implement ensemble of models
            raise NotImplementedError("Ensemble not yet implemented")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[tuple] = None,
        **kwargs
    ) -> 'LoanDefaultClassifier':
        """
        Fit the classification model.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features
        y : pd.Series or np.ndarray
            Training labels (0 or 1)
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
        Predict loan default (0 or 1).
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
            
        Returns
        -------
        np.ndarray
            Binary predictions (0 = no default, 1 = default)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        X_array = self._validate_input(X)
        return self.model.predict(X_array)
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict probability of loan default.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
            
        Returns
        -------
        np.ndarray
            Probability estimates, shape (n_samples, 2)
            Column 0: P(no default), Column 1: P(default)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        X_array = self._validate_input(X)
        return self.model.predict_proba(X_array)
    
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
            importances = np.abs(self.model.coef_[0])
        else:
            return None
        
        feature_names = self.feature_names or [
            f'feature_{i}' for i in range(len(importances))
        ]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)


if __name__ == "__main__":
    # Example usage
    print("Loan Default Classifier Module")
    print("=" * 50)
    print("Supported models:", LoanDefaultClassifier.SUPPORTED_MODELS)
    print()
    print("Usage:")
    print("  from src.models.binary_classifier import LoanDefaultClassifier")
    print("  model = LoanDefaultClassifier(model_type='random_forest')")
    print("  model.fit(X_train, y_train)")
    print("  predictions = model.predict(X_test)")
