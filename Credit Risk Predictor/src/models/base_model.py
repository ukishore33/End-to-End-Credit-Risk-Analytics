"""
Base Model Module

This module provides a base class for all credit risk models.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pathlib import Path


class BaseModel(ABC):
    """
    Abstract base class for credit risk models.
    
    All model implementations should inherit from this class
    and implement the abstract methods.
    """
    
    def __init__(
        self,
        model_type: str = 'default',
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """
        Initialize the base model.
        
        Parameters
        ----------
        model_type : str, default='default'
            Type of model algorithm to use
        model_params : Dict[str, Any], optional
            Model hyperparameters
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> 'BaseModel':
        """
        Fit the model on training data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : pd.Series or np.ndarray
            Target variable
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        self
            Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Model predictions
        """
        pass
    
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Validate and convert input data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data
            
        Returns
        -------
        np.ndarray
            Validated array
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                # Ensure columns match training data
                missing_cols = set(self.feature_names) - set(X.columns)
                if missing_cols:
                    raise ValueError(f"Missing columns: {missing_cols}")
                X = X[self.feature_names]
            return X.values
        return np.asarray(X)
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Parameters
        ----------
        path : str
            Path to save model
        """
        import joblib
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, path)
        
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Parameters
        ----------
        path : str
            Path to saved model
            
        Returns
        -------
        BaseModel
            Loaded model instance
        """
        import joblib
        
        model_data = joblib.load(path)
        
        instance = cls(
            model_type=model_data['model_type'],
            model_params=model_data['model_params'],
            random_state=model_data['random_state']
        )
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        return instance
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns
        -------
        Dict[str, Any]
            Model parameters
        """
        return {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Parameters
        ----------
        **params : dict
            Parameters to set
            
        Returns
        -------
        self
            Model instance
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


if __name__ == "__main__":
    print("Base Model Module")
    print("=" * 50)
    print("This module provides the abstract base class for all models.")
    print("Inherit from BaseModel and implement fit() and predict() methods.")
