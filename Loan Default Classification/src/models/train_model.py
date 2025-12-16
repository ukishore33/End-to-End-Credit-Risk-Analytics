"""Model training script for loan eligibility prediction.

This module handles training various machine learning models.
"""

import argparse
import logging
from pathlib import Path
import joblib

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def load_training_data(data_path: str):
    """Load training data.
    
    Args:
        data_path: Path to training data CSV
        
    Returns:
        X_train, y_train
    """
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Assuming 'target' is the label column
    if 'target' in df.columns:
        y = df['target']
        X = df.drop(columns=['target'])
    else:
        raise ValueError("Target column 'target' not found in data")
    
    logger.info(f"Training data loaded. Shape: {X.shape}")
    return X, y


def get_model(model_name: str, **params):
    """Get model instance based on name.
    
    Args:
        model_name: Name of the model
        **params: Model parameters
        
    Returns:
        Model instance
    """
    models = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    logger.info(f"Creating {model_name} model with params: {params}")
    return models[model_name](**params)


def train_model(X_train, y_train, model_name: str, model_params: dict):
    """Train a machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model to train
        model_params: Model hyperparameters
        
    Returns:
        Trained model
    """
    logger.info(f"Training {model_name} model")
    
    model = get_model(model_name, **model_params)
    model.fit(X_train, y_train)
    
    # Calculate training metrics
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    logger.info(f"Training completed. Training accuracy: {train_accuracy:.4f}")
    
    return model


def evaluate_model(model, X, y):
    """Evaluate model performance.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating model")
    
    y_pred = model.predict(X)
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0),
    }
    
    # Calculate AUC-ROC if model has predict_proba
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X)
            metrics['roc_auc'] = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
        except:
            pass
    
    logger.info(f"Metrics: {metrics}")
    return metrics


def save_model(model, output_path: str):
    """Save trained model.
    
    Args:
        model: Trained model
        output_path: Path to save the model
    """
    logger.info(f"Saving model to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info("Model saved successfully")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train loan eligibility prediction model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', 
                       help='Path to model config file')
    parser.add_argument('--model', type=str, default='random_forest', 
                       choices=['logistic_regression', 'random_forest', 'gradient_boosting'],
                       help='Model to train')
    parser.add_argument('--output', type=str, default='models/saved_models/model.pkl',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    model_params = config.get(args.model, {})
    
    # Load data
    X_train, y_train = load_training_data(args.data)
    
    # Train model
    model = train_model(X_train, y_train, args.model, model_params)
    
    # Evaluate on training data
    metrics = evaluate_model(model, X_train, y_train)
    
    # Save model
    save_model(model, args.output)
    
    logger.info("Training pipeline completed successfully")


if __name__ == '__main__':
    main()
