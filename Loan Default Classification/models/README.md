# Models Directory

This directory contains trained models and checkpoints.

## Structure

- **saved_models/**: Serialized production-ready models
  - Final trained models (.pkl, .h5, .pt)
  - Model metadata and configs
  - Version-controlled model artifacts

- **checkpoints/**: Training checkpoints
  - Intermediate model states
  - For resuming training
  - Experiment tracking

## Saved Models

Models are saved in various formats:
- `.pkl` - Scikit-learn models (using joblib)
- `.h5` - Keras/TensorFlow models
- `.pt` or `.pth` - PyTorch models
- `.onnx` - ONNX format for deployment

## Model Naming Convention

Use descriptive names for saved models:
```
{model_type}_{version}_{date}_{metric}.pkl

Example:
random_forest_v1_20231110_acc85.pkl
xgboost_v2_20231111_f1_87.pkl
```

## Model Registry

Keep track of your models:

| Model Name | Type | Date | Accuracy | F1 Score | Notes |
|------------|------|------|----------|----------|-------|
| rf_v1.pkl | Random Forest | 2023-11-10 | 0.85 | 0.83 | Baseline model |
| xgb_v1.pkl | XGBoost | 2023-11-11 | 0.87 | 0.85 | With feature engineering |

## Usage

Save a model:
```python
import joblib
joblib.dump(model, 'models/saved_models/my_model.pkl')
```

Load a model:
```python
import joblib
model = joblib.load('models/saved_models/my_model.pkl')
```

## Model Versioning

Consider using tools like:
- MLflow for experiment tracking
- DVC for model versioning
- Weights & Biases for monitoring

## Security Note

⚠️ Model files are gitignored by default. Use proper model versioning tools or artifact storage for production models.
