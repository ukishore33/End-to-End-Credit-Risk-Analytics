"""
Configuration Module

This module provides configuration management for the credit risk analytics pipeline.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


def load_config(
    config_path: Union[str, Path],
    env: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to configuration file
    env : str, optional
        Environment name to load (e.g., 'dev', 'prod')
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
        
    Examples
    --------
    >>> config = load_config('configs/model_config.yaml')
    >>> print(config['model']['type'])
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # If environment specified, merge environment-specific config
    if env and env in config:
        base_config = {k: v for k, v in config.items() if not isinstance(v, dict) or k == env}
        env_config = config.get(env, {})
        config = {**base_config, **env_config}
    
    # Replace environment variables
    config = _replace_env_vars(config)
    
    return config


def _replace_env_vars(config: Any) -> Any:
    """Replace ${ENV_VAR} placeholders with environment variables."""
    if isinstance(config, dict):
        return {k: _replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(v) for v in config]
    elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
        env_var = config[2:-1]
        return os.environ.get(env_var, config)
    return config


def get_config_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None
) -> Any:
    """
    Get a value from nested configuration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    key_path : str
        Dot-separated path to value (e.g., 'model.params.n_estimators')
    default : Any, optional
        Default value if key not found
        
    Returns
    -------
    Any
        Configuration value
        
    Examples
    --------
    >>> value = get_config_value(config, 'model.params.n_estimators', default=100)
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def save_config(
    config: Dict[str, Any],
    config_path: Union[str, Path]
) -> None:
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    config_path : str or Path
        Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


if __name__ == "__main__":
    print("Configuration Module")
    print("=" * 50)
    print("Usage:")
    print("  from src.utils.config import load_config")
    print("  config = load_config('configs/model_config.yaml')")
