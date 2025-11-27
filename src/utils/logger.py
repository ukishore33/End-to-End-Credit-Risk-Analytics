"""
Logger Module

This module provides logging utilities for the credit risk analytics pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = 'credit_risk',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Parameters
    ----------
    name : str, default='credit_risk'
        Logger name
    level : int, default=logging.INFO
        Logging level
    log_file : str, optional
        Path to log file
    format_string : str, optional
        Custom format string
        
    Returns
    -------
    logging.Logger
        Configured logger
        
    Examples
    --------
    >>> logger = setup_logger('my_module', logging.DEBUG)
    >>> logger.info("Starting process...")
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'credit_risk') -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Parameters
    ----------
    name : str, default='credit_risk'
        Logger name
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        setup_logger(name)
    
    return logger


if __name__ == "__main__":
    # Example usage
    logger = setup_logger('test', logging.DEBUG)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
