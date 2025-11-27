"""
Helper Functions Module

This module provides general helper functions for the credit risk analytics pipeline.
"""

import os
import random
import time
import functools
import numpy as np
from typing import Any, Callable


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value
        
    Examples
    --------
    >>> set_seed(42)
    >>> # All random operations will be reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set seeds for deep learning frameworks if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Parameters
    ----------
    func : Callable
        Function to time
        
    Returns
    -------
    Callable
        Wrapped function
        
    Examples
    --------
    >>> @timer
    ... def my_function():
    ...     time.sleep(1)
    >>> my_function()  # Prints execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper


def memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns
    -------
    float
        Memory usage in megabytes
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 2
    except ImportError:
        return 0.0


def flatten_dict(
    d: dict,
    parent_key: str = '',
    sep: str = '.'
) -> dict:
    """
    Flatten a nested dictionary.
    
    Parameters
    ----------
    d : dict
        Dictionary to flatten
    parent_key : str, default=''
        Parent key prefix
    sep : str, default='.'
        Separator for keys
        
    Returns
    -------
    dict
        Flattened dictionary
        
    Examples
    --------
    >>> d = {'a': {'b': 1, 'c': 2}, 'd': 3}
    >>> flatten_dict(d)
    {'a.b': 1, 'a.c': 2, 'd': 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunks(lst: list, n: int):
    """
    Yield successive n-sized chunks from list.
    
    Parameters
    ----------
    lst : list
        List to chunk
    n : int
        Chunk size
        
    Yields
    ------
    list
        Chunks of size n
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    print("Helper Functions Module")
    print("=" * 50)
    print("Functions:")
    print("  - set_seed(seed)")
    print("  - timer(func)")
    print("  - memory_usage()")
    print("  - flatten_dict(d)")
    print("  - chunks(lst, n)")
