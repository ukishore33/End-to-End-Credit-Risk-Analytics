"""
Cross Validation Module

This module provides cross-validation strategies for credit risk models.
"""

import pandas as pd
import numpy as np
from typing import Any, Iterator, Optional, Tuple, Union
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    GroupKFold
)


def get_cv_strategy(
    strategy: str = 'stratified',
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    **kwargs
) -> Any:
    """
    Get a cross-validation strategy.
    
    Parameters
    ----------
    strategy : str, default='stratified'
        CV strategy type
        Options: 'kfold', 'stratified', 'time_series', 'group'
    n_splits : int, default=5
        Number of splits
    shuffle : bool, default=True
        Whether to shuffle before splitting
    random_state : int, default=42
        Random seed
    **kwargs : dict
        Additional arguments for specific strategies
        
    Returns
    -------
    Any
        Cross-validator object
        
    Examples
    --------
    >>> cv = get_cv_strategy('stratified', n_splits=5)
    >>> for train_idx, val_idx in cv.split(X, y):
    ...     X_train, X_val = X[train_idx], X[val_idx]
    """
    if strategy == 'kfold':
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None
        )
        
    elif strategy == 'stratified':
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None
        )
        
    elif strategy == 'time_series':
        return TimeSeriesSplit(
            n_splits=n_splits,
            **kwargs
        )
        
    elif strategy == 'group':
        return GroupKFold(n_splits=n_splits)
    
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")


def time_series_cv(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    dates: pd.Series,
    n_splits: int = 5,
    test_size: Optional[int] = None,
    gap: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Time-based cross-validation for credit risk models.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    dates : pd.Series
        Date series for temporal ordering
    n_splits : int, default=5
        Number of splits
    test_size : int, optional
        Fixed test set size (if None, increases with each split)
    gap : int, default=0
        Gap between train and test sets
        
    Yields
    ------
    Tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices)
        
    Examples
    --------
    >>> for train_idx, test_idx in time_series_cv(X, y, dates, n_splits=5):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    ...     y_train, y_test = y[train_idx], y[test_idx]
    """
    # Sort by date
    date_order = np.argsort(dates)
    n_samples = len(X)
    
    # Calculate split points
    if test_size is None:
        test_size = n_samples // (n_splits + 1)
    
    for i in range(n_splits):
        # Calculate indices
        test_end = n_samples - (n_splits - i - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start - gap
        
        train_indices = date_order[:train_end]
        test_indices = date_order[test_start:test_end]
        
        yield train_indices, test_indices


def expanding_window_cv(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    initial_train_size: int,
    test_size: int,
    step_size: int = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding window cross-validation.
    
    The training set grows with each fold while test set remains fixed size.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    initial_train_size : int
        Initial training set size
    test_size : int
        Test set size
    step_size : int, optional
        Step size between folds (default: test_size)
        
    Yields
    ------
    Tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices)
    """
    if step_size is None:
        step_size = test_size
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    train_end = initial_train_size
    
    while train_end + test_size <= n_samples:
        train_indices = indices[:train_end]
        test_indices = indices[train_end:train_end + test_size]
        
        yield train_indices, test_indices
        
        train_end += step_size


def sliding_window_cv(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    train_size: int,
    test_size: int,
    step_size: int = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Sliding window cross-validation.
    
    Both training and test windows move forward with each fold.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    train_size : int
        Training set size (fixed)
    test_size : int
        Test set size
    step_size : int, optional
        Step size between folds (default: test_size)
        
    Yields
    ------
    Tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices)
    """
    if step_size is None:
        step_size = test_size
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    train_start = 0
    
    while train_start + train_size + test_size <= n_samples:
        train_end = train_start + train_size
        test_end = train_end + test_size
        
        train_indices = indices[train_start:train_end]
        test_indices = indices[train_end:test_end]
        
        yield train_indices, test_indices
        
        train_start += step_size


def purged_cv(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    groups: np.ndarray,
    n_splits: int = 5,
    purge_length: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Purged cross-validation to prevent data leakage.
    
    Removes samples from training that are too close to test samples
    in time (commonly used in financial applications).
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    groups : np.ndarray
        Group labels (e.g., time periods)
    n_splits : int, default=5
        Number of splits
    purge_length : int, default=0
        Number of groups to purge around test set
        
    Yields
    ------
    Tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices)
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    group_fold_size = n_groups // n_splits
    
    for i in range(n_splits):
        # Define test groups
        test_start = i * group_fold_size
        test_end = (i + 1) * group_fold_size if i < n_splits - 1 else n_groups
        test_groups = unique_groups[test_start:test_end]
        
        # Define purge zone
        purge_start = max(0, test_start - purge_length)
        purge_end = min(n_groups, test_end + purge_length)
        purge_groups = unique_groups[purge_start:purge_end]
        
        # Get indices
        train_mask = ~np.isin(groups, purge_groups)
        test_mask = np.isin(groups, test_groups)
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        yield train_indices, test_indices


if __name__ == "__main__":
    print("Cross Validation Module")
    print("=" * 50)
    print("Available strategies:")
    print("  - kfold: Standard K-Fold")
    print("  - stratified: Stratified K-Fold")
    print("  - time_series: Time Series Split")
    print("  - group: Group K-Fold")
    print()
    print("Usage:")
    print("  from src.hyperparameter_tuning.cross_validation import get_cv_strategy")
    print("  cv = get_cv_strategy('stratified', n_splits=5)")
