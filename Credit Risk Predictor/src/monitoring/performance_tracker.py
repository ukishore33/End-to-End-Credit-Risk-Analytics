"""
Performance Tracker Module

This module provides performance tracking for credit risk models in production.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from collections import deque


class PerformanceTracker:
    """
    Track model performance over time.
    
    This class provides methods to log predictions, track metrics,
    and detect performance degradation.
    
    Parameters
    ----------
    model_name : str
        Name of the model being tracked
    window_size : int, default=1000
        Size of rolling window for metrics
    alert_threshold : float, default=0.1
        Threshold for performance degradation alerts
        
    Examples
    --------
    >>> tracker = PerformanceTracker('loan_default_v1')
    >>> tracker.log_prediction(features, 0.75, actual=1)
    >>> report = tracker.generate_report()
    """
    
    def __init__(
        self,
        model_name: str,
        window_size: int = 1000,
        alert_threshold: float = 0.1,
        baseline_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the performance tracker.
        
        Parameters
        ----------
        model_name : str
            Model identifier
        window_size : int, default=1000
            Rolling window size
        alert_threshold : float, default=0.1
            Degradation threshold
        baseline_metrics : Dict[str, float], optional
            Baseline performance metrics
        """
        self.model_name = model_name
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.baseline_metrics = baseline_metrics or {}
        
        # Storage for predictions
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.features_logged = deque(maxlen=window_size)
        
        # Historical metrics
        self.metrics_history = []
    
    def log_prediction(
        self,
        features: Union[Dict, np.ndarray],
        prediction: float,
        actual: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Log a prediction for tracking.
        
        Parameters
        ----------
        features : Dict or np.ndarray
            Input features
        prediction : float
            Model prediction (probability or score)
        actual : int, optional
            Actual outcome (if known)
        timestamp : datetime, optional
            Prediction timestamp
        """
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp or datetime.now())
        self.features_logged.append(features)
    
    def update_actual(
        self,
        index: int,
        actual: int
    ) -> None:
        """
        Update actual outcome for a logged prediction.
        
        Parameters
        ----------
        index : int
            Index of prediction to update
        actual : int
            Actual outcome
        """
        if 0 <= index < len(self.actuals):
            self.actuals[index] = actual
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Calculate current performance metrics.
        
        Returns
        -------
        Dict[str, float]
            Current performance metrics
        """
        # Filter to only records with actual outcomes
        valid_idx = [i for i, a in enumerate(self.actuals) if a is not None]
        
        if len(valid_idx) < 10:  # Minimum samples needed
            return {'error': 'Insufficient data with actual outcomes'}
        
        predictions = np.array([self.predictions[i] for i in valid_idx])
        actuals = np.array([self.actuals[i] for i in valid_idx])
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
        
        pred_binary = (predictions >= 0.5).astype(int)
        
        metrics = {
            'n_samples': len(valid_idx),
            'accuracy': accuracy_score(actuals, pred_binary),
            'precision': precision_score(actuals, pred_binary, zero_division=0),
            'recall': recall_score(actuals, pred_binary, zero_division=0),
            'f1_score': f1_score(actuals, pred_binary, zero_division=0),
            'auc_roc': roc_auc_score(actuals, predictions),
            'mean_prediction': predictions.mean(),
            'positive_rate': actuals.mean()
        }
        
        return metrics
    
    def check_degradation(self) -> Dict[str, Any]:
        """
        Check for performance degradation.
        
        Returns
        -------
        Dict[str, Any]
            Degradation check results
        """
        current_metrics = self.get_current_metrics()
        
        if 'error' in current_metrics:
            return {'degradation_detected': False, 'reason': current_metrics['error']}
        
        result = {
            'degradation_detected': False,
            'degraded_metrics': [],
            'current_metrics': current_metrics,
            'baseline_metrics': self.baseline_metrics
        }
        
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                
                # Check for significant degradation
                if baseline_value > 0:
                    degradation = (baseline_value - current_value) / baseline_value
                    
                    if degradation > self.alert_threshold:
                        result['degradation_detected'] = True
                        result['degraded_metrics'].append({
                            'metric': metric,
                            'baseline': baseline_value,
                            'current': current_value,
                            'degradation_pct': degradation * 100
                        })
        
        return result
    
    def generate_report(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate performance tracking report.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save report
            
        Returns
        -------
        str
            Formatted report
        """
        current_metrics = self.get_current_metrics()
        degradation = self.check_degradation()
        
        lines = [
            "=" * 60,
            f"PERFORMANCE TRACKING REPORT - {self.model_name}",
            "=" * 60,
            f"Report Generated: {datetime.now().isoformat()}",
            f"Window Size: {self.window_size}",
            f"Total Predictions Logged: {len(self.predictions)}",
            ""
        ]
        
        if 'error' not in current_metrics:
            lines.append("CURRENT METRICS")
            lines.append("-" * 40)
            for metric, value in current_metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {metric}: {value:.4f}")
                else:
                    lines.append(f"  {metric}: {value}")
            lines.append("")
        
        if degradation['degradation_detected']:
            lines.append("⚠️  PERFORMANCE DEGRADATION DETECTED")
            lines.append("-" * 40)
            for item in degradation['degraded_metrics']:
                lines.append(f"  {item['metric']}:")
                lines.append(f"    Baseline: {item['baseline']:.4f}")
                lines.append(f"    Current: {item['current']:.4f}")
                lines.append(f"    Degradation: {item['degradation_pct']:.1f}%")
            lines.append("")
        
        lines.append("=" * 60)
        
        report_text = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def get_prediction_distribution(
        self,
        time_window: Optional[timedelta] = None
    ) -> pd.DataFrame:
        """
        Get prediction distribution over time.
        
        Parameters
        ----------
        time_window : timedelta, optional
            Time window to analyze
            
        Returns
        -------
        pd.DataFrame
            Prediction distribution data
        """
        if time_window:
            cutoff = datetime.now() - time_window
            valid_idx = [i for i, t in enumerate(self.timestamps) if t >= cutoff]
        else:
            valid_idx = range(len(self.predictions))
        
        data = {
            'timestamp': [self.timestamps[i] for i in valid_idx],
            'prediction': [self.predictions[i] for i in valid_idx],
            'actual': [self.actuals[i] for i in valid_idx]
        }
        
        return pd.DataFrame(data)


def track_prediction(
    tracker: PerformanceTracker,
    features: Union[Dict, np.ndarray],
    prediction: float,
    actual: Optional[int] = None
) -> None:
    """
    Quick function to track a prediction.
    
    Parameters
    ----------
    tracker : PerformanceTracker
        Tracker instance
    features : Dict or np.ndarray
        Input features
    prediction : float
        Model prediction
    actual : int, optional
        Actual outcome
    """
    tracker.log_prediction(features, prediction, actual)


if __name__ == "__main__":
    print("Performance Tracker Module")
    print("=" * 50)
    print("Usage:")
    print("  from src.monitoring.performance_tracker import PerformanceTracker")
    print("  tracker = PerformanceTracker('model_v1')")
    print("  tracker.log_prediction(features, 0.75, actual=1)")
    print("  report = tracker.generate_report()")
