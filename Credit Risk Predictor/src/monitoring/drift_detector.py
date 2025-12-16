"""
Drift Detector Module

This module provides data and concept drift detection for credit risk models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime


class DriftDetector:
    """
    Drift detection for model monitoring.
    
    This class provides methods to detect data drift and concept drift
    in credit risk model inputs.
    
    Parameters
    ----------
    reference_data : pd.DataFrame
        Reference/baseline data (typically training data)
    feature_names : List[str], optional
        Names of features to monitor
    psi_threshold : float, default=0.25
        PSI threshold for significant drift
    ks_threshold : float, default=0.05
        KS test p-value threshold
        
    Examples
    --------
    >>> detector = DriftDetector(X_train)
    >>> report = detector.detect_drift(X_new)
    >>> if report['drift_detected']:
    ...     print("Drift detected in features:", report['drifted_features'])
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        psi_threshold: float = 0.25,
        ks_threshold: float = 0.05
    ):
        """
        Initialize the drift detector.
        
        Parameters
        ----------
        reference_data : pd.DataFrame
            Reference data distribution
        feature_names : List[str], optional
            Features to monitor
        psi_threshold : float, default=0.25
            PSI threshold for drift
        ks_threshold : float, default=0.05
            KS test threshold
        """
        self.reference_data = reference_data
        self.feature_names = feature_names or reference_data.columns.tolist()
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        
        # Store reference statistics
        self.reference_stats = self._compute_reference_stats()
    
    def _compute_reference_stats(self) -> Dict[str, Any]:
        """Compute and store reference data statistics."""
        stats = {}
        
        for feature in self.feature_names:
            if feature not in self.reference_data.columns:
                continue
                
            col = self.reference_data[feature]
            
            if pd.api.types.is_numeric_dtype(col):
                stats[feature] = {
                    'type': 'numeric',
                    'mean': col.mean(),
                    'std': col.std(),
                    'min': col.min(),
                    'max': col.max(),
                    'median': col.median(),
                    'quantiles': col.quantile([0.25, 0.5, 0.75]).to_dict()
                }
            else:
                stats[feature] = {
                    'type': 'categorical',
                    'value_counts': col.value_counts(normalize=True).to_dict(),
                    'unique_values': col.unique().tolist()
                }
        
        return stats
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        methods: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Parameters
        ----------
        current_data : pd.DataFrame
            Current data to check for drift
        methods : List[str], optional
            Drift detection methods to use
            Default: ['psi', 'ks']
            
        Returns
        -------
        Dict[str, Any]
            Drift detection report
        """
        if methods is None:
            methods = ['psi', 'ks']
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_reference_samples': len(self.reference_data),
            'n_current_samples': len(current_data),
            'drift_detected': False,
            'drifted_features': [],
            'feature_reports': {}
        }
        
        for feature in self.feature_names:
            if feature not in current_data.columns:
                continue
            
            feature_report = self._check_feature_drift(
                feature, current_data[feature], methods
            )
            report['feature_reports'][feature] = feature_report
            
            if feature_report['drift_detected']:
                report['drift_detected'] = True
                report['drifted_features'].append(feature)
        
        return report
    
    def _check_feature_drift(
        self,
        feature: str,
        current_values: pd.Series,
        methods: List[str]
    ) -> Dict[str, Any]:
        """Check drift for a single feature."""
        reference_values = self.reference_data[feature]
        feature_stats = self.reference_stats.get(feature, {})
        
        result = {
            'drift_detected': False,
            'methods_triggered': [],
            'details': {}
        }
        
        if feature_stats.get('type') == 'numeric':
            # PSI test
            if 'psi' in methods:
                psi_value, _ = calculate_psi(reference_values, current_values)
                result['details']['psi'] = psi_value
                if psi_value >= self.psi_threshold:
                    result['drift_detected'] = True
                    result['methods_triggered'].append('psi')
            
            # KS test
            if 'ks' in methods:
                from scipy import stats
                ks_stat, p_value = stats.ks_2samp(reference_values, current_values)
                result['details']['ks_statistic'] = ks_stat
                result['details']['ks_p_value'] = p_value
                if p_value < self.ks_threshold:
                    result['drift_detected'] = True
                    result['methods_triggered'].append('ks')
            
            # Basic statistics comparison
            result['details']['reference_mean'] = float(reference_values.mean())
            result['details']['current_mean'] = float(current_values.mean())
            result['details']['mean_shift'] = float(
                current_values.mean() - reference_values.mean()
            )
            
        else:  # Categorical
            # Chi-square test
            if 'chi2' in methods:
                # TODO: Implement chi-square test
                pass
            
            # Value distribution change
            ref_dist = reference_values.value_counts(normalize=True)
            curr_dist = current_values.value_counts(normalize=True)
            result['details']['reference_distribution'] = ref_dist.to_dict()
            result['details']['current_distribution'] = curr_dist.to_dict()
        
        return result
    
    def generate_drift_report(
        self,
        drift_result: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate human-readable drift report.
        
        Parameters
        ----------
        drift_result : Dict[str, Any]
            Output from detect_drift()
        output_path : str, optional
            Path to save report
            
        Returns
        -------
        str
            Formatted report
        """
        lines = [
            "=" * 60,
            "DATA DRIFT REPORT",
            "=" * 60,
            f"Timestamp: {drift_result['timestamp']}",
            f"Reference Samples: {drift_result['n_reference_samples']}",
            f"Current Samples: {drift_result['n_current_samples']}",
            "",
            f"Overall Drift Detected: {'YES' if drift_result['drift_detected'] else 'NO'}",
            ""
        ]
        
        if drift_result['drift_detected']:
            lines.append("DRIFTED FEATURES:")
            lines.append("-" * 40)
            for feature in drift_result['drifted_features']:
                report = drift_result['feature_reports'][feature]
                lines.append(f"  • {feature}")
                lines.append(f"    Methods triggered: {report['methods_triggered']}")
                for key, value in report['details'].items():
                    if isinstance(value, float):
                        lines.append(f"    {key}: {value:.4f}")
            lines.append("")
        
        lines.append("=" * 60)
        
        report_text = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Quick function to detect data drift.
    
    Parameters
    ----------
    reference : pd.DataFrame
        Reference data
    current : pd.DataFrame
        Current data
    threshold : float, default=0.25
        PSI threshold
        
    Returns
    -------
    Dict[str, Any]
        Drift detection results
    """
    detector = DriftDetector(reference, psi_threshold=threshold)
    return detector.detect_drift(current)


def calculate_psi(
    expected: Union[pd.Series, np.ndarray],
    actual: Union[pd.Series, np.ndarray],
    n_bins: int = 10
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate Population Stability Index (PSI).
    
    Parameters
    ----------
    expected : pd.Series or np.ndarray
        Expected distribution (reference)
    actual : pd.Series or np.ndarray
        Actual distribution (current)
    n_bins : int, default=10
        Number of bins
        
    Returns
    -------
    Tuple[float, pd.DataFrame]
        (PSI value, breakdown by bin)
        
    Notes
    -----
    PSI Interpretation:
    - < 0.10: No significant shift
    - 0.10 - 0.25: Moderate shift
    - >= 0.25: Significant shift
    """
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    
    # Create bins based on expected distribution
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # Calculate distributions
    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]
    
    # Add small epsilon to avoid log(0)
    epsilon = 0.0001
    expected_pct = (expected_counts + epsilon) / (len(expected) + epsilon * n_bins)
    actual_pct = (actual_counts + epsilon) / (len(actual) + epsilon * n_bins)
    
    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = float(np.sum(psi_values))
    
    breakdown = pd.DataFrame({
        'bin': range(1, n_bins + 1),
        'expected_pct': expected_pct,
        'actual_pct': actual_pct,
        'psi_contribution': psi_values
    })
    
    return psi, breakdown


if __name__ == "__main__":
    print("Drift Detector Module")
    print("=" * 50)
    print("Usage:")
    print("  from src.monitoring.drift_detector import DriftDetector")
    print("  detector = DriftDetector(X_train)")
    print("  report = detector.detect_drift(X_new)")
