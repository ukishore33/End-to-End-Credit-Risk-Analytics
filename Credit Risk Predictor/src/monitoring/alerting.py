"""
Alerting Module

This module provides alerting functionality for model monitoring.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert:
    """
    Represents a monitoring alert.
    
    Parameters
    ----------
    alert_type : str
        Type of alert
    severity : AlertSeverity
        Severity level
    message : str
        Alert message
    details : Dict, optional
        Additional details
    """
    
    def __init__(
        self,
        alert_type: str,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict] = None
    ):
        """
        Initialize the alert.
        
        Parameters
        ----------
        alert_type : str
            Type of alert
        severity : AlertSeverity
            Severity level
        message : str
            Alert message
        details : Dict, optional
            Additional details
        """
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        self.acknowledged = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_type': self.alert_type,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }
    
    def __str__(self) -> str:
        """String representation of alert."""
        return (
            f"[{self.severity.value.upper()}] {self.alert_type}: "
            f"{self.message} ({self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})"
        )


class AlertManager:
    """
    Manages alerts for model monitoring.
    
    This class provides methods to create, track, and send alerts
    based on monitoring conditions.
    
    Parameters
    ----------
    model_name : str
        Name of the model being monitored
    notification_handlers : List[Callable], optional
        Functions to call when alerts are triggered
        
    Examples
    --------
    >>> manager = AlertManager('loan_default_v1')
    >>> manager.add_rule('accuracy', 0.85, AlertSeverity.WARNING)
    >>> manager.check_and_alert({'accuracy': 0.80})
    """
    
    def __init__(
        self,
        model_name: str,
        notification_handlers: Optional[List[Callable]] = None
    ):
        """
        Initialize the alert manager.
        
        Parameters
        ----------
        model_name : str
            Model identifier
        notification_handlers : List[Callable], optional
            Notification handlers
        """
        self.model_name = model_name
        self.notification_handlers = notification_handlers or []
        self.alerts: List[Alert] = []
        self.rules: List[Dict] = []
    
    def add_rule(
        self,
        metric: str,
        threshold: float,
        severity: AlertSeverity,
        comparison: str = 'below',
        message_template: Optional[str] = None
    ) -> None:
        """
        Add an alerting rule.
        
        Parameters
        ----------
        metric : str
            Metric to monitor
        threshold : float
            Threshold value
        severity : AlertSeverity
            Alert severity
        comparison : str, default='below'
            Comparison type ('below', 'above', 'equals')
        message_template : str, optional
            Custom message template
        """
        self.rules.append({
            'metric': metric,
            'threshold': threshold,
            'severity': severity,
            'comparison': comparison,
            'message_template': message_template
        })
    
    def check_and_alert(
        self,
        metrics: Dict[str, float]
    ) -> List[Alert]:
        """
        Check metrics against rules and generate alerts.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Current metric values
            
        Returns
        -------
        List[Alert]
            Generated alerts
        """
        triggered_alerts = []
        
        for rule in self.rules:
            metric = rule['metric']
            if metric not in metrics:
                continue
            
            value = metrics[metric]
            threshold = rule['threshold']
            comparison = rule['comparison']
            
            triggered = False
            if comparison == 'below' and value < threshold:
                triggered = True
            elif comparison == 'above' and value > threshold:
                triggered = True
            elif comparison == 'equals' and value == threshold:
                triggered = True
            
            if triggered:
                message = rule.get('message_template') or (
                    f"{metric} is {comparison} threshold: "
                    f"{value:.4f} {'<' if comparison == 'below' else '>'} {threshold:.4f}"
                )
                
                alert = Alert(
                    alert_type=f"metric_{metric}",
                    severity=rule['severity'],
                    message=message,
                    details={
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'model': self.model_name
                    }
                )
                
                self.alerts.append(alert)
                triggered_alerts.append(alert)
                
                # Send notifications
                self._send_notifications(alert)
        
        return triggered_alerts
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error in notification handler: {e}")
    
    def add_notification_handler(
        self,
        handler: Callable[[Alert], None]
    ) -> None:
        """
        Add a notification handler.
        
        Parameters
        ----------
        handler : Callable[[Alert], None]
            Function to call with alerts
        """
        self.notification_handlers.append(handler)
    
    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        acknowledged: Optional[bool] = None
    ) -> List[Alert]:
        """
        Get alerts with optional filtering.
        
        Parameters
        ----------
        severity : AlertSeverity, optional
            Filter by severity
        acknowledged : bool, optional
            Filter by acknowledgment status
            
        Returns
        -------
        List[Alert]
            Filtered alerts
        """
        alerts = self.alerts
        
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        return alerts
    
    def acknowledge_alert(self, alert: Alert) -> None:
        """
        Acknowledge an alert.
        
        Parameters
        ----------
        alert : Alert
            Alert to acknowledge
        """
        alert.acknowledged = True
    
    def clear_alerts(self, acknowledged_only: bool = False) -> None:
        """
        Clear alerts.
        
        Parameters
        ----------
        acknowledged_only : bool, default=False
            Only clear acknowledged alerts
        """
        if acknowledged_only:
            self.alerts = [a for a in self.alerts if not a.acknowledged]
        else:
            self.alerts = []
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of current alerts.
        
        Returns
        -------
        Dict[str, Any]
            Alert summary
        """
        return {
            'total': len(self.alerts),
            'critical': len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]),
            'warning': len([a for a in self.alerts if a.severity == AlertSeverity.WARNING]),
            'info': len([a for a in self.alerts if a.severity == AlertSeverity.INFO]),
            'unacknowledged': len([a for a in self.alerts if not a.acknowledged])
        }


def send_alert(
    alert: Alert,
    method: str = 'console',
    **kwargs
) -> bool:
    """
    Send an alert via specified method.
    
    Parameters
    ----------
    alert : Alert
        Alert to send
    method : str, default='console'
        Notification method ('console', 'email', 'slack')
    **kwargs : dict
        Additional arguments for notification method
        
    Returns
    -------
    bool
        True if alert was sent successfully
    """
    if method == 'console':
        print(str(alert))
        if alert.details:
            for key, value in alert.details.items():
                print(f"  {key}: {value}")
        return True
    
    elif method == 'email':
        # TODO: Implement email notification
        # recipient = kwargs.get('recipient')
        # subject = f"[{alert.severity.value.upper()}] {alert.alert_type}"
        # body = alert.message
        # send_email(recipient, subject, body)
        print("TODO: Implement email notifications")
        return False
    
    elif method == 'slack':
        # TODO: Implement Slack notification
        # webhook_url = kwargs.get('webhook_url')
        # message = f"*{alert.alert_type}*\n{alert.message}"
        # send_slack_message(webhook_url, message)
        print("TODO: Implement Slack notifications")
        return False
    
    else:
        raise ValueError(f"Unknown notification method: {method}")


if __name__ == "__main__":
    print("Alerting Module")
    print("=" * 50)
    print("Usage:")
    print("  from src.monitoring.alerting import AlertManager, AlertSeverity")
    print("  manager = AlertManager('model_v1')")
    print("  manager.add_rule('accuracy', 0.85, AlertSeverity.WARNING)")
    print("  alerts = manager.check_and_alert({'accuracy': 0.80})")
