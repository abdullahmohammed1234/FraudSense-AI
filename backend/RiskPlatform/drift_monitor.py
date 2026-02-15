"""
DriftMonitor Module for FraudSense AI.

Monitors concept drift in input data and predictions to detect
model degradation over time.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from datetime import datetime

from .config import DriftConfig


class DriftMonitor:
    """
    Monitors concept drift in the fraud detection system.
    
    Tracks feature distributions and prediction patterns to detect
    when the input data has shifted significantly from training data.
    """
    
    def __init__(
        self,
        drift_threshold: float = DriftConfig.DRIFT_THRESHOLD,
        window_size: int = DriftConfig.WINDOW_SIZE
    ):
        """
        Initialize the DriftMonitor.
        
        Args:
            drift_threshold: Z-score threshold for drift detection.
            window_size: Size of rolling window for tracking.
        """
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        
        # Rolling windows for predictions
        self.prediction_window: deque = deque(maxlen=window_size)
        self.feature_windows: Dict[str, deque] = {}
        
        # Drift statistics
        self.total_checks = 0
        self.drift_detected_count = 0
        self.drift_history: List[Dict[str, Any]] = []
        
        # Training statistics (to be set externally)
        self.training_mean: Optional[np.ndarray] = None
        self.training_std: Optional[np.ndarray] = None
    
    def set_training_statistics(
        self,
        mean: np.ndarray,
        std: np.ndarray
    ) -> None:
        """
        Set training statistics for drift comparison.
        
        Args:
            mean: Training feature means.
            std: Training feature standard deviations.
        """
        self.training_mean = mean
        self.training_std = std
    
    def check_drift(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Check for drift in the input features.
        
        Args:
            features: Input feature array.
            
        Returns:
            Tuple of (drift_detected, drift_score).
        """
        if self.training_mean is None or self.training_std is None:
            return False, 0.0
        
        self.total_checks += 1
        
        # Calculate z-score for each feature
        z_scores = np.abs((features - self.training_mean) / (self.training_std + 1e-10))
        
        # Average z-score
        avg_z_score = np.mean(z_scores)
        max_z_score = np.max(z_scores)
        
        # Drift detected if average z-score exceeds threshold
        drift_detected = avg_z_score > self.drift_threshold
        
        if drift_detected:
            self.drift_detected_count += 1
            
            # Record drift event
            self.drift_history.append({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "avg_z_score": float(avg_z_score),
                "max_z_score": float(max_z_score),
                "feature_count": len(z_scores)
            })
            
            # Keep only recent history
            if len(self.drift_history) > 100:
                self.drift_history = self.drift_history[-100:]
        
        return drift_detected, float(avg_z_score)
    
    def update_prediction(self, fraud_probability: float) -> None:
        """
        Update prediction tracking.
        
        Args:
            fraud_probability: Fraud probability from prediction.
        """
        self.prediction_window.append(fraud_probability)
    
    def check_prediction_drift(self) -> Tuple[bool, float]:
        """
        Check for drift in prediction patterns.
        
        Returns:
            Tuple of (drift_detected, drift_score).
        """
        if len(self.prediction_window) < 10:
            return False, 0.0
        
        # Calculate recent mean
        recent_mean = np.mean(self.prediction_window)
        recent_std = np.std(self.prediction_window)
        
        # Check if recent predictions are significantly different from expected
        # (assuming fraud rate around 0.0017 for credit card data)
        expected_fraud_rate = 0.0017
        
        # Calculate drift score based on deviation from expected
        drift_score = abs(recent_mean - expected_fraud_rate) / (expected_fraud_rate + 1e-10)
        
        # Drift detected if score is very high
        drift_detected = drift_score > 3.0  # 3x expected rate
        
        return drift_detected, float(drift_score)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current drift monitoring status.
        
        Returns:
            Dictionary with drift status.
        """
        # Determine overall status
        if self.total_checks == 0:
            status = "no_data"
            status_color = "gray"
        elif self.drift_detected_count / self.total_checks > 0.1:
            status = "critical"
            status_color = "red"
        elif self.drift_detected_count / self.total_checks > 0.05:
            status = "warning"
            status_color = "yellow"
        else:
            status = "healthy"
            status_color = "green"
        
        return {
            "status": status,
            "status_color": status_color,
            "total_checks": self.total_checks,
            "drift_detected_count": self.drift_detected_count,
            "drift_rate": round(self.drift_detected_count / max(self.total_checks, 1), 4),
            "recent_drift_events": self.drift_history[-10:] if self.drift_history else [],
            "window_size": self.window_size,
            "threshold": self.drift_threshold
        }
    
    def reset(self) -> None:
        """Reset all drift monitoring data."""
        self.prediction_window.clear()
        self.feature_windows.clear()
        self.total_checks = 0
        self.drift_detected_count = 0
        self.drift_history.clear()


# Global drift monitor instance
_drift_monitor: Optional[DriftMonitor] = None


def get_drift_monitor() -> DriftMonitor:
    """
    Get or create the global drift monitor instance.
    
    Returns:
        DriftMonitor instance.
    """
    global _drift_monitor
    
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    
    return _drift_monitor


if __name__ == "__main__":
    # Test the drift monitor
    monitor = get_drift_monitor()
    
    # Set mock training statistics
    mean = np.random.randn(30)
    std = np.ones(30) * 2
    monitor.set_training_statistics(mean, std)
    
    # Test drift detection
    features = np.random.randn(30) * 3  # Higher variance than training
    drift_detected, score = monitor.check_drift(features)
    
    print(f"Drift detected: {drift_detected}")
    print(f"Drift score: {score}")
    
    # Get status
    status = monitor.get_status()
    print(f"Status: {status}")
