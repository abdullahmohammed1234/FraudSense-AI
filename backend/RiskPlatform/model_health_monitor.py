"""
ModelHealthMonitor Module for FraudSense AI.

Monitors model health and performance with configurable rule-based health checks:
- Rolling average fraud probability
- Drift frequency trend
- Risk distribution changes
- Anomaly score trend
- Detection stability score
"""

import threading
from collections import deque, Counter
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class HealthStatus:
    """Health status constants."""
    HEALTHY = "Healthy"
    WARNING = "Warning"
    DEGRADED = "Degraded"


class ModelHealthMonitor:
    """
    Model health monitoring with rule-based health checks.
    
    Tracks model performance metrics and provides health status
    based on configurable thresholds.
    """
    
    DEFAULT_WINDOW_SIZE = 500
    
    # Default health thresholds (configurable)
    DEFAULT_THRESHOLDS = {
        "max_avg_fraud_probability": 0.30,  # Warning if avg fraud prob > 30%
        "max_drift_frequency": 0.15,  # Warning if drift rate > 15%
        "max_risk_distribution_shift": 0.25,  # Warning if distribution shifts > 25%
        "max_avg_anomaly_score": 0.60,  # Warning if avg anomaly > 60%
        "min_stability_index": 0.70,  # Degraded if stability < 70%
        "max_fraud_rate_variance": 0.10,  # Warning if variance > 10%
    }
    
    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the ModelHealthMonitor.
        
        Args:
            window_size: Size of rolling window for metrics
            thresholds: Custom health thresholds
        """
        self.window_size = window_size
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        
        self._lock = threading.RLock()
        
        # Initialize tracking deques
        self.fraud_probabilities: deque = deque(maxlen=window_size)
        self.anomaly_scores: deque = deque(maxlen=window_size)
        self.risk_bands: deque = deque(maxlen=window_size)
        self.drift_flags: deque = deque(maxlen=window_size)
        
        # Historical data for trend analysis
        self.historical_fraud_probs: deque = deque(maxlen=window_size * 2)
        self.historical_anomaly_scores: deque = deque(maxlen=window_size * 2)
        self.historical_drift_flags: deque = deque(maxlen=window_size * 2)
        
        # Stability tracking
        self.prediction_stability_scores: deque = deque(maxlen=100)
        self.consecutive_predictions: deque = deque(maxlen=20)
        
        # Risk distribution tracking
        self.risk_distribution_history: deque = deque(maxlen=50)
        
        # Start time
        self.start_time = datetime.utcnow()
    
    def record_prediction(
        self,
        fraud_probability: float,
        anomaly_score: float,
        risk_band: str,
        drift_detected: bool,
        actual_label: Optional[bool] = None
    ) -> None:
        """
        Record a prediction for health monitoring.
        
        Args:
            fraud_probability: Fraud probability (0-1)
            anomaly_score: Anomaly score (0-1)
            risk_band: Risk band (Low, Medium, High, Critical)
            drift_detected: Whether drift was detected
            actual_label: Actual fraud label if known
        """
        with self._lock:
            # Add to current window
            self.fraud_probabilities.append(fraud_probability)
            self.anomaly_scores.append(anomaly_score)
            self.risk_bands.append(risk_band)
            self.drift_flags.append(1 if drift_detected else 0)
            
            # Add to historical data
            self.historical_fraud_probs.append(fraud_probability)
            self.historical_anomaly_scores.append(anomaly_score)
            self.historical_drift_flags.append(1 if drift_detected else 0)
            
            # Track stability (consecutive high/low predictions)
            self._update_stability(fraud_probability)
            
            # Update risk distribution history
            if len(self.risk_bands) >= self.window_size:
                self._update_risk_distribution_history()
    
    def _update_stability(self, fraud_probability: float) -> None:
        """Update prediction stability tracking."""
        # Check if we're seeing consistent predictions
        self.consecutive_predictions.append(fraud_probability)
        
        if len(self.consecutive_predictions) >= 10:
            probs = list(self.consecutive_predictions)
            variance = np.var(probs)
            
            # Low variance = high stability
            stability = max(0, 1 - (variance * 10))
            self.prediction_stability_scores.append(stability)
    
    def _update_risk_distribution_history(self) -> None:
        """Update risk distribution history."""
        current_dist = self._get_risk_distribution()
        self.risk_distribution_history.append(current_dist)
    
    def _get_risk_distribution(self) -> Dict[str, int]:
        """Get current risk distribution."""
        counts = Counter(self.risk_bands)
        return {
            "Low": counts.get("Low", 0),
            "Medium": counts.get("Medium", 0),
            "High": counts.get("High", 0),
            "Critical": counts.get("Critical", 0)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current model health status.
        
        Returns:
            Dictionary with health status and metrics
        """
        with self._lock:
            if len(self.fraud_probabilities) < 10:
                return self._get_initializing_status()
            
            return self._calculate_health_status()
    
    def _get_initializing_status(self) -> Dict[str, Any]:
        """Get status when not enough data."""
        return {
            "health_status": HealthStatus.HEALTHY,
            "confidence_score": 0.0,
            "drift_trend": 0.0,
            "performance_trend": 0.0,
            "stability_index": 1.0,
            "status": "initializing",
            "message": "Collecting data for health analysis",
            "sample_size": len(self.fraud_probabilities)
        }
    
    def _calculate_health_status(self) -> Dict[str, Any]:
        """Calculate health status based on all metrics."""
        # Get current metrics
        avg_fraud_prob = np.mean(self.fraud_probabilities)
        drift_frequency = np.mean(self.drift_flags) if self.drift_flags else 0
        avg_anomaly = np.mean(self.anomaly_scores)
        
        # Calculate stability index
        stability_index = np.mean(self.prediction_stability_scores) if self.prediction_stability_scores else 1.0
        
        # Calculate trends
        drift_trend = self._calculate_drift_trend()
        performance_trend = self._calculate_performance_trend()
        risk_distribution_change = self._calculate_risk_distribution_change()
        
        # Determine health status based on thresholds
        health_issues = []
        
        if avg_fraud_prob > self.thresholds["max_avg_fraud_probability"]:
            health_issues.append(f"High average fraud probability: {avg_fraud_prob:.2%}")
        
        if drift_frequency > self.thresholds["max_drift_frequency"]:
            health_issues.append(f"High drift frequency: {drift_frequency:.2%}")
        
        if risk_distribution_change > self.thresholds["max_risk_distribution_shift"]:
            health_issues.append(f"Risk distribution shift: {risk_distribution_change:.2%}")
        
        if avg_anomaly > self.thresholds["max_avg_anomaly_score"]:
            health_issues.append(f"High average anomaly score: {avg_anomaly:.2f}")
        
        if stability_index < self.thresholds["min_stability_index"]:
            health_issues.append(f"Low stability index: {stability_index:.2f}")
        
        # Determine health status
        if len(health_issues) >= 3:
            health_status = HealthStatus.DEGRADED
        elif len(health_issues) >= 1:
            health_status = HealthStatus.WARNING
        else:
            health_status = HealthStatus.HEALTHY
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            avg_fraud_prob, drift_frequency, stability_index, risk_distribution_change
        )
        
        return {
            "health_status": health_status,
            "confidence_score": round(confidence_score, 4),
            "drift_trend": round(drift_trend, 4),
            "performance_trend": round(performance_trend, 4),
            "stability_index": round(stability_index, 4),
            "metrics": {
                "average_fraud_probability": round(avg_fraud_prob, 4),
                "drift_frequency": round(drift_frequency, 4),
                "average_anomaly_score": round(avg_anomaly, 4),
                "risk_distribution": self._get_risk_distribution(),
                "risk_distribution_change": round(risk_distribution_change, 4),
                "sample_size": len(self.fraud_probabilities)
            },
            "health_issues": health_issues,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
        }
    
    def _calculate_confidence_score(
        self,
        avg_fraud_prob: float,
        drift_frequency: float,
        stability_index: float,
        risk_distribution_change: float
    ) -> float:
        """Calculate overall confidence score (0-1)."""
        # Start with base confidence
        confidence = 1.0
        
        # Penalize for high fraud probability
        if avg_fraud_prob > 0.2:
            confidence -= 0.2
        
        # Penalize for high drift frequency
        confidence -= drift_frequency * 0.5
        
        # Penalize for low stability
        confidence -= (1 - stability_index) * 0.3
        
        # Penalize for large distribution changes
        confidence -= risk_distribution_change * 0.2
        
        return max(0, min(1, confidence))
    
    def _calculate_drift_trend(self) -> float:
        """Calculate drift frequency trend (-1 to 1)."""
        if len(self.historical_drift_flags) < 20:
            return 0.0
        
        # Compare recent drift frequency to older data
        flags = list(self.historical_drift_flags)
        mid_point = len(flags) // 2
        
        older_drift_rate = np.mean(flags[:mid_point])
        recent_drift_rate = np.mean(flags[mid_point:])
        
        # Positive = increasing drift
        return recent_drift_rate - older_drift_rate
    
    def _calculate_performance_trend(self) -> float:
        """Calculate fraud probability performance trend (-1 to 1)."""
        if len(self.historical_fraud_probs) < 20:
            return 0.0
        
        probs = list(self.historical_fraud_probs)
        mid_point = len(probs) // 2
        
        older_avg = np.mean(probs[:mid_point])
        recent_avg = np.mean(probs[mid_point:])
        
        # Positive = increasing fraud detection (could indicate issues or real fraud)
        return recent_avg - older_avg
    
    def _calculate_risk_distribution_change(self) -> float:
        """Calculate risk distribution change from history."""
        if len(self.risk_distribution_history) < 2:
            return 0.0
        
        history = list(self.risk_distribution_history)
        old_dist = history[0]
        new_dist = history[-1]
        
        # Calculate total change across all bands
        total_change = 0.0
        for band in ["Low", "Medium", "High", "Critical"]:
            old_pct = old_dist.get(band, 0) / max(1, sum(old_dist.values()))
            new_pct = new_dist.get(band, 0) / max(1, sum(new_dist.values()))
            total_change += abs(new_pct - old_pct)
        
        return total_change / 2  # Normalize to 0-1
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get detailed health metrics.
        
        Returns:
            Detailed metrics dictionary
        """
        with self._lock:
            if len(self.fraud_probabilities) < 10:
                return {"status": "insufficient_data", "sample_size": len(self.fraud_probabilities)}
            
            probs = list(self.fraud_probabilities)
            anomalies = list(self.anomaly_scores)
            
            return {
                "average_fraud_probability": round(np.mean(probs), 4),
                "median_fraud_probability": round(np.median(probs), 4),
                "std_fraud_probability": round(np.std(probs), 4),
                "min_fraud_probability": round(min(probs), 4),
                "max_fraud_probability": round(max(probs), 4),
                "average_anomaly_score": round(np.mean(anomalies), 4),
                "drift_frequency": round(np.mean(self.drift_flags), 4),
                "stability_index": round(
                    np.mean(self.prediction_stability_scores), 4
                ) if self.prediction_stability_scores else 1.0,
                "current_risk_distribution": self._get_risk_distribution(),
                "sample_size": len(probs)
            }
    
    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Update health thresholds.
        
        Args:
            thresholds: New threshold values
        """
        with self._lock:
            self.thresholds.update(thresholds)
    
    def reset(self) -> None:
        """Reset all monitoring data."""
        with self._lock:
            self.fraud_probabilities.clear()
            self.anomaly_scores.clear()
            self.risk_bands.clear()
            self.drift_flags.clear()
            self.historical_fraud_probs.clear()
            self.historical_anomaly_scores.clear()
            self.historical_drift_flags.clear()
            self.prediction_stability_scores.clear()
            self.consecutive_predictions.clear()
            self.risk_distribution_history.clear()
            self.start_time = datetime.utcnow()


# Global health monitor instance
_health_monitor: Optional[ModelHealthMonitor] = None


def get_model_health_monitor() -> ModelHealthMonitor:
    """
    Get or create the global model health monitor instance.
    
    Returns:
        ModelHealthMonitor instance
    """
    global _health_monitor
    
    if _health_monitor is None:
        _health_monitor = ModelHealthMonitor()
    
    return _health_monitor


def reset_model_health() -> None:
    """Reset model health monitoring data."""
    monitor = get_model_health_monitor()
    monitor.reset()


if __name__ == "__main__":
    # Test the model health monitor
    monitor = get_model_health_monitor()
    
    # Record some test predictions
    import random
    random.seed(42)
    
    for i in range(100):
        fraud_prob = random.random() * 0.5  # 0-0.5
        anomaly_score = random.random() * 0.6
        risk_band = random.choice(["Low", "Medium", "High", "Critical"])
        drift = random.random() < 0.1  # 10% drift
        
        monitor.record_prediction(
            fraud_probability=fraud_prob,
            anomaly_score=anomaly_score,
            risk_band=risk_band,
            drift_detected=drift
        )
    
    # Get health status
    health = monitor.get_health_status()
    print("Health Status:")
    for key, value in health.items():
        if key != "health_issues":
            print(f"  {key}: {value}")
    
    if health.get("health_issues"):
        print("\nHealth Issues:")
        for issue in health["health_issues"]:
            print(f"  - {issue}")
    
    # Get detailed metrics
    print("\nDetailed Metrics:")
    metrics = monitor.get_detailed_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
