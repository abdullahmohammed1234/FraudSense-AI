"""
MetricsTracker Module for FraudSense AI.

Thread-safe live metrics engine that tracks real-time statistics including:
- total_transactions
- total_fraud_detected
- auto_block_count
- manual_review_count
- approval_count
- drift_events
- average_inference_time_ms
- p95_latency
- fraud_rate
- risk_band_distribution
- rolling_window_metrics (last 500 predictions)
"""

import threading
from collections import deque, Counter
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np


class MetricsTracker:
    """
    Thread-safe metrics tracker for real-time fraud detection statistics.
    
    Provides comprehensive metrics tracking with rolling window support
    and atomic counter operations.
    """
    
    ROLLING_WINDOW_SIZE = 500
    
    def __init__(self):
        """Initialize the MetricsTracker with thread-safe counters."""
        self._lock = threading.Lock()
        self._reset_counters()
        
    def _reset_counters(self) -> None:
        """Initialize all counters to zero."""
        # Core transaction counters
        self.total_transactions = 0
        self.total_fraud_detected = 0
        self.auto_block_count = 0
        self.manual_review_count = 0
        self.approval_count = 0
        self.drift_events = 0
        
        # Latency tracking
        self.total_latency_ms = 0.0
        self.max_latency_ms = 0.0
        self.min_latency_ms = float('inf')
        
        # Rolling window buffers (last 500 predictions)
        self.rolling_predictions: deque = deque(maxlen=self.ROLLING_WINDOW_SIZE)
        self.rolling_latencies: deque = deque(maxlen=self.ROLLING_WINDOW_SIZE)
        
        # Aggregated data for quick calculations
        self.fraud_probabilities: deque = deque(maxlen=self.ROLLING_WINDOW_SIZE)
        self.anomaly_scores: deque = deque(maxlen=self.ROLLING_WINDOW_SIZE)
        self.risk_scores: deque = deque(maxlen=self.ROLLING_WINDOW_SIZE)
        
        # Risk band distribution
        self.risk_band_counts: Counter = Counter()
        
        # Timestamp tracking
        self.start_time = datetime.utcnow()
        self.last_reset = datetime.utcnow()
    
    def record_prediction(
        self,
        fraud_probability: float,
        anomaly_score: float,
        risk_score: float,
        risk_band: str,
        action: str,
        latency_ms: float,
        drift_detected: bool = False
    ) -> None:
        """
        Record a prediction with all metrics.
        
        Args:
            fraud_probability: Fraud probability (0-1)
            anomaly_score: Anomaly score (0-1)
            risk_score: Ensemble risk score (0-1)
            risk_band: Risk band (Low, Medium, High, Critical)
            action: Action taken (Auto-Block, Manual Review, Approve)
            latency_ms: Inference latency in milliseconds
            drift_detected: Whether drift was detected
        """
        with self._lock:
            # Core counters
            self.total_transactions += 1
            
            if fraud_probability >= 0.5:
                self.total_fraud_detected += 1
            
            if "Block" in action:
                self.auto_block_count += 1
            elif "Review" in action:
                self.manual_review_count += 1
            elif "Approve" in action:
                self.approval_count += 1
            
            if drift_detected:
                self.drift_events += 1
            
            # Latency tracking
            self.total_latency_ms += latency_ms
            self.max_latency_ms = max(self.max_latency_ms, latency_ms)
            self.min_latency_ms = min(self.min_latency_ms, latency_ms)
            
            # Rolling window data
            prediction_record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "fraud_probability": fraud_probability,
                "anomaly_score": anomaly_score,
                "risk_score": risk_score,
                "risk_band": risk_band,
                "action": action,
                "latency_ms": latency_ms,
                "drift_detected": drift_detected
            }
            self.rolling_predictions.append(prediction_record)
            self.rolling_latencies.append(latency_ms)
            
            # Aggregated data
            self.fraud_probabilities.append(fraud_probability)
            self.anomaly_scores.append(anomaly_score)
            self.risk_scores.append(risk_score)
            
            # Risk band distribution
            self.risk_band_counts[risk_band] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all current metrics as a dictionary.
        
        Returns:
            Dictionary with all real-time metrics
        """
        with self._lock:
            return self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate all metrics from current state."""
        # Calculate fraud rate
        fraud_rate = 0.0
        if self.total_transactions > 0:
            fraud_rate = (self.total_fraud_detected / self.total_transactions) * 100
        
        # Calculate average latency
        avg_latency = 0.0
        if self.total_transactions > 0:
            avg_latency = self.total_latency_ms / self.total_transactions
        
        # Calculate p95 latency from rolling window
        p95_latency = self._calculate_percentile(self.rolling_latencies, 0.95)
        
        # Get risk band distribution
        risk_band_dist = {
            "Low": self.risk_band_counts.get("Low", 0),
            "Medium": self.risk_band_counts.get("Medium", 0),
            "High": self.risk_band_counts.get("High", 0),
            "Critical": self.risk_band_counts.get("Critical", 0)
        }
        
        # Calculate auto-block and manual review percentages
        auto_block_pct = 0.0
        manual_review_pct = 0.0
        approval_pct = 0.0
        if self.total_transactions > 0:
            auto_block_pct = (self.auto_block_count / self.total_transactions) * 100
            manual_review_pct = (self.manual_review_count / self.total_transactions) * 100
            approval_pct = (self.approval_count / self.total_transactions) * 100
        
        # Get rolling window data
        rolling_data = list(self.rolling_predictions)
        
        return {
            "total_transactions": self.total_transactions,
            "total_fraud_detected": self.total_fraud_detected,
            "auto_block_count": self.auto_block_count,
            "manual_review_count": self.manual_review_count,
            "approval_count": self.approval_count,
            "drift_events": self.drift_events,
            "average_inference_time_ms": round(avg_latency, 2),
            "p95_latency": round(p95_latency, 2),
            "fraud_rate": round(fraud_rate, 4),
            "risk_band_distribution": risk_band_dist,
            "auto_block_percentage": round(auto_block_pct, 2),
            "manual_review_percentage": round(manual_review_pct, 2),
            "approval_percentage": round(approval_pct, 2),
            "rolling_window_metrics": rolling_data,
            "rolling_window_size": len(rolling_data),
            "max_latency_ms": round(self.max_latency_ms, 2) if self.max_latency_ms != 0 else 0,
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float('inf') else 0,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
        }
    
    def _calculate_percentile(self, data: deque, percentile: float) -> float:
        """Calculate percentile from deque data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * percentile)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]
    
    def get_rolling_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from the rolling window.
        
        Returns:
            Rolling window statistics
        """
        with self._lock:
            if not self.fraud_probabilities:
                return {
                    "sample_size": 0,
                    "average_fraud_probability": 0.0,
                    "average_anomaly_score": 0.0,
                    "average_risk_score": 0.0,
                    "average_latency_ms": 0.0,
                    "fraud_detection_rate": 0.0,
                    "auto_block_rate": 0.0
                }
            
            probs = list(self.fraud_probabilities)
            anomalies = list(self.anomaly_scores)
            risks = list(self.risk_scores)
            latencies = list(self.rolling_latencies)
            
            fraud_count = sum(1 for p in probs if p >= 0.5)
            block_count = sum(1 for p in self.rolling_predictions if "Block" in p.get("action", ""))
            
            return {
                "sample_size": len(probs),
                "average_fraud_probability": round(np.mean(probs), 4),
                "average_anomaly_score": round(np.mean(anomalies), 4),
                "average_risk_score": round(np.mean(risks), 4),
                "average_latency_ms": round(np.mean(latencies), 2) if latencies else 0,
                "fraud_detection_rate": round((fraud_count / len(probs)) * 100, 2),
                "auto_block_rate": round((block_count / len(probs)) * 100, 2),
                "p50_latency": round(np.percentile(latencies, 50), 2) if latencies else 0,
                "p95_latency": round(np.percentile(latencies, 95), 2) if latencies else 0,
                "p99_latency": round(np.percentile(latencies, 99), 2) if latencies else 0
            }
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._reset_counters()
    
    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get a lightweight snapshot of current metrics.
        
        Returns:
            Lightweight metrics snapshot
        """
        with self._lock:
            fraud_rate = 0.0
            if self.total_transactions > 0:
                fraud_rate = (self.total_fraud_detected / self.total_transactions) * 100
            
            return {
                "total_transactions": self.total_transactions,
                "total_fraud_detected": self.total_fraud_detected,
                "fraud_rate": round(fraud_rate, 4),
                "auto_block_count": self.auto_block_count,
                "manual_review_count": self.manual_review_count,
                "approval_count": self.approval_count,
                "drift_events": self.drift_events,
                "average_inference_time_ms": round(
                    self.total_latency_ms / self.total_transactions, 2
                ) if self.total_transactions > 0 else 0,
                "p95_latency": round(self._calculate_percentile(self.rolling_latencies, 0.95), 2)
            }


# Global metrics tracker instance
_metrics_tracker: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    """
    Get or create the global metrics tracker instance.
    
    Returns:
        MetricsTracker instance
    """
    global _metrics_tracker
    
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()
    
    return _metrics_tracker


def reset_metrics() -> None:
    """Reset all metrics to initial state."""
    tracker = get_metrics_tracker()
    tracker.reset()


if __name__ == "__main__":
    # Test the metrics tracker
    tracker = get_metrics_tracker()
    
    # Record some test predictions
    tracker.record_prediction(
        fraud_probability=0.85,
        anomaly_score=0.72,
        risk_score=0.78,
        risk_band="High",
        action="Auto-Block Transaction",
        latency_ms=45.2,
        drift_detected=False
    )
    
    tracker.record_prediction(
        fraud_probability=0.15,
        anomaly_score=0.10,
        risk_score=0.12,
        risk_band="Low",
        action="Approve",
        latency_ms=32.1,
        drift_detected=False
    )
    
    tracker.record_prediction(
        fraud_probability=0.55,
        anomaly_score=0.45,
        risk_score=0.52,
        risk_band="Medium",
        action="Manual Review Required",
        latency_ms=38.5,
        drift_detected=True
    )
    
    # Get metrics
    metrics = tracker.get_metrics()
    print("Metrics:")
    for key, value in metrics.items():
        if key != "rolling_window_metrics":
            print(f"  {key}: {value}")
    
    print("\nRolling Statistics:")
    rolling_stats = tracker.get_rolling_statistics()
    for key, value in rolling_stats.items():
        print(f"  {key}: {value}")
