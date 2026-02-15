"""
AnalyticsTracker Module for FraudSense AI.

Tracks real-time analytics and statistics for the fraud detection system.
"""

from typing import Dict, Any, List, Optional
from collections import Counter, deque
from datetime import datetime
import threading

from .config import AnalyticsConfig


class AnalyticsTracker:
    """
    Real-time analytics tracker for fraud detection.
    
    Tracks transaction statistics, risk distributions, and
    provides aggregate analytics.
    """
    
    def __init__(self, max_rolling: int = AnalyticsConfig.MAX_ROLLING):
        """
        Initialize the AnalyticsTracker.
        
        Args:
            max_rolling: Maximum size of rolling windows.
        """
        self.max_rolling = max_rolling
        self._lock = threading.Lock()
        
        # Transaction statistics
        self.total_transactions = 0
        self.total_fraud_detected = 0
        self.total_blocked = 0
        self.total_approved = 0
        self.total_review_required = 0
        
        # Sum of probabilities and scores
        self.sum_fraud_probability = 0.0
        self.sum_anomaly_score = 0.0
        self.sum_risk_score = 0.0
        
        # Rolling windows
        self.rolling_fraud_probabilities: deque = deque(maxlen=max_rolling)
        self.rolling_anomaly_scores: deque = deque(maxlen=max_rolling)
        self.rolling_risk_scores: deque = deque(maxlen=max_rolling)
        
        # Feature tracking
        self.feature_counter: Counter = Counter()
        
        # Risk level distribution
        self.risk_level_counts: Counter = Counter()
        
        # Action distribution
        self.action_counts: Counter = Counter()
        
        # Latency tracking
        self.latency_sum = 0.0
        self.latency_count = 0
        self.latency_max = 0.0
        self.rolling_latencies: deque = deque(maxlen=max_rolling)
    
    def record_transaction(
        self,
        fraud_probability: float,
        anomaly_score: float,
        risk_score: float,
        risk_level: str,
        action: str,
        top_factors: List[Dict[str, Any]],
        latency_ms: float = 0.0
    ) -> None:
        """
        Record a transaction for analytics.
        
        Args:
            fraud_probability: Fraud probability.
            anomaly_score: Anomaly score.
            risk_score: Final risk score.
            risk_level: Risk level.
            action: Action taken.
            top_factors: Top risk factors.
            latency_ms: Inference latency in milliseconds.
        """
        with self._lock:
            self.total_transactions += 1
            self.sum_fraud_probability += fraud_probability
            self.sum_anomaly_score += anomaly_score
            self.sum_risk_score += risk_score
            
            # Update rolling windows
            self.rolling_fraud_probabilities.append(fraud_probability)
            self.rolling_anomaly_scores.append(anomaly_score)
            self.rolling_risk_scores.append(risk_score)
            
            # Count risk levels
            self.risk_level_counts[risk_level] += 1
            
            # Count actions
            self.action_counts[action] += 1
            
            # Track fraud detection
            if fraud_probability >= 0.5:
                self.total_fraud_detected += 1
            
            # Track by action
            if "Block" in action:
                self.total_blocked += 1
            elif "Review" in action:
                self.total_review_required += 1
            elif "Approve" in action:
                self.total_approved += 1
            
            # Track top features
            if top_factors:
                top_feature = top_factors[0].get("feature", "Unknown") if top_factors else "Unknown"
                self.feature_counter[top_feature] += 1
            
            # Track latency
            if latency_ms > 0:
                self.latency_sum += latency_ms
                self.latency_count += 1
                self.latency_max = max(self.latency_max, latency_ms)
                self.rolling_latencies.append(latency_ms)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get analytics summary.
        
        Returns:
            Dictionary with analytics data.
        """
        with self._lock:
            if self.total_transactions == 0:
                return self._empty_summary()
            
            # Calculate averages
            avg_fraud_prob = self.sum_fraud_probability / self.total_transactions
            avg_anomaly = self.sum_anomaly_score / self.total_transactions
            avg_risk_score = self.sum_risk_score / self.total_transactions
            
            # Calculate fraud rate
            fraud_rate = (self.total_fraud_detected / self.total_transactions) * 100
            
            # Get average latency
            avg_latency = self.latency_sum / self.latency_count if self.latency_count > 0 else 0.0
            
            # Get top feature
            top_feature = "N/A"
            if self.feature_counter:
                top_feature = self.feature_counter.most_common(1)[0][0]
            
            # Get risk distribution
            risk_distribution = {
                level: self.risk_level_counts.get(level, 0)
                for level in ["Low", "Medium", "High", "Critical"]
            }
            
            # Get action distribution
            action_distribution = dict(self.action_counts)
            
            return {
                "total_transactions": self.total_transactions,
                "fraud_detected": self.total_fraud_detected,
                "fraud_rate_percentage": round(fraud_rate, 4),
                "actions": {
                    "blocked": self.total_blocked,
                    "review_required": self.total_review_required,
                    "approved": self.total_approved
                },
                "averages": {
                    "fraud_probability": round(avg_fraud_prob, 4),
                    "anomaly_score": round(avg_anomaly, 4),
                    "risk_score": round(avg_risk_score, 4)
                },
                "risk_distribution": risk_distribution,
                "action_distribution": action_distribution,
                "top_risk_feature": top_feature,
                "latency": {
                    "average_ms": round(avg_latency, 2),
                    "max_ms": round(self.latency_max, 2),
                    "total_measurements": self.latency_count
                },
                "rolling_window": {
                    "fraud_probabilities": list(self.rolling_fraud_probabilities),
                    "size": len(self.rolling_fraud_probabilities)
                }
            }
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Get empty summary when no data."""
        return {
            "total_transactions": 0,
            "fraud_detected": 0,
            "fraud_rate_percentage": 0.0,
            "actions": {
                "blocked": 0,
                "review_required": 0,
                "approved": 0
            },
            "averages": {
                "fraud_probability": 0.0,
                "anomaly_score": 0.0,
                "risk_score": 0.0
            },
            "risk_distribution": {
                "Low": 0,
                "Medium": 0,
                "High": 0,
                "Critical": 0
            },
            "action_distribution": {},
            "top_risk_feature": "N/A",
            "latency": {
                "average_ms": 0.0,
                "max_ms": 0.0,
                "total_measurements": 0
            },
            "rolling_window": {
                "fraud_probabilities": [],
                "size": 0
            }
        }
    
    def get_recent_analytics(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get recent analytics based on rolling window.
        
        Args:
            limit: Number of recent transactions to consider.
            
        Returns:
            Dictionary with recent analytics.
        """
        with self._lock:
            if not self.rolling_fraud_probabilities:
                return {
                    "recent_count": 0,
                    "average_fraud_probability": 0.0,
                    "average_anomaly_score": 0.0,
                    "average_risk_score": 0.0
                }
            
            # Get recent values
            recent_probs = list(self.rolling_fraud_probabilities)[-limit:]
            recent_anomalies = list(self.rolling_anomaly_scores)[-limit:]
            recent_risks = list(self.rolling_risk_scores)[-limit:]
            
            return {
                "recent_count": len(recent_probs),
                "average_fraud_probability": round(sum(recent_probs) / len(recent_probs), 4),
                "average_anomaly_score": round(sum(recent_anomalies) / len(recent_anomalies), 4),
                "average_risk_score": round(sum(recent_risks) / len(recent_risks), 4)
            }
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """
        Get latency statistics.
        
        Returns:
            Dictionary with latency stats.
        """
        with self._lock:
            if not self.rolling_latencies:
                return {
                    "average_ms": 0.0,
                    "max_ms": 0.0,
                    "min_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "total_measurements": 0
                }
            
            latencies = sorted(self.rolling_latencies)
            n = len(latencies)
            
            return {
                "average_ms": round(self.latency_sum / self.latency_count, 2) if self.latency_count > 0 else 0.0,
                "max_ms": round(self.latency_max, 2),
                "min_ms": round(min(latencies), 2),
                "p95_ms": round(latencies[int(n * 0.95)] if n > 0 else 0.0, 2),
                "p99_ms": round(latencies[int(n * 0.99)] if n > 0 else 0.0, 2),
                "total_measurements": self.latency_count
            }
    
    def reset(self) -> None:
        """Reset all analytics data."""
        with self._lock:
            self.total_transactions = 0
            self.total_fraud_detected = 0
            self.total_blocked = 0
            self.total_approved = 0
            self.total_review_required = 0
            self.sum_fraud_probability = 0.0
            self.sum_anomaly_score = 0.0
            self.sum_risk_score = 0.0
            self.rolling_fraud_probabilities.clear()
            self.rolling_anomaly_scores.clear()
            self.rolling_risk_scores.clear()
            self.feature_counter.clear()
            self.risk_level_counts.clear()
            self.action_counts.clear()
            self.latency_sum = 0.0
            self.latency_count = 0
            self.latency_max = 0.0
            self.rolling_latencies.clear()


# Global analytics tracker instance
_analytics_tracker: Optional[AnalyticsTracker] = None


def get_analytics_tracker() -> AnalyticsTracker:
    """
    Get or create the global analytics tracker instance.
    
    Returns:
        AnalyticsTracker instance.
    """
    global _analytics_tracker
    
    if _analytics_tracker is None:
        _analytics_tracker = AnalyticsTracker()
    
    return _analytics_tracker


if __name__ == "__main__":
    # Test the analytics tracker
    tracker = get_analytics_tracker()
    
    # Record some test transactions
    tracker.record_transaction(
        fraud_probability=0.85,
        anomaly_score=0.72,
        risk_score=0.78,
        risk_level="High",
        action="Auto-Block Transaction",
        top_factors=[{"feature": "V14", "impact": 0.45}],
        latency_ms=45.2
    )
    
    tracker.record_transaction(
        fraud_probability=0.15,
        anomaly_score=0.10,
        risk_score=0.12,
        risk_level="Low",
        action="Approve",
        top_factors=[],
        latency_ms=32.1
    )
    
    # Get summary
    summary = tracker.get_summary()
    print(f"Analytics Summary: {summary}")
