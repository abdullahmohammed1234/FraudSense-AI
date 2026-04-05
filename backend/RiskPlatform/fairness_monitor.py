"""
Fairness Monitoring Module for FraudSense AI.

Provides bias detection and fairness metrics across demographic groups:
- Demographic parity
- Equalized odds
- Disparate impact analysis
- Fairness alerts and reporting
"""

import os
import json
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class FairnessMetric(Enum):
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    DISPARATE_IMPACT = "disparate_impact"
    ACCURACY_PARITY = "accuracy_parity"


class ProtectedAttribute(Enum):
    AMOUNT_TIER = "amount_tier"
    TIME_OF_DAY = "time_of_day"
    TRANSACTION_FREQUENCY = "transaction_frequency"


class FairnessMonitor:
    """
    Fairness monitoring for fraud detection model predictions.
    
    Tracks fairness metrics across different transaction groups and
    detects potential bias in model predictions.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._metrics_history: List[Dict[str, Any]] = []
        self._group_statistics: Dict[str, Dict[str, Any]] = {}
        self._thresholds = {
            "disparate_impact_min": 0.8,
            "demographic_parity_max_diff": 0.2,
            "equalized_odds_max_diff": 0.2,
            "accuracy_parity_max_diff": 0.1
        }
        self.max_history = 500
        
    def record_prediction(
        self,
        transaction_id: str,
        features: Dict[str, float],
        prediction: int,
        actual_label: Optional[int] = None,
        fraud_probability: float = 0.0
    ) -> None:
        """
        Record a prediction for fairness monitoring.
        
        Args:
            transaction_id: Transaction identifier
            features: Transaction features
            prediction: Model prediction (0/1)
            actual_label: Actual fraud label (if available)
            fraud_probability: Predicted fraud probability
        """
        with self._lock:
            # Determine group memberships
            groups = self._determine_groups(features)
            
            # Initialize group statistics if needed
            for group_key in groups.keys():
                if group_key not in self._group_statistics:
                    self._group_statistics[group_key] = {
                        "total": 0,
                        "predicted_fraud": 0,
                        "true_fraud": 0,
                        "correct_predictions": 0,
                        "false_positives": 0,
                        "false_negatives": 0,
                        "true_negatives": 0
                    }
            
            # Update group statistics
            for group_key, group_value in groups.items():
                stats = self._group_statistics[group_key]
                stats["total"] += 1
                
                if prediction == 1:
                    stats["predicted_fraud"] += 1
                
                if actual_label is not None:
                    if actual_label == 1 and prediction == 1:
                        stats["true_fraud"] += 1
                        stats["correct_predictions"] += 1
                    elif actual_label == 0 and prediction == 1:
                        stats["false_positives"] += 1
                    elif actual_label == 1 and prediction == 0:
                        stats["false_negatives"] += 1
                    elif actual_label == 0 and prediction == 0:
                        stats["true_negatives"] += 1
                        stats["correct_predictions"] += 1
    
    def _determine_groups(self, features: Dict[str, float]) -> Dict[str, str]:
        """Determine group memberships based on protected attributes."""
        groups = {}
        
        # Amount tier group
        amount = features.get("Amount", 0.0)
        if amount < 50:
            groups["amount_tier"] = "low"
        elif amount < 200:
            groups["amount_tier"] = "medium"
        elif amount < 500:
            groups["amount_tier"] = "high"
        else:
            groups["amount_tier"] = "very_high"
        
        # Time of day group
        time = features.get("Time", 0.0)
        if time < 360:
            groups["time_of_day"] = "night"
        elif time < 720:
            groups["time_of_day"] = "morning"
        elif time < 1080:
            groups["time_of_day"] = "afternoon"
        else:
            groups["time_of_day"] = "evening"
        
        return groups
    
    def calculate_fairness_metrics(
        self,
        protected_attribute: ProtectedAttribute = ProtectedAttribute.AMOUNT_TIER
    ) -> Dict[str, Any]:
        """
        Calculate fairness metrics for a protected attribute.
        
        Args:
            protected_attribute: The attribute to analyze
            
        Returns:
            Fairness metrics and bias analysis
        """
        with self._lock:
            # Get group data
            group_data = self._get_group_data_for_attribute(protected_attribute)
            
            if not group_data:
                return {"error": "Insufficient data for fairness analysis"}
            
            metrics = {
                "protected_attribute": protected_attribute.value,
                "groups": {},
                "overall_metrics": {},
                "bias_detected": False,
                "recommendations": []
            }
            
            # Calculate metrics for each group
            group_rates = {}
            for group_name, stats in group_data.items():
                rates = self._calculate_group_rates(stats)
                metrics["groups"][group_name] = rates
                group_rates[group_name] = rates
            
            # Calculate fairness metrics
            metrics["disparate_impact"] = self._calculate_disparate_impact(group_rates)
            metrics["demographic_parity"] = self._calculate_demographic_parity(group_rates)
            metrics["accuracy_parity"] = self._calculate_accuracy_parity(group_data)
            metrics["equalized_odds"] = self._calculate_equalized_odds(group_data)
            
            # Check for bias
            metrics["bias_detected"] = self._check_for_bias(metrics)
            
            # Generate recommendations
            metrics["recommendations"] = self._generate_recommendations(metrics)
            
            # Store metrics
            self._store_metrics(metrics)
            
            return metrics
    
    def _get_group_data_for_attribute(
        self,
        attribute: ProtectedAttribute
    ) -> Dict[str, Dict[str, Any]]:
        """Get group data for a specific protected attribute."""
        group_data = {}
        
        prefix = attribute.value
        for group_key, stats in self._group_statistics.items():
            if group_key.startswith(prefix):
                group_name = group_key.split("_")[-1]
                group_data[group_name] = stats
        
        return group_data
    
    def _calculate_group_rates(self, stats: Dict[str, int]) -> Dict[str, float]:
        """Calculate rates for a group."""
        total = stats.get("total", 1)
        
        return {
            "positive_rate": round(stats.get("predicted_fraud", 0) / total, 4),
            "true_positive_rate": round(
                stats.get("true_fraud", 0) / max(stats.get("true_fraud", 0) + stats.get("false_negatives", 0), 1),
                4
            ) if stats.get("true_fraud", 0) + stats.get("false_negatives", 0) > 0 else 0.0,
            "false_positive_rate": round(
                stats.get("false_positives", 0) / max(stats.get("false_positives", 0) + stats.get("true_negatives", 0), 1),
                4
            ) if stats.get("false_positives", 0) + stats.get("true_negatives", 0) > 0 else 0.0,
            "accuracy": round(stats.get("correct_predictions", 0) / total, 4),
            "total_transactions": total
        }
    
    def _calculate_disparate_impact(self, group_rates: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate disparate impact metric."""
        if not group_rates:
            return {"value": 1.0, "compliant": True}
        
        positive_rates = [r["positive_rate"] for r in group_rates.values()]
        
        if not positive_rates:
            return {"value": 1.0, "compliant": True}
        
        min_rate = min(positive_rates)
        max_rate = max(positive_rates)
        
        if max_rate == 0:
            impact_ratio = 1.0
        else:
            impact_ratio = min_rate / max_rate
        
        return {
            "value": round(impact_ratio, 4),
            "min_rate": round(min_rate, 4),
            "max_rate": round(max_rate, 4),
            "compliant": impact_ratio >= self._thresholds["disparate_impact_min"],
            "threshold": self._thresholds["disparate_impact_min"]
        }
    
    def _calculate_demographic_parity(self, group_rates: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate demographic parity (difference in positive rates)."""
        if not group_rates:
            return {"value": 0.0, "compliant": True}
        
        positive_rates = [r["positive_rate"] for r in group_rates.values()]
        
        if len(positive_rates) < 2:
            return {"value": 0.0, "compliant": True}
        
        max_diff = max(positive_rates) - min(positive_rates)
        
        return {
            "value": round(max_diff, 4),
            "compliant": max_diff <= self._thresholds["demographic_parity_max_diff"],
            "threshold": self._thresholds["demographic_parity_max_diff"]
        }
    
    def _calculate_accuracy_parity(self, group_data: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Calculate accuracy parity across groups."""
        if not group_data:
            return {"value": 0.0, "compliant": True}
        
        accuracies = []
        for stats in group_data.values():
            total = stats.get("total", 1)
            if total > 0:
                accuracies.append(stats.get("correct_predictions", 0) / total)
        
        if len(accuracies) < 2:
            return {"value": 0.0, "compliant": True}
        
        max_diff = max(accuracies) - min(accuracies)
        
        return {
            "value": round(max_diff, 4),
            "compliant": max_diff <= self._thresholds["accuracy_parity_max_diff"],
            "threshold": self._thresholds["accuracy_parity_max_diff"]
        }
    
    def _calculate_equalized_odds(self, group_data: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Calculate equalized odds (TPR and FPR parity)."""
        if not group_data:
            return {"value": 0.0, "compliant": True}
        
        tprs = []
        fprs = []
        
        for stats in group_data.values():
            true_fraud = stats.get("true_fraud", 0)
            false_neg = stats.get("false_negatives", 0)
            false_pos = stats.get("false_positives", 0)
            true_neg = stats.get("true_negatives", 0)
            
            tpr = true_fraud / (true_fraud + false_neg) if (true_fraud + false_neg) > 0 else 0
            fpr = false_pos / (false_pos + true_neg) if (false_pos + true_neg) > 0 else 0
            
            tprs.append(tpr)
            fprs.append(fpr)
        
        tpr_diff = max(tprs) - min(tprs) if tprs else 0
        fpr_diff = max(fprs) - min(fprs) if fprs else 0
        max_diff = max(tpr_diff, fpr_diff)
        
        return {
            "value": round(max_diff, 4),
            "tpr_difference": round(tpr_diff, 4),
            "fpr_difference": round(fpr_diff, 4),
            "compliant": max_diff <= self._thresholds["equalized_odds_max_diff"],
            "threshold": self._thresholds["equalized_odds_max_diff"]
        }
    
    def _check_for_bias(self, metrics: Dict[str, Any]) -> bool:
        """Check if any fairness metric indicates bias."""
        if metrics.get("disparate_impact", {}).get("compliant", True) is False:
            return True
        if metrics.get("demographic_parity", {}).get("compliant", True) is False:
            return True
        if metrics.get("equalized_odds", {}).get("compliant", True) is False:
            return True
        if metrics.get("accuracy_parity", {}).get("compliant", True) is False:
            return True
        
        return False
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on fairness analysis."""
        recommendations = []
        
        if not metrics.get("disparate_impact", {}).get("compliant", True):
            recommendations.append(
                f"Disparate impact detected: {metrics['disparate_impact']['value']:.2f} "
                f"(threshold: {metrics['disparate_impact']['threshold']:.2f}). "
                "Consider rebalancing training data or adjusting decision thresholds."
            )
        
        if not metrics.get("demographic_parity", {}).get("compliant", True):
            recommendations.append(
                f"Demographic parity violation: {metrics['demographic_parity']['value']:.2f} "
                f"(threshold: {metrics['demographic_parity']['threshold']:.2f}). "
                "Review model for systematic biases."
            )
        
        if not metrics.get("equalized_odds", {}).get("compliant", True):
            eo = metrics.get("equalized_odds", {})
            recommendations.append(
                f"Equalized odds violation. TPR diff: {eo.get('tpr_difference', 0):.2f}, "
                f"FPR diff: {eo.get('fpr_difference', 0):.2f}. "
                "Consider fairness-aware training."
            )
        
        if not recommendations:
            recommendations.append("All fairness metrics within acceptable thresholds.")
        
        return recommendations
    
    def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store metrics in history."""
        if len(self._metrics_history) >= self.max_history:
            self._metrics_history.pop(0)
        
        metrics["timestamp"] = datetime.utcnow().isoformat() + "Z"
        self._metrics_history.append(metrics)
    
    def get_fairness_alerts(self) -> List[Dict[str, Any]]:
        """Get recent fairness alerts."""
        with self._lock:
            alerts = []
            for metrics in reversed(self._metrics_history[-10:]):
                if metrics.get("bias_detected", False):
                    alerts.append({
                        "timestamp": metrics.get("timestamp"),
                        "protected_attribute": metrics.get("protected_attribute"),
                        "recommendations": metrics.get("recommendations", [])
                    })
            return alerts
    
    def get_fairness_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get fairness trends over time."""
        with self._lock:
            if not self._metrics_history:
                return {"error": "No historical data available"}
            
            recent_metrics = self._metrics_history[-days:] if len(self._metrics_history) > days else self._metrics_history
            
            return {
                "period_days": len(recent_metrics),
                "bias_incidents": sum(1 for m in recent_metrics if m.get("bias_detected")),
                "metrics_trend": {
                    "disparate_impact": [m.get("disparate_impact", {}).get("value", 1.0) for m in recent_metrics],
                    "demographic_parity": [m.get("demographic_parity", {}).get("value", 0.0) for m in recent_metrics]
                }
            }
    
    def get_group_statistics(self) -> Dict[str, Any]:
        """Get current group statistics."""
        with self._lock:
            return self._group_statistics.copy()
    
    def set_thresholds(
        self,
        disparate_impact_min: Optional[float] = None,
        demographic_parity_max: Optional[float] = None,
        equalized_odds_max: Optional[float] = None,
        accuracy_parity_max: Optional[float] = None
    ) -> None:
        """Update fairness thresholds."""
        with self._lock:
            if disparate_impact_min is not None:
                self._thresholds["disparate_impact_min"] = disparate_impact_min
            if demographic_parity_max is not None:
                self._thresholds["demographic_parity_max_diff"] = demographic_parity_max
            if equalized_odds_max is not None:
                self._thresholds["equalized_odds_max_diff"] = equalized_odds_max
            if accuracy_parity_max is not None:
                self._thresholds["accuracy_parity_max_diff"] = accuracy_parity_max
    
    def reset_statistics(self) -> None:
        """Reset all fairness statistics."""
        with self._lock:
            self._group_statistics.clear()
            self._metrics_history.clear()


# Global instance
_fairness_monitor: Optional[FairnessMonitor] = None


def get_fairness_monitor() -> FairnessMonitor:
    """Get or create global fairness monitor."""
    global _fairness_monitor
    if _fairness_monitor is None:
        _fairness_monitor = FairnessMonitor()
    return _fairness_monitor


if __name__ == "__main__":
    monitor = get_fairness_monitor()
    
    # Simulate transactions for different groups
    test_transactions = [
        # Low amount transactions (should have lower fraud rate)
        {"Amount": 25.0, "Time": 100.0, "prediction": 0, "actual": 0},
        {"Amount": 45.0, "Time": 200.0, "prediction": 0, "actual": 0},
        {"Amount": 30.0, "Time": 300.0, "prediction": 1, "actual": 0},  # FP
        {"Amount": 49.0, "Time": 400.0, "prediction": 0, "actual": 0},
        
        # High amount transactions (higher fraud rate)
        {"Amount": 500.0, "Time": 1500.0, "prediction": 1, "actual": 1},
        {"Amount": 750.0, "Time": 1600.0, "prediction": 1, "actual": 1},
        {"Amount": 600.0, "Time": 1700.0, "prediction": 1, "actual": 0},  # FP
        {"Amount": 800.0, "Time": 1800.0, "prediction": 1, "actual": 1},
        
        # Very high amount
        {"Amount": 2000.0, "Time": 100.0, "prediction": 1, "actual": 1},
        {"Amount": 5000.0, "Time": 200.0, "prediction": 1, "actual": 1},
    ]
    
    for i, txn in enumerate(test_transactions):
        monitor.record_prediction(
            transaction_id=f"TXN-{i}",
            features=txn,
            prediction=txn["prediction"],
            actual_label=txn.get("actual"),
            fraud_probability=0.7 if txn["prediction"] == 1 else 0.2
        )
    
    # Calculate fairness metrics
    metrics = monitor.calculate_fairness_metrics(ProtectedAttribute.AMOUNT_TIER)
    
    print("Fairness Metrics:")
    print(f"  Protected Attribute: {metrics['protected_attribute']}")
    print(f"  Bias Detected: {metrics['bias_detected']}")
    print(f"  Disparate Impact: {metrics['disparate_impact']}")
    print(f"  Demographic Parity: {metrics['demographic_parity']}")
    print(f"  Equalized Odds: {metrics['equalized_odds']}")
    print(f"\nRecommendations:")
    for rec in metrics["recommendations"]:
        print(f"  - {rec}")