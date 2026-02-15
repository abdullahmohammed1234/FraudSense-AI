"""
ThresholdSimulator Module for FraudSense AI.

Provides threshold simulation without model retraining:
- Simulate decision outcomes using historical predictions
- Calculate projected fraud detection rate
- Calculate projected false positive rate
- Calculate projected decision distribution
- Calculate projected auto-block rate
"""

import threading
from collections import deque, Counter
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class ThresholdSimulator:
    """
    Threshold simulation engine.
    
    Simulates different classification thresholds using stored predictions
    to estimate decision outcomes without retraining the model.
    """
    
    def __init__(self, max_predictions: int = 500):
        """
        Initialize the ThresholdSimulator.
        
        Args:
            max_predictions: Maximum number of predictions to store
        """
        self.max_predictions = max_predictions
        self._lock = threading.RLock()
        
        # Store prediction history
        self.prediction_history: deque = deque(maxlen=max_predictions)
        
        # Default thresholds for simulation
        self.default_block_threshold = 0.75
        self.default_review_threshold = 0.40
    
    def record_prediction(
        self,
        fraud_probability: float,
        anomaly_score: float,
        risk_score: float,
        actual_label: Optional[bool] = None,
        transaction_id: Optional[str] = None
    ) -> None:
        """
        Record a prediction for future simulation.
        
        Args:
            fraud_probability: Predicted fraud probability
            anomaly_score: Anomaly score
            risk_score: Ensemble risk score
            actual_label: Actual fraud label (if known)
            transaction_id: Transaction ID
        """
        with self._lock:
            prediction = {
                "fraud_probability": fraud_probability,
                "anomaly_score": anomaly_score,
                "risk_score": risk_score,
                "actual_label": actual_label,
                "transaction_id": transaction_id
            }
            self.prediction_history.append(prediction)
    
    def simulate_threshold(
        self,
        threshold: float,
        review_threshold: Optional[float] = None,
        block_anomaly_threshold: float = 0.70
    ) -> Dict[str, Any]:
        """
        Simulate decision outcomes with a given threshold.
        
        Args:
            threshold: Classification threshold for fraud detection
            review_threshold: Threshold for manual review (default: 0.4)
            block_anomaly_threshold: Anomaly threshold for auto-block
            
        Returns:
            Simulation results dictionary
        """
        with self._lock:
            if not self.prediction_history:
                return {
                    "error": "No prediction history available",
                    "sample_size": 0
                }
            
            if review_threshold is None:
                review_threshold = self.default_review_threshold
            
            predictions = list(self.prediction_history)
            total = len(predictions)
            
            # Track decision outcomes
            decisions = {
                "auto_block": 0,
                "manual_review": 0,
                "approve": 0
            }
            
            # For fraud detection rate calculation
            detected_fraud = 0
            total_actual_fraud = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            true_positives = 0
            
            for pred in predictions:
                fraud_prob = pred["fraud_probability"]
                anomaly = pred["anomaly_score"]
                actual = pred["actual_label"]
                
                # Determine decision based on threshold
                if fraud_prob >= threshold and anomaly >= block_anomaly_threshold:
                    decision = "auto_block"
                elif fraud_prob >= threshold:
                    decision = "manual_review"
                elif fraud_prob >= review_threshold or anomaly >= (block_anomaly_threshold * 0.6):
                    decision = "manual_review"
                else:
                    decision = "approve"
                
                decisions[decision] += 1
                
                # Calculate metrics if actual label is known
                if actual is not None:
                    if actual:  # Actually fraud
                        total_actual_fraud += 1
                        if fraud_prob >= threshold:
                            detected_fraud += 1
                            true_positives += 1
                        else:
                            false_negatives += 1
                    else:  # Not fraud
                        if fraud_prob >= threshold:
                            false_positives += 1
                        else:
                            true_negatives += 1
            
            # Calculate rates
            fraud_detection_rate = (detected_fraud / total_actual_fraud * 100) if total_actual_fraud > 0 else 0
            false_positive_rate = (false_positives / (total - total_actual_fraud) * 100) if (total - total_actual_fraud) > 0 else 0
            
            # Calculate decision distribution
            decision_distribution = {
                "auto_block": {
                    "count": decisions["auto_block"],
                    "percentage": round((decisions["auto_block"] / total) * 100, 2)
                },
                "manual_review": {
                    "count": decisions["manual_review"],
                    "percentage": round((decisions["manual_review"] / total) * 100, 2)
                },
                "approve": {
                    "count": decisions["approve"],
                    "percentage": round((decisions["approve"] / total) * 100, 2)
                }
            }
            
            # Auto-block rate
            auto_block_rate = (decisions["auto_block"] / total) * 100
            
            # Calculate confidence interval for detection rate
            if total_actual_fraud > 0:
                detection_variance = (fraud_detection_rate * (100 - fraud_detection_rate)) / total_actual_fraud
                confidence_margin = 1.96 * np.sqrt(detection_variance)
            else:
                confidence_margin = 0
            
            return {
                "threshold": threshold,
                "review_threshold": review_threshold,
                "block_anomaly_threshold": block_anomaly_threshold,
                "sample_size": total,
                "projected_fraud_detection_rate": round(fraud_detection_rate, 2),
                "projected_false_positive_rate": round(false_positive_rate, 2),
                "projected_decision_distribution": decision_distribution,
                "projected_auto_block_rate": round(auto_block_rate, 2),
                "confidence_margin_95": round(confidence_margin, 2),
                "metrics": {
                    "true_positives": true_positives,
                    "true_negatives": true_negatives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "total_with_actual_label": true_positives + true_negatives + false_positives + false_negatives
                }
            }
    
    def find_optimal_threshold(
        self,
        target_fpr: float = 1.0,
        min_detection_rate: float = 80.0
    ) -> Dict[str, Any]:
        """
        Find optimal threshold that meets target FPR while maintaining detection rate.
        
        Args:
            target_fpr: Target false positive rate (%)
            min_detection_rate: Minimum acceptable detection rate (%)
            
        Returns:
            Optimal threshold information
        """
        with self._lock:
            if not self.prediction_history:
                return {"error": "No prediction history available"}
            
            predictions = list(self.prediction_history)
            
            # Get unique fraud probabilities
            fraud_probs = sorted(set(p["fraud_probability"] for p in predictions))
            
            best_threshold = 0.5
            best_fpr = 100.0
            best_detection_rate = 0.0
            
            for threshold in fraud_probs:
                result = self.simulate_threshold(threshold)
                
                if result.get("error"):
                    continue
                
                fpr = result["projected_false_positive_rate"]
                detection_rate = result["projected_fraud_detection_rate"]
                
                if fpr <= target_fpr and detection_rate >= min_detection_rate:
                    # This threshold meets our criteria
                    if detection_rate > best_detection_rate:
                        best_threshold = threshold
                        best_fpr = fpr
                        best_detection_rate = detection_rate
            
            return {
                "optimal_threshold": round(best_threshold, 4),
                "projected_fpr": round(best_fpr, 2),
                "projected_detection_rate": round(best_detection_rate, 2),
                "target_fpr": target_fpr,
                "min_detection_rate": min_detection_rate
            }
    
    def compare_thresholds(
        self,
        thresholds: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple thresholds.
        
        Args:
            thresholds: List of thresholds to compare
            
        Returns:
            List of simulation results for each threshold
        """
        results = []
        
        for threshold in thresholds:
            result = self.simulate_threshold(threshold)
            results.append(result)
        
        return results
    
    def get_threshold_analysis(
        self,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        steps: int = 9
    ) -> Dict[str, Any]:
        """
        Get threshold analysis across a range.
        
        Args:
            min_threshold: Minimum threshold
            max_threshold: Maximum threshold
            steps: Number of steps
            
        Returns:
            Threshold analysis
        """
        thresholds = np.linspace(min_threshold, max_threshold, steps)
        results = self.compare_thresholds(thresholds.tolist())
        
        # Find trade-off points
        detection_rates = [r.get("projected_fraud_detection_rate", 0) for r in results]
        fprs = [r.get("projected_false_positive_rate", 0) for r in results]
        
        return {
            "thresholds_tested": [round(t, 2) for t in thresholds],
            "detection_rates": detection_rates,
            "false_positive_rates": fprs,
            "results": results
        }
    
    def get_prediction_history(self) -> List[Dict[str, Any]]:
        """Get stored prediction history."""
        with self._lock:
            return list(self.prediction_history)
    
    def clear_history(self) -> int:
        """Clear prediction history."""
        with self._lock:
            count = len(self.prediction_history)
            self.prediction_history.clear()
            return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored predictions."""
        with self._lock:
            if not self.prediction_history:
                return {
                    "total_predictions": 0,
                    "has_actual_labels": False
                }
            
            predictions = list(self.prediction_history)
            fraud_probs = [p["fraud_probability"] for p in predictions]
            has_labels = any(p["actual_label"] is not None for p in predictions)
            
            return {
                "total_predictions": len(predictions),
                "has_actual_labels": has_labels,
                "average_fraud_probability": round(np.mean(fraud_probs), 4),
                "min_fraud_probability": round(min(fraud_probs), 4),
                "max_fraud_probability": round(max(fraud_probs), 4),
                "std_fraud_probability": round(np.std(fraud_probs), 4)
            }


# Global threshold simulator instance
_threshold_simulator: Optional[ThresholdSimulator] = None


def get_threshold_simulator() -> ThresholdSimulator:
    """
    Get or create the global threshold simulator.
    
    Returns:
        ThresholdSimulator instance
    """
    global _threshold_simulator
    
    if _threshold_simulator is None:
        _threshold_simulator = ThresholdSimulator()
    
    return _threshold_simulator


if __name__ == "__main__":
    # Test the threshold simulator
    simulator = get_threshold_simulator()
    
    # Record some test predictions
    import random
    random.seed(42)
    
    for i in range(100):
        fraud_prob = random.random()
        anomaly_score = random.random() * 0.8
        risk_score = (fraud_prob + anomaly_score) / 2
        
        # Add some with known labels
        actual = random.random() < 0.02  # ~2% fraud
        
        simulator.record_prediction(
            fraud_probability=fraud_prob,
            anomaly_score=anomaly_score,
            risk_score=risk_score,
            actual_label=actual
        )
    
    # Simulate different thresholds
    print("Threshold Simulation:")
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        result = simulator.simulate_threshold(threshold)
        print(f"\nThreshold {threshold}:")
        print(f"  Detection Rate: {result.get('projected_fraud_detection_rate')}%")
        print(f"  False Positive Rate: {result.get('projected_false_positive_rate')}%")
        print(f"  Auto-block Rate: {result.get('projected_auto_block_rate')}%")
    
    # Find optimal threshold
    print("\nOptimal Threshold:")
    optimal = simulator.find_optimal_threshold(target_fpr=1.0, min_detection_rate=80.0)
    print(f"  Threshold: {optimal.get('optimal_threshold')}")
    print(f"  Detection Rate: {optimal.get('projected_detection_rate')}%")
    print(f"  FPR: {optimal.get('projected_fpr')}%")
