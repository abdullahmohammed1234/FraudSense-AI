"""
RiskScorer Module for FraudSense AI.

Provides ensemble risk scoring combining fraud probability,
anomaly score, and drift signal with configurable weights.
"""

from typing import Dict, Any, Optional, Tuple

from .config import RiskScoringConfig, DecisionConfig


class RiskScorer:
    """
    Ensemble Risk Scorer.
    
    Combines multiple risk signals with configurable weights
    to produce a final risk score and risk band.
    """
    
    def __init__(
        self,
        fraud_weight: float = RiskScoringConfig.FRAUD_PROBABILITY_WEIGHT,
        anomaly_weight: float = RiskScoringConfig.ANOMALY_SCORE_WEIGHT,
        drift_weight: float = RiskScoringConfig.DRIFT_SIGNAL_WEIGHT,
        drift_penalty: float = RiskScoringConfig.DRIFT_PENALTY
    ):
        """
        Initialize the RiskScorer.
        
        Args:
            fraud_weight: Weight for fraud probability.
            anomaly_weight: Weight for anomaly score.
            drift_weight: Weight for drift signal.
            drift_penalty: Multiplier for drift penalty.
        """
        self.fraud_weight = fraud_weight
        self.anomaly_weight = anomaly_weight
        self.drift_weight = drift_weight
        self.drift_penalty = drift_penalty
        
        # Validate weights sum to 1.0
        total = fraud_weight + anomaly_weight + drift_weight
        if abs(total - 1.0) > 0.001:
            # Normalize weights
            self.fraud_weight /= total
            self.anomaly_weight /= total
            self.drift_weight /= total
    
    def calculate_drift_signal(self, drift_detected: bool) -> float:
        """
        Calculate drift signal value.
        
        Args:
            drift_detected: Whether drift was detected.
            
        Returns:
            Drift signal value (0-1).
        """
        return 1.0 if drift_detected else 0.0
    
    def calculate_risk_score(
        self,
        fraud_probability: float,
        anomaly_score: float,
        drift_detected: bool = False
    ) -> Tuple[float, str]:
        """
        Calculate ensemble risk score.
        
        final_risk_score = weighted combination of:
        - fraud_probability (weight: 0.50)
        - anomaly_score (weight: 0.35)
        - drift_signal (weight: 0.15)
        
        Args:
            fraud_probability: Fraud probability (0-1).
            anomaly_score: Anomaly score (0-1).
            drift_detected: Whether drift was detected.
            
        Returns:
            Tuple of (final_risk_score, risk_band).
        """
        # Calculate drift signal
        drift_signal = self.calculate_drift_signal(drift_detected)
        
        # Apply drift penalty if drift detected
        drift_component = drift_signal * self.drift_weight
        if drift_detected:
            drift_component *= self.drift_penalty
        
        # Calculate weighted components
        fraud_component = fraud_probability * self.fraud_weight
        anomaly_component = anomaly_score * self.anomaly_weight
        
        # Calculate final score (normalized)
        raw_score = fraud_component + anomaly_component + drift_component
        
        # Normalize to 0-1 (considering drift penalty might push above 1)
        final_score = min(raw_score, 1.0)
        
        # Determine risk band
        risk_band = self._get_risk_band(final_score)
        
        return final_score, risk_band
    
    def _get_risk_band(self, score: float) -> str:
        """
        Determine risk band from score.
        
        Args:
            score: Final risk score (0-1).
            
        Returns:
            Risk band string (Low/Medium/High/Critical).
        """
        if score < DecisionConfig.LOW_RISK_THRESHOLD:
            return "Low"
        elif score < DecisionConfig.MEDIUM_RISK_THRESHOLD:
            return "Medium"
        elif score < DecisionConfig.HIGH_RISK_THRESHOLD:
            return "High"
        else:
            return "Critical"
    
    def get_score_breakdown(
        self,
        fraud_probability: float,
        anomaly_score: float,
        drift_detected: bool
    ) -> Dict[str, Any]:
        """
        Get detailed breakdown of risk score components.
        
        Args:
            fraud_probability: Fraud probability.
            anomaly_score: Anomaly score.
            drift_detected: Whether drift was detected.
            
        Returns:
            Dictionary with score breakdown.
        """
        drift_signal = self.calculate_drift_signal(drift_detected)
        
        fraud_component = fraud_probability * self.fraud_weight
        anomaly_component = anomaly_score * self.anomaly_weight
        
        drift_component = drift_signal * self.drift_weight
        if drift_detected:
            drift_component *= self.drift_penalty
        
        final_score, risk_band = self.calculate_risk_score(
            fraud_probability, anomaly_score, drift_detected
        )
        
        return {
            "final_risk_score": round(final_score, 4),
            "risk_band": risk_band,
            "components": {
                "fraud_probability": {
                    "value": fraud_probability,
                    "weight": self.fraud_weight,
                    "contribution": round(fraud_component, 4)
                },
                "anomaly_score": {
                    "value": anomaly_score,
                    "weight": self.anomaly_weight,
                    "contribution": round(anomaly_component, 4)
                },
                "drift_signal": {
                    "value": drift_signal,
                    "weight": self.drift_weight,
                    "penalty_applied": drift_detected,
                    "contribution": round(drift_component, 4)
                }
            },
            "weights": {
                "fraud_probability": self.fraud_weight,
                "anomaly_score": self.anomaly_weight,
                "drift_signal": self.drift_weight
            }
        }
    
    def set_weights(
        self,
        fraud_weight: Optional[float] = None,
        anomaly_weight: Optional[float] = None,
        drift_weight: Optional[float] = None
    ) -> None:
        """
        Update weights.
        
        Args:
            fraud_weight: New fraud probability weight.
            anomaly_weight: New anomaly score weight.
            drift_weight: New drift signal weight.
        """
        if fraud_weight is not None:
            self.fraud_weight = fraud_weight
        if anomaly_weight is not None:
            self.anomaly_weight = anomaly_weight
        if drift_weight is not None:
            self.drift_weight = drift_weight
        
        # Normalize weights
        total = self.fraud_weight + self.anomaly_weight + self.drift_weight
        if total > 0:
            self.fraud_weight /= total
            self.anomaly_weight /= total
            self.drift_weight /= total
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Dictionary with current settings.
        """
        return {
            "weights": {
                "fraud_probability": self.fraud_weight,
                "anomaly_score": self.anomaly_weight,
                "drift_signal": self.drift_weight
            },
            "drift_penalty": self.drift_penalty,
            "risk_bands": {
                "Low": f"< {DecisionConfig.LOW_RISK_THRESHOLD}",
                "Medium": f"{DecisionConfig.LOW_RISK_THRESHOLD} - {DecisionConfig.MEDIUM_RISK_THRESHOLD}",
                "High": f"{DecisionConfig.MEDIUM_RISK_THRESHOLD} - {DecisionConfig.HIGH_RISK_THRESHOLD}",
                "Critical": f"> {DecisionConfig.HIGH_RISK_THRESHOLD}"
            }
        }


# Global risk scorer instance
_risk_scorer: Optional[RiskScorer] = None


def get_risk_scorer() -> RiskScorer:
    """
    Get or create the global risk scorer instance.
    
    Returns:
        RiskScorer instance.
    """
    global _risk_scorer
    
    if _risk_scorer is None:
        _risk_scorer = RiskScorer()
    
    return _risk_scorer


def calculate_risk_score(
    fraud_probability: float,
    anomaly_score: float,
    drift_detected: bool = False
) -> Tuple[float, str]:
    """
    Calculate ensemble risk score (convenience function).
    
    Args:
        fraud_probability: Fraud probability.
        anomaly_score: Anomaly score.
        drift_detected: Whether drift was detected.
        
    Returns:
        Tuple of (final_risk_score, risk_band).
    """
    scorer = get_risk_scorer()
    return scorer.calculate_risk_score(fraud_probability, anomaly_score, drift_detected)


if __name__ == "__main__":
    # Test the risk scorer
    scorer = get_risk_scorer()
    
    # Test cases
    test_cases = [
        (0.85, 0.72, False),
        (0.30, 0.15, False),
        (0.60, 0.45, True),
        (0.10, 0.05, False),
    ]
    
    for fraud_prob, anomaly, drift in test_cases:
        score, band = scorer.calculate_risk_score(fraud_prob, anomaly, drift)
        breakdown = scorer.get_score_breakdown(fraud_prob, anomaly, drift)
        
        print(f"\nFraud: {fraud_prob}, Anomaly: {anomaly}, Drift: {drift}")
        print(f"  Final Score: {score:.4f}, Band: {band}")
        print(f"  Breakdown: {breakdown}")
