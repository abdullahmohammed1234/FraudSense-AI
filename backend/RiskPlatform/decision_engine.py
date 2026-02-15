"""
DecisionEngine Module for FraudSense AI.

Provides intelligent decision recommendation based on fraud probability,
anomaly score, and configurable thresholds.
"""

from typing import Dict, Any, Optional
from enum import Enum

from .config import DecisionConfig, GovernanceConfig


class ActionRecommendation(str, Enum):
    """Possible action recommendations."""
    AUTO_BLOCK = "Auto-Block Transaction"
    MANUAL_REVIEW = "Manual Review Required"
    APPROVE = "Approve"


class RiskLevel(str, Enum):
    """Risk level categories."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class DecisionEngine:
    """
    Decision Recommendation Engine.
    
    Analyzes fraud probability, anomaly score, and other signals
    to provide actionable recommendations.
    """
    
    def __init__(
        self,
        block_threshold: float = DecisionConfig.BLOCK_THRESHOLD,
        review_threshold: float = DecisionConfig.REVIEW_THRESHOLD,
        high_anomaly_threshold: float = DecisionConfig.HIGH_ANOMALY_THRESHOLD,
        moderate_anomaly_threshold: float = DecisionConfig.MODERATE_ANOMALY_THRESHOLD
    ):
        """
        Initialize the DecisionEngine.
        
        Args:
            block_threshold: Threshold for auto-blocking.
            review_threshold: Threshold for manual review.
            high_anomaly_threshold: High anomaly score threshold.
            moderate_anomaly_threshold: Moderate anomaly score threshold.
        """
        self.block_threshold = block_threshold
        self.review_threshold = review_threshold
        self.high_anomaly_threshold = high_anomaly_threshold
        self.moderate_anomaly_threshold = moderate_anomaly_threshold
    
    def get_recommendation(
        self,
        fraud_probability: float,
        anomaly_score: float,
        drift_detected: bool = False
    ) -> Dict[str, Any]:
        """
        Get action recommendation based on fraud probability and anomaly score.
        
        Logic:
        - If fraud_probability > block_threshold AND anomaly_score high → Auto-Block
        - If moderate signals → Manual Review Required
        - Otherwise → Approve
        
        Args:
            fraud_probability: Fraud probability (0-1).
            anomaly_score: Anomaly score (0-1).
            drift_detected: Whether drift was detected.
            
        Returns:
            Dictionary with recommendation and reasoning.
        """
        # Determine action based on logic
        if fraud_probability >= self.block_threshold and anomaly_score >= self.high_anomaly_threshold:
            # High probability AND high anomaly = Auto-Block
            recommendation = ActionRecommendation.AUTO_BLOCK
            reasoning = f"High fraud probability ({fraud_probability:.2f}) combined with high anomaly score ({anomaly_score:.2f})"
        elif fraud_probability >= self.block_threshold:
            # High probability but moderate anomaly = Manual Review
            recommendation = ActionRecommendation.MANUAL_REVIEW
            reasoning = f"High fraud probability ({fraud_probability:.2f}) - requires manual verification"
        elif fraud_probability >= self.review_threshold or anomaly_score >= self.moderate_anomaly_threshold:
            # Either moderate probability OR moderate anomaly = Manual Review
            recommendation = ActionRecommendation.MANUAL_REVIEW
            if fraud_probability >= self.review_threshold:
                reasoning = f"Moderate fraud probability ({fraud_probability:.2f})"
            else:
                reasoning = f"Moderate anomaly score ({anomaly_score:.2f})"
        else:
            # Low probability and low anomaly = Approve
            recommendation = ActionRecommendation.APPROVE
            reasoning = f"Low fraud probability ({fraud_probability:.2f}) and low anomaly score ({anomaly_score:.2f})"
        
        # Add drift warning if applicable
        if drift_detected:
            reasoning += " (Warning: Drift detected in input data)"
        
        return {
            "action_recommendation": recommendation.value,
            "reasoning": reasoning,
            "confidence": self._calculate_confidence(fraud_probability, anomaly_score),
            "requires_escalation": recommendation != ActionRecommendation.APPROVE
        }
    
    def _calculate_confidence(self, fraud_probability: float, anomaly_score: float) -> str:
        """
        Calculate confidence level of the recommendation.
        
        Args:
            fraud_probability: Fraud probability.
            anomaly_score: Anomaly score.
            
        Returns:
            Confidence level (High, Medium, Low).
        """
        # Calculate a combined signal strength
        combined = (fraud_probability + anomaly_score) / 2
        
        if combined >= 0.75:
            return "High"
        elif combined >= 0.50:
            return "Medium"
        else:
            return "Low"
    
    def get_risk_level(
        self,
        fraud_probability: float,
        anomaly_score: float = 0.0,
        final_risk_score: Optional[float] = None
    ) -> str:
        """
        Determine risk level based on various inputs.
        
        Args:
            fraud_probability: Fraud probability (0-1).
            anomaly_score: Anomaly score (0-1).
            final_risk_score: Optional ensemble risk score.
            
        Returns:
            Risk level string.
        """
        # Use final_risk_score if provided, otherwise use fraud_probability
        score = final_risk_score if final_risk_score is not None else fraud_probability
        
        if score < DecisionConfig.LOW_RISK_THRESHOLD:
            return RiskLevel.LOW.value
        elif score < DecisionConfig.MEDIUM_RISK_THRESHOLD:
            return RiskLevel.MEDIUM.value
        elif score < DecisionConfig.HIGH_RISK_THRESHOLD:
            return RiskLevel.HIGH.value
        elif score < DecisionConfig.CRITICAL_RISK_THRESHOLD:
            return RiskLevel.HIGH.value
        else:
            return RiskLevel.CRITICAL.value
    
    def get_threshold_used(self) -> float:
        """
        Get the current threshold used for decisions.
        
        Returns:
            Threshold value.
        """
        return self.block_threshold
    
    def set_thresholds(
        self,
        block_threshold: Optional[float] = None,
        review_threshold: Optional[float] = None,
        high_anomaly_threshold: Optional[float] = None,
        moderate_anomaly_threshold: Optional[float] = None
    ) -> None:
        """
        Update thresholds.
        
        Args:
            block_threshold: New block threshold.
            review_threshold: New review threshold.
            high_anomaly_threshold: New high anomaly threshold.
            moderate_anomaly_threshold: New moderate anomaly threshold.
        """
        if block_threshold is not None:
            self.block_threshold = block_threshold
        if review_threshold is not None:
            self.review_threshold = review_threshold
        if high_anomaly_threshold is not None:
            self.high_anomaly_threshold = high_anomaly_threshold
        if moderate_anomaly_threshold is not None:
            self.moderate_anomaly_threshold = moderate_anomaly_threshold
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Dictionary with current settings.
        """
        return {
            "block_threshold": self.block_threshold,
            "review_threshold": self.review_threshold,
            "high_anomaly_threshold": self.high_anomaly_threshold,
            "moderate_anomaly_threshold": self.moderate_anomaly_threshold
        }


# Global decision engine instance
_decision_engine: Optional[DecisionEngine] = None


def get_decision_engine() -> DecisionEngine:
    """
    Get or create the global decision engine instance.
    
    Returns:
        DecisionEngine instance.
    """
    global _decision_engine
    
    if _decision_engine is None:
        _decision_engine = DecisionEngine()
    
    return _decision_engine


def get_recommendation(
    fraud_probability: float,
    anomaly_score: float,
    drift_detected: bool = False
) -> Dict[str, Any]:
    """
    Get action recommendation (convenience function).
    
    Args:
        fraud_probability: Fraud probability.
        anomaly_score: Anomaly score.
        drift_detected: Whether drift was detected.
        
    Returns:
        Recommendation dictionary.
    """
    engine = get_decision_engine()
    return engine.get_recommendation(fraud_probability, anomaly_score, drift_detected)


if __name__ == "__main__":
    # Test the decision engine
    engine = get_decision_engine()
    
    # Test cases
    test_cases = [
        (0.85, 0.72, False),   # High prob, high anomaly
        (0.80, 0.30, False),   # High prob, low anomaly
        (0.45, 0.55, False),   # Moderate prob, moderate anomaly
        (0.15, 0.10, False),   # Low prob, low anomaly
        (0.60, 0.45, True),    # With drift detected
    ]
    
    for fraud_prob, anomaly, drift in test_cases:
        result = engine.get_recommendation(fraud_prob, anomaly, drift)
        risk_level = engine.get_risk_level(fraud_prob, anomaly)
        
        print(f"\nFraud Prob: {fraud_prob}, Anomaly: {anomaly}, Drift: {drift}")
        print(f"  Action: {result['action_recommendation']}")
        print(f"  Risk Level: {risk_level}")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Confidence: {result['confidence']}")
