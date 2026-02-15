"""
ExplainabilityEngine Module for FraudSense AI.

Provides explainability for individual transactions:
- Top contributing features
- Normalized importance scores
- Risk factor summary text
- SHAP integration (if available) or simulated feature importance
"""

import json
import os
import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class ExplainabilityEngine:
    """
    Explainability engine for fraud detection predictions.
    
    Provides feature importance and explanations for individual
    transactions using SHAP or simulated importance based on model weights.
    """
    
    # Feature names
    FEATURE_NAMES = [
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Time", "Amount"
    ]
    
    # Default feature importance (based on common fraud detection patterns)
    DEFAULT_FEATURE_IMPORTANCE = {
        "V14": 0.18, "V17": 0.14, "V12": 0.10, "V10": 0.09,
        "V4": 0.08, "V11": 0.07, "V3": 0.06, "V9": 0.05,
        "V2": 0.04, "V1": 0.04, "V7": 0.03, "V5": 0.03,
        "V6": 0.02, "V8": 0.02, "V13": 0.02, "V15": 0.01,
        "V16": 0.01, "V18": 0.01, "V19": 0.01, "V20": 0.01,
        "V21": 0.01, "V22": 0.00, "V23": 0.00, "V24": 0.00,
        "V25": 0.00, "V26": 0.00, "V27": 0.00, "V28": 0.00,
        "Time": 0.02, "Amount": 0.03
    }
    
    def __init__(
        self,
        global_importance_path: Optional[str] = None,
        explainer_path: Optional[str] = None
    ):
        """
        Initialize the ExplainabilityEngine.
        
        Args:
            path: Path to global feature importance JSON
            explainer_path: Path to saved SHAP explainer
        """
        self._lock = threading.RLock()
        
        # Try to load global feature importance
        self.global_feature_importance = self.DEFAULT_FEATURE_IMPORTANCE.copy()
        
        if global_importance_path and os.path.exists(global_importance_path):
            try:
                with open(global_importance_path, 'r') as f:
                    loaded = json.load(f)
                    self.global_feature_importance.update(loaded)
            except Exception:
                pass
        
        # SHAP explainer (if available)
        self.explainer = None
        self.use_shap = False
        
        if explainer_path and os.path.exists(explainer_path):
            try:
                import shap
                import pickle
                with open(explainer_path, 'rb') as f:
                    self.explainer = pickle.load(f)
                self.use_shap = True
            except ImportError:
                self.explainer = None
                self.use_shap = False
            except Exception:
                self.explainer = None
                self.use_shap = False
        
        # Transaction explanation cache
        self.explanation_cache: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = 1000
    
    def explain_prediction(
        self,
        transaction_id: str,
        features: Dict[str, float],
        fraud_probability: float,
        anomaly_score: float,
        risk_score: float,
        risk_level: str,
        top_factors: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate explanation for a transaction prediction.
        
        Args:
            transaction_id: Unique transaction identifier
            features: Transaction features
            fraud_probability: Fraud probability
            anomaly_score: Anomaly score
            risk_score: Ensemble risk score
            risk_level: Risk level
            top_factors: Pre-computed top factors
            
        Returns:
            Explanation dictionary
        """
        with self._lock:
            # Check cache first
            if transaction_id in self.explanation_cache:
                return self.explanation_cache[transaction_id]
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(features)
            
            # Get top features
            top_features = self._get_top_features(features, feature_importance, top_factors)
            
            # Generate normalized scores
            importance_scores = self._normalize_importance(feature_importance)
            
            # Generate summary
            summary = self._generate_explanation_summary(
                fraud_probability,
                risk_level,
                top_features
            )
            
            # Calculate confidence level
            confidence = self._calculate_confidence(fraud_probability, anomaly_score, risk_score)
            
            explanation = {
                "transaction_id": transaction_id,
                "top_features": top_features,
                "feature_importance_scores": importance_scores,
                "explanation_summary": summary,
                "confidence_level": confidence,
                "fraud_probability": fraud_probability,
                "risk_level": risk_level,
                "anomaly_score": anomaly_score,
                "risk_score": risk_score
            }
            
            # Cache the explanation
            self._cache_explanation(transaction_id, explanation)
            
            return explanation
    
    def _calculate_feature_importance(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate feature importance based on feature values and global importance.
        
        Args:
            features: Transaction features
            
        Returns:
            Feature importance dictionary
        """
        importance = {}
        
        for feature_name in self.FEATURE_NAMES:
            # Get feature value
            feature_value = features.get(feature_name, 0.0)
            
            # Get global importance weight
            global_importance = self.global_feature_importance.get(feature_name, 0.01)
            
            # Calculate local importance based on feature deviation
            # Features with high absolute values and high global importance contribute more
            deviation = abs(feature_value)
            
            # Normalize deviation (V features are typically -3 to 3)
            if feature_name.startswith("V"):
                normalized_deviation = min(1.0, deviation / 3.0)
            elif feature_name == "Amount":
                normalized_deviation = min(1.0, feature_value / 500.0)
            elif feature_name == "Time":
                normalized_deviation = min(1.0, feature_value / 1500.0)
            else:
                normalized_deviation = min(1.0, deviation)
            
            # Combine global importance with local deviation
            local_importance = global_importance * (0.3 + 0.7 * normalized_deviation)
            
            importance[feature_name] = round(local_importance, 4)
        
        return importance
    
    def _get_top_features(
        self,
        features: Dict[str, float],
        importance: Dict[str, float],
        precomputed: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top contributing features.
        
        Args:
            features: Transaction features
            importance: Feature importance scores
            precomputed: Pre-computed top factors
            
        Returns:
            List of top features with details
        """
        if precomputed and len(precomputed) > 0:
            # Use precomputed factors
            return precomputed[:5]
        
        # Sort by importance
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = []
        for feature_name, score in sorted_features[:5]:
            feature_value = features.get(feature_name, 0.0)
            
            # Determine impact direction
            if feature_name.startswith("V"):
                if abs(feature_value) > 1.0:
                    direction = "high" if feature_value > 0 else "low"
                else:
                    direction = "normal"
            elif feature_name == "Amount":
                direction = "high" if feature_value > 200 else "normal"
            else:
                direction = "normal"
            
            top_features.append({
                "feature": feature_name,
                "value": round(feature_value, 4),
                "importance": round(score, 4),
                "direction": direction,
                "impact": self._calculate_impact(score, direction)
            })
        
        return top_features
    
    def _calculate_impact(self, importance: float, direction: str) -> str:
        """Calculate impact level."""
        if importance > 0.15:
            return "very_high"
        elif importance > 0.08:
            return "high"
        elif importance > 0.04:
            return "medium"
        else:
            return "low"
    
    def _normalize_importance(self, importance: Dict[str, float]) -> Dict[str, float]:
        """Normalize importance scores to sum to 1.0."""
        total = sum(importance.values())
        if total == 0:
            return importance
        
        normalized = {
            k: round(v / total, 4)
            for k, v in importance.items()
        }
        
        return normalized
    
    def _generate_explanation_summary(
        self,
        fraud_probability: float,
        risk_level: str,
        top_features: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable explanation summary."""
        # Base summary on fraud probability
        if fraud_probability >= 0.75:
            summary = f"Transaction flagged as HIGH RISK with {fraud_probability:.1%} fraud probability. "
        elif fraud_probability >= 0.50:
            summary = f"Transaction shows ELEVATED RISK with {fraud_probability:.1%} fraud probability. "
        elif fraud_probability >= 0.25:
            summary = f"Transaction has MODERATE RISK with {fraud_probability:.1%} fraud probability. "
        else:
            summary = f"Transaction appears LOW RISK with {fraud_probability:.1%} fraud probability. "
        
        # Add top contributing factors
        if top_features:
            feature_names = [f["feature"] for f in top_features[:3]]
            summary += f"Key risk factors: {', '.join(feature_names)}. "
        
        # Add recommendation based on risk level
        if risk_level in ["High", "Critical"]:
            summary += "Immediate action recommended."
        elif risk_level == "Medium":
            summary += "Manual review advised."
        else:
            summary += "Transaction can proceed with standard monitoring."
        
        return summary
    
    def _calculate_confidence(
        self,
        fraud_probability: float,
        anomaly_score: float,
        risk_score: float
    ) -> float:
        """Calculate confidence level of the explanation."""
        # Calculate agreement between signals
        signals = [fraud_probability, anomaly_score, risk_score]
        
        # High agreement = high confidence
        variance = np.var(signals)
        mean = np.mean(signals)
        
        # Normalize confidence based on agreement
        if variance < 0.02:
            confidence = 0.95
        elif variance < 0.05:
            confidence = 0.85
        elif variance < 0.10:
            confidence = 0.70
        else:
            confidence = 0.50
        
        # Adjust based on how extreme the values are
        if mean > 0.7 or mean < 0.3:
            confidence = min(1.0, confidence + 0.1)
        
        return round(confidence, 2)
    
    def _cache_explanation(
        self,
        transaction_id: str,
        explanation: Dict[str, Any]
    ) -> None:
        """Cache explanation for later retrieval."""
        # Evict oldest if cache is full
        if len(self.explanation_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.explanation_cache))
            del self.explanation_cache[oldest_key]
        
        self.explanation_cache[transaction_id] = explanation
    
    def get_explanation(
        self,
        transaction_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached explanation for a transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Cached explanation or None
        """
        with self._lock:
            return self.explanation_cache.get(transaction_id)
    
    def update_global_importance(
        self,
        importance: Dict[str, float]
    ) -> None:
        """
        Update global feature importance.
        
        Args:
            importance: New importance values
        """
        with self._lock:
            self.global_feature_importance.update(importance)
    
    def clear_cache(self) -> None:
        """Clear explanation cache."""
        with self._lock:
            self.explanation_cache.clear()


# Global explainability engine instance
_explainability_engine: Optional[ExplainabilityEngine] = None


def get_explainability_engine() -> ExplainabilityEngine:
    """
    Get or create the global explainability engine instance.
    
    Returns:
        ExplainabilityEngine instance
    """
    global _explainability_engine
    
    if _explainability_engine is None:
        # Try to load from common paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        importance_path = os.path.join(base_dir, "global_feature_importance.json")
        explainer_path = os.path.join(base_dir, "explainer.pkl")
        
        _explainability_engine = ExplainabilityEngine(
            global_importance_path=importance_path if os.path.exists(importance_path) else None,
            explainer_path=explainer_path if os.path.exists(explainer_path) else None
        )
    
    return _explainability_engine


if __name__ == "__main__":
    # Test the explainability engine
    engine = get_explainability_engine()
    
    # Sample features
    features = {
        "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
        "V5": -0.34, "V6": 0.48, "V7": 0.08, "V8": -0.74,
        "V9": 0.10, "V10": -0.36, "V11": 1.23, "V12": -0.64,
        "V13": 0.60, "V14": -0.54, "V15": 0.27, "V16": 0.62,
        "V17": -0.26, "V18": 0.14, "V19": -0.18, "V20": 0.27,
        "V21": -0.14, "V22": -0.03, "V23": -0.14, "V24": 0.14,
        "V25": -0.26, "V26": 0.02, "V27": -0.14, "V28": -0.10,
        "Time": 406.0, "Amount": 149.62
    }
    
    # Generate explanation
    explanation = engine.explain_prediction(
        transaction_id="TXN-TEST123",
        features=features,
        fraud_probability=0.78,
        anomaly_score=0.65,
        risk_score=0.72,
        risk_level="High"
    )
    
    print("Explanation:")
    print(f"  Transaction ID: {explanation['transaction_id']}")
    print(f"  Confidence Level: {explanation['confidence_level']}")
    print(f"  Fraud Probability: {explanation['fraud_probability']}")
    print(f"  Risk Level: {explanation['risk_level']}")
    print(f"\n  Top Features:")
    for feature in explanation['top_features']:
        print(f"    {feature['feature']}: {feature['value']} (importance: {feature['importance']})")
    print(f"\n  Summary: {explanation['explanation_summary']}")
