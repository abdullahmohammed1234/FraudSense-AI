"""
Model module for FraudSense AI.

This module handles loading the trained model and SHAP explainer,
making predictions, and generating explanations.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional, Tuple
import os
from collections import Counter

# Default paths
MODEL_PATH = "model.pkl"
EXPLAINER_PATH = "explainer.pkl"
ANOMALY_MODEL_PATH = "anomaly_model.pkl"
TRAINING_STATS_PATH = "training_stats.json"

# Drift detection threshold
DRIFT_THRESHOLD = 2.5  # Standard deviations


class FraudDetectionModel:
    """
    Fraud detection model wrapper that handles predictions and explanations.
    """
    
    def __init__(self, model_path: str = MODEL_PATH, explainer_path: str = EXPLAINER_PATH):
        """
        Initialize the model by loading from disk.
        
        Args:
            model_path: Path to the saved model.
            explainer_path: Path to the saved SHAP explainer.
        """
        # Load model
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.best_threshold = model_data.get("best_threshold", 0.5)  # Default to 0.5
        
        # Load explainer if available
        self.explainer = None
        if os.path.exists(EXPLAINER_PATH):
            try:
                self.explainer = joblib.load(EXPLAINER_PATH)
                print(f"Loaded explainer from {EXPLAINER_PATH}")
            except Exception as e:
                print(f"Warning: Could not load explainer: {e}")
        
        # Load anomaly model
        self.anomaly_model = None
        self.anomaly_min = -1.0
        self.anomaly_max = -0.5
        if os.path.exists(ANOMALY_MODEL_PATH):
            try:
                anomaly_data = joblib.load(ANOMALY_MODEL_PATH)
                self.anomaly_model = anomaly_data["model"]
                self.anomaly_min = anomaly_data.get("min_score", -1.0)
                self.anomaly_max = anomaly_data.get("max_score", -0.5)
                print(f"Loaded anomaly model from {ANOMALY_MODEL_PATH}")
            except Exception as e:
                print(f"Warning: Could not load anomaly model: {e}")
        
        # Load training statistics for drift detection
        self.training_mean = None
        self.training_std = None
        if os.path.exists(TRAINING_STATS_PATH):
            try:
                import json
                with open(TRAINING_STATS_PATH, 'r') as f:
                    stats = json.load(f)
                self.training_mean = np.array(stats["mean"])
                self.training_std = np.array(stats["std"])
                print(f"Loaded training statistics from {TRAINING_STATS_PATH}")
            except Exception as e:
                print(f"Warning: Could not load training statistics: {e}")
        
        print(f"Model loaded successfully")
        print(f"Features: {len(self.feature_names)}")
        print(f"Optimized threshold: {self.best_threshold:.4f}")
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            features: Dictionary of feature values.
            
        Returns:
            numpy array of features in correct order.
        """
        # Create a DataFrame with all features
        feature_array = []
        
        for fname in self.feature_names:
            if fname in features:
                feature_array.append(features[fname])
            else:
                # Default to 0 if not provided
                feature_array.append(0.0)
        
        # Ensure proper numpy array with correct dtype
        return np.array(feature_array, dtype=np.float64).reshape(1, -1)
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a prediction with fraud probability and explanation.
        
        Args:
            features: Dictionary of transaction features.
            
        Returns:
            Dictionary containing prediction results.
        """
        # Prepare features
        X = self._prepare_features(features)
        
        # Get prediction and probability
        proba = self.model.predict_proba(X)
        
        # Handle case where model only returns one class
        if proba.shape[1] == 1:
            # Model was trained on single class, assume probability = 0
            probability = 0.0
        else:
            probability = float(proba[0][1])
        
        # Use optimized threshold for prediction
        predicted_class = int(probability >= self.best_threshold)
        
        # Get top contributing features using SHAP
        top_factors = self._get_top_factors(X)
        
        # Determine risk level
        risk_level = self._get_risk_level(probability)
        
        # Get anomaly score
        anomaly_score = self._get_anomaly_score(X)
        
        # Check for concept drift
        drift_detected, drift_distance = self._check_drift(X)
        
        # Generate human-readable explanation
        explanation_summary = self._generate_explanation(top_factors, X)
        
        return {
            "predicted_class": predicted_class,
            "fraud_probability": probability,
            "risk_level": risk_level,
            "top_factors": top_factors,
            "anomaly_score": anomaly_score,
            "drift_detected": drift_detected,
            "explanation_summary": explanation_summary
        }
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Determine risk level based on fraud probability.
        
        Args:
            probability: Fraud probability.
            
        Returns:
            Risk level string.
        """
        if probability < 0.2:
            return "Low"
        elif probability <= 0.6:
            return "Medium"
        else:
            return "High"
    
    def _get_top_factors(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Get top contributing features using SHAP values.
        
        Args:
            X: Feature array.
            
        Returns:
            List of top 3 contributing features with their impacts.
        """
        if self.explainer is None:
            return []
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # For binary classification, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Take positive class
            
            # Flatten to ensure it's a proper array
            shap_values = np.asarray(shap_values)
            
            # Ensure proper 2D array
            if shap_values.ndim > 2:
                shap_values = shap_values.reshape(-1, shap_values.shape[-1])
            
            # Get absolute values for importance
            abs_shap = np.abs(shap_values[0])
            
            # Get top 3 indices
            top_indices = np.argsort(abs_shap)[-3:][::-1]
            
            # Build result
            top_factors = []
            for idx in top_indices:
                feature_name = self.feature_names[idx]
                impact = float(shap_values[0][idx])
                top_factors.append({
                    "feature": feature_name,
                    "impact": round(impact, 4)
                })
            
            return top_factors
        
        except Exception as e:
            print(f"Warning: Could not compute SHAP values: {e}")
            return []

    def _get_anomaly_score(self, X: np.ndarray) -> float:
        """
        Get anomaly score using IsolationForest.
        
        Args:
            X: Feature array.
            
        Returns:
            Normalized anomaly score (0-1).
        """
        if self.anomaly_model is None:
            return 0.0
        
        try:
            # Get raw anomaly score (more negative = more anomalous)
            raw_score = self.anomaly_model.score_samples(X)
            # Handle scalar return value
            if hasattr(raw_score, 'item'):
                raw_score = raw_score.item()
            # Convert to 0-1 scale (higher = more anomalous)
            # Score_samples returns negative scores, so we invert and scale
            normalized = 1.0 - (raw_score - self.anomaly_min) / (self.anomaly_max - self.anomaly_min + 1e-10)
            return float(np.clip(normalized, 0.0, 1.0))
        except Exception as e:
            print(f"Warning: Could not compute anomaly score: {e}")
            return 0.0

    def _check_drift(self, X: np.ndarray) -> Tuple[bool, float]:
        """
        Check for concept drift by comparing to training feature means.
        
        Args:
            X: Feature array.
            
        Returns:
            Tuple of (drift_detected, distance_from_mean).
        """
        # Drift detection is currently disabled to prevent false positives
        # The feature statistics comparison can trigger incorrectly on normal transactions
        # This can be re-enabled with a more robust drift detection algorithm in the future
        return False, 0.0

    def _generate_explanation(self, top_factors: List[Dict], X: np.ndarray) -> str:
        """
        Generate human-readable explanation from top factors.
        
        Args:
            top_factors: List of top contributing features.
            X: Feature array.
            
        Returns:
            Human-readable explanation string.
        """
        if not top_factors:
            return "Transaction appears normal based on feature patterns."
        
        explanations = []
        for factor in top_factors[:3]:
            feature = factor["feature"]
            impact = factor["impact"]
            
            # Get actual feature value
            try:
                feat_idx = self.feature_names.index(feature)
                feat_value = X[0][feat_idx]
            except:
                feat_value = 0.0
            
            direction = "unusually high" if impact > 0 else "unusually low"
            
            explanations.append(f"{feature} is {direction} ({feat_value:.2f})")
        
        if len(explanations) == 1:
            return f"This transaction was flagged due to {explanations[0].lower()} compared to typical patterns."
        elif len(explanations) == 2:
            return f"This transaction was flagged due to {explanations[0].lower()} and {explanations[1].lower()}."
        else:
            return f"This transaction was flagged due to {explanations[0].lower()}, {explanations[1].lower()}, and {explanations[2].lower()}."


class FraudRiskEngine:
    """
    Fraud Risk Intelligence Engine.
    
    Tracks rolling statistics for fraud detection analytics.
    """
    
    def __init__(self):
        """Initialize the risk engine with default statistics."""
        self.total_analyzed = 0
        self.high_risk_count = 0
        self.sum_probabilities = 0.0
        self.rolling_probabilities: List[float] = []
        self.feature_counter: Counter = Counter()
        self.max_rolling = 50
    
    def update(self, fraud_probability: float, top_factors: List[Dict[str, Any]]) -> None:
        """
        Update statistics with a new prediction.
        
        Args:
            fraud_probability: Fraud probability from prediction.
            top_factors: Top contributing features.
        """
        self.total_analyzed += 1
        self.sum_probabilities += fraud_probability
        
        # Track high-risk count
        if fraud_probability >= 0.6:
            self.high_risk_count += 1
        
        # Update rolling probabilities
        self.rolling_probabilities.append(fraud_probability)
        if len(self.rolling_probabilities) > self.max_rolling:
            self.rolling_probabilities.pop(0)
        
        # Track top SHAP features
        if top_factors:
            top_feature = top_factors[0]["feature"] if top_factors else None
            if top_feature:
                self.feature_counter[top_feature] += 1
    
    def get_analytics(self) -> Dict[str, Any]:
        """
        Get current analytics summary.
        
        Returns:
            Dictionary with analytics data.
        """
        avg_prob = self.sum_probabilities / self.total_analyzed if self.total_analyzed > 0 else 0.0
        fraud_rate = (self.high_risk_count / self.total_analyzed * 100) if self.total_analyzed > 0 else 0.0
        
        # Get most frequent top feature
        top_feature = "N/A"
        if self.feature_counter:
            top_feature = self.feature_counter.most_common(1)[0][0]
        
        return {
            "total_transactions": self.total_analyzed,
            "high_risk_count": self.high_risk_count,
            "fraud_rate_percentage": round(fraud_rate, 2),
            "average_probability": round(avg_prob, 4),
            "top_global_risk_feature": top_feature,
            "rolling_probabilities": self.rolling_probabilities.copy()
        }


# Global model instance
_model: Optional[FraudDetectionModel] = None

# Global risk engine instance
_risk_engine: Optional[FraudRiskEngine] = None


def load_model(model_path: str = MODEL_PATH, explainer_path: str = EXPLAINER_PATH) -> FraudDetectionModel:
    """
    Load and return the fraud detection model.
    
    Args:
        model_path: Path to the saved model.
        explainer_path: Path to the saved SHAP explainer.
        
    Returns:
        Loaded FraudDetectionModel instance.
    """
    global _model
    
    if _model is None:
        _model = FraudDetectionModel(model_path, explainer_path)
    
    return _model


def predict_fraud(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Make a fraud prediction.
    
    Args:
        features: Dictionary of transaction features.
        
    Returns:
        Prediction result dictionary.
    """
    model = load_model()
    result = model.predict(features)
    
    # Update risk engine
    risk_engine = get_risk_engine()
    risk_engine.update(result["fraud_probability"], result["top_factors"])
    
    return result


def get_risk_engine() -> FraudRiskEngine:
    """
    Get or create the global risk engine instance.
    
    Returns:
        FraudRiskEngine instance.
    """
    global _risk_engine
    
    if _risk_engine is None:
        _risk_engine = FraudRiskEngine()
    
    return _risk_engine


def get_analytics() -> Dict[str, Any]:
    """
    Get analytics from the risk engine.
    
    Returns:
        Analytics dictionary.
    """
    risk_engine = get_risk_engine()
    return risk_engine.get_analytics()


if __name__ == "__main__":
    # Test the model
    print("Testing FraudDetectionModel...")
    
    # Load model
    model = load_model()
    
    # Test with sample features
    sample_features = {
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536346,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551599,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62,
        "Time": 406.0
    }
    
    result = model.predict(sample_features)
    print("\nPrediction result:")
    print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Top Factors: {result['top_factors']}")
