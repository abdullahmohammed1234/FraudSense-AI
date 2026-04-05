"""
Counterfactual Explanations Module for FraudSense AI.

Generates actionable insights for false positives by suggesting minimal
feature changes that would flip a fraud prediction to non-fraud.
"""

import os
import json
import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class CounterfactualExplainer:
    """
    Counterfactual explanation generator for fraud predictions.
    
    Generates actionable insights showing what feature changes would
    result in a different prediction (false positive -> genuine).
    """
    
    FEATURE_NAMES = [
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Time", "Amount"
    ]
    
    # Feature constraints (min/max for realistic counterfactuals)
    FEATURE_CONSTRAINTS = {
        "V1": (-3.0, 3.0), "V2": (-3.0, 3.0), "V3": (-3.0, 3.0),
        "V4": (-3.0, 3.0), "V5": (-3.0, 3.0), "V6": (-3.0, 3.0),
        "V7": (-3.0, 3.0), "V8": (-3.0, 3.0), "V9": (-3.0, 3.0),
        "V10": (-3.0, 3.0), "V11": (-3.0, 3.0), "V12": (-3.0, 3.0),
        "V13": (-3.0, 3.0), "V14": (-3.0, 3.0), "V15": (-3.0, 3.0),
        "V16": (-3.0, 3.0), "V17": (-3.0, 3.0), "V18": (-3.0, 3.0),
        "V19": (-3.0, 3.0), "V20": (-3.0, 3.0), "V21": (-3.0, 3.0),
        "V22": (-3.0, 3.0), "V23": (-3.0, 3.0), "V24": (-3.0, 3.0),
        "V25": (-3.0, 3.0), "V26": (-3.0, 3.0), "V27": (-3.0, 3.0),
        "V28": (-3.0, 3.0), "Time": (0.0, 2000.0), "Amount": (0.0, 25000.0)
    }
    
    # Actionable feature mappings (real-world interpretations)
    ACTIONABLE_FEATURES = {
        "Amount": {
            "name": "Transaction Amount",
            "interpretation": "Lower transaction amount reduces fraud risk",
            "action": "Consider splitting large transactions or verifying with customer"
        },
        "Time": {
            "name": "Transaction Time",
            "interpretation": "Time of day affects fraud likelihood",
            "action": "Transactions during unusual hours may need additional verification"
        },
        "V14": {"name": "V14 Feature", "interpretation": "High V14 value indicates unusual pattern", "action": "Review transaction behavior patterns"},
        "V17": {"name": "V17 Feature", "interpretation": "V17 captures transaction velocity", "action": "Check for rapid successive transactions"},
        "V12": {"name": "V12 Feature", "interpretation": "V12 indicates deviation from normal", "action": "Verify with cardholder"}
    }
    
    def __init__(self):
        self._lock = threading.RLock()
        self._feature_weights = self._initialize_weights()
        self._counterfactual_cache: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = 500
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize feature weights based on fraud detection importance."""
        weights = {f"V{i}": 0.0 for i in range(1, 29)}
        weights.update({
            "V14": 0.18, "V17": 0.14, "V12": 0.10, "V10": 0.09,
            "V4": 0.08, "V11": 0.07, "V3": 0.06, "V9": 0.05,
            "V2": 0.04, "V1": 0.04, "V7": 0.03, "V5": 0.03,
            "V6": 0.02, "V8": 0.02, "V13": 0.02, "V15": 0.01,
            "V16": 0.01, "V18": 0.01, "V19": 0.01, "V20": 0.01,
            "V21": 0.01, "V22": 0.00, "V23": 0.00, "V24": 0.00,
            "V25": 0.00, "V26": 0.00, "V27": 0.00, "V28": 0.00,
            "Time": 0.02, "Amount": 0.03
        })
        return weights
    
    def generate_counterfactual(
        self,
        transaction_id: str,
        features: Dict[str, float],
        fraud_probability: float,
        target_probability: float = 0.3,
        model_predict_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanation for a false positive.
        
        Args:
            transaction_id: Unique transaction identifier
            features: Current transaction features
            fraud_probability: Current fraud probability
            target_probability: Target probability to achieve
            model_predict_fn: Optional function to get actual model prediction
            
        Returns:
            Counterfactual explanation with actionable insights
        """
        with self._lock:
            cache_key = f"{transaction_id}_{fraud_probability:.2f}"
            if cache_key in self._counterfactual_cache:
                return self._counterfactual_cache[cache_key]
            
            # Find minimal changes needed
            changes = self._find_minimal_changes(
                features, fraud_probability, target_probability, model_predict_fn
            )
            
            # Generate actionable insights
            actionable_changes = self._generate_actionable_changes(changes, features)
            
            # Calculate sparsity (how many features changed)
            sparsity = len([c for c in changes if c["changed"]]) / len(changes) if changes else 1.0
            
            counterfactual = {
                "transaction_id": transaction_id,
                "original_probability": fraud_probability,
                "target_probability": target_probability,
                "changes": actionable_changes,
                "sparsity": round(sparsity, 3),
                "num_changes": len([c for c in actionable_changes if c["changed"]]),
                "feasibility": self._evaluate_feasibility(actionable_changes),
                "summary": self._generate_summary(fraud_probability, target_probability, actionable_changes)
            }
            
            self._cache_counterfactual(cache_key, counterfactual)
            return counterfactual
    
    def _find_minimal_changes(
        self,
        features: Dict[str, float],
        current_prob: float,
        target_prob: float,
        model_fn: Optional[callable]
    ) -> List[Dict[str, Any]]:
        """Find minimal feature changes to achieve target probability."""
        changes = []
        
        # Calculate importance-adjusted threshold
        importance_threshold = (current_prob - target_prob) / len(self.FEATURE_NAMES)
        
        for feature_name in self.FEATURE_NAMES:
            if feature_name not in features:
                continue
                
            current_value = features[feature_name]
            weight = self._feature_weights.get(feature_name, 0.01)
            constraint = self.FEATURE_CONSTRAINTS.get(feature_name, (-3.0, 3.0))
            
            # Determine direction needed (if fraud prob is high, move towards normal range)
            if current_prob > target_prob:
                # Need to reduce fraud likelihood
                # Move towards mean (0 for V features)
                if feature_name.startswith("V"):
                    target_value = 0.0
                elif feature_name == "Amount":
                    target_value = min(current_value * 0.5, constraint[1] * 0.3)
                elif feature_name == "Time":
                    target_value = 800.0  # Normal business hours
                else:
                    target_value = current_value
            else:
                target_value = current_value
            
            # Calculate change
            change_needed = target_value - current_value
            
            # Only include if change would have meaningful impact
            impact = abs(change_needed) * weight
            if impact > importance_threshold * 0.5:
                changes.append({
                    "feature": feature_name,
                    "current_value": round(current_value, 4),
                    "proposed_value": round(target_value, 4),
                    "change": round(change_needed, 4),
                    "impact": round(impact, 4),
                    "changed": abs(change_needed) > 0.01
                })
        
        # Sort by impact and return top changes
        changes.sort(key=lambda x: x["impact"], reverse=True)
        return changes[:5]
    
    def _generate_actionable_changes(
        self,
        changes: List[Dict[str, Any]],
        original_features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Convert numerical changes to actionable recommendations."""
        actionable = []
        
        for change in changes:
            feature = change["feature"]
            action_info = self.ACTIONABLE_FEATURES.get(feature, {
                "name": f"Feature {feature}",
                "interpretation": f"Feature {feature} contributes to fraud risk",
                "action": "Review transaction details"
            })
            
            # Determine action type
            if feature == "Amount":
                if change["change"] < 0:
                    action_type = "reduce_amount"
                    specific_action = f"Reduce amount by ${abs(change['change']):.2f}"
                else:
                    action_type = "verify_amount"
                    specific_action = "Verify transaction amount with customer"
            elif feature == "Time":
                action_type = "verify_timing"
                specific_action = "Transaction at unusual hour - consider verification"
            elif feature.startswith("V") and abs(change["current_value"]) > 1.5:
                action_type = "behavioral_review"
                specific_action = "Unusual pattern detected - manual review recommended"
            else:
                action_type = "general_review"
                specific_action = "Review transaction context"
            
            actionable.append({
                "feature": feature,
                "feature_name": action_info["name"],
                "current_value": change["current_value"],
                "proposed_value": change["proposed_value"],
                "change": change["change"],
                "interpretation": action_info["interpretation"],
                "action": action_info["action"],
                "action_type": action_type,
                "specific_action": specific_action,
                "impact": change["impact"],
                "changed": change["changed"],
                "feasibility": self._assess_change_feasibility(change, feature)
            })
        
        return actionable
    
    def _assess_change_feasibility(self, change: Dict[str, Any], feature: str) -> str:
        """Assess how feasible the proposed change is."""
        if feature == "Amount":
            return "high"  # Can easily reduce amount
        elif feature == "Time":
            return "none"  # Cannot change transaction time
        elif feature.startswith("V"):
            return "medium"  # Requires behavioral change
        return "low"
    
    def _evaluate_feasibility(self, changes: List[Dict[str, Any]]) -> str:
        """Evaluate overall feasibility of counterfactual."""
        feasibilities = [c.get("feasibility", "low") for c in changes if c.get("changed")]
        if not feasibilities:
            return "high"
        
        if all(f == "high" for f in feasibilities):
            return "high"
        elif all(f == "none" for f in feasibilities):
            return "impossible"
        elif "none" in feasibilities:
            return "partial"
        return "medium"
    
    def _generate_summary(
        self,
        original_prob: float,
        target_prob: float,
        changes: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable summary."""
        num_changes = len([c for c in changes if c.get("changed", False)])
        
        if num_changes == 0:
            return "Transaction appears to be a true positive. No feasible counterfactual found."
        
        actionable = [c for c in changes if c.get("feasibility") in ["high", "medium"]]
        
        if not actionable:
            return f"To reduce fraud probability from {original_prob:.1%} to {target_prob:.1%}, behavioral changes required. Consider manual review."
        
        summary = f"To reduce fraud probability from {original_prob:.1%} to {target_prob:.1%}, consider {num_changes} changes: "
        summary += ", ".join([c.get("specific_action", c["feature"]) for c in actionable[:3]])
        summary += "."
        
        return summary
    
    def _cache_counterfactual(self, key: str, cf: Dict[str, Any]) -> None:
        """Cache counterfactual for later retrieval."""
        if len(self._counterfactual_cache) >= self.max_cache_size:
            oldest = next(iter(self._counterfactual_cache))
            del self._counterfactual_cache[oldest]
        self._counterfactual_cache[key] = cf
    
    def get_counterfactual(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get cached counterfactual."""
        with self._lock:
            for key, cf in self._counterfactual_cache.items():
                if key.startswith(transaction_id):
                    return cf
            return None
    
    def clear_cache(self) -> None:
        """Clear counterfactual cache."""
        with self._lock:
            self._counterfactual_cache.clear()


# Global instance
_counterfactual_explainer: Optional[CounterfactualExplainer] = None


def get_counterfactual_explainer() -> CounterfactualExplainer:
    """Get or create global counterfactual explainer."""
    global _counterfactual_explainer
    if _counterfactual_explainer is None:
        _counterfactual_explainer = CounterfactualExplainer()
    return _counterfactual_explainer


if __name__ == "__main__":
    explainer = get_counterfactual_explainer()
    
    features = {
        "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
        "V5": -0.34, "V6": 0.48, "V7": 0.08, "V8": -0.74,
        "V9": 0.10, "V10": -0.36, "V11": 1.23, "V12": -0.64,
        "V13": 0.60, "V14": -0.54, "V15": 0.27, "V16": 0.62,
        "V17": -0.26, "V18": 0.14, "V19": -0.18, "V20": 0.27,
        "V21": -0.14, "V22": -0.03, "V23": -0.14, "V24": 0.14,
        "V25": -0.26, "V26": 0.02, "V27": -0.14, "V28": -0.10,
        "Time": 406.0, "Amount": 450.0
    }
    
    cf = explainer.generate_counterfactual(
        transaction_id="TXN-TEST123",
        features=features,
        fraud_probability=0.78,
        target_probability=0.3
    )
    
    print("Counterfactual Explanation:")
    print(f"  Original: {cf['original_probability']:.1%}")
    print(f"  Target: {cf['target_probability']:.1%}")
    print(f"  Changes: {cf['num_changes']}")
    print(f"  Feasibility: {cf['feasibility']}")
    print(f"  Summary: {cf['summary']}")
    print("\nActionable Changes:")
    for c in cf["changes"]:
        if c.get("changed"):
            print(f"  - {c['feature_name']}: {c['specific_action']}")