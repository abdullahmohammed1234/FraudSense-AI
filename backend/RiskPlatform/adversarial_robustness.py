"""
Adversarial Robustness Module for FraudSense AI.

Provides protection against model evasion attacks through:
- Adversarial input detection
- Input preprocessing/defensing
- Model robustness testing
- Attack detection and alerting
"""

import os
import json
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np


class AdversarialDetector:
    """
    Adversarial attack detection and robustness enhancement.
    
    Detects and mitigates adversarial attacks on fraud detection models.
    """
    
    FEATURE_NAMES = [
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Time", "Amount"
    ]
    
    # Statistical bounds for normal transactions (based on typical distribution)
    NORMAL_BOUNDS = {
        f"V{i}": (-3.0, 3.0) for i in range(1, 29)
    }
    NORMAL_BOUNDS.update({
        "Time": (0.0, 2000.0),
        "Amount": (0.0, 25000.0)
    })
    
    # Known adversarial patterns
    ADVERSARIAL_PATTERNS = {
        "extreme_values": {"threshold": 4.5, "consecutive": 3},
        "gradient_signs": {"min_signs": 5},
        "frequency_anomaly": {"threshold": 100}
    }
    
    def __init__(self):
        self._lock = threading.RLock()
        self._detection_history: List[Dict[str, Any]] = []
        self._model_statistics = self._initialize_statistics()
        self._defense_enabled = True
        self._sensitivity = "medium"
        self.max_history = 1000
        
    def _initialize_statistics(self) -> Dict[str, Any]:
        """Initialize model statistics for anomaly detection."""
        return {
            "mean_variation": 0.0,
            "std_variation": 1.0,
            "feature_means": {f: 0.0 for f in self.FEATURE_NAMES},
            "feature_stds": {f: 1.0 for f in self.FEATURE_NAMES},
            "max_recent_variation": 2.0
        }
    
    def analyze_input(
        self,
        features: Dict[str, float],
        transaction_id: str
    ) -> Dict[str, Any]:
        """
        Analyze input for adversarial characteristics.
        
        Args:
            features: Transaction features
            transaction_id: Transaction identifier
            
        Returns:
            Analysis results with adversarial flags
        """
        with self._lock:
            analysis = {
                "transaction_id": transaction_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "is_adversarial": False,
                "adversarial_score": 0.0,
                "flags": [],
                "defense_action": "none",
                "details": {}
            }
            
            # Run detection checks
            analysis["details"]["statistical_check"] = self._check_statistical_anomalies(features)
            analysis["details"]["pattern_check"] = self._check_adversarial_patterns(features)
            analysis["details"]["gradient_check"] = self._check_gradient_signs(features)
            analysis["details"]["frequency_check"] = self._check_frequency_anomaly(features)
            
            # Calculate overall adversarial score
            scores = []
            for check_name, check_result in analysis["details"].items():
                if check_result.get("flag", False):
                    scores.append(check_result.get("score", 0.5))
                    analysis["flags"].append(check_name)
            
            if scores:
                analysis["adversarial_score"] = round(sum(scores) / len(scores), 4)
                analysis["is_adversarial"] = analysis["adversarial_score"] > self._get_threshold()
            
            # Determine defense action
            if analysis["is_adversarial"]:
                analysis["defense_action"] = self._determine_defense_action(analysis["adversarial_score"])
            
            # Store in history
            self._store_detection(analysis)
            
            return analysis
    
    def _check_statistical_anomalies(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Check for statistical anomalies in features."""
        result = {"flag": False, "score": 0.0, "anomalies": []}
        
        extreme_count = 0
        for feature_name in self.FEATURE_NAMES:
            if feature_name not in features:
                continue
                
            value = features[feature_name]
            bounds = self.NORMAL_BOUNDS.get(feature_name, (-3.0, 3.0))
            
            # Check for extreme values
            if abs(value) > bounds[1] * 1.5:
                extreme_count += 1
                result["anomalies"].append({
                    "feature": feature_name,
                    "value": round(value, 4),
                    "expected_bounds": bounds
                })
        
        if extreme_count >= 3:
            result["flag"] = True
            result["score"] = min(1.0, extreme_count / 10)
        
        return result
    
    def _check_adversarial_patterns(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Check for known adversarial patterns."""
        result = {"flag": False, "score": 0.0, "patterns_found": []}
        
        # Check for consecutive extreme values
        extreme_features = []
        for feature_name in self.FEATURE_NAMES:
            if feature_name not in features:
                continue
            if abs(features[feature_name]) > self.ADVERSARIAL_PATTERNS["extreme_values"]["threshold"]:
                extreme_features.append(feature_name)
        
        if len(extreme_features) >= self.ADVERSARIAL_PATTERNS["extreme_values"]["consecutive"]:
            result["flag"] = True
            result["score"] = 0.7
            result["patterns_found"].append("consecutive_extreme_values")
        
        # Check for suspicious sign patterns (common in adversarial examples)
        positive_count = sum(1 for f in self.FEATURE_NAMES if f in features and features[f] > 0)
        negative_count = sum(1 for f in self.FEATURE_NAMES if f in features and features[f] < 0)
        
        if positive_count > 25 or negative_count > 25:
            result["flag"] = True
            result["score"] = max(result["score"], 0.5)
            result["patterns_found"].append("suspicious_sign_distribution")
        
        return result
    
    def _check_gradient_signs(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Check for suspicious gradient patterns."""
        result = {"flag": False, "score": 0.0, "sign_patterns": {}}
        
        # Calculate feature gradients (simplified)
        values = [features.get(f, 0.0) for f in self.FEATURE_NAMES if f in features]
        if len(values) < 2:
            return result
        
        # Check for alternating signs (common in adversarial)
        signs = [1 if v > 0 else -1 for v in values]
        alternations = sum(1 for i in range(len(signs)-1) if signs[i] != signs[i+1])
        
        if alternations > len(signs) * 0.7:
            result["flag"] = True
            result["score"] = 0.6
            result["sign_patterns"]["alternating_ratio"] = round(alternations / len(signs), 3)
        
        return result
    
    def _check_frequency_anomaly(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Check for frequency-based anomalies."""
        result = {"flag": False, "score": 0.0}
        
        # Check Amount for suspicious patterns
        amount = features.get("Amount", 0.0)
        
        # Round amounts are suspicious
        if amount > 0 and amount == round(amount):
            if amount % 100 == 0 and amount > 500:
                result["flag"] = True
                result["score"] = 0.4
        
        # Very specific amounts
        suspicious_amounts = [99.99, 199.99, 299.99, 499.99, 999.99]
        if amount in suspicious_amounts:
            result["flag"] = True
            result["score"] = max(result["score"], 0.5)
        
        return result
    
    def _get_threshold(self) -> float:
        """Get detection threshold based on sensitivity."""
        thresholds = {"low": 0.7, "medium": 0.5, "high": 0.3}
        return thresholds.get(self._sensitivity, 0.5)
    
    def _determine_defense_action(self, adversarial_score: float) -> str:
        """Determine defense action based on adversarial score."""
        if adversarial_score > 0.8:
            return "block"
        elif adversarial_score > 0.6:
            return "review"
        else:
            return "flag"
    
    def preprocess_input(
        self,
        features: Dict[str, float],
        apply_defense: bool = True
    ) -> Dict[str, float]:
        """
        Apply preprocessing to defend against adversarial attacks.
        
        Args:
            features: Original features
            apply_defense: Whether to apply defense techniques
            
        Returns:
            Preprocessed features
        """
        if not apply_defense or not self._defense_enabled:
            return features
        
        defended = features.copy()
        
        # Feature squeezing: reduce precision
        for key in defended:
            if isinstance(defended[key], (int, float)):
                defended[key] = round(defended[key], 3)
        
        # Clip extreme values
        for feature_name in self.FEATURE_NAMES:
            if feature_name in defended:
                bounds = self.NORMAL_BOUNDS.get(feature_name, (-3.0, 3.0))
                defended[feature_name] = np.clip(defended[feature_name], bounds[0], bounds[1])
        
        # Add small noise (randomization defense)
        noise_level = 0.01
        for feature_name in self.FEATURE_NAMES:
            if feature_name in defended:
                noise = np.random.normal(0, noise_level)
                defended[feature_name] += noise
        
        return defended
    
    def _store_detection(self, analysis: Dict[str, Any]) -> None:
        """Store detection result in history."""
        if len(self._detection_history) >= self.max_history:
            self._detection_history.pop(0)
        self._detection_history.append(analysis)
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        with self._lock:
            total = len(self._detection_history)
            if total == 0:
                return {"total_analyzed": 0, "adversarial_detected": 0, "attack_rate": 0.0}
            
            adversarial = sum(1 for a in self._detection_history if a.get("is_adversarial", False))
            
            return {
                "total_analyzed": total,
                "adversarial_detected": adversarial,
                "attack_rate": round(adversarial / total, 4),
                "defense_actions": self._count_defense_actions(),
                "flags_breakdown": self._get_flags_breakdown()
            }
    
    def _count_defense_actions(self) -> Dict[str, int]:
        """Count defense actions taken."""
        actions = {"block": 0, "review": 0, "flag": 0, "none": 0}
        for analysis in self._detection_history:
            action = analysis.get("defense_action", "none")
            actions[action] = actions.get(action, 0) + 1
        return actions
    
    def _get_flags_breakdown(self) -> Dict[str, int]:
        """Get breakdown of detection flags."""
        flags = {}
        for analysis in self._detection_history:
            for flag in analysis.get("flags", []):
                flags[flag] = flags.get(flag, 0) + 1
        return flags
    
    def set_sensitivity(self, sensitivity: str) -> None:
        """Set detection sensitivity (low/medium/high)."""
        if sensitivity in ["low", "medium", "high"]:
            self._sensitivity = sensitivity
    
    def enable_defense(self, enabled: bool) -> None:
        """Enable or disable defense mechanisms."""
        self._defense_enabled = enabled
    
    def clear_history(self) -> None:
        """Clear detection history."""
        with self._lock:
            self._detection_history.clear()


class AdversarialRobustnessTester:
    """
    Test model robustness against adversarial attacks.
    """
    
    def __init__(self, detector: AdversarialDetector):
        self._detector = detector
    
    def generate_adversarial_samples(
        self,
        features: Dict[str, float],
        num_samples: int = 10,
        epsilon: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Generate adversarial test samples."""
        samples = []
        
        for i in range(num_samples):
            perturbed = {}
            
            # FGSM-style perturbation
            for feature_name in self.FEATURE_NAMES:
                if feature_name in features:
                    original = features[feature_name]
                    # Random direction perturbation
                    direction = np.random.choice([-1, 1])
                    perturbed[feature_name] = original + direction * epsilon * np.random.uniform(0.5, 1.5)
            
            # Add Amount and Time
            perturbed["Amount"] = features.get("Amount", 100.0)
            perturbed["Time"] = features.get("Time", 500.0)
            
            result = self._detector.analyze_input(perturbed, f"ADV-TEST-{i}")
            result["original_features"] = features
            result["perturbed_features"] = perturbed
            result["perturbation_epsilon"] = epsilon
            
            samples.append(result)
        
        return samples
    
    def test_robustness(
        self,
        features: Dict[str, float],
        num_attacks: int = 20
    ) -> Dict[str, Any]:
        """Test model robustness against various attack strategies."""
        results = {
            "original_detection": self._detector.analyze_input(features, "ORIGINAL"),
            "adversarial_samples": [],
            "robustness_score": 0.0,
            "vulnerabilities": []
        }
        
        epsilons = [0.05, 0.1, 0.2, 0.3, 0.5]
        
        for eps in epsilons:
            samples = self.generate_adversarial_samples(features, num_attacks // len(epsilons), eps)
            results["adversarial_samples"].extend(samples)
        
        # Calculate robustness score
        detected_count = sum(1 for s in results["adversarial_samples"] if s.get("is_adversarial", False))
        results["robustness_score"] = round(detected_count / len(results["adversarial_samples"]), 4)
        
        if results["robustness_score"] < 0.5:
            results["vulnerabilities"].append("Low detection rate on perturbed inputs")
        
        return results


# Global instances
_adversarial_detector: Optional[AdversarialDetector] = None
_robustness_tester: Optional[AdversarialRobustnessTester] = None


def get_adversarial_detector() -> AdversarialDetector:
    """Get or create global adversarial detector."""
    global _adversarial_detector
    if _adversarial_detector is None:
        _adversarial_detector = AdversarialDetector()
    return _adversarial_detector


def get_robustness_tester() -> AdversarialRobustnessTester:
    """Get or create global robustness tester."""
    global _robustness_tester
    if _robustness_tester is None:
        _robustness_tester = AdversarialRobustnessTester(get_adversarial_detector())
    return _robustness_tester


if __name__ == "__main__":
    detector = get_adversarial_detector()
    tester = get_robustness_tester()
    
    # Test with normal transaction
    normal_features = {
        "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
        "V5": -0.34, "V6": 0.48, "V7": 0.08, "V8": -0.74,
        "V9": 0.10, "V10": -0.36, "V11": 1.23, "V12": -0.64,
        "V13": 0.60, "V14": -0.54, "V15": 0.27, "V16": 0.62,
        "V17": -0.26, "V18": 0.14, "V19": -0.18, "V20": 0.27,
        "V21": -0.14, "V22": -0.03, "V23": -0.14, "V24": 0.14,
        "V25": -0.26, "V26": 0.02, "V27": -0.14, "V28": -0.10,
        "Time": 406.0, "Amount": 149.62
    }
    
    result = detector.analyze_input(normal_features, "TXN-NORMAL")
    print(f"Normal Transaction Analysis:")
    print(f"  Adversarial: {result['is_adversarial']}")
    print(f"  Score: {result['adversarial_score']}")
    print(f"  Flags: {result['flags']}")
    
    # Test with adversarial-looking transaction
    adversarial_features = {
        "V1": 4.8, "V2": -4.5, "V3": 4.2, "V4": -4.9,
        "V5": 4.1, "V6": -3.8, "V7": 4.6, "V8": -4.3,
        "V9": 4.0, "V10": -4.7, "V11": 4.2, "V12": -4.1,
        "V13": 3.9, "V14": -4.8, "V15": 4.0, "V16": -4.2,
        "V17": 4.5, "V18": -4.0, "V19": 3.8, "V20": -4.6,
        "V21": 4.1, "V22": -3.9, "V23": 4.3, "V24": -4.4,
        "V25": 3.7, "V26": -4.5, "V27": 4.2, "V28": -4.1,
        "Time": 1800.0, "Amount": 999.99
    }
    
    result = detector.analyze_input(adversarial_features, "TXN-ADVERSARIAL")
    print(f"\nAdversarial Transaction Analysis:")
    print(f"  Adversarial: {result['is_adversarial']}")
    print(f"  Score: {result['adversarial_score']}")
    print(f"  Flags: {result['flags']}")
    print(f"  Defense Action: {result['defense_action']}")
    
    # Test preprocessing defense
    defended = detector.preprocess_input(adversarial_features)
    print(f"\nDefended features (V1): {defended['V1']}")