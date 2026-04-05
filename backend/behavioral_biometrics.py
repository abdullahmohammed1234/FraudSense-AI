"""
Behavioral Biometrics Module.

This module provides user behavior analytics and device fingerprinting:
- Keystroke dynamics analysis
- Mouse movement patterns
- Device fingerprinting
- Session behavior analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import statistics


@dataclass
class BiometricSample:
    """Single biometric sample."""
    timestamp: datetime
    user_id: str
    session_id: str
    event_type: str
    features: Dict[str, Any]


@dataclass
class KeystrokeFeatures:
    """Keystroke dynamics features."""
    dwell_time: float
    flight_time: float
    press_press_latency: float
    release_press_latency: float
    typing_speed: float
    error_rate: float
    rhythm_variance: float


@dataclass
class MouseFeatures:
    """Mouse movement features."""
    avg_velocity: float
    max_velocity: float
    avg_acceleration: float
    curvature: float
    click_frequency: float
    scroll_frequency: float
    stillness_ratio: float
    movement_entropy: float


@dataclass
class DeviceFingerprint:
    """Device fingerprint."""
    fingerprint_hash: str
    user_agent: str
    screen_resolution: Tuple[int, int]
    timezone: str
    language: str
    platform: str
    plugins_count: int
    canvas_fingerprint: str
    audio_fingerprint: str


class KeystrokeAnalyzer:
    """Analyzer for keystroke dynamics."""
    
    KEYSTROKE_WEIGHTS = {
        "anomaly_dwell": 0.25,
        "anomaly_flight": 0.25,
        "high_error_rate": 0.3,
        "irregular_rhythm": 0.2,
    }
    
    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {
            "dwell_times": [],
            "flight_times": [],
            "typing_speeds": [],
            "error_rates": []
        })
    
    def analyze_keystroke(self, keystroke_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze keystroke patterns."""
        user_id = keystroke_data.get("user_id", "unknown")
        
        # Extract features
        dwell_time = keystroke_data.get("dwell_time", 0.0)
        flight_time = keystroke_data.get("flight_time", 0.0)
        typing_speed = keystroke_data.get("typing_speed", 0.0)
        error_rate = keystroke_data.get("error_rate", 0.0)
        
        # Get user profile
        profile = self.user_profiles[user_id]
        
        # Calculate anomaly scores
        risk_score = 0.0
        indicators = []
        
        # Check dwell time anomaly
        if profile["dwell_times"]:
            avg_dwell = statistics.mean(profile["dwell_times"])
            std_dwell = statistics.stdev(profile["dwell_times"]) if len(profile["dwell_times"]) > 1 else 0.1
            
            if abs(dwell_time - avg_dwell) > 2 * std_dwell:
                risk_score += self.KEYSTROKE_WEIGHTS["anomaly_dwell"]
                indicators.append("Unusual key press duration")
        
        # Check flight time anomaly
        if profile["flight_times"]:
            avg_flight = statistics.mean(profile["flight_times"])
            std_flight = statistics.stdev(profile["flight_times"]) if len(profile["flight_times"]) > 1 else 0.1
            
            if abs(flight_time - avg_flight) > 2 * std_flight:
                risk_score += self.KEYSTROKE_WEIGHTS["anomaly_flight"]
                indicators.append("Unusual key transition time")
        
        # Check error rate
        if error_rate > 0.1:
            risk_score += self.KEYSTROKE_WEIGHTS["high_error_rate"]
            indicators.append("High typing error rate")
        
        # Check rhythm irregularity
        if profile["typing_speeds"]:
            recent_speeds = profile["typing_speeds"][-10:]
            if len(recent_speeds) > 3:
                cv = statistics.stdev(recent_speeds) / statistics.mean(recent_speeds) if statistics.mean(recent_speeds) > 0 else 0
                if cv > 0.5:
                    risk_score += self.KEYSTROKE_WEIGHTS["irregular_rhythm"]
                    indicators.append("Irregular typing rhythm")
        
        # Update profile
        profile["dwell_times"].append(dwell_time)
        profile["flight_times"].append(flight_time)
        profile["typing_speeds"].append(typing_speed)
        profile["error_rates"].append(error_rate)
        
        # Keep only last 100 samples
        for key in profile:
            profile[key] = profile[key][-100:]
        
        risk_score = min(risk_score, 1.0)
        
        return {
            "biometric_risk_score": round(risk_score, 4),
            "fraud_indicators": indicators,
            "confidence": 0.8,
            "features": {
                "dwell_time": dwell_time,
                "flight_time": flight_time,
                "typing_speed": typing_speed,
                "error_rate": error_rate
            }
        }


class MouseMovementAnalyzer:
    """Analyzer for mouse movement patterns."""
    
    MOUSE_WEIGHTS = {
        "unusual_velocity": 0.2,
        "high_curvature": 0.2,
        "abnormal_click_pattern": 0.25,
        "robotic_movement": 0.25,
        "low_entropy": 0.1,
    }
    
    def __init__(self):
        self.user_baselines: Dict[str, Dict[str, float]] = {}
    
    def analyze_mouse_movement(self, mouse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mouse movement patterns."""
        user_id = mouse_data.get("user_id", "unknown")
        
        # Extract features
        avg_velocity = mouse_data.get("avg_velocity", 0.0)
        max_velocity = mouse_data.get("max_velocity", 0.0)
        curvature = mouse_data.get("curvature", 0.0)
        click_frequency = mouse_data.get("click_frequency", 0.0)
        stillness_ratio = mouse_data.get("stillness_ratio", 0.0)
        movement_entropy = mouse_data.get("movement_entropy", 0.0)
        
        risk_score = 0.0
        indicators = []
        
        # Get or create baseline
        if user_id in self.user_baselines:
            baseline = self.user_baselines[user_id]
            
            # Check velocity anomaly
            if abs(avg_velocity - baseline.get("avg_velocity", 0)) > baseline.get("velocity_std", 50) * 2:
                risk_score += self.MOUSE_WEIGHTS["unusual_velocity"]
                indicators.append("Unusual mouse movement speed")
            
            # Check curvature
            if curvature > baseline.get("avg_curvature", 0) * 1.5:
                risk_score += self.MOUSE_WEIGHTS["high_curvature"]
                indicators.append("Unusual movement patterns")
            
            # Check for robotic movement (very regular)
            if movement_entropy < 0.3:
                risk_score += self.MOUSE_WEIGHTS["robotic_movement"]
                indicators.append("Suspiciously regular mouse movement")
        else:
            # Create baseline
            self.user_baselines[user_id] = {
                "avg_velocity": avg_velocity,
                "velocity_std": max(avg_velocity * 0.3, 10),
                "avg_curvature": curvature,
            }
        
        # Check click patterns
        if click_frequency > 5:
            risk_score += self.MOUSE_WEIGHTS["abnormal_click_pattern"]
            indicators.append("Abnormal click pattern")
        
        # Check stillness
        if stillness_ratio > 0.8:
            indicators.append("Unusual stillness periods")
        
        risk_score = min(risk_score, 1.0)
        
        return {
            "biometric_risk_score": round(risk_score, 4),
            "fraud_indicators": indicators,
            "confidence": 0.75,
            "features": {
                "avg_velocity": avg_velocity,
                "max_velocity": max_velocity,
                "curvature": curvature,
                "click_frequency": click_frequency,
                "stillness_ratio": stillness_ratio
            }
        }


class DeviceFingerprinter:
    """Device fingerprinting system."""
    
    FINGERPRINT_WEIGHTS = {
        "new_device": 0.3,
        "new_browser": 0.2,
        "suspicious_fingerprint": 0.35,
        "emulator_detected": 0.5,
        " vpn_detected": 0.25,
    }
    
    def __init__(self):
        self.device_history: Dict[str, List[str]] = defaultdict(list)
        self.fingerprint_cache: Dict[str, DeviceFingerprint] = {}
    
    def create_fingerprint(self, device_data: Dict[str, Any]) -> DeviceFingerprint:
        """Create device fingerprint from device data."""
        
        # Combine device attributes
        fingerprint_components = [
            device_data.get("user_agent", ""),
            str(device_data.get("screen_resolution", "")),
            device_data.get("timezone", ""),
            device_data.get("language", ""),
            device_data.get("platform", ""),
            device_data.get("canvas_fingerprint", ""),
            device_data.get("audio_fingerprint", ""),
        ]
        
        # Create hash
        fingerprint_string = "|".join(fingerprint_components)
        fingerprint_hash = hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]
        
        return DeviceFingerprint(
            fingerprint_hash=fingerprint_hash,
            user_agent=device_data.get("user_agent", ""),
            screen_resolution=tuple(device_data.get("screen_resolution", [0, 0])),
            timezone=device_data.get("timezone", ""),
            language=device_data.get("language", ""),
            platform=device_data.get("platform", ""),
            plugins_count=device_data.get("plugins_count", 0),
            canvas_fingerprint=device_data.get("canvas_fingerprint", ""),
            audio_fingerprint=device_data.get("audio_fingerprint", "")
        )
    
    def analyze_device(self, device_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Analyze device for fraud indicators."""
        
        fingerprint = self.create_fingerprint(device_data)
        
        risk_score = 0.0
        indicators = []
        
        # Check if new device
        if fingerprint.fingerprint_hash not in self.device_history[user_id]:
            risk_score += self.FINGERPRINT_WEIGHTS["new_device"]
            indicators.append("New device detected")
            self.device_history[user_id].append(fingerprint.fingerprint_hash)
        
        # Check for emulator
        if device_data.get("is_emulator", False):
            risk_score += self.FINGERPRINT_WEIGHTS["emulator_detected"]
            indicators.append("Emulator detected")
        
        # Check for VPN
        if device_data.get("is_vpn", False):
            risk_score += self.FINGERPRINT_WEIGHTS[" vpn_detected"]
            indicators.append("VPN detected")
        
        # Check fingerprint consistency
        if fingerprint.user_agent == "" or fingerprint.canvas_fingerprint == "":
            risk_score += self.FINGERPRINT_WEIGHTS["suspicious_fingerprint"]
            indicators.append("Incomplete device fingerprint")
        
        # Keep only last 10 devices
        self.device_history[user_id] = self.device_history[user_id][-10:]
        
        risk_score = min(risk_score, 1.0)
        
        return {
            "device_risk_score": round(risk_score, 4),
            "fraud_indicators": indicators,
            "confidence": 0.85,
            "fingerprint": fingerprint.fingerprint_hash,
            "is_new_device": fingerprint.fingerprint_hash not in self.device_history[user_id][:-1]
        }


class SessionAnalyzer:
    """Analyzer for session behavior patterns."""
    
    SESSION_WEIGHTS = {
        "unusual_time": 0.2,
        "short_session": 0.25,
        "rapid_page_transitions": 0.2,
        "no_navigation_pattern": 0.25,
        "copy_paste_activity": 0.15,
    }
    
    def __init__(self):
        self.session_history: Dict[str, Dict[str, Any]] = {}
    
    def analyze_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze session for fraud indicators."""
        
        session_id = session_data.get("session_id", "unknown")
        user_id = session_data.get("user_id", "unknown")
        
        risk_score = 0.0
        indicators = []
        
        # Check session time
        current_hour = datetime.now().hour
        if current_hour < 5 or current_hour > 23:
            risk_score += self.SESSION_WEIGHTS["unusual_time"]
            indicators.append("Unusual session time")
        
        # Check session duration
        session_duration = session_data.get("duration_seconds", 0)
        if session_duration < 30:
            risk_score += self.SESSION_WEIGHTS["short_session"]
            indicators.append("Very short session")
        
        # Check page transition speed
        page_transitions = session_data.get("page_transitions", 0)
        transition_rate = page_transitions / max(session_duration, 1) * 60
        
        if transition_rate > 20:
            risk_score += self.SESSION_WEIGHTS["rapid_page_transitions"]
            indicators.append("Rapid page navigation")
        
        # Check for copy-paste activity
        paste_count = session_data.get("paste_events", 0)
        if paste_count > 5:
            risk_score += self.SESSION_WEIGHTS["copy_paste_activity"]
            indicators.append("Excessive copy-paste activity")
        
        # Check navigation pattern
        if session_data.get("has_navigation_pattern", True) == False:
            risk_score += self.SESSION_WEIGHTS["no_navigation_pattern"]
            indicators.append("No natural navigation pattern")
        
        risk_score = min(risk_score, 1.0)
        
        return {
            "session_risk_score": round(risk_score, 4),
            "fraud_indicators": indicators,
            "confidence": 0.7,
            "features": {
                "session_duration": session_duration,
                "page_transitions": page_transitions,
                "paste_events": paste_count
            }
        }


class BehavioralBiometricsEngine:
    """Main behavioral biometrics engine."""
    
    def __init__(self):
        self.keystroke_analyzer = KeystrokeAnalyzer()
        self.mouse_analyzer = MouseMovementAnalyzer()
        self.device_fingerprinter = DeviceFingerprinter()
        self.session_analyzer = SessionAnalyzer()
    
    def analyze_behavior(
        self,
        biometric_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze behavioral biometrics and return combined risk score."""
        
        user_id = biometric_data.get("user_id", "unknown")
        
        keystroke_result = {"biometric_risk_score": 0.0, "fraud_indicators": [], "confidence": 0.0}
        mouse_result = {"biometric_risk_score": 0.0, "fraud_indicators": [], "confidence": 0.0}
        device_result = {"device_risk_score": 0.0, "fraud_indicators": [], "confidence": 0.0}
        session_result = {"session_risk_score": 0.0, "fraud_indicators": [], "confidence": 0.0}
        
        # Analyze keystroke
        if "keystroke" in biometric_data:
            keystroke_result = self.keystroke_analyzer.analyze_keystroke(biometric_data["keystroke"])
        
        # Analyze mouse
        if "mouse" in biometric_data:
            mouse_result = self.mouse_analyzer.analyze_mouse_movement(biometric_data["mouse"])
        
        # Analyze device
        if "device" in biometric_data:
            device_result = self.device_fingerprinter.analyze_device(biometric_data["device"], user_id)
        
        # Analyze session
        if "session" in biometric_data:
            session_result = self.session_analyzer.analyze_session(biometric_data["session"])
        
        # Combine scores (weighted average)
        weights = {
            "keystroke": 0.25,
            "mouse": 0.2,
            "device": 0.35,
            "session": 0.2
        }
        
        combined_score = (
            keystroke_result["biometric_risk_score"] * weights["keystroke"] +
            mouse_result["biometric_risk_score"] * weights["mouse"] +
            device_result["device_risk_score"] * weights["device"] +
            session_result["session_risk_score"] * weights["session"]
        )
        
        # Combine indicators
        all_indicators = (
            keystroke_result["fraud_indicators"] +
            mouse_result["fraud_indicators"] +
            device_result["fraud_indicators"] +
            session_result["fraud_indicators"]
        )
        
        # Average confidence
        confidences = [
            keystroke_result["confidence"],
            mouse_result["confidence"],
            device_result["confidence"],
            session_result["confidence"]
        ]
        avg_confidence = statistics.mean([c for c in confidences if c > 0]) if any(c > 0 for c in confidences) else 0.0
        
        return {
            "behavioral_risk_score": round(combined_score, 4),
            "fraud_indicators": all_indicators,
            "confidence": round(avg_confidence, 2),
            "component_scores": {
                "keystroke": keystroke_result.get("biometric_risk_score", 0),
                "mouse": mouse_result.get("biometric_risk_score", 0),
                "device": device_result.get("device_risk_score", 0),
                "session": session_result.get("session_risk_score", 0)
            },
            "recommendation": "block" if combined_score > 0.6 else "review" if combined_score > 0.3 else "allow"
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get behavioral biometrics configuration."""
        return {
            "keystroke_weights": self.keystroke_analyzer.KEYSTROKE_WEIGHTS,
            "mouse_weights": self.mouse_analyzer.MOUSE_WEIGHTS,
            "fingerprint_weights": self.device_fingerprinter.FINGERPRINT_WEIGHTS,
            "session_weights": self.session_analyzer.SESSION_WEIGHTS,
            "thresholds": {
                "block": 0.6,
                "review": 0.3
            }
        }


_global_engine: Optional[BehavioralBiometricsEngine] = None


def get_biometrics_engine() -> BehavioralBiometricsEngine:
    """Get global behavioral biometrics engine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = BehavioralBiometricsEngine()
    return _global_engine


def analyze_biometric_data(biometric_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to analyze biometric data."""
    return get_biometrics_engine().analyze_behavior(biometric_data)