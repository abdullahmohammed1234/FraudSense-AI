"""
RiskPlatform Configuration Module.

Contains all configurable thresholds, weights, and settings for the
enterprise fraud detection platform.
"""

from typing import Dict, Any
import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base directory for the platform
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Logs directory
LOGS_DIR = os.path.join(os.path.dirname(BASE_DIR), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Transaction logs path
TRANSACTION_LOGS_PATH = os.path.join(LOGS_DIR, "transactions_log.json")

# Model metadata path
MODEL_METADATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "model_metadata.json")


# ============================================================================
# DECISION ENGINE CONFIGURATION
# ============================================================================

class DecisionConfig:
    """Configuration for the Decision Recommendation Engine."""
    
    # Threshold for auto-blocking (fraud_probability)
    BLOCK_THRESHOLD: float = 0.75
    
    # Threshold for manual review
    REVIEW_THRESHOLD: float = 0.40
    
    # Anomaly score threshold for high risk
    HIGH_ANOMALY_THRESHOLD: float = 0.70
    
    # Moderate anomaly threshold
    MODERATE_ANOMALY_THRESHOLD: float = 0.40
    
    # Risk level thresholds
    LOW_RISK_THRESHOLD: float = 0.20
    MEDIUM_RISK_THRESHOLD: float = 0.50
    HIGH_RISK_THRESHOLD: float = 0.75
    CRITICAL_RISK_THRESHOLD: float = 0.90


# ============================================================================
# ENSEMBLE RISK SCORING CONFIGURATION
# ============================================================================

class RiskScoringConfig:
    """Configuration for Ensemble Risk Scoring."""
    
    # Weights for the ensemble scoring (must sum to 1.0)
    FRAUD_PROBABILITY_WEIGHT: float = 0.50
    ANOMALY_SCORE_WEIGHT: float = 0.35
    DRIFT_SIGNAL_WEIGHT: float = 0.15
    
    # Risk band thresholds
    RISK_BANDS: Dict[str, tuple] = {
        "Low": (0.0, 0.25),
        "Medium": (0.25, 0.50),
        "High": (0.50, 0.75),
        "Critical": (0.75, 1.0)
    }
    
    # Drift signal weight multiplier (when drift is detected)
    DRIFT_PENALTY: float = 1.5


# ============================================================================
# MODEL GOVERNANCE CONFIGURATION
# ============================================================================

class GovernanceConfig:
    """Configuration for Model Governance."""
    
    # Number of logs to return in audit log endpoint
    AUDIT_LOG_LIMIT: int = 50
    
    # Auto-create log file if missing
    AUTO_CREATE_LOGS: bool = True
    
    # Model version
    MODEL_VERSION: str = "1.0.0"
    
    # Default threshold
    DEFAULT_THRESHOLD: float = 0.5


# ============================================================================
# STRESS TEST CONFIGURATION
# ============================================================================

class StressTestConfig:
    """Configuration for Stress Testing."""
    
    # Number of transactions to simulate
    DEFAULT_TEST_SIZE: int = 1000
    
    # Timeout for each prediction (seconds)
    PREDICTION_TIMEOUT: float = 5.0


# ============================================================================
# DRIFT MONITOR CONFIGURATION
# ============================================================================

class DriftConfig:
    """Configuration for Drift Monitoring."""
    
    # Z-score threshold for drift detection
    DRIFT_THRESHOLD: float = 2.5
    
    # Minimum samples before checking drift
    MIN_SAMPLES: int = 100
    
    # Rolling window size for drift tracking
    WINDOW_SIZE: int = 1000


# ============================================================================
# ANALYTICS CONFIGURATION
# ============================================================================

class AnalyticsConfig:
    """Configuration for Analytics Tracking."""
    
    # Maximum rolling probabilities to track
    MAX_ROLLING: int = 50
    
    # High risk probability threshold
    HIGH_RISK_THRESHOLD: float = 0.6
    
    # Feature importance update frequency
    FEATURE_UPDATE_FREQ: int = 100
