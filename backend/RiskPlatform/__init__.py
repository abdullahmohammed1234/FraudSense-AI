"""
RiskPlatform - Enterprise Fraud Detection Platform.

A modular, production-grade fraud detection system with:
- Audit logging and governance
- Model versioning and metadata management
- Decision recommendation engine
- Ensemble risk scoring
- Drift monitoring
- Real-time analytics

Modules:
- config: Configuration and settings
- audit_logger: Audit logging for compliance
- model_manager: Model versioning and metadata
- decision_engine: Decision recommendation
- risk_scorer: Ensemble risk scoring
- drift_monitor: Concept drift detection
- analytics_tracker: Real-time analytics
"""

# Configuration
from .config import (
    DecisionConfig,
    RiskScoringConfig,
    GovernanceConfig,
    StressTestConfig,
    DriftConfig,
    AnalyticsConfig,
    LOGS_DIR,
    TRANSACTION_LOGS_PATH,
    MODEL_METADATA_PATH
)

# Core modules
from .audit_logger import (
    AuditLogger,
    get_audit_logger,
    generate_transaction_id
)

from .model_manager import (
    ModelMetadata,
    ModelManager,
    get_model_manager,
    get_model_info
)

from .decision_engine import (
    DecisionEngine,
    ActionRecommendation,
    RiskLevel,
    get_decision_engine,
    get_recommendation
)

from .risk_scorer import (
    RiskScorer,
    get_risk_scorer,
    calculate_risk_score
)

from .drift_monitor import (
    DriftMonitor,
    get_drift_monitor
)

from .analytics_tracker import (
    AnalyticsTracker,
    get_analytics_tracker
)

from .metrics_tracker import (
    MetricsTracker,
    get_metrics_tracker,
    reset_metrics
)

from .model_health_monitor import (
    ModelHealthMonitor,
    HealthStatus,
    get_model_health_monitor,
    reset_model_health
)

from .explainability_engine import (
    ExplainabilityEngine,
    get_explainability_engine
)

from .threshold_simulator import (
    ThresholdSimulator,
    get_threshold_simulator
)

from .auth_middleware import (
    AuthMiddleware,
    APIKeyManager,
    RateLimiter,
    UserRole,
    Permission,
    get_api_key_manager,
    get_rate_limiter,
    get_auth_middleware,
    check_permission
)

from .logging_manager import (
    LoggingManager,
    get_logging_manager,
    log_request,
    log_prediction,
    log_error
)

# Version
__version__ = "2.0.0"

# All exported symbols
__all__ = [
    # Config
    "DecisionConfig",
    "RiskScoringConfig", 
    "GovernanceConfig",
    "StressTestConfig",
    "DriftConfig",
    "AnalyticsConfig",
    "LOGS_DIR",
    "TRANSACTION_LOGS_PATH",
    "MODEL_METADATA_PATH",
    
    # Audit
    "AuditLogger",
    "get_audit_logger",
    "generate_transaction_id",
    
    # Model
    "ModelMetadata",
    "ModelManager",
    "get_model_manager",
    "get_model_info",
    
    # Decision
    "DecisionEngine",
    "ActionRecommendation",
    "RiskLevel",
    "get_decision_engine",
    "get_recommendation",
    
    # Risk Scoring
    "RiskScorer",
    "get_risk_scorer",
    "calculate_risk_score",
    
    # Drift
    "DriftMonitor",
    "get_drift_monitor",
    
    # Analytics
    "AnalyticsTracker",
    "get_analytics_tracker",
    
    # Metrics
    "MetricsTracker",
    "get_metrics_tracker",
    "reset_metrics",
    
    # Model Health
    "ModelHealthMonitor",
    "HealthStatus",
    "get_model_health_monitor",
    "reset_model_health",
    
    # Explainability
    "ExplainabilityEngine",
    "get_explainability_engine",
    
    # Threshold Simulation
    "ThresholdSimulator",
    "get_threshold_simulator",
    
    # Auth
    "AuthMiddleware",
    "APIKeyManager",
    "RateLimiter",
    "UserRole",
    "Permission",
    "get_api_key_manager",
    "get_rate_limiter",
    "get_auth_middleware",
    "check_permission",
    
    # Logging
    "LoggingManager",
    "get_logging_manager",
    "log_request",
    "log_prediction",
    "log_error",
]
