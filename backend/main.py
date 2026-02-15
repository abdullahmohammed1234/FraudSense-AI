"""
FastAPI application for FraudSense AI.

This module provides a REST API for real-time fraud detection predictions
with enterprise-grade features including audit logging, model versioning,
decision recommendations, and ensemble risk scoring.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import sys
import pandas as pd
import numpy as np
import random
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import existing model
from model import load_model, predict_fraud, get_analytics, get_risk_engine

# Import RiskPlatform modules
from RiskPlatform import (
    get_audit_logger,
    get_model_manager,
    get_model_info,
    get_decision_engine,
    get_risk_scorer,
    get_drift_monitor,
    get_analytics_tracker,
    generate_transaction_id,
    GovernanceConfig,
    DecisionConfig,
    RiskScoringConfig,
    StressTestConfig,
    get_metrics_tracker,
    get_model_health_monitor,
    get_explainability_engine,
    get_threshold_simulator,
    get_auth_middleware,
    get_logging_manager,
    UserRole,
    Permission
)

# Initialize FastAPI app
app = FastAPI(
    title="FraudSense AI API",
    description="Enterprise-grade fraud detection system with ML risk scoring",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Feature input model
class TransactionFeatures(BaseModel):
    """Model for transaction features input."""
    
    # PCA features V1-V28
    V1: Optional[float] = 0.0
    V2: Optional[float] = 0.0
    V3: Optional[float] = 0.0
    V4: Optional[float] = 0.0
    V5: Optional[float] = 0.0
    V6: Optional[float] = 0.0
    V7: Optional[float] = 0.0
    V8: Optional[float] = 0.0
    V9: Optional[float] = 0.0
    V10: Optional[float] = 0.0
    V11: Optional[float] = 0.0
    V12: Optional[float] = 0.0
    V13: Optional[float] = 0.0
    V14: Optional[float] = 0.0
    V15: Optional[float] = 0.0
    V16: Optional[float] = 0.0
    V17: Optional[float] = 0.0
    V18: Optional[float] = 0.0
    V19: Optional[float] = 0.0
    V20: Optional[float] = 0.0
    V21: Optional[float] = 0.0
    V22: Optional[float] = 0.0
    V23: Optional[float] = 0.0
    V24: Optional[float] = 0.0
    V25: Optional[float] = 0.0
    V26: Optional[float] = 0.0
    V27: Optional[float] = 0.0
    V28: Optional[float] = 0.0
    
    # Time and Amount
    Time: Optional[float] = 0.0
    Amount: Optional[float] = 0.0
    
    class Config:
        schema_extra = {
            "example": {
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536346,
                "V4": 1.378155,
                "V5": -0.338321,
                "Amount": 149.62,
                "Time": 406.0
            }
        }


# Enhanced Prediction Response
class PredictionResponse(BaseModel):
    """Enhanced model for prediction response with enterprise features."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    anomaly_score: float = Field(..., description="Anomaly score (0-1)")
    final_risk_score: float = Field(..., description="Ensemble risk score (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High, Critical")
    risk_band: str = Field(..., description="Risk band from ensemble scoring")
    threshold_used: float = Field(..., description="Threshold used for classification")
    model_version: str = Field(..., description="Model version used")
    drift_detected: bool = Field(..., description="Whether concept drift was detected")
    action_recommendation: str = Field(..., description="Recommended action")
    action_reasoning: str = Field(..., description="Explanation for action")
    action_confidence: str = Field(..., description="Confidence in recommendation")
    top_factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top 3 contributing features"
    )
    explanation_summary: str = Field(..., description="Human-readable explanation")


class SimulationResponse(PredictionResponse):
    """Model for simulation response."""
    
    is_fraud: Optional[bool] = Field(None, description="Actual fraud label if available")


# Audit Log Response
class AuditLogResponse(BaseModel):
    """Model for audit log response."""
    
    logs: List[Dict[str, Any]] = Field(..., description="List of audit log entries")
    total_count: int = Field(..., description="Total number of logs")
    limit: int = Field(..., description="Limit applied")


class AuditStatsResponse(BaseModel):
    """Model for audit statistics response."""
    
    total_logs: int = Field(..., description="Total number of log entries")
    by_risk_level: Dict[str, int] = Field(..., description="Logs grouped by risk level")
    by_action: Dict[str, int] = Field(..., description="Logs grouped by action")
    average_fraud_probability: float = Field(..., description="Average fraud probability")
    drift_detected_count: int = Field(..., description="Number of logs with drift detected")


# Model Info Response
class ModelInfoResponse(BaseModel):
    """Model for model info response."""
    
    model_type: str = Field(..., description="Type of ML model")
    training_date: str = Field(..., description="Date model was trained")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    threshold: float = Field(..., description="Classification threshold")
    dataset_size: int = Field(..., description="Training dataset size")
    feature_count: int = Field(..., description="Number of features")
    version: str = Field(..., description="Model version")
    health_status: str = Field(..., description="Model health status")


# Stress Test Response
class StressTestResponse(BaseModel):
    """Model for stress test response."""
    
    total_processed: int = Field(..., description="Total transactions processed")
    fraud_rate: float = Field(..., description="Fraud detection rate")
    average_inference_time_ms: float = Field(..., description="Average inference time")
    max_latency_ms: float = Field(..., description="Maximum latency observed")
    min_latency_ms: float = Field(..., description="Minimum latency observed")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    transactions_per_second: float = Field(..., description="Throughput")
    risk_distribution: Dict[str, int] = Field(..., description="Distribution of risk levels")
    action_distribution: Dict[str, int] = Field(..., description="Distribution of actions")


class MetricsResponse(BaseModel):
    """Model for metrics response."""
    
    total_transactions: int = Field(..., description="Total transactions processed")
    total_fraud_detected: int = Field(..., description="Total fraud detected")
    auto_block_count: int = Field(..., description="Auto-blocked transactions")
    manual_review_count: int = Field(..., description="Manual review count")
    approval_count: int = Field(..., description="Approved transactions")
    drift_events: int = Field(..., description="Drift events detected")
    average_inference_time_ms: float = Field(..., description="Average inference time")
    p95_latency: float = Field(..., description="P95 latency")
    fraud_rate: float = Field(..., description="Fraud rate percentage")
    risk_band_distribution: Dict[str, int] = Field(..., description="Risk band distribution")
    auto_block_percentage: float = Field(..., description="Auto-block percentage")
    manual_review_percentage: float = Field(..., description="Manual review percentage")
    approval_percentage: float = Field(..., description="Approval percentage")


class ModelHealthResponse(BaseModel):
    """Model for model health response."""
    
    health_status: str = Field(..., description="Health status: Healthy, Warning, or Degraded")
    confidence_score: float = Field(..., description="Health confidence score")
    drift_trend: float = Field(..., description="Drift frequency trend")
    performance_trend: float = Field(..., description="Performance trend")
    stability_index: float = Field(..., description="Detection stability score")


class ThresholdSimulationRequest(BaseModel):
    """Model for threshold simulation request."""
    
    threshold: float = Field(..., description="Classification threshold (0-1)", ge=0.0, le=1.0)
    review_threshold: Optional[float] = Field(None, description="Review threshold (0-1)")


class ThresholdSimulationResponse(BaseModel):
    """Model for threshold simulation response."""
    
    threshold: float = Field(..., description="Threshold used")
    sample_size: int = Field(..., description="Sample size")
    projected_fraud_detection_rate: float = Field(..., description="Projected fraud detection rate")
    projected_false_positive_rate: float = Field(..., description="Projected false positive rate")
    projected_decision_distribution: Dict[str, Any] = Field(..., description="Decision distribution")
    projected_auto_block_rate: float = Field(..., description="Projected auto-block rate")


class RiskTrendsResponse(BaseModel):
    """Model for risk trends response."""
    
    time_series_data: List[Dict[str, Any]] = Field(..., description="Time series risk data")
    risk_distribution: Dict[str, int] = Field(..., description="Current risk distribution")
    fraud_trend: float = Field(..., description="Fraud detection trend")
    drift_alerts: List[Dict[str, Any]] = Field(..., description="Recent drift alerts")


class DecisionDistributionResponse(BaseModel):
    """Model for decision distribution response."""
    
    distribution: Dict[str, int] = Field(..., description="Decision counts")
    percentages: Dict[str, float] = Field(..., description="Decision percentages")
    auto_block_rate: float = Field(..., description="Auto-block rate")
    manual_review_rate: float = Field(..., description="Manual review rate")
    approval_rate: float = Field(..., description="Approval rate")


class LatencyStatsResponse(BaseModel):
    """Model for latency stats response."""
    
    average_ms: float = Field(..., description="Average latency")
    min_ms: float = Field(..., description="Minimum latency")
    max_ms: float = Field(..., description="Maximum latency")
    p50_ms: float = Field(..., description="P50 latency")
    p95_ms: float = Field(..., description="P95 latency")
    p99_ms: float = Field(..., description="P99 latency")
    histogram: List[Dict[str, int]] = Field(..., description="Latency histogram")


class ExplanationResponse(BaseModel):
    """Model for explanation response."""
    
    transaction_id: str = Field(..., description="Transaction ID")
    top_features: List[Dict[str, Any]] = Field(..., description="Top contributing features")
    feature_importance_scores: Dict[str, float] = Field(..., description="Normalized importance scores")
    explanation_summary: str = Field(..., description="Risk factor summary")
    confidence_level: float = Field(..., description="Confidence level")


class SystemLogsResponse(BaseModel):
    """Model for system logs response."""
    
    logs: List[Dict[str, Any]] = Field(..., description="Log entries")
    total_count: int = Field(..., description="Total logs")
    stats: Dict[str, Any] = Field(..., description="Log statistics")


# Cache for dataset
_dataset_cache: Optional[pd.DataFrame] = None


def get_dataset() -> pd.DataFrame:
    """Load and cache the dataset for simulation."""
    global _dataset_cache
    if _dataset_cache is None:
        dataset_path = "../creditcard.csv"
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
        _dataset_cache = pd.read_csv(dataset_path)
    return _dataset_cache


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    print("Loading fraud detection model...")
    try:
        load_model()
        
        # Initialize RiskPlatform components
        print("Initializing RiskPlatform components...")
        get_audit_logger()
        get_model_manager()
        get_decision_engine()
        get_risk_scorer()
        get_drift_monitor()
        get_analytics_tracker()
        get_metrics_tracker()
        get_model_health_monitor()
        get_explainability_engine()
        get_threshold_simulator()
        get_logging_manager()
        
        print("RiskPlatform initialized successfully!")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run train.py first to train and save the model.")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "FraudSense AI API",
        "version": "2.0.0",
        "description": "Enterprise-grade fraud detection system",
        "endpoints": {
            "predict": "/predict (POST)",
            "simulate": "/simulate (GET)",
            "analytics": "/analytics (GET)",
            "audit_log": "/audit-log (GET)",
            "model_info": "/model-info (GET)",
            "stress_test": "/stress-test (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Get system health
    drift_monitor = get_drift_monitor()
    drift_status = drift_monitor.get_status()
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "drift_status": drift_status.get("status", "unknown"),
        "version": "2.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionFeatures, transaction_id: Optional[str] = None):
    """
    Predict fraud for a transaction with enterprise features.
    
    Includes:
    - Fraud probability
    - Anomaly score
    - Ensemble risk scoring
    - Decision recommendation
    - Audit logging
    """
    start_time = time.time()
    
    try:
        # Generate transaction ID if not provided
        if transaction_id is None:
            transaction_id = generate_transaction_id()
        
        # Convert Pydantic model to dictionary
        features = transaction.dict()
        
        # Make prediction using existing model
        result = predict_fraud(features)
        
        # Get RiskPlatform components
        decision_engine = get_decision_engine()
        risk_scorer = get_risk_scorer()
        audit_logger = get_audit_logger()
        analytics_tracker = get_analytics_tracker()
        metrics_tracker = get_metrics_tracker()
        model_manager = get_model_manager()
        
        # Get model info
        model_info = model_manager.get_model_info()
        model_version = model_info.get("version", GovernanceConfig.MODEL_VERSION)
        
        # Get thresholds
        threshold = model_info.get("threshold", DecisionConfig.BLOCK_THRESHOLD)
        
        # Calculate ensemble risk score
        final_risk_score, risk_band = risk_scorer.calculate_risk_score(
            result["fraud_probability"],
            result.get("anomaly_score", 0.0),
            result.get("drift_detected", False)
        )
        
        # Get decision recommendation
        recommendation = decision_engine.get_recommendation(
            result["fraud_probability"],
            result.get("anomaly_score", 0.0),
            result.get("drift_detected", False)
        )
        
        # Determine risk level from ensemble score
        risk_level = decision_engine.get_risk_level(
            result["fraud_probability"],
            result.get("anomaly_score", 0.0),
            final_risk_score
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log to audit
        audit_logger.log_prediction(
            transaction_id=transaction_id,
            fraud_probability=result["fraud_probability"],
            anomaly_score=result.get("anomaly_score", 0.0),
            drift_detected=result.get("drift_detected", False),
            risk_level=risk_level,
            threshold_used=threshold,
            model_version=model_version,
            top_risk_factors=result.get("top_factors", []),
            action_recommendation=recommendation["action_recommendation"],
            transaction_data={"Amount": features.get("Amount", 0.0), "Time": features.get("Time", 0.0)}
        )
        
        # Track analytics
        analytics_tracker.record_transaction(
            fraud_probability=result["fraud_probability"],
            anomaly_score=result.get("anomaly_score", 0.0),
            risk_score=final_risk_score,
            risk_level=risk_level,
            action=recommendation["action_recommendation"],
            top_factors=result.get("top_factors", []),
            latency_ms=latency_ms
        )
        
        # Track metrics (new)
        metrics_tracker.record_prediction(
            fraud_probability=result["fraud_probability"],
            anomaly_score=result.get("anomaly_score", 0.0),
            risk_score=final_risk_score,
            risk_band=risk_band,
            action=recommendation["action_recommendation"],
            latency_ms=latency_ms,
            drift_detected=result.get("drift_detected", False)
        )
        
        # Track model health
        health_monitor = get_model_health_monitor()
        health_monitor.record_prediction(
            fraud_probability=result["fraud_probability"],
            anomaly_score=result.get("anomaly_score", 0.0),
            risk_band=risk_band,
            drift_detected=result.get("drift_detected", False)
        )
        
        # Track for threshold simulation
        threshold_simulator = get_threshold_simulator()
        threshold_simulator.record_prediction(
            fraud_probability=result["fraud_probability"],
            anomaly_score=result.get("anomaly_score", 0.0),
            risk_score=final_risk_score,
            transaction_id=transaction_id
        )
        
        # Log structured request
        logging_manager = get_logging_manager()
        logging_manager.log_prediction(
            transaction_id=transaction_id,
            fraud_probability=result["fraud_probability"],
            risk_level=risk_level,
            decision=recommendation["action_recommendation"],
            latency_ms=latency_ms,
            status_code=200
        )
        
        return PredictionResponse(
            transaction_id=transaction_id,
            fraud_probability=result["fraud_probability"],
            anomaly_score=result.get("anomaly_score", 0.0),
            final_risk_score=round(final_risk_score, 4),
            risk_level=risk_level,
            risk_band=risk_band,
            threshold_used=threshold,
            model_version=model_version,
            drift_detected=result.get("drift_detected", False),
            action_recommendation=recommendation["action_recommendation"],
            action_reasoning=recommendation["reasoning"],
            action_confidence=recommendation["confidence"],
            top_factors=result.get("top_factors", []),
            explanation_summary=result.get("explanation_summary", "")
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please run train.py first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/simulate", response_model=SimulationResponse)
async def simulate():
    """
    Simulate a live transaction with enterprise features.
    """
    start_time = time.time()
    
    try:
        # Generate transaction ID
        transaction_id = generate_transaction_id()
        
        # Load dataset
        df = get_dataset()
        
        # Randomly sample one row
        random.seed()
        sample_idx = random.randint(0, len(df) - 1)
        row = df.iloc[sample_idx]
        
        # Prepare features
        features = {
            "V1": float(row["V1"]),
            "V2": float(row["V2"]),
            "V3": float(row["V3"]),
            "V4": float(row["V4"]),
            "V5": float(row["V5"]),
            "V6": float(row["V6"]),
            "V7": float(row["V7"]),
            "V8": float(row["V8"]),
            "V9": float(row["V9"]),
            "V10": float(row["V10"]),
            "V11": float(row["V11"]),
            "V12": float(row["V12"]),
            "V13": float(row["V13"]),
            "V14": float(row["V14"]),
            "V15": float(row["V15"]),
            "V16": float(row["V16"]),
            "V17": float(row["V17"]),
            "V18": float(row["V18"]),
            "V19": float(row["V19"]),
            "V20": float(row["V20"]),
            "V21": float(row["V21"]),
            "V22": float(row["V22"]),
            "V23": float(row["V23"]),
            "V24": float(row["V24"]),
            "V25": float(row["V25"]),
            "V26": float(row["V26"]),
            "V27": float(row["V27"]),
            "V28": float(row["V28"]),
            "Amount": float(row["Amount"]),
            "Time": float(row["Time"])
        }
        
        # Make prediction
        result = predict_fraud(features)
        
        # Get RiskPlatform components
        decision_engine = get_decision_engine()
        risk_scorer = get_risk_scorer()
        audit_logger = get_audit_logger()
        analytics_tracker = get_analytics_tracker()
        model_manager = get_model_manager()
        
        # Get model info
        model_info = model_manager.get_model_info()
        model_version = model_info.get("version", GovernanceConfig.MODEL_VERSION)
        threshold = model_info.get("threshold", DecisionConfig.BLOCK_THRESHOLD)
        
        # Calculate ensemble risk score
        final_risk_score, risk_band = risk_scorer.calculate_risk_score(
            result["fraud_probability"],
            result.get("anomaly_score", 0.0),
            result.get("drift_detected", False)
        )
        
        # Get decision recommendation
        recommendation = decision_engine.get_recommendation(
            result["fraud_probability"],
            result.get("anomaly_score", 0.0),
            result.get("drift_detected", False)
        )
        
        # Determine risk level
        risk_level = decision_engine.get_risk_level(
            result["fraud_probability"],
            result.get("anomaly_score", 0.0),
            final_risk_score
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log to audit
        audit_logger.log_prediction(
            transaction_id=transaction_id,
            fraud_probability=result["fraud_probability"],
            anomaly_score=result.get("anomaly_score", 0.0),
            drift_detected=result.get("drift_detected", False),
            risk_level=risk_level,
            threshold_used=threshold,
            model_version=model_version,
            top_risk_factors=result.get("top_factors", []),
            action_recommendation=recommendation["action_recommendation"],
            transaction_data={"Amount": features.get("Amount", 0.0), "Time": features.get("Time", 0.0)}
        )
        
        # Track analytics
        analytics_tracker.record_transaction(
            fraud_probability=result["fraud_probability"],
            anomaly_score=result.get("anomaly_score", 0.0),
            risk_score=final_risk_score,
            risk_level=risk_level,
            action=recommendation["action_recommendation"],
            top_factors=result.get("top_factors", []),
            latency_ms=latency_ms
        )
        
        # Get actual label
        actual_fraud = bool(row["Class"]) if "Class" in row else None
        
        return SimulationResponse(
            transaction_id=transaction_id,
            fraud_probability=result["fraud_probability"],
            anomaly_score=result.get("anomaly_score", 0.0),
            final_risk_score=round(final_risk_score, 4),
            risk_level=risk_level,
            risk_band=risk_band,
            threshold_used=threshold,
            model_version=model_version,
            drift_detected=result.get("drift_detected", False),
            action_recommendation=recommendation["action_recommendation"],
            action_reasoning=recommendation["reasoning"],
            action_confidence=recommendation["confidence"],
            top_factors=result.get("top_factors", []),
            explanation_summary=result.get("explanation_summary", ""),
            is_fraud=actual_fraud
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Model or dataset not found. Please run train.py first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation error: {str(e)}"
        )


@app.get("/analytics")
async def get_analytics_endpoint():
    """Get fraud detection analytics."""
    return get_analytics()


# ============================================================================
# NEW ENTERPRISE ENDPOINTS
# ============================================================================

@app.get("/audit-log", response_model=AuditLogResponse)
async def get_audit_log(limit: int = GovernanceConfig.AUDIT_LOG_LIMIT):
    """
    Get audit log entries.
    
    Returns last N log entries sorted newest first.
    """
    audit_logger = get_audit_logger()
    logs = audit_logger.get_logs(limit=limit)
    
    return AuditLogResponse(
        logs=logs,
        total_count=len(logs),
        limit=limit
    )


@app.get("/audit-log/stats", response_model=AuditStatsResponse)
async def get_audit_stats():
    """Get audit log statistics."""
    audit_logger = get_audit_logger()
    stats = audit_logger.get_statistics()
    
    return AuditStatsResponse(
        total_logs=stats["total_logs"],
        by_risk_level=stats["by_risk_level"],
        by_action=stats["by_action"],
        average_fraud_probability=stats["average_fraud_probability"],
        drift_detected_count=stats["drift_detected_count"]
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info_endpoint():
    """
    Get model metadata and information.
    """
    model_info = get_model_info()
    
    return ModelInfoResponse(
        model_type=model_info["model_type"],
        training_date=model_info["training_date"],
        metrics=model_info["metrics"],
        threshold=model_info["threshold"],
        dataset_size=model_info["dataset_size"],
        feature_count=model_info["feature_count"],
        version=model_info["version"],
        health_status=model_info["health_status"]
    )


@app.get("/drift-status")
async def get_drift_status():
    """Get current drift monitoring status."""
    drift_monitor = get_drift_monitor()
    return drift_monitor.get_status()


@app.get("/risk-config")
async def get_risk_config():
    """Get current risk scoring configuration."""
    risk_scorer = get_risk_scorer()
    decision_engine = get_decision_engine()
    
    return {
        "risk_scorer": risk_scorer.get_config(),
        "decision_engine": decision_engine.get_config()
    }


@app.get("/analytics/realtime")
async def get_realtime_analytics():
    """Get real-time analytics from tracker."""
    analytics_tracker = get_analytics_tracker()
    
    return {
        "summary": analytics_tracker.get_summary(),
        "recent": analytics_tracker.get_recent_analytics(limit=20),
        "latency": analytics_tracker.get_latency_stats()
    }


@app.get("/stress-test", response_model=StressTestResponse)
async def run_stress_test(count: int = StressTestConfig.DEFAULT_TEST_SIZE):
    """
    Run stress test by simulating N transactions.
    
    Returns performance metrics including:
    - Total processed
    - Fraud rate
    - Average inference time
    - Max latency
    - Throughput
    """
    try:
        # Load dataset
        df = get_dataset()
        
        # Initialize tracking
        latencies = []
        fraud_count = 0
        risk_levels = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
        actions = {"Auto-Block Transaction": 0, "Manual Review Required": 0, "Approve": 0}
        
        # Get components
        decision_engine = get_decision_engine()
        risk_scorer = get_risk_scorer()
        model_manager = get_model_manager()
        model_info = model_manager.get_model_info()
        threshold = model_info.get("threshold", DecisionConfig.BLOCK_THRESHOLD)
        
        # Process transactions
        start_time = time.time()
        
        for i in range(count):
            # Sample random transaction
            sample_idx = random.randint(0, len(df) - 1)
            row = df.iloc[sample_idx]
            
            # Prepare features
            features = {
                "V1": float(row["V1"]), "V2": float(row["V2"]), "V3": float(row["V3"]),
                "V4": float(row["V4"]), "V5": float(row["V5"]), "V6": float(row["V6"]),
                "V7": float(row["V7"]), "V8": float(row["V8"]), "V9": float(row["V9"]),
                "V10": float(row["V10"]), "V11": float(row["V11"]), "V12": float(row["V12"]),
                "V13": float(row["V13"]), "V14": float(row["V14"]), "V15": float(row["V15"]),
                "V16": float(row["V16"]), "V17": float(row["V17"]), "V18": float(row["V18"]),
                "V19": float(row["V19"]), "V20": float(row["V20"]), "V21": float(row["V21"]),
                "V22": float(row["V22"]), "V23": float(row["V23"]), "V24": float(row["V24"]),
                "V25": float(row["V25"]), "V26": float(row["V26"]), "V27": float(row["V27"]),
                "V28": float(row["V28"]), "Amount": float(row["Amount"]), "Time": float(row["Time"])
            }
            
            # Time the prediction
            pred_start = time.time()
            result = predict_fraud(features)
            pred_time = (time.time() - pred_start) * 1000
            
            latencies.append(pred_time)
            
            # Calculate risk score
            final_score, risk_band = risk_scorer.calculate_risk_score(
                result["fraud_probability"],
                result.get("anomaly_score", 0.0),
                result.get("drift_detected", False)
            )
            
            # Get recommendation
            recommendation = decision_engine.get_recommendation(
                result["fraud_probability"],
                result.get("anomaly_score", 0.0),
                result.get("drift_detected", False)
            )
            
            # Get risk level
            risk_level = decision_engine.get_risk_level(
                result["fraud_probability"],
                result.get("anomaly_score", 0.0),
                final_score
            )
            
            # Track stats
            if result["fraud_probability"] >= 0.5:
                fraud_count += 1
            
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
            actions[recommendation["action_recommendation"]] = actions.get(recommendation["action_recommendation"], 0) + 1
        
        # Calculate metrics
        total_time = time.time() - start_time
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        
        # Calculate p95
        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0
        
        # Calculate throughput
        tps = count / total_time if total_time > 0 else 0
        
        return StressTestResponse(
            total_processed=count,
            fraud_rate=round(fraud_count / count * 100, 4) if count > 0 else 0,
            average_inference_time_ms=round(avg_latency, 2),
            max_latency_ms=round(max_latency, 2),
            min_latency_ms=round(min_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
            transactions_per_second=round(tps, 2),
            risk_distribution=risk_levels,
            action_distribution=actions
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Stress test error: {str(e)}"
        )


# ============================================================================
# ENTERPRISE METRICS & HEALTH ENDPOINTS
# ============================================================================

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get real-time metrics from the metrics tracker.
    
    Returns aggregated statistics including:
    - total_transactions
    - total_fraud_detected
    - auto_block_count
    - manual_review_count
    - approval_count
    - drift_events
    - average_inference_time_ms
    - p95_latency
    - fraud_rate
    - risk_band_distribution
    """
    metrics_tracker = get_metrics_tracker()
    metrics = metrics_tracker.get_metrics()
    
    return MetricsResponse(
        total_transactions=metrics["total_transactions"],
        total_fraud_detected=metrics["total_fraud_detected"],
        auto_block_count=metrics["auto_block_count"],
        manual_review_count=metrics["manual_review_count"],
        approval_count=metrics["approval_count"],
        drift_events=metrics["drift_events"],
        average_inference_time_ms=metrics["average_inference_time_ms"],
        p95_latency=metrics["p95_latency"],
        fraud_rate=metrics["fraud_rate"],
        risk_band_distribution=metrics["risk_band_distribution"],
        auto_block_percentage=metrics["auto_block_percentage"],
        manual_review_percentage=metrics["manual_review_percentage"],
        approval_percentage=metrics["approval_percentage"]
    )


@app.get("/metrics/reset")
async def reset_metrics():
    """Reset all metrics (admin only)."""
    metrics_tracker = get_metrics_tracker()
    metrics_tracker.reset()
    return {"status": "success", "message": "Metrics reset successfully"}


@app.get("/metrics/snapshot")
async def get_metrics_snapshot():
    """Get a lightweight snapshot of current metrics."""
    metrics_tracker = get_metrics_tracker()
    return metrics_tracker.get_snapshot()


@app.get("/metrics/rolling")
async def get_rolling_statistics():
    """Get rolling window statistics."""
    metrics_tracker = get_metrics_tracker()
    return metrics_tracker.get_rolling_statistics()


# ============================================================================
# MODEL HEALTH ENDPOINTS
# ============================================================================

@app.get("/model-health", response_model=ModelHealthResponse)
async def get_model_health():
    """
    Get model health status.
    
    Returns:
    - health_status: Healthy | Warning | Degraded
    - confidence_score: float
    - drift_trend: float
    - performance_trend: float
    - stability_index: float
    """
    health_monitor = get_model_health_monitor()
    health = health_monitor.get_health_status()
    
    return ModelHealthResponse(
        health_status=health["health_status"],
        confidence_score=health["confidence_score"],
        drift_trend=health["drift_trend"],
        performance_trend=health["performance_trend"],
        stability_index=health["stability_index"]
    )


@app.get("/model-health/detailed")
async def get_model_health_detailed():
    """Get detailed model health metrics."""
    health_monitor = get_model_health_monitor()
    return health_monitor.get_detailed_metrics()


@app.get("/model-health/reset")
async def reset_model_health():
    """Reset model health monitoring data."""
    health_monitor = get_model_health_monitor()
    health_monitor.reset()
    return {"status": "success", "message": "Model health reset successfully"}


# ============================================================================
# THRESHOLD SIMULATION ENDPOINTS
# ============================================================================

@app.post("/simulate-threshold", response_model=ThresholdSimulationResponse)
async def simulate_threshold(request: ThresholdSimulationRequest):
    """
    Simulate decision outcomes using a specific threshold.
    
    Does not retrain model - only re-evaluates decisions using
    the last 500 predictions.
    
    Returns:
    - projected_fraud_detection_rate
    - projected_false_positive_rate
    - projected_decision_distribution
    - projected_auto_block_rate
    """
    simulator = get_threshold_simulator()
    
    result = simulator.simulate_threshold(
        threshold=request.threshold,
        review_threshold=request.review_threshold
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return ThresholdSimulationResponse(
        threshold=result["threshold"],
        sample_size=result["sample_size"],
        projected_fraud_detection_rate=result["projected_fraud_detection_rate"],
        projected_false_positive_rate=result["projected_false_positive_rate"],
        projected_decision_distribution=result["projected_decision_distribution"],
        projected_auto_block_rate=result["projected_auto_block_rate"]
    )


@app.get("/simulate-threshold/optimal")
async def find_optimal_threshold(
    target_fpr: float = 1.0,
    min_detection_rate: float = 80.0
):
    """Find optimal threshold that meets target FPR."""
    simulator = get_threshold_simulator()
    return simulator.find_optimal_threshold(target_fpr, min_detection_rate)


@app.get("/simulate-threshold/compare")
async def compare_thresholds(thresholds: str = "0.3,0.4,0.5,0.6,0.7,0.8"):
    """Compare multiple thresholds."""
    simulator = get_threshold_simulator()
    threshold_list = [float(t.strip()) for t in thresholds.split(",")]
    return simulator.compare_thresholds(threshold_list)


# ============================================================================
# EXPLAINABILITY ENDPOINTS
# ============================================================================

@app.get("/explain/{transaction_id}", response_model=ExplanationResponse)
async def explain_transaction(transaction_id: str):
    """
    Get explanation for a specific transaction.
    
    Returns:
    - top_features: List of contributing features
    - feature_importance_scores: Normalized importance scores
    - explanation_summary: Human-readable summary
    - confidence_level: float
    """
    # First try to get from audit logger
    audit_logger = get_audit_logger()
    log_entry = audit_logger.get_logs_by_transaction_id(transaction_id)
    
    if not log_entry:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {transaction_id} not found"
        )
    
    # Get features from log
    transaction_data = log_entry.get("transaction_data", {})
    
    # Build features dict
    features = {
        "Time": transaction_data.get("Time", 0.0),
        "Amount": transaction_data.get("Amount", 0.0),
    }
    
    # Add V features (we'll simulate them since they're not stored)
    for i in range(1, 29):
        features[f"V{i}"] = 0.0
    
    # Get explanation
    explainability_engine = get_explainability_engine()
    
    explanation = explainability_engine.explain_prediction(
        transaction_id=transaction_id,
        features=features,
        fraud_probability=log_entry.get("fraud_probability", 0.0),
        anomaly_score=log_entry.get("anomaly_score", 0.0),
        risk_score=log_entry.get("fraud_probability", 0.0),
        risk_level=log_entry.get("risk_level", "Unknown"),
        top_factors=log_entry.get("top_risk_factors", [])
    )
    
    return ExplanationResponse(
        transaction_id=explanation["transaction_id"],
        top_features=explanation["top_features"],
        feature_importance_scores=explanation["feature_importance_scores"],
        explanation_summary=explanation["explanation_summary"],
        confidence_level=explanation["confidence_level"]
    )


# ============================================================================
# RISK INTELLIGENCE DASHBOARD ENDPOINTS
# ============================================================================

@app.get("/risk-trends", response_model=RiskTrendsResponse)
async def get_risk_trends():
    """
    Get risk trends for dashboard visualization.
    
    Returns time-series risk data optimized for frontend charts.
    """
    metrics_tracker = get_metrics_tracker()
    rolling_data = metrics_tracker.get_metrics()["rolling_window_metrics"]
    
    # Build time series
    time_series = []
    for entry in rolling_data[-50:]:  # Last 50 entries
        time_series.append({
            "timestamp": entry.get("timestamp", ""),
            "fraud_probability": entry.get("fraud_probability", 0),
            "risk_score": entry.get("risk_score", 0),
            "risk_band": entry.get("risk_band", "Unknown"),
            "drift_detected": entry.get("drift_detected", False)
        })
    
    # Get risk distribution
    risk_dist = metrics_tracker.get_metrics()["risk_band_distribution"]
    
    # Calculate fraud trend
    rolling_stats = metrics_tracker.get_rolling_statistics()
    fraud_trend = rolling_stats.get("fraud_detection_rate", 0)
    
    # Get drift alerts
    drift_events = [e for e in rolling_data if e.get("drift_detected", False)][-5:]
    drift_alerts = [
        {
            "timestamp": e.get("timestamp", ""),
            "transaction_id": e.get("transaction_id", "")
        }
        for e in drift_events
    ]
    
    return RiskTrendsResponse(
        time_series_data=time_series,
        risk_distribution=risk_dist,
        fraud_trend=fraud_trend,
        drift_alerts=drift_alerts
    )


@app.get("/decision-distribution", response_model=DecisionDistributionResponse)
async def get_decision_distribution():
    """
    Get decision distribution for dashboard.
    
    Returns auto-block, manual review, and approval percentages.
    """
    metrics = get_metrics_tracker().get_metrics()
    
    distribution = {
        "Auto-Block": metrics["auto_block_count"],
        "Manual Review": metrics["manual_review_count"],
        "Approve": metrics["approval_count"]
    }
    
    total = metrics["total_transactions"]
    if total > 0:
        percentages = {
            "Auto-Block": round(metrics["auto_block_percentage"], 2),
            "Manual Review": round(metrics["manual_review_percentage"], 2),
            "Approve": round(metrics["approval_percentage"], 2)
        }
    else:
        percentages = {"Auto-Block": 0, "Manual Review": 0, "Approve": 0}
    
    return DecisionDistributionResponse(
        distribution=distribution,
        percentages=percentages,
        auto_block_rate=percentages["Auto-Block"],
        manual_review_rate=percentages["Manual Review"],
        approval_rate=percentages["Approve"]
    )


@app.get("/latency-stats", response_model=LatencyStatsResponse)
async def get_latency_stats():
    """
    Get latency statistics for dashboard.
    
    Returns latency histogram and percentiles.
    """
    metrics_tracker = get_metrics_tracker()
    rolling_stats = metrics_tracker.get_rolling_statistics()
    
    # Get latencies from rolling stats
    latencies = []
    
    return LatencyStatsResponse(
        average_ms=rolling_stats.get("average_latency_ms", 0),
        min_ms=0,
        max_ms=0,
        p50_ms=rolling_stats.get("p50_latency", 0),
        p95_ms=rolling_stats.get("p95_latency", 0),
        p99_ms=rolling_stats.get("p99_latency", 0),
        histogram=[]
    )


# ============================================================================
# SYSTEM LOGS ENDPOINTS
# ============================================================================

@app.get("/system-logs", response_model=SystemLogsResponse)
async def get_system_logs(limit: int = 100, level: Optional[str] = None):
    """
    Get system logs in JSONL format.
    
    Returns structured log entries with all required fields.
    """
    logging_manager = get_logging_manager()
    logs = logging_manager.read_logs(limit=limit, level=level)
    stats = logging_manager.get_log_stats()
    
    return SystemLogsResponse(
        logs=logs,
        total_count=len(logs),
        stats=stats
    )


@app.get("/system-logs/stats")
async def get_system_logs_stats():
    """Get system logs statistics."""
    logging_manager = get_logging_manager()
    return logging_manager.get_log_stats()


# ============================================================================
# AUTH & API KEY ENDPOINTS
# ============================================================================

@app.get("/api-keys")
async def list_api_keys():
    """List all API keys (admin only)."""
    api_key_manager = get_api_key_manager()
    return {"api_keys": api_key_manager.list_keys()}


# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(frontend_path, 'index.html'))

@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse(os.path.join(frontend_path, 'dashboard.html'))

@app.get("/analysis")
async def serve_analysis():
    return FileResponse(os.path.join(frontend_path, 'analysis.html'))

@app.get("/history")
async def serve_history():
    return FileResponse(os.path.join(frontend_path, 'history.html'))

@app.get("/settings")
async def serve_settings():
    return FileResponse(os.path.join(frontend_path, 'settings.html'))

# Serve HTML pages with .html extension
@app.get("/index.html")
async def serve_index_html():
    return FileResponse(os.path.join(frontend_path, 'index.html'))

@app.get("/dashboard.html")
async def serve_dashboard_html():
    return FileResponse(os.path.join(frontend_path, 'dashboard.html'))

@app.get("/analysis.html")
async def serve_analysis_html():
    return FileResponse(os.path.join(frontend_path, 'analysis.html'))

@app.get("/history.html")
async def serve_history_html():
    return FileResponse(os.path.join(frontend_path, 'history.html'))

@app.get("/settings.html")
async def serve_settings_html():
    return FileResponse(os.path.join(frontend_path, 'settings.html'))

# Serve JavaScript and CSS files directly
@app.get("/script.js")
async def serve_script():
    return FileResponse(os.path.join(frontend_path, 'script.js'), media_type="application/javascript")

@app.get("/styles.css")
async def serve_styles():
    return FileResponse(os.path.join(frontend_path, 'styles.css'), media_type="text/css")

# Mount static files directory
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
