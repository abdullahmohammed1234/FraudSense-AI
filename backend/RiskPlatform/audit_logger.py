"""
AuditLogger Module for FraudSense AI.

Handles logging of all predictions (manual and simulated) with comprehensive
audit trail including timestamp, transaction_id, fraud_probability, 
anomaly_score, drift_detected, risk_level, threshold_used, model_version,
top_risk_factors, and action_recommendation.
"""

import json
import os
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .config import TRANSACTION_LOGS_PATH, GovernanceConfig


class AuditLogger:
    """
    Thread-safe audit logger for transaction predictions.
    
    Logs are stored in JSON format with all required fields for
    compliance and governance purposes.
    """
    
    def __init__(self, log_path: str = TRANSACTION_LOGS_PATH):
        """
        Initialize the AuditLogger.
        
        Args:
            log_path: Path to the transaction log file.
        """
        self.log_path = log_path
        self._lock = threading.Lock()
        
        # Ensure log file exists
        self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self) -> None:
        """Create log file and directory if they don't exist."""
        if GovernanceConfig.AUTO_CREATE_LOGS:
            # Create directory if needed
            log_dir = os.path.dirname(self.log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Create empty JSON array if file doesn't exist
            if not os.path.exists(self.log_path):
                with open(self.log_path, 'w') as f:
                    json.dump([], f)
    
    def _read_logs(self) -> List[Dict[str, Any]]:
        """
        Read all logs from the file.
        
        Returns:
            List of log entries.
        """
        try:
            with open(self.log_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_logs(self, logs: List[Dict[str, Any]]) -> None:
        """
        Write logs to the file.
        
        Args:
            logs: List of log entries to write.
        """
        with open(self.log_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def log_prediction(
        self,
        transaction_id: str,
        fraud_probability: float,
        anomaly_score: float,
        drift_detected: bool,
        risk_level: str,
        threshold_used: float,
        model_version: str,
        top_risk_factors: List[Dict[str, Any]],
        action_recommendation: str,
        transaction_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a prediction with all required fields.
        
        Args:
            transaction_id: Unique identifier for the transaction.
            fraud_probability: Fraud probability (0-1).
            anomaly_score: Anomaly score (0-1).
            drift_detected: Whether drift was detected.
            risk_level: Risk level (Low/Medium/High/Critical).
            threshold_used: Threshold used for classification.
            model_version: Version of the model used.
            top_risk_factors: List of top risk factors.
            action_recommendation: Recommended action.
            transaction_data: Optional additional transaction data.
            
        Returns:
            The created log entry.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "transaction_id": transaction_id,
            "fraud_probability": round(fraud_probability, 6),
            "anomaly_score": round(anomaly_score, 6),
            "drift_detected": drift_detected,
            "risk_level": risk_level,
            "threshold_used": threshold_used,
            "model_version": model_version,
            "top_risk_factors": top_risk_factors,
            "action_recommendation": action_recommendation
        }
        
        # Add optional transaction data
        if transaction_data:
            log_entry["transaction_data"] = {
                "amount": transaction_data.get("Amount", 0.0),
                "time": transaction_data.get("Time", 0.0)
            }
        
        # Thread-safe append
        with self._lock:
            logs = self._read_logs()
            logs.append(log_entry)
            self._write_logs(logs)
        
        return log_entry
    
    def get_logs(self, limit: int = GovernanceConfig.AUDIT_LOG_LIMIT) -> List[Dict[str, Any]]:
        """
        Get recent logs, sorted newest first.
        
        Args:
            limit: Maximum number of logs to return.
            
        Returns:
            List of log entries (newest first).
        """
        with self._lock:
            logs = self._read_logs()
        
        # Sort by timestamp (newest first) and limit
        sorted_logs = sorted(
            logs,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        return sorted_logs[:limit]
    
    def get_logs_by_risk_level(self, risk_level: str) -> List[Dict[str, Any]]:
        """
        Get logs filtered by risk level.
        
        Args:
            risk_level: Risk level to filter by.
            
        Returns:
            List of matching log entries.
        """
        with self._lock:
            logs = self._read_logs()
        
        return [log for log in logs if log.get("risk_level") == risk_level]
    
    def get_logs_by_transaction_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific log entry by transaction ID.
        
        Args:
            transaction_id: Transaction ID to search for.
            
        Returns:
            Log entry if found, None otherwise.
        """
        with self._lock:
            logs = self._read_logs()
        
        for log in logs:
            if log.get("transaction_id") == transaction_id:
                return log
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get audit log statistics.
        
        Returns:
            Dictionary with statistics.
        """
        with self._lock:
            logs = self._read_logs()
        
        if not logs:
            return {
                "total_logs": 0,
                "by_risk_level": {},
                "by_action": {},
                "average_fraud_probability": 0.0,
                "drift_detected_count": 0
            }
        
        risk_level_counts = {}
        action_counts = {}
        total_fraud_prob = 0.0
        drift_count = 0
        
        for log in logs:
            # Count by risk level
            rl = log.get("risk_level", "Unknown")
            risk_level_counts[rl] = risk_level_counts.get(rl, 0) + 1
            
            # Count by action
            action = log.get("action_recommendation", "Unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Sum fraud probabilities
            total_fraud_prob += log.get("fraud_probability", 0.0)
            
            # Count drift
            if log.get("drift_detected", False):
                drift_count += 1
        
        return {
            "total_logs": len(logs),
            "by_risk_level": risk_level_counts,
            "by_action": action_counts,
            "average_fraud_probability": round(total_fraud_prob / len(logs), 4),
            "drift_detected_count": drift_count
        }
    
    def clear_logs(self) -> int:
        """
        Clear all logs (use with caution).
        
        Returns:
            Number of logs cleared.
        """
        with self._lock:
            logs = self._read_logs()
            count = len(logs)
            self._write_logs([])
        
        return count
    
    def export_logs(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Export logs within a date range.
        
        Args:
            start_date: Start date in ISO format.
            end_date: End date in ISO format.
            
        Returns:
            List of log entries within the date range.
        """
        with self._lock:
            logs = self._read_logs()
        
        if not start_date and not end_date:
            return logs
        
        filtered_logs = []
        for log in logs:
            timestamp = log.get("timestamp", "")
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                continue
            filtered_logs.append(log)
        
        return filtered_logs


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """
    Get or create the global audit logger instance.
    
    Returns:
        AuditLogger instance.
    """
    global _audit_logger
    
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    
    return _audit_logger


def generate_transaction_id() -> str:
    """
    Generate a unique transaction ID.
    
    Returns:
        Unique transaction ID string.
    """
    return f"TXN-{uuid.uuid4().hex[:12].upper()}"


if __name__ == "__main__":
    # Test the audit logger
    logger = get_audit_logger()
    
    # Log a test prediction
    logger.log_prediction(
        transaction_id=generate_transaction_id(),
        fraud_probability=0.85,
        anomaly_score=0.72,
        drift_detected=False,
        risk_level="High",
        threshold_used=0.5,
        model_version="1.0.0",
        top_risk_factors=[
            {"feature": "V14", "impact": 0.45},
            {"feature": "V17", "impact": 0.32}
        ],
        action_recommendation="Auto-Block Transaction",
        transaction_data={"Amount": 150.0, "Time": 400.0}
    )
    
    # Get logs
    logs = logger.get_logs(limit=10)
    print(f"Total logs: {len(logs)}")
    print(f"Latest log: {json.dumps(logs[0], indent=2)}")
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2)}")
