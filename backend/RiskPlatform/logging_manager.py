"""
LoggingManager Module for FraudSense AI.

Provides structured JSONL logging with:
- timestamp
- request_id
- endpoint
- latency_ms
- user_role
- transaction_id (if applicable)
- risk_level
- decision
- status_code
"""

import json
import os
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging


class LoggingManager:
    """
    Structured JSONL logging manager.
    
    Provides thread-safe JSONL logging with all required fields.
    """
    
    LOG_DIR = "logs"
    SYSTEM_LOGS_FILE = "system_logs.jsonl"
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        max_file_size_mb: int = 100,
        backup_count: int = 5
    ):
        """
        Initialize the LoggingManager.
        
        Args:
            log_dir: Directory for log files
            log_file: Log file name
            max_file_size_mb: Maximum log file size in MB
            backup_count: Number of backup files to keep
        """
        self.log_dir = log_dir or self.LOG_DIR
        self.log_file = log_file or self.SYSTEM_LOGS_FILE
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.backup_count = backup_count
        
        self._lock = threading.RLock()
        self._log_path = os.path.join(self.log_dir, self.log_file)
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize Python logger
        self._logger = logging.getLogger("RiskPlatform")
        self._logger.setLevel(logging.INFO)
        
        # Add file handler if not already present
        if not self._logger.handlers:
            handler = logging.FileHandler(self._log_path)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self._logger.addHandler(handler)
    
    def _should_rotate(self) -> bool:
        """Check if log file should be rotated."""
        if not os.path.exists(self._log_path):
            return False
        return os.path.getsize(self._log_path) > self.max_file_size
    
    def _rotate_logs(self) -> None:
        """Rotate log files."""
        # Remove oldest backup
        oldest_backup = os.path.join(
            self.log_dir,
            f"{self.log_file}.{self.backup_count}"
        )
        if os.path.exists(oldest_backup):
            os.remove(oldest_backup)
        
        # Shift backups
        for i in range(self.backup_count - 1, 0, -1):
            src = os.path.join(self.log_dir, f"{self.log_file}.{i}")
            dst = os.path.join(self.log_dir, f"{self.log_file}.{i + 1}")
            if os.path.exists(src):
                os.rename(src, dst)
        
        # Rotate current log
        if os.path.exists(self._log_path):
            os.rename(self._log_path, os.path.join(self.log_dir, f"{self.log_file}.1"))
    
    def log(
        self,
        level: str,
        message: str,
        request_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        latency_ms: Optional[float] = None,
        user_role: Optional[str] = None,
        transaction_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        decision: Optional[str] = None,
        status_code: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Log a structured JSONL entry.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: Log message
            request_id: Unique request identifier
            endpoint: API endpoint
            latency_ms: Request latency in milliseconds
            user_role: User role
            transaction_id: Transaction ID
            risk_level: Risk level
            decision: Decision made
            status_code: HTTP status code
            extra: Additional fields
            **kwargs: Additional custom fields
            
        Returns:
            The log entry
        """
        with self._lock:
            # Check rotation
            if self._should_rotate():
                self._rotate_logs()
            
            # Generate request ID if not provided
            if not request_id:
                request_id = f"REQ-{uuid.uuid4().hex[:12].upper()}"
            
            # Build log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": level.upper(),
                "message": message,
                "request_id": request_id,
            }
            
            # Add optional fields
            if endpoint:
                log_entry["endpoint"] = endpoint
            
            if latency_ms is not None:
                log_entry["latency_ms"] = round(latency_ms, 2)
            
            if user_role:
                log_entry["user_role"] = user_role
            
            if transaction_id:
                log_entry["transaction_id"] = transaction_id
            
            if risk_level:
                log_entry["risk_level"] = risk_level
            
            if decision:
                log_entry["decision"] = decision
            
            if status_code:
                log_entry["status_code"] = status_code
            
            # Add extra fields
            if extra:
                log_entry.update(extra)
            
            # Add custom kwargs
            if kwargs:
                log_entry.update(kwargs)
            
            # Write to JSONL file
            try:
                with open(self._log_path, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                # Fallback to Python logger
                self._logger.error(f"Failed to write JSONL: {e}")
            
            # Also log to Python logger
            log_msg = f"[{level}] {message}"
            if endpoint:
                log_msg += f" | endpoint={endpoint}"
            if status_code:
                log_msg += f" | status={status_code}"
            
            if level == "ERROR":
                self._logger.error(log_msg)
            elif level == "WARNING":
                self._logger.warning(log_msg)
            else:
                self._logger.info(log_msg)
            
            return log_entry
    
    def log_request(
        self,
        endpoint: str,
        method: str,
        request_id: Optional[str] = None,
        user_role: Optional[str] = None,
        status_code: Optional[int] = None,
        latency_ms: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Log an HTTP request."""
        return self.log(
            level="INFO",
            message=f"{method} {endpoint}",
            request_id=request_id,
            endpoint=f"{method} {endpoint}",
            user_role=user_role,
            status_code=status_code,
            latency_ms=latency_ms,
            **kwargs
        )
    
    def log_prediction(
        self,
        transaction_id: str,
        fraud_probability: float,
        risk_level: str,
        decision: str,
        request_id: Optional[str] = None,
        user_role: Optional[str] = None,
        latency_ms: Optional[float] = None,
        status_code: int = 200,
        **kwargs
    ) -> Dict[str, Any]:
        """Log a prediction request."""
        return self.log(
            level="INFO",
            message=f"Prediction for transaction {transaction_id}",
            request_id=request_id,
            endpoint="/predict",
            user_role=user_role,
            transaction_id=transaction_id,
            risk_level=risk_level,
            decision=decision,
            status_code=status_code,
            latency_ms=latency_ms,
            fraud_probability=round(fraud_probability, 4),
            **kwargs
        )
    
    def log_error(
        self,
        message: str,
        error: Optional[Exception] = None,
        request_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """Log an error."""
        extra = {}
        if error:
            extra["error_type"] = type(error).__name__
            extra["error_message"] = str(error)
        
        return self.log(
            level="ERROR",
            message=message,
            request_id=request_id,
            endpoint=endpoint,
            status_code=status_code,
            extra=extra,
            **kwargs
        )
    
    def read_logs(
        self,
        limit: int = 100,
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Read logs from the JSONL file.
        
        Args:
            limit: Maximum number of logs to return
            level: Filter by log level
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of log entries
        """
        logs = []
        
        if not os.path.exists(self._log_path):
            return logs
        
        try:
            with open(self._log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Apply filters
                        if level and entry.get("level") != level.upper():
                            continue
                        
                        if start_time:
                            entry_time = datetime.fromisoformat(entry["timestamp"].rstrip("Z"))
                            if entry_time < start_time:
                                continue
                        
                        if end_time:
                            entry_time = datetime.fromisoformat(entry["timestamp"].rstrip("Z"))
                            if entry_time > end_time:
                                continue
                        
                        logs.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        
        # Return most recent first
        return logs[-limit:][::-1]
    
    def clear_logs(self) -> int:
        """
        Clear all logs.
        
        Returns:
            Number of logs cleared
        """
        with self._lock:
            if not os.path.exists(self._log_path):
                return 0
            
            # Count lines
            with open(self._log_path, 'r') as f:
                count = sum(1 for _ in f)
            
            # Clear file
            with open(self._log_path, 'w') as f:
                pass
            
            return count
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get log statistics."""
        if not os.path.exists(self._log_path):
            return {
                "total_logs": 0,
                "file_size_bytes": 0,
                "levels": {}
            }
        
        levels = {}
        total = 0
        
        try:
            with open(self._log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        total += 1
                        level = entry.get("level", "UNKNOWN")
                        levels[level] = levels.get(level, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        
        return {
            "total_logs": total,
            "file_size_bytes": os.path.getsize(self._log_path),
            "levels": levels
        }


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def get_logging_manager() -> LoggingManager:
    """
    Get or create the global logging manager.
    
    Returns:
        LoggingManager instance
    """
    global _logging_manager
    
    if _logging_manager is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, "logs")
        _logging_manager = LoggingManager(log_dir=log_dir)
    
    return _logging_manager


def log_request(**kwargs) -> Dict[str, Any]:
    """Convenience function to log a request."""
    return get_logging_manager().log_request(**kwargs)


def log_prediction(**kwargs) -> Dict[str, Any]:
    """Convenience function to log a prediction."""
    return get_logging_manager().log_prediction(**kwargs)


def log_error(**kwargs) -> Dict[str, Any]:
    """Convenience function to log an error."""
    return get_logging_manager().log_error(**kwargs)


if __name__ == "__main__":
    # Test the logging manager
    manager = get_logging_manager()
    
    # Log some test entries
    print("Logging test entries...")
    
    manager.log_request(
        endpoint="/predict",
        method="POST",
        request_id="REQ-TEST001",
        user_role="Admin",
        status_code=200,
        latency_ms=45.2
    )
    
    manager.log_prediction(
        transaction_id="TXN-TEST123",
        fraud_probability=0.78,
        risk_level="High",
        decision="Auto-Block Transaction",
        request_id="REQ-TEST001",
        user_role="Admin",
        latency_ms=45.2,
        status_code=200
    )
    
    manager.log_error(
        message="Test error",
        error=ValueError("Test error message"),
        request_id="REQ-TEST002",
        endpoint="/predict"
    )
    
    # Read logs
    print("\nReading logs...")
    logs = manager.read_logs(limit=10)
    for log in logs:
        print(f"  {log.get('timestamp')}: {log.get('message')}")
    
    # Get stats
    print("\nLog statistics:")
    stats = manager.get_log_stats()
    print(f"  Total logs: {stats['total_logs']}")
    print(f"  By level: {stats['levels']}")
