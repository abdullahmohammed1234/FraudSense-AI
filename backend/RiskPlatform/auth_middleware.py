"""
AuthMiddleware Module for FraudSense AI.

Provides authentication and authorization:
- API Key validation middleware
- Role-Based Access Control (RBAC)
- Permission matrix
- JWT support (optional)
"""

import os
import hashlib
import secrets
import threading
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from functools import wraps
from datetime import datetime, timedelta


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "Admin"
    ANALYST = "Analyst"
    AUDITOR = "Auditor"
    # Default role for unauthenticated requests
    NONE = "None"


class Permission(str, Enum):
    """Permission constants."""
    # Prediction permissions
    PREDICT = "predict"
    SIMULATE = "simulate"
    
    # Analytics permissions
    GET_METRICS = "get_metrics"
    GET_ANALYTICS = "get_analytics"
    GET_AUDIT_LOG = "get_audit_log"
    
    # Health monitoring
    GET_MODEL_HEALTH = "get_model_health"
    
    # Simulation
    SIMULATE_THRESHOLD = "simulate_threshold"
    
    # Explainability
    EXPLAIN_TRANSACTION = "explain_transaction"
    
    # Dashboard
    RISK_TRENDS = "risk_trends"
    DECISION_DISTRIBUTION = "decision_distribution"
    LATENCY_STATS = "latency_stats"
    
    # Admin permissions
    ADMIN_ACCESS = "admin_access"
    RESET_METRICS = "reset_metrics"
    UPDATE_CONFIG = "update_config"


# Permission matrix: role -> set of permissions
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: {
        Permission.PREDICT,
        Permission.SIMULATE,
        Permission.GET_METRICS,
        Permission.GET_ANALYTICS,
        Permission.GET_AUDIT_LOG,
        Permission.GET_MODEL_HEALTH,
        Permission.SIMULATE_THRESHOLD,
        Permission.EXPLAIN_TRANSACTION,
        Permission.RISK_TRENDS,
        Permission.DECISION_DISTRIBUTION,
        Permission.LATENCY_STATS,
        Permission.ADMIN_ACCESS,
        Permission.RESET_METRICS,
        Permission.UPDATE_CONFIG,
    },
    UserRole.ANALYST: {
        Permission.PREDICT,
        Permission.SIMULATE,
        Permission.GET_METRICS,
        Permission.GET_ANALYTICS,
        Permission.GET_MODEL_HEALTH,
        Permission.SIMULATE_THRESHOLD,
        Permission.EXPLAIN_TRANSACTION,
        Permission.RISK_TRENDS,
        Permission.DECISION_DISTRIBUTION,
        Permission.LATENCY_STATS,
    },
    UserRole.AUDITOR: {
        Permission.GET_AUDIT_LOG,
        Permission.GET_METRICS,
    },
    UserRole.NONE: set(),
}


class APIKeyManager:
    """
    API Key Manager for authentication.
    
    Manages API keys with role assignments and rate limiting.
    """
    
    # Default API keys (should be overridden by environment variables)
    DEFAULT_KEYS = {
        "dev-key-001": {
            "name": "Development Key",
            "role": UserRole.ADMIN,
            "rate_limit": 1000,  # requests per minute
            "enabled": True,
            "created_at": "2024-01-01T00:00:00Z",
        },
        "analyst-key-001": {
            "name": "Analyst Key",
            "role": UserRole.ANALYST,
            "rate_limit": 500,
            "enabled": True,
            "created_at": "2024-01-01T00:00:00Z",
        },
        "auditor-key-001": {
            "name": "Auditor Key",
            "role": UserRole.AUDITOR,
            "rate_limit": 200,
            "enabled": True,
            "created_at": "2024-01-01T00:00:00Z",
        },
    }
    
    def __init__(self):
        """Initialize the API Key Manager."""
        self._lock = threading.RLock()
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._key_hashes: Dict[str, str] = {}
        
        # Load keys from environment or defaults
        self._load_keys()
    
    def _load_keys(self) -> None:
        """Load API keys from environment or defaults."""
        # Try to load from environment
        env_keys = os.environ.get("API_KEYS", "")
        
        if env_keys:
            # Parse comma-separated key:role:rate_limit format
            for key_spec in env_keys.split(","):
                parts = key_spec.split(":")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    role = parts[1].strip()
                    rate_limit = int(parts[2].strip()) if len(parts) > 2 else 500
                    
                    self._api_keys[key] = {
                        "name": f"Env Key ({key[:8]}...)",
                        "role": UserRole(role),
                        "rate_limit": rate_limit,
                        "enabled": True,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                    }
        else:
            # Use default keys
            self._api_keys = self.DEFAULT_KEYS.copy()
        
        # Store hashed versions for secure comparison
        for key in self._api_keys:
            self._key_hashes[self._hash_key(key)] = key
    
    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def validate_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Key info if valid, None otherwise
        """
        if not api_key:
            return None
        
        with self._lock:
            key_hash = self._hash_key(api_key)
            original_key = self._key_hashes.get(key_hash)
            
            if not original_key:
                return None
            
            key_info = self._api_keys.get(original_key)
            
            if not key_info or not key_info.get("enabled", False):
                return None
            
            return {
                "key": original_key,
                "name": key_info["name"],
                "role": key_info["role"],
                "rate_limit": key_info["rate_limit"],
            }
    
    def add_key(
        self,
        name: str,
        role: UserRole,
        rate_limit: int = 500
    ) -> str:
        """
        Add a new API key.
        
        Args:
            name: Name/description of the key
            role: User role
            rate_limit: Requests per minute
            
        Returns:
            Generated API key
        """
        with self._lock:
            # Generate secure random key
            api_key = f"fs-{secrets.token_urlsafe(32)}"
            
            self._api_keys[api_key] = {
                "name": name,
                "role": role,
                "rate_limit": rate_limit,
                "enabled": True,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            
            self._key_hashes[self._hash_key(api_key)] = api_key
            
            return api_key
    
    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked, False if not found
        """
        with self._lock:
            if api_key in self._api_keys:
                self._api_keys[api_key]["enabled"] = False
                return True
            return False
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """
        List all API keys (without exposing the actual keys).
        
        Returns:
            List of key info
        """
        with self._lock:
            return [
                {
                    "name": info["name"],
                    "role": info["role"].value,
                    "rate_limit": info["rate_limit"],
                    "enabled": info["enabled"],
                    "created_at": info["created_at"],
                }
                for info in self._api_keys.values()
            ]
    
    def get_role_permissions(self, role: UserRole) -> Set[Permission]:
        """
        Get permissions for a role.
        
        Args:
            role: User role
            
        Returns:
            Set of permissions
        """
        return ROLE_PERMISSIONS.get(role, set())


class RateLimiter:
    """
    Rate limiter using sliding window algorithm.
    
    Tracks request frequency per API key.
    """
    
    def __init__(self):
        """Initialize the rate limiter."""
        self._lock = threading.RLock()
        self._request_timestamps: Dict[str, List[datetime]] = {}
    
    def check_rate_limit(
        self,
        api_key: str,
        rate_limit: int,
        window_seconds: int = 60
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit.
        
        Args:
            api_key: API key
            rate_limit: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        with self._lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=window_seconds)
            
            # Get timestamps for this key
            if api_key not in self._request_timestamps:
                self._request_timestamps[api_key] = []
            
            timestamps = self._request_timestamps[api_key]
            
            # Remove old timestamps
            timestamps = [ts for ts in timestamps if ts > cutoff]
            self._request_timestamps[api_key] = timestamps
            
            # Check if limit exceeded
            current_count = len(timestamps)
            remaining = max(0, rate_limit - current_count)
            
            if current_count >= rate_limit:
                # Rate limit exceeded
                return False, {
                    "limit": rate_limit,
                    "remaining": 0,
                    "reset_at": (cutoff + timedelta(seconds=window_seconds)).isoformat() + "Z",
                    "retry_after": window_seconds
                }
            
            # Add current timestamp
            timestamps.append(now)
            
            return True, {
                "limit": rate_limit,
                "remaining": remaining - 1,
                "reset_at": (now + timedelta(seconds=window_seconds)).isoformat() + "Z",
            }
    
    def reset(self, api_key: Optional[str] = None) -> None:
        """
        Reset rate limit tracking.
        
        Args:
            api_key: Specific key to reset, or None for all
        """
        with self._lock:
            if api_key:
                self._request_timestamps.pop(api_key, None)
            else:
                self._request_timestamps.clear()


class AuthMiddleware:
    """
    Authentication middleware for FastAPI.
    
    Combines API key validation, RBAC, and rate limiting.
    """
    
    def __init__(self):
        """Initialize the auth middleware."""
        self.api_key_manager = APIKeyManager()
        self.rate_limiter = RateLimiter()
    
    def validate_request(
        self,
        api_key: Optional[str],
        required_permission: Optional[Permission] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Validate an API request.
        
        Args:
            api_key: API key from request header
            required_permission: Required permission
            
        Returns:
            Tuple of (is_valid, user_info, error_message)
        """
        # If no API key required, allow with limited access
        if not api_key:
            # Allow limited access for health check
            if required_permission in [None]:
                return True, {"role": UserRole.NONE}, None
            return False, None, "API key required"
        
        # Validate API key
        key_info = self.api_key_manager.validate_key(api_key)
        
        if not key_info:
            return False, None, "Invalid API key"
        
        # Check rate limit
        is_allowed, rate_info = self.rate_limiter.check_rate_limit(
            api_key,
            key_info["rate_limit"]
        )
        
        if not is_allowed:
            return False, None, f"Rate limit exceeded. {rate_info['retry_after']}s until reset"
        
        # Check permission if required
        if required_permission:
            role = key_info["role"]
            permissions = self.api_key_manager.get_role_permissions(role)
            
            if required_permission not in permissions:
                return False, None, f"Insufficient permissions. Required: {required_permission.value}"
        
        return True, key_info, None
    
    def require_permission(self, permission: Permission):
        """
        Decorator to require a specific permission.
        
        Args:
            permission: Required permission
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get API key from kwargs or context
                api_key = kwargs.get("api_key") or kwargs.get("x_api_key")
                
                is_valid, user_info, error = self.validate_request(api_key, permission)
                
                if not is_valid:
                    return {
                        "error": error,
                        "status_code": 401 if "Invalid" in str(error) else 403
                    }
                
                # Add user info to kwargs
                kwargs["user_info"] = user_info
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator


# Global instances
_api_key_manager: Optional[APIKeyManager] = None
_rate_limiter: Optional[RateLimiter] = None
_auth_middleware: Optional[AuthMiddleware] = None


def get_api_key_manager() -> APIKeyManager:
    """Get or create the API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def get_rate_limiter() -> RateLimiter:
    """Get or create the rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_auth_middleware() -> AuthMiddleware:
    """Get or create the auth middleware."""
    global _auth_middleware
    if _auth_middleware is None:
        _auth_middleware = AuthMiddleware()
    return _auth_middleware


def check_permission(role: UserRole, permission: Permission) -> bool:
    """Check if a role has a specific permission."""
    return permission in ROLE_PERMISSIONS.get(role, set())


if __name__ == "__main__":
    # Test the auth system
    manager = get_api_key_manager()
    limiter = get_rate_limiter()
    middleware = get_auth_middleware()
    
    # Test key validation
    key_info = manager.validate_key("dev-key-001")
    print(f"Key validation: {key_info}")
    
    # Test permissions
    print(f"\nAdmin permissions: {manager.get_role_permissions(UserRole.ADMIN)}")
    print(f"Analyst permissions: {manager.get_role_permissions(UserRole.ANALYST)}")
    print(f"Auditor permissions: {manager.get_role_permissions(UserRole.AUDITOR)}")
    
    # Test rate limiting
    print("\nRate limit test:")
    for i in range(5):
        allowed, info = limiter.check_rate_limit("test-key", 3)
        print(f"  Request {i+1}: allowed={allowed}, remaining={info['remaining']}")
    
    # Test middleware validation
    print("\nMiddleware validation:")
    is_valid, user_info, error = middleware.validate_request(
        "dev-key-001",
        Permission.GET_METRICS
    )
    print(f"  Valid: {is_valid}, User: {user_info}, Error: {error}")
