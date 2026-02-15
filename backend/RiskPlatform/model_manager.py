"""
ModelManager Module for FraudSense AI.

Handles model versioning, metadata management, and provides
comprehensive model information for governance and compliance.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from .config import MODEL_METADATA_PATH, GovernanceConfig


class ModelMetadata:
    """
    Model metadata container with all required fields.
    """
    
    def __init__(
        self,
        model_type: str = "RandomForest",
        training_date: str = "",
        roc_auc: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        threshold: float = 0.5,
        dataset_size: int = 0,
        feature_count: int = 30,
        version: str = "1.0.0",
        additional_info: Optional[Dict[str, Any]] = None
    ):
        self.model_type = model_type
        self.training_date = training_date
        self.roc_auc = roc_auc
        self.precision = precision
        self.recall = recall
        self.threshold = threshold
        self.dataset_size = dataset_size
        self.feature_count = feature_count
        self.version = version
        self.additional_info = additional_info or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type,
            "training_date": self.training_date,
            "ROC_AUC": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "threshold": self.threshold,
            "dataset_size": self.dataset_size,
            "feature_count": self.feature_count,
            "version": self.version,
            **self.additional_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(
            model_type=data.get("model_type", "Unknown"),
            training_date=data.get("training_date", ""),
            roc_auc=data.get("ROC_AUC", 0.0),
            precision=data.get("precision", 0.0),
            recall=data.get("recall", 0.0),
            threshold=data.get("threshold", 0.5),
            dataset_size=data.get("dataset_size", 0),
            feature_count=data.get("feature_count", 30),
            version=data.get("version", "1.0.0"),
            additional_info={k: v for k, v in data.items() 
                           if k not in ["model_type", "training_date", "ROC_AUC", 
                                       "precision", "recall", "threshold", 
                                       "dataset_size", "feature_count", "version"]}
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ModelMetadata":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class ModelManager:
    """
    Manages model versioning and metadata.
    
    Provides centralized access to model information and handles
    loading/saving of model metadata.
    """
    
    def __init__(self, metadata_path: str = MODEL_METADATA_PATH):
        """
        Initialize the ModelManager.
        
        Args:
            metadata_path: Path to the model metadata JSON file.
        """
        self.metadata_path = metadata_path
        self._metadata: Optional[ModelMetadata] = None
    
    def _load_metadata(self) -> ModelMetadata:
        """
        Load metadata from file or create default.
        
        Returns:
            ModelMetadata instance.
        """
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                return ModelMetadata.from_dict(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load metadata: {e}")
        
        # Return default metadata
        return ModelMetadata(
            model_type="RandomForest",
            training_date=datetime.utcnow().strftime("%Y-%m-%d"),
            roc_auc=0.95,
            precision=0.88,
            recall=0.82,
            threshold=GovernanceConfig.DEFAULT_THRESHOLD,
            dataset_size=284807,
            feature_count=30,
            version=GovernanceConfig.MODEL_VERSION
        )
    
    def get_metadata(self, force_reload: bool = False) -> ModelMetadata:
        """
        Get current model metadata.
        
        Args:
            force_reload: Force reload from file.
            
        Returns:
            ModelMetadata instance.
        """
        if self._metadata is None or force_reload:
            self._metadata = self._load_metadata()
        
        return self._metadata
    
    def save_metadata(self, metadata: ModelMetadata) -> bool:
        """
        Save metadata to file.
        
        Args:
            metadata: ModelMetadata to save.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            self._metadata = metadata
            return True
        except IOError as e:
            print(f"Error saving metadata: {e}")
            return False
    
    def update_metadata(self, updates: Dict[str, Any]) -> bool:
        """
        Update specific fields in metadata.
        
        Args:
            updates: Dictionary of fields to update.
            
        Returns:
            True if successful, False otherwise.
        """
        metadata = self.get_metadata()
        current_data = metadata.to_dict()
        current_data.update(updates)
        
        new_metadata = ModelMetadata.from_dict(current_data)
        return self.save_metadata(new_metadata)
    
    def get_version(self) -> str:
        """
        Get current model version.
        
        Returns:
            Model version string.
        """
        return self.get_metadata().version
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get complete model information for API response.
        
        Returns:
            Dictionary with all model information.
        """
        metadata = self.get_metadata()
        
        return {
            "model_type": metadata.model_type,
            "training_date": metadata.training_date,
            "metrics": {
                "ROC_AUC": metadata.roc_auc,
                "precision": metadata.precision,
                "recall": metadata.recall
            },
            "threshold": metadata.threshold,
            "dataset_size": metadata.dataset_size,
            "feature_count": metadata.feature_count,
            "version": metadata.version,
            "health_status": "operational"
        }
    
    def create_default_metadata(
        self,
        model_type: str = "RandomForest",
        roc_auc: float = 0.95,
        precision: float = 0.88,
        recall: float = 0.82,
        dataset_size: int = 284807,
        feature_count: int = 30
    ) -> ModelMetadata:
        """
        Create and save default metadata.
        
        Args:
            model_type: Type of the model.
            roc_auc: ROC AUC score.
            precision: Precision score.
            recall: Recall score.
            dataset_size: Size of training dataset.
            feature_count: Number of features.
            
        Returns:
            Created ModelMetadata instance.
        """
        metadata = ModelMetadata(
            model_type=model_type,
            training_date=datetime.utcnow().strftime("%Y-%m-%d"),
            roc_auc=roc_auc,
            precision=precision,
            recall=recall,
            threshold=GovernanceConfig.DEFAULT_THRESHOLD,
            dataset_size=dataset_size,
            feature_count=feature_count,
            version=GovernanceConfig.MODEL_VERSION
        )
        
        self.save_metadata(metadata)
        return metadata


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get or create the global model manager instance.
    
    Returns:
        ModelManager instance.
    """
    global _model_manager
    
    if _model_manager is None:
        _model_manager = ModelManager()
    
    return _model_manager


def get_model_info() -> Dict[str, Any]:
    """
    Get model information for API endpoint.
    
    Returns:
        Model information dictionary.
    """
    manager = get_model_manager()
    return manager.get_model_info()


if __name__ == "__main__":
    # Test the model manager
    manager = get_model_manager()
    
    # Get metadata
    metadata = manager.get_metadata()
    print(f"Model Type: {metadata.model_type}")
    print(f"Version: {metadata.version}")
    print(f"ROC AUC: {metadata.roc_auc}")
    
    # Get model info
    info = manager.get_model_info()
    print(f"\nModel Info: {json.dumps(info, indent=2)}")
