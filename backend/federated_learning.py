"""
Federated Learning Module.

This module provides privacy-preserving training across institutions:
- Secure aggregation of model updates
- Differential privacy
- Communication-efficient training
- Cross-institution collaboration
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import base64


@dataclass
class LocalModelUpdate:
    """Model update from a single institution."""
    institution_id: str
    round_number: int
    timestamp: datetime
    model_weights: Dict[str, np.ndarray]
    num_samples: int
    metrics: Dict[str, float]
    gradient_norm: float


@dataclass
class AggregatedModel:
    """Aggregated model from multiple institutions."""
    round_number: int
    timestamp: datetime
    model_weights: Dict[str, np.ndarray]
    participating_institutions: List[str]
    total_samples: int
    aggregation_method: str


@dataclass
class Institution:
    """Participating institution in federated learning."""
    institution_id: str
    name: str
    data_size: int
    trust_score: float
    is_active: bool
    last_seen: datetime


class SecureAggregator:
    """Secure aggregation of model updates."""
    
    def __init__(self, noise_multiplier: float = 1.0, clipping_norm: float = 1.0):
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.institutional_weights: Dict[str, float] = {}
    
    def add_noise(self, gradient: np.ndarray) -> np.ndarray:
        """Add Gaussian noise for differential privacy."""
        noise = np.random.normal(0, self.noise_multiplier * self.clipping_norm, gradient.shape)
        return gradient + noise
    
    def clip_gradients(self, gradient: np.ndarray) -> np.ndarray:
        """Clip gradients to bounded norm."""
        norm = np.linalg.norm(gradient)
        if norm > self.clipping_norm:
            return gradient * (self.clipping_norm / norm)
        return gradient
    
    def federated_averaging(
        self,
        updates: List[LocalModelUpdate]
    ) -> Dict[str, np.ndarray]:
        """FedAvg aggregation of model weights."""
        
        if not updates:
            return {}
        
        aggregated = {}
        
        # Get all weight keys
        all_keys = set()
        for update in updates:
            all_keys.update(update.model_weights.keys())
        
        # Aggregate each weight
        for key in all_keys:
            weighted_sum = None
            total_weight = 0
            
            for update in updates:
                if key in update.model_weights:
                    weight = update.num_samples
                    if update.institution_id in self.institutional_weights:
                        weight *= self.institutional_weights[update.institution_id]
                    
                    if weighted_sum is None:
                        weighted_sum = update.model_weights[key] * weight
                    else:
                        weighted_sum += update.model_weights[key] * weight
                    
                    total_weight += weight
            
            if weighted_sum is not None and total_weight > 0:
                aggregated[key] = weighted_sum / total_weight
        
        return aggregated
    
    def secure_aggregate(
        self,
        updates: List[LocalModelUpdate],
        apply_dp: bool = True
    ) -> Dict[str, np.ndarray]:
        """Perform secure aggregation with optional differential privacy."""
        
        aggregated = self.federated_averaging(updates)
        
        if apply_dp:
            for key in aggregated:
                clipped = self.clip_gradients(aggregated[key])
                aggregated[key] = self.add_noise(clipped)
        
        return aggregated


class DifferentialPrivacy:
    """Differential privacy mechanisms."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
    
    def compute_sigma(self, sensitivity: float) -> float:
        """Compute noise scale based on privacy budget."""
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_laplace_noise(self, data: np.ndarray, sensitivity: float) -> np.ndarray:
        """Add Laplace noise for differential privacy."""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise
    
    def add_gaussian_noise(self, data: np.ndarray, sensitivity: float) -> np.ndarray:
        """Add Gaussian noise for differential privacy."""
        sigma = self.compute_sigma(sensitivity)
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise
    
    def get_privacy_budget(self, num_iterations: int) -> Tuple[float, float]:
        """Calculate cumulative privacy budget."""
        composition_epsilon = self.epsilon * np.sqrt(2 * np.log(1 / self.delta) * num_iterations)
        composition_delta = self.delta * num_iterations
        return composition_epsilon, composition_delta
    
    def randomize_response(self, true_result: np.ndarray, epsilon: float) -> np.ndarray:
        """Exponential mechanism for private query responses."""
        return true_result + np.random.laplace(0, 1/epsilon, true_result.shape)


class CommunicationOptimizer:
    """Optimize communication in federated learning."""
    
    def __init__(
        self,
        compression_scheme: str = "top_k",
        compression_ratio: float = 0.1,
        quantization_bits: int = 8
    ):
        self.compression_scheme = compression_scheme
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits
    
    def compress_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Compress model weights for transmission."""
        
        if self.compression_scheme == "top_k":
            return self._top_k_compression(weights)
        elif self.compression_scheme == "quantization":
            return self._quantize_weights(weights)
        elif self.compression_scheme == "pruning":
            return self._prune_weights(weights)
        else:
            return weights, {"method": "none"}
    
    def _top_k_compression(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Keep only top k% by absolute value."""
        flat = weights.flatten()
        k = int(len(flat) * self.compression_ratio)
        
        if k == 0:
            return weights, {"method": "top_k", "k": 0}
        
        indices = np.argpartition(np.abs(flat), -k)[-k:]
        values = flat[indices]
        
        # Create sparse representation
        compressed = np.zeros_like(flat)
        compressed[indices] = values
        
        return compressed.reshape(weights.shape), {
            "method": "top_k",
            "k": k,
            "compression_ratio": self.compression_ratio
        }
    
    def _quantize_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Quantize weights to reduced precision."""
        min_val = weights.min()
        max_val = weights.max()
        
        num_bins = 2 ** self.quantization_bits
        scale = (max_val - min_val) / num_bins
        
        quantized = np.round((weights - min_val) / scale) * scale + min_val
        
        return quantized, {
            "method": "quantization",
            "bits": self.quantization_bits,
            "num_bins": num_bins
        }
    
    def _prune_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Zero out small magnitude weights."""
        threshold = np.percentile(np.abs(weights), 100 * (1 - self.compression_ratio))
        pruned = weights.copy()
        pruned[np.abs(pruned) < threshold] = 0
        
        sparsity = np.sum(pruned == 0) / pruned.size
        
        return pruned, {
            "method": "pruning",
            "threshold": threshold,
            "sparsity": sparsity
        }
    
    def decompress_weights(self, weights: np.ndarray, metadata: Dict) -> np.ndarray:
        """Decompress weights."""
        return weights


class FederatedLearningCoordinator:
    """Main coordinator for federated learning."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        max_grad_norm: float = 1.0,
        compression_scheme: str = "top_k"
    ):
        self.secure_aggregator = SecureAggregator(
            noise_multiplier=epsilon,
            clipping_norm=max_grad_norm
        )
        self.dp = DifferentialPrivacy(epsilon=epsilon, max_grad_norm=max_grad_norm)
        self.compressor = CommunicationOptimizer(compression_scheme=compression_scheme)
        
        self.institutions: Dict[str, Institution] = {}
        self.update_history: List[LocalModelUpdate] = []
        self.current_round = 0
    
    def register_institution(
        self,
        institution_id: str,
        name: str,
        data_size: int
    ) -> Institution:
        """Register a new institution."""
        
        institution = Institution(
            institution_id=institution_id,
            name=name,
            data_size=data_size,
            trust_score=1.0,
            is_active=True,
            last_seen=datetime.now()
        )
        
        self.institutions[institution_id] = institution
        
        # Set institutional weight based on data size
        self.secure_aggregator.institutional_weights[institution_id] = min(data_size / 100000, 2.0)
        
        return institution
    
    def receive_update(
        self,
        institution_id: str,
        model_weights: Dict[str, np.ndarray],
        num_samples: int,
        metrics: Dict[str, float]
    ) -> LocalModelUpdate:
        """Receive model update from an institution."""
        
        # Compress weights
        compressed_weights = {}
        for key, weights in model_weights.items():
            compressed, _ = self.compressor.compress_weights(weights)
            compressed_weights[key] = compressed
        
        # Calculate gradient norm
        flat_weights = np.concatenate([w.flatten() for w in compressed_weights.values()])
        gradient_norm = float(np.linalg.norm(flat_weights))
        
        update = LocalModelUpdate(
            institution_id=institution_id,
            round_number=self.current_round,
            timestamp=datetime.now(),
            model_weights=compressed_weights,
            num_samples=num_samples,
            metrics=metrics,
            gradient_norm=gradient_norm
        )
        
        self.update_history.append(update)
        
        # Update institution last seen
        if institution_id in self.institutions:
            self.institutions[institution_id].last_seen = datetime.now()
        
        return update
    
    def aggregate_updates(
        self,
        apply_dp: bool = True
    ) -> AggregatedModel:
        """Aggregate all received updates for current round."""
        
        current_updates = [
            u for u in self.update_history
            if u.round_number == self.current_round
        ]
        
        if not current_updates:
            raise ValueError("No updates available for aggregation")
        
        # Perform secure aggregation
        aggregated_weights = self.secure_aggregator.secure_aggregate(
            current_updates,
            apply_dp=apply_dp
        )
        
        # Get participation info
        institutions = [u.institution_id for u in current_updates]
        total_samples = sum(u.num_samples for u in current_updates)
        
        # Decompress weights for final model
        final_weights = {}
        for key, weights in aggregated_weights.items():
            final_weights[key] = self.compressor.decompress_weights(weights, {})
        
        return AggregatedModel(
            round_number=self.current_round,
            timestamp=datetime.now(),
            model_weights=final_weights,
            participating_institutions=institutions,
            total_samples=total_samples,
            aggregation_method="fedavg_dp" if apply_dp else "fedavg"
        )
    
    def next_round(self) -> None:
        """Move to next training round."""
        self.current_round += 1
    
    def get_round_summary(self) -> Dict[str, Any]:
        """Get summary of current round."""
        
        current_updates = [
            u for u in self.update_history
            if u.round_number == self.current_round
        ]
        
        return {
            "round": self.current_round,
            "updates_received": len(current_updates),
            "institutions": [u.institution_id for u in current_updates],
            "total_samples": sum(u.num_samples for u in current_updates),
            "avg_gradient_norm": np.mean([u.gradient_norm for u in current_updates]) if current_updates else 0,
            "privacy_budget": self.dp.get_privacy_budget(self.current_round + 1)
        }
    
    def get_institution_stats(self) -> Dict[str, Any]:
        """Get statistics about participating institutions."""
        
        return {
            "total_institutions": len(self.institutions),
            "active_institutions": sum(1 for i in self.institutions.values() if i.is_active),
            "institutions": [
                {
                    "id": inst.institution_id,
                    "name": inst.name,
                    "data_size": inst.data_size,
                    "trust_score": inst.trust_score,
                    "last_seen": inst.last_seen.isoformat()
                }
                for inst in self.institutions.values()
            ]
        }
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Get privacy budget report."""
        
        total_updates = len(self.update_history)
        epsilon, delta = self.dp.get_privacy_budget(total_updates)
        
        return {
            "current_round": self.current_round,
            "total_updates": total_updates,
            "epsilon": epsilon,
            "delta": delta,
            "privacy_guarantee": f"(ε={epsilon:.2f}, δ={delta:.2e})-differential privacy",
            "noise_multiplier": self.secure_aggregator.noise_multiplier,
            "clipping_norm": self.secure_aggregator.clipping_norm
        }


class CrossInstitutionAnalyzer:
    """Analyze patterns across institutions."""
    
    def __init__(self):
        self.coordinator = FederatedLearningCoordinator()
    
    def compare_fraud_patterns(
        self,
        institution_fraud_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Compare fraud patterns across institutions."""
        
        pattern_comparison = {}
        
        for institution_id, fraud_data in institution_fraud_data.items():
            if not fraud_data:
                continue
            
            amounts = [d.get("amount", 0) for d in fraud_data]
            probabilities = [d.get("fraud_probability", 0) for d in fraud_data]
            
            pattern_comparison[institution_id] = {
                "fraud_count": len(fraud_data),
                "avg_amount": float(np.mean(amounts)),
                "avg_fraud_probability": float(np.mean(probabilities)),
                "max_fraud_probability": float(np.max(probabilities)),
                "fraud_rate": len(fraud_data) / max(len(fraud_data) + 100, 1)
            }
        
        return pattern_comparison
    
    def detect_coordinated_fraud(
        self,
        cross_institution_transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect fraud that spans multiple institutions."""
        
        suspicious_clusters = []
        
        # Simple clustering based on shared attributes
        by_amount = defaultdict(list)
        for txn in cross_institution_transactions:
            amount = txn.get("amount", 0)
            amount_bucket = int(amount / 100) * 100
            by_amount[amount_bucket].append(txn)
        
        # Find amounts with multiple transactions
        for bucket, txns in by_amount.items():
            if len(txns) >= 3:
                avg_prob = np.mean([t.get("fraud_probability", 0) for t in txns])
                if avg_prob > 0.5:
                    suspicious_clusters.append({
                        "amount_range": f"{bucket}-{bucket+100}",
                        "transaction_count": len(txns),
                        "avg_fraud_probability": avg_prob,
                        "institutions": list(set(t.get("institution_id", "unknown") for t in txns))
                    })
        
        return {
            "suspicious_clusters": suspicious_clusters,
            "total_clusters": len(suspicious_clusters),
            "total_cross_institution_txns": len(cross_institution_transactions)
        }
    
    def get_collaboration_recommendations(
        self,
        institution_metrics: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Recommend collaboration opportunities between institutions."""
        
        recommendations = []
        
        institution_ids = list(institution_metrics.keys())
        
        for i, inst1 in enumerate(institution_ids):
            for inst2 in institution_ids[i+1:]:
                m1 = institution_metrics.get(inst1, {})
                m2 = institution_metrics.get(inst2, {})
                
                # Calculate similarity
                roc_auc1 = m1.get("roc_auc", 0)
                roc_auc2 = m2.get("roc_auc", 0)
                
                similarity = 1 - abs(roc_auc1 - roc_auc2)
                
                if similarity > 0.8:
                    recommendations.append({
                        "institution_pair": [inst1, inst2],
                        "similarity": similarity,
                        "recommendation": "share_best_practices" if abs(roc_auc1 - roc_auc2) < 0.05 else "collaborate_on_model"
                    })
        
        return recommendations


_global_coordinator: Optional[FederatedLearningCoordinator] = None


def get_federated_coordinator() -> FederatedLearningCoordinator:
    """Get global federated learning coordinator."""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = FederatedLearningCoordinator()
    return _global_coordinator


def federated_train_step(
    institution_id: str,
    local_data: pd.DataFrame,
    current_global_model: Dict[str, np.ndarray]
) -> LocalModelUpdate:
    """Perform one federated training step."""
    
    coordinator = get_federated_coordinator()
    
    # Simulate local training
    num_samples = len(local_data)
    
    # Generate mock gradients (in practice, these would come from actual training)
    model_weights = {}
    for key in current_global_model.keys():
        model_weights[key] = current_global_model[key] + np.random.randn(*current_global_model[key].shape) * 0.01
    
    metrics = {
        "loss": np.random.random() * 0.5,
        "accuracy": 0.8 + np.random.random() * 0.15
    }
    
    return coordinator.receive_update(institution_id, model_weights, num_samples, metrics)