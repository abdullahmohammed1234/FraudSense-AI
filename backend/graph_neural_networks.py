"""
Graph Neural Networks Module.

This module provides fraud ring detection using graph-based analysis:
- Transaction graph construction
- Graph convolutional networks for node classification
- Community detection for coordinated fraud
- Graph embedding for similarity analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import hashlib
from typing import TypeVar, Generic


T = TypeVar('T')


class Node:
    """Base node class for graph."""
    
    def __init__(self, node_id: str, node_type: str):
        self.node_id = node_id
        self.node_type = node_type
        self.features: Dict[str, Any] = {}
        self.embedding: Optional[np.ndarray] = None
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return self.node_id == other.node_id


class TransactionNode(Node):
    """Transaction node in the fraud graph."""
    
    def __init__(self, transaction_id: str, amount: float, timestamp: datetime):
        super().__init__(transaction_id, "transaction")
        self.amount = amount
        self.timestamp = timestamp
        self.is_fraud = False
        self.fraud_probability = 0.0


class UserNode(Node):
    """User node in the fraud graph."""
    
    def __init__(self, user_id: str):
        super().__init__(user_id, "user")
        self.accounts: Set[str] = set()
        self.devices: Set[str] = set()
        self.transactions: Set[str] = set()
        self.risk_score = 0.0


class AccountNode(Node):
    """Account node in the fraud graph."""
    
    def __init__(self, account_id: str, account_type: str = "checking"):
        super().__init__(account_id, "account")
        self.account_type = account_type
        self.owner_id: Optional[str] = None
        self.balance = 0.0


class DeviceNode(Node):
    """Device node in the fraud graph."""
    
    def __init__(self, device_id: str):
        super().__init__(device_id, "device")
        self.fingerprint: Optional[str] = None


class Edge:
    """Edge in the fraud graph."""
    
    def __init__(self, source: str, target: str, edge_type: str, weight: float = 1.0):
        self.source = source
        self.target = target
        self.edge_type = edge_type
        self.weight = weight
        self.timestamp: Optional[datetime] = None
    
    def __hash__(self):
        return hash((self.source, self.target, self.edge_type))


class FraudGraph:
    """Fraud detection graph."""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[Edge]] = defaultdict(list)
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges[edge.source].append(edge)
        self.adjacency[edge.source].add(edge.target)
        self.adjacency[edge.target].add(edge.source)
    
    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get neighbors of a node."""
        return self.adjacency.get(node_id, set())
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def bfs(self, start: str, depth: int = 3) -> List[str]:
        """Breadth-first search from a node."""
        visited = set()
        queue = deque([(start, 0)])
        result = []
        
        while queue:
            node_id, d = queue.popleft()
            if node_id in visited or d > depth:
                continue
            
            visited.add(node_id)
            result.append(node_id)
            
            for neighbor in self.get_neighbors(node_id):
                if neighbor not in visited:
                    queue.append((neighbor, d + 1))
        
        return result
    
    def get_subgraph(self, node_ids: List[str]) -> 'FraudGraph':
        """Extract subgraph containing specified nodes."""
        subgraph = FraudGraph()
        
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
        
        for node_id in node_ids:
            for edge in self.edges.get(node_id, []):
                if edge.target in node_ids:
                    subgraph.add_edge(edge)
        
        return subgraph
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        node_types = defaultdict(int)
        for node in self.nodes.values():
            node_types[node.node_type] += 1
        
        edge_types = defaultdict(int)
        for edges in self.edges.values():
            for edge in edges:
                edge_types[edge.edge_type] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": sum(len(e) for e in self.edges.values()),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "avg_degree": sum(len(n) for n in self.adjacency.values()) / max(len(self.adjacency), 1)
        }


class FraudRingDetector:
    """Detector for coordinated fraud rings."""
    
    def __init__(self):
        self.graph = FraudGraph()
        self.ring_cache: Dict[str, List[str]] = {}
    
    def build_transaction_graph(
        self,
        transactions: List[Dict[str, Any]],
        user_transactions: Dict[str, List[str]]
    ) -> FraudGraph:
        """Build transaction graph from transaction data."""
        
        graph = FraudGraph()
        
        # Add transaction nodes
        for txn in transactions:
            node = TransactionNode(
                transaction_id=txn.get("transaction_id", ""),
                amount=txn.get("amount", 0.0),
                timestamp=datetime.fromisoformat(txn.get("timestamp", datetime.now().isoformat()))
            )
            node.is_fraud = txn.get("is_fraud", False)
            node.fraud_probability = txn.get("fraud_probability", 0.0)
            graph.add_node(node)
        
        # Add edges based on user transactions
        for user_id, txn_ids in user_transactions.items():
            # Add user node
            user_node = UserNode(user_id)
            graph.add_node(user_node)
            
            # Connect user to their transactions
            for txn_id in txn_ids:
                edge = Edge(user_id, txn_id, "conducted", 1.0)
                graph.add_edge(edge)
        
        return graph
    
    def find_fraud_rings(
        self,
        suspicious_nodes: List[str],
        min_ring_size: int = 3
    ) -> List[List[str]]:
        """Find fraud rings among suspicious nodes."""
        
        rings = []
        
        # Build connectivity graph among suspicious nodes
        suspicious_set = set(suspicious_nodes)
        connectivity: Dict[str, Set[str]] = defaultdict(set)
        
        for node_id in suspicious_nodes:
            neighbors = self.graph.get_neighbors(node_id)
            for neighbor in neighbors:
                if neighbor in suspicious_set:
                    connectivity[node_id].add(neighbor)
        
        # Find connected components
        visited = set()
        
        def dfs(node: str, component: List[str]):
            visited.add(node)
            component.append(node)
            for neighbor in connectivity[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node_id in suspicious_nodes:
            if node_id not in visited:
                component = []
                dfs(node_id, component)
                if len(component) >= min_ring_size:
                    rings.append(component)
        
        return rings
    
    def calculate_ring_risk(self, ring: List[str]) -> float:
        """Calculate risk score for a fraud ring."""
        
        if not ring:
            return 0.0
        
        fraud_count = 0
        total_prob = 0.0
        
        for node_id in ring:
            node = self.graph.get_node(node_id)
            if node and isinstance(node, TransactionNode):
                if node.is_fraud:
                    fraud_count += 1
                total_prob += node.fraud_probability
        
        # Ring risk based on fraud density and average probability
        fraud_density = fraud_count / len(ring)
        avg_prob = total_prob / len(ring)
        
        return min((fraud_density * 0.6 + avg_prob * 0.4), 1.0)
    
    def get_ring_features(self, ring: List[str]) -> Dict[str, Any]:
        """Extract features from a fraud ring."""
        
        amounts = []
        timestamps = []
        
        for node_id in ring:
            node = self.graph.get_node(node_id)
            if node and isinstance(node, TransactionNode):
                amounts.append(node.amount)
                timestamps.append(node.timestamp)
        
        return {
            "ring_size": len(ring),
            "total_amount": sum(amounts),
            "avg_amount": np.mean(amounts) if amounts else 0,
            "std_amount": np.std(amounts) if amounts else 0,
            "time_span_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600 if len(timestamps) > 1 else 0,
            "transaction_frequency": len(ring) / max((max(timestamps) - min(timestamps)).total_seconds() / 3600, 1) if len(timestamps) > 1 else 0
        }


class GraphEmbeddingEngine:
    """Engine for graph embedding and similarity."""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
    
    def simple_node_embedding(self, graph: FraudGraph) -> Dict[str, np.ndarray]:
        """Generate simple node embeddings based on graph structure."""
        
        embeddings = {}
        
        # Initialize with node features
        for node_id, node in graph.nodes.items():
            if isinstance(node, TransactionNode):
                emb = np.array([
                    node.amount / 1000.0,
                    node.fraud_probability,
                    node.timestamp.hour / 24.0,
                    float(node.is_fraud)
                ])
            elif isinstance(node, UserNode):
                emb = np.array([
                    len(node.accounts) / 10.0,
                    len(node.devices) / 5.0,
                    len(node.transactions) / 100.0,
                    node.risk_score
                ])
            else:
                emb = np.zeros(4)
            
            # Pad to embedding dimension
            if len(emb) < self.embedding_dim:
                emb = np.pad(emb, (0, self.embedding_dim - len(emb)))
            
            embeddings[node_id] = emb[:self.embedding_dim]
        
        # Apply simple graph convolution (1-hop neighborhood aggregation)
        for node_id in graph.nodes:
            neighbors = graph.get_neighbors(node_id)
            if neighbors:
                neighbor_embeddings = [embeddings[n] for n in neighbors if n in embeddings]
                if neighbor_embeddings:
                    neighbor_avg = np.mean(neighbor_embeddings, axis=0)
                    embeddings[node_id] = 0.7 * embeddings[node_id] + 0.3 * neighbor_avg
        
        return embeddings
    
    def calculate_similarity(
        self,
        node_id1: str,
        node_id2: str,
        embeddings: Dict[str, np.ndarray]
    ) -> float:
        """Calculate cosine similarity between two nodes."""
        
        if node_id1 not in embeddings or node_id2 not in embeddings:
            return 0.0
        
        emb1 = embeddings[node_id1]
        emb2 = embeddings[node_id2]
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_nodes(
        self,
        node_id: str,
        embeddings: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find most similar nodes to a given node."""
        
        if node_id not in embeddings:
            return []
        
        similarities = []
        target_emb = embeddings[node_id]
        
        for other_id, emb in embeddings.items():
            if other_id != node_id:
                sim = self.calculate_similarity(node_id, other_id, embeddings)
                similarities.append((other_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class GNNFraudDetector:
    """Main GNN-based fraud detector."""
    
    def __init__(self):
        self.ring_detector = FraudRingDetector()
        self.embedding_engine = GraphEmbeddingEngine()
    
    def analyze_graph(
        self,
        transactions: List[Dict[str, Any]],
        user_transactions: Dict[str, List[str]],
        suspicious_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Analyze transaction graph for fraud patterns."""
        
        # Build graph
        graph = self.ring_detector.build_transaction_graph(transactions, user_transactions)
        
        # Find suspicious nodes
        suspicious_nodes = []
        for node_id, node in graph.nodes.items():
            if isinstance(node, TransactionNode):
                if node.fraud_probability >= suspicious_threshold:
                    suspicious_nodes.append(node_id)
        
        # Find fraud rings
        rings = self.ring_detector.find_fraud_rings(suspicious_nodes)
        
        # Calculate ring risks
        ring_results = []
        for ring in rings:
            risk = self.ring_detector.calculate_ring_risk(ring)
            features = self.ring_detector.get_ring_features(ring)
            ring_results.append({
                "ring_nodes": ring,
                "risk_score": round(risk, 4),
                "size": len(ring),
                "features": features
            })
        
        # Generate embeddings
        embeddings = self.embedding_engine.simple_node_embedding(graph)
        
        # Find most suspicious communities
        high_risk_rings = [r for r in ring_results if r["risk_score"] > 0.6]
        
        return {
            "graph_stats": graph.get_statistics(),
            "suspicious_nodes_count": len(suspicious_nodes),
            "fraud_rings_detected": len(rings),
            "rings": ring_results,
            "high_risk_rings": high_risk_rings,
            "embedding_dimension": self.embedding_engine.embedding_dim,
            "recommendation": "investigate" if len(high_risk_rings) > 0 else "normal"
        }
    
    def analyze_node_relationships(
        self,
        node_id: str,
        transactions: List[Dict[str, Any]],
        user_transactions: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Analyze relationships for a specific node."""
        
        graph = self.ring_detector.build_transaction_graph(transactions, user_transactions)
        
        # Get node
        node = graph.get_node(node_id)
        if not node:
            return {"error": "Node not found"}
        
        # Get neighbors
        neighbors = graph.get_neighbors(node_id)
        
        # Get subgraph
        subgraph = graph.get_subgraph(list(neighbors) + [node_id])
        
        # Generate embeddings for this subgraph
        embeddings = self.embedding_engine.simple_node_embedding(subgraph)
        
        # Find similar nodes
        similar = self.embedding_engine.find_similar_nodes(node_id, embeddings, top_k=5)
        
        return {
            "node_id": node_id,
            "node_type": node.node_type,
            "neighbors_count": len(neighbors),
            "neighbors": list(neighbors)[:10],
            "similar_nodes": similar,
            "subgraph_stats": subgraph.get_statistics()
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get GNN configuration."""
        return {
            "embedding_dimension": self.embedding_engine.embedding_dim,
            "ring_detection": {
                "min_ring_size": 3,
                "suspicious_threshold": 0.5
            },
            "thresholds": {
                "high_risk_ring": 0.6,
                "investigate": 0.3
            }
        }


_global_detector: Optional[GNNFraudDetector] = None


def get_gnn_detector() -> GNNFraudDetector:
    """Get global GNN fraud detector."""
    global _global_detector
    if _global_detector is None:
        _global_detector = GNNFraudDetector()
    return _global_detector


def analyze_fraud_graph(
    transactions: List[Dict[str, Any]],
    user_transactions: Dict[str, List[str]],
    suspicious_threshold: float = 0.5
) -> Dict[str, Any]:
    """Convenience function to analyze fraud graph."""
    return get_gnn_detector().analyze_graph(transactions, user_transactions, suspicious_threshold)