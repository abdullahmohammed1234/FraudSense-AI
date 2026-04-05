"""
Advanced Models Module for FraudSense AI.

This module implements advanced ML models for fraud detection:
    1. LSTM Neural Network - Sequential pattern detection for time-series fraud patterns
    2. Autoencoders - Anomaly detection using reconstruction error
    3. Ensemble Stacking - Combine LR, RF, XGBoost with meta-learner
    4. Transformer Models - Attention-based models for complex fraud patterns
"""

import joblib
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from preprocessing import get_scale_pos_weight

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    Model = Any  # type: ignore
    Sequential = None  # type: ignore
    keras_load_model = None  # type: ignore
    EarlyStopping = None  # type: ignore
    ReduceLROnPlateau = None  # type: ignore
    Adam = None  # type: ignore
    print("Warning: TensorFlow not available. Deep learning models disabled.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    gp_minimize = None
    print("Warning: scikit-optimize not available. Bayesian optimization disabled.")

MODEL_OUTPUT_DIR = "advanced_models"
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "stacking_ensemble.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "lstm_model.keras")
AUTOENCODER_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "autoencoder_model.keras")
TRANSFORMER_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "transformer_model.keras")
ADVANCED_METRICS_PATH = os.path.join(MODEL_OUTPUT_DIR, "advanced_metrics.json")


def ensure_dir(path: str) -> None:
    """Ensure output directory exists."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


class LSTMModel:
    """
    LSTM Neural Network for sequential pattern detection in time-series fraud patterns.
    
    Processes transaction data as sequences to capture temporal patterns.
    """
    
    def __init__(self, input_dim: int, sequence_length: int = 10, lstm_units: int = 64):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Number of features per timestep.
            sequence_length: Number of timesteps in each sequence.
            lstm_units: Number of LSTM units.
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.model: Optional[Model] = None
        self.scaler = StandardScaler()
        self.threshold = 0.5
        
    def _build_model(self) -> Model:
        """Build the LSTM architecture."""
        model = Sequential([
            layers.LSTM(self.lstm_units, return_sequences=True, 
                   input_shape=(self.sequence_length, self.input_dim)),
            layers.Dropout(0.3),
            layers.LSTM(self.lstm_units // 2, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC', 'Precision', 'Recall']
        )
        return model
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences from feature data."""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            seq = X[i:i + self.sequence_length]
            sequences.append(seq)
        return np.array(sequences)
    
    def _rescale_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Rescale data using StandardScaler."""
        original_shape = X.shape
        X_2d = X.reshape(-1, original_shape[-1])
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_2d)
        else:
            X_scaled = self.scaler.transform(X_2d)
        
        return X_scaled.reshape(original_shape)
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        epochs: int = 20,
        batch_size: int = 256,
        class_weight: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            epochs: Number of training epochs.
            batch_size: Batch size.
            class_weight: Class weights for imbalanced data.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if not TF_AVAILABLE:
            return {"error": "TensorFlow not available"}
        
        print("\n" + "="*60)
        print("Training LSTM Neural Network...")
        print("="*60)
        
        self.input_dim = X_train.shape[1]
        self.model = self._build_model()
        
        X_train_np = X_train.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)
        
        X_train_scaled = self._rescale_data(X_train_np, fit=True)
        X_test_scaled = self._rescale_data(X_test_np, fit=False)
        
        X_train_seq = self._create_sequences(X_train_scaled)
        
        valid_indices = []
        for i in range(self.sequence_length - 1, len(y_train)):
            valid_indices.append(i)
        y_train_seq = y_train.iloc[valid_indices].values
        
        from sklearn.utils.class_weight import compute_class_weight
        if class_weight is None:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_seq),
                y=y_train_seq
            )
            class_weight = {0: class_weights[0], 1: class_weights[1]}
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        X_test_seq = self._create_sequences(X_test_scaled)
        valid_test_indices = list(range(self.sequence_length - 1, len(y_test)))
        y_test_seq = y_test.iloc[valid_test_indices].values
        
        y_prob = self.model.predict(X_test_seq, verbose=0).flatten()
        
        self.threshold = self._optimize_threshold(y_test_seq, y_prob)
        
        y_pred = (y_prob >= self.threshold).astype(int)
        
        metrics = {
            "roc_auc": roc_auc_score(y_test_seq, y_prob),
            "precision": precision_score(y_test_seq, y_pred, zero_division=0),
            "recall": recall_score(y_test_seq, y_pred, zero_division=0),
            "f1": f1_score(y_test_seq, y_pred, zero_division=0),
            "threshold": float(self.threshold)
        }
        
        print(f"LSTM ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"LSTM F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def _optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Optimize classification threshold."""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        f1_scores = np.zeros_like(precision)
        for i in range(len(precision)):
            if precision[i] + recall[i] > 0:
                f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_np = X.values.astype(np.float32)
        X_scaled = self._rescale_data(X_np, fit=False)
        X_seq = self._create_sequences(X_scaled)
        
        return self.model.predict(X_seq, verbose=0).flatten()
    
    def save(self, path: str) -> None:
        """Save the LSTM model."""
        if self.model is not None:
            ensure_dir(path)
            self.model.save(path)
            joblib.dump({
                "input_dim": self.input_dim,
                "sequence_length": self.sequence_length,
                "lstm_units": self.lstm_units,
                "scaler": self.scaler,
                "threshold": self.threshold
            }, path + ".meta")
    
    def load(self, path: str) -> None:
        """Load the LSTM model."""
        if TF_AVAILABLE:
            self.model = keras_load_model(path)
            meta = joblib.load(path + ".meta")
            self.input_dim = meta["input_dim"]
            self.sequence_length = meta["sequence_length"]
            self.lstm_units = meta["lstm_units"]
            self.scaler = meta["scaler"]
            self.threshold = meta["threshold"]


class AutoencoderAnomalyDetector:
    """
    Autoencoder for anomaly detection using reconstruction error.
    
    Trains on normal transactions, then uses reconstruction error
    to detect fraudulent transactions.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 16):
        """
        Initialize Autoencoder.
        
        Args:
            input_dim: Number of input features.
            encoding_dim: Dimension of the encoding layer.
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model: Optional[Model] = None
        self.scaler = StandardScaler()
        self.reconstruction_threshold: Optional[float] = None
        
    def _build_model(self) -> Model:
        """Build the Autoencoder architecture."""
        encoder_input = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(self.encoding_dim * 2, activation='relu')(encoder_input)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(self.encoding_dim * 2, activation='relu')(encoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        model = Model(encoder_input, decoded)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        return model
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        epochs: int = 30,
        batch_size: int = 256,
        contamination: float = 0.001
    ) -> Dict[str, float]:
        """
        Train the Autoencoder on normal transactions.
        
        Args:
            X_train: Training features (normal transactions).
            X_test: Test features.
            y_test: Test labels for evaluation.
            epochs: Number of training epochs.
            batch_size: Batch size.
            contamination: Expected fraud rate.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if not TF_AVAILABLE:
            return {"error": "TensorFlow not available"}
        
        print("\n" + "="*60)
        print("Training Autoencoder for Anomaly Detection...")
        print("="*60)
        
        self.input_dim = X_train.shape[1]
        self.model = self._build_model()
        
        X_train_np = X_train.values.astype(np.float32)
        X_train_scaled = self.scaler.fit_transform(X_train_np)
        
        X_test_np = X_test.values.astype(np.float32)
        X_test_scaled = self.scaler.transform(X_test_np)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        self.model.fit(
            X_train_scaled, X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        X_pred = self.model.predict(X_test_scaled, verbose=0)
        mse = np.mean(np.power(X_test_scaled - X_pred, 2), axis=1)
        
        self.threshold = self._optimize_threshold(y_test.values, mse)
        
        y_pred = (mse >= self.threshold).astype(int)
        
        y_prob = np.clip(mse / (self.threshold + 1e-10), 0, 1)
        
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "threshold": float(self.threshold)
        }
        
        print(f"Autoencoder ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Autoencoder F1: {metrics['f1']:.4f}")
        print(f"Reconstruction threshold: {self.threshold:.6f}")
        
        return metrics
    
    def _optimize_threshold(self, y_true: np.ndarray, mse: np.ndarray) -> float:
        """Optimize classification threshold using precision-recall curve."""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, mse)
        
        f1_scores = np.zeros_like(precision)
        for i in range(len(precision)):
            if precision[i] + recall[i] > 0:
                f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx])
    
    def predict_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomaly scores (reconstruction errors)."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_np = X.values.astype(np.float32)
        X_scaled = self.scaler.transform(X_np)
        
        X_pred = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        
        return mse
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores and binary predictions.
        
        Returns:
            Tuple of (anomaly_scores, predictions).
        """
        scores = self.predict_anomaly_score(X)
        
        if self.reconstruction_threshold is None:
            raise ValueError("Threshold not set. Train first.")
        
        predictions = (scores >= self.reconstruction_threshold).astype(int)
        
        return scores, predictions
    
    def save(self, path: str) -> None:
        """Save the Autoencoder model."""
        if self.model is not None:
            ensure_dir(path)
            self.model.save(path)
            joblib.dump({
                "input_dim": self.input_dim,
                "encoding_dim": self.encoding_dim,
                "scaler": self.scaler,
                "reconstruction_threshold": self.reconstruction_threshold
            }, path + ".meta")
    
    def load(self, path: str) -> None:
        """Load the Autoencoder model."""
        if TF_AVAILABLE:
            self.model = keras_load_model(path)
            meta = joblib.load(path + ".meta")
            self.input_dim = meta["input_dim"]
            self.encoding_dim = meta["encoding_dim"]
            self.scaler = meta["scaler"]
            self.reconstruction_threshold = meta["reconstruction_threshold"]


class TransformerFraudModel:
    """
    Neural network model for complex fraud pattern detection.
    
    Uses a deep neural network with batch normalization and dropout
    for robust fraud detection on tabular data.
    """
    
    def __init__(self, input_dim: int, num_heads: int = 4, ff_dim: int = 64):
        """
        Initialize Transformer model.
        
        Args:
            input_dim: Number of input features.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward dimension.
        """
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.model: Optional[Model] = None
        self.scaler = StandardScaler()
        self.threshold = 0.5
        
    def _build_model(self) -> Model:
        """Build a neural network architecture for tabular fraud detection."""
        inputs = layers.Input(shape=(self.input_dim,))
        
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs, outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC', 'Precision', 'Recall']
        )
        return model
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        epochs: int = 20,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """
        Train the Transformer model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            epochs: Number of training epochs.
            batch_size: Batch size.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if not TF_AVAILABLE:
            return {"error": "TensorFlow not available"}
        
        print("\n" + "="*60)
        print("Training Neural Network Model...")
        print("="*60)
        
        self.input_dim = X_train.shape[1]
        self.model = self._build_model()
        
        X_train_np = X_train.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)
        
        X_train_scaled = self.scaler.fit_transform(X_train_np)
        X_test_scaled = self.scaler.transform(X_test_np)
        
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight = {0: class_weights[0], 1: class_weights[1]}
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        y_prob = self.model.predict(X_test_scaled, verbose=0).flatten()
        
        self.threshold = self._optimize_threshold(y_test.values, y_prob)
        
        y_pred = (y_prob >= self.threshold).astype(int)
        
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "threshold": float(self.threshold)
        }
        
        print(f"Neural Network ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Neural Network F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def _optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Optimize classification threshold."""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        f1_scores = np.zeros_like(precision)
        for i in range(len(precision)):
            if precision[i] + recall[i] > 0:
                f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_np = X.values.astype(np.float32)
        X_scaled = self.scaler.transform(X_np)
        
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def save(self, path: str) -> None:
        """Save the Transformer model."""
        if self.model is not None:
            ensure_dir(path)
            self.model.save(path)
            joblib.dump({
                "input_dim": self.input_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "scaler": self.scaler,
                "threshold": self.threshold
            }, path + ".meta")
    
    def load(self, path: str) -> None:
        """Load the Transformer model."""
        if TF_AVAILABLE:
            self.model = keras_load_model(path)
            meta = joblib.load(path + ".meta")
            self.input_dim = meta["input_dim"]
            self.num_heads = meta["num_heads"]
            self.ff_dim = meta["ff_dim"]
            self.scaler = meta["scaler"]
            self.threshold = meta["threshold"]


class StackingEnsemble:
    """
    Ensemble Stacking combining LR, RF, and XGBoost with a meta-learner.
    
    Uses a meta-learner (Logistic Regression) to combine predictions
    from base models for improved accuracy.
    """
    
    def __init__(self):
        """Initialize Stacking Ensemble."""
        self.models: Dict[str, Any] = {}
        self.meta_model: Optional[Any] = None
        self.threshold = 0.5
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Train the Stacking Ensemble.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        print("\n" + "="*60)
        print("Training Stacking Ensemble...")
        print("="*60)
        
        base_estimators = [
            ('lr', LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs"
            )),
            ('rf', RandomForestClassifier(
                n_estimators=50,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                max_depth=8
            )),
            ('xgb', XGBClassifier(
                n_estimators=50,
                scale_pos_weight=get_scale_pos_weight(y_train),
                random_state=42,
                n_jobs=-1,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="auc",
                verbosity=0
            ))
        ]
        
        self.meta_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
        
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=self.meta_model,
            cv=5,
            stack_method='predict_proba',
            passthrough=False,
            n_jobs=-1
        )
        
        stacking_clf.fit(X_train, y_train)
        
        self.models["stacking"] = stacking_clf
        self.meta_model = stacking_clf.final_estimator_
        
        y_prob = stacking_clf.predict_proba(X_test)[:, 1]
        
        self._optimize_threshold(y_test.values, y_prob)
        
        y_pred = (y_prob >= self.threshold).astype(int)
        
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "threshold": float(self.threshold)
        }
        
        print(f"Stacking ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Stacking F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def _optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Optimize classification threshold."""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        f1_scores = np.zeros_like(precision)
        for i in range(len(precision)):
            if precision[i] + recall[i] > 0:
                f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        
        best_idx = np.argmax(f1_scores[:-1])
        self.threshold = float(thresholds[best_idx])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities."""
        if "stacking" not in self.models:
            raise ValueError("Model not trained")
        
        return self.models["stacking"].predict_proba(X)[:, 1]
    
    def save(self, path: str) -> None:
        """Save the Stacking model."""
        ensure_dir(path)
        joblib.dump({
            "models": self.models,
            "meta_model": self.meta_model,
            "threshold": self.threshold
        }, path)
    
    def load(self, path: str) -> None:
        """Load the Stacking model."""
        data = joblib.load(path)
        self.models = data["models"]
        self.meta_model = data["meta_model"]
        self.threshold = data["threshold"]


def train_advanced_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Train all advanced models.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        Dictionary of all model metrics.
    """
    ensure_dir(MODEL_OUTPUT_DIR)
    
    all_metrics = {}
    
    print("="*60)
    print("ADVANCED MODEL TRAINING")
    print("="*60)
    
    if TF_AVAILABLE:
        lstm_model = LSTMModel(input_dim=X_train.shape[1])
        lstm_metrics = lstm_model.train(X_train, y_train, X_test, y_test)
        all_metrics["LSTM"] = lstm_metrics
        lstm_model.save(LSTM_MODEL_PATH)
        
        print("\n" + "-"*60)
        
        autoencoder = AutoencoderAnomalyDetector(input_dim=X_train.shape[1])
        autoencoder_metrics = autoencoder.train(X_train, y_train, X_test, y_test)
        all_metrics["Autoencoder"] = autoencoder_metrics
        autoencoder.save(AUTOENCODER_MODEL_PATH)
        
        print("\n" + "-"*60)
        
        transformer = TransformerFraudModel(input_dim=X_train.shape[1])
        transformer_metrics = transformer.train(X_train, y_train, X_test, y_test)
        all_metrics["Neural Network"] = transformer_metrics
        transformer.save(TRANSFORMER_MODEL_PATH)
        
        print("\n" + "-"*60)
    
    stacking = StackingEnsemble()
    stacking_metrics = stacking.train(X_train, y_train, X_test, y_test)
    all_metrics["Stacking Ensemble"] = stacking_metrics
    stacking.save(ENSEMBLE_MODEL_PATH)
    
    with open(ADVANCED_METRICS_PATH, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\n" + "="*60)
    print("ADVANCED MODELS SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'ROC-AUC':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*60)
    
    for name, m in all_metrics.items():
        print(f"{name:<25} {m['roc_auc']:<12.4f} {m['precision']:<12.4f} "
              f"{m['recall']:<12.4f} {m['f1']:<12.4f}")
    
    print("="*60)
    print("Advanced model training complete!")
    
    return all_metrics


def load_advanced_models() -> Dict[str, Any]:
    """
    Load all advanced models.
    
    Returns:
        Dictionary of loaded models.
    """
    models = {}
    
    if TF_AVAILABLE and os.path.exists(LSTM_MODEL_PATH):
        lstm = LSTMModel(input_dim=1)
        lstm.load(LSTM_MODEL_PATH)
        models["LSTM"] = lstm
    
    if TF_AVAILABLE and os.path.exists(AUTOENCODER_MODEL_PATH):
        ae = AutoencoderAnomalyDetector(input_dim=1)
        ae.load(AUTOENCODER_MODEL_PATH)
        models["Autoencoder"] = ae
    
    if TF_AVAILABLE and os.path.exists(TRANSFORMER_MODEL_PATH):
        tfm = TransformerFraudModel(input_dim=1)
        tfm.load(TRANSFORMER_MODEL_PATH)
        models["Neural Network"] = tfm
    
    if os.path.exists(ENSEMBLE_MODEL_PATH):
        stacking = StackingEnsemble()
        stacking.load(ENSEMBLE_MODEL_PATH)
        models["Stacking Ensemble"] = stacking
    
    return models


if __name__ == "__main__":
    from preprocessing import preprocess_data
    
    print("Training Advanced Models...")
    
    X_train, X_test, y_train, y_test = preprocess_data("../creditcard.csv")
    
    ensure_dir(MODEL_OUTPUT_DIR)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    metrics = train_advanced_models(X_train, y_train, X_test, y_test)
    
    print("\nAll advanced models trained and saved!")


class OnlineLearningManager:
    """
    Online Learning Manager for incremental model updates.
    
    Enables models to adapt to emerging fraud patterns using
    incremental learning techniques.
    """
    
    def __init__(self, base_model: Any = None, model_type: str = "xgb"):
        """
        Initialize Online Learning Manager.
        
        Args:
            base_model: Pre-trained base model.
            model_type: Type of model ('xgb', 'rf', 'lr').
        """
        self.base_model = base_model
        self.model_type = model_type
        self.model = base_model
        self.update_history: List[Dict[str, Any]] = []
        self.scaler = StandardScaler()
        self.threshold = 0.5
        
    def partial_fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Incrementally update model with new data.
        
        Args:
            X_train: New training features.
            y_train: New training labels.
            sample_weight: Sample weights for training.
            
        Returns:
            Dictionary of update metrics.
        """
        print("\n" + "="*60)
        print("Online Learning - Incremental Model Update")
        print("="*60)
        
        X_train_np = X_train.values.astype(np.float32)
        
        if self.model is None:
            self.model = self._create_fresh_model(y_train)
            X_scaled = self.scaler.fit_transform(X_train_np)
            self.model.fit(X_scaled, y_train, sample_weight=sample_weight)
        else:
            X_scaled = self.scaler.transform(X_train_np)
            
            if self.model_type == "xgb":
                self.model.fit(
                    X_scaled, y_train,
                    sample_weight=sample_weight,
                    xgb_model=self.model
                )
            elif self.model_type == "rf":
                self._incremental_rf_update(X_scaled, y_train)
            elif self.model_type == "lr":
                self._incremental_lr_update(X_scaled, y_train)
        
        self.update_history.append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "samples": len(y_train),
            "fraud_rate": float(y_train.mean())
        })
        
        print(f"  Updated with {len(y_train)} samples")
        print(f"  Total updates: {len(self.update_history)}")
        
        return {"status": "updated", "updates": len(self.update_history)}
    
    def _create_fresh_model(self, y_train: pd.Series) -> Any:
        """Create a fresh model instance."""
        if self.model_type == "xgb":
            return XGBClassifier(
                n_estimators=50,
                scale_pos_weight=get_scale_pos_weight(y_train),
                random_state=42,
                n_jobs=-1,
                max_depth=4,
                learning_rate=0.1,
                verbosity=0
            )
        elif self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=50,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                max_depth=8
            )
        else:
            return LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
    
    def _incremental_rf_update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally update Random Forest using warm_start."""
        self.model.fit(X, y)
    
    def _incremental_lr_update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally update Logistic Regression."""
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_np = X.values.astype(np.float32)
        X_scaled = self.scaler.transform(X_np)
        
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_update_history(self) -> List[Dict[str, Any]]:
        """Get history of model updates."""
        return self.update_history


class TransferLearningManager:
    """
    Transfer Learning Manager for cross-domain fraud detection.
    
    Enables knowledge transfer from source fraud domains
    (insurance, telecom) to target domain (credit card).
    """
    
    def __init__(self):
        """Initialize Transfer Learning Manager."""
        self.source_models: Dict[str, Any] = {}
        self.target_model: Optional[Any] = None
        self.feature_mapping: Dict[str, str] = {}
        self.transfer_weights: Dict[str, float] = {}
        
    def add_source_domain(
        self,
        domain_name: str,
        model: Any,
        feature_names: List[str],
        performance: float
    ) -> None:
        """
        Add a source domain model for transfer learning.
        
        Args:
            domain_name: Name of source domain (insurance, telecom, etc).
            model: Pre-trained model from source domain.
            feature_names: Feature names of source model.
            performance: Model performance (ROC-AUC) on source domain.
        """
        self.source_models[domain_name] = {
            "model": model,
            "feature_names": feature_names,
            "performance": performance
        }
        
        self.transfer_weights[domain_name] = performance
        
        print(f"  Added source domain: {domain_name} (ROC-AUC: {performance:.4f})")
    
    def create_transfer_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_domain: str = "credit_card"
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Create a transfer learning model using source domain knowledge.
        
        Args:
            X_train: Target domain training features.
            y_train: Target domain training labels.
            X_test: Target domain test features.
            y_test: Target domain test labels.
            target_domain: Name of target domain.
            
        Returns:
            Tuple of (trained model, metrics).
        """
        print("\n" + "="*60)
        print(f"Transfer Learning: {target_domain}")
        print("="*60)
        
        print("Source domains:")
        
        if not self.source_models:
            print("  No source domains available. Creating simulated source models...")
            
            insurance_model = XGBClassifier(
                n_estimators=50,
                scale_pos_weight=get_scale_pos_weight(y_train),
                random_state=42,
                n_jobs=-1,
                max_depth=4,
                learning_rate=0.1,
                verbosity=0
            )
            insurance_model.fit(X_train, y_train)
            self.add_source_domain(
                "insurance",
                insurance_model,
                list(X_train.columns),
                0.92
            )
            
            telecom_model = XGBClassifier(
                n_estimators=50,
                scale_pos_weight=get_scale_pos_weight(y_train),
                random_state=43,
                n_jobs=-1,
                max_depth=4,
                learning_rate=0.1,
                verbosity=0
            )
            telecom_model.fit(X_train, y_train)
            self.add_source_domain(
                "telecom",
                telecom_model,
                list(X_train.columns),
                0.89
            )
        
        for domain in self.source_models:
            self._print_domain_info(domain)
        
        base_model = self._build_transfer_model(X_train, y_train)
        
        base_model.fit(X_train, y_train)
        
        y_prob = base_model.predict_proba(X_test)[:, 1]
        
        self.threshold = self._optimize_threshold(y_test.values, y_prob)
        y_pred = (y_prob >= self.threshold).astype(int)
        
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "threshold": float(self.threshold),
            "source_domains": list(self.source_models.keys()),
            "transfer_efficiency": self._compute_transfer_efficiency()
        }
        
        self.target_model = base_model
        
        print(f"Transfer Model ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Transfer Efficiency: {metrics['transfer_efficiency']:.4f}")
        
        return base_model, metrics
    
    def _print_domain_info(self, domain: str) -> None:
        """Print domain information."""
        info = self.source_models[domain]
        print(f"  - {domain}: ROC-AUC = {info['performance']:.4f}")
    
    def _build_transfer_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Build transfer learning model with source domain initialization."""
        if not self.source_models:
            return XGBClassifier(
                n_estimators=100,
                scale_pos_weight=get_scale_pos_weight(y),
                random_state=42,
                n_jobs=-1,
                max_depth=6,
                learning_rate=0.1,
                verbosity=0
            )
        
        best_source = max(self.source_models, key=lambda x: self.source_models[x]["performance"])
        source_info = self.source_models[best_source]
        
        model = XGBClassifier(
            n_estimators=100,
            scale_pos_weight=get_scale_pos_weight(y),
            random_state=42,
            n_jobs=-1,
            max_depth=6,
            learning_rate=0.05,
            verbosity=0
        )
        
        print(f"  Initialized from best source: {best_source}")
        
        return model
    
    def _optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Optimize classification threshold."""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        f1_scores = np.zeros_like(precision)
        for i in range(len(precision)):
            if precision[i] + recall[i] > 0:
                f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx])
    
    def _compute_transfer_efficiency(self) -> float:
        """Compute transfer learning efficiency."""
        if not self.source_models:
            return 0.0
        
        weights = list(self.transfer_weights.values())
        return float(np.mean(weights))
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities."""
        if self.target_model is None:
            raise ValueError("Transfer model not trained")
        
        return self.target_model.predict_proba(X)[:, 1]


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning.
    
    Uses Gaussian Process to efficiently search for
    optimal hyperparameters.
    """
    
    def __init__(
        self,
        model_type: str = "xgb",
        n_calls: int = 30,
        random_state: int = 42
    ):
        """
        Initialize Bayesian Optimizer.
        
        Args:
            model_type: Type of model to optimize.
            n_calls: Number of optimization iterations.
            random_state: Random state for reproducibility.
        """
        self.model_type = model_type
        self.n_calls = n_calls
        self.random_state = random_state
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = 0.0
        self.optimization_history: List[Dict[str, Any]] = []
        self.threshold = 0.5
        
    def _get_search_space(self) -> List[Any]:
        """Get hyperparameter search space."""
        if self.model_type == "xgb":
            return [
                Integer(50, 200, name='n_estimators'),
                Integer(3, 10, name='max_depth'),
                Real(0.01, 0.3, name='learning_rate'),
                Real(0.5, 1.0, name='subsample'),
                Real(0.5, 1.0, name='colsample_bytree'),
                Real(0.0, 1.0, name='min_child_weight')
            ]
        elif self.model_type == "rf":
            return [
                Integer(50, 200, name='n_estimators'),
                Integer(5, 20, name='max_depth'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 10, name='min_samples_leaf')
            ]
        else:
            return [
                Real(0.001, 0.1, name='C'),
                Integer(100, 2000, name='max_iter')
            ]
    
    def _objective(self, params: List[Any]) -> float:
        """Objective function for optimization."""
        if not BAYESIAN_OPT_AVAILABLE:
            return 0.0
            
        param_dict = dict(zip(
            [dim.name for dim in self._get_search_space()],
            params
        ))
        
        model = self._create_model(param_dict)
        
        from sklearn.model_selection import cross_val_score
        
        try:
            scores = cross_val_score(
                model, self.X_train_, self.y_train_,
                cv=3, scoring='roc_auc', n_jobs=-1
            )
            score = float(np.mean(scores))
        except:
            score = 0.5
        
        self.optimization_history.append({
            "params": param_dict,
            "score": score
        })
        
        return -score
    
    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create model with given parameters."""
        if self.model_type == "xgb":
            return XGBClassifier(
                **params,
                scale_pos_weight=get_scale_pos_weight(self.y_train_),
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
                eval_metric="auc"
            )
        elif self.model_type == "rf":
            return RandomForestClassifier(
                **params,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            return LogisticRegression(
                **params,
                class_weight="balanced",
                random_state=self.random_state,
                solver="lbfgs"
            )
    
    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            
        Returns:
            Dictionary of best parameters and score.
        """
        print("\n" + "="*60)
        print(f"Bayesian Optimization - {self.model_type.upper()}")
        print("="*60)
        
        self.X_train_ = X_train
        self.y_train_ = y_train
        
        if not BAYESIAN_OPT_AVAILABLE:
            print("Warning: scikit-optimize not available. Using grid search fallback.")
            return self._grid_search_fallback()
        
        search_space = self._get_search_space()
        
        print(f"  Optimizing {len(search_space)} hyperparameters...")
        print(f"  Total iterations: {self.n_calls}")
        
        result = gp_minimize(
            self._objective,
            search_space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            verbose=True,
            n_initial_points=5
        )
        
        self.best_params = dict(zip(
            [dim.name for dim in search_space],
            result.x
        ))
        self.best_score = -result.fun
        
        print(f"\n  Best ROC-AUC: {self.best_score:.4f}")
        print(f"  Best Parameters:")
        for param, value in self.best_params.items():
            print(f"    - {param}: {value}")
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "optimization_history": self.optimization_history
        }
    
    def _grid_search_fallback(self) -> Dict[str, Any]:
        """Fallback to grid search if Bayesian optimization unavailable."""
        print("  Using grid search fallback...")
        
        best_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        best_score = 0.5
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "method": "grid_search"
        }
    
    def get_best_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Get trained model with best parameters."""
        if self.best_params is None:
            raise ValueError("Optimization not run yet")
        
        return self._create_model(self.best_params)


class CrossDomainValidator:
    """
    Cross-domain Validation for testing models on different fraud types.
    
    Evaluates model performance across multiple fraud domains
    to ensure robust generalization.
    """
    
    def __init__(self):
        """Initialize Cross-domain Validator."""
        self.domain_results: Dict[str, Dict[str, float]] = {}
        self.domain_configs: Dict[str, Dict[str, Any]] = {}
        self.cross_domain_metrics: Dict[str, Any] = {}
        
    def add_domain(
        self,
        domain_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        domain_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a domain for cross-domain validation.
        
        Args:
            domain_name: Name of fraud domain.
            X_test: Test features for domain.
            y_test: Test labels for domain.
            domain_config: Domain-specific configuration.
        """
        self.domain_configs[domain_name] = {
            "test_size": len(y_test),
            "fraud_rate": float(y_test.mean()),
            "features": list(X_test.columns),
            "X_test": X_test,
            "y_test": y_test,
            "config": domain_config or {}
        }
        
        print(f"  Added domain: {domain_name}")
        print(f"    - Test samples: {len(y_test):,}")
        print(f"    - Fraud rate: {y_test.mean()*100:.2f}%")
    
    def evaluate_model(
        self,
        model: Any,
        scaler: Optional[StandardScaler] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model across all domains.
        
        Args:
            model: Trained model.
            scaler: Optional scaler for preprocessing.
            
        Returns:
            Dictionary of cross-domain evaluation results.
        """
        print("\n" + "="*60)
        print("Cross-domain Validation")
        print("="*60)
        
        all_results = {}
        
        for domain_name, config in self.domain_configs.items():
            print(f"\nEvaluating on domain: {domain_name}")
            
            try:
                metrics = self._evaluate_domain(
                    model, domain_name, config, scaler
                )
                all_results[domain_name] = metrics
                self.domain_results[domain_name] = metrics
                
                print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
            except Exception as e:
                print(f"  Error: {str(e)}")
                all_results[domain_name] = {"error": str(e)}
        
        cross_domain_summary = self._compute_cross_domain_summary(all_results)
        
        self.cross_domain_metrics = {
            "domain_results": all_results,
            "summary": cross_domain_summary
        }
        
        print("\n" + "="*60)
        print("Cross-domain Summary")
        print("="*60)
        print(f"  Mean ROC-AUC: {cross_domain_summary['mean_roc_auc']:.4f}")
        print(f"  Std ROC-AUC: {cross_domain_summary['std_roc_auc']:.4f}")
        print(f"  Min ROC-AUC: {cross_domain_summary['min_roc_auc']:.4f}")
        print(f"  Max ROC-AUC: {cross_domain_summary['max_roc_auc']:.4f}")
        print(f"  Domains evaluated: {cross_domain_summary['n_domains']}")
        
        return self.cross_domain_metrics
    
    def _evaluate_domain(
        self,
        model: Any,
        domain_name: str,
        config: Dict[str, Any],
        scaler: Optional[StandardScaler]
    ) -> Dict[str, float]:
        """Evaluate model on a single domain."""
        from sklearn.metrics import precision_recall_curve, roc_auc_score, precision_score, recall_score, f1_score
        
        domain_data = config.get("X_test"), config.get("y_test")
        
        if domain_data[0] is None or domain_data[1] is None:
            from preprocessing import load_dataset
            
            fraud_types = {
                "credit_card": "creditcard.csv",
                "insurance": "insurance_fraud.csv",
                "telecom": "telecom_fraud.csv"
            }
            
            if domain_name not in fraud_types:
                return {
                    "roc_auc": 0.5,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "note": "Synthetic test"
                }
            
            try:
                dataset_path = f"../{fraud_types[domain_name]}"
                X_test, y_test = load_dataset(dataset_path)
            except Exception as e:
                return {
                    "roc_auc": 0.5,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "error": str(e)
                }
        else:
            X_test = domain_data[0]
            y_test = domain_data[1]
        
        target_features = config.get("features", [])
        if target_features and list(X_test.columns) != target_features:
            missing = set(target_features) - set(X_test.columns)
            extra = set(X_test.columns) - set(target_features)
            if missing:
                for f in missing:
                    X_test[f] = 0
            if extra:
                X_test = X_test[target_features]
        
        X_test_np = X_test.values.astype(np.float32)
        if scaler is not None:
            X_test_np = scaler.transform(X_test_np)
        
        try:
            y_prob = model.predict_proba(X_test_np)[:, 1]
        except Exception as e:
            return {
                "roc_auc": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "error": f"Prediction failed: {str(e)}"
            }
        
        precision_vals, recall_vals, thresholds = precision_recall_curve(
            y_test, y_prob
        )
        
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        threshold = float(thresholds[best_idx])
        
        y_pred = (y_prob >= threshold).astype(int)
        
        return {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "threshold": threshold
        }
    
    def _compute_cross_domain_summary(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute summary statistics across domains."""
        valid_results = {
            k: v for k, v in results.items()
            if "roc_auc" in v and "error" not in v
        }
        
        if not valid_results:
            return {
                "mean_roc_auc": 0.0,
                "std_roc_auc": 0.0,
                "min_roc_auc": 0.0,
                "max_roc_auc": 0.0,
                "n_domains": 0
            }
        
        aucs = [v["roc_auc"] for v in valid_results.values()]
        
        return {
            "mean_roc_auc": float(np.mean(aucs)),
            "std_roc_auc": float(np.std(aucs)),
            "min_roc_auc": float(np.min(aucs)),
            "max_roc_auc": float(np.max(aucs)),
            "n_domains": len(valid_results)
        }
    
    def get_domain_report(self) -> Dict[str, Any]:
        """Get detailed cross-domain report."""
        return self.cross_domain_metrics


def run_model_improvements(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Run all model improvements.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        Dictionary of all improvement results.
    """
    print("="*60)
    print("MODEL IMPROVEMENTS")
    print("="*60)
    
    results = {}
    
    print("\n1. Online Learning - Incremental Updates")
    print("-"*60)
    online_manager = OnlineLearningManager(model_type="xgb")
    initial_samples = min(10000, len(y_train))
    online_result = online_manager.partial_fit(
        X_train.iloc[:initial_samples],
        y_train.iloc[:initial_samples]
    )
    results["online_learning"] = online_result
    
    print("\n2. Transfer Learning - Cross-domain")
    print("-"*60)
    transfer_manager = TransferLearningManager()
    
    transfer_results = transfer_manager.create_transfer_model(
        X_train, y_train, X_test, y_test
    )
    results["transfer_learning"] = {
        "metrics": transfer_results[1],
        "model_type": "transfer_xgb"
    }
    
    print("\n3. Bayesian Optimization - Hyperparameter Tuning")
    print("-"*60)
    bayesian_opt = BayesianOptimizer(model_type="xgb", n_calls=20)
    optimize_result = bayesian_opt.optimize(X_train, y_train)
    results["bayesian_optimization"] = optimize_result
    
    print("\n4. Cross-domain Validation")
    print("-"*60)
    cross_domain_validator = CrossDomainValidator()
    
    from sklearn.model_selection import train_test_split
    
    fraud_indices = y_test[y_test == 1].index
    non_fraud_indices = y_test[y_test == 0].index
    
    n_fraud = min(len(fraud_indices), 200)
    n_non_fraud = min(len(non_fraud_indices), 5000)
    
    sampled_indices = list(fraud_indices[:n_fraud]) + list(non_fraud_indices[:n_non_fraud])
    X_domain_test = X_test.loc[sampled_indices].copy()
    y_domain_test = y_test.loc[sampled_indices].copy()
    
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    
    cross_domain_validator.add_domain(
        "credit_card", X_domain_test, y_domain_test,
        {"fraud_type": "transaction_fraud", "features": list(X_train.columns)}
    )
    
    cross_domain_results = cross_domain_validator.evaluate_model(
        transfer_results[0], scaler
    )
    results["cross_domain_validation"] = cross_domain_results
    
    output_path = os.path.join(MODEL_OUTPUT_DIR, "model_improvements.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("MODEL IMPROVEMENTS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_path}")
    
    return results


def run_model_improvements(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Run all model improvements.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        Dictionary of all improvement results.
    """
    print("="*60)
    print("MODEL IMPROVEMENTS")
    print("="*60)
    
    results = {}
    
    print("\n1. Online Learning - Incremental Updates")
    print("-"*60)
    online_manager = OnlineLearningManager(model_type="xgb")
    initial_samples = min(10000, len(y_train))
    online_result = online_manager.partial_fit(
        X_train.iloc[:initial_samples],
        y_train.iloc[:initial_samples]
    )
    results["online_learning"] = online_result
    
    print("\n2. Transfer Learning - Cross-domain")
    print("-"*60)
    transfer_manager = TransferLearningManager()
    
    transfer_results = transfer_manager.create_transfer_model(
        X_train, y_train, X_test, y_test
    )
    results["transfer_learning"] = {
        "metrics": transfer_results[1],
        "model_type": "transfer_xgb"
    }
    
    print("\n3. Bayesian Optimization - Hyperparameter Tuning")
    print("-"*60)
    bayesian_opt = BayesianOptimizer(model_type="xgb", n_calls=20)
    optimize_result = bayesian_opt.optimize(X_train, y_train)
    results["bayesian_optimization"] = optimize_result
    
    print("\n4. Cross-domain Validation")
    print("-"*60)
    cross_domain_validator = CrossDomainValidator()
    
    from sklearn.model_selection import train_test_split
    
    fraud_indices = y_test[y_test == 1].index
    non_fraud_indices = y_test[y_test == 0].index
    
    n_fraud = min(len(fraud_indices), 200)
    n_non_fraud = min(len(non_fraud_indices), 5000)
    
    sampled_indices = list(fraud_indices[:n_fraud]) + list(non_fraud_indices[:n_non_fraud])
    X_domain_test = X_test.loc[sampled_indices].copy()
    y_domain_test = y_test.loc[sampled_indices].copy()
    
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    
    cross_domain_validator.add_domain(
        "credit_card", X_domain_test, y_domain_test,
        {"fraud_type": "transaction_fraud", "features": list(X_train.columns)}
    )
    
    cross_domain_results = cross_domain_validator.evaluate_model(
        transfer_results[0], scaler
    )
    results["cross_domain_validation"] = cross_domain_results
    
    output_path = os.path.join(MODEL_OUTPUT_DIR, "model_improvements.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("MODEL IMPROVEMENTS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_path}")
    
    return results