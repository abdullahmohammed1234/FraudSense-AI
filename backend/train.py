"""
Training module for FraudSense AI.

This module trains multiple ML models (Logistic Regression, Random Forest, XGBoost),
evaluates them, selects the best model based on ROC-AUC, and saves the model along
with a SHAP explainer.

Features:
    - Threshold optimization using precision-recall curve and F1 score
    - Stratified cross-validation (5-fold)
    - Metric visualization (ROC curve, Confusion Matrix, PR curve)
    - Global SHAP feature importance
"""

import joblib
import shap
import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    auc
)
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_data, get_class_weights, get_scale_pos_weight, load_dataset


# Default dataset path
DEFAULT_DATASET_PATH = "../creditcard.csv"
MODEL_PATH = "model.pkl"
EXPLAINER_PATH = "explainer.pkl"
ANOMALY_MODEL_PATH = "anomaly_model.pkl"
TRAINING_STATS_PATH = "training_stats.json"
METRICS_DIR = "metrics"
GLOBAL_FEATURE_IMPORTANCE_PATH = "global_feature_importance.json"
MODEL_CONFIG_PATH = "model_config.json"


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        threshold: Classification threshold.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }


def optimize_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Optimize classification threshold using precision-recall curve.
    
    Finds the threshold that maximizes F1 score.
    
    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        
    Returns:
        Tuple of (best_threshold, precision_array, recall_array).
    """
    print("\n" + "="*60)
    print("Optimizing Classification Threshold...")
    print("="*60)
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Compute F1 for each threshold
    # Handle the last element where precision + recall might be 0
    f1_scores = np.zeros_like(precision)
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:
            f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    
    # Find best threshold (excluding the last element which is just a boundary)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Precision at best threshold: {precision[best_idx]:.4f}")
    print(f"Recall at best threshold: {recall[best_idx]:.4f}")
    
    return best_threshold, precision, recall


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_cv: bool = False
) -> Tuple[Any, Dict[str, float], float]:
    """
    Train Logistic Regression with balanced class weights.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        use_cv: Whether to perform cross-validation.
        
    Returns:
        Tuple of (trained model, metrics dictionary, optimized threshold).
    """
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    print("="*60)
    
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Optimize threshold
    best_threshold, _, _ = optimize_threshold(y_test, y_prob)
    
    # Apply optimized threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Cross-validation if requested
    cv_scores = []
    if use_cv:
        cv_scores = stratified_cross_validation(model, X_train, y_train, "lr")
    
    return model, metrics, best_threshold, cv_scores


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_cv: bool = False
) -> Tuple[Any, Dict[str, float], float, List[float]]:
    """
    Train Random Forest with balanced class weights.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        use_cv: Whether to perform cross-validation.
        
    Returns:
        Tuple of (trained model, metrics dictionary, optimized threshold, cv_scores).
    """
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_depth=10
    )
    
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Optimize threshold
    best_threshold, _, _ = optimize_threshold(y_test, y_prob)
    
    # Apply optimized threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Cross-validation if requested
    cv_scores = []
    if use_cv:
        cv_scores = stratified_cross_validation(model, X_train, y_train, "rf")
    
    return model, metrics, best_threshold, cv_scores


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_cv: bool = False
) -> Tuple[Any, Dict[str, float], float, List[float]]:
    """
    Train XGBoost with scale_pos_weight for class imbalance.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        use_cv: Whether to perform cross-validation.
        
    Returns:
        Tuple of (trained model, metrics dictionary, optimized threshold, cv_scores).
    """
    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)
    
    scale_pos_weight = get_scale_pos_weight(y_train)
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="auc"
    )
    
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Optimize threshold
    best_threshold, _, _ = optimize_threshold(y_test, y_prob)
    
    # Apply optimized threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Cross-validation if requested
    cv_scores = []
    if use_cv:
        cv_scores = stratified_cross_validation(model, X_train, y_train, "xgb")
    
    return model, metrics, best_threshold, cv_scores


def create_shap_explainer(model: Any, X_train: pd.DataFrame) -> Any:
    """
    Create SHAP TreeExplainer for tree-based models.
    
    Args:
        model: Trained model.
        X_train: Training features.
        
    Returns:
        SHAP TreeExplainer or None if not applicable.
    """
    print("\n" + "="*60)
    print("Creating SHAP Explainer...")
    print("="*60)
    
    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    
    # Calculate baseline (expected value)
    # Use a sample for faster computation
    sample_size = min(1000, X_train.shape[0])
    X_sample = X_train.sample(n=sample_size, random_state=42)
    explainer.shap_values(X_sample)
    
    print("SHAP Explainer created successfully")
    
    return explainer


def stratified_cross_validation(
    model_template: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str
) -> List[float]:
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        model_template: Model template to use.
        X_train: Training features.
        y_train: Training labels.
        model_type: Type of model ('lr', 'rf', 'xgb').
        
    Returns:
        List of ROC-AUC scores for each fold.
    """
    print("\n" + "-"*60)
    print("Performing Stratified 5-Fold Cross-Validation...")
    print("-"*60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Create fresh model instance
        if model_type == "lr":
            fold_model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs"
            )
        elif model_type == "rf":
            fold_model = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                max_depth=10
            )
        elif model_type == "xgb":
            scale_pos_weight = get_scale_pos_weight(y_fold_train)
            fold_model = XGBClassifier(
                n_estimators=100,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="auc"
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Predict
        y_prob = fold_model.predict_proba(X_fold_val)[:, 1]
        
        # Calculate ROC-AUC
        fold_auc = roc_auc_score(y_fold_val, y_prob)
        cv_scores.append(fold_auc)
        print(f"  Fold {fold}: ROC-AUC = {fold_auc:.4f}")
    
    mean_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)
    
    print(f"\n  Mean ROC-AUC: {mean_auc:.4f}")
    print(f"  Std ROC-AUC:  {std_auc:.4f}")
    print("-"*60)
    
    return cv_scores


def create_metrics_visualization(
    y_true: pd.Series,
    y_prob: np.ndarray,
    best_threshold: float,
    output_dir: str
) -> None:
    """
    Generate and save metric visualizations.
    
    Creates:
        - ROC Curve (PNG)
        - Confusion Matrix (PNG)
        - Precision-Recall Curve (PNG)
    
    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        best_threshold: Optimized classification threshold.
        output_dir: Directory to save plots.
    """
    print("\n" + "="*60)
    print("Generating Metric Visualizations...")
    print("="*60)
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = (y_prob >= best_threshold).astype(int)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Fraud Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'roc_curve.png')}")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Fraud Detection')
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'confusion_matrix.png')}")
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.axhline(y=y_true.mean(), color='red', linestyle='--', label='Baseline (random)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Fraud Detection')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'precision_recall_curve.png')}")
    
    print("="*60)


def train_isolation_forest(X_train: pd.DataFrame, contamination: float = 0.001) -> Tuple[Any, Dict[str, float]]:
    """
    Train IsolationForest for anomaly detection.
    
    Args:
        X_train: Training features.
        contamination: Expected proportion of outliers.
        
    Returns:
        Tuple of (trained model, score range).
    """
    print("\n" + "="*60)
    print("Training IsolationForest for Anomaly Detection...")
    print("="*60)
    
    # Use a sample for faster training on large datasets
    sample_size = min(50000, X_train.shape[0])
    X_sample = X_train.sample(n=sample_size, random_state=42)
    
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_sample)
    
    # Get score range for normalization
    scores = model.score_samples(X_sample)
    score_min = float(scores.min())
    score_max = float(scores.max())
    
    print(f"  Sample size: {sample_size:,}")
    print(f"  Score range: [{score_min:.4f}, {score_max:.4f}]")
    print("  IsolationForest trained successfully!")
    
    return model, {"min_score": score_min, "max_score": score_max}


def save_training_stats(X_train: pd.DataFrame, output_path: str) -> None:
    """
    Save training feature statistics for drift detection.
    
    Args:
        X_train: Training features.
        output_path: Path to save JSON file.
    """
    print("\n" + "="*60)
    print("Computing Training Feature Statistics...")
    print("="*60)
    
    # Use a sample for statistics
    sample_size = min(50000, X_train.shape[0])
    X_sample = X_train.sample(n=sample_size, random_state=42)
    
    mean = X_sample.mean().tolist()
    std = X_sample.std().tolist()
    
    stats = {
        "mean": mean,
        "std": std,
        "sample_size": sample_size,
        "features": list(X_train.columns)
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Sample size: {sample_size:,}")
    print(f"  Saved training statistics to: {output_path}")
    print("="*60)


def compute_global_shap_importance(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: List[str],
    output_path: str
) -> None:
    """
    Compute and save global SHAP feature importance.
    
    Args:
        model: Trained model.
        X_test: Test features.
        feature_names: List of feature names.
        output_path: Path to save JSON file.
    """
    print("\n" + "="*60)
    print("Computing Global SHAP Feature Importance...")
    print("="*60)
    
    # Use a sample for faster computation
    sample_size = min(5000, X_test.shape[0])
    X_sample = X_test.sample(n=sample_size, random_state=42)
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Handle case where shap_values is a list (binary classification)
    if isinstance(shap_values, list):
        shap_values_array = np.array(shap_values[1])  # Take positive class
    else:
        shap_values_array = np.array(shap_values)
    
    # Handle 3D array (some XGBoost versions)
    if len(shap_values_array.shape) == 3:
        shap_values_array = shap_values_array[:, :, 1]  # Take positive class
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
    
    # Get top 10
    top_10 = importance_df.head(10)
    
    # Save to JSON
    importance_dict = {
        "top_10_features": [
            {
                "rank": i + 1,
                "feature": row['feature'],
                "mean_abs_shap": float(row['mean_abs_shap'])
            }
            for i, (_, row) in enumerate(top_10.iterrows())
        ],
        "sample_size": sample_size
    }
    
    with open(output_path, 'w') as f:
        json.dump(importance_dict, f, indent=2)
    
    print(f"  Top 10 Features saved to: {output_path}")
    for i, (_, row) in enumerate(top_10.iterrows()):
        print(f"    {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    # Also generate SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_array, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    summary_path = os.path.join(os.path.dirname(output_path), 'shap_summary.png')
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"  Saved SHAP summary plot: {summary_path}")
    
    print("="*60)


def save_model_and_explainer(
    model: Any,
    explainer: Any,
    feature_names: list,
    best_threshold: float,
    model_path: str,
    explainer_path: str
) -> None:
    """
    Save the trained model and SHAP explainer to disk.
    
    Args:
        model: Trained model.
        explainer: SHAP explainer.
        feature_names: List of feature names.
        best_threshold: Optimized classification threshold.
        model_path: Path to save model.
        explainer_path: Path to save explainer.
    """
    print("\n" + "="*60)
    print("Saving model and explainer...")
    print("="*60)
    
    # Save model with feature names and optimized threshold
    model_data = {
        "model": model,
        "feature_names": feature_names,
        "best_threshold": best_threshold
    }
    joblib.dump(model_data, model_path)
    print(f"Model saved to: {model_path}")
    print(f"  - Optimized threshold: {best_threshold:.4f}")
    
    # Save explainer
    joblib.dump(explainer, explainer_path)
    print(f"Explainer saved to: {explainer_path}")


def print_training_summary(
    models: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    best_model_name: str,
    thresholds: Dict[str, float] = None
) -> None:
    """
    Print a summary of all trained models.
    
    Args:
        models: Dictionary of trained models.
        metrics: Dictionary of metrics for each model.
        best_model_name: Name of the best model.
        thresholds: Dictionary of optimized thresholds.
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'ROC-AUC':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Threshold':<12}")
    print("-"*85)
    
    for name, model_metrics in metrics.items():
        marker = " ***BEST***" if name == best_model_name else ""
        threshold_str = f"{thresholds.get(name, 0.5):.4f}" if thresholds else "0.5000"
        print(f"{name:<25} {model_metrics['roc_auc']:<12.4f} "
              f"{model_metrics['precision']:<12.4f} {model_metrics['recall']:<12.4f} "
              f"{model_metrics['f1']:<12.4f} {threshold_str:<12}{marker}")
    
    print("="*60)
    print(f"\nBest Model: {best_model_name}")
    print(f"ROC-AUC: {metrics[best_model_name]['roc_auc']:.4f}")
    print(f"Optimized Threshold: {thresholds.get(best_model_name, 0.5):.4f}")
    
    if metrics[best_model_name]['roc_auc'] >= 0.95:
        print("✓ Target ROC-AUC ≥ 0.95 achieved!")
    else:
        print(f"✗ Target ROC-AUC ≥ 0.95 NOT achieved (current: {metrics[best_model_name]['roc_auc']:.4f})")


def train_models(dataset_path: str = DEFAULT_DATASET_PATH, use_cv: bool = True) -> None:
    """
    Main training function that trains all models and saves the best one.
    
    Args:
        dataset_path: Path to the dataset CSV file.
        use_cv: Whether to perform cross-validation.
    """
    print("="*60)
    print("FRAUDSENSE AI - MODEL TRAINING")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Cross-Validation: {'Enabled' if use_cv else 'Disabled'}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(dataset_path)
    
    # Store models and their metrics
    models = {}
    metrics = {}
    thresholds = {}
    cv_results = {}
    
    # Train all models with cross-validation
    lr_model, lr_metrics, lr_threshold, lr_cv = train_logistic_regression(
        X_train, y_train, X_test, y_test, use_cv=use_cv
    )
    models["Logistic Regression"] = lr_model
    metrics["Logistic Regression"] = lr_metrics
    thresholds["Logistic Regression"] = lr_threshold
    cv_results["Logistic Regression"] = lr_cv
    
    rf_model, rf_metrics, rf_threshold, rf_cv = train_random_forest(
        X_train, y_train, X_test, y_test, use_cv=use_cv
    )
    models["Random Forest"] = rf_model
    metrics["Random Forest"] = rf_metrics
    thresholds["Random Forest"] = rf_threshold
    cv_results["Random Forest"] = rf_cv
    
    xgb_model, xgb_metrics, xgb_threshold, xgb_cv = train_xgboost(
        X_train, y_train, X_test, y_test, use_cv=use_cv
    )
    models["XGBoost"] = xgb_model
    metrics["XGBoost"] = xgb_metrics
    thresholds["XGBoost"] = xgb_threshold
    cv_results["XGBoost"] = xgb_cv
    
    # Select best model based on ROC-AUC
    best_model_name = max(metrics, key=lambda x: metrics[x]["roc_auc"])
    best_model = models[best_model_name]
    best_metrics = metrics[best_model_name]
    best_threshold = thresholds[best_model_name]
    
    # Print summary
    print_training_summary(models, metrics, best_model_name, thresholds)
    
    # Create SHAP explainer only for tree-based models
    explainer = None
    if best_model_name in ["Random Forest", "XGBoost"]:
        explainer = create_shap_explainer(best_model, X_train)
    
    # Generate metric visualizations
    y_prob = best_model.predict_proba(X_test)[:, 1]
    create_metrics_visualization(y_test, y_prob, best_threshold, METRICS_DIR)
    
    # Compute global SHAP feature importance for tree-based models
    if best_model_name in ["Random Forest", "XGBoost"]:
        compute_global_shap_importance(
            best_model,
            X_test,
            list(X_train.columns),
            GLOBAL_FEATURE_IMPORTANCE_PATH
        )
    
    # Save model and explainer
    save_model_and_explainer(
        model=best_model,
        explainer=explainer,
        feature_names=list(X_train.columns),
        best_threshold=best_threshold,
        model_path=MODEL_PATH,
        explainer_path=EXPLAINER_PATH
    )
    
    # Train IsolationForest for anomaly detection
    anomaly_model, score_range = train_isolation_forest(X_train)
    
    # Save anomaly model
    anomaly_data = {
        "model": anomaly_model,
        "min_score": score_range["min_score"],
        "max_score": score_range["max_score"]
    }
    joblib.dump(anomaly_data, ANOMALY_MODEL_PATH)
    print(f"Anomaly model saved to: {ANOMALY_MODEL_PATH}")
    
    # Save training statistics for drift detection
    save_training_stats(X_train, TRAINING_STATS_PATH)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nSaved files:")
    print(f"  - Model: {MODEL_PATH}")
    print(f"  - Explainer: {EXPLAINER_PATH}")
    print(f"  - Anomaly Model: {ANOMALY_MODEL_PATH}")
    print(f"  - Training Stats: {TRAINING_STATS_PATH}")
    print(f"  - Metrics: {METRICS_DIR}/*")
    print(f"  - Global SHAP: {GLOBAL_FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    # Get dataset path from command line or use default
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_PATH
    
    # Change to backend directory for execution
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    train_models(dataset_path)
