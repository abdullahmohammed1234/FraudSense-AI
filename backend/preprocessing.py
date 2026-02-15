"""
Preprocessing module for FraudSense AI.

This module handles data loading, feature/target separation, and train/test splitting
for the Credit Card Fraud Detection dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the Credit Card Fraud Detection dataset from CSV.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        DataFrame containing the dataset.
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValueError: If the dataset is empty or missing required columns.
    """
    df = pd.read_csv(filepath)
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' column")
    
    print(f"Dataset loaded: {df.shape[0]:,} transactions, {df.shape[1]} columns")
    print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
    
    return df


def separate_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple of (features DataFrame, target Series).
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Features DataFrame.
        y: Target Series.
        test_size: Proportion of data for testing (default 0.2).
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Testing set: {X_test.shape[0]:,} samples")
    print(f"Training fraud rate: {y_train.mean()*100:.2f}%")
    print(f"Testing fraud rate: {y_test.mean()*100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def get_class_weights(y: pd.Series) -> dict:
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        y: Target Series.
        
    Returns:
        Dictionary with class weights.
    """
    class_counts = y.value_counts()
    total = len(y)
    
    # Calculate balanced weights
    weights = {
        0: total / (2 * class_counts[0]),
        1: total / (2 * class_counts[1])
    }
    
    return weights


def get_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost.
    
    Args:
        y: Target Series.
        
    Returns:
        Scale pos weight value.
    """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    
    return neg_count / pos_count


def preprocess_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Main preprocessing function that loads data and splits into train/test sets.
    
    Args:
        filepath: Path to the CSV dataset.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # Load dataset
    df = load_dataset(filepath)
    
    # Separate features and target
    X, y = separate_features_target(df)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    
    return X_train, X_test, y_train, y_test
