import os
import pytest
import numpy as np
import pandas as pd

def test_chronological_temporal_boundaries():
    """Formally asserts zero data leakage between train, val, and test splits."""
    train_path = "data/processed/train_features.parquet"
    val_path = "data/processed/val_features.parquet"
    test_path = "data/processed/test_features.parquet"
    
    assert os.path.exists(train_path), f"Train features missing at {train_path}"
    assert os.path.exists(val_path), f"Val features missing at {val_path}"
    assert os.path.exists(test_path), f"Test features missing at {test_path}"
    
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    
    # Parse true composite timestamps
    t_train = pd.to_datetime(df_train["date"].astype(str) + " " + df_train["time"].astype(str))
    t_val = pd.to_datetime(df_val["date"].astype(str) + " " + df_val["time"].astype(str))
    t_test = pd.to_datetime(df_test["date"].astype(str) + " " + df_test["time"].astype(str))
    
    max_train = t_train.max()
    min_val = t_val.min()
    max_val = t_val.max()
    min_test = t_test.min()
    
    assert max_train <= min_val, f"Temporal leakage: max(train) {max_train} > min(val) {min_val}"
    assert max_val <= min_test, f"Temporal leakage: max(val) {max_val} > min(test) {min_test}"

def test_clean_unsupervised_training_set():
    """Asserts training split contains strictly nominal records (zero anomalies)."""
    df_train = pd.read_parquet("data/processed/train_features.parquet")
    assert df_train["label"].sum() == 0, "Training set contains dirty anomaly labels!"
    assert (df_train["anomaly_type"] == "NORMAL").all(), "Non-normal anomaly types found in training set!"

def test_ground_truth_prevalence():
    """Asserts validation and test sets maintain realistic anomaly prevalence (~2.5% to 3.5%)."""
    df_val = pd.read_parquet("data/processed/val_features.parquet")
    df_test = pd.read_parquet("data/processed/test_features.parquet")
    
    val_prev = df_val["label"].mean() * 100.0
    test_prev = df_test["label"].mean() * 100.0
    
    assert 2.0 <= val_prev <= 4.0, f"Val prevalence {val_prev:.2f}% out of realistic 2-4% bound"
    assert 2.0 <= test_prev <= 4.0, f"Test prevalence {test_prev:.2f}% out of realistic 2-4% bound"
