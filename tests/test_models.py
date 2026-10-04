import os
import json
import pytest
import numpy as np
import onnxruntime as ort
import xgboost as xgb

FEATURE_COLS = [
    "temperature", "humidity", "light", "voltage",
    "delta_temp_3m", "delta_volt_3m",
    "z_temp_15m", "z_volt_15m",
    "temp_slope_15m", "var_temp_15m"
]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def test_isolation_forest_onnx_inference():
    """Validates Isolation Forest ONNX model loading, scoring, and calibration."""
    model_path = "models/isolation_forest.onnx"
    calib_path = "models/if_calibration.json"
    
    assert os.path.exists(model_path)
    assert os.path.exists(calib_path)
    
    with open(calib_path) as f:
        calib = json.load(f)
    assert "coef" in calib and "intercept" in calib
    
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    dummy_input = np.random.randn(5, 10).astype(np.float32)
    outs = session.run(None, {input_name: dummy_input})
    raw_scores = outs[1][:, 0] if len(outs) > 1 and hasattr(outs[1], 'shape') else outs[0].astype(np.float32)
    
    probs = sigmoid(calib["coef"] * (-raw_scores) + calib["intercept"])
    assert len(probs) == 5
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_autoencoder_onnx_inference():
    """Validates PyTorch Autoencoder ONNX model loading, reconstruction, and MSE."""
    model_path = "models/autoencoder.onnx"
    calib_path = "models/ae_calibration.json"
    
    assert os.path.exists(model_path)
    assert os.path.exists(calib_path)
    
    with open(calib_path) as f:
        calib = json.load(f)
    assert "means" in calib and "stds" in calib and len(calib["means"]) == 10
    
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    dummy_input = np.random.randn(5, 10).astype(np.float32)
    recon = session.run(None, {input_name: dummy_input})[0]
    assert recon.shape == dummy_input.shape
    
    mse = np.mean((dummy_input - recon) ** 2, axis=1)
    probs = sigmoid(calib["coef"] * mse + calib["intercept"])
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_xgboost_fallback_inference():
    """Validates Tier 2 XGBoost model loading and inference on 12-feature vectors."""
    model_path = "models/xgboost_fallback.json"
    assert os.path.exists(model_path)
    
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    
    dummy_12_features = np.random.randn(5, 12).astype(np.float32)
    preds = clf.predict(dummy_12_features)
    probs = clf.predict_proba(dummy_12_features)
    
    assert len(preds) == 5
    assert probs.shape == (5, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
