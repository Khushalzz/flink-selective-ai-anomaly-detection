import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

FEATURE_COLS = [
    "temperature", "humidity", "light", "voltage",
    "delta_temp_3m", "delta_volt_3m",
    "z_temp_15m", "z_volt_15m",
    "temp_slope_15m", "var_temp_15m"
]

def main():
    os.makedirs("models", exist_ok=True)
    
    print("Loading train & validation features...")
    train_df = pd.read_parquet("data/processed/train_features.parquet")
    val_df = pd.read_parquet("data/processed/val_features.parquet")
    
    X_train = train_df[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    X_val = val_df[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.int32)
    
    print(f"Training Isolation Forest on {len(X_train):,} samples...")
    # 100 trees, 0.03 expected anomaly rate
    model = IsolationForest(
        n_estimators=100,
        max_samples=512,
        contamination=0.03,
        random_state=42,
        n_jobs=4
    )
    model.fit(X_train)
    print("Isolation Forest trained!")
    
    # Raw decision function on validation
    raw_scores_val = model.decision_function(X_val) # Higher = more normal, lower = more anomalous
    
    # Calibrate to probability of anomaly using logistic regression
    # (Flip sign so higher score = higher probability of anomaly)
    X_calib = (-raw_scores_val).reshape(-1, 1)
    calibrator = LogisticRegression(class_weight="balanced")
    calibrator.fit(X_calib, y_val)
    
    calib_params = {
        "coef": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
        "features": FEATURE_COLS
    }
    with open("models/if_calibration.json", "w") as f:
        json.dump(calib_params, f, indent=2)
    print(f"Calibrated probability model saved: P(anomaly) = sigmoid({calib_params['coef']:.4f} * (-raw_score) + {calib_params['intercept']:.4f})")
    
    # Export to ONNX for embedded Flink Java execution
    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset={'': 12, 'ai.onnx.ml': 3})
    
    onnx_path = "models/isolation_forest.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported Isolation Forest to ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")
    
    # Test ONNX inference
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    test_out = session.run(None, {input_name: X_val[:5]})
    print(f"ONNX Test Inference Succeeded! Output keys: {len(test_out)}")

if __name__ == "__main__":
    main()
