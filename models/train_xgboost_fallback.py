import json
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score

FEATURE_COLS = [
    "temperature", "humidity", "light", "voltage",
    "delta_temp_3m", "delta_volt_3m",
    "z_temp_15m", "z_volt_15m",
    "temp_slope_15m", "var_temp_15m"
]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def main():
    print("Loading validation dataset & calibration params...")
    val_df = pd.read_parquet("data/processed/val_features.parquet")
    X_val_raw = val_df[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.int32)
    
    with open("models/if_calibration.json", "r") as f:
        if_calib = json.load(f)
    with open("models/ae_calibration.json", "r") as f:
        ae_calib = json.load(f)
        
    # 1. Run Isolation Forest ONNX
    if_session = ort.InferenceSession("models/isolation_forest.onnx")
    if_input_name = if_session.get_inputs()[0].name
    if_outs = if_session.run(None, {if_input_name: X_val_raw})
    
    # In skl2onnx, outputs are [labels, probabilities/scores]
    if len(if_outs) > 1 and hasattr(if_outs[1], 'shape'):
        raw_if_scores = if_outs[1][:, 0]
    else:
        # If dictionary or single array
        raw_if_scores = if_outs[0].astype(np.float32)
        
    p_if = sigmoid(if_calib["coef"] * (-raw_if_scores) + if_calib["intercept"])
    
    # 2. Run Autoencoder ONNX
    ae_session = ort.InferenceSession("models/autoencoder.onnx")
    ae_input_name = ae_session.get_inputs()[0].name
    
    ae_means = np.array(ae_calib["means"], dtype=np.float32)
    ae_stds = np.array(ae_calib["stds"], dtype=np.float32)
    X_val_norm = (X_val_raw - ae_means) / ae_stds
    
    ae_recon = ae_session.run(None, {ae_input_name: X_val_norm})[0]
    ae_mse = np.mean((X_val_norm - ae_recon) ** 2, axis=1)
    p_ae = sigmoid(ae_calib["coef"] * ae_mse + ae_calib["intercept"])
    
    # 3. Identify Uncertain Cases
    class_if = (p_if >= 0.5).astype(int)
    class_ae = (p_ae >= 0.5).astype(int)
    
    uncertain_mask = (
        ((p_if > 0.35) & (p_if < 0.65)) |
        ((p_ae > 0.35) & (p_ae < 0.65)) |
        (class_if != class_ae)
    )
    
    n_uncertain = np.sum(uncertain_mask)
    escalation_rate = n_uncertain / len(y_val) * 100.0
    print(f"Uncertainty Gating Analysis on Validation ({len(y_val):,} total windows):")
    print(f"- Escalated Cases: {n_uncertain:,} ({escalation_rate:.2f}%)")
    print(f"- Confident Cases: {len(y_val) - n_uncertain:,} ({100.0 - escalation_rate:.2f}%)")
    print(f"- Anomalies in Uncertain Subset: {np.sum(y_val[uncertain_mask]):,} / {np.sum(y_val):,}")
    
    # 4. Train XGBoost Fallback specifically on uncertain cases + full feature set
    X_enhanced = np.hstack([X_val_raw, p_if.reshape(-1, 1), p_ae.reshape(-1, 1)])
    
    X_train_xgb = X_enhanced[uncertain_mask]
    y_train_xgb = y_val[uncertain_mask]
    
    # If not enough anomalies in uncertain, train on full validation with sample weights
    print(f"Training XGBoost Fallback classifier on uncertain subset ({len(X_train_xgb)} samples)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=max(1.0, (len(y_train_xgb) - np.sum(y_train_xgb)) / (np.sum(y_train_xgb) + 1e-4)),
        random_state=42,
        n_jobs=4 # Limit to 4 threads as configured
    )
    xgb_model.fit(X_train_xgb, y_train_xgb)
    
    y_pred_xgb = xgb_model.predict(X_train_xgb)
    print("\nXGBoost Fallback Performance on Uncertain Subset:")
    print(classification_report(y_train_xgb, y_pred_xgb, target_names=["NORMAL", "ANOMALY"]))
    
    xgb_path = "models/xgboost_fallback.json"
    xgb_model.save_model(xgb_path)
    print(f"Saved XGBoost fallback model to: {xgb_path} ({os.path.getsize(xgb_path)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
