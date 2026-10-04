"""
Standalone Interactive Streaming Anomaly Detection Simulation
Runs a local Python replay over precomputed feature rows. It uses Python ONNX
Runtime and XGBoost; it does not launch Flink, recompute online features, or call
the FastAPI sidecar.

Path exercised:
  Held-out Parquet rows -> Python ONNX inference -> uncertainty gate
                       -> local XGBoost fallback -> CLI metrics

Usage:
  python demo_streaming_pipeline.py --samples 5000 --speed 2000
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import onnxruntime as ort
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

FEATURE_COLS = [
    "temperature", "humidity", "light", "voltage",
    "delta_temp_3m", "delta_volt_3m",
    "z_temp_15m", "z_volt_15m",
    "temp_slope_15m", "var_temp_15m"
]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def parse_args():
    parser = argparse.ArgumentParser(description="BDT Real-Time Streaming Demonstration")
    parser.add_argument("--samples", type=int, default=5000, help="Number of sensor events to stream (max 40000)")
    parser.add_argument("--speed", type=int, default=0, help="Events per second throttle (0 = unthrottled max line-rate)")
    parser.add_argument("--data-file", type=str, default="data/processed/test_features.parquet", help="Path to test parquet file")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 85)
    print("  🚀 BDT: BOUNDED DECISION TIERING STREAMING DEMONSTRATION")
    print("  Sub-millisecond IoT Anomaly Detection with Selective Epistemic Escalation")
    print("=" * 85)
    
    if not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} not found. Run preprocessing/inject_anomalies.py first.")
        sys.exit(1)
        
    print(f"Loading held-out test telemetry from {args.data_file}...")
    df = pd.read_parquet(args.data_file)
    n_total = min(args.samples, len(df))
    df = df.iloc[:n_total].copy()
    
    # Load calibration parameters
    with open("models/if_calibration.json") as f: if_calib = json.load(f)
    with open("models/ae_calibration.json") as f: ae_calib = json.load(f)
    
    print("Initializing embedded ONNX runtimes and Tier 2 XGBoost model...")
    sess_if = ort.InferenceSession("models/isolation_forest.onnx")
    if_in_name = sess_if.get_inputs()[0].name
    
    sess_ae = ort.InferenceSession("models/autoencoder.onnx")
    ae_in_name = sess_ae.get_inputs()[0].name
    ae_means = np.array(ae_calib["means"], dtype=np.float32)
    ae_stds = np.array(ae_calib["stds"], dtype=np.float32)
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost_fallback.json")
    
    X_raw = df[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    labels = df["label"].values.astype(np.int32)
    anomaly_types = df["anomaly_type"].values
    
    # Tracking accumulators
    n_escalated = 0
    y_preds = []
    y_probs = []
    latencies_us = []
    
    family_counts = {"SPIKE_TEMP": 0, "SPIKE_VOLT_DROP": 0, "DRIFT": 0, "FLATLINE": 0}
    family_detected = {"SPIKE_TEMP": 0, "SPIKE_VOLT_DROP": 0, "DRIFT": 0, "FLATLINE": 0}
    
    print("\nStreaming pipeline running across sensor motes...")
    print("-" * 85)
    
    t_pipeline_start = time.perf_counter()
    report_interval = 500
    
    for i in range(n_total):
        t0 = time.perf_counter_ns()
        feat = X_raw[i:i+1]
        true_label = labels[i]
        anom_type = anomaly_types[i]
        if anom_type in family_counts:
            family_counts[anom_type] += 1
            
        # 1. Python ONNX Runtime fast path (not the Flink/JVM runtime)
        outs_if = sess_if.run(None, {if_in_name: feat})
        raw_if = outs_if[1][0, 0] if len(outs_if) > 1 and hasattr(outs_if[1], 'shape') else outs_if[0][0]
        p_if = float(sigmoid(if_calib["coef"] * (-raw_if) + if_calib["intercept"]))
        pred_if = 1 if p_if >= 0.5 else 0
        
        feat_norm = (feat - ae_means) / ae_stds
        recon = sess_ae.run(None, {ae_in_name: feat_norm})[0]
        mse = float(np.mean((feat_norm - recon) ** 2))
        p_ae = float(sigmoid(ae_calib["coef"] * mse + ae_calib["intercept"]))
        pred_ae = 1 if p_ae >= 0.5 else 0
        
        p_c = 0.5 * (p_if + p_ae)
        
        # 2. Epistemic Uncertainty Gating
        is_uncertain = (
            (0.35 < p_if < 0.65) or
            (0.35 < p_ae < 0.65) or
            (pred_if != pred_ae)
        )
        
        if is_uncertain:
            # 3. Tier 2: Specialized Fallback Escalation (XGBoost)
            n_escalated += 1
            feat_enh = np.hstack([feat, np.array([[p_if, p_ae]], dtype=np.float32)])
            p_final = float(xgb_model.predict_proba(feat_enh)[0, 1])
            pred_final = 1 if p_final >= 0.5 else 0
        else:
            # Fast Path Pass-Through
            p_final = p_c
            pred_final = 1 if p_final >= 0.5 else 0
            
        t_elap_us = (time.perf_counter_ns() - t0) / 1000.0
        latencies_us.append(t_elap_us)
        y_preds.append(pred_final)
        y_probs.append(p_final)
        
        if pred_final == 1 and anom_type in family_detected:
            family_detected[anom_type] += 1
            
        # Optional artificial stream rate throttle
        if args.speed > 0:
            target_sleep = 1.0 / args.speed
            time.sleep(target_sleep)
            
        # Periodic Dashboard Update
        if (i + 1) % report_interval == 0 or (i + 1) == n_total:
            curr_processed = i + 1
            total_elapsed = time.perf_counter() - t_pipeline_start
            curr_rate = curr_processed / total_elapsed if total_elapsed > 0 else 0
            
            p_curr = precision_score(labels[:curr_processed], y_preds[:curr_processed], zero_division=0)
            r_curr = recall_score(labels[:curr_processed], y_preds[:curr_processed], zero_division=0)
            f1_curr = f1_score(labels[:curr_processed], y_preds[:curr_processed], zero_division=0)
            esc_pct = (n_escalated / curr_processed) * 100.0
            
            p50 = np.percentile(latencies_us, 50)
            p95 = np.percentile(latencies_us, 95)
            
            print(f"[{curr_processed:6d}/{n_total}] Rate: {curr_rate:7.0f} ev/s | "
                  f"Escalated: {esc_pct:4.1f}% | "
                  f"F1: {f1_curr:.4f} (P: {p_curr:.3f}, R: {r_curr:.3f}) | "
                  f"p50: {p50:4.1f}µs, p95: {p95:4.1f}µs")
                  
    # Final Metrics Summary Table
    total_time = time.perf_counter() - t_pipeline_start
    overall_p = precision_score(labels, y_preds, zero_division=0)
    overall_r = recall_score(labels, y_preds, zero_division=0)
    overall_f1 = f1_score(labels, y_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, y_preds).ravel()
    fpr = (fp / (fp + tn)) * 100.0
    
    print("\n" + "=" * 85)
    print("  🏁 STREAMING SIMULATION COMPLETE - SUMMARY REPORT")
    print("=" * 85)
    print(f"Total Stream Events Processed : {n_total:,}")
    print(f"Total Wall-Clock Time Elapsed : {total_time:.2f} seconds")
    print(f"Sustained Pipeline Rate       : {n_total/total_time:,.0f} events/second")
    print(f"Fast-Path Pass-Through        : {n_total - n_escalated:,} ({(1 - n_escalated/n_total)*100:.1f}%)")
    print(f"Tier 2 Escalations (XGBoost)  : {n_escalated:,} ({(n_escalated/n_total)*100:.1f}%)")
    print("-" * 85)
    print(f"Detection Quality (F1-Score)  : {overall_f1:.4f}")
    print(f"Precision (PPV)               : {overall_p:.4f}  (True Positives: {tp:,}, False Positives: {fp:,})")
    print(f"Recall (Sensitivity)          : {overall_r:.4f}  (Anomalies Caught: {tp}/{tp+fn})")
    print(f"False Positive Rate (FPR)     : {fpr:.2f}% (False Alarms: {fp:,} out of {tn+fp:,} normal windows)")
    print("-" * 85)
    print("Python Model Inference Timing (local loop; excludes Flink/network/sinks):")
    print(f"  p50 (Median)                : {np.percentile(latencies_us, 50):.2f} µs")
    print(f"  p95                         : {np.percentile(latencies_us, 95):.2f} µs")
    print(f"  p99                         : {np.percentile(latencies_us, 99):.2f} µs")
    print("-" * 85)
    print("Disaggregated Recall by Anomaly Family:")
    for fam, cnt in family_counts.items():
        det = family_detected[fam]
        rec = (det / cnt * 100.0) if cnt > 0 else 0.0
        print(f"  • {fam:18s} : {det:3d} / {cnt:3d} caught ({rec:5.1f}%)")
    print("=" * 85)

if __name__ == "__main__":
    main()
