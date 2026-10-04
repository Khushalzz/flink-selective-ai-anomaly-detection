"""Offline escalation sweep with explicitly estimated model-only capacity.

Detection scores are evaluated on the held-out Parquet split. Capacity is an
analytical estimate from batched Python model timings and an assumed transport
cap; this script does not run or measure Flink, Kafka, HTTP, or persistence.
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

FEATURE_COLS = [
    "temperature", "humidity", "light", "voltage",
    "delta_temp_3m", "delta_volt_3m",
    "z_temp_15m", "z_volt_15m",
    "temp_slope_15m", "var_temp_15m",
]
SWEEP_RATES = [0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 1.00]
ASSUMED_TRANSPORT_CAPACITY_EV_S = 35_000.0


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -25.0, 25.0)))


def run_sweep():
    test_df = pd.read_parquet("data/processed/test_features.parquet")
    X = test_df[FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
    y = test_df["label"].to_numpy(dtype=np.int32)
    n_samples = len(y)

    with open("models/if_calibration.json", encoding="utf-8") as handle:
        if_calib = json.load(handle)
    with open("models/ae_calibration.json", encoding="utf-8") as handle:
        ae_calib = json.load(handle)

    if_session = ort.InferenceSession("models/isolation_forest.onnx")
    if_input = if_session.get_inputs()[0].name
    ae_session = ort.InferenceSession("models/autoencoder.onnx")
    ae_input = ae_session.get_inputs()[0].name

    # Measure batched Python model time; this is not a streaming latency measurement.
    sample_count = min(1000, n_samples)
    x_sample = X[:sample_count]
    ae_means = np.asarray(ae_calib["means"], dtype=np.float32)
    ae_stds = np.asarray(ae_calib["stds"], dtype=np.float32)
    x_sample_norm = (x_sample - ae_means) / ae_stds
    t0 = time.perf_counter()
    if_session.run(None, {if_input: x_sample})
    ae_session.run(None, {ae_input: x_sample_norm})
    fast_batch_us = (time.perf_counter() - t0) * 1e6 / sample_count

    if_outputs = if_session.run(None, {if_input: X})
    raw_if = if_outputs[1][:, 0] if len(if_outputs) > 1 and hasattr(if_outputs[1], "shape") else if_outputs[0].astype(np.float32)
    p_if = sigmoid(if_calib["coef"] * (-raw_if) + if_calib["intercept"])

    x_norm = (X - ae_means) / ae_stds
    reconstruction = ae_session.run(None, {ae_input: x_norm})[0]
    mse = np.mean((x_norm - reconstruction) ** 2, axis=1)
    p_ae = sigmoid(ae_calib["coef"] * mse + ae_calib["intercept"])
    p_fast = 0.5 * (p_if + p_ae)

    uncertainty = (
        np.abs(p_if - p_ae) * 1.5
        + (1.0 - 2.0 * np.abs(p_if - 0.5))
        + (1.0 - 2.0 * np.abs(p_ae - 0.5))
    )
    ranked = np.argsort(-uncertainty)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost_fallback.json")
    x_enhanced = np.hstack((X, p_if[:, None], p_ae[:, None]))
    t0 = time.perf_counter()
    xgb_model.predict_proba(x_enhanced[:sample_count])
    xgb_batch_us = (time.perf_counter() - t0) * 1e6 / sample_count
    p_xgb = xgb_model.predict_proba(x_enhanced)[:, 1]

    records = []
    for rate in SWEEP_RATES:
        count = int(np.round(rate * n_samples))
        selected = ranked[:count]
        p_final = p_fast.copy()
        p_final[selected] = p_xgb[selected]
        predicted = (p_final >= 0.5).astype(np.int32)
        tn, fp, _, _ = confusion_matrix(y, predicted).ravel()

        estimated_service_us = fast_batch_us + rate * xgb_batch_us
        estimated_capacity = min(
            ASSUMED_TRANSPORT_CAPACITY_EV_S,
            1e6 / estimated_service_us if estimated_service_us > 0 else 0.0,
        )
        records.append({
            "Escalation_Rate": rate,
            "Escalated_Windows": count,
            "F1_Score": f1_score(y, predicted, zero_division=0),
            "Precision": precision_score(y, predicted, zero_division=0),
            "Recall": recall_score(y, predicted, zero_division=0),
            "PR_AUC": average_precision_score(y, p_final),
            "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
            "Estimated_Model_Service_us_per_event": estimated_service_us,
            "Assumed_Transport_Capacity_ev_s": ASSUMED_TRANSPORT_CAPACITY_EV_S,
            "Estimated_Model_Capacity_ev_s": estimated_capacity,
        })

    results = pd.DataFrame(records)
    results.to_csv("experiments/results/escalation_sweep_results.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    os.makedirs("assets", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        results["Estimated_Model_Capacity_ev_s"], results["F1_Score"],
        "o-", color="#1f77b4", linewidth=2.5, markersize=8,
        label="Offline XGBoost escalation; estimated capacity",
    )
    for _, row in results.iterrows():
        ax.annotate(
            f" {row['Escalation_Rate'] * 100:.1f}%",
            (row["Estimated_Model_Capacity_ev_s"], row["F1_Score"]),
            fontsize=8.5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Estimated Model-Service Capacity (events/s; log scale)")
    ax.set_ylabel("Offline held-out F1 score")
    ax.set_title("Offline F1 vs. Estimated Model-Service Capacity")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("assets/f1_vs_estimated_capacity.png", dpi=300)
    plt.close(fig)

    print(f"Wrote offline metrics and model-only capacity estimates for {n_samples:,} rows.")
    print("Capacity is estimated from batched Python timings and an assumed transport cap; it is not Flink throughput.")
    print("Saved experiments/results/escalation_sweep_results.csv and assets/f1_vs_estimated_capacity.png")


if __name__ == "__main__":
    run_sweep()
