import os
import time
import json
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Enforce strict thread bounds: 4 threads max for CPU sidecar
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from sidecar.laya_runner import get_laya_agent, format_sensor_state, predict_laya_batch

app = FastAPI(title="Streaming Anomaly Escalation Sidecar")

class SensorState(BaseModel):
    moteid: int
    epoch: int
    temperature: float
    humidity: float
    light: float
    voltage: float
    delta_temp_3m: float
    delta_volt_3m: float
    z_temp_15m: float
    z_volt_15m: float
    temp_slope_15m: float
    var_temp_15m: float
    p_if: float
    p_ae: float

class DecisionResponse(BaseModel):
    decision: str        # "NORMAL" or "ANOMALY"
    confidence: float    # Continuous probability P(anomaly) in [0.0, 1.0]
    latency_ms: float
    engine: str

class BatchSensorState(BaseModel):
    items: List[SensorState]

# Global Model Registry
xgb_model = None

@app.on_event("startup")
def load_models():
    global xgb_model
    xgb_path = os.path.join("models", "xgboost_fallback.json")
    if os.path.exists(xgb_path):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_path)
        print(f"[Sidecar] Loaded XGBoost fallback model from {xgb_path}")
    else:
        print("[Sidecar] Warning: models/xgboost_fallback.json not found!")
    
    # Warm up Laya
    try:
        _ = get_laya_agent()
        print("[Sidecar] Laya agent initialized.")
    except Exception as e:
        print(f"[Sidecar] Warning: Laya initialization failed: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "xgb_ready": xgb_model is not None, "laya_ready": True}

def to_feature_vector(s: SensorState) -> np.ndarray:
    return np.array([
        s.temperature, s.humidity, s.light, s.voltage,
        s.delta_temp_3m, s.delta_volt_3m,
        s.z_temp_15m, s.z_volt_15m,
        s.temp_slope_15m, s.var_temp_15m,
        s.p_if, s.p_ae
    ], dtype=np.float32).reshape(1, -1)

@app.post("/decide/xgb", response_model=DecisionResponse)
def decide_xgb(state: SensorState):
    """System D: Conventional ML Fallback via XGBoost"""
    t0 = time.perf_counter()
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="XGBoost model not loaded")
    
    vec = to_feature_vector(state)
    probs = xgb_model.predict_proba(vec)[0]
    p_anomaly = float(probs[1])
    decision = "ANOMALY" if p_anomaly >= 0.5 else "NORMAL"
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return DecisionResponse(
        decision=decision,
        confidence=p_anomaly, # Continuous P(anomaly)
        latency_ms=round(latency_ms, 2),
        engine="xgboost"
    )

@app.post("/decide/heuristic", response_model=DecisionResponse)
def decide_heuristic(state: SensorState):
    """System E: Hand-crafted heuristic rule fallback (Experimental Control)"""
    t0 = time.perf_counter()
    z_shock = abs(state.z_temp_15m) + abs(state.z_volt_15m)
    rate_of_change = abs(state.temp_slope_15m) + abs(state.delta_temp_3m)
    disagreement = abs(state.p_if - state.p_ae)
    
    logit = (
        1.8 * state.p_if +
        2.0 * state.p_ae +
        0.7 * z_shock +
        1.2 * rate_of_change +
        0.5 * disagreement - 2.2
    )
    p_heur = float(1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0))))
    decision = "ANOMALY" if p_heur >= 0.5 else "NORMAL"
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return DecisionResponse(
        decision=decision,
        confidence=p_heur,
        latency_ms=round(latency_ms, 2),
        engine="heuristic-control"
    )

@app.post("/decide/laya", response_model=DecisionResponse)
def decide_laya(state: SensorState):
    """System F: Real Laya / Jev Non-Autoregressive Decision Engine"""
    t0 = time.perf_counter()
    st_text = format_sensor_state(
        moteid=state.moteid, epoch=state.epoch,
        temp=state.temperature, volt=state.voltage, hum=state.humidity, light=state.light,
        delta_t=state.delta_temp_3m, delta_v=state.delta_volt_3m,
        z_t=state.z_temp_15m, z_v=state.z_volt_15m,
        slope_t=state.temp_slope_15m, var_t=state.var_temp_15m,
        p_if=state.p_if, p_ae=state.p_ae
    )
    res = predict_laya_batch([st_text], batch_size=1)[0]
    decision, p_anomaly = res
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return DecisionResponse(
        decision=decision,
        confidence=p_anomaly,
        latency_ms=round(latency_ms, 2),
        engine="real-laya-modernbert"
    )

@app.post("/decide/batch")
def decide_batch(batch: BatchSensorState, engine: str = "xgb"):
    t0 = time.perf_counter()
    results = []
    
    if engine == "xgb" and xgb_model is not None:
        X = np.vstack([to_feature_vector(s) for s in batch.items])
        probs = xgb_model.predict_proba(X)
        for i, s in enumerate(batch.items):
            p_anom = float(probs[i][1])
            results.append({
                "moteid": s.moteid,
                "epoch": s.epoch,
                "decision": "ANOMALY" if p_anom >= 0.5 else "NORMAL",
                "p_anomaly": p_anom
            })
    elif engine == "laya":
        state_texts = [
            format_sensor_state(
                moteid=s.moteid, epoch=s.epoch,
                temp=s.temperature, volt=s.voltage, hum=s.humidity, light=s.light,
                delta_t=s.delta_temp_3m, delta_v=s.delta_volt_3m,
                z_t=s.z_temp_15m, z_v=s.z_volt_15m,
                slope_t=s.temp_slope_15m, var_t=s.var_temp_15m,
                p_if=s.p_if, p_ae=s.p_ae
            )
            for s in batch.items
        ]
        batch_out = predict_laya_batch(state_texts, batch_size=32)
        for i, (choice, p_anom) in enumerate(batch_out):
            results.append({
                "moteid": batch.items[i].moteid,
                "epoch": batch.items[i].epoch,
                "decision": choice,
                "p_anomaly": p_anom
            })
    else: # heuristic
        for s in batch.items:
            res = decide_heuristic(s)
            results.append({
                "moteid": s.moteid,
                "epoch": s.epoch,
                "decision": res.decision,
                "p_anomaly": res.confidence
            })
            
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "engine": engine,
        "count": len(results),
        "total_latency_ms": round(total_ms, 2),
        "avg_latency_ms": round(total_ms / len(results), 3) if results else 0,
        "results": results
    }
