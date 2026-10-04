import os
import time
from typing import List, Tuple, Dict, Any
import torch
import laya

# Budgeted thread limit for Python Sidecar / Laya
torch.set_num_threads(4)

_AGENT = None

QUESTION = {
    'is_anomaly': {
        'type': 'choice',
        'instructions': 'Does this sensor telemetry indicate normal operation or an anomalous condition (e.g. spike, drop, drift, flatline)?',
        'criteria': {
            'NORMAL': 'Normal sensor operation following stable baseline temperature and voltage patterns',
            'ANOMALY': 'Anomalous behavior such as extreme temperature spike, battery voltage collapse, drift, or frozen flatline'
        }
    }
}

def get_laya_agent():
    global _AGENT
    if _AGENT is None:
        print("[Laya Runner] Initializing official convaiinnovations/laya (typed-decisions, 4 threads)...")
        _AGENT = laya.load('convaiinnovations/laya', subfolder='typed-decisions', device='cpu')
        print("[Laya Runner] Model loaded successfully.")
    return _AGENT

def format_sensor_state(
    moteid: int, epoch: int,
    temp: float, volt: float, hum: float, light: float,
    delta_t: float, delta_v: float,
    z_t: float, z_v: float,
    slope_t: float, var_t: float,
    p_if: float, p_ae: float
) -> str:
    """Formats structured textual context for Laya non-autoregressive decision model."""
    return (
        f"Telemetry Report: Mote {moteid}, Epoch {epoch}. "
        f"Temperature={temp:.2f}C (z-score={z_t:.2f}, 3m-shock={delta_t:+.2f}C, 15m-slope={slope_t:+.3f}C/step). "
        f"Voltage={volt:.2f}V (z-score={z_v:.2f}, 3m-shock={delta_v:+.2f}V). "
        f"Humidity={hum:.1f}%, Light={light:.1f} Lux. "
        f"Fast Path Prior: IF anomaly probability={p_if:.3f}, AE anomaly probability={p_ae:.3f}."
    )

def predict_laya_batch(states: List[str], batch_size: int = 64) -> List[Tuple[str, float]]:
    """
    Evaluates micro-batches of states using real Laya model.
    Returns: List of (choice, p_anomaly).
    """
    agent = get_laya_agent()
    results = []
    total = len(states)
    t_start = time.perf_counter()
    
    for i in range(0, total, batch_size):
        chunk = states[i:i + batch_size]
        batch_out = agent.predict_batch(chunk, QUESTION)
        for item in batch_out:
            ans = item['answers']['is_anomaly']
            choice = ans['choice']
            p_anom = float(ans['probabilities'].get('ANOMALY', 0.5))
            results.append((choice, p_anom))
        
        processed = len(results)
        if processed % 512 == 0 or processed == total:
            elapsed = time.perf_counter() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  [Laya Progress] {processed:,} / {total:,} evaluated ({processed/total*100:.1f}%) - {rate:.1f} items/sec")
            
    return results

if __name__ == "__main__":
    agent = get_laya_agent()
    test_state = format_sensor_state(
        moteid=17, epoch=500,
        temp=38.5, volt=2.15, hum=45.0, light=120.0,
        delta_t=5.2, delta_v=-0.45,
        z_t=3.8, z_v=-3.2,
        slope_t=0.15, var_t=4.2,
        p_if=0.62, p_ae=0.58
    )
    res = predict_laya_batch([test_state])
    print("Test Laya Output:", res)
