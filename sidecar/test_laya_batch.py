import time
import laya
import numpy as np

print("Initializing Laya agent...")
agent = laya.load('convaiinnovations/laya', subfolder='typed-decisions', device='cpu')

question = {
    'is_anomaly': {
        'type': 'choice',
        'instructions': 'Does this sensor telemetry indicate normal operation or an anomaly?',
        'criteria': {
            'NORMAL': 'Normal sensor operation adhering to expected baseline',
            'ANOMALY': 'Anomalous outlier, hardware spike, drop, or unexpected behavior'
        }
    }
}

states = [
    "Sensor mote_1: Temp=22.5C, Volt=2.7V. Baseline normal, z=0.1.",
    "Sensor mote_2: Temp=45.8C (z=+4.8), Volt dropped to 2.05V (z=-3.9). Severe outlier.",
    "Sensor mote_3: Temp=20.1C, Volt=2.68V. Baseline normal, z=0.05.",
    "Sensor mote_4: Temp stuck at 19.5C for 45 minutes, variance=0. Flatline sensor fault."
]

t0 = time.perf_counter()
res = agent.predict_batch(states, question)
total_ms = (time.perf_counter() - t0) * 1000.0

print(f"\nBatch of {len(states)} processed in {total_ms:.1f}ms ({total_ms/len(states):.1f}ms per window):")
for i, r in enumerate(res):
    p_anom = r['is_anomaly']['probabilities']['ANOMALY']
    choice = r['is_anomaly']['choice']
    print(f"  Item {i+1}: {choice} (P_anomaly = {p_anom:.4f}) | State: {states[i][:45]}...")
