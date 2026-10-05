# Runtime Architecture

This document describes the current runnable local pipeline. Python remains useful for model training and offline comparisons; the main streaming inference path runs inside Flink on the JVM.

## End-to-end data flow

```text
Labeled Intel Lab replay
        │
        ▼
Kafka topic (one unique topic per run)
        │
        ▼
Flink Kafka source → keyed per-mote feature windows
        │
        ├── Isolation Forest ONNX ─┐
        ├── Autoencoder ONNX ──────┴─ fast decision
        │                              │ uncertain/disagreement
        │                              ▼
        │                      XGBoost ONNX fallback
        │                              │
        └──────────────────────────────┴──► ClickHouse + Redis
                                              │
                                              ▼
                                      Dashboard API + UI

Uncertain events ── optional Flink Async I/O ──► Laya CPU sidecar ──► ClickHouse
```

The default `bdt` job does not make network calls for model inference: Isolation Forest, Autoencoder, and XGBoost run in the Flink JVM through ONNX Runtime. The `fast` mode is the comparison baseline with no XGBoost escalation.

## Feature and decision contract

Each Kafka record carries the raw Intel Lab readings, an event identity, a run identity, the injection label, and a send timestamp. Flink keys state by mote and maintains the same short and long rolling windows used by the offline feature pipeline (6 readings and 30 readings). The 10 model inputs are:

`temperature, humidity, light, voltage, delta_temp_3m, delta_volt_3m, z_temp_15m, z_volt_15m, temp_slope_15m, var_temp_15m`.

The model bundle loads the IF/AE ONNX files and calibration settings once per operator. The gate escalates uncertain scores in the open interval 0.35–0.65 and detector disagreement. Escalated BDT records also include the two detector probabilities for the XGBoost model. Parity tests compare these results with the checked-in Python reference fixture.

## Sinks and recovery

Each run uses a unique Kafka topic and `run_id`, so the dashboard reads only the latest selected run. The job checkpoints every 10 seconds with at-least-once semantics. ClickHouse inserts are acknowledged synchronously in batches and flushed periodically; Redis write failures are surfaced to Flink. Neither sink participates in a cross-system transaction, so a restored job can replay records. The ClickHouse `ReplacingMergeTree` uses `(run_id, event_id, engine)` as its replacement key.

The Compose ClickHouse service uses its existing anonymous data volume. Dashboard startup creates `default.anomaly_events` if it is absent and does not drop or reinitialize existing tables such as `sensor_readings`.

## Dashboard

`dashboard/api.py` serves the static dashboard and a read-only API. It reads model events and aggregate metrics from ClickHouse, run metadata from Redis, and Flink health from the Flink REST endpoint. The UI has a live view and a separate labeled offline preview. Live figures are based on actual rows from the selected run; the preview figures are illustrative offline results.

## Laya comparison branch

The repository contains an optional CPU-only Laya sidecar and Flink comparison route, but this path is experimental. The integrated sensor replay encountered Laya timeouts and backpressure, so the branch is excluded from the verified live demo and throughput benchmark. Laya is also not trained for this sensor dataset. Revisit its queueing, timeout, and sensor accuracy evaluation before relying on it.

## Measurements and limits

`scripts/run_benchmark.py` replays the same held-out split through `fast` and `bdt`, then writes measured ClickHouse-backed throughput, processing latency, F1, precision, and recall to `experiments/results/streaming_integration_benchmark.csv`. These are local Docker measurements, not a cluster capacity guarantee. Report the hardware, Docker limits, replay size, and both model quality and throughput when presenting them. The separate `experiments/streaming_benchmark_sweep.py` remains an offline analytical estimate and must not be described as measured Flink capacity.
