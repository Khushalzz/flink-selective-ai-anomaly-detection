<div align="center">
  <h1>BDT · Bounded Decision Tiering</h1>
  <p><strong>Selective escalation for streaming IoT anomaly detection</strong></p>
  <p>
    <a href="https://flink.apache.org/"><img src="https://img.shields.io/badge/Apache%20Flink-1.20.2-E6526F?logo=apacheflink&logoColor=white" alt="Apache Flink 1.20.2"></a>
    <a href="https://kafka.apache.org/"><img src="https://img.shields.io/badge/Apache%20Kafka-3.9.2-231F20?logo=apachekafka&logoColor=white" alt="Apache Kafka 3.9.2"></a>
    <a href="https://onnxruntime.ai/"><img src="https://img.shields.io/badge/ONNX%20Runtime-Python%20evaluation-005CED?logo=onnx&logoColor=white" alt="Python ONNX Runtime"></a>
    <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/XGBoost-fallback-2088FF?logo=xgboost&logoColor=white" alt="XGBoost"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache 2.0 license"></a>
    <a href=".github/workflows/ci.yml"><img src="https://github.com/Khushalzz/flink-selective-ai-anomaly-detection/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  </p>
</div>

<p align="center">
  <img src="assets/intel-lab-sensor-map.png" width="100%" alt="Intel Lab floor plan with numbered sensor locations">
</p>
<p align="center"><em>Numbered sensor locations across the Intel Lab floor plan.</em></p>

> **Research prototype:** The Java Flink job and the Python model experiments are separate today. The Flink job uses physical thresholds and per-sensor lifetime statistics; it does not load ONNX models or call the Python sidecar. Offline benchmark results below are not Flink throughput measurements.

## What this project explores

Can a fast first-stage detector preserve streaming capacity while escalating only uncertain sensor readings to a more expressive decision model?

This repository studies that question with Intel Lab telemetry, controlled anomaly injection, Python ONNX Runtime, XGBoost, and a separate Java/Flink runtime example.

## Two paths in the repository

| Path | Current behavior |
| --- | --- |
| **Flink runtime** | Kafka → keyed physical thresholds and lifetime Welford statistics → ClickHouse, Redis alerts, and stdout. |
| **Offline model study** | Prepared Parquet → Python ONNX Runtime (Isolation Forest + Autoencoder) → uncertainty gate → XGBoost or Laya evaluation. |

The paths are not yet connected end to end. See [the architecture notes](docs/ARCHITECTURE.md) for the exact implementation boundary.

## Frontend preview

A standalone responsive dashboard preview is available in `dashboard/`. It uses sample data and the Intel Lab floor-plan image; it does not connect to Kafka or Flink.

```bash
python -m http.server 4173 --bind 127.0.0.1
```

Open [http://localhost:4173/dashboard/](http://localhost:4173/dashboard/).

## Offline results

The checked-in evaluation uses 40,000 chronological test windows with synthetic injected anomalies (2.85% prevalence). The scores are offline model results, not results from the Java Flink job.

| System | Precision | Recall | F1 | False-positive rate | Escalated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isolation Forest | 0.1807 | 0.8603 | 0.2986 | 11.43% | 0% |
| IF + Autoencoder | 0.2677 | 0.8989 | 0.4125 | 7.20% | 0% |
| IF + AE → XGBoost | **0.7340** | **0.9385** | **0.8238** | **1.00%** | 20.5% |

The historical System F row in the committed six-system CSV came from a deterministic proxy, not an actual Laya model run. It is marked unverified in the chart and should not be cited as Laya performance. The canonical evaluator calls the real Laya model.

<p align="center">
  <img src="assets/rigorous_6_system_comparison.png" width="100%" alt="Offline detection metrics for six model configurations; the legacy F row is marked as an unverified proxy">
</p>

### Capacity estimate

The sweep chart combines held-out offline F1 scores with a **model-only capacity estimate** calculated from batched Python timings and an assumed transport cap. It does not measure Flink, Kafka, HTTP, or sink throughput.

<p align="center">
  <img src="assets/f1_vs_estimated_capacity.png" width="100%" alt="Offline F1 compared with estimated model-service capacity">
</p>

## Run the offline demo

The repository includes processed Parquet splits and model artifacts for a quick local replay:

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python demo_streaming_pipeline.py --samples 2000
```

This replays prepared feature rows with Python ONNX Runtime and local XGBoost inference. It does not launch Flink, recompute online features, or call the FastAPI sidecar.

## Start the local infrastructure

```bash
docker compose up -d
docker compose ps
```

The Flink dashboard is at [http://localhost:8081](http://localhost:8081). Compose starts the services, but does not build or submit the Flink job.

To build and submit the current Java job:

```bash
cd flink-job
mvn package -DskipTests
cd ..
docker exec -it bdt-flink-jobmanager flink run   -c bdt.Job   /opt/flink/usrlib/bdt-flink-job-1.0.0-shaded.jar   kafka=kafka:9092   topic=intel-lab-sensors   clickhouseUrl=http://clickhouse:8123   redisHost=redis   parallelism=4
```

The job is threshold-based. Checkpoints use at-least-once semantics; ClickHouse and Redis may see duplicates after recovery.

## Reproduce the offline study

The public checkout does **not** include `data_chronological.parquet`, the raw source expected by preprocessing. Add its provenance, license, and checksum before distributing or regenerating the dataset.

```bash
python preprocessing/inject_anomalies.py --validation-seed 42 --test-seed 43
python experiments/verify_no_leakage.py
python models/train_isolation_forest.py
python models/train_autoencoder.py
python models/train_xgboost_fallback.py
python experiments/evaluate_systems.py
python experiments/plot_benchmark.py
python experiments/streaming_benchmark_sweep.py
```

The seed defaults remain `42/42` for the checked-in snapshot. Use distinct seeds for new experiments, retrain the models, and regenerate the results and figures together. The sweep reports estimated capacity only; its CSV labels that estimate explicitly.

## Project guide

- [Architecture and runtime limitations](docs/ARCHITECTURE.md)
- [Benchmark reproduction and measurement scope](docs/BENCHMARK_REPRODUCTION.md)
- [Mathematical formulation](docs/MATHEMATICAL_FORMULATION.md)
- [Interactive offline notebook](notebooks/01_benchmark_walkthrough.ipynb)
- [Tests](tests/)
- [Security policy](SECURITY.md)

## Dataset and citation

The processed train, validation, and test Parquet files are included. The raw chronological input is not. The dataset-source citation and license should be added when its provenance is confirmed.

```bibtex
@software{bdt_streaming_anomaly_2026,
  author = {Khushalzz},
  title = {{Selective Escalation of Uncertain Streaming Sensor Anomalies via Tiered Decision Architectures}},
  year = {2026},
  url = {https://github.com/Khushalzz/flink-selective-ai-anomaly-detection},
  version = {1.0.0}
}
```

Licensed under Apache-2.0. See [LICENSE](LICENSE).
