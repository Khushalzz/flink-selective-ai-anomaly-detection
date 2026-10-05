# Experimental Benchmark Reproduction Guide

This guide separates the live Flink integration measurement from the offline Python evaluation and analytical capacity estimate.

---

## 1. Benchmarking Hardware & Software Environment

The repository records the following machine as benchmark context. The offline scripts do not exercise the Flink/Kafka deployment, so this table alone is not evidence of end-to-end streaming measurements:

| Parameter | Specification |
| :--- | :--- |
| **Processor** | Intel Core i5-10400 CPU @ 2.90 GHz (6 Physical Cores, 12 Threads) |
| **System Memory** | 16 GB Dual-Channel DDR4-2666 MHz |
| **Storage Subsystem** | NVMe M.2 Solid State Drive (PCIe 3.0 x4) |
| **Operating System** | Windows 11 Pro / Ubuntu 22.04 LTS via WSL2 |
| **Java Environment** | Eclipse Temurin OpenJDK 17.0.10 (LTS) |
| **Apache Flink** | Version 1.20.2 (Standalone Cluster Mode, 4-6 Worker Slots) |
| **Apache Kafka** | Version 3.9.2 (KRaft Mode, 4 Topic Partitions) |
| **Python Runtimes** | Python 3.11.15 (Conda / Venv) |

## 2. Measured Flink pipeline comparison

To replay the 40,000-row held-out split through both Flink modes and save the measured results:

```bash
python scripts/run_benchmark.py
```

The script builds and starts the local stack, then runs `fast` and `bdt` on unique Kafka topics. It records ClickHouse-backed throughput and processing latency alongside label-based precision, recall, F1, false-positive rate, and escalation rate in `experiments/results/streaming_integration_benchmark.csv`. These measurements describe this local Docker run only; include the host and Docker memory/CPU limits with results you share. Startup/model warm-up and input pacing can change latency, so do not compare this figure with offline batch inference time.

Latest recorded 40,000-event run:

| Mode | F1 | FPR | Escalation | Throughput | p95 end-to-end latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fast | 0.4125 | 7.20% | 0% | 1,283 events/s | 29.86 s |
| BDT | 0.8238 | 1.00% | 20.51% | 1,332 events/s | 28.99 s |

One run per mode is a local snapshot, not a statistical comparison. The replay producer is unpaced, so the p95 latency includes queueing while Kafka input exceeds Flink's sustained rate; it is not ONNX model inference time.

For one smaller live run and its dashboard, use:

```bash
python scripts/run_demo.py --mode bdt --limit 5000
```

The live view is [http://localhost:4173/dashboard/](http://localhost:4173/dashboard/); Flink's job manager is [http://localhost:8081](http://localhost:8081).

## 3. Offline evaluation hardware context

The machine below is the recorded context for the offline experiments. It is not necessarily the current host used for the Flink integration measurements.

### Thread Budgeting Context
The following limits are intended for a separately launched sidecar; this repository does not currently connect it to the Flink job:
- **Flink TaskManager:** Allocated **4 task slots** in the checked-in Compose file.
- **Python Fallback Sidecar:** Budgeted to **4 threads maximum**:
  ```bash
  export OMP_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  ```

---

## 4. Zero-Leakage Dataset Verification

Preprocessing requires `data_chronological.parquet`, which is not present in the public repository checkout. Obtain the source dataset and verify its provenance/checksum before generating splits. Use distinct injection seeds for validation and test in new runs; the defaults remain 42 for both to reproduce the checked-in snapshot. Then verify chronological boundaries:

```bash
python experiments/verify_no_leakage.py
```

### Expected Output
```text
======================================================================
TEMPORAL LEAKAGE AUDIT: CHRONOLOGICAL BOUNDARY VERIFICATION
======================================================================
Train Split: 120,000 samples | 2004-02-28 00:58:46 -> 2004-02-29 09:44:24
Val Split:    40,000 samples | 2004-02-29 09:44:24 -> 2004-02-29 20:06:26
Test Split:   40,000 samples | 2004-02-29 20:06:27 -> 2004-03-01 06:45:23

[PASSED] Strict temporal boundary: max(Train) <= min(Val)
[PASSED] Strict temporal boundary: max(Val)   <= min(Test)
[PASSED] Clean unsupervised training: 0 anomalies in Train split.
[PASSED] Target ground-truth test prevalence: 1,138 anomalies (2.85%).
======================================================================
AUDIT RESULT: 100% ZERO TEMPORAL DATA LEAKAGE CONFIRMED.
======================================================================
```

---

## 5. Executing the 6-System Comprehensive Evaluation

To run the canonical offline evaluator across the 40,000 held-out test windows (including the Laya model call):

```bash
python experiments/evaluate_systems.py
```

This generates `experiments/results/rigorous_6_system_results.csv` and outputs the master comparison table.

*The time column is batch elapsed time divided by batch size. It excludes Flink, Kafka, HTTP, and sink time; do not read it as streaming latency.*

### Historical result snapshot

The checked-in System F row was generated by a legacy deterministic proxy, not the Laya model. It is labeled unverified and should not be cited as Laya performance. The canonical evaluator now calls the real Laya model.

| System | Precision | Recall | $F_1$-Score | PR-AUC | ROC-AUC | FPR (%) | Batch-average model time* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Isolation Forest** | 0.1807 | 0.8603 | **0.2986** | 0.6908 | 0.9581 | 11.43% | $130.5\ \mu\text{s}$ |
| **B: Autoencoder** | 0.3383 | 0.6924 | **0.4546** | 0.6255 | 0.9481 | 3.97% | $0.5\ \mu\text{s}$ |
| **C: IF + AE Ensemble** | 0.2677 | 0.8989 | **0.4125** | 0.6896 | 0.9690 | 7.20% | $131.0\ \mu\text{s}$ |
| **D: IF + AE $\rightarrow$ XGBoost** | **0.7340** | **0.9385** | **0.8238** | **0.9227** | **0.9833** | **1.00%** | **$131.2\ \mu\text{s}$** |
| **E: IF + AE $\rightarrow$ Heuristic** | 0.1700 | 0.9596 | **0.2888** | 0.6282 | 0.9719 | 13.72% | $131.2\ \mu\text{s}$ |
| **F: legacy Laya proxy (unverified)** | 0.2057 | 0.9561 | **0.3385** | 0.8054 | 0.9795 | 10.81% | **$171.0\text{ ms}$** |

---

## 6. Offline escalation sweep and capacity estimates

To evaluate offline F1 at selected escalation rates and calculate model-only capacity estimates:

```bash
python experiments/streaming_benchmark_sweep.py
```

Outputs:
- Data table: `experiments/results/escalation_sweep_results.csv`
- Offline F1 vs. estimated model capacity: `assets/f1_vs_estimated_capacity.png`

Capacity is derived from batched Python model timings and an assumed transport cap. It is not measured Flink throughput, and the output does not claim measured p95/p99 latency.

To generate the offline model comparison figure:
```bash
python experiments/plot_benchmark.py
```
Output:
- `assets/rigorous_6_system_comparison.png`

---

## 7. Running the Standalone Python Replay

For an offline replay without Docker:

```bash
# Replay 5,000 prepared feature rows through Python inference
python demo_streaming_pipeline.py --samples 5000
```
