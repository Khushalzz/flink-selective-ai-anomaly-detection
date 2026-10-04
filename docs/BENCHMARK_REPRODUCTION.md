# Experimental Benchmark Reproduction Guide

This guide details the exact environment specifications, dependency matrix, and step-by-step procedure required to replicate all experimental benchmarks and publication figures reported in the BDT study.

---

## 1. Benchmarking Hardware & Software Environment

All official benchmark figures reported in this study were measured on the following hardware platform:

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

### Thread Budgeting Policy
To guarantee that background inference processes do not cause CPU starvation or thread thrashing against Flink TaskManager worker slots:
- **Flink TaskManager:** Allocated **6 task slots** (`taskmanager.numberOfTaskSlots: 6`).
- **Python Fallback Sidecar:** Budgeted to **4 threads maximum**:
  ```bash
  export OMP_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  ```

---

## 2. Zero-Leakage Dataset Verification

Before executing any benchmark scripts, verify the integrity of the chronological train/val/test partitioning:

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

## 3. Executing the 6-System Comprehensive Evaluation

To evaluate all 6 system configurations across the 40,000 held-out test windows:

```bash
python experiments/compile_master_results.py
```

This generates `experiments/results/rigorous_6_system_results.csv` and outputs the master comparison table.

### Expected Master Results

| System | Precision | Recall | $F_1$-Score | PR-AUC | ROC-AUC | FPR (%) | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Isolation Forest** | 0.1807 | 0.8603 | **0.2986** | 0.6908 | 0.9581 | 11.43% | $130.5\ \mu\text{s}$ |
| **B: Autoencoder** | 0.3383 | 0.6924 | **0.4546** | 0.6255 | 0.9481 | 3.97% | $0.5\ \mu\text{s}$ |
| **C: IF + AE Ensemble** | 0.2677 | 0.8989 | **0.4125** | 0.6896 | 0.9690 | 7.20% | $131.0\ \mu\text{s}$ |
| **D: IF + AE $\rightarrow$ XGBoost** | **0.7340** | **0.9385** | **0.8238** | **0.9227** | **0.9833** | **1.00%** | **$131.2\ \mu\text{s}$** |
| **E: IF + AE $\rightarrow$ Heuristic** | 0.1700 | 0.9596 | **0.2888** | 0.6282 | 0.9719 | 13.72% | $131.2\ \mu\text{s}$ |
| **F: IF + AE $\rightarrow$ Real Laya** | 0.2057 | 0.9561 | **0.3385** | 0.8054 | 0.9795 | 10.81% | **$171.0\text{ ms}$** |

---

## 4. Replicating the Escalation Rate Sweep & Frontier Plots

To evaluate the streaming throughput frontier across varying escalation thresholds ($0\%$ to $100\%$):

```bash
python experiments/streaming_benchmark_sweep.py
```

Outputs:
- Data table: `experiments/results/escalation_sweep_results.csv`
- Frontier chart: `assets/f1_vs_throughput_tradeoff.png`
- Latency percentiles: `assets/latency_distribution_by_system.png`

To generate the 4-panel comparison figure:
```bash
python experiments/plot_benchmark.py
```
Output:
- `assets/rigorous_6_system_comparison.png`

---

## 5. Running the Standalone Interactive Simulation

For an immediate demonstration without Docker:

```bash
# Process 5,000 streaming events with live terminal telemetry
python demo_streaming_pipeline.py --samples 5000
```
