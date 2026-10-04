# Contributing to BDT

Thank you for your interest in contributing to **BDT (Bounded Decision Tiering)**!

## Code of Conduct
We are committed to providing a welcoming, inclusive, and harassment-free environment for all contributors.

## How to Contribute
1. **Fork the repository** on GitHub: [https://github.com/Khushalzz/flink-selective-ai-anomaly-detection](https://github.com/Khushalzz/flink-selective-ai-anomaly-detection)
2. **Create a topic branch**: `git checkout -b feature/my-new-detector`
3. **Set up environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest flake8
   ```
4. **Make your changes** with accompanying unit tests in `tests/`.
5. **Verify zero temporal leakage**:
   ```bash
   python experiments/verify_no_leakage.py
   ```
6. **Run tests**:
   ```bash
   pytest tests/ -v
   ```
7. **Commit using conventional commits** (e.g. `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).
8. **Push to your fork and submit a Pull Request**.

## Benchmarking Protocol
If you are adding a new detector or fallback model:
- All training must be done strictly on `data/processed/train_features.parquet` (unsupervised) or `val_features.parquet` (supervised calibration/fallback).
- Never access `data/processed/test_features.parquet` until final scoring.
- Ensure all predicted anomaly scores are continuous probabilities $P(\text{anomaly}) \in [0, 1]$ to allow valid PR-AUC and ROC-AUC computation.
