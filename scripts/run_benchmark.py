"""Measure fast-path versus BDT through the real local Flink pipeline."""

import csv
import argparse
import json
import time
from pathlib import Path
from run_demo import run_demo

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Compare fast and BDT Flink replay modes.")
    parser.add_argument("--skip-build", action="store_true", help="Use the already built Flink job jar")
    args = parser.parse_args()
    records = []
    for index, mode in enumerate(("fast", "bdt")):
        result = run_demo(mode=mode, limit=40_000, delay=0.0, laya=False,
                          build=(index == 0 and not args.skip_build))
        summary = result["bdt"]
        records.append({
            "run_id": result["run_id"],
            "mode": mode,
            "events": summary.get("total"),
            "f1": summary.get("f1"),
            "precision": summary.get("precision"),
            "recall": summary.get("recall"),
            "fpr": summary.get("fpr"),
            "escalation_rate": summary.get("escalation_rate"),
            "throughput_events_per_second": summary.get("throughput_eps"),
            "latency_p50_ms": summary.get("p50_ms"),
            "latency_p95_ms": summary.get("p95_ms"),
            "latency_p99_ms": summary.get("p99_ms"),
        })
    output = ROOT / "experiments/results/streaming_integration_benchmark.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote measured Flink comparison to {output}")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
