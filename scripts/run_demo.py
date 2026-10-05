"""Build and run one reproducible local BDT replay."""

import argparse
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_URL = "http://localhost:4173/dashboard/"
FLINK_URL = "http://localhost:8081"


def compose(*args, check=True, capture=False):
    command = ["docker", "compose", *args]
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=check, text=True,
                          capture_output=capture)


def get_json(url, timeout=2):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_until_ready(url, description, timeout=180):
    end = time.time() + timeout
    last_error = None
    while time.time() < end:
        try:
            get_json(url, timeout=3)
            print(f"Ready: {description}", flush=True)
            return
        except Exception as error:
            last_error = error
            time.sleep(2)
    raise RuntimeError(f"Timed out waiting for {description}: {last_error}")


def cancel_previous_demo_jobs():
    try:
        jobs = get_json(f"{FLINK_URL}/jobs/overview").get("jobs", [])
    except Exception:
        return
    for job in jobs:
        state = job.get("state", job.get("status"))
        if state == "RUNNING" and str(job.get("name", "")).startswith("BDT "):
            compose("exec", "-T", "jobmanager", "flink", "cancel", job["jid"], check=False)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            running = [job for job in get_json(f"{FLINK_URL}/jobs/overview").get("jobs", [])
                       if job.get("state", job.get("status")) == "RUNNING"
                       and str(job.get("name", "")).startswith("BDT ")]
        except Exception:
            running = []
        if not running:
            return
        time.sleep(1)
    raise RuntimeError("Previous BDT replay jobs did not stop; refusing to start another run")


def create_kafka_topic(topic):
    command = [
        "exec", "-T", "kafka", "/opt/kafka/bin/kafka-topics.sh",
        "--bootstrap-server", "kafka:9092", "--create", "--if-not-exists",
        "--topic", topic, "--partitions", "4", "--replication-factor", "1",
    ]
    for attempt in range(30):
        result = compose(*command, check=False, capture=True)
        if result.returncode == 0:
            print((result.stdout or result.stderr or "Kafka topic is ready.").strip(), flush=True)
            return
        if attempt == 29:
            raise RuntimeError(f"Could not create Kafka topic {topic}: {result.stderr or result.stdout}")
        time.sleep(2)


def wait_for_job(job_id, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        try:
            job = get_json(f"{FLINK_URL}/jobs/{job_id}")
            state = job.get("state")
            if state == "RUNNING":
                print(f"Ready: Flink job {job_id}", flush=True)
                return
            if state in {"FAILED", "CANCELED", "FINISHED"}:
                try:
                    failure = get_json(f"{FLINK_URL}/jobs/{job_id}/exceptions").get("root-exception", "")
                except Exception:
                    failure = ""
                raise RuntimeError(f"Flink job entered {state}: {failure[:1200]}")
        except urllib.error.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"Flink job {job_id} did not become ready within {timeout}s")


def run_demo(mode="bdt", limit=5000, delay=0.002, laya=False, build=True):
    if mode not in {"fast", "bdt"}:
        raise ValueError("mode must be fast or bdt")
    run_id = uuid.uuid4().hex
    topic = f"bdt-{run_id[:12]}"

    compose("up", "-d", "kafka", "redis", "clickhouse", "jobmanager", "taskmanager", "dashboard")
    wait_until_ready(f"{FLINK_URL}/overview", "Flink REST API")
    wait_until_ready("http://localhost:4173/api/health", "dashboard API")

    if build:
        compose("--profile", "tools", "run", "--rm", "flink-builder")

    if laya:
        compose("--profile", "laya", "up", "-d", "laya-sidecar")
        wait_until_ready("http://localhost:8000/health", "optional Laya sidecar", timeout=900)
        health = get_json("http://localhost:8000/health")
        if not health.get("laya_ready"):
            raise RuntimeError("Laya sidecar is reachable but the model is not ready")

    # Build the producer before starting Flink so image setup time cannot inflate
    # end-to-end event latency or leave records queued behind a cold model.
    compose("--profile", "demo", "build", "producer")
    create_kafka_topic(topic)
    cancel_previous_demo_jobs()
    jar = "/opt/flink/usrlib/bdt-flink-job-1.0.0.jar"
    submission = compose(
        "exec", "-T", "jobmanager", "flink", "run", "-d", "-c", "bdt.Job", jar,
        "kafka=kafka:9092", f"topic={topic}", "clickhouseUrl=http://clickhouse:8123",
        "redisHost=redis", "parallelism=4", f"group=bdt-{run_id[:12]}",
        "modelDir=/workspace/models", f"mode={mode}", f"layaEnabled={str(laya).lower()}",
        "layaUrl=http://laya-sidecar:8000/decide/laya",
        capture=True,
    )
    submission_output = (submission.stdout or "") + (submission.stderr or "")
    print(submission_output, end="", flush=True)
    job_match = re.search(r"Job has been submitted with JobID ([0-9a-f]+)", submission_output)
    if not job_match:
        raise RuntimeError("Flink did not return a submitted job ID")
    wait_for_job(job_match.group(1))

    producer_args = [
        "--file", "data/processed/test_features.parquet",
        "--broker", "kafka:9092", "--topic", topic,
        "--delay", str(delay), "--run-id", run_id, "--redis-host", "redis",
    ]
    if limit is not None:
        producer_args += ["--limit", str(limit)]
    if laya:
        producer_args.append("--laya-enabled")
    compose("--profile", "demo", "run", "--rm", "producer", *producer_args)

    expected = limit or 40_000
    summary_url = f"http://localhost:4173/api/summary?system={mode}"
    end = time.time() + 300
    summary = {}
    while time.time() < end:
        try:
            summary = get_json(summary_url)
            if summary.get("run_id") == run_id and summary.get("total", 0) >= expected:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        raise RuntimeError(f"Run {run_id} has not reached {expected} ClickHouse rows; see dashboard and container logs")

    laya_summary = None
    if laya:
        end = time.time() + 900
        while time.time() < end:
            laya_summary = get_json("http://localhost:4173/api/summary?system=laya")
            if laya_summary.get("run_id") == run_id and laya_summary.get("status") in {"complete", "disabled"}:
                break
            time.sleep(2)

    result = {"run_id": run_id, "topic": topic, "mode": mode, "bdt": summary, "laya": laya_summary}
    print(json.dumps(result, indent=2))
    print(f"Dashboard: {DASHBOARD_URL}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run a local end-to-end BDT replay.")
    parser.add_argument("--mode", choices=["fast", "bdt"], default="bdt")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--delay", type=float, default=0.002)
    parser.add_argument("--laya", action="store_true", help="Also score uncertain events through the optional async Laya branch")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    run_demo(args.mode, args.limit, args.delay, args.laya, not args.skip_build)


if __name__ == "__main__":
    main()
