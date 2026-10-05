"""Read-only API and static app for the live BDT dashboard."""

import json
import os
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import redis
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = Path(os.environ.get("ASSET_DIR", "/app/assets"))
SCHEMA_PATH = Path(os.environ.get("SCHEMA_PATH", "/app/init.sql"))
CLICKHOUSE_URL = os.environ.get("CLICKHOUSE_URL", "http://clickhouse:8123").rstrip("/")
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
FLINK_REST_URL = os.environ.get("FLINK_REST_URL", "http://jobmanager:8081").rstrip("/")

app = FastAPI(title="BDT Dashboard API", version="1.0.0")
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_timeout=2)


def clickhouse_request(query: str, timeout: float = 5.0) -> str:
    request = Request(
        f"{CLICKHOUSE_URL}/?query={quote(query)}",
        data=b"",
        headers={"Content-Type": "text/plain; charset=utf-8"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def query_rows(query: str) -> list[dict]:
    normalized = query.strip().rstrip(";")
    if " FORMAT " not in normalized.upper():
        normalized += " FORMAT JSONEachRow"
    body = clickhouse_request(normalized)
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def active_run_id() -> str | None:
    run_id = redis_client.get("bdt:active_run")
    if not run_id or len(run_id) > 80 or not all(c.isalnum() or c in "-_" for c in run_id):
        return None
    return run_id


def system_clause(system: str) -> str:
    if system in {"fast", "bdt", "laya"}:
        return f"system_name = '{system}'"
    raise HTTPException(status_code=422, detail="system must be fast, bdt, or laya")


@app.on_event("startup")
def initialize_schema():
    if not SCHEMA_PATH.exists():
        print(f"[Dashboard] Schema file missing: {SCHEMA_PATH}")
        return
    last_error = None
    for _ in range(30):
        try:
            clickhouse_request(SCHEMA_PATH.read_text(encoding="utf-8"), timeout=3)
            print("[Dashboard] ClickHouse schema is ready.")
            return
        except Exception as error:
            last_error = error
            time.sleep(1)
    print(f"[Dashboard] ClickHouse schema initialization deferred: {last_error}")


@app.get("/api/health")
def health():
    checks = {"redis": False, "clickhouse": False, "flink": False}
    try:
        checks["redis"] = bool(redis_client.ping())
    except Exception:
        pass
    try:
        clickhouse_request("SELECT 1")
        checks["clickhouse"] = True
    except Exception:
        pass
    try:
        request = Request(f"{FLINK_REST_URL}/overview", method="GET")
        with urlopen(request, timeout=2) as response:
            overview = json.loads(response.read().decode("utf-8"))
        checks["flink"] = True
    except Exception:
        overview = None
    return {
        "status": "ok" if all(checks.values()) else "degraded",
        "checks": checks,
        "run_id": active_run_id(),
        "flink": overview,
    }


@app.get("/api/summary")
def summary(system: str = "bdt"):
    run_id = active_run_id()
    if not run_id:
        return {"run_id": None, "status": "idle", "system": system, "total": 0, "anomalies": 0,
                "escalated": 0, "escalation_rate": 0, "throughput_eps": 0, "f1": None,
                "precision": None, "recall": None, "fpr": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    clause = system_clause(system)
    try:
        rows = query_rows(f"""
            SELECT
                count() AS total,
                countIf(is_anomaly = 1) AS anomalies,
                countIf(uncertain = 1) AS uncertain,
                countIf(escalated = 1) AS escalated,
                countIf(ground_truth = 1 AND is_anomaly = 1) AS tp,
                countIf(ground_truth = 0 AND is_anomaly = 1) AS fp,
                countIf(ground_truth = 1 AND is_anomaly = 0) AS fn,
                countIf(ground_truth = 0 AND is_anomaly = 0) AS tn,
                quantileTDigestOrDefault(0.50)(processing_latency_ms) AS p50_ms,
                quantileTDigestOrDefault(0.95)(processing_latency_ms) AS p95_ms,
                quantileTDigestOrDefault(0.99)(processing_latency_ms) AS p99_ms,
                min(processed_at_epoch_ms) AS first_ms,
                max(processed_at_epoch_ms) AS last_ms
            FROM default.anomaly_events FINAL
            WHERE run_id = '{run_id}' AND {clause}
        """)
        row = rows[0] if rows else {}
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"ClickHouse query failed: {error}") from error

    total = int(row.get("total", 0))
    anomalies = int(row.get("anomalies", 0))
    uncertain = int(row.get("uncertain", 0))
    escalated = int(row.get("escalated", 0))
    tp, fp = int(row.get("tp", 0)), int(row.get("fp", 0))
    fn, tn = int(row.get("fn", 0)), int(row.get("tn", 0))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    fpr = fp / (fp + tn) if fp + tn else None
    first_ms, last_ms = int(row.get("first_ms", 0)), int(row.get("last_ms", 0))
    throughput = total / ((last_ms - first_ms) / 1000.0) if total > 1 and last_ms > first_ms else 0.0
    run_meta = redis_client.hgetall(f"bdt:run:{run_id}")
    expected_total = int(run_meta.get("expected_count") or 0)
    laya_enabled = run_meta.get("laya_enabled") == "1"
    producer_done = run_meta.get("status") == "producer_finished"
    if system == "laya":
        try:
            comparison_rows = query_rows(f"SELECT countIf(uncertain = 1) AS expected FROM default.anomaly_events FINAL WHERE run_id = '{run_id}' AND system_name IN ('fast', 'bdt')")
            expected = int(comparison_rows[0].get("expected", 0)) if comparison_rows else 0
        except Exception:
            expected = 0
        run_status = "disabled" if not laya_enabled else ("complete" if producer_done and total >= expected else "running")
    else:
        expected = expected_total
        run_status = "complete" if producer_done and (not expected or total >= expected) else "running"
    return {
        "run_id": run_id,
        "status": run_status,
        "system": system,
        "total": total,
        "expected": expected,
        "anomalies": anomalies,
        "ground_truth_count": tp + fp + fn + tn,
        "uncertain": uncertain,
        "escalated": escalated,
        "escalation_rate": escalated / total if total else 0.0,
        "throughput_eps": round(throughput, 2),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "p50_ms": float(row.get("p50_ms") or 0),
        "p95_ms": float(row.get("p95_ms") or 0),
        "p99_ms": float(row.get("p99_ms") or 0),
    }


@app.get("/api/events")
def events(limit: int = Query(default=50, ge=1, le=250), mote_id: int | None = None,
           system: str = "bdt", priority: str = "all"):
    run_id = active_run_id()
    if not run_id:
        return {"run_id": None, "events": []}
    clause = system_clause(system)
    if priority not in {"all", "high", "review"}:
        raise HTTPException(status_code=422, detail="priority must be all, high, or review")
    extra = ""
    if mote_id is not None:
        extra += f" AND moteid = {int(mote_id)}"
    if priority == "high":
        extra += " AND is_anomaly = 1"
    elif priority == "review":
        extra += " AND escalated = 1"
    try:
        rows = query_rows(f"""
            SELECT run_id, event_id, timestamp, moteid, epoch, temperature, humidity, light, voltage,
                   ground_truth, prediction, is_anomaly, p_if, p_ae, p_final, escalated, engine,
                   inference_status, anomaly_reasons, processing_latency_ms
            FROM default.anomaly_events FINAL
            WHERE run_id = '{run_id}' AND {clause}{extra}
            ORDER BY processed_at_epoch_ms DESC
            LIMIT {limit}
            FORMAT JSONEachRow
        """)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"ClickHouse query failed: {error}") from error
    return {"run_id": run_id, "events": rows}


@app.get("/api/series")
def series(window: str = "24h", system: str = "bdt"):
    run_id = active_run_id()
    if not run_id:
        return {"run_id": None, "series": []}
    window_config = {"1h": (3600, 60), "6h": (21600, 300), "24h": (86400, 1800)}
    if window not in window_config:
        raise HTTPException(status_code=422, detail="window must be 1h, 6h, or 24h")
    seconds, bucket_seconds = window_config[window]
    clause = system_clause(system)
    try:
        rows = query_rows(f"""
            SELECT
                toUnixTimestamp(toStartOfInterval(toDateTime(intDiv(processed_at_epoch_ms, 1000)), INTERVAL {bucket_seconds} SECOND)) * 1000 AS bucket_ms,
                count() AS readings,
                countIf(is_anomaly = 1) AS anomalies,
                countIf(escalated = 1) AS escalated
            FROM default.anomaly_events FINAL
            WHERE run_id = '{run_id}' AND {clause}
              AND processed_at_epoch_ms >= (SELECT max(processed_at_epoch_ms) - {seconds * 1000} FROM default.anomaly_events WHERE run_id = '{run_id}')
            GROUP BY bucket_ms
            ORDER BY bucket_ms
            FORMAT JSONEachRow
        """)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"ClickHouse query failed: {error}") from error
    return {"run_id": run_id, "window": window, "series": rows}


@app.get("/")
def home():
    return RedirectResponse(url="/dashboard/")


app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="assets")
app.mount("/dashboard", StaticFiles(directory=APP_DIR, html=True), name="dashboard")
