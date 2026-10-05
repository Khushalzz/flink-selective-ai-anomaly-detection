"""Replay Intel Lab readings to Kafka with stable per-run event identity."""

import argparse
import os
import sys
import time
import uuid

import orjson
from kafka import KafkaProducer


def iter_parquet_rows(file_path, batch_size):
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(file_path)
    expected = parquet_file.metadata.num_rows
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        values = batch.to_pydict()
        for index in range(len(values["moteid"])):
            date = values["date"][index]
            clock = values["time"][index]
            row = {
                "date": date,
                "time": clock,
                "timestamp": f"{date}T{clock}",
                "epoch": int(values["epoch"][index]),
                "moteid": int(values["moteid"][index]),
                "temperature": float(values["temperature"][index]),
                "humidity": float(values["humidity"][index]),
                "light": float(values["light"][index]),
                "voltage": float(values["voltage"][index]),
            }
            if "label" in values and values["label"][index] is not None:
                row["label"] = int(values["label"][index])
            if "anomaly_type" in values and values["anomaly_type"][index] is not None:
                row["anomaly_type"] = str(values["anomaly_type"][index])
            yield row
    return expected


def iter_text_rows(file_path):
    with open(file_path, "r", encoding="utf-8") as source:
        for line in source:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            date, clock = parts[0], parts[1]
            yield {
                "date": date,
                "time": clock,
                "timestamp": f"{date}T{clock}",
                "epoch": int(parts[2]),
                "moteid": int(parts[3]),
                "temperature": float(parts[4]),
                "humidity": float(parts[5]),
                "light": float(parts[6]),
                "voltage": float(parts[7]),
            }


def run_producer(file_path, bootstrap_servers, topic, delay=0.0, max_records=None,
                 batch_size=2000, run_id=None, redis_host=None, redis_port=6379, laya_enabled=False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Telemetry file not found: {file_path}")
    run_id = run_id or uuid.uuid4().hex
    expected = None
    if file_path.endswith(".parquet"):
        import pyarrow.parquet as pq
        expected = pq.ParquetFile(file_path).metadata.num_rows
    if max_records is not None and expected is not None:
        expected = min(expected, max_records)

    run_store = None
    if redis_host:
        import redis
        run_store = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        run_store.ping()
        run_store.set("bdt:active_run", run_id)
        run_store.hset(f"bdt:run:{run_id}", mapping={
            "status": "starting",
            "expected_count": expected or "",
            "started_at_epoch_ms": int(time.time() * 1000),
            "sent_count": 0,
            "laya_enabled": "1" if laya_enabled else "0",
        })

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers.split(","),
        value_serializer=orjson.dumps,
        key_serializer=lambda key: str(key).encode("utf-8") if key is not None else None,
        acks="all",
        linger_ms=5,
        batch_size=65536,
        compression_type="lz4",
    )
    print(f"Run {run_id}: streaming {file_path} to {topic} at {bootstrap_servers}")
    sent = 0
    started = time.perf_counter()
    rows = iter_parquet_rows(file_path, batch_size) if file_path.endswith(".parquet") else iter_text_rows(file_path)
    try:
        for row in rows:
            sent += 1
            row["run_id"] = run_id
            row["event_id"] = f"{run_id}:{sent}"
            row["sent_at_epoch_ms"] = int(time.time() * 1000)
            producer.send(topic, key=row["moteid"], value=row)
            if delay > 0:
                time.sleep(delay)
            if sent % 1000 == 0:
                elapsed = time.perf_counter() - started
                print(f"sent={sent:,} rate={sent / elapsed:,.0f} events/s")
            if max_records is not None and sent >= max_records:
                break
    except KeyboardInterrupt:
        print("Replay interrupted; flushing records already sent.")
    finally:
        producer.flush()
        producer.close()
        elapsed = time.perf_counter() - started
        if run_store:
            run_store.hset(f"bdt:run:{run_id}", mapping={
                "status": "producer_finished",
                "sent_count": sent,
                "producer_finished_at_epoch_ms": int(time.time() * 1000),
            })
            run_store.close()
        print(f"run_id={run_id} sent={sent:,} elapsed={elapsed:.2f}s rate={sent / elapsed if elapsed else 0:,.0f} events/s")
    return run_id


def main():
    default_file = "data/processed/test_features.parquet"
    if not os.path.exists(default_file):
        default_file = "data_chronological.parquet" if os.path.exists("data_chronological.parquet") else os.path.join("archive (5)", "data.txt")
    parser = argparse.ArgumentParser(description="Replay Intel Lab readings to Kafka.")
    parser.add_argument("--file", default=default_file)
    parser.add_argument("--broker", default="localhost:29092")
    parser.add_argument("--topic", default="intel-lab-sensors")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between records; 0 runs at maximum producer rate")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--redis-host", default=None)
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--laya-enabled", action="store_true")
    args = parser.parse_args()
    run_producer(
        file_path=args.file,
        bootstrap_servers=args.broker,
        topic=args.topic,
        delay=args.delay,
        max_records=args.limit,
        batch_size=args.batch_size,
        run_id=args.run_id,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        laya_enabled=args.laya_enabled,
    )


if __name__ == "__main__":
    main()
