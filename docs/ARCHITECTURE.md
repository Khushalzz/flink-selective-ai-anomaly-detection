# Current System Architecture

This document describes what the checked-in code runs today. The repository has an offline model-evaluation path and a separate Java Flink example; they are not wired together as a deployed BDT pipeline.

## Flink runtime

`flink-job/src/main/java/bdt/Job.java` reads JSON records from Kafka, keys them by mote ID, and runs `AnomalyDetectorFunction`. That function applies fixed physical bounds and Z-score checks against a per-key Welford accumulator. The accumulator is lifetime state; it is not a 15-minute rolling window. The current job does not invoke the ONNX models or the FastAPI sidecar.

The job enables Flink checkpoints in at-least-once mode. Kafka offsets and keyed state can be restored from a checkpoint. Compose stores checkpoint files in the named local `flink_checkpoints` volume. ClickHouse and Redis are not transactional Flink sinks, so recovery may replay records and create duplicates. Downstream consumers should use `(moteid, epoch)` as an event identity where deduplication is needed.

ClickHouse inserts are batched and acknowledged synchronously. A non-success HTTP response fails the sink task so Flink can restart from a checkpoint. The ClickHouse table schema is initialized for new ClickHouse data directories from `docker/clickhouse/init.sql`. Redis write failures are surfaced to Flink rather than silently discarded.

## Offline Python evaluation

`experiments/evaluate_systems.py` loads precomputed feature rows and evaluates Python ONNX Runtime, XGBoost, and Laya models. `demo_streaming_pipeline.py` replays those rows locally, but does not recompute streaming features or communicate with Kafka/Flink/sidecar.

`experiments/streaming_benchmark_sweep.py` computes offline detection scores at selected escalation rates. Its capacity column is an analytical model-service estimate based on batched Python timings and an assumed transport cap; it excludes Flink, Kafka, HTTP, and sink work. It does not include Laya or measured latency percentiles. The estimate is not end-to-end streaming performance.

## FastAPI sidecar

`sidecar/app.py` exposes XGBoost, heuristic, Laya, and batch endpoints. The health response reports model readiness. Batch calls are limited to 256 records, and the engine name is validated. The Flink job currently does not call these endpoints.

## Local services

`docker-compose.yml` starts a single Kafka broker, Flink JobManager and TaskManager, Redis, ClickHouse, and MongoDB. Compose starts the infrastructure only; it does not build or submit the Flink job or start the Python sidecar. The MongoDB service is not currently used by the runtime code.
