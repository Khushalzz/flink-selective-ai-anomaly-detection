CREATE TABLE IF NOT EXISTS default.anomaly_events
(
    run_id String,
    event_id String,
    timestamp DateTime64(3, 'UTC'),
    date Date,
    epoch Int32,
    moteid Int32,
    temperature Float32,
    humidity Float32,
    light Float32,
    voltage Float32,
    ground_truth Nullable(UInt8),
    is_anomaly Nullable(UInt8),
    prediction LowCardinality(String),
    p_if Float32,
    p_ae Float32,
    p_final Nullable(Float32),
    uncertain UInt8,
    escalated UInt8,
    system_name LowCardinality(String),
    engine LowCardinality(String),
    inference_status LowCardinality(String),
    anomaly_score Float32,
    anomaly_reasons String,
    anomaly_type String,
    sent_at_epoch_ms UInt64,
    processed_at_epoch_ms UInt64,
    processing_latency_ms Float32
)
ENGINE = ReplacingMergeTree
ORDER BY (run_id, event_id, engine);
