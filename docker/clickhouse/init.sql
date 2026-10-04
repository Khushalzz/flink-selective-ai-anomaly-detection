CREATE TABLE IF NOT EXISTS default.sensor_readings
(
    timestamp DateTime64(3, 'UTC'),
    date Date,
    epoch UInt64,
    moteid Int32,
    temperature Float32,
    humidity Float32,
    light Float32,
    voltage Float32,
    is_anomaly UInt8,
    anomaly_score Float32,
    anomaly_reasons String
)
ENGINE = ReplacingMergeTree
ORDER BY (moteid, epoch);
