package bdt;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

public class Job {
    public static void main(String[] args) throws Exception {
        Map<String, String> config = parseArgs(args);

        String kafkaBootstrap = config.getOrDefault("kafka", "kafka:9092");
        String topic = config.getOrDefault("topic", "intel-lab-sensors");
        String groupId = config.getOrDefault("group", "bdt-anomaly-group-v2");
        String redisHost = config.getOrDefault("redisHost", "redis");
        int redisPort = Integer.parseInt(config.getOrDefault("redisPort", "6379"));
        String clickhouseUrl = config.getOrDefault("clickhouseUrl", "http://clickhouse:8123");
        int parallelism = Integer.parseInt(config.getOrDefault("parallelism", "4"));
        long checkpointIntervalMs = Long.parseLong(config.getOrDefault("checkpointIntervalMs", "10000"));
        if (checkpointIntervalMs <= 0) {
            throw new IllegalArgumentException("checkpointIntervalMs must be greater than zero");
        }

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.enableCheckpointing(checkpointIntervalMs, CheckpointingMode.AT_LEAST_ONCE);
        env.getCheckpointConfig().setCheckpointTimeout(Math.max(60_000L, checkpointIntervalMs * 6));
        env.setParallelism(parallelism);
        env.setRestartStrategy(RestartStrategies.fixedDelayRestart(5, Duration.ofSeconds(2)));

        KafkaSource<SensorReading> kafkaSource = KafkaSource.<SensorReading>builder()
                .setBootstrapServers(kafkaBootstrap)
                .setTopics(topic)
                .setGroupId(groupId)
                .setStartingOffsets(OffsetsInitializer.earliest())
                .setValueOnlyDeserializer(new SensorReadingDeserializer())
                .setProperty("commit.offsets.on.checkpoint", "true")
                .build();

        DataStream<SensorReading> inputStream = env.fromSource(
                kafkaSource,
                WatermarkStrategy.noWatermarks(),
                "KafkaSensorSource"
        ).setParallelism(parallelism);

        // Key by sensor mote ID: load is uniformly partitioned across all 4 worker slots
        DataStream<AnomalyRecord> processedStream = inputStream
                .keyBy(SensorReading::getMoteid)
                .process(new AnomalyDetectorFunction())
                .name("AnomalyDetector")
                .setParallelism(parallelism);

        // Sink 1: ClickHouse (acknowledged micro-batches; backpressure on insert failure)
        processedStream.addSink(new ClickHouseSink(clickhouseUrl))
                .name("ClickHouseSink")
                .setParallelism(parallelism);

        // Sink 2: Redis (Instant alert queue for detected anomalies)
        processedStream.addSink(new RedisAlertSink(redisHost, redisPort))
                .name("RedisAlertSink")
                .setParallelism(parallelism);

        // Sink 3: High-efficiency sampled stdout (only logs true anomalies to avoid Docker console drag)
        processedStream
                .filter(record -> record.getIsAnomaly() == 1)
                .print("ALERT")
                .setParallelism(parallelism);

        env.execute("Optimized Parallel Anomaly Pipeline");
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        for (String arg : args) {
            int idx = arg.indexOf('=');
            if (idx > 0) {
                map.put(arg.substring(0, idx).trim(), arg.substring(idx + 1).trim());
            }
        }
        return map;
    }
}
