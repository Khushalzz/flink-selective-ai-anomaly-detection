package bdt;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.AsyncDataStream;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.OutputTag;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class Job {
    private static final OutputTag<LayaRequest> LAYA_REQUESTS = new OutputTag<LayaRequest>("laya-requests") {};

    public static void main(String[] args) throws Exception {
        Map<String, String> config = parseArgs(args);
        String kafkaBootstrap = config.getOrDefault("kafka", "kafka:9092");
        String topic = config.getOrDefault("topic", "intel-lab-sensors");
        String groupId = config.getOrDefault("group", "bdt-run-group");
        String redisHost = config.getOrDefault("redisHost", "redis");
        int redisPort = Integer.parseInt(config.getOrDefault("redisPort", "6379"));
        String clickhouseUrl = config.getOrDefault("clickhouseUrl", "http://clickhouse:8123");
        String modelDirectory = config.getOrDefault("modelDir", "/workspace/models");
        String layaUrl = config.getOrDefault("layaUrl", "http://laya-sidecar:8000/decide/laya");
        boolean layaEnabled = Boolean.parseBoolean(config.getOrDefault("layaEnabled", "false"));
        String mode = config.getOrDefault("mode", "bdt").toLowerCase();
        if (!mode.equals("bdt") && !mode.equals("fast")) {
            throw new IllegalArgumentException("mode must be fast or bdt");
        }
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

        DataStream<SensorReading> input = env.fromSource(
                kafkaSource, WatermarkStrategy.noWatermarks(), "KafkaSensorSource")
                .setParallelism(parallelism);

        SingleOutputStreamOperator<AnomalyRecord> bdtResults = input
                .keyBy(SensorReading::getMoteid)
                .process(new AnomalyDetectorFunction(modelDirectory, mode.equals("bdt"), layaEnabled, LAYA_REQUESTS))
                .name("JVMModelScoring")
                .setParallelism(parallelism);

        bdtResults.addSink(new ClickHouseSink(clickhouseUrl))
                .name("ClickHouseBdtSink")
                .setParallelism(parallelism);
        bdtResults.addSink(new RedisAlertSink(redisHost, redisPort))
                .name("RedisBdtAlertSink")
                .setParallelism(parallelism);
        bdtResults.filter(record -> Integer.valueOf(1).equals(record.getIsAnomaly()))
                .print("BDT_ALERT")
                .setParallelism(parallelism);

        if (layaEnabled) {
            DataStream<LayaRequest> uncertain = bdtResults.getSideOutput(LAYA_REQUESTS);
            DataStream<AnomalyRecord> layaResults = AsyncDataStream.unorderedWait(
                    uncertain,
                    new LayaAsyncFunction(layaUrl),
                    15,
                    TimeUnit.SECONDS,
                    1
            );
            layaResults.addSink(new ClickHouseSink(clickhouseUrl))
                    .name("ClickHouseLayaComparisonSink")
                    .setParallelism(parallelism);
        }

        env.execute("BDT " + mode.toUpperCase() + " Pipeline");
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        for (String arg : args) {
            int idx = arg.indexOf('=');
            if (idx > 0) map.put(arg.substring(0, idx).trim(), arg.substring(idx + 1).trim());
        }
        return map;
    }
}
