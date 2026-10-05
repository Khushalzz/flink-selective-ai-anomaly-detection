package bdt;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class ClickHouseSink extends RichSinkFunction<AnomalyRecord> {
    private static final long serialVersionUID = 1L;
    private static final long FLUSH_INTERVAL_MS = 1_000L;

    private final String clickhouseUrl;
    private final int batchSize;
    private transient HttpClient httpClient;
    private transient ObjectMapper objectMapper;
    private transient List<String> buffer;
    private transient ScheduledExecutorService flushScheduler;
    private transient volatile String backgroundFlushFailure;

    public ClickHouseSink(String clickhouseUrl, int batchSize) {
        this.clickhouseUrl = clickhouseUrl;
        this.batchSize = batchSize;
    }

    public ClickHouseSink(String clickhouseUrl) {
        this(clickhouseUrl, 1000);
    }

    @Override
    public void open(OpenContext openContext) {
        httpClient = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
        objectMapper = new ObjectMapper();
        buffer = new ArrayList<>(batchSize);
        flushScheduler = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "clickhouse-periodic-flush");
            thread.setDaemon(true);
            return thread;
        });
        flushScheduler.scheduleAtFixedRate(() -> {
            try {
                flushBuffer();
            } catch (Exception error) {
                backgroundFlushFailure = error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage();
            }
        }, FLUSH_INTERVAL_MS, FLUSH_INTERVAL_MS, TimeUnit.MILLISECONDS);
    }

    @Override
    public void invoke(AnomalyRecord record, Context context) throws Exception {
        synchronized (this) {
            if (backgroundFlushFailure != null) {
                throw new IOException("Previous ClickHouse insert failed: " + backgroundFlushFailure);
            }
            buffer.add(objectMapper.writeValueAsString(toRow(record)));
            if (buffer.size() >= batchSize) flushBuffer();
        }
    }

    private Map<String, Object> toRow(AnomalyRecord record) {
        Map<String, Object> row = new HashMap<>();
        String timestamp = record.getTimestamp();
        if (timestamp != null) {
            timestamp = timestamp.replace("T", " ");
            if (timestamp.length() > 23) timestamp = timestamp.substring(0, 23);
        } else {
            timestamp = record.getDate() + " " + record.getTime();
        }
        row.put("run_id", record.getRunId());
        row.put("event_id", record.getEventId());
        row.put("timestamp", timestamp);
        row.put("date", record.getDate());
        row.put("epoch", record.getEpoch());
        row.put("moteid", record.getMoteid());
        row.put("temperature", record.getTemperature());
        row.put("humidity", record.getHumidity());
        row.put("light", record.getLight());
        row.put("voltage", record.getVoltage());
        row.put("ground_truth", record.getGroundTruthLabel());
        row.put("is_anomaly", record.getIsAnomaly());
        row.put("prediction", record.getPrediction());
        row.put("p_if", record.getPIf());
        row.put("p_ae", record.getPAe());
        row.put("p_final", record.getPFinal());
        row.put("uncertain", record.isUncertain() ? 1 : 0);
        row.put("escalated", record.isEscalated() ? 1 : 0);
        row.put("system_name", record.getSystem());
        row.put("engine", record.getEngine());
        row.put("inference_status", record.getInferenceStatus());
        row.put("anomaly_score", record.getAnomalyScore());
        row.put("anomaly_reasons", record.getAnomalyReasons() == null ? "" : record.getAnomalyReasons());
        row.put("anomaly_type", record.getAnomalyType() == null ? "" : record.getAnomalyType());
        row.put("sent_at_epoch_ms", record.getSentAtEpochMs());
        row.put("processed_at_epoch_ms", record.getProcessedAtEpochMs());
        row.put("processing_latency_ms", record.getProcessingLatencyMs());
        return row;
    }

    private synchronized void flushBuffer() throws IOException, InterruptedException {
        if (buffer == null || buffer.isEmpty()) return;
        String body = String.join("\n", buffer) + "\n";
        String query = URLEncoder.encode("INSERT INTO anomaly_events FORMAT JSONEachRow", StandardCharsets.UTF_8);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(clickhouseUrl + "/?query=" + query))
                .timeout(Duration.ofSeconds(10))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new IOException("ClickHouse insert failed (HTTP " + response.statusCode() + "): " + response.body());
        }
        buffer.clear();
        backgroundFlushFailure = null;
    }

    @Override
    public void close() throws Exception {
        if (flushScheduler != null) {
            flushScheduler.shutdown();
            if (!flushScheduler.awaitTermination(15, TimeUnit.SECONDS)) flushScheduler.shutdownNow();
        }
        flushBuffer();
        if (backgroundFlushFailure != null) {
            throw new IOException("ClickHouse periodic flush failed: " + backgroundFlushFailure);
        }
    }
}
