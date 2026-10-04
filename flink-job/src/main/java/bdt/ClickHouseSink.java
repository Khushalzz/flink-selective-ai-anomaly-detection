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

public class ClickHouseSink extends RichSinkFunction<AnomalyRecord> {
    private static final long serialVersionUID = 1L;
    private static final long FLUSH_INTERVAL_MS = 1_000L;

    private final String clickhouseUrl;
    private final int batchSize;
    private transient HttpClient httpClient;
    private transient ObjectMapper objectMapper;
    private transient List<String> buffer;
    private transient long lastFlushTime;

    public ClickHouseSink(String clickhouseUrl, int batchSize) {
        this.clickhouseUrl = clickhouseUrl;
        this.batchSize = batchSize;
    }

    public ClickHouseSink(String clickhouseUrl) {
        this(clickhouseUrl, 2500);
    }

    @Override
    public void open(OpenContext openContext) {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
        this.objectMapper = new ObjectMapper();
        this.buffer = new ArrayList<>(batchSize);
        this.lastFlushTime = System.currentTimeMillis();
    }

    @Override
    public void invoke(AnomalyRecord record, Context context) throws Exception {
        Map<String, Object> map = new HashMap<>();
        String ts = record.getTimestamp();
        if (ts != null) {
            ts = ts.replace("T", " ");
            if (ts.length() > 23) {
                ts = ts.substring(0, 23);
            }
        } else {
            ts = record.getDate() + " " + record.getTime();
        }
        map.put("timestamp", ts);
        map.put("date", record.getDate());
        map.put("epoch", record.getEpoch());
        map.put("moteid", record.getMoteid());
        map.put("temperature", (float) record.getTemperature());
        map.put("humidity", (float) record.getHumidity());
        map.put("light", (float) record.getLight());
        map.put("voltage", (float) record.getVoltage());
        map.put("is_anomaly", record.getIsAnomaly());
        map.put("anomaly_score", (float) record.getAnomalyScore());
        map.put("anomaly_reasons", record.getAnomalyReasons() != null ? record.getAnomalyReasons() : "");

        buffer.add(objectMapper.writeValueAsString(map));
        long now = System.currentTimeMillis();
        if (buffer.size() >= batchSize || now - lastFlushTime >= FLUSH_INTERVAL_MS) {
            flushBuffer();
        }
    }

    private void flushBuffer() throws IOException, InterruptedException {
        if (buffer == null || buffer.isEmpty()) {
            return;
        }

        String body = String.join("\n", buffer) + "\n";
        String query = URLEncoder.encode(
                "INSERT INTO sensor_readings FORMAT JSONEachRow",
                StandardCharsets.UTF_8
        );
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

        // Clear only after a successful server response so failures reach Flink's restart policy.
        buffer.clear();
        lastFlushTime = System.currentTimeMillis();
    }

    @Override
    public void close() throws Exception {
        flushBuffer();
    }
}
