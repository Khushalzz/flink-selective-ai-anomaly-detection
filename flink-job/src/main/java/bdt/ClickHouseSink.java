package bdt;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClickHouseSink extends RichSinkFunction<AnomalyRecord> {
    private static final long serialVersionUID = 1L;

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
        this(clickhouseUrl, 2500); // 2,500 rows per batch for peak columnar compression
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
    public void invoke(AnomalyRecord record, Context context) {
        try {
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
            if (buffer.size() >= batchSize || (now - lastFlushTime > 1000 && !buffer.isEmpty())) {
                flushBufferAsync();
            }
        } catch (Exception e) {
            System.err.println("ClickHouseSink invoke error: " + e.getMessage());
        }
    }

    private void flushBufferAsync() {
        if (buffer == null || buffer.isEmpty()) return;
        try {
            StringBuilder sb = new StringBuilder();
            for (String line : buffer) {
                sb.append(line).append("\n");
            }
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(clickhouseUrl + "/?query=INSERT+INTO+sensor_readings+FORMAT+JSONEachRow"))
                    .timeout(Duration.ofSeconds(10))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(sb.toString()))
                    .build();

            // Non-blocking asynchronous send: worker thread never blocks on database I/O!
            httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                    .thenAccept(response -> {
                        if (response.statusCode() >= 400) {
                            System.err.println("ClickHouse async insert error HTTP " + response.statusCode() + ": " + response.body());
                        }
                    })
                    .exceptionally(ex -> {
                        System.err.println("ClickHouse async transport failure: " + ex.getMessage());
                        return null;
                    });

        } catch (Exception e) {
            System.err.println("ClickHouse async dispatch error: " + e.getMessage());
        } finally {
            buffer.clear();
            lastFlushTime = System.currentTimeMillis();
        }
    }

    @Override
    public void close() {
        flushBufferAsync();
    }
}
