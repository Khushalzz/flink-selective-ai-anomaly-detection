package bdt;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import redis.clients.jedis.Jedis;

public class RedisAlertSink extends RichSinkFunction<AnomalyRecord> {
    private static final long serialVersionUID = 1L;

    private final String host;
    private final int port;
    private transient Jedis jedis;
    private transient ObjectMapper objectMapper;

    public RedisAlertSink(String host, int port) {
        this.host = host;
        this.port = port;
    }

    @Override
    public void open(OpenContext openContext) {
        init();
    }

    private void init() {
        try {
            jedis = new Jedis(host, port);
            objectMapper = new ObjectMapper();
        } catch (Exception e) {
            System.err.println("Warning: Redis connection failed in worker: " + e.getMessage());
        }
    }

    @Override
    public void invoke(AnomalyRecord record, Context context) {
        // Only publish when an anomaly is flagged!
        if (record.getIsAnomaly() == 1) {
            try {
                if (jedis == null) {
                    init();
                }
                String alertJson = objectMapper.writeValueAsString(record);
                jedis.rpush("alerts:anomalies", alertJson);
                jedis.ltrim("alerts:anomalies", -2000, -1); // Keep last 2,000 alerts
                jedis.publish("channel:anomalies", alertJson);
            } catch (Exception e) {
                // Reconnect on transient error
                try {
                    if (jedis != null) jedis.close();
                } catch (Exception ignored) {}
                jedis = null;
            }
        }
    }

    @Override
    public void close() {
        if (jedis != null) {
            try {
                jedis.close();
            } catch (Exception ignored) {}
        }
    }
}
