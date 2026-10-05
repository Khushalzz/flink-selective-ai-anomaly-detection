package bdt;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.api.common.functions.OpenContext;
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
    public void open(OpenContext openContext) throws Exception {
        connect();
    }

    private void connect() throws Exception {
        Jedis connection = new Jedis(host, port);
        connection.ping();
        jedis = connection;
        objectMapper = new ObjectMapper();
    }

    @Override
    public void invoke(AnomalyRecord record, Context context) throws Exception {
        if (!Integer.valueOf(1).equals(record.getIsAnomaly())) {
            return;
        }

        try {
            if (jedis == null) {
                connect();
            }
            String alertJson = objectMapper.writeValueAsString(record);
            String key = "alerts:anomalies:" + record.getRunId();
            jedis.rpush(key, alertJson);
            jedis.ltrim(key, -2000, -1);
            jedis.publish("channel:anomalies", alertJson);
        } catch (Exception e) {
            if (jedis != null) {
                try {
                    jedis.close();
                } catch (Exception ignored) {
                    // Preserve the original Redis failure.
                }
            }
            jedis = null;
            objectMapper = null;
            throw e;
        }
    }

    @Override
    public void close() {
        if (jedis != null) {
            try {
                jedis.close();
            } catch (Exception ignored) {
                // Best-effort client cleanup.
            }
        }
    }
}
