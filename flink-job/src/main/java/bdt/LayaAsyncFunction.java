package bdt;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.async.RichAsyncFunction;
import org.apache.flink.streaming.api.functions.async.ResultFuture;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Collections;
import java.util.concurrent.CompletionException;

/** Optional System F comparison route. The normal BDT decision never waits for Laya. */
public class LayaAsyncFunction extends RichAsyncFunction<LayaRequest, AnomalyRecord> {
    private static final long serialVersionUID = 1L;
    private final String endpoint;
    private transient HttpClient client;
    private transient ObjectMapper mapper;

    public LayaAsyncFunction(String endpoint) {
        this.endpoint = endpoint;
    }

    @Override
    public void open(Configuration configuration) {
        client = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(2)).build();
        mapper = new ObjectMapper();
    }

    @Override
    public void asyncInvoke(LayaRequest request, ResultFuture<AnomalyRecord> resultFuture) {
        try {
            HttpRequest httpRequest = HttpRequest.newBuilder(URI.create(endpoint))
                    .timeout(Duration.ofSeconds(15))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(toPayload(request))))
                    .build();
            client.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
                    .whenComplete((response, error) -> {
                        long finishedAt = System.currentTimeMillis();
                        try {
                            if (error != null) {
                                resultFuture.complete(Collections.singleton(AnomalyRecord.layaFailure(
                                        request, "FAILED:" + rootMessage(error), finishedAt)));
                            } else if (response.statusCode() < 200 || response.statusCode() >= 300) {
                                String detail = response.body() == null ? "" : response.body().replaceAll("\\s+", " ").trim();
                                if (detail.length() > 220) detail = detail.substring(0, 220);
                                String status = "HTTP_" + response.statusCode() + (detail.isEmpty() ? "" : ":" + detail);
                                resultFuture.complete(Collections.singleton(AnomalyRecord.layaFailure(
                                        request, status, finishedAt)));
                            } else {
                                JsonNode body = mapper.readTree(response.body());
                                resultFuture.complete(Collections.singleton(AnomalyRecord.layaSuccess(
                                        request, body.path("decision").asText("UNKNOWN"),
                                        body.path("confidence").asDouble(0.0), finishedAt)));
                            }
                        } catch (Exception parseError) {
                            resultFuture.complete(Collections.singleton(AnomalyRecord.layaFailure(
                                    request, "INVALID_RESPONSE:" + rootMessage(parseError), finishedAt)));
                        }
                    });
        } catch (Exception error) {
            resultFuture.complete(Collections.singleton(AnomalyRecord.layaFailure(
                    request, "REQUEST_ERROR:" + rootMessage(error), System.currentTimeMillis())));
        }
    }

    @Override
    public void timeout(LayaRequest request, ResultFuture<AnomalyRecord> resultFuture) {
        resultFuture.complete(Collections.singleton(AnomalyRecord.layaFailure(
                request, "TIMEOUT", System.currentTimeMillis())));
    }

    private ObjectNode toPayload(LayaRequest request) {
        SensorReading reading = request.getReading();
        double[] f = request.getFeatures();
        ObjectNode json = mapper.createObjectNode();
        json.put("moteid", reading.getMoteid());
        json.put("epoch", reading.getEpoch());
        json.put("temperature", reading.getTemperature());
        json.put("humidity", reading.getHumidity());
        json.put("light", reading.getLight());
        json.put("voltage", reading.getVoltage());
        json.put("delta_temp_3m", f[4]);
        json.put("delta_volt_3m", f[5]);
        json.put("z_temp_15m", f[6]);
        json.put("z_volt_15m", f[7]);
        json.put("temp_slope_15m", f[8]);
        json.put("var_temp_15m", f[9]);
        json.put("p_if", request.getPIf());
        json.put("p_ae", request.getPAe());
        return json;
    }

    private static String rootMessage(Throwable error) {
        Throwable cause = error;
        while (cause instanceof CompletionException && cause.getCause() != null) {
            cause = cause.getCause();
        }
        String message = cause.getMessage();
        return message == null ? cause.getClass().getSimpleName() : message;
    }
}
