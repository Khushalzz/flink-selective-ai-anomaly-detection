package bdt;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.io.Serializable;

public class SensorReading implements Serializable {
    private static final long serialVersionUID = 1L;

    private String date;
    private String time;
    private String timestamp;
    private String runId;
    private String eventId;
    private String anomalyType;
    private long sentAtEpochMs;
    private Integer label;
    private int epoch;
    private int moteid;
    private double temperature;
    private double humidity;
    private double light;
    private double voltage;

    public SensorReading() {}

    public String getDate() { return date; }
    public void setDate(String date) { this.date = date; }
    public String getTime() { return time; }
    public void setTime(String time) { this.time = time; }
    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }
    @JsonProperty("run_id")
    public String getRunId() { return runId; }
    @JsonProperty("run_id")
    public void setRunId(String runId) { this.runId = runId; }
    @JsonProperty("event_id")
    public String getEventId() { return eventId; }
    @JsonProperty("event_id")
    public void setEventId(String eventId) { this.eventId = eventId; }
    @JsonProperty("anomaly_type")
    public String getAnomalyType() { return anomalyType; }
    @JsonProperty("anomaly_type")
    public void setAnomalyType(String anomalyType) { this.anomalyType = anomalyType; }
    @JsonProperty("sent_at_epoch_ms")
    public long getSentAtEpochMs() { return sentAtEpochMs; }
    @JsonProperty("sent_at_epoch_ms")
    public void setSentAtEpochMs(long sentAtEpochMs) { this.sentAtEpochMs = sentAtEpochMs; }
    public Integer getLabel() { return label; }
    public void setLabel(Integer label) { this.label = label; }
    public int getEpoch() { return epoch; }
    public void setEpoch(int epoch) { this.epoch = epoch; }
    public int getMoteid() { return moteid; }
    public void setMoteid(int moteid) { this.moteid = moteid; }
    public double getTemperature() { return temperature; }
    public void setTemperature(double temperature) { this.temperature = temperature; }
    public double getHumidity() { return humidity; }
    public void setHumidity(double humidity) { this.humidity = humidity; }
    public double getLight() { return light; }
    public void setLight(double light) { this.light = light; }
    public double getVoltage() { return voltage; }
    public void setVoltage(double voltage) { this.voltage = voltage; }
}
