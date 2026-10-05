package bdt;

import java.io.Serializable;

public class AnomalyRecord implements Serializable {
    private static final long serialVersionUID = 1L;

    private String runId;
    private String eventId;
    private String date;
    private String time;
    private String timestamp;
    private String anomalyType;
    private String system;
    private String engine;
    private String inferenceStatus;
    private String prediction;
    private String anomalyReasons;
    private long sentAtEpochMs;
    private long processedAtEpochMs;
    private int epoch;
    private int moteid;
    private double temperature;
    private double humidity;
    private double light;
    private double voltage;
    private Integer groundTruthLabel;
    private Integer isAnomaly;
    private double pIf;
    private double pAe;
    private Double pFinal;
    private boolean uncertain;
    private boolean escalated;
    private double anomalyScore;
    private double processingLatencyMs;

    public AnomalyRecord() {}

    public AnomalyRecord(SensorReading reading, double pIf, double pAe, Double pFinal,
                         boolean uncertain, boolean escalated, String system, String engine, String inferenceStatus, String reasons,
                         long processedAtEpochMs) {
        this.runId = reading.getRunId();
        this.eventId = reading.getEventId();
        this.date = reading.getDate();
        this.time = reading.getTime();
        this.timestamp = reading.getTimestamp();
        this.anomalyType = reading.getAnomalyType();
        this.sentAtEpochMs = reading.getSentAtEpochMs();
        this.processedAtEpochMs = processedAtEpochMs;
        this.epoch = reading.getEpoch();
        this.moteid = reading.getMoteid();
        this.temperature = reading.getTemperature();
        this.humidity = reading.getHumidity();
        this.light = reading.getLight();
        this.voltage = reading.getVoltage();
        this.groundTruthLabel = reading.getLabel();
        this.pIf = pIf;
        this.pAe = pAe;
        this.pFinal = pFinal;
        this.uncertain = uncertain;
        this.escalated = escalated;
        this.system = system;
        this.engine = engine;
        this.inferenceStatus = inferenceStatus;
        this.isAnomaly = pFinal == null ? null : (pFinal >= 0.5 ? 1 : 0);
        this.prediction = pFinal == null ? "UNKNOWN" : (pFinal >= 0.5 ? "ANOMALY" : "NORMAL");
        this.anomalyScore = pFinal == null ? 0.0 : pFinal;
        this.anomalyReasons = reasons;
        this.processingLatencyMs = sentAtEpochMs > 0 ? Math.max(0.0, processedAtEpochMs - sentAtEpochMs) : 0.0;
    }

    public static AnomalyRecord layaSuccess(LayaRequest request, String decision, double confidence, long processedAtEpochMs) {
        AnomalyRecord record = new AnomalyRecord(request.getReading(), request.getPIf(), request.getPAe(), confidence,
                true, true, "laya", "laya", "OK", "Laya comparison", processedAtEpochMs);
        record.prediction = decision;
        record.isAnomaly = "ANOMALY".equalsIgnoreCase(decision) ? 1 : 0;
        return record;
    }

    public static AnomalyRecord layaFailure(LayaRequest request, String status, long processedAtEpochMs) {
        AnomalyRecord record = new AnomalyRecord(request.getReading(), request.getPIf(), request.getPAe(), null,
                true, true, "laya", "laya", status, "Laya comparison unavailable", processedAtEpochMs);
        record.prediction = "UNKNOWN";
        return record;
    }

    public String getRunId() { return runId; }
    public void setRunId(String runId) { this.runId = runId; }
    public String getEventId() { return eventId; }
    public void setEventId(String eventId) { this.eventId = eventId; }
    public String getDate() { return date; }
    public void setDate(String date) { this.date = date; }
    public String getTime() { return time; }
    public void setTime(String time) { this.time = time; }
    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }
    public String getAnomalyType() { return anomalyType; }
    public void setAnomalyType(String anomalyType) { this.anomalyType = anomalyType; }
    public String getSystem() { return system; }
    public void setSystem(String system) { this.system = system; }
    public String getEngine() { return engine; }
    public void setEngine(String engine) { this.engine = engine; }
    public String getPrediction() { return prediction; }
    public void setPrediction(String prediction) { this.prediction = prediction; }
    public String getInferenceStatus() { return inferenceStatus; }
    public void setInferenceStatus(String inferenceStatus) { this.inferenceStatus = inferenceStatus; }
    public String getAnomalyReasons() { return anomalyReasons; }
    public void setAnomalyReasons(String anomalyReasons) { this.anomalyReasons = anomalyReasons; }
    public long getSentAtEpochMs() { return sentAtEpochMs; }
    public void setSentAtEpochMs(long sentAtEpochMs) { this.sentAtEpochMs = sentAtEpochMs; }
    public long getProcessedAtEpochMs() { return processedAtEpochMs; }
    public void setProcessedAtEpochMs(long processedAtEpochMs) { this.processedAtEpochMs = processedAtEpochMs; }
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
    public Integer getGroundTruthLabel() { return groundTruthLabel; }
    public void setGroundTruthLabel(Integer groundTruthLabel) { this.groundTruthLabel = groundTruthLabel; }
    public Integer getIsAnomaly() { return isAnomaly; }
    public void setIsAnomaly(Integer isAnomaly) { this.isAnomaly = isAnomaly; }
    public double getPIf() { return pIf; }
    public void setPIf(double pIf) { this.pIf = pIf; }
    public double getPAe() { return pAe; }
    public void setPAe(double pAe) { this.pAe = pAe; }
    public Double getPFinal() { return pFinal; }
    public void setPFinal(Double pFinal) { this.pFinal = pFinal; }
    public boolean isUncertain() { return uncertain; }
    public void setUncertain(boolean uncertain) { this.uncertain = uncertain; }
    public boolean isEscalated() { return escalated; }
    public void setEscalated(boolean escalated) { this.escalated = escalated; }
    public double getAnomalyScore() { return anomalyScore; }
    public void setAnomalyScore(double anomalyScore) { this.anomalyScore = anomalyScore; }
    public double getProcessingLatencyMs() { return processingLatencyMs; }
    public void setProcessingLatencyMs(double processingLatencyMs) { this.processingLatencyMs = processingLatencyMs; }
}
