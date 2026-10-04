package bdt;

import java.io.Serializable;

public class AnomalyRecord implements Serializable {
    private static final long serialVersionUID = 1L;

    private String date;
    private String time;
    private String timestamp;
    private int epoch;
    private int moteid;
    private double temperature;
    private double humidity;
    private double light;
    private double voltage;
    private int isAnomaly; // 1 = anomaly, 0 = normal
    private double anomalyScore;
    private String anomalyReasons;

    public AnomalyRecord() {}

    public AnomalyRecord(SensorReading reading, int isAnomaly, double anomalyScore, String anomalyReasons) {
        this.date = reading.getDate();
        this.time = reading.getTime();
        this.timestamp = reading.getTimestamp();
        this.epoch = reading.getEpoch();
        this.moteid = reading.getMoteid();
        this.temperature = reading.getTemperature();
        this.humidity = reading.getHumidity();
        this.light = reading.getLight();
        this.voltage = reading.getVoltage();
        this.isAnomaly = isAnomaly;
        this.anomalyScore = anomalyScore;
        this.anomalyReasons = anomalyReasons;
    }

    public String getDate() { return date; }
    public void setDate(String date) { this.date = date; }

    public String getTime() { return time; }
    public void setTime(String time) { this.time = time; }

    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }

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

    public int getIsAnomaly() { return isAnomaly; }
    public void setIsAnomaly(int isAnomaly) { this.isAnomaly = isAnomaly; }

    public double getAnomalyScore() { return anomalyScore; }
    public void setAnomalyScore(double anomalyScore) { this.anomalyScore = anomalyScore; }

    public String getAnomalyReasons() { return anomalyReasons; }
    public void setAnomalyReasons(String anomalyReasons) { this.anomalyReasons = anomalyReasons; }

    @Override
    public String toString() {
        return "AnomalyRecord{" +
                "moteid=" + moteid +
                ", epoch=" + epoch +
                ", isAnomaly=" + isAnomaly +
                ", score=" + String.format("%.2f", anomalyScore) +
                ", reasons='" + anomalyReasons + '\'' +
                ", temp=" + temperature +
                ", hum=" + humidity +
                ", volt=" + voltage +
                '}';
    }
}
