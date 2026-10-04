package bdt;

import java.io.Serializable;

public class SensorReading implements Serializable {
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

    public SensorReading() {}

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

    @Override
    public String toString() {
        return "SensorReading{" +
                "moteid=" + moteid +
                ", epoch=" + epoch +
                ", temp=" + temperature +
                ", hum=" + humidity +
                ", light=" + light +
                ", volt=" + voltage +
                '}';
    }
}
