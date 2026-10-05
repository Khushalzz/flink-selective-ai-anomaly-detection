package bdt;

import java.util.List;

/** Reproduces preprocessing/inject_anomalies.py feature order and window semantics. */
public final class SensorFeatureWindow {
    private SensorFeatureWindow() {}

    public static double[] compute(List<SensorReading> readings) {
        if (readings == null || readings.isEmpty()) {
            throw new IllegalArgumentException("At least one reading is required");
        }
        int count = readings.size();
        int shortStart = Math.max(0, count - 6);
        int longStart = Math.max(0, count - 30);
        double shortTempMean = 0.0;
        double shortVoltMean = 0.0;
        double longTempMean = 0.0;
        double longVoltMean = 0.0;

        for (int i = shortStart; i < count; i++) {
            shortTempMean += readings.get(i).getTemperature();
            shortVoltMean += readings.get(i).getVoltage();
        }
        int shortCount = count - shortStart;
        shortTempMean /= shortCount;
        shortVoltMean /= shortCount;

        int longCount = count - longStart;
        for (int i = longStart; i < count; i++) {
            longTempMean += readings.get(i).getTemperature();
            longVoltMean += readings.get(i).getVoltage();
        }
        longTempMean /= longCount;
        longVoltMean /= longCount;

        double tempStd = 0.1;
        double voltStd = 0.01;
        double zTemp = 0.0;
        double zVolt = 0.0;
        if (longCount >= 3) {
            double tempSquares = 0.0;
            double voltSquares = 0.0;
            for (int i = longStart; i < count; i++) {
                double dt = readings.get(i).getTemperature() - longTempMean;
                double dv = readings.get(i).getVoltage() - longVoltMean;
                tempSquares += dt * dt;
                voltSquares += dv * dv;
            }
            tempStd = Math.sqrt(tempSquares / (longCount - 1));
            voltStd = Math.sqrt(voltSquares / (longCount - 1));
            SensorReading current = readings.get(count - 1);
            zTemp = (current.getTemperature() - longTempMean) / (tempStd + 1e-4);
            zVolt = (current.getVoltage() - longVoltMean) / (voltStd + 1e-4);
        }

        SensorReading current = readings.get(count - 1);
        double slope = count > 10
                ? (current.getTemperature() - readings.get(count - 11).getTemperature()) / 10.0
                : 0.0;
        return new double[]{
                current.getTemperature(), current.getHumidity(), current.getLight(), current.getVoltage(),
                current.getTemperature() - shortTempMean,
                current.getVoltage() - shortVoltMean,
                zTemp, zVolt, slope, tempStd * tempStd
        };
    }
}
