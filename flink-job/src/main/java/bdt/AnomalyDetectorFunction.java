package bdt;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class AnomalyDetectorFunction extends KeyedProcessFunction<Integer, SensorReading, AnomalyRecord> {
    private static final long serialVersionUID = 1L;

    public static class SensorStats implements Serializable {
        private static final long serialVersionUID = 1L;
        public long count = 0;
        public double meanTemp = 0, m2Temp = 0;
        public double meanHum = 0, m2Hum = 0;
        public double meanLight = 0, m2Light = 0;
        public double meanVolt = 0, m2Volt = 0;

        public void update(double temp, double hum, double light, double volt) {
            count++;
            double deltaT = temp - meanTemp;
            meanTemp += deltaT / count;
            m2Temp += deltaT * (temp - meanTemp);

            double deltaH = hum - meanHum;
            meanHum += deltaH / count;
            m2Hum += deltaH * (hum - meanHum);

            double deltaL = light - meanLight;
            meanLight += deltaL / count;
            m2Light += deltaL * (light - meanLight);

            double deltaV = volt - meanVolt;
            meanVolt += deltaV / count;
            m2Volt += deltaV * (volt - meanVolt);
        }

        public double getStdTemp() { return count > 1 ? Math.sqrt(m2Temp / (count - 1)) : 0.0; }
        public double getStdHum() { return count > 1 ? Math.sqrt(m2Hum / (count - 1)) : 0.0; }
        public double getStdLight() { return count > 1 ? Math.sqrt(m2Light / (count - 1)) : 0.0; }
        public double getStdVolt() { return count > 1 ? Math.sqrt(m2Volt / (count - 1)) : 0.0; }
    }

    private transient ValueState<SensorStats> statsState;

    @Override
    public void open(OpenContext openContext) {
        ValueStateDescriptor<SensorStats> descriptor =
                new ValueStateDescriptor<>("sensor-rolling-stats", SensorStats.class);
        statsState = getRuntimeContext().getState(descriptor);
    }

    @Override
    public void processElement(SensorReading reading, Context ctx, Collector<AnomalyRecord> out) throws Exception {
        SensorStats stats = statsState.value();
        if (stats == null) {
            stats = new SensorStats();
        }

        List<String> reasons = new ArrayList<>();
        double maxScore = 0.0;

        // 1. Physical Hardware Boundary Thresholds
        if (reading.getVoltage() < 2.4) {
            reasons.add(String.format("LOW_BATTERY_VOLTAGE(%.2fV)", reading.getVoltage()));
            maxScore = Math.max(maxScore, 4.0);
        } else if (reading.getVoltage() > 3.3) {
            reasons.add(String.format("VOLTAGE_SPIKE(%.2fV)", reading.getVoltage()));
            maxScore = Math.max(maxScore, 4.0);
        }

        if (reading.getTemperature() < 0.0 || reading.getTemperature() > 45.0) {
            reasons.add(String.format("EXTREME_TEMP(%.2fC)", reading.getTemperature()));
            maxScore = Math.max(maxScore, 5.0);
        }

        if (reading.getHumidity() < 5.0 || reading.getHumidity() > 95.0) {
            reasons.add(String.format("EXTREME_HUMIDITY(%.2f%%)", reading.getHumidity()));
            maxScore = Math.max(maxScore, 4.5);
        }

        // 2. Statistical Z-Score Outlier Detection (after warmup of 10 readings)
        if (stats.count >= 10) {
            double stdT = stats.getStdTemp();
            if (stdT > 0.05) {
                double zT = Math.abs(reading.getTemperature() - stats.meanTemp) / stdT;
                if (zT > 3.0) {
                    reasons.add(String.format("TEMP_ZSCORE(%.2f)", zT));
                    maxScore = Math.max(maxScore, zT);
                }
            }

            double stdH = stats.getStdHum();
            if (stdH > 0.05) {
                double zH = Math.abs(reading.getHumidity() - stats.meanHum) / stdH;
                if (zH > 3.0) {
                    reasons.add(String.format("HUMIDITY_ZSCORE(%.2f)", zH));
                    maxScore = Math.max(maxScore, zH);
                }
            }

            double stdL = stats.getStdLight();
            if (stdL > 1.0) {
                double zL = Math.abs(reading.getLight() - stats.meanLight) / stdL;
                if (zL > 3.5) {
                    reasons.add(String.format("LIGHT_ZSCORE(%.2f)", zL));
                    maxScore = Math.max(maxScore, zL);
                }
            }

            double stdV = stats.getStdVolt();
            if (stdV > 0.01) {
                double zV = Math.abs(reading.getVoltage() - stats.meanVolt) / stdV;
                if (zV > 3.0) {
                    reasons.add(String.format("VOLT_ZSCORE(%.2f)", zV));
                    maxScore = Math.max(maxScore, zV);
                }
            }
        }

        // Update statistics with current reading
        stats.update(reading.getTemperature(), reading.getHumidity(), reading.getLight(), reading.getVoltage());
        statsState.update(stats);

        int isAnomaly = reasons.isEmpty() ? 0 : 1;
        String reasonsStr = reasons.isEmpty() ? "NORMAL" : String.join(", ", reasons);

        out.collect(new AnomalyRecord(reading, isAnomaly, maxScore, reasonsStr));
    }
}
