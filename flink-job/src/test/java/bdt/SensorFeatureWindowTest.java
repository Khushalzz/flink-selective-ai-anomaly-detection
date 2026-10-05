package bdt;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SensorFeatureWindowTest {
    @Test
    void computesTheSameSixAndThirtyReadingWindowContract() {
        List<SensorReading> history = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            SensorReading reading = new SensorReading();
            reading.setMoteid(1);
            reading.setEpoch(i);
            reading.setTemperature(10.0 + i);
            reading.setHumidity(30.0 + i);
            reading.setLight(100.0 + 2 * i);
            reading.setVoltage(2.0 + 0.01 * i);
            history.add(reading);
        }
        double[] features = SensorFeatureWindow.compute(history);
        assertEquals(39.0, features[0], 1e-9);
        assertEquals(2.5, features[4], 1e-9);
        assertEquals(2.29, features[3], 1e-9);
        assertEquals(0.025, features[5], 1e-9);
        assertEquals(1.0, features[8], 1e-9);
        assertEquals(77.5, features[9], 1e-9);
    }

    @Test
    void usesDocumentedWarmupDefaultsForTheFirstReading() {
        SensorReading reading = new SensorReading();
        reading.setTemperature(22.0);
        reading.setHumidity(40.0);
        reading.setLight(100.0);
        reading.setVoltage(2.7);
        double[] features = SensorFeatureWindow.compute(List.of(reading));
        assertEquals(0.0, features[6], 1e-9);
        assertEquals(0.0, features[7], 1e-9);
        assertEquals(0.01, features[9], 1e-9);
    }
}
