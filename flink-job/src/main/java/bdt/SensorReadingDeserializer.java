package bdt;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.api.common.serialization.DeserializationSchema;
import org.apache.flink.api.common.typeinfo.TypeInformation;

import java.io.IOException;

public class SensorReadingDeserializer implements DeserializationSchema<SensorReading> {
    private static final long serialVersionUID = 1L;
    private transient ObjectMapper objectMapper;

    @Override
    public void open(InitializationContext context) {
        objectMapper = new ObjectMapper();
    }

    @Override
    public SensorReading deserialize(byte[] message) throws IOException {
        if (objectMapper == null) {
            objectMapper = new ObjectMapper();
        }
        return objectMapper.readValue(message, SensorReading.class);
    }

    @Override
    public boolean isEndOfStream(SensorReading nextElement) {
        return false;
    }

    @Override
    public TypeInformation<SensorReading> getProducedType() {
        return TypeInformation.of(SensorReading.class);
    }
}
