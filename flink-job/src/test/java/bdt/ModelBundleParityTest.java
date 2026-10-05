package bdt;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.io.InputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelBundleParityTest {
    @Test
    void javaOnnxScoresMatchPythonReferenceFixture() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        try (InputStream stream = getClass().getResourceAsStream("/model_parity_fixture.json")) {
            assertTrue(stream != null, "parity fixture must be on the test classpath");
            JsonNode cases = mapper.readTree(stream);
            String modelDirectory = System.getProperty("bdt.model.dir", "../models");
            try (ModelBundle models = new ModelBundle(modelDirectory)) {
                for (JsonNode item : cases) {
                    double[] features = new double[item.path("features").size()];
                    for (int i = 0; i < features.length; i++) {
                        features[i] = item.path("features").get(i).asDouble();
                    }
                    ModelBundle.Scores actual = models.score(features);
                    assertEquals(item.path("p_if").asDouble(), actual.pIf, 1e-5, "Isolation Forest probability");
                    assertEquals(item.path("p_ae").asDouble(), actual.pAe, 1e-5, "Autoencoder probability");
                    assertEquals(item.path("p_final").asDouble(), actual.pFinal, 1e-5, "Final probability");
                    assertEquals(item.path("escalated").asBoolean(), actual.escalated, "Gate result");
                }
            }
        }
    }
}
