package bdt;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** Loads the three Python-trained artifacts once per Flink operator instance. */
public final class ModelBundle implements AutoCloseable {
    public static final List<String> FEATURE_ORDER = List.of(
            "temperature", "humidity", "light", "voltage",
            "delta_temp_3m", "delta_volt_3m", "z_temp_15m", "z_volt_15m",
            "temp_slope_15m", "var_temp_15m"
    );
    private static final double MARGIN_LOW = 0.35;
    private static final double MARGIN_HIGH = 0.65;

    private final OrtEnvironment environment;
    private final OrtSession.SessionOptions sessionOptions;
    private final OrtSession isolationForest;
    private final OrtSession autoencoder;
    private final OrtSession xgboost;
    private final String ifInputName;
    private final String aeInputName;
    private final String xgbInputName;
    private final Calibration ifCalibration;
    private final Calibration aeCalibration;

    public ModelBundle(String modelDirectory) throws Exception {
        Path root = Paths.get(modelDirectory);
        ObjectMapper mapper = new ObjectMapper();
        ifCalibration = mapper.readValue(root.resolve("if_calibration.json").toFile(), Calibration.class);
        aeCalibration = mapper.readValue(root.resolve("ae_calibration.json").toFile(), Calibration.class);
        validateFeatureContract(ifCalibration.features, "if_calibration.json");
        validateFeatureContract(aeCalibration.features, "ae_calibration.json");
        if (aeCalibration.means == null || aeCalibration.stds == null
                || aeCalibration.means.size() != FEATURE_ORDER.size()
                || aeCalibration.stds.size() != FEATURE_ORDER.size()) {
            throw new IllegalArgumentException("Autoencoder calibration must contain ten means and standard deviations");
        }

        environment = OrtEnvironment.getEnvironment();
        sessionOptions = new OrtSession.SessionOptions();
        sessionOptions.setIntraOpNumThreads(1);
        isolationForest = environment.createSession(root.resolve("isolation_forest.onnx").toString(), sessionOptions);
        autoencoder = environment.createSession(root.resolve("autoencoder.onnx").toString(), sessionOptions);
        xgboost = environment.createSession(root.resolve("xgboost_fallback.onnx").toString(), sessionOptions);
        ifInputName = isolationForest.getInputNames().iterator().next();
        aeInputName = autoencoder.getInputNames().iterator().next();
        xgbInputName = xgboost.getInputNames().iterator().next();
    }

    private static void validateFeatureContract(List<String> actual, String source) {
        if (actual == null || !FEATURE_ORDER.equals(actual)) {
            throw new IllegalArgumentException(source + " feature order does not match Flink: " + actual);
        }
    }

    public Scores score(double[] input) throws Exception {
        return score(input, true);
    }

    public Scores score(double[] input, boolean applyFallback) throws Exception {
        if (input.length != FEATURE_ORDER.size()) {
            throw new IllegalArgumentException("Expected ten features, received " + input.length);
        }
        float[] features = toFloatArray(input);
        float[] normalized = new float[features.length];
        for (int i = 0; i < features.length; i++) {
            normalized[i] = (float) ((features[i] - aeCalibration.means.get(i)) / aeCalibration.stds.get(i));
        }

        double pIf;
        try (OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(features), new long[]{1, features.length});
             OrtSession.Result result = isolationForest.run(Collections.singletonMap(ifInputName, tensor))) {
            float[][] rawScores = (float[][]) result.get(1).getValue();
            pIf = sigmoid(ifCalibration.coef * -rawScores[0][0] + ifCalibration.intercept);
        }

        double pAe;
        try (OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(normalized), new long[]{1, normalized.length});
             OrtSession.Result result = autoencoder.run(Collections.singletonMap(aeInputName, tensor))) {
            float[][] reconstruction = (float[][]) result.get(0).getValue();
            double mse = 0.0;
            for (int i = 0; i < normalized.length; i++) {
                double difference = normalized[i] - reconstruction[0][i];
                mse += difference * difference;
            }
            mse /= normalized.length;
            pAe = sigmoid(aeCalibration.coef * mse + aeCalibration.intercept);
        }

        boolean disagreement = (pIf >= 0.5) != (pAe >= 0.5);
        boolean uncertain = inMargin(pIf) || inMargin(pAe) || disagreement;
        double pFinal = 0.5 * (pIf + pAe);
        if (uncertain && applyFallback) {
            float[] enhanced = new float[features.length + 2];
            System.arraycopy(features, 0, enhanced, 0, features.length);
            enhanced[features.length] = (float) pIf;
            enhanced[features.length + 1] = (float) pAe;
            try (OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(enhanced), new long[]{1, enhanced.length});
                 OrtSession.Result result = xgboost.run(Collections.singletonMap(xgbInputName, tensor))) {
                float[][] probabilities = (float[][]) result.get(1).getValue();
                pFinal = probabilities[0][1];
            }
        }
        return new Scores(pIf, pAe, pFinal, uncertain, uncertain && applyFallback, disagreement);
    }

    private static boolean inMargin(double probability) {
        return probability > MARGIN_LOW && probability < MARGIN_HIGH;
    }

    private static float[] toFloatArray(double[] source) {
        float[] result = new float[source.length];
        for (int i = 0; i < source.length; i++) {
            result[i] = (float) (Double.isFinite(source[i]) ? source[i] : 0.0);
        }
        return result;
    }

    private static double sigmoid(double value) {
        double clipped = Math.max(-25.0, Math.min(25.0, value));
        return 1.0 / (1.0 + Math.exp(-clipped));
    }

    @Override
    public void close() throws Exception {
        xgboost.close();
        autoencoder.close();
        isolationForest.close();
        sessionOptions.close();
    }

    public static final class Scores {
        public final double pIf;
        public final double pAe;
        public final double pFinal;
        public final boolean uncertain;
        public final boolean escalated;
        public final boolean disagreement;

        public Scores(double pIf, double pAe, double pFinal, boolean uncertain, boolean escalated, boolean disagreement) {
            this.pIf = pIf;
            this.pAe = pAe;
            this.pFinal = pFinal;
            this.uncertain = uncertain;
            this.escalated = escalated;
            this.disagreement = disagreement;
        }
    }

    public static final class Calibration {
        public double coef;
        public double intercept;
        public List<Double> means;
        public List<Double> stds;
        public List<String> features;
    }
}
