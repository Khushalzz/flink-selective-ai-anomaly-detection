package bdt;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import java.util.ArrayList;
import java.util.List;

public class AnomalyDetectorFunction extends KeyedProcessFunction<Integer, SensorReading, AnomalyRecord> {
    private static final long serialVersionUID = 1L;
    private static final int HISTORY_LIMIT = 30;

    private final String modelDirectory;
    private final boolean emitLayaRequests;
    private final boolean applyFallback;
    private final OutputTag<LayaRequest> layaOutputTag;
    private transient ListState<SensorReading> historyState;
    private transient ModelBundle models;

    public AnomalyDetectorFunction(String modelDirectory, boolean applyFallback, boolean emitLayaRequests,
                                   OutputTag<LayaRequest> layaOutputTag) {
        this.modelDirectory = modelDirectory;
        this.applyFallback = applyFallback;
        this.emitLayaRequests = emitLayaRequests;
        this.layaOutputTag = layaOutputTag;
    }

    @Override
    public void open(OpenContext openContext) throws Exception {
        historyState = getRuntimeContext().getListState(
                new ListStateDescriptor<>("sensor-last-30-readings", SensorReading.class));
        models = new ModelBundle(modelDirectory);
    }

    @Override
    public void processElement(SensorReading reading, Context ctx, Collector<AnomalyRecord> out) throws Exception {
        long startMs = System.currentTimeMillis();
        if (reading.getRunId() == null || reading.getRunId().isBlank()) {
            reading.setRunId("manual");
        }
        if (reading.getEventId() == null || reading.getEventId().isBlank()) {
            reading.setEventId(reading.getRunId() + ":" + reading.getMoteid() + ":" + reading.getEpoch());
        }
        if (reading.getSentAtEpochMs() <= 0) {
            reading.setSentAtEpochMs(startMs);
        }

        List<SensorReading> history = new ArrayList<>();
        for (SensorReading previous : historyState.get()) {
            history.add(previous);
        }
        history.add(reading);
        if (history.size() > HISTORY_LIMIT) {
            history = new ArrayList<>(history.subList(history.size() - HISTORY_LIMIT, history.size()));
        }
        historyState.update(history);

        double[] features = SensorFeatureWindow.compute(history);
        ModelBundle.Scores scores = models.score(features, applyFallback);
        String reasons = buildReasons(scores);
        String engine = scores.escalated ? "xgboost" : "fast-path";
        String system = applyFallback ? "bdt" : "fast";
        AnomalyRecord bdt = new AnomalyRecord(reading, scores.pIf, scores.pAe, scores.pFinal,
                scores.uncertain, scores.escalated, system, engine, "OK", reasons, System.currentTimeMillis());
        out.collect(bdt);

        if (emitLayaRequests && scores.uncertain) {
            ctx.output(layaOutputTag, new LayaRequest(reading, features, scores.pIf, scores.pAe));
        }
    }

    private static String buildReasons(ModelBundle.Scores scores) {
        if (scores.pFinal < 0.5) {
            return scores.escalated ? "XGBOOST_NORMAL" : "NORMAL";
        }
        if (scores.escalated) {
            return scores.disagreement ? "XGBOOST_AFTER_DISAGREEMENT" : "XGBOOST_AFTER_MARGIN_UNCERTAINTY";
        }
        return "FAST_PATH_ANOMALY";
    }

    @Override
    public void close() throws Exception {
        if (models != null) models.close();
    }
}
