package bdt;

import java.io.Serializable;

public class LayaRequest implements Serializable {
    private static final long serialVersionUID = 1L;
    private final SensorReading reading;
    private final double[] features;
    private final double pIf;
    private final double pAe;

    public LayaRequest(SensorReading reading, double[] features, double pIf, double pAe) {
        this.reading = reading;
        this.features = features.clone();
        this.pIf = pIf;
        this.pAe = pAe;
    }

    public SensorReading getReading() { return reading; }
    public double[] getFeatures() { return features.clone(); }
    public double getPIf() { return pIf; }
    public double getPAe() { return pAe; }
}
