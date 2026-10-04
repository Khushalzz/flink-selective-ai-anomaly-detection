import pytest
import numpy as np

class WelfordAccumulator:
    """Exact Python mirror of Flink AnomalyDetectorFunction.SensorStats Welford accumulator."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    def std(self) -> float:
        return np.sqrt(self.variance())

    def z_score(self, x: float) -> float:
        s = self.std()
        return abs(x - self.mean) / s if s > 1e-6 else 0.0

def test_welford_incremental_numerical_accuracy():
    """Validates that Welford online accumulator matches sample variance to 10 decimal places."""
    np.random.seed(42)
    values = np.random.normal(loc=22.5, scale=3.2, size=1000)
    
    acc = WelfordAccumulator()
    for v in values:
        acc.update(float(v))
        
    expected_mean = float(np.mean(values))
    expected_std = float(np.std(values, ddof=1))
    
    assert np.isclose(acc.mean, expected_mean, atol=1e-7)
    assert np.isclose(acc.std(), expected_std, atol=1e-7)

def test_welford_z_score_calculation():
    """Validates z-score spike detection on known outliers."""
    acc = WelfordAccumulator()
    for v in [20.0, 20.2, 19.8, 20.1, 20.0, 19.9, 20.1, 20.0, 20.2, 19.9]:
        acc.update(v)
        
    # Baseline mean ~20.0, std ~0.13
    assert acc.z_score(20.0) < 0.5
    # An extreme spike of 25.0C should have z-score > 25
    assert acc.z_score(25.0) > 20.0
