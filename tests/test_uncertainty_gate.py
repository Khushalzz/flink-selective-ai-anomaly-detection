import pytest
import numpy as np

def evaluate_uncertainty_gate(p_if, p_ae, margin_low=0.35, margin_high=0.65):
    """
    Evaluates dual epistemic uncertainty criteria:
      1. Margin Uncertainty: detector probability lies in ambiguous band [margin_low, margin_high]
      2. Detector Disagreement: one detector predicts normal (< 0.5) while other predicts anomaly (>= 0.5)
    """
    pred_if = int(p_if >= 0.5)
    pred_ae = int(p_ae >= 0.5)
    
    in_margin_if = (margin_low < p_if < margin_high)
    in_margin_ae = (margin_low < p_ae < margin_high)
    disagree = (pred_if != pred_ae)
    
    return bool(in_margin_if or in_margin_ae or disagree)

def test_confident_normal_no_escalation():
    """Clear normal readings should pass through with zero escalation."""
    assert not evaluate_uncertainty_gate(0.05, 0.08)
    assert not evaluate_uncertainty_gate(0.12, 0.20)
    assert not evaluate_uncertainty_gate(0.01, 0.30)

def test_confident_anomaly_no_escalation():
    """Clear severe anomalies should pass through with zero escalation."""
    assert not evaluate_uncertainty_gate(0.85, 0.92)
    assert not evaluate_uncertainty_gate(0.70, 0.88)
    assert not evaluate_uncertainty_gate(0.99, 0.99)

def test_detector_disagreement_escalation():
    """When IF predicts anomaly but AE predicts normal (or vice versa), must escalate."""
    assert evaluate_uncertainty_gate(0.80, 0.20)
    assert evaluate_uncertainty_gate(0.20, 0.80)
    assert evaluate_uncertainty_gate(0.95, 0.10)

def test_margin_uncertainty_escalation():
    """When either detector is near 0.5 decision boundary, must escalate."""
    assert evaluate_uncertainty_gate(0.48, 0.49)
    assert evaluate_uncertainty_gate(0.52, 0.55)
    assert evaluate_uncertainty_gate(0.36, 0.20)
    assert evaluate_uncertainty_gate(0.20, 0.64)
