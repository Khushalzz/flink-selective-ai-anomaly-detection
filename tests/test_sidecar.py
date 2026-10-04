import pytest
from fastapi.testclient import TestClient
from sidecar.app import app, load_models

@pytest.fixture(scope="module")
def client():
    load_models()
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    """Asserts sidecar health check returns OK and models ready."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "xgb_ready" in data

def test_decide_xgb_endpoint(client):
    """Asserts Tier 2 XGBoost endpoint returns valid decision and continuous confidence."""
    payload = {
        "moteid": 1,
        "epoch": 100,
        "temperature": 35.5,
        "humidity": 45.0,
        "light": 150.0,
        "voltage": 2.2,
        "delta_temp_3m": 4.5,
        "delta_volt_3m": -0.3,
        "z_temp_15m": 3.8,
        "z_volt_15m": -2.9,
        "temp_slope_15m": 0.25,
        "var_temp_15m": 3.5,
        "p_if": 0.62,
        "p_ae": 0.58
    }
    response = client.post("/decide/xgb", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] in ["NORMAL", "ANOMALY"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["engine"] == "xgboost"
    assert data["latency_ms"] >= 0.0

def test_decide_heuristic_endpoint(client):
    """Asserts Heuristic decision endpoint returns valid decision and continuous confidence."""
    payload = {
        "moteid": 2,
        "epoch": 200,
        "temperature": 18.0,
        "humidity": 40.0,
        "light": 80.0,
        "voltage": 2.7,
        "delta_temp_3m": 0.1,
        "delta_volt_3m": 0.0,
        "z_temp_15m": 0.2,
        "z_volt_15m": 0.1,
        "temp_slope_15m": 0.01,
        "var_temp_15m": 0.05,
        "p_if": 0.45,
        "p_ae": 0.48
    }
    response = client.post("/decide/heuristic", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] in ["NORMAL", "ANOMALY"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["engine"] == "heuristic-control"
