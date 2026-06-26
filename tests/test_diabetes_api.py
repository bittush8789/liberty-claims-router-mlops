import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add root folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "diabetes_predictions_total" in response.text

def test_predict_endpoint():
    payload = {
        "Pregnancies": 2,
        "Glucose": 115,
        "BloodPressure": 72,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.450,
        "Age": 30
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["Diabetic", "Non-Diabetic"]
    assert "probability" in data
