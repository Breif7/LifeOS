"""
Unit tests for FastAPI REST Endpoints (WBS 4.1).
"""

import pytest
from starlette.testclient import TestClient
from lifeos.api import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project_code"] == "PRJ-LIFEOS-ML-2026-Q4"
    assert data["status"] == "OPERATIONAL"


def test_api_cpm_schedule():
    response = client.get("/cpm/schedule")
    assert response.status_code == 200
    data = response.json()
    assert data["project_duration_hours"] == 166.0
    assert "1.1" in data["critical_path"]
    assert len(data["wbs_packages"]) == 16


def test_api_nlp_classify_endpoint():
    payload = {"text": "Apartar $4,208 MXN para liquidar pago para no generar intereses de AMEX antes del día 9"}
    response = client.post("/nlp/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["covey_quadrant"] == "Q1"
    assert data["is_blocked"] is False


def test_api_nlp_classify_blocked():
    payload = {"text": "Rediseñar paleta de colores CSS y sombras de botones del dashboard por estética"}
    response = client.post("/nlp/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["covey_quadrant"] == "Q4"
    assert data["is_blocked"] is True


def test_api_evm_metrics():
    response = client.get("/evm/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["cutoff_sprint"] == 3
    assert data["cutoff_metrics"]["technical"]["BAC"] == 240.0
    assert data["cutoff_metrics"]["financial"]["BAC"] == 65400.0
    assert data["ml_forecast"]["eac_ml_hours"] == 235.63


def test_api_finance_schedule():
    response = client.get("/finance/schedule")
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["total_debt_balance"] == 47238.0
    assert "quincena_1" in data["payment_schedule"]
    assert "quincena_2" in data["payment_schedule"]


def test_api_forge3d_quote():
    payload = {
        "job_name": "Sensor Support Bracket",
        "grams_per_unit": 35.0,
        "hours_per_unit": 2.5,
        "units": 4,
        "material": "PETG",
        "post_processing_minutes": 10.0,
        "margin_percent": 35.0,
    }
    response = client.post("/forge3d/quote", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["units"] == 4
    assert data["material"] == "PETG"
    assert data["final_price_total"] > 0


def test_api_risks():
    response = client.get("/risks")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 6
    risk_ids = [r["risk_id"] for r in data]
    assert "R-01" in risk_ids
    assert "R-02" in risk_ids
