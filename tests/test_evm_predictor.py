"""
Unit tests for Earned Value Management (EVM) tracker and Polynomial Ridge Regression predictor.
"""

import pytest
from lifeos.evm_predictor import EVMTracker, PolynomialRidgeRegressor


def test_evm_cutoff_metrics():
    tracker = EVMTracker()
    metrics = tracker.compute_cutoff_metrics()

    tech = metrics["technical"]
    fin = metrics["financial"]

    # --- Technical Dimension ---
    assert tech["BAC"] == 240.0
    assert tech["PV"] == 114.0
    assert tech["EV"] == 114.0
    assert tech["AC"] == 111.0
    assert tech["CV"] == 3.0
    assert tech["SV"] == 0.0
    assert abs(tech["CPI"] - 1.0270) < 1e-3
    assert abs(tech["SPI"] - 1.0000) < 1e-3
    assert abs(tech["CSI"] - 1.0270) < 1e-3
    assert abs(tech["EAC_typical"] - 233.68) < 0.1
    assert abs(tech["EAC_ml"] - 235.63) < 0.1
    assert abs(tech["ETC"] - 122.68) < 0.1
    assert abs(tech["VAC"] - 6.32) < 0.1
    assert abs(tech["TCPI"] - 0.9767) < 1e-3

    # --- Financial Dimension ---
    assert fin["BAC"] == 65400.0
    assert fin["PV"] == 32700.0
    assert fin["EV"] == 32700.0
    assert fin["AC"] == 31850.0
    assert fin["CV"] == 850.0
    assert fin["SV"] == 0.0
    assert abs(fin["CPI"] - 1.0267) < 1e-3
    assert abs(fin["SPI"] - 1.0000) < 1e-3
    assert abs(fin["CSI"] - 1.0267) < 1e-3
    assert abs(fin["EAC_typical"] - 63699.91) < 1.0
    assert abs(fin["EAC_ml"] - 64120.00) < 1.0
    assert abs(fin["ETC"] - 31849.91) < 1.0
    assert abs(fin["VAC"] - 1700.09) < 1.0
    assert abs(fin["TCPI"] - 0.9747) < 1e-3


def test_polynomial_ridge_regressor():
    import numpy as np

    # Synthetic quadratic data: y = 2 + 3*x + 0.5*x^2
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = 2.0 + 3.0 * X[:, 0] + 0.5 * (X[:, 0] ** 2)

    model = PolynomialRidgeRegressor(degree=2, l2_reg=1e-5)
    model.fit(X, y)
    preds = model.predict(X)

    # MSE should be close to 0
    mse = float(np.mean((preds - y) ** 2))
    assert mse < 1e-2


def test_ml_future_forecast_structure():
    tracker = EVMTracker()
    forecast = tracker.predict_future_sprints_ml()

    assert forecast["eac_ml_hours"] == 235.63
    assert forecast["eac_ml_cost"] == 64120.00
    assert len(forecast["sprints"]) == 6
    assert len(forecast["pv_hours"]) == 6
    assert len(forecast["ac_hours_ml"]) == 6
    assert forecast["ac_hours_ml"][-1] == 235.63
