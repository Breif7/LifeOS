"""
Unit tests for Financial Management, credit cards optimization, and daily spending cap.
"""

import pytest
from lifeos.finance_manager import FinanceManager


def test_portfolio_summary():
    mgr = FinanceManager()
    summary = mgr.get_portfolio_summary()

    assert summary["total_debt_balance"] == 47238.00
    assert summary["monthly_no_interest_payment"] == 11341.00
    assert summary["monthly_minimum_payment"] == 3518.00
    assert summary["monthly_income"] == 29382.00
    assert summary["monthly_opex_bac"] == 21800.00
    assert summary["interest_paid_rate"] == 0.0


def test_payment_schedule_split():
    mgr = FinanceManager()
    schedule = mgr.get_biweekly_payment_schedule()

    q1 = schedule["quincena_1"]
    q2 = schedule["quincena_2"]

    # Quincena 1 should cover cards due day 5 to 15 (Platinum, Didi, BBVA Oro, AMEX)
    q1_card_names = [c["name"] for c in q1["cards"]]
    assert "Platinum" in q1_card_names
    assert "Didi" in q1_card_names
    assert "BBVA Oro" in q1_card_names
    assert "AMEX" in q1_card_names
    assert q1["is_covered"] is True

    # Quincena 2 should cover cards due day 16 to 26 (Nu, MP, Revolut, Jun, Plata, LikeU)
    q2_card_names = [c["name"] for c in q2["cards"]]
    assert "Nu" in q2_card_names
    assert "Mercado Pago" in q2_card_names
    assert "LikeU" in q2_card_names
    assert q2["is_covered"] is True


def test_discretionary_daily_cap_and_alerts():
    mgr = FinanceManager()

    # Expense 1: Under $300 limit
    res1 = mgr.register_expense(
        description="Almuerzo de trabajo",
        amount=180.0,
        category="discretionary",
        day_of_month=3,
    )
    assert res1["daily_discretionary_total"] == 180.0
    assert res1["exceeds_cap"] is False
    assert res1["alert"] is None

    # Expense 2: Exceeding $300 limit on same day
    res2 = mgr.register_expense(
        description="Café de especialidad y postre",
        amount=150.0,
        category="discretionary",
        day_of_month=3,
    )
    assert res2["daily_discretionary_total"] == 330.0
    assert res2["exceeds_cap"] is True
    assert "ALERTA RIESGO R-04" in res2["alert"]
