"""
Unit tests for Dynamic Configuration & State Management (DynamicConfigManager).
Validates dynamic persistence, cache read/write, manual updates, and FinanceManager integration.
"""

import os
import pytest
from lifeos.dynamic_config import DynamicConfigManager
from lifeos.finance_manager import FinanceManager


def test_default_baseline_fallback(tmp_path):
    cache_file = tmp_path / "non_existent.json"
    mgr = DynamicConfigManager(cache_path=str(cache_file))

    assert mgr.income == 14691.00
    assert mgr.bac_opex == 10900.00
    assert mgr.discretionary_cap == 300.00
    assert mgr.liquidity_buffer == 3791.00
    assert mgr.total_debt == 47238.00
    assert len(mgr.cards) == 10
    assert mgr.sync_source == "hardcoded_baseline"


def test_save_and_load_cache(tmp_path):
    cache_file = tmp_path / "state_test.json"
    mgr1 = DynamicConfigManager(cache_path=str(cache_file))

    # Update values
    mgr1.set_financial_baseline(income=16500.0, bac_opex=11200.0, discretionary_cap=350.0)
    mgr1.update_card("AMEX", balance=4000.0, no_interest_payment=2000.0, due_day=10)

    assert os.path.exists(cache_file)

    # Initialize a new instance pointing to same cache
    mgr2 = DynamicConfigManager(cache_path=str(cache_file))
    assert mgr2.income == 16500.0
    assert mgr2.bac_opex == 11200.0
    assert mgr2.discretionary_cap == 350.0
    assert mgr2.sync_source == "local_cache"

    amex_card = next(c for c in mgr2.cards if c.name == "AMEX")
    assert amex_card.balance == 4000.0
    assert amex_card.no_interest_payment == 2000.0
    assert amex_card.due_day == 10


def test_dynamic_card_addition(tmp_path):
    cache_file = tmp_path / "cards_test.json"
    mgr = DynamicConfigManager(cache_path=str(cache_file))
    initial_count = len(mgr.cards)

    mgr.update_card("Santander Fiesta", balance=8500.0, no_interest_payment=1500.0, due_day=22)

    assert len(mgr.cards) == initial_count + 1
    new_card = next(c for c in mgr.cards if c.name == "Santander Fiesta")
    assert new_card.balance == 8500.0
    assert new_card.due_day == 22


def test_finance_manager_dynamic_integration(tmp_path):
    cache_file = tmp_path / "fin_test.json"
    mgr = DynamicConfigManager(cache_path=str(cache_file))
    mgr.set_financial_baseline(income=18000.0, bac_opex=12000.0, discretionary_cap=400.0)

    fin_mgr = FinanceManager(config=mgr)
    summary = fin_mgr.get_portfolio_summary()

    # Monthly is biweekly * 2
    assert summary["monthly_income"] == 36000.0
    assert summary["monthly_opex_bac"] == 24000.0
    assert summary["monthly_free_cashflow"] == 12000.0

    # Expense registered under new cap
    res = fin_mgr.register_expense("Cena", 380.0, category="discretionary", day_of_month=5)
    assert res["daily_cap"] == 400.0
    assert res["exceeds_cap"] is False
