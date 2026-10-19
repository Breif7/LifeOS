"""
Unit tests for Forge3D manufacturing quotation and lot planning.
"""

import pytest
from lifeos.forge3d import Forge3DEngine


def test_quote_calculation_pla():
    engine = Forge3DEngine()
    quote = engine.calculate_quote(
        job_name="Case Enclosure ML Device",
        grams_per_unit=50.0,
        hours_per_unit=3.0,
        units=5,
        material="PLA",
        post_processing_minutes=10.0,
        margin_percent=40.0,
    )

    assert quote.units == 5
    assert quote.material == "PLA"
    assert quote.total_print_hours == 15.0
    assert quote.base_cost_total > 0
    assert quote.final_price_total > quote.base_cost_total
    assert quote.price_per_unit == round(quote.final_price_total / 5.0, 2)


def test_preventive_maintenance_threshold():
    engine = Forge3DEngine(maintenance_cycle_hours=30.0)

    # Job 1: 20 hours -> accumulated 20h < 30h
    q1 = engine.calculate_quote("Job 1", 100, 20.0, 1)
    assert q1.calibration_needed is False

    # Job 2: 15 hours -> accumulated 35h >= 30h
    q2 = engine.calculate_quote("Job 2", 100, 15.0, 1)
    assert q2.calibration_needed is True
