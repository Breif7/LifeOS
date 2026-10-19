"""
Unit tests for CPM/PERT network and Monte Carlo simulation.
"""

import pytest
from lifeos.cpm_pert import CPMNetwork


def test_cpm_schedule_calculation():
    net = CPMNetwork()
    packages = net.compute_schedule()

    # 1. Total packages count
    assert len(packages) == 16

    # 2. Project duration along critical path
    assert net.project_duration == 166.0

    # 3. Critical Path tasks
    expected_critical = [
        "1.1", "1.2", "2.1", "3.1", "3.2", "3.3", "4.2", "4.4", "5.1", "5.2", "5.3"
    ]
    assert net.critical_path == expected_critical

    # 4. Check critical tasks have zero total slack
    for cid in expected_critical:
        assert packages[cid].is_critical is True
        assert abs(packages[cid].total_slack) < 1e-4

    # 5. Check non-critical tasks and their slacks
    assert packages["1.3"].is_critical is False
    assert packages["1.3"].total_slack == 6.0

    assert packages["2.2"].is_critical is False
    assert packages["2.2"].total_slack == 22.0

    assert packages["2.3"].is_critical is False
    assert packages["2.3"].total_slack == 6.0

    assert packages["4.1"].is_critical is False
    assert packages["4.1"].total_slack == 2.0

    assert packages["4.3"].is_critical is False
    assert packages["4.3"].total_slack == 4.0


def test_pert_statistics():
    net = CPMNetwork()
    stats = net.get_pert_statistics()

    assert stats["deterministic_cp_hours"] == 166.0
    assert stats["total_scope_hours"] == 240.0
    # Expected PERT mean around 170.15 - 170.17h (within 0.1h tolerance)
    assert abs(stats["pert_mean_cp_hours"] - 170.17) < 0.2
    assert abs(stats["pert_std_cp_hours"] - 3.79) < 0.2


def test_monte_carlo_simulation():
    net = CPMNetwork()
    sim_res = net.run_monte_carlo_simulation(iterations=2000, target_hours=255.0, random_seed=42)

    assert sim_res["iterations"] == 2000
    assert sim_res["mean_hours"] > 165.0
    assert sim_res["p85_hours"] < 255.0
    # Probability under 255 hours should be very high (> 85%)
    assert sim_res["prob_under_threshold"] >= 85.0
