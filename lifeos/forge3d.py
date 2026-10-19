"""
Forge3D: 3D Printing Batch Quotation & Lot Scheduling Module.
PRJ-LIFEOS-ML-2026-Q4 — WBS 4.3
Tailored for Anycubic Kobra S1 manufacturing operations.
Calculates material cost, power consumption, machine depreciation,
preventive maintenance reserves (R-05), and lot delivery lead times.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FilamentProfile:
    material_name: str
    cost_per_kg_mxn: float
    density_g_cm3: float = 1.24  # Standard PLA
    recommended_nozzle_temp: int = 210
    recommended_bed_temp: int = 60


DEFAULT_FILAMENTS: Dict[str, FilamentProfile] = {
    "PLA": FilamentProfile("PLA", 420.0, 1.24, 210, 60),
    "PETG": FilamentProfile("PETG", 480.0, 1.27, 235, 80),
    "ABS": FilamentProfile("ABS", 520.0, 1.05, 245, 95),
    "TPU": FilamentProfile("TPU", 650.0, 1.21, 220, 50),
}


@dataclass
class PrintQuote:
    job_name: str
    units: int
    material: str
    grams_per_unit: float
    hours_per_unit: float
    total_material_cost: float
    total_energy_cost: float
    total_depreciation_cost: float
    total_maintenance_reserve: float
    total_labor_cost: float
    base_cost_total: float
    margin_percent: float
    final_price_total: float
    price_per_unit: float
    total_print_hours: float
    calibration_needed: bool


class Forge3DEngine:
    """
    Quotation and operational planning engine for 3D additive manufacturing.
    """

    def __init__(
        self,
        printer_power_watts: float = 250.0,
        electricity_kwh_cost_mxn: float = 3.50,
        machine_depreciation_per_hour_mxn: float = 4.00,
        labor_rate_per_hour_mxn: float = 150.0,
        maintenance_cycle_hours: float = 30.0,
    ):
        self.printer_power_watts = printer_power_watts
        self.electricity_kwh_cost_mxn = electricity_kwh_cost_mxn
        self.machine_depreciation_per_hour_mxn = machine_depreciation_per_hour_mxn
        self.labor_rate_per_hour_mxn = labor_rate_per_hour_mxn
        self.maintenance_cycle_hours = maintenance_cycle_hours
        self.filaments = DEFAULT_FILAMENTS
        self.accumulated_print_hours: float = 0.0

    def calculate_quote(
        self,
        job_name: str,
        grams_per_unit: float,
        hours_per_unit: float,
        units: int = 1,
        material: str = "PLA",
        post_processing_minutes: float = 15.0,
        margin_percent: float = 40.0,
    ) -> PrintQuote:
        """
        Computes accurate manufacturing quote for an additive manufacturing batch.
        """
        fil_profile = self.filaments.get(material.upper(), self.filaments["PLA"])

        total_grams = grams_per_unit * units
        total_hours = hours_per_unit * units

        # 1. Raw material cost
        material_cost = (total_grams / 1000.0) * fil_profile.cost_per_kg_mxn

        # 2. Electricity cost: kWh = (Watts * hours) / 1000
        total_kwh = (self.printer_power_watts * total_hours) / 1000.0
        energy_cost = total_kwh * self.electricity_kwh_cost_mxn

        # 3. Machine wear & depreciation
        depreciation_cost = total_hours * self.machine_depreciation_per_hour_mxn

        # 4. Preventive maintenance reserve (R-05: boquilla repuesto y lubricación)
        # Allocate $6.50 MXN per print hour towards nozzle and lubrication upkeep
        maintenance_reserve = total_hours * 6.50

        # 5. Labor / post-processing
        total_labor_hours = (post_processing_minutes * units) / 60.0
        labor_cost = total_labor_hours * self.labor_rate_per_hour_mxn

        base_cost = material_cost + energy_cost + depreciation_cost + maintenance_reserve + labor_cost

        # 6. Commercial price with target margin
        final_price = round(base_cost * (1.0 + margin_percent / 100.0), 2)
        unit_price = round(final_price / units, 2)

        # Check maintenance threshold
        self.accumulated_print_hours += total_hours
        calibration_needed = self.accumulated_print_hours >= self.maintenance_cycle_hours

        return PrintQuote(
            job_name=job_name,
            units=units,
            material=material.upper(),
            grams_per_unit=round(grams_per_unit, 2),
            hours_per_unit=round(hours_per_unit, 2),
            total_material_cost=round(material_cost, 2),
            total_energy_cost=round(energy_cost, 2),
            total_depreciation_cost=round(depreciation_cost, 2),
            total_maintenance_reserve=round(maintenance_reserve, 2),
            total_labor_cost=round(labor_cost, 2),
            base_cost_total=round(base_cost, 2),
            margin_percent=margin_percent,
            final_price_total=round(final_price, 2),
            price_per_unit=round(unit_price, 2),
            total_print_hours=round(total_hours, 2),
            calibration_needed=calibration_needed,
        )
