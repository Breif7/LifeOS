"""
LifeOS Daily Standup & Card Check Script.
PRJ-LIFEOS-ML-2026-Q4 — Automation Helper
"""

import sys
from datetime import datetime
from typing import List, Tuple

from lifeos.config import CREDIT_CARDS, DAILY_DISCRETIONARY_CAP
from lifeos.finance_manager import FinanceManager
from lifeos.nlp_classifier import LifeOSNLPClassifier


def get_upcoming_cards(target_day: int, days_ahead: int = 3) -> List[Tuple[any, int]]:
    """Returns cards whose due date is within [target_day, target_day + days_ahead]."""
    upcoming = []
    for card in CREDIT_CARDS:
        diff = card.due_day - target_day
        if 0 <= diff <= days_ahead:
            upcoming.append((card, diff))
    upcoming.sort(key=lambda x: x[1])
    return upcoming


def run_daily_standup(day: int = None):
    if day is None:
        day = datetime.now().day

    print(f"============================================================")
    print(f"        LIFEOS-ML: REPORTE DIARIO MATUTINO (Día {day})")
    print(f"============================================================")

    upcoming = get_upcoming_cards(day, days_ahead=3)
    if upcoming:
        print("\n[ALERTA DE PAGOS] Tarjetas con vencimiento en los próximos 3 días:")
        total_due_soon = 0.0
        for card, diff in upcoming:
            total_due_soon += card.no_interest_payment
            when_str = "¡VENCE HOY!" if diff == 0 else f"Vence en {diff} día(s) (Día {card.due_day})"
            print(f"  * {card.name:14s} | {when_str:22s} | Pago sin intereses: ${card.no_interest_payment:,.2f} MXN (CAT: {card.cat_rate:.1f}%)")
        print(f"  -> Total requerido en esta ventana: ${total_due_soon:,.2f} MXN")
    else:
        print("\n[FINANZAS] No hay tarjetas con vencimiento en los próximos 3 días.")

    fin_mgr = FinanceManager()
    summary = fin_mgr.get_portfolio_summary()

    print("\n[LÍNEA BASE FINANCIERA & DISCIPLINA]")
    print(f"  * Tope diario de gasto discrecional: ${DAILY_DISCRETIONARY_CAP:.2f} MXN (Alerta R-04 si se supera)")
    print(f"  * Pago mensual total para 0% intereses: ${summary['monthly_no_interest_payment']:,.2f} MXN")
    print(f"  * Deuda total en administración: ${summary['total_debt_balance']:,.2f} MXN")

    print("\n[PLANIFICACIÓN KANBAN DEL DÍA (Límite WIP = 3)]")
    print("  1. Maestría UNIR:         [Definir 1 tarea de posgrado]")
    print("  2. Ingeniería ML/Código:  [Definir 1 tarea técnica de portafolio]")
    print("  3. Finanzas/Operaciones:  [Definir 1 tarea de pagos o Forge3D]")

    print("\n[HÁBITO CUADRANTE II]")
    print("  * 30 minutos de ejercicio físico diario programados.")
    print(f"============================================================\n")


if __name__ == "__main__":
    current_day = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_daily_standup(current_day)
