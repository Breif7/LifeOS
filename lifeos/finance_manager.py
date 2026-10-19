"""
Personal Financial Management & Credit Card Optimization Module.
PRJ-LIFEOS-ML-2026-Q4 — LifeOS-ML
Manages the 10 credit cards, zero-interest payment scheduling,
cash flow synchronization with payroll, and daily $300 discretionary expense cap.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import date

from lifeos.config import (
    BIWEEKLY_AMORTIZATION,
    BIWEEKLY_BAC_OPEX,
    BIWEEKLY_INCOME,
    BIWEEKLY_INVESTMENT,
    BIWEEKLY_LIQUIDITY_BUFFER,
    CREDIT_CARDS,
    DAILY_DISCRETIONARY_CAP,
    MONTHLY_MINIMUM_PAYMENT,
    MONTHLY_NO_INTEREST_PAYMENT,
    TOTAL_CREDIT_DEBT,
    CreditCard,
)


@dataclass
class ExpenseRecord:
    description: str
    amount: float
    category: str  # 'discretionary', 'fixed_opex', 'debt_payment', 'investment'
    day_of_month: int
    wbs_id: Optional[str] = None


class FinanceManager:
    """
    Financial governor for LifeOS-ML.
    Synchronizes biweekly payroll with credit card deadlines and strictly enforces
    zero interest charges and discretionary spend caps.
    """

    def __init__(self, cards: Optional[List[CreditCard]] = None):
        self.cards = cards if cards is not None else CREDIT_CARDS
        self.expenses: List[ExpenseRecord] = []

    def get_portfolio_summary(self) -> Dict[str, float]:
        """Calculates total debt, required monthly payments, and coverage ratios."""
        total_balance = sum(c.balance for c in self.cards)
        total_no_interest = sum(c.no_interest_payment for c in self.cards)
        total_minimum = sum(c.minimum_payment for c in self.cards)

        monthly_income = BIWEEKLY_INCOME * 2
        monthly_opex = BIWEEKLY_BAC_OPEX * 2
        monthly_free_cashflow = monthly_income - monthly_opex

        return {
            "total_debt_balance": round(total_balance, 2),
            "monthly_no_interest_payment": round(total_no_interest, 2),
            "monthly_minimum_payment": round(total_minimum, 2),
            "monthly_income": round(monthly_income, 2),
            "monthly_opex_bac": round(monthly_opex, 2),
            "monthly_free_cashflow": round(monthly_free_cashflow, 2),
            "debt_coverage_ratio": round(monthly_free_cashflow / total_no_interest, 2),
            "interest_paid_rate": 0.0,  # 0% Target Win Condition
        }

    def get_biweekly_payment_schedule(self) -> Dict[str, Dict]:
        """
        Splits credit card payments into Quincena 1 (payroll day 1 to 15)
        and Quincena 2 (payroll day 16 to end of month).
        """
        q1_cards = [c for c in self.cards if c.due_day <= 15]
        q2_cards = [c for c in self.cards if c.due_day > 15]

        q1_total = sum(c.no_interest_payment for c in q1_cards)
        q2_total = sum(c.no_interest_payment for c in q2_cards)

        return {
            "quincena_1": {
                "period": "Día 1 al 15 (Vencimientos día 5 al 9)",
                "allocated_payroll": BIWEEKLY_INCOME,
                "bac_opex": BIWEEKLY_BAC_OPEX,
                "liquidity_buffer": BIWEEKLY_LIQUIDITY_BUFFER,
                "total_no_interest_required": round(q1_total, 2),
                "is_covered": BIWEEKLY_BAC_OPEX >= q1_total,
                "cards": [
                    {
                        "name": c.name,
                        "due_day": c.due_day,
                        "amount_due": c.no_interest_payment,
                        "cat_rate": c.cat_rate,
                    }
                    for c in q1_cards
                ],
            },
            "quincena_2": {
                "period": "Día 16 al fin de mes (Vencimientos día 16 al 26)",
                "allocated_payroll": BIWEEKLY_INCOME,
                "bac_opex": BIWEEKLY_BAC_OPEX,
                "liquidity_buffer": BIWEEKLY_LIQUIDITY_BUFFER,
                "total_no_interest_required": round(q2_total, 2),
                "is_covered": BIWEEKLY_BAC_OPEX >= q2_total,
                "cards": [
                    {
                        "name": c.name,
                        "due_day": c.due_day,
                        "amount_due": c.no_interest_payment,
                        "cat_rate": c.cat_rate,
                    }
                    for c in q2_cards
                ],
            },
        }

    def register_expense(
        self,
        description: str,
        amount: float,
        category: str = "discretionary",
        day_of_month: int = 1,
        wbs_id: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Registers an operational or discretionary expense.
        Monitors R-04 risk: Discretionary expenses exceeding $300 MXN/day.
        """
        expense = ExpenseRecord(
            description=description,
            amount=amount,
            category=category,
            day_of_month=day_of_month,
            wbs_id=wbs_id,
        )
        self.expenses.append(expense)

        # Check daily discretionary cap
        daily_discretionary = sum(
            e.amount
            for e in self.expenses
            if e.category == "discretionary" and e.day_of_month == day_of_month
        )

        exceeds_cap = daily_discretionary > DAILY_DISCRETIONARY_CAP
        alert_msg = None
        if exceeds_cap:
            overflow = daily_discretionary - DAILY_DISCRETIONARY_CAP
            alert_msg = (
                f"ALERTA RIESGO R-04: Gasto discrecional acumulado en el día {day_of_month} "
                f"(${daily_discretionary:.2f} MXN) supera el tope diario de ${DAILY_DISCRETIONARY_CAP:.2f} MXN "
                f"en ${overflow:.2f} MXN. Requiere ajuste compensatorio en rubros variables."
            )

        return {
            "registered_expense": expense,
            "daily_discretionary_total": round(daily_discretionary, 2),
            "daily_cap": DAILY_DISCRETIONARY_CAP,
            "exceeds_cap": exceeds_cap,
            "alert": alert_msg,
        }
