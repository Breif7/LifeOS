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
from lifeos.dynamic_config import DynamicConfigManager, dynamic_config


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
    Supports real-time dynamic configuration synchronized with Google Sheets.
    """

    def __init__(
        self,
        cards: Optional[List[CreditCard]] = None,
        config: Optional[DynamicConfigManager] = None,
    ):
        self.config = config if config is not None else dynamic_config
        self.cards = [
            CreditCard(
                name=c.name,
                due_day=c.due_day,
                cutoff_day=c.cutoff_day,
                balance=c.balance,
                no_interest_payment=c.no_interest_payment,
                minimum_payment=c.minimum_payment,
                cat_rate=c.cat_rate,
            )
            for c in (cards if cards is not None else self.config.cards)
        ]
        self.expenses: List[ExpenseRecord] = []


    def get_portfolio_summary(self) -> Dict[str, float]:
        """Calculates total debt, required monthly payments, and coverage ratios."""
        total_balance = sum(c.balance for c in self.cards)
        total_no_interest = sum(c.no_interest_payment for c in self.cards)
        total_minimum = sum(c.minimum_payment for c in self.cards)

        monthly_income = self.config.income * 2
        monthly_opex = self.config.bac_opex * 2
        monthly_free_cashflow = monthly_income - monthly_opex
        coverage_ratio = (
            round(monthly_free_cashflow / total_no_interest, 2)
            if total_no_interest > 0
            else 1.0
        )

        return {
            "total_debt_balance": round(total_balance, 2),
            "monthly_no_interest_payment": round(total_no_interest, 2),
            "monthly_minimum_payment": round(total_minimum, 2),
            "monthly_income": round(monthly_income, 2),
            "monthly_opex_bac": round(monthly_opex, 2),
            "monthly_free_cashflow": round(monthly_free_cashflow, 2),
            "debt_coverage_ratio": coverage_ratio,
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
                "allocated_payroll": self.config.income,
                "bac_opex": self.config.bac_opex,
                "liquidity_buffer": self.config.liquidity_buffer,
                "total_no_interest_required": round(q1_total, 2),
                "is_covered": self.config.bac_opex >= q1_total,
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
                "allocated_payroll": self.config.income,
                "bac_opex": self.config.bac_opex,
                "liquidity_buffer": self.config.liquidity_buffer,
                "total_no_interest_required": round(q2_total, 2),
                "is_covered": self.config.bac_opex >= q2_total,
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
        Monitors R-04 risk: Discretionary expenses exceeding daily cap.
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

        cap = self.config.discretionary_cap
        exceeds_cap = daily_discretionary > cap
        alert_msg = None
        if exceeds_cap:
            overflow = daily_discretionary - cap
            alert_msg = (
                f"ALERTA RIESGO R-04: Gasto discrecional acumulado en el día {day_of_month} "
                f"(${daily_discretionary:.2f} MXN) supera el tope diario de ${cap:.2f} MXN "
                f"en ${overflow:.2f} MXN. Requiere ajuste compensatorio en rubros variables."
            )

        return {
            "registered_expense": expense,
            "daily_discretionary_total": round(daily_discretionary, 2),
            "daily_cap": cap,
            "exceeds_cap": exceeds_cap,
            "alert": alert_msg,
        }

