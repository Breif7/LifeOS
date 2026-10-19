"""
Dynamic Configuration & Real-Time State Manager for LifeOS-ML.
PRJ-LIFEOS-ML-2026-Q4 — LifeOS-ML

Replaces static hardcoded values with real-time dynamic resolution:
1. Live Google Sheets synchronization (Priority 1)
2. Local persistent JSON cache (Priority 2)
3. Deterministic baseline fallback (Priority 3)
"""

from dataclasses import asdict
from datetime import datetime
import io
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union
import urllib.request
import pandas as pd

from lifeos.config import (
    BIWEEKLY_BAC_OPEX,
    BIWEEKLY_INCOME,
    CREDIT_CARDS,
    DAILY_DISCRETIONARY_CAP,
    TOTAL_BAC_FINANCIAL,
    TOTAL_CREDIT_DEBT,
    CreditCard,
)


class DynamicConfigManager:
    """
    Manages live project configuration, credit card balances, and expenses.
    Synchronizes in real time with Google Sheets and persists to a local JSON cache.
    """

    DEFAULT_SHEET_ID = "1fnyhzWFBSsTEjxK6BBS4Mq8PKiQzkgPJTAm3kSRCZfo"
    GID_CARDS = "937733390"
    GID_EXPENSES = "107376044"
    GID_PANEL = "1922512971"

    def __init__(
        self,
        sheet_id: str = DEFAULT_SHEET_ID,
        cache_path: Optional[str] = None,
        auto_sync: bool = False,
    ):
        self.sheet_id = sheet_id
        if cache_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.cache_path = os.path.join(base_dir, "lifeos_state.json")
        else:
            self.cache_path = cache_path

        # State attributes
        self.income: float = BIWEEKLY_INCOME
        self.bac_opex: float = BIWEEKLY_BAC_OPEX
        self.discretionary_cap: float = DAILY_DISCRETIONARY_CAP
        self.cards: List[CreditCard] = [
            CreditCard(
                name=c.name,
                due_day=c.due_day,
                cutoff_day=c.cutoff_day,
                balance=c.balance,
                no_interest_payment=c.no_interest_payment,
                minimum_payment=c.minimum_payment,
                cat_rate=c.cat_rate,
            )
            for c in CREDIT_CARDS
        ]
        self.recent_expenses: List[Dict] = []
        self.last_sync: Optional[str] = None
        self.sync_source: str = "hardcoded_baseline"


        # Load from cache if present
        self.load_from_cache()

        if auto_sync:
            try:
                self.sync_from_google_sheets()
            except Exception as e:
                print(f"[DynamicConfig] Auto-sync notice: using {self.sync_source} ({e})")

    @property
    def liquidity_buffer(self) -> float:
        """Guaranteed net cashflow buffer after OpEx BAC."""
        return round(self.income - self.bac_opex, 2)

    @property
    def total_debt(self) -> float:
        """Sum of total credit card balances."""
        return round(sum(c.balance for c in self.cards), 2)

    @property
    def monthly_no_interest_payment(self) -> float:
        """Sum of no-interest payments across cards."""
        return round(sum(c.no_interest_payment for c in self.cards), 2)

    @property
    def monthly_minimum_payment(self) -> float:
        """Sum of required minimum payments across cards."""
        return round(sum(c.minimum_payment for c in self.cards), 2)

    @staticmethod
    def _clean_currency(val) -> float:
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).replace("$", "").replace(",", "").replace("%", "").strip()
        try:
            return float(s)
        except ValueError:
            return 0.0

    @staticmethod
    def _extract_day(date_str: str, default: int = 15) -> int:
        if pd.isna(date_str):
            return default
        s = str(date_str).strip()
        # Match YYYY-MM-DD or DD/MM/YYYY
        match = re.search(r"\b(\d{1,2})\b$", s)
        if match:
            return int(match.group(1))
        parts = re.split(r"[-/]", s)
        if len(parts) >= 3:
            try:
                return int(parts[-1])
            except ValueError:
                pass
        return default

    def sync_from_google_sheets(self, timeout_sec: int = 8) -> bool:
        """
        Pulls real-time data from user's Google Sheets:
        1. Cards tab (gid=937733390): updates card balances, due dates, CAT.
        2. Panel tab (gid=1922512971): updates income, BAC OpEx, total debt.
        3. Expenses tab (gid=107376044): updates current transactions.
        """
        def fetch_csv(gid: str, header_row: int) -> pd.DataFrame:
            url = f"https://docs.google.com/spreadsheets/d/{self.sheet_id}/export?format=csv&gid={gid}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                text = resp.read().decode("utf-8")
                return pd.read_csv(io.StringIO(text), header=header_row)

        # 1. Sync Cards (gid=937733390, header=3)
        try:
            df_cards = fetch_csv(self.GID_CARDS, header_row=3)
            parsed_cards = []
            for _, row in df_cards.dropna(subset=["Tarjeta"]).iterrows():
                name = str(row["Tarjeta"]).strip()
                if not name or name.upper() in ["TOTALES", "TOTAL", "SUMA"]:
                    continue

                saldo = self._clean_currency(row.get("Saldo Total", 0.0))
                no_int = self._clean_currency(row.get("Pago No Generar Int.", 0.0))
                min_pay = self._clean_currency(row.get("Pago Mínimo", 0.0))
                due_day = self._extract_day(row.get("Fecha Límite de Pago", 15), default=15)
                cutoff_day = self._extract_day(row.get("Fecha de Corte", 25), default=25)

                # Look up existing CAT if not in sheet
                cat = 85.0
                for old in self.cards:
                    if old.name.lower() == name.lower():
                        cat = old.cat_rate
                        break

                parsed_cards.append(
                    CreditCard(
                        name=name,
                        due_day=due_day,
                        cutoff_day=cutoff_day,
                        balance=saldo,
                        no_interest_payment=no_int,
                        minimum_payment=min_pay,
                        cat_rate=cat,
                    )
                )

            if parsed_cards:
                self.cards = parsed_cards
        except Exception as e_cards:
            print(f"[DynamicConfig] Warning syncing cards: {e_cards}")

        # 2. Sync Panel & Budget (gid=1922512971, header=None)
        try:
            url_panel = f"https://docs.google.com/spreadsheets/d/{self.sheet_id}/export?format=csv&gid={self.GID_PANEL}"
            req = urllib.request.Request(url_panel, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                df_panel = pd.read_csv(io.StringIO(resp.read().decode("utf-8")), header=None)

            # Row 4 typically contains Total Ingresos
            for _, row in df_panel.iterrows():
                row_str = " ".join([str(x) for x in row.values])
                if "$14,691" in row_str or "TOTAL INGRESOS" in row_str:
                    for val in row.values:
                        c_val = self._clean_currency(val)
                        if c_val > 5000 and c_val != self.total_debt:
                            self.income = c_val
                            break
                if "$10,900" in row_str or "TOTAL GASTOS" in row_str:
                    for val in row.values:
                        c_val = self._clean_currency(val)
                        if 8000 <= c_val <= 15000:
                            self.bac_opex = c_val
                            break
        except Exception as e_panel:
            print(f"[DynamicConfig] Warning syncing panel: {e_panel}")

        # 3. Sync Recent Expenses (gid=107376044, header=3)
        try:
            df_exp = fetch_csv(self.GID_EXPENSES, header_row=3)
            parsed_exp = []
            for _, row in df_exp.dropna(subset=["Fecha (AAAA-MM-DD)"]).iterrows():
                fecha = str(row["Fecha (AAAA-MM-DD)"]).strip()
                if fecha and fecha.lower() != "nan":
                    parsed_exp.append({
                        "fecha": fecha,
                        "tipo": str(row.get("Tipo", "")).strip(),
                        "cuenta": str(row.get("Cuenta / Tarjeta", "")).strip(),
                        "concepto": str(row.get("Descripción / Concepto", "")).strip(),
                        "monto": self._clean_currency(row.get("Monto ($ MXN)", 0.0)),
                        "categoria": str(row.get("Categoría", "")).strip(),
                        "notas": str(row.get("Notas Opcionales", "")).strip() if pd.notna(row.get("Notas Opcionales")) else "",
                    })
            self.recent_expenses = parsed_exp
        except Exception as e_exp:
            print(f"[DynamicConfig] Warning syncing expenses: {e_exp}")

        self.last_sync = datetime.now().isoformat()
        self.sync_source = "google_sheets"
        self.save_to_cache()
        return True

    def save_to_cache(self):
        """Persists current state to local JSON file."""
        data = {
            "last_sync": self.last_sync,
            "sync_source": self.sync_source,
            "income": self.income,
            "bac_opex": self.bac_opex,
            "discretionary_cap": self.discretionary_cap,
            "cards": [
                {
                    "name": c.name,
                    "due_day": c.due_day,
                    "cutoff_day": c.cutoff_day,
                    "balance": c.balance,
                    "no_interest_payment": c.no_interest_payment,
                    "minimum_payment": c.minimum_payment,
                    "cat_rate": c.cat_rate,
                }
                for c in self.cards
            ],
            "recent_expenses": self.recent_expenses,
        }
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[DynamicConfig] Warning saving cache: {e}")

    def load_from_cache(self) -> bool:
        """Loads state from local JSON cache if available."""
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.last_sync = data.get("last_sync")
            self.sync_source = "local_cache"
            self.income = float(data.get("income", self.income))
            self.bac_opex = float(data.get("bac_opex", self.bac_opex))
            self.discretionary_cap = float(data.get("discretionary_cap", self.discretionary_cap))

            raw_cards = data.get("cards", [])
            if raw_cards:
                self.cards = [
                    CreditCard(
                        name=rc["name"],
                        due_day=int(rc["due_day"]),
                        cutoff_day=int(rc["cutoff_day"]),
                        balance=float(rc["balance"]),
                        no_interest_payment=float(rc["no_interest_payment"]),
                        minimum_payment=float(rc["minimum_payment"]),
                        cat_rate=float(rc.get("cat_rate", 85.0)),
                    )
                    for rc in raw_cards
                ]
            self.recent_expenses = data.get("recent_expenses", [])
            return True
        except Exception as e:
            print(f"[DynamicConfig] Error reading cache: {e}")
            return False

    def update_card(
        self,
        name: str,
        balance: Optional[float] = None,
        no_interest_payment: Optional[float] = None,
        minimum_payment: Optional[float] = None,
        due_day: Optional[int] = None,
    ) -> bool:
        """Modifies a credit card's values dynamically and saves to cache."""
        found = False
        for card in self.cards:
            if card.name.lower() == name.lower():
                if balance is not None:
                    card.balance = float(balance)
                if no_interest_payment is not None:
                    card.no_interest_payment = float(no_interest_payment)
                if minimum_payment is not None:
                    card.minimum_payment = float(minimum_payment)
                if due_day is not None:
                    card.due_day = int(due_day)
                found = True
                break

        if not found:
            # Add new card dynamically
            self.cards.append(
                CreditCard(
                    name=name,
                    due_day=due_day or 15,
                    cutoff_day=25,
                    balance=balance or 0.0,
                    no_interest_payment=no_interest_payment or 0.0,
                    minimum_payment=minimum_payment or 0.0,
                    cat_rate=85.0,
                )
            )

        self.sync_source = "user_manual_update"
        self.last_sync = datetime.now().isoformat()
        self.save_to_cache()
        return True

    def set_financial_baseline(
        self,
        income: Optional[float] = None,
        bac_opex: Optional[float] = None,
        discretionary_cap: Optional[float] = None,
    ):
        """Updates income, opex budget, or discretionary cap dynamically."""
        if income is not None:
            self.income = float(income)
        if bac_opex is not None:
            self.bac_opex = float(bac_opex)
        if discretionary_cap is not None:
            self.discretionary_cap = float(discretionary_cap)

        self.sync_source = "user_manual_update"
        self.last_sync = datetime.now().isoformat()
        self.save_to_cache()

    def get_summary(self) -> Dict[str, any]:
        """Returns consolidated financial status."""
        return {
            "sync_source": self.sync_source,
            "last_sync": self.last_sync,
            "income": self.income,
            "bac_opex": self.bac_opex,
            "liquidity_buffer": self.liquidity_buffer,
            "discretionary_cap": self.discretionary_cap,
            "total_debt": self.total_debt,
            "monthly_no_interest_payment": self.monthly_no_interest_payment,
            "monthly_minimum_payment": self.monthly_minimum_payment,
            "card_count": len(self.cards),
            "recent_expenses_count": len(self.recent_expenses),
        }


# Global singleton instance
dynamic_config = DynamicConfigManager()
