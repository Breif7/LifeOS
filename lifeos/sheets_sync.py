"""
Google Sheets Bidirectional Synchronization Module for LifeOS-ML.
PRJ-LIFEOS-ML-2026-Q4 — LifeOS-ML
Supports:
1. Colab Native Auth (google.colab.auth + gspread)
2. Service Account Auth (service_account.json)
3. Public CSV Link Sharing Mode
4. Flexible column parsing for Credit Cards, Daily Expenses, and EVM Dashboard sync.
"""

import io
import os
import re
import urllib.request
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from lifeos.config import (
    CREDIT_CARDS,
    CreditCard,
)


class GoogleSheetsSync:
    """
    Bidirectional synchronization engine between Google Sheets and LifeOS-ML.
    """

    DEFAULT_SHEET_ID = "1fnyhzWFBSsTEjxK6BBS4Mq8PKiQzkgPJTAm3kSRCZfo"
    DEFAULT_GID = "921608852"

    def __init__(
        self,
        sheet_id: str = DEFAULT_SHEET_ID,
        service_account_path: Optional[str] = None,
    ):
        self.sheet_id = sheet_id
        self.service_account_path = service_account_path
        self.gc = None  # gspread client if authenticated

    def init_client(self):
        """Initializes gspread client via Service Account or Colab native auth."""
        import gspread

        # 1. Check Service Account JSON
        sa_file = self.service_account_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if sa_file and os.path.exists(sa_file):
            self.gc = gspread.service_account(filename=sa_file)
            return self.gc

        # 2. Check Google Colab environment
        try:
            import google.colab
            from google.colab import auth
            from google.auth import default

            auth.authenticate_user()
            creds, _ = default()
            self.gc = gspread.authorize(creds)
            return self.gc
        except Exception:
            pass

        raise RuntimeError(
            "No se detectaron credenciales activas de Google. Para sincronización en segundo plano, "
            "proporciona un archivo 'service_account.json' o ejecuta la autenticación nativa en Google Colab."
        )

    def fetch_via_public_csv(self, gid: Optional[str] = None) -> pd.DataFrame:
        """
        Fetches worksheet data directly via Google Sheets CSV export URL.
        Requires sheet sharing set to 'Cualquier persona con el enlace (Lector)'.
        """
        gid_str = gid if gid is not None else self.DEFAULT_GID
        url = f"https://docs.google.com/spreadsheets/d/{self.sheet_id}/export?format=csv&gid={gid_str}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        try:
            with urllib.request.urlopen(req) as resp:
                csv_bytes = resp.read()
                df = pd.read_csv(io.BytesIO(csv_bytes))
                return df
        except urllib.error.HTTPError as e:
            if e.code == 401 or e.code == 403:
                raise PermissionError(
                    f"Acceso denegado (HTTP {e.code}). La hoja de Google Sheets está privada. "
                    "Para lectura directa sin credenciales, activa en Google Sheets: Compartir -> 'Cualquier persona con el enlace (Lector)'. "
                    "O bien, usa una Cuenta de Servicio (service_account.json)."
                )
            raise e

    def fetch_worksheet(self, gid: Optional[str] = None, title: Optional[str] = None) -> pd.DataFrame:
        """Fetches worksheet using gspread authenticated client."""
        if self.gc is None:
            self.init_client()

        spreadsheet = self.gc.open_by_key(self.sheet_id)
        if title:
            ws = spreadsheet.worksheet(title)
        elif gid:
            ws = spreadsheet.get_worksheet_by_id(int(gid))
        else:
            ws = spreadsheet.sheet1

        records = ws.get_all_records()
        return pd.DataFrame(records)

    @staticmethod
    def parse_cards_dataframe(df: pd.DataFrame) -> List[CreditCard]:
        """
        Fuzzy-parses DataFrame columns to extract credit card attributes:
        Card Name, Balance, No-Interest Payment, Minimum Payment, Due Day, Cutoff Day, CAT.
        """
        def normalize_str(s: str) -> str:
            s = str(s).lower()
            for src, dst in [('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'), ('ñ', 'n')]:
                s = s.replace(src, dst)
            return re.sub(r"[^\w]", "", s)

        # Normalize column headers
        col_map = {}
        for col in df.columns:
            clean = normalize_str(col)
            if any(k in clean for k in ["tarjeta", "banco", "nombre", "card", "cuenta"]):
                col_map["name"] = col
            elif any(k in clean for k in ["nointeres", "sininteres", "pagonointeres", "intereses", "parano"]):
                col_map["no_interest"] = col
            elif any(k in clean for k in ["minimo", "pagominimo"]):
                col_map["minimum"] = col
            elif any(k in clean for k in ["saldo", "balance", "total"]):
                col_map["balance"] = col
            elif any(k in clean for k in ["diapago", "fechapago", "limite", "vence", "fechalimite"]):
                col_map["due_day"] = col
            elif any(k in clean for k in ["corte", "diacorte", "fechacorte"]):
                col_map["cutoff_day"] = col
            elif "cat" in clean:
                col_map["cat"] = col

        if "name" not in col_map:
            raise ValueError(f"No se encontró columna para nombre de tarjeta en: {df.columns.tolist()}")

        parsed_cards: List[CreditCard] = []

        def clean_num(val, default=0.0):
            if pd.isna(val):
                return default
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).replace("$", "").replace(",", "").replace("%", "").strip()
            try:
                return float(s)
            except ValueError:
                return default

        for _, row in df.iterrows():
            name = str(row[col_map["name"]]).strip()
            clean_name = normalize_str(name)
            if not name or any(t in clean_name for t in ["total", "promedio", "suma", "general"]):
                continue

            balance = clean_num(row.get(col_map.get("balance"), 0.0))
            no_interest = clean_num(row.get(col_map.get("no_interest"), 0.0))
            minimum = clean_num(row.get(col_map.get("minimum"), 0.0))
            due_day = int(clean_num(row.get(col_map.get("due_day"), 15), 15))
            cutoff_day = int(clean_num(row.get(col_map.get("cutoff_day"), 25), 25))
            cat = clean_num(row.get(col_map.get("cat"), 85.0), 85.0)

            parsed_cards.append(
                CreditCard(
                    name=name,
                    due_day=due_day,
                    cutoff_day=cutoff_day,
                    balance=balance,
                    no_interest_payment=no_interest,
                    minimum_payment=minimum,
                    cat_rate=cat,
                )
            )

        return parsed_cards

    def append_expense_row(
        self,
        description: str,
        amount: float,
        category: str,
        covey_quadrant: str,
        wbs_id: str,
        worksheet_title: str = "Gastos_LifeOS",
    ) -> bool:
        """Appends a new expense transaction row to the Google Sheet."""
        if self.gc is None:
            self.init_client()

        spreadsheet = self.gc.open_by_key(self.sheet_id)
        try:
            ws = spreadsheet.worksheet(worksheet_title)
        except Exception:
            # Create if not exists
            ws = spreadsheet.add_worksheet(title=worksheet_title, rows=1000, cols=10)
            ws.append_row(["Fecha", "Descripción", "Monto (MXN)", "Categoría", "Cuadrante Covey", "WBS_ID"])

        from datetime import datetime
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([now_str, description, amount, category, covey_quadrant, wbs_id])
        return True

    def write_evm_summary(
        self,
        cutoff_metrics: Dict[str, Dict[str, float]],
        worksheet_title: str = "EVM_Dashboard",
    ) -> bool:
        """Writes or updates EVM summary metrics table in Google Sheets."""
        if self.gc is None:
            self.init_client()

        spreadsheet = self.gc.open_by_key(self.sheet_id)
        try:
            ws = spreadsheet.worksheet(worksheet_title)
        except Exception:
            ws = spreadsheet.add_worksheet(title=worksheet_title, rows=100, cols=10)

        tech = cutoff_metrics["technical"]
        fin = cutoff_metrics["financial"]

        rows = [
            ["Métrica EVM", "Dimensión Técnica (Horas)", "Dimensión Financiera (MXN)"],
            ["BAC (Presupuesto Conclusión)", tech["BAC"], fin["BAC"]],
            ["PV (Valor Planificado)", tech["PV"], fin["PV"]],
            ["EV (Valor Ganado)", tech["EV"], fin["EV"]],
            ["AC (Costo/Esfuerzo Real)", tech["AC"], fin["AC"]],
            ["CV (Variación Coste)", tech["CV"], fin["CV"]],
            ["SV (Variación Cronograma)", tech["SV"], fin["SV"]],
            ["CPI (Índice Coste)", tech["CPI"], fin["CPI"]],
            ["SPI (Índice Plazo)", tech["SPI"], fin["SPI"]],
            ["CSI (Índice Coste-Plazo)", tech["CSI"], fin["CSI"]],
            ["EAC Típico (Lineal PMBOK)", tech["EAC_typical"], fin["EAC_typical"]],
            ["EAC ML (Ridge L2)", tech["EAC_ml"], fin["EAC_ml"]],
            ["VAC (Ahorro Previsto)", tech["VAC"], fin["VAC"]],
        ]

        ws.clear()
        ws.update("A1", rows)
        return True
