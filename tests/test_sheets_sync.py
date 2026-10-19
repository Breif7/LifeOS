"""
Unit tests for Google Sheets parsing and synchronization logic.
"""

import pandas as pd
import pytest
from lifeos.sheets_sync import GoogleSheetsSync


def test_parse_cards_dataframe():
    data = {
        "Nombre Tarjeta": ["Platinum Santander", "BBVA Oro", "AMEX Verde", "Total General"],
        "Saldo al Corte": ["$6,200.00", "$8,900.50", "7,500.00", "$22,600.50"],
        "Pago para No Generar Intereses": ["1,850.00", "2,100.00", "2,200.00", "6,150.00"],
        "Pago Mínimo": ["520", "680", "600", "1800"],
        "Día Límite de Pago": [5, 9, 9, 0],
        "Día de Corte": [15, 19, 19, 0],
        "CAT Promedio": ["92.4%", "84.5%", "89.2%", "88.7%"],
    }
    df = pd.DataFrame(data)

    cards = GoogleSheetsSync.parse_cards_dataframe(df)

    # Total row should be automatically skipped
    assert len(cards) == 3

    assert cards[0].name == "Platinum Santander"
    assert cards[0].balance == 6200.0
    assert cards[0].no_interest_payment == 1850.0
    assert cards[0].due_day == 5
    assert cards[0].cat_rate == 92.4

    assert cards[1].name == "BBVA Oro"
    assert cards[1].due_day == 9
    assert cards[1].no_interest_payment == 2100.0

    assert cards[2].name == "AMEX Verde"
    assert cards[2].due_day == 9
