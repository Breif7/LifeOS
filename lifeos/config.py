"""
LifeOS-ML Configuration & Project Baseline
PRJ-LIFEOS-ML-2026-Q4 — LifeOS-ML: Sprint, Finance & Study EVM Tracker
Author: Ing. Miguel Martínez Ortega
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --- Project Metadata ---
PROJECT_CODE = "PRJ-LIFEOS-ML-2026-Q4"
PROJECT_NAME = "LifeOS-ML: Sprint, Finance & Study EVM Tracker"
PROJECT_AUTHOR = "Ing. Miguel Martínez Ortega"
TOTAL_SPRINTS = 6
TOTAL_WEEKS = 12
WORK_DAYS = 60
BAC_HOURS = 240.0
WIP_LIMIT = 3

# --- Financial Baseline ---
BIWEEKLY_INCOME = 14691.00
BIWEEKLY_BAC_OPEX = 10900.00
TOTAL_BAC_FINANCIAL = 65400.00  # 6 * 10,900
BIWEEKLY_LIQUIDITY_BUFFER = 3791.00
BIWEEKLY_AMORTIZATION = 1000.00
BIWEEKLY_INVESTMENT = 500.00
DAILY_DISCRETIONARY_CAP = 300.00

# Total Debt Baseline
TOTAL_CREDIT_DEBT = 47238.00
MONTHLY_NO_INTEREST_PAYMENT = 11341.00
MONTHLY_MINIMUM_PAYMENT = 3518.00


@dataclass
class WorkPackage:
    wbs_id: str
    name: str
    sprint: str
    duration: float  # Deterministic hours
    bac_cost: float  # MXN
    predecessors: List[str]
    pert_o: float  # Optimistic hours
    pert_m: float  # Most likely hours
    pert_p: float  # Pessimistic hours
    es: float = 0.0
    ef: float = 0.0
    ls: float = 0.0
    lf: float = 0.0
    total_slack: float = 0.0
    is_critical: bool = False

    @property
    def pert_te(self) -> float:
        """Expected PERT duration: (O + 4M + P) / 6"""
        return round((self.pert_o + 4.0 * self.pert_m + self.pert_p) / 6.0, 2)

    @property
    def pert_variance(self) -> float:
        """PERT variance: ((P - O) / 6)^2"""
        return ((self.pert_p - self.pert_o) / 6.0) ** 2


# 16 Work Packages aligned with Table 2 of the Project Management Plan (PGP)
WBS_CATALOG: Dict[str, WorkPackage] = {
    "1.1": WorkPackage(
        wbs_id="1.1",
        name="Definición de URD, Win Conditions y Acta Kick-off",
        sprint="S1",
        duration=10.0,
        bac_cost=2500.0,
        predecessors=[],
        pert_o=8.5,
        pert_m=10.0,
        pert_p=13.5,
    ),
    "1.2": WorkPackage(
        wbs_id="1.2",
        name="Auditoría de Tarjetas y Línea Base Financiera",
        sprint="S1",
        duration=12.0,
        bac_cost=3000.0,
        predecessors=["1.1"],
        pert_o=10.5,
        pert_m=12.0,
        pert_p=15.5,
    ),
    "1.3": WorkPackage(
        wbs_id="1.3",
        name="Diseño del PGP Maestro y Matriz de Riesgos",
        sprint="S1",
        duration=14.0,
        bac_cost=3500.0,
        predecessors=["1.1"],
        pert_o=11.0,
        pert_m=14.0,
        pert_p=19.0,
    ),
    "2.1": WorkPackage(
        wbs_id="2.1",
        name="Pipeline ETL de Estados de Cuenta y Captura Quincenal",
        sprint="S2",
        duration=18.0,
        bac_cost=4800.0,
        predecessors=["1.2"],
        pert_o=15.5,
        pert_m=18.0,
        pert_p=22.5,
    ),
    "2.2": WorkPackage(
        wbs_id="2.2",
        name="Alertas de Pago Cero Intereses (CAT 0%) y Tope $300",
        sprint="S2",
        duration=14.0,
        bac_cost=3800.0,
        predecessors=["2.1"],
        pert_o=11.0,
        pert_m=14.0,
        pert_p=19.0,
    ),
    "2.3": WorkPackage(
        wbs_id="2.3",
        name="Sincronización Sprints con Maestría UNIR y Hábitos",
        sprint="S2",
        duration=10.0,
        bac_cost=2800.0,
        predecessors=["1.3"],
        pert_o=8.0,
        pert_m=10.0,
        pert_p=14.0,
    ),
    "3.1": WorkPackage(
        wbs_id="3.1",
        name="Curación de Corpus Bilingüe Finanzas y Tareas IA",
        sprint="S3",
        duration=16.0,
        bac_cost=4500.0,
        predecessors=["2.1", "2.3"],
        pert_o=13.5,
        pert_m=16.0,
        pert_p=20.5,
    ),
    "3.2": WorkPackage(
        wbs_id="3.2",
        name="Entrenamiento Clasificador NLP (TF-IDF + Softmax L2)",
        sprint="S3",
        duration=20.0,
        bac_cost=5600.0,
        predecessors=["3.1"],
        pert_o=17.5,
        pert_m=20.0,
        pert_p=26.5,
    ),
    "3.3": WorkPackage(
        wbs_id="3.3",
        name="Motor Predictivo Burn-Rate y EVM (Regresión Polinomial Ridge L2)",
        sprint="S4",
        duration=22.0,
        bac_cost=6200.0,
        predecessors=["2.2", "3.2"],
        pert_o=19.0,
        pert_m=22.0,
        pert_p=29.0,
    ),
    "4.1": WorkPackage(
        wbs_id="4.1",
        name="API REST FastAPI para Inferencia ML y Motor CPM",
        sprint="S4",
        duration=20.0,
        bac_cost=5800.0,
        predecessors=["3.2"],
        pert_o=17.0,
        pert_m=20.0,
        pert_p=25.0,
    ),
    "4.2": WorkPackage(
        wbs_id="4.2",
        name="Dashboard Control EVM (Curva S) y Tablero Kanban",
        sprint="S5",
        duration=18.0,
        bac_cost=5200.0,
        predecessors=["3.3", "4.1"],
        pert_o=15.5,
        pert_m=18.0,
        pert_p=22.5,
    ),
    "4.3": WorkPackage(
        wbs_id="4.3",
        name="Módulo Cotización y Planificación Lotes 3D (Forge3D)",
        sprint="S5",
        duration=16.0,
        bac_cost=4600.0,
        predecessors=["4.1"],
        pert_o=13.0,
        pert_m=16.0,
        pert_p=21.0,
    ),
    "4.4": WorkPackage(
        wbs_id="4.4",
        name="Pruebas Unitarias, Auditoría ISO 9000 y Validación",
        sprint="S5",
        duration=14.0,
        bac_cost=4100.0,
        predecessors=["4.2", "4.3"],
        pert_o=11.5,
        pert_m=14.0,
        pert_p=18.5,
    ),
    "5.1": WorkPackage(
        wbs_id="5.1",
        name="Publicación Repositorio GitHub, Colab y Docs",
        sprint="S6",
        duration=14.0,
        bac_cost=3600.0,
        predecessors=["4.4"],
        pert_o=12.0,
        pert_m=14.0,
        pert_p=17.0,
    ),
    "5.2": WorkPackage(
        wbs_id="5.2",
        name="Optimización CV ATS, Casos STAR y Postulaciones ML",
        sprint="S6",
        duration=14.0,
        bac_cost=3200.0,
        predecessors=["5.1"],
        pert_o=11.5,
        pert_m=14.0,
        pert_p=18.5,
    ),
    "5.3": WorkPackage(
        wbs_id="5.3",
        name="Cierre Administrativo, Evaluación Final y Lecciones",
        sprint="S6",
        duration=8.0,
        bac_cost=2000.0,
        predecessors=["5.2"],
        pert_o=6.5,
        pert_m=8.0,
        pert_p=11.5,
    ),
}

# --- Credit Card Master Catalog ---
@dataclass
class CreditCard:
    name: str
    due_day: int
    cutoff_day: int
    balance: float
    no_interest_payment: float
    minimum_payment: float
    cat_rate: float  # Annual Total Cost %


CREDIT_CARDS: List[CreditCard] = [
    CreditCard("Platinum", 5, 15, 6200.0, 1850.0, 520.0, 92.4),
    CreditCard("Didi", 7, 17, 3400.0, 950.0, 310.0, 115.0),
    CreditCard("BBVA Oro", 9, 19, 8900.0, 2100.0, 680.0, 84.5),
    CreditCard("AMEX", 9, 19, 7500.0, 2200.0, 600.0, 89.2),
    CreditCard("Nu", 16, 26, 4800.0, 1150.0, 360.0, 98.0),
    CreditCard("Mercado Pago", 17, 27, 3900.0, 820.0, 290.0, 108.5),
    CreditCard("Revolut", 19, 29, 2100.0, 450.0, 180.0, 75.0),
    CreditCard("Jun", 19, 29, 1838.0, 380.0, 150.0, 82.0),
    CreditCard("Plata", 22, 2, 4300.0, 791.0, 238.0, 96.0),
    CreditCard("LikeU", 26, 6, 4300.0, 650.0, 190.0, 79.8),
]

# --- Risk Register Catalog ---
@dataclass
class RiskItem:
    risk_id: str
    description: str
    category: str
    probability: int  # 1 (Muy Bajo) to 5 (Muy Alto)
    impact: int       # 1 to 5
    priority: str
    response_strategy: str
    contingency_plan: str


RISK_CATALOG: List[RiskItem] = [
    RiskItem(
        risk_id="R-01",
        description="Desfase en fecha límite de pago de tarjeta de crédito con cobro de intereses moratorios (CAT 80%-120%).",
        category="Financiero",
        probability=2,
        impact=5,
        priority="B (Alta)",
        response_strategy="Evitar / Reducir",
        contingency_plan="Separación automática del fondo de pago sin intereses el mismo día de dispersión de nómina y alerta Q1 5 días antes del corte.",
    ),
    RiskItem(
        risk_id="R-02",
        description="Sobrecarga académica en semanas de exámenes y entregas finales de UNIR (Sprint 5 · 48 h programadas).",
        category="Cronograma",
        probability=4,
        impact=4,
        priority="A (Muy Alta)",
        response_strategy="Reducir",
        contingency_plan="Reducción temporal del límite WIP Kanban a 2 tareas activas y consumo controlado de las 4 horas de holgura del paquete no crítico WBS 4.3.",
    ),
    RiskItem(
        risk_id="R-03",
        description="Corrupción del alcance (Scope Creep) o perfeccionamiento estético innecesario (Gold Plating) en el código.",
        category="Alcance",
        probability=3,
        impact=4,
        priority="B (Alta)",
        response_strategy="Evitar",
        contingency_plan="Filtrado automático mediante el clasificador NLP de Covey (bloqueo de tareas Q4) y requisito de solicitud de cambio formal.",
    ),
    RiskItem(
        risk_id="R-04",
        description="Acumulación de gastos hormiga en comida fuera u ocio que supere el tope diario de $300.00 MXN.",
        category="Costes",
        probability=3,
        impact=3,
        priority="C (Media)",
        response_strategy="Reducir",
        contingency_plan="Monitoreo del índice CPI financiero a mitad de quincena (día 7) y ajuste compensatorio inmediato en rubros variables.",
    ),
    RiskItem(
        risk_id="R-05",
        description="Paro no programado o atasco de filamento en impresora 3D Anycubic Kobra S1 durante lote de pedidos.",
        category="Operativo 3D",
        probability=2,
        impact=3,
        priority="C (Media)",
        response_strategy="Reducir / Retener",
        contingency_plan="Rutina de calibración preventiva cada 30 horas de impresión y stock de boquilla de repuesto en taller.",
    ),
    RiskItem(
        risk_id="R-06",
        description="Desviación en las predicciones del modelo ML por cambios abruptos en el patrón de gasto estacional (diciembre).",
        category="Calidad IA",
        probability=3,
        impact=3,
        priority="C (Media)",
        response_strategy="Reducir",
        contingency_plan="Reentrenamiento incremental al cierre de cada quincena combinando el estimador ML con la cota conservadora EAC = BAC / (CPI x SPI).",
    ),
]
