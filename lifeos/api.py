"""
FastAPI REST Service for LifeOS-ML (WBS 4.1).
Exposes endpoints for CPM/PERT scheduling, NLP classification, EVM forecasting,
credit cards cashflow management, and Forge3D batch quoting.
"""

from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lifeos.config import (
    PROJECT_AUTHOR,
    PROJECT_CODE,
    PROJECT_NAME,
    RISK_CATALOG,
    WBS_CATALOG,
)
from lifeos.cpm_pert import CPMNetwork
from lifeos.dynamic_config import dynamic_config
from lifeos.evm_predictor import EVMTracker
from lifeos.finance_manager import FinanceManager
from lifeos.forge3d import Forge3DEngine
from lifeos.nlp_classifier import LifeOSNLPClassifier

app = FastAPI(
    title="LifeOS-ML Cognitive Project Management & ML API",
    description="Cognitive PMBOK 7 + Scrum/Kanban + EVM Machine Learning API",
    version="1.0.0",
)

# Shared singleton instances
cpm_engine = CPMNetwork()
nlp_engine = LifeOSNLPClassifier()
evm_tracker = EVMTracker()
finance_mgr = FinanceManager(config=dynamic_config)
forge3d_engine = Forge3DEngine()


# --- Pydantic Request/Response Models ---
class UpdateCardRequest(BaseModel):
    balance: Optional[float] = None
    no_interest_payment: Optional[float] = None
    minimum_payment: Optional[float] = None
    due_day: Optional[int] = None


class UpdateBaselineRequest(BaseModel):
    income: Optional[float] = None
    bac_opex: Optional[float] = None
    discretionary_cap: Optional[float] = None


class NLPClassifyRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "Apartar $4,208 MXN para liquidar pago para no generar intereses de AMEX antes del día 9"})



class MonteCarloRequest(BaseModel):
    iterations: int = Field(default=10000, ge=100, le=50000)
    target_hours: float = Field(default=255.0, gt=0)


class ExpenseRequest(BaseModel):
    description: str
    amount: float
    category: str = "discretionary"
    day_of_month: int = 1
    wbs_id: Optional[str] = None


class Forge3DQuoteRequest(BaseModel):
    job_name: str
    grams_per_unit: float
    hours_per_unit: float
    units: int = 1
    material: str = "PLA"
    post_processing_minutes: float = 15.0
    margin_percent: float = 40.0


# --- Endpoints ---
@app.get("/")
def get_root():
    return {
        "project_code": PROJECT_CODE,
        "project_name": PROJECT_NAME,
        "author": PROJECT_AUTHOR,
        "status": "OPERATIONAL",
        "endpoints": [
            "/cpm/schedule",
            "/cpm/monte-carlo",
            "/nlp/classify",
            "/nlp/evaluate-cv",
            "/evm/metrics",
            "/finance/schedule",
            "/finance/expense",
            "/forge3d/quote",
            "/risks",
        ],
    }


@app.get("/cpm/schedule")
def get_cpm_schedule():
    cpm_engine.compute_schedule()
    summary = cpm_engine.get_summary_table()
    stats = cpm_engine.get_pert_statistics()
    return {
        "project_duration_hours": cpm_engine.project_duration,
        "critical_path": cpm_engine.critical_path,
        "pert_statistics": stats,
        "wbs_packages": summary,
    }


@app.post("/cpm/monte-carlo")
def run_monte_carlo(req: MonteCarloRequest):
    result = cpm_engine.run_monte_carlo_simulation(
        iterations=req.iterations,
        target_hours=req.target_hours,
    )
    # Exclude large raw arrays from response
    return {
        "iterations": result["iterations"],
        "mean_hours": result["mean_hours"],
        "std_hours": result["std_hours"],
        "min_hours": result["min_hours"],
        "max_hours": result["max_hours"],
        "p50_hours": result["p50_hours"],
        "p85_hours": result["p85_hours"],
        "p90_hours": result["p90_hours"],
        "p95_hours": result["p95_hours"],
        "prob_under_threshold": result["prob_under_threshold"],
        "threshold_hours": result["threshold_hours"],
    }


@app.post("/nlp/classify")
def classify_text(req: NLPClassifyRequest):
    return nlp_engine.classify(req.text)


@app.get("/nlp/evaluate-cv")
def evaluate_nlp_cv():
    return nlp_engine.evaluate_stratified_cv()


@app.get("/evm/metrics")
def get_evm_metrics():
    metrics = evm_tracker.compute_cutoff_metrics()
    forecast = evm_tracker.predict_future_sprints_ml()
    return {
        "cutoff_sprint": 3,
        "cutoff_metrics": metrics,
        "ml_forecast": {
            "eac_ml_hours": forecast["eac_ml_hours"],
            "eac_ml_cost": forecast["eac_ml_cost"],
            "sprints": forecast["sprints"],
            "pv_hours": forecast["pv_hours"],
            "ev_hours": forecast["ev_hours"],
            "ac_hours_actual": forecast["ac_hours_actual"],
            "ac_hours_ml": forecast["ac_hours_ml"],
        },
    }


@app.get("/finance/schedule")
def get_finance_schedule():
    summary = finance_mgr.get_portfolio_summary()
    schedule = finance_mgr.get_biweekly_payment_schedule()
    return {
        "summary": summary,
        "payment_schedule": schedule,
    }


@app.post("/finance/expense")
def register_expense(req: ExpenseRequest):
    res = finance_mgr.register_expense(
        description=req.description,
        amount=req.amount,
        category=req.category,
        day_of_month=req.day_of_month,
        wbs_id=req.wbs_id,
    )
    return {
        "status": "REGISTERED",
        "details": {
            "description": res["registered_expense"].description,
            "amount": res["registered_expense"].amount,
            "category": res["registered_expense"].category,
            "day_of_month": res["registered_expense"].day_of_month,
        },
        "daily_discretionary_total": res["daily_discretionary_total"],
        "exceeds_cap": res["exceeds_cap"],
        "alert": res["alert"],
    }


@app.post("/forge3d/quote")
def quote_3d_batch(req: Forge3DQuoteRequest):
    quote = forge3d_engine.calculate_quote(
        job_name=req.job_name,
        grams_per_unit=req.grams_per_unit,
        hours_per_unit=req.hours_per_unit,
        units=req.units,
        material=req.material,
        post_processing_minutes=req.post_processing_minutes,
        margin_percent=req.margin_percent,
    )
    return quote


@app.get("/risks")
def get_risk_register():
    return [
        {
            "risk_id": r.risk_id,
            "description": r.description,
            "category": r.category,
            "probability": r.probability,
            "impact": r.impact,
            "priority": r.priority,
            "response_strategy": r.response_strategy,
            "contingency_plan": r.contingency_plan,
        }
        for r in RISK_CATALOG
    ]


# --- Dynamic Configuration & Real-Time Sync Endpoints ---
@app.post("/config/sync-sheets")
def sync_config_from_sheets():
    """Triggers live synchronization with user's Google Sheets."""
    try:
        success = dynamic_config.sync_from_google_sheets()
        return {
            "status": "SUCCESS" if success else "FAILED",
            "last_sync": dynamic_config.last_sync,
            "sync_source": dynamic_config.sync_source,
            "summary": dynamic_config.get_summary(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Sheets sync failed: {str(e)}")


@app.get("/config/live")
def get_live_config():
    """Returns the current dynamic financial and operational baseline."""
    return dynamic_config.get_summary()


@app.get("/config/cards")
def get_live_cards():
    """Returns all current credit cards with dynamic balances and deadlines."""
    return [
        {
            "name": c.name,
            "due_day": c.due_day,
            "cutoff_day": c.cutoff_day,
            "balance": c.balance,
            "no_interest_payment": c.no_interest_payment,
            "minimum_payment": c.minimum_payment,
            "cat_rate": c.cat_rate,
        }
        for c in dynamic_config.cards
    ]


@app.put("/config/cards/{card_name}")
def update_live_card(card_name: str, req: UpdateCardRequest):
    """Updates or inserts a credit card's balance/deadline dynamically."""
    updated = dynamic_config.update_card(
        name=card_name,
        balance=req.balance,
        no_interest_payment=req.no_interest_payment,
        minimum_payment=req.minimum_payment,
        due_day=req.due_day,
    )
    return {
        "status": "UPDATED",
        "card_name": card_name,
        "summary": dynamic_config.get_summary(),
    }


@app.put("/config/baseline")
def update_live_baseline(req: UpdateBaselineRequest):
    """Updates biweekly income, OpEx BAC, or daily discretionary cap dynamically."""
    dynamic_config.set_financial_baseline(
        income=req.income,
        bac_opex=req.bac_opex,
        discretionary_cap=req.discretionary_cap,
    )
    return {
        "status": "UPDATED",
        "summary": dynamic_config.get_summary(),
    }

