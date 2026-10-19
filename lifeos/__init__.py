"""
LifeOS-ML Package
PRJ-LIFEOS-ML-2026-Q4 — LifeOS-ML: Sprint, Finance & Study EVM Tracker
Author: Ing. Miguel Martínez Ortega
"""

from lifeos.config import (
    CREDIT_CARDS,
    RISK_CATALOG,
    WBS_CATALOG,
    WorkPackage,
)
from lifeos.cpm_pert import CPMNetwork
from lifeos.evm_predictor import EVMTracker, PolynomialRidgeRegressor
from lifeos.finance_manager import FinanceManager
from lifeos.forge3d import Forge3DEngine
from lifeos.nlp_classifier import (
    LifeOSNLPClassifier,
    NumpyTfidfSoftmaxClassifier,
    SublinearTfidfVectorizer,
)

__version__ = "1.0.0"
__all__ = [
    "CPMNetwork",
    "LifeOSNLPClassifier",
    "NumpyTfidfSoftmaxClassifier",
    "SublinearTfidfVectorizer",
    "EVMTracker",
    "PolynomialRidgeRegressor",
    "FinanceManager",
    "Forge3DEngine",
    "WBS_CATALOG",
    "CREDIT_CARDS",
    "RISK_CATALOG",
]
