"""
Unit tests for pure NumPy NLP classifier, sublinear TF-IDF vectorizer,
Covey quadrant categorization, and scope creep blocking.
"""

import pytest
from lifeos.nlp_classifier import (
    LifeOSNLPClassifier,
    NumpyTfidfSoftmaxClassifier,
    SublinearTfidfVectorizer,
)


def test_vectorizer_properties():
    vec = SublinearTfidfVectorizer(min_df=1)
    docs = [
        "Pagar tarjeta BBVA Oro antes del corte",
        "Entrenar clasificador en Colab",
    ]
    X = vec.fit_transform(docs)
    assert X.shape[0] == 2
    assert X.shape[1] > 0
    # L2 normalized rows should have unit norm
    assert abs(float(sum(X[0] ** 2)) - 1.0) < 1e-4
    assert abs(float(sum(X[1] ** 2)) - 1.0) < 1e-4


def test_nlp_sample_classification():
    clf = LifeOSNLPClassifier()
    clf.train_on_corpus()

    # Sample 1 from PGP Section 4
    sample_financial = "Apartar $4,208 MXN para liquidar pago para no generar intereses de AMEX antes del día 9"
    res1 = clf.classify(sample_financial)
    assert res1["covey_quadrant"] == "Q1"
    assert res1["is_blocked"] is False
    assert res1["covey_confidence"] > 0.6

    # Sample 2 from PGP Section 4
    sample_ml = "Entrenar clasificador DistilRoBERTa y documentar curva S en Colab"
    res2 = clf.classify(sample_ml)
    assert res2["covey_quadrant"] == "Q2"
    assert res2["is_blocked"] is False
    assert res2["covey_confidence"] > 0.6


def test_scope_creep_and_gold_plating_blocking():
    clf = LifeOSNLPClassifier()
    clf.train_on_corpus()

    # Distractor / Gold Plating activity
    sample_gold_plating = "Rediseñar paleta de colores CSS y sombras de botones del dashboard por estética"
    res = clf.classify(sample_gold_plating)

    assert res["covey_quadrant"] == "Q4"
    assert res["is_blocked"] is True
    assert "BLOQUEO DIRECTO PGP" in res["block_reason"]


def test_nlp_stratified_cv_performance():
    clf = LifeOSNLPClassifier()
    cv_res = clf.evaluate_stratified_cv(k_folds=4)

    # Win condition: Accuracy > 90% (target > 95%) and Macro F1 > 0.85
    assert cv_res["mean_cv_accuracy"] >= 0.90
    assert cv_res["mean_macro_f1"] >= 0.85
