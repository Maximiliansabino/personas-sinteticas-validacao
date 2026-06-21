"""
Testes para src/classifier.py — _fbeta, _run_cv.

Usa mini-corpus sintético. Não chama run_experiment() para não tocar
reports/baseline_results.csv (contrato com o KNIME).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.classifier import _fbeta, _run_cv, _make_pipeline_svm, _make_pipeline_nb


# ---------------------------------------------------------------------------
# _fbeta
# ---------------------------------------------------------------------------

def test_fbeta_valores_ideais():
    assert _fbeta(1.0, 1.0) == pytest.approx(1.0)


def test_fbeta_zero_divide_retorna_zero():
    assert _fbeta(0.0, 0.0) == 0.0


def test_fbeta_precisao_zero():
    assert _fbeta(0.0, 1.0) == 0.0


def test_fbeta_recall_zero():
    assert _fbeta(1.0, 0.0) == 0.0


def test_fbeta_prioriza_precisao():
    # F0.5 penaliza recall mais do que precisão
    f_alta_prec = _fbeta(0.9, 0.5)
    f_alta_rec = _fbeta(0.5, 0.9)
    assert f_alta_prec > f_alta_rec


def test_fbeta_exemplo_panzariello():
    # Referência: F0.5 com P=1.0, R=0.9 deve ser próximo de P (≈0.98)
    resultado = _fbeta(1.0, 0.9)
    assert resultado > 0.96


# ---------------------------------------------------------------------------
# _run_cv (mini-corpus sintético — não toca CSV do KNIME)
# ---------------------------------------------------------------------------

def _mini_corpus(n_per_class: int = 30) -> tuple[pd.Series, pd.Series]:
    """
    Corpus binário simples: textos predatórios vs normais com termos distintos.
    n_per_class=30 → 60 amostras, suficiente para CV=5 estratificada.
    """
    predatory = [f"grooming victim trust isolate {i}" for i in range(n_per_class)]
    normal = [f"homework school friends game {i}" for i in range(n_per_class)]
    texts = pd.Series(predatory + normal)
    labels = pd.Series([1] * n_per_class + [0] * n_per_class)
    return texts, labels


def test_run_cv_retorna_chaves_corretas():
    X, y = _mini_corpus()
    result = _run_cv(_make_pipeline_svm(), X, y)
    esperadas = {"f05_mean", "f05_std", "f1_mean", "precision_mean", "recall_mean"}
    assert set(result.keys()) == esperadas


def test_run_cv_valores_entre_0_e_1():
    X, y = _mini_corpus()
    result = _run_cv(_make_pipeline_svm(), X, y)
    for key, val in result.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} fora do intervalo [0,1]"


def test_run_cv_corpus_separavel_f05_alto():
    # Corpus claramente separável → F0.5 deve ser alto
    X, y = _mini_corpus(n_per_class=40)
    result = _run_cv(_make_pipeline_svm(), X, y)
    assert result["f05_mean"] > 0.7, f"F0.5 esperado alto, obtido: {result['f05_mean']:.4f}"


def test_run_cv_balanced_usa_undersampling():
    # Com balanced=True, deve ainda retornar métricas válidas
    X, y = _mini_corpus()
    result = _run_cv(_make_pipeline_svm(), X, y, balanced=True)
    assert "f05_mean" in result
    assert 0.0 <= result["f05_mean"] <= 1.0


def test_run_cv_nb_pipeline_funciona():
    X, y = _mini_corpus()
    result = _run_cv(_make_pipeline_nb(), X, y)
    assert "f05_mean" in result
    assert result["f05_mean"] >= 0.0
