"""
Testes para src/mining.py — extract_discriminative_features, save_mining_csv.

Usa mini-corpus sintético com termos plantados (não requer PAN/OSAEBA).
Verificação por pertinência (termo plantado aparece no top-N), não por
coeficiente exato (evita dependência de versão do sklearn).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.mining import extract_discriminative_features, save_mining_csv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mini_corpus():
    predatory = [
        "grooming victim trust isolate send photo",
        "make friend special secret keep between us",
        "nobody understand you like i do meet me",
        "send picture private just for me trust me",
        "you are special secret between us only us",
    ] * 6  # 30 amostras
    normal = [
        "homework school friends game tonight",
        "teacher class project deadline tomorrow",
        "football match weekend friends family",
        "birthday party friends everyone invited",
        "study group library exam preparation",
    ] * 6
    return predatory, normal


# ---------------------------------------------------------------------------
# extract_discriminative_features
# ---------------------------------------------------------------------------

def test_retorna_dataframe(mini_corpus):
    pred, norm = mini_corpus
    df = extract_discriminative_features(pred, norm, top_n=5)
    assert isinstance(df, pd.DataFrame)


def test_colunas_corretas(mini_corpus):
    pred, norm = mini_corpus
    df = extract_discriminative_features(pred, norm, top_n=5)
    assert set(df.columns) == {"termo", "coeficiente", "classe"}


def test_numero_de_linhas(mini_corpus):
    pred, norm = mini_corpus
    top_n = 5
    df = extract_discriminative_features(pred, norm, top_n=top_n)
    assert len(df) == top_n * 2


def test_classes_esperadas(mini_corpus):
    pred, norm = mini_corpus
    df = extract_discriminative_features(pred, norm, top_n=5)
    classes = set(df["classe"].unique())
    assert classes == {"predatorio", "normal"}


def test_termos_plantados_predatorio_aparecem_no_top(mini_corpus):
    pred, norm = mini_corpus
    df = extract_discriminative_features(pred, norm, top_n=10)
    termos_pred = set(df[df["classe"] == "predatorio"]["termo"].tolist())
    # "grooming" ou "victim" ou "secret" devem estar no top-10 predatório
    planted = {"grooming", "victim", "secret", "trust", "isolate"}
    assert planted & termos_pred, (
        f"Nenhum termo plantado encontrado no top predatório. Termos: {termos_pred}"
    )


def test_coeficientes_predatorio_positivos(mini_corpus):
    pred, norm = mini_corpus
    df = extract_discriminative_features(pred, norm, top_n=5)
    pred_coefs = df[df["classe"] == "predatorio"]["coeficiente"]
    assert (pred_coefs > 0).all(), "Coeficientes predatórios devem ser positivos"


def test_coeficientes_normal_negativos(mini_corpus):
    pred, norm = mini_corpus
    df = extract_discriminative_features(pred, norm, top_n=5)
    norm_coefs = df[df["classe"] == "normal"]["coeficiente"]
    assert (norm_coefs < 0).all(), "Coeficientes normais devem ser negativos"


def test_listas_vazias_levanta_erro():
    with pytest.raises(ValueError):
        extract_discriminative_features([], ["texto normal"], top_n=5)
    with pytest.raises(ValueError):
        extract_discriminative_features(["texto pred"], [], top_n=5)


# ---------------------------------------------------------------------------
# save_mining_csv
# ---------------------------------------------------------------------------

def test_save_cria_arquivo(mini_corpus, tmp_path):
    pred, norm = mini_corpus
    df = extract_discriminative_features(pred, norm, top_n=5)
    out = save_mining_csv(df, path=tmp_path / "test_mining.csv")
    assert out.exists()


def test_save_colunas_csv(mini_corpus, tmp_path):
    pred, norm = mini_corpus
    df = extract_discriminative_features(pred, norm, top_n=5)
    out = save_mining_csv(df, path=tmp_path / "mining.csv")
    loaded = pd.read_csv(out)
    assert set(loaded.columns) == {"termo", "coeficiente", "classe"}


def test_save_numero_de_linhas(mini_corpus, tmp_path):
    pred, norm = mini_corpus
    top_n = 7
    df = extract_discriminative_features(pred, norm, top_n=top_n)
    out = save_mining_csv(df, path=tmp_path / "mining.csv")
    loaded = pd.read_csv(out)
    assert len(loaded) == top_n * 2


def test_save_cria_diretorio_pai(mini_corpus, tmp_path):
    pred, norm = mini_corpus
    df = extract_discriminative_features(pred, norm, top_n=3)
    nested = tmp_path / "subdir" / "deep" / "mining.csv"
    out = save_mining_csv(df, path=nested)
    assert out.exists()
