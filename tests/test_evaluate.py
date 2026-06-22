"""
Testes para src/evaluate.py — E3 (LOO) e E4 (Jaccard).

_get_db é monkeypatchado para (None, False) em todos os testes:
evita timeout de 10s e não requer MongoDB.
"""
from __future__ import annotations

import pandas as pd
import pytest

import src.evaluate as ev


@pytest.fixture(autouse=True)
def mock_get_db(monkeypatch):
    """Desabilita conexão MongoDB em todos os testes deste módulo."""
    monkeypatch.setattr(ev, "_get_db", lambda: (None, False))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth_df(n_per_class: int = 10, n_msgs: int = 10) -> pd.DataFrame:
    """
    DataFrame sintético com formato esperado por evaluate.py.
    Colunas: [conversation_id, n_msgs, text, label]
    """
    ids = [f"SYN_{i}" for i in range(n_per_class * 2)]
    texts = (
        [f"grooming victim trust isolate control {i}" for i in range(n_per_class)]
        + [f"homework school friends game {i}" for i in range(n_per_class)]
    )
    labels = [1] * n_per_class + [0] * n_per_class
    return pd.DataFrame({
        "conversation_id": ids,
        "n_msgs": n_msgs,
        "text": texts,
        "label": labels,
    })


# ---------------------------------------------------------------------------
# experiment_e3_leave_one_out
# ---------------------------------------------------------------------------

def test_e3_retorna_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(ev, "N_MSGS_LIST", [10])
    monkeypatch.setattr(ev, "BOOTSTRAP_N", 10)  # rápido
    df_input = _synth_df(n_per_class=8, n_msgs=10)
    result = ev.experiment_e3_leave_one_out(df_input)
    import pandas as pd
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) >= {"n_msgs", "f05", "ic_lower", "ic_upper"}


def test_e3_metricas_no_intervalo(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(ev, "N_MSGS_LIST", [10])
    monkeypatch.setattr(ev, "BOOTSTRAP_N", 10)
    df_input = _synth_df(n_per_class=8, n_msgs=10)
    result = ev.experiment_e3_leave_one_out(df_input)
    if len(result) > 0:
        assert (result["f05"] >= 0.0).all()
        assert (result["f05"] <= 1.0).all()
        assert (result["ic_lower"] <= result["ic_upper"]).all()


def test_e3_corpus_insuficiente_retorna_df_vazio(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(ev, "N_MSGS_LIST", [5])
    monkeypatch.setattr(ev, "BOOTSTRAP_N", 10)
    # n_msgs=5 mas dados com n_msgs=10 → nenhum subset → DataFrame vazio
    df_input = _synth_df(n_per_class=4, n_msgs=10)
    result = ev.experiment_e3_leave_one_out(df_input)
    import pandas as pd
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# experiment_e4_jaccard
# ---------------------------------------------------------------------------

def test_e4_retorna_float_e_lista(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(ev, "N_MSGS_LIST", [10])
    pan_df = _synth_df(n_per_class=10, n_msgs=10)
    synth_df = _synth_df(n_per_class=10, n_msgs=10)
    jaccard, common = ev.experiment_e4_jaccard(pan_df, synth_df, top_n=5)
    assert 0.0 <= jaccard <= 1.0
    assert isinstance(common, list)


def test_e4_corpora_identicos_jaccard_um(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(ev, "N_MSGS_LIST", [10])
    df = _synth_df(n_per_class=10, n_msgs=10)
    jaccard, common = ev.experiment_e4_jaccard(df, df, top_n=5)
    assert jaccard == pytest.approx(1.0)


def test_e4_corpora_distintos_jaccard_baixo(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(ev, "N_MSGS_LIST", [10])
    pan_df = _synth_df(n_per_class=10, n_msgs=10)
    # corpus sintético com vocabulário completamente diferente
    synth_texts = [f"xyz abc def ghi jkl {i}" for i in range(20)]
    synth_df = pan_df.copy()
    synth_df["text"] = synth_texts
    jaccard, _ = ev.experiment_e4_jaccard(pan_df, synth_df, top_n=5)
    assert jaccard < 0.5
