"""Testes unitários para src/summarize_results.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.summarize_results import (
    HEADER_AVISO,
    _summarize_e1,
    _summarize_e2,
    _summarize_e3,
    _summarize_e4,
    _summarize_e5,
    summarize_r10_results,
)

# ---------------------------------------------------------------------------
# Constantes de caminho (absolutas para compatibilidade com pytest de qualquer cwd)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_REPORTS_R10 = _REPO_ROOT / "reports" / "r10"
_BASELINE_CSV = _REPO_ROOT / "reports" / "baseline_results.csv"


# ---------------------------------------------------------------------------
# test_retorna_string
# ---------------------------------------------------------------------------


def test_retorna_string(tmp_path: Path) -> None:
    """Verifica que a função retorna uma string não vazia dado um diretório vazio."""
    # Diretório existente mas sem CSVs — todos os DataFrames serão vazios,
    # mas a função deve completar sem erro e retornar string.
    resultado = summarize_r10_results(results_dir=str(tmp_path))
    assert isinstance(resultado, str), "O retorno deve ser do tipo str"
    assert len(resultado) > 0, "O retorno não pode ser string vazia"


# ---------------------------------------------------------------------------
# test_contem_status_em_analise
# ---------------------------------------------------------------------------


def test_contem_status_final(tmp_path: Path) -> None:
    """Verifica que o relatório contém marcação de resultado final consolidado."""
    resultado = summarize_r10_results(results_dir=str(tmp_path))
    assert "Relatório final" in resultado, "O relatório deve mencionar status final"
    assert "reports/final" in resultado, "O relatório deve mencionar a origem consolidada"


def test_header_exato(tmp_path: Path) -> None:
    """Verifica que o cabeçalho de aviso aparece literalmente no relatório."""
    resultado = summarize_r10_results(results_dir=str(tmp_path))
    assert HEADER_AVISO in resultado, (
        f"O cabeçalho exato não foi encontrado. Esperado:\n{HEADER_AVISO}"
    )


# ---------------------------------------------------------------------------
# test_com_csvs_reais
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _REPORTS_R10.exists(),
    reason="Diretório reports/r10 não encontrado no disco — pulando teste de integração",
)
def test_com_csvs_reais() -> None:
    """Se os arquivos reais existem, executa sem erro e retorna string não vazia."""
    baseline = str(_BASELINE_CSV) if _BASELINE_CSV.exists() else None
    resultado = summarize_r10_results(
        results_dir=str(_REPORTS_R10),
        baseline_csv=baseline,
    )
    assert isinstance(resultado, str)
    assert len(resultado) > 100, "Relatório com dados reais deve ter mais de 100 caracteres"
    # Deve conter seções de todos os experimentos
    for secao in ["E1", "E2", "E3", "E4", "E5", "Síntese"]:
        assert secao in resultado, f"Seção '{secao}' não encontrada no relatório"


# ---------------------------------------------------------------------------
# test_mock_e1
# ---------------------------------------------------------------------------


def _make_baseline_df() -> pd.DataFrame:
    """Cria um DataFrame mockado pequeno com estrutura de baseline_results.csv.

    Retorna
    -------
    pd.DataFrame
        DataFrame com 4 linhas: 2 classificadores × 2 tipos de balanceamento,
        todos para N=10.
    """
    return pd.DataFrame(
        {
            "experiment_id": [
                "baseline_full_sem_LinearSVC_10msgs_20260101",
                "baseline_full_sem_NB_10msgs_20260101",
                "baseline_full_und_LinearSVC_10msgs_20260101",
                "baseline_full_und_NB_10msgs_20260101",
            ],
            "experiment_name": ["baseline_full"] * 4,
            "experiment_type": [
                "sem_balanceamento",
                "sem_balanceamento",
                "undersampling",
                "undersampling",
            ],
            "n_msgs": [10, 10, 10, 10],
            "classifier": ["LinearSVC", "MultinomialNB", "LinearSVC", "MultinomialNB"],
            "f05_mean": [0.9070, 0.8878, 0.6362, 0.6515],
            "f05_std": [0.01, 0.02, 0.03, 0.02],
            "f1_mean": [0.89, 0.87, 0.78, 0.79],
            "precision_mean": [0.9174, 0.9680, 0.5847, 0.6021],
            "recall_mean": [0.8687, 0.6682, 0.9836, 0.9707],
            "n_total": [66927, 66927, 66927, 66927],
            "n_predatory": [4946, 4946, 4946, 4946],
            "n_normal": [61981, 61981, 61981, 61981],
            "corpus": ["pan2012"] * 4,
            "n_conversations": [500, 500, 500, 500],
        }
    )


def test_mock_e1_retorna_string_nao_vazia() -> None:
    """Testa parsing do baseline_results.csv com DataFrame mockado pequeno."""
    df_mock = _make_baseline_df()
    resultado = _summarize_e1(df_mock)
    assert isinstance(resultado, str)
    assert len(resultado) > 0


def test_mock_e1_contem_classificadores() -> None:
    """Verifica que o resultado do E1 mockado menciona os classificadores esperados."""
    df_mock = _make_baseline_df()
    resultado = _summarize_e1(df_mock)
    assert "LinearSVC" in resultado, "LinearSVC deve aparecer na seção E1"
    assert "MultinomialNB" in resultado, "MultinomialNB deve aparecer na seção E1"


def test_mock_e1_contem_tipos_balanceamento() -> None:
    """Verifica que ambos os tipos de balanceamento aparecem no relatório E1."""
    df_mock = _make_baseline_df()
    resultado = _summarize_e1(df_mock)
    assert "Sem subamostragem" in resultado or "sem_balanceamento" in resultado or "Sem" in resultado
    assert "undersampling" in resultado.lower() or "Undersampling" in resultado or "Com" in resultado


def test_mock_e1_sem_baseline_full_retorna_aviso() -> None:
    """Se não houver linhas baseline_full, a função deve retornar aviso adequado."""
    df_vazio = _make_baseline_df().copy()
    df_vazio["experiment_name"] = "outra_corrida"  # nenhuma linha é baseline_full
    resultado = _summarize_e1(df_vazio)
    assert "Sem dados" in resultado or "baseline_full" in resultado or "disponíveis" in resultado


# ---------------------------------------------------------------------------
# Testes dos helpers de outros experimentos com mocks mínimos
# ---------------------------------------------------------------------------


def test_mock_e2_vazio() -> None:
    """Verifica que _summarize_e2 lida com DataFrame vazio sem erro."""
    resultado = _summarize_e2(pd.DataFrame())
    assert "E2" in resultado
    assert "disponíveis" in resultado or "Sem dados" in resultado


def test_mock_e3_vazio() -> None:
    """Verifica que _summarize_e3 lida com DataFrame vazio sem erro."""
    resultado = _summarize_e3(pd.DataFrame())
    assert "E3" in resultado


def test_mock_e4_vazio() -> None:
    """Verifica que _summarize_e4 lida com DataFrame vazio sem erro."""
    resultado = _summarize_e4(pd.DataFrame())
    assert "E4" in resultado or "Jaccard" in resultado


def test_mock_e5_vazio() -> None:
    """Verifica que _summarize_e5 lida com DataFrame vazio sem erro."""
    resultado = _summarize_e5(pd.DataFrame())
    assert "E5" in resultado


def test_diretorio_inexistente_levanta_erro() -> None:
    """Verifica que FileNotFoundError é levantado se o diretório não existir."""
    with pytest.raises(FileNotFoundError):
        summarize_r10_results(results_dir="/caminho/que/nao/existe/r10")
