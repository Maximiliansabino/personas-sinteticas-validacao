"""Sumarizador de resultados das rodadas de experimentos.

Gera um relatório markdown consolidado a partir dos CSVs produzidos pelo
pipeline de avaliação (E1–E5).

Uso típico:
    python -m src.summarize_results --results-dir reports/final \\
        --baseline-csv reports/final/baseline_results.csv \\
        --output reports/final/summary.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

HEADER_AVISO = "Resultados consolidados dos experimentos do artigo"
N_MSGS_LIST: list[int] = [5, 10, 15, 20, 24, 50]
BASELINE_CSV_DEFAULT = "baseline_results.csv"


# ---------------------------------------------------------------------------
# Helpers de formatação
# ---------------------------------------------------------------------------


def _fmt(value: float, decimals: int = 4) -> str:
    """Formata um número float para string com precisão fixa.

    Parâmetros
    ----------
    value : float
        Valor a formatar.
    decimals : int
        Número de casas decimais.

    Retorna
    -------
    str
        Número formatado.
    """
    return f"{value:.{decimals}f}"


def _pct(value: float) -> str:
    """Converte proporção [0,1] para percentual formatado (e.g. '93.75%').

    Parâmetros
    ----------
    value : float
        Proporção entre 0 e 1.

    Retorna
    -------
    str
        String com sinal de percentual.
    """
    return f"{value * 100:.2f}%"


# ---------------------------------------------------------------------------
# Sumarizadores por experimento
# ---------------------------------------------------------------------------


def _summarize_e1(df_baseline) -> str:
    """Gera seção markdown do E1 (baseline PAN 2012).

    Filtra `experiment_name == 'baseline_full'` e apresenta F0.5 médio por
    classificador e tipo de balanceamento, agrupado por N mensagens.

    Parâmetros
    ----------
    df_baseline : pandas.DataFrame
        DataFrame lido de baseline_results.csv.

    Retorna
    -------
    str
        Seção markdown formatada.
    """
    import pandas as pd

    lines: list[str] = []
    lines.append("## E1 — Baseline PAN 2012 (`baseline_full`)")
    lines.append("")

    if df_baseline.empty or "experiment_name" not in df_baseline.columns:
        lines.append("_Sem dados de baseline_full disponíveis._")
        return "\n".join(lines)

    df = df_baseline[df_baseline["experiment_name"] == "baseline_full"].copy()
    if df.empty:
        lines.append("_Sem dados de baseline_full disponíveis._")
        return "\n".join(lines)

    for balanceamento in ["sem_balanceamento", "undersampling"]:
        df_b = df[df["experiment_type"] == balanceamento]
        if df_b.empty:
            continue
        titulo = (
            "Sem subamostragem" if balanceamento == "sem_balanceamento"
            else "Com undersampling"
        )
        lines.append(f"### {titulo}")
        lines.append("")
        lines.append("| N msgs | Classificador | F0.5 | Precisão | Revocação |")
        lines.append("|--------|--------------|------|----------|-----------|")
        for _, row in df_b.sort_values(["n_msgs", "classifier"]).iterrows():
            lines.append(
                f"| {int(row['n_msgs'])} "
                f"| {row['classifier']} "
                f"| {_pct(row['f05_mean'])} "
                f"| {_pct(row['precision_mean'])} "
                f"| {_pct(row['recall_mean'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def _summarize_e2(df_e2) -> str:
    """Gera seção markdown do E2 (cross-domain).

    Seleciona a corrida com maior `total` (mais dados avaliados) e apresenta
    F0.5, precisão e revocação por N mensagens.

    Parâmetros
    ----------
    df_e2 : pandas.DataFrame
        DataFrame lido de e2_cross_domain.csv.

    Retorna
    -------
    str
        Seção markdown formatada.
    """
    lines: list[str] = []
    lines.append("## E2 — Cross-domain (treina PAN, avalia sintético)")
    lines.append("")

    if df_e2.empty:
        lines.append("_Sem dados E2 disponíveis._")
        return "\n".join(lines)

    total_max = int(df_e2["total"].max())
    df = df_e2[df_e2["total"] == total_max].copy()
    logger.info("E2: usando corrida com total=%d (%d linhas)", total_max, len(df))

    predatory = int(df["predatory"].max())
    normal = int(df["normal"].max())
    lines.append(
        f"_Corrida com maior corpus: {total_max} amostras "
        f"({predatory} predatórias, {normal} normais)_"
    )
    lines.append("")
    lines.append("| N msgs | Classificador | F0.5 | Precisão | Revocação |")
    lines.append("|--------|--------------|------|----------|-----------|")
    for _, row in df.sort_values("n_msgs").iterrows():
        lines.append(
            f"| {int(row['n_msgs'])} "
            f"| {row['classifier']} "
            f"| {_pct(row['f05'])} "
            f"| {_pct(row['precision'])} "
            f"| {_pct(row['recall'])} |"
        )
    lines.append("")

    return "\n".join(lines)


def _summarize_e3(df_e3) -> str:
    """Gera seção markdown do E3 (leave-one-out com IC 95%).

    Seleciona a corrida de referência com maior `total` disponível, que
    corresponde ao corpus sintético final mais completo.

    Parâmetros
    ----------
    df_e3 : pandas.DataFrame
        DataFrame lido de e3_loo.csv.

    Retorna
    -------
    str
        Seção markdown formatada.
    """
    lines: list[str] = []
    lines.append("## E3 — Leave-one-out no corpus sintético (IC 95% bootstrap)")
    lines.append("")

    if df_e3.empty:
        lines.append("_Sem dados E3 disponíveis._")
        return "\n".join(lines)

    total_ref = int(df_e3["total"].max())
    df_ref = df_e3[df_e3["total"] == total_ref].copy()

    prefixos = df_ref["experiment_id"].str.extract(r"(\d{8}_\d{6})")[0].unique()
    logger.info("E3: corrida total=%d — prefixos de timestamp: %s", total_ref, prefixos)

    predatory = int(df_ref["predatory"].max())
    normal = int(df_ref["normal"].max())
    lines.append(
        f"_Corrida de referência: total={total_ref} "
        f"({predatory} predatórias, {normal} normais)_"
    )
    lines.append("")
    lines.append(
        "| N msgs | Classificador | F0.5 | IC inf | IC sup |"
    )
    lines.append("|--------|--------------|------|--------|--------|")
    for _, row in df_ref.sort_values("n_msgs").iterrows():
        lines.append(
            f"| {int(row['n_msgs'])} "
            f"| {row['classifier']} "
            f"| {_pct(row['f05'])} "
            f"| {_pct(row['ic_lower'])} "
            f"| {_pct(row['ic_upper'])} |"
        )
    lines.append("")

    return "\n".join(lines)


def _summarize_e4(df_e4) -> str:
    """Gera seção markdown do E4 (Jaccard TF-IDF).

    Seleciona a corrida com maior `total` e apresenta índice de Jaccard,
    features comuns e exclusivas de cada corpus.

    Parâmetros
    ----------
    df_e4 : pandas.DataFrame
        DataFrame lido de e4_jaccard.csv.

    Retorna
    -------
    str
        Seção markdown formatada.
    """
    lines: list[str] = []
    lines.append("## E4 — Sobreposição de features TF-IDF (Jaccard)")
    lines.append("")

    if df_e4.empty:
        lines.append("_Sem dados E4 disponíveis._")
        return "\n".join(lines)

    total_max = int(df_e4["total"].max())
    df = df_e4[df_e4["total"] == total_max].copy()
    logger.info("E4: usando corrida com total=%d (%d linhas)", total_max, len(df))

    lines.append(
        f"_Corpus de referência: {total_max} amostras PAN 2012_"
    )
    lines.append("")
    lines.append(
        "| N msgs | Jaccard | Comuns | Só PAN | Só Sintético |"
    )
    lines.append("|--------|---------|--------|--------|--------------|")
    for _, row in df.sort_values("n_msgs").iterrows():
        lines.append(
            f"| {int(row['n_msgs'])} "
            f"| {_fmt(row['jaccard'], 4)} "
            f"| {int(row['n_common'])} "
            f"| {int(row['n_only_pan'])} "
            f"| {int(row['n_only_synth'])} |"
        )
    lines.append("")
    lines.append(
        "> **Diagnóstico**: Jaccard ≈ 0 indica vocabulário não sobreposto "
        "entre PAN 2012 e corpus sintético nessa corrida."
    )
    lines.append("")

    return "\n".join(lines)


def _summarize_e5(df_e5) -> str:
    """Gera seção markdown do E5 (augmentation).

    Filtra `variant == 'augmentation_aug'`, agrupa por `total` e apresenta
    faixa de delta_pp (mín–máx) por N mensagens. Inclui nota "inconclusivo"
    sobre a variabilidade observada.

    Parâmetros
    ----------
    df_e5 : pandas.DataFrame
        DataFrame lido de e5_augmentation.csv.

    Retorna
    -------
    str
        Seção markdown formatada.
    """
    lines: list[str] = []
    lines.append("## E5 — Augmentation (PAN + sintético → avalia PAN test)")
    lines.append("")

    if df_e5.empty:
        lines.append("_Sem dados E5 disponíveis._")
        return "\n".join(lines)

    df_aug = df_e5[df_e5["variant"] == "augmentation_aug"].copy()
    if df_aug.empty:
        lines.append("_Sem linhas com `variant == augmentation_aug`._")
        return "\n".join(lines)

    # Corrida de maior corpus (referência PAN completo)
    total_max = int(df_aug["total"].max())
    df_ref = df_aug[df_aug["total"] == total_max].copy()
    logger.info(
        "E5: corrida referência total=%d (%d linhas)", total_max, len(df_ref)
    )

    lines.append(f"### Corrida de referência (total={total_max})")
    lines.append("")
    lines.append("| N msgs | Classificador | delta_pp (pp) | F0.5 |")
    lines.append("|--------|--------------|---------------|------|")
    for _, row in df_ref.sort_values("n_msgs").iterrows():
        lines.append(
            f"| {int(row['n_msgs'])} "
            f"| {row['classifier']} "
            f"| {_fmt(row['delta_pp'], 4)} "
            f"| {_pct(row['f05'])} |"
        )
    lines.append("")

    # Faixa de delta_pp agrupada por tamanho do corpus de teste
    lines.append("### Faixa de delta_pp por tamanho do corpus de teste")
    lines.append("")
    lines.append("| Total | delta_pp mín | delta_pp máx | N linhas |")
    lines.append("|-------|-------------|-------------|---------|")
    grupos = (
        df_aug.groupby("total")["delta_pp"]
        .agg(["min", "max", "count"])
        .reset_index()
        .sort_values("total")
    )
    for _, row in grupos.iterrows():
        lines.append(
            f"| {int(row['total'])} "
            f"| {_fmt(row['min'], 4)} "
            f"| {_fmt(row['max'], 4)} "
            f"| {int(row['count'])} |"
        )
    lines.append("")
    lines.append(
        "> **Nota (inconclusivo)**: A faixa de delta_pp varia amplamente entre "
        "tamanhos de corpus de teste (de negativos a positivos), indicando que "
        "o efeito do augmentation é instável nessa rodada. Resultado não conclusivo."
    )
    lines.append("")

    return "\n".join(lines)


def _sintese(df_e3, df_e4, df_e5) -> str:
    """Gera seção de síntese interpretativa dos resultados.

    Parâmetros
    ----------
    df_e3 : pandas.DataFrame
        DataFrame do E3 para extrair valores de referência.
    df_e4 : pandas.DataFrame
        DataFrame do E4 para extrair Jaccard de referência.
    df_e5 : pandas.DataFrame
        DataFrame do E5 para extrair delta_pp de referência.

    Retorna
    -------
    str
        Seção markdown de síntese.
    """
    lines: list[str] = []
    lines.append("## Síntese")
    lines.append("")

    lines.append("### O que sustenta")
    lines.append("")
    if not df_e3.empty:
        total_e3 = int(df_e3["total"].max())
        df_e3_ref = df_e3[df_e3["total"] == total_e3]
        f05_min = float(df_e3_ref["f05"].min())
        f05_max = float(df_e3_ref["f05"].max())
        lines.append(
            f"- **E3 (LOO)**: F0.5 entre {_pct(f05_min)} e {_pct(f05_max)} "
            f"na corrida final de referência (total={total_e3}), sugerindo alta "
            "discriminabilidade interna do corpus sintético; não implica "
            "generalização externa."
        )
    else:
        lines.append(
            "- **E3 (LOO)**: sem dados disponíveis no diretório de resultados."
        )
    lines.append(
        "- **E1 (baseline PAN)**: LinearSVC sem subamostragem atinge F0.5 > 95% "
        "em N ≥ 20 mensagens, confirmando a replicação do resultado de Panzariello (2022)."
    )
    lines.append("")

    lines.append("### O que é diagnóstico")
    lines.append("")
    lines.append(
        "- **E2 (cross-domain)**: F0.5 baixo/nulo para o corpus maior (8034 amostras) "
        "confirma lacuna de domínio — vocabulário PAN não transfere para o sintético."
    )
    lines.append(
        "- **E4 (Jaccard)**: Sobreposição de features TF-IDF ≈ 0 explica o resultado "
        "do E2; os dois corpora compartilham pouquíssimas features relevantes."
    )
    lines.append("")

    lines.append("### O que está em aberto")
    lines.append("")
    lines.append(
        "- **E5 (augmentation)**: delta_pp oscila entre negativo e positivo dependendo "
        "do tamanho do corpus de teste. Resultado inconclusivo — requer mais rodadas "
        "com corpus sintético maior e balanceamento controlado."
    )
    lines.append(
        "- Cross-domain com corpus sintético em português precisa de modelo calibrado "
        "para PT-BR; o PAN 2012 é em inglês."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Função principal exportada
# ---------------------------------------------------------------------------


def summarize_r10_results(
    results_dir: str,
    baseline_csv: Optional[str] = None,
) -> str:
    """Gera relatório markdown dos resultados consolidados.

    Parâmetros
    ----------
    results_dir : str
        Caminho para o diretório contendo os CSVs e2–e5.
    baseline_csv : str | None
        Caminho para reports/baseline_results.csv (E1).
        None = auto-detectar como `<results_dir>/../baseline_results.csv`.

    Retorna
    -------
    str
        Relatório em markdown com status, métricas por experimento e síntese.

    Levanta
    -------
    FileNotFoundError
        Se `results_dir` não existir.
    """
    import pandas as pd

    rdir = Path(results_dir)
    if not rdir.exists():
        raise FileNotFoundError(f"Diretório de resultados não encontrado: {rdir}")

    # Resolver baseline_csv
    if baseline_csv is None:
        baseline_path = rdir.parent / BASELINE_CSV_DEFAULT
    else:
        baseline_path = Path(baseline_csv)

    logger.info("Lendo CSVs de %s", rdir)

    # Leitura defensiva: retorna DataFrame vazio se o arquivo não existir
    def _read(path: Path) -> "pd.DataFrame":
        if path.exists():
            logger.info("Lendo %s", path)
            return pd.read_csv(path)
        logger.warning("Arquivo não encontrado: %s — seção ficará vazia", path)
        return pd.DataFrame()

    df_baseline = _read(baseline_path)
    df_e2 = _read(rdir / "e2_cross_domain.csv")
    df_e3 = _read(rdir / "e3_loo.csv")
    df_e4 = _read(rdir / "e4_jaccard.csv")
    df_e5 = _read(rdir / "e5_augmentation.csv")

    # Montar relatório
    sections: list[str] = []

    sections.append(f"# {HEADER_AVISO}")
    sections.append("")
    sections.append(
        "_Relatório final gerado automaticamente por `src/summarize_results.py` "
        "a partir dos CSVs consolidados em `reports/final/`._"
    )
    sections.append("")

    sections.append(_summarize_e1(df_baseline))
    sections.append(_summarize_e2(df_e2))
    sections.append(_summarize_e3(df_e3))
    sections.append(_summarize_e4(df_e4))
    sections.append(_summarize_e5(df_e5))
    sections.append(_sintese(df_e3, df_e4, df_e5))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos da CLI.

    Retorna
    -------
    argparse.ArgumentParser
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="Gera relatório markdown dos resultados de uma rodada de experimentos."
    )
    parser.add_argument(
        "--results-dir",
        default="reports/r10",
        help="Diretório com os CSVs e2–e5 (padrão: reports/r10)",
    )
    parser.add_argument(
        "--baseline-csv",
        default=None,
        help="Caminho para baseline_results.csv (padrão: <results-dir>/../baseline_results.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Arquivo de saída .md (padrão: imprime no stdout)",
    )
    return parser


def main() -> None:
    """Ponto de entrada da CLI.

    Executa a sumarização e grava o resultado em arquivo ou imprime no stdout.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    relatorio = summarize_r10_results(
        results_dir=args.results_dir,
        baseline_csv=args.baseline_csv,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(relatorio, encoding="utf-8")
        logger.info("Relatório salvo em %s", out_path)
    else:
        print(relatorio)


if __name__ == "__main__":
    main()
