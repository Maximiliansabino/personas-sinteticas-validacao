"""Gera um dossie Markdown da execucao final do artigo.

O arquivo produzido e pensado como fonte de trabalho para escrita cientifica:
registra configuracao da execucao, artefatos gerados, inventario do corpus,
metricas E1-E5, riscos/limitacoes e instrucoes para o agente de escrita.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Le um CSV como lista de dicionarios. Retorna lista vazia se ausente."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_parquet_profile(path: Path) -> dict[str, Any]:
    """Extrai estatisticas basicas de um parquet usado no pipeline."""
    if not path.exists():
        return {"path": str(path), "exists": False}

    import pandas as pd

    df = pd.read_parquet(path)
    profile: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "rows": len(df),
        "columns": list(df.columns),
    }
    if "conversation_id" in df.columns:
        profile["n_conversations"] = int(df["conversation_id"].nunique())
    if "n_msgs" in df.columns:
        profile["n_msgs"] = sorted(int(v) for v in df["n_msgs"].dropna().unique())
    if "label" in df.columns:
        profile["labels_rows"] = {
            str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()
        }
        if "conversation_id" in df.columns:
            by_conv = df.drop_duplicates("conversation_id")
            profile["labels_conversations"] = {
                str(k): int(v)
                for k, v in by_conv["label"].value_counts().sort_index().items()
            }
    if "status" in df.columns:
        profile["status"] = {
            str(k): int(v) for k, v in df["status"].value_counts().sort_index().items()
        }
    return profile


def _xml_profile(synthetic_dir: Path) -> dict[str, Any]:
    """Conta XMLs sinteticos por label, idioma e status."""
    profile: dict[str, Any] = {
        "dir": str(synthetic_dir),
        "exists": synthetic_dir.exists(),
        "total_xml": 0,
        "by_label": {},
        "by_lang": {},
        "by_status": {},
        "guardrail_sessions": [],
    }
    if not synthetic_dir.exists():
        return profile

    from lxml import etree

    for xml_path in sorted(synthetic_dir.glob("*.xml")):
        profile["total_xml"] += 1
        try:
            tree = etree.parse(str(xml_path))
            conv = tree.getroot().find("conversation")
            label = conv.get("label", "unknown") if conv is not None else "unknown"
            lang = conv.get("lang", "unknown") if conv is not None else "unknown"
            interrupted = (
                conv is not None and conv.get("guardrail_interrupted") == "true"
            )
            status = "guardrail_interrupted" if interrupted else "complete_or_partial"
            profile["by_label"][label] = profile["by_label"].get(label, 0) + 1
            profile["by_lang"][lang] = profile["by_lang"].get(lang, 0) + 1
            profile["by_status"][status] = profile["by_status"].get(status, 0) + 1
            if interrupted:
                profile["guardrail_sessions"].append(xml_path.stem)
        except Exception:
            profile["by_status"]["parse_error"] = profile["by_status"].get("parse_error", 0) + 1
    return profile


def _git_metadata() -> dict[str, str]:
    """Coleta metadados Git sem alterar o repositorio."""
    def _cmd(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "indisponivel"

    return {
        "branch": _cmd(["git", "branch", "--show-current"]),
        "commit": _cmd(["git", "rev-parse", "--short", "HEAD"]),
        "dirty": _cmd(["git", "status", "--short"]),
    }


def _fmt_float(value: str | float | None) -> str:
    """Formata numero decimal de forma defensiva."""
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _append_profile(lines: list[str], title: str, profile: dict[str, Any]) -> None:
    lines.extend(["", f"## {title}", ""])
    if not profile.get("exists"):
        lines.append(f"- Arquivo ausente: `{profile.get('path')}`")
        return
    lines.append(f"- Caminho: `{profile['path']}`")
    lines.append(f"- Linhas: {profile.get('rows', '-')}")
    if "n_conversations" in profile:
        lines.append(f"- Conversas unicas: {profile['n_conversations']}")
    if "n_msgs" in profile:
        lines.append(f"- Cortes n_msgs: {profile['n_msgs']}")
    if "labels_rows" in profile:
        lines.append(f"- Labels por linha: {profile['labels_rows']}")
    if "labels_conversations" in profile:
        lines.append(f"- Labels por conversa: {profile['labels_conversations']}")
    if "status" in profile:
        lines.append(f"- Status: {profile['status']}")


def _append_csv_table(
    lines: list[str],
    title: str,
    rows: list[dict[str, str]],
    columns: list[tuple[str, str]],
    limit: int | None = None,
) -> None:
    lines.extend(["", f"## {title}", ""])
    if not rows:
        lines.append("_CSV nao encontrado ou vazio._")
        return
    selected = rows if limit is None else rows[:limit]
    lines.append("| " + " | ".join(label for label, _ in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in selected:
        values = []
        for _, key in columns:
            aliases = key.split("|")
            value = "-"
            for alias in aliases:
                if alias in row and row.get(alias, "") != "":
                    value = row.get(alias, "-")
                    break
            if key.startswith(("f", "precision", "recall", "jaccard")):
                value = _fmt_float(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")


def gerar_relatorio(
    run_id: str,
    logs_dir: Path,
    reports_dir: Path,
    pan_path: Path,
    synth_path: Path,
    synth_train_path: Path,
    synth_test_path: Path,
    output: Path,
) -> None:
    """Gera o Markdown detalhado da execucao final."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git = _git_metadata()
    pan_profile = _read_parquet_profile(pan_path)
    synth_profile = _read_parquet_profile(synth_path)
    synth_train_profile = _read_parquet_profile(synth_train_path)
    synth_test_profile = _read_parquet_profile(synth_test_path)
    xmls = _xml_profile(Path("data/synthetic"))

    baseline = _read_csv(reports_dir / "baseline_results.csv")
    e2 = _read_csv(reports_dir / "e2_cross_domain.csv")
    e3 = _read_csv(reports_dir / "e3_loo.csv")
    e4 = _read_csv(reports_dir / "e4_jaccard.csv")
    e5 = _read_csv(reports_dir / "e5_augmentation.csv")

    lines: list[str] = [
        "---",
        "tags: [mestrado, bmt, personas-sinteticas, experimento, artigo]",
        f"created: {datetime.now().strftime('%Y-%m-%d')}",
        f"updated: {datetime.now().strftime('%Y-%m-%d')}",
        "type: artefato",
        "---",
        "",
        f"# Dossie da execucao final do artigo - {run_id}",
        "",
        "## Finalidade",
        "",
        "Este dossie registra a execucao final do pipeline de personas sinteticas",
        "para apoiar a escrita do artigo cientifico em padrao IEEE/LaTeX.",
        "Ele deve ser usado como fonte de evidencias junto com os CSVs em `reports/`,",
        "o codigo atual do repositorio e os agentes `.agents/escrita_cientifca.md`",
        "e `.agents/revisao_cientifica.md`.",
        "",
        "## Identificacao da execucao",
        "",
        f"- RUN_ID: `{run_id}`",
        f"- Gerado em: {generated_at}",
        f"- Branch Git: `{git['branch']}`",
        f"- Commit Git: `{git['commit']}`",
        f"- Logs: `{logs_dir}`",
        f"- Reports/KNIME: `{reports_dir}`",
        f"- Worktree sujo no momento do relatorio: {'sim' if git['dirty'] else 'nao'}",
        "",
        "## Escopo executado",
        "",
        "- Pre-flight de arquivos locais, variaveis de ambiente e provider OpenAI.",
        "- Teste de conectividade, escrita, leitura e limpeza no MongoDB Atlas.",
        "- Validacao estrutural das fichas de personas.",
        "- Execucao do pipeline LLM canonicamente especificado no repositorio: Groq + Anthropic.",
        "- OpenAI foi tratado como opcional porque nao ha provider `openai/` no `ModelRouter` atual.",
        "- Consolidacao do parquet sintetico final, split treino/teste, E1, E2, E3, E4 e E5.",
        "- Exportacao dos CSVs finais para consumo no KNIME.",
    ]

    lines.extend(["", "## Inventario dos XMLs sinteticos", ""])
    lines.append(f"- Diretorio: `{xmls['dir']}`")
    lines.append(f"- Total de XMLs: {xmls['total_xml']}")
    lines.append(f"- Por label: {xmls['by_label']}")
    lines.append(f"- Por idioma: {xmls['by_lang']}")
    lines.append(f"- Por status: {xmls['by_status']}")
    if xmls["guardrail_sessions"]:
        lines.append(f"- Sessoes com guardrail: {xmls['guardrail_sessions']}")
    else:
        lines.append("- Sessoes com guardrail: nenhuma detectada no inventario XML")

    _append_profile(lines, "PAN 2012 processado", pan_profile)
    _append_profile(lines, "Corpus sintetico final", synth_profile)
    _append_profile(lines, "Split sintetico de treino", synth_train_profile)
    _append_profile(lines, "Split sintetico de teste", synth_test_profile)

    _append_csv_table(
        lines,
        "E1 - Baseline PAN",
        baseline[-24:],
        [
            ("Experimento", "experiment_name"),
            ("Tipo", "experiment_type"),
            ("Classificador", "classifier"),
            ("n_msgs", "n_msgs"),
            ("F0.5", "f05_mean"),
            ("Precisao", "precision_mean"),
            ("Recall", "recall_mean"),
            ("Corpus", "corpus"),
        ],
    )
    _append_csv_table(
        lines,
        "E2 - Cross-domain PAN para sintetico",
        e2,
        [
            ("n_msgs", "n_msgs"),
            ("F0.5", "f05"),
            ("F1", "f1"),
            ("Precisao", "precision"),
            ("Recall", "recall"),
            ("Total", "total"),
            ("Pred.", "predatory"),
            ("Norm.", "normal"),
        ],
    )
    _append_csv_table(
        lines,
        "E3 - Leave-one-out no sintetico",
        e3,
        [
            ("n_msgs", "n_msgs"),
            ("F0.5", "f05"),
            ("IC inf.", "ic_lower|f05_ci_low"),
            ("IC sup.", "ic_upper|f05_ci_high"),
            ("Precisao", "precision"),
            ("Recall", "recall"),
            ("Amostras", "total|n_total"),
        ],
    )
    _append_csv_table(
        lines,
        "E4 - Jaccard TF-IDF",
        e4,
        [
            ("n_msgs", "n_msgs"),
            ("Jaccard", "jaccard"),
            ("Comuns", "n_common"),
            ("So PAN", "n_only_pan"),
            ("So sintetico", "n_only_synth"),
            ("Top N", "top_n"),
        ],
    )
    _append_csv_table(
        lines,
        "E5 - Aumento de dados",
        e5,
        [
            ("n_msgs", "n_msgs"),
            ("Variante", "variant|experiment_type"),
            ("F0.5", "f05|f05_augmented|f05_baseline"),
            ("Delta p.p.", "delta_pp"),
            ("Precisao", "precision"),
            ("Recall", "recall"),
            ("Total treino", "total|train_size"),
        ],
    )

    lines.extend([
        "",
        "## Leitura cientifica recomendada",
        "",
        "- O artigo deve apresentar o trabalho como uma metodologia geral de geracao",
        "  de personas sinteticas via LLM para dominios com restricao de dados reais.",
        "- A deteccao precoce de grooming online e o caso de validacao, nao o limite",
        "  conceitual da metodologia.",
        "- PAN 2012 deve ser descrito como corpus de baseline, treino e calibracao",
        "  linguistica/classificatoria.",
        "- OSAEBA, quando usado, deve ser descrito como fonte contextual e demografica,",
        "  nao como corpus de conversas.",
        "- E2 e E4 devem ser interpretados com cautela quando indicarem barreira",
        "  lexical/idioma entre PAN e corpus sintetico.",
        "- E3 e a evidencia interna mais direta do corpus sintetico, mas nao prova",
        "  generalizacao externa.",
        "- E5 deve ser discutido como aumento de dados e tratado como inconclusivo",
        "  se os resultados forem ruidosos ou dependentes de poucas amostras.",
        "",
        "## Figuras e tabelas sugeridas para o artigo",
        "",
        "- Tabela de configuracao experimental: corpus, modelos, idiomas, n_msgs e splits.",
        "- Tabela E1 com baseline PAN por classificador e balanceamento.",
        "- Tabela E2-E5 com F0.5, precisao e recall nos cortes principais.",
        "- Figura da arquitetura multiagente e fluxo de persistencia MongoDB/XML/Parquet.",
        "- Figura de crescimento/cobertura do corpus sintetico por rodada ou provider.",
        "- Figura de diagnostico lexical E4 para discutir distancia PAN-sintetico.",
        "",
        "## Pendencias para escrita",
        "",
        "- Conferir manualmente os CSVs exportados no KNIME antes de congelar tabelas finais.",
        "- Verificar se todos os numeros usados no texto do artigo foram copiados deste dossie",
        "  ou diretamente dos CSVs finais.",
        "- Nao preencher lacunas bibliograficas com memoria ou conhecimento externo sem citacao.",
        "- Registrar qualquer rodada descartada, sessao parcial ou guardrail como limitacao.",
        "",
        "## Comando de reproducao",
        "",
        "```bash",
        "FINAL_EXECUTION=1 bash run_execucao_final_artigo.sh",
        "```",
        "",
        "## Logs produzidos",
        "",
    ])

    if logs_dir.exists():
        for log in sorted(logs_dir.glob("*.log")):
            lines.append(f"- `{log}` ({log.stat().st_size} bytes)")
    else:
        lines.append(f"- Diretorio de logs ausente: `{logs_dir}`")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Dossie final gerado: {output}")


def main() -> None:
    """Entrada CLI."""
    parser = argparse.ArgumentParser(description="Gera dossie Markdown da execucao final")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--pan", required=True)
    parser.add_argument("--synth", required=True)
    parser.add_argument("--synth-train", required=True)
    parser.add_argument("--synth-test", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gerar_relatorio(
        run_id=args.run_id,
        logs_dir=Path(args.logs_dir),
        reports_dir=Path(args.reports_dir),
        pan_path=Path(args.pan),
        synth_path=Path(args.synth),
        synth_train_path=Path(args.synth_train),
        synth_test_path=Path(args.synth_test),
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
