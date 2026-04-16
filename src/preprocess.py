"""
Pipeline de pré-processamento da base PAN 2012 replicando
Panzariello (2022), Estratégia 1.

Gera um DataFrame com linhas (conversation_id, n_msgs, text, label)
para cada ponto de corte em n_msgs_list = [5, 10, 15, 20, 24, 50].

Uso:
    python -m src.preprocess \\
        --train  data/pan2012/train/pan12-...-training-corpus-2012-05-01.xml \\
        --predators data/pan2012/train/pan12-...-predators-2012-05-01.txt \\
        --output data/processed/pan2012_train.parquet
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from lxml import etree

logger = logging.getLogger(__name__)

# Pontos de corte padrão de Panzariello (2022)
DEFAULT_N_MSGS: list[int] = [5, 10, 15, 20, 24, 50]

# Filtros Panzariello (2022)
_MIN_MSGS_PER_AUTHOR = 6
_MAX_NON_ALNUM_CHARS = 8
_REQUIRED_AUTHORS = 2


# ---------------------------------------------------------------------------
# Utilitários NLTK
# ---------------------------------------------------------------------------

def _ensure_nltk_stopwords() -> set[str]:
    """Garante que o corpus de stopwords do nltk está disponível."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        logger.info("Baixando nltk stopwords...")
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Limpeza textual básica
# ---------------------------------------------------------------------------

def basic_clean(text: str, stopwords: set[str] | None = None) -> str:
    """
    Limpeza básica de mensagem de chat (inglês).

    Passos:
    1. Lowercase
    2. Remove URLs
    3. Normaliza repetições de letras (noooo → no, lolll → lol) — 3+ → 2
    4. Remove stopwords inglês (nltk) se stopwords for fornecido

    Args:
        text:       Texto bruto da mensagem.
        stopwords:  Conjunto de stopwords para remoção. Se None, não remove.

    Returns:
        Texto limpo como string.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Normaliza repetições de letras: 3 ou mais ocorrências → 2
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 4. Remove stopwords
    if stopwords:
        tokens = text.split()
        tokens = [t for t in tokens if t not in stopwords]
        text = " ".join(tokens)

    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Carregamento do XML PAN 2012
# ---------------------------------------------------------------------------

def _load_predator_ids(predators_path: str) -> frozenset[str]:
    """Carrega o arquivo de IDs de predadores (um ID por linha)."""
    path = Path(predators_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de predadores não encontrado: {predators_path}")

    ids = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            pid = line.strip()
            if pid:
                ids.add(pid)

    logger.info("Predadores carregados: %d IDs de '%s'", len(ids), predators_path)
    return frozenset(ids)


def _parse_xml(xml_path: str) -> list[dict]:
    """
    Parseia o XML PAN 2012 com lxml.

    Formato esperado::

        <corpus>
          <conversation id="conv-id">
            <message line="1" author="hash...">
              texto da mensagem
            </message>
          </conversation>
        </corpus>

    Returns:
        Lista de dicionários com keys: id, messages.
        Cada mensagem: {line, author, text}
    """
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"XML não encontrado: {xml_path}")

    logger.info("Parseando XML: %s", xml_path)
    tree = etree.parse(str(path))
    conversations = []

    for conv_el in tree.findall(".//conversation"):
        conv_id = conv_el.get("id", "")
        messages = []
        for msg_el in conv_el.findall("message"):
            raw_text = (msg_el.text or "").strip()
            messages.append({
                "line": int(msg_el.get("line", 0)),
                "author": msg_el.get("author", ""),
                "text": raw_text,
            })

        # Garante ordem cronológica pelo número de linha
        messages.sort(key=lambda m: m["line"])
        conversations.append({"id": conv_id, "messages": messages})

    logger.info("Conversas brutas parseadas: %d", len(conversations))
    return conversations


# ---------------------------------------------------------------------------
# Filtros Panzariello (2022)
# ---------------------------------------------------------------------------

def _apply_filters(conversations: list[dict]) -> list[dict]:
    """
    Aplica os filtros de Panzariello (2022) na seguinte ordem:

    1. Remove mensagens vazias.
    2. Remove mensagens com mais de 8 caracteres não-alfanuméricos.
    3. Mantém apenas conversas com exatamente 2 autores distintos.
    4. Mantém apenas conversas em que cada autor possui >= 6 mensagens.

    Returns:
        Lista filtrada de dicionários de conversa.
    """
    kept = []
    stats = Counter({"vazia": 0, "nao_alnum": 0, "autores": 0, "min_msgs": 0, "ok": 0})

    for conv in conversations:
        msgs = conv["messages"]

        # 1. Remove mensagens vazias
        msgs = [m for m in msgs if m["text"]]
        removed_empty = len(conv["messages"]) - len(msgs)
        stats["vazia"] += removed_empty

        # 2. Remove mensagens com mais de 8 chars não-alfanuméricos
        def _non_alnum_count(text: str) -> int:
            return sum(1 for ch in text if not ch.isalnum() and not ch.isspace())

        before = len(msgs)
        msgs = [m for m in msgs if _non_alnum_count(m["text"]) <= _MAX_NON_ALNUM_CHARS]
        stats["nao_alnum"] += before - len(msgs)

        # 3. Exatamente 2 autores distintos
        authors = {m["author"] for m in msgs}
        if len(authors) != _REQUIRED_AUTHORS:
            stats["autores"] += 1
            continue

        # 4. Cada autor com >= 6 mensagens
        msgs_per_author = Counter(m["author"] for m in msgs)
        if any(count < _MIN_MSGS_PER_AUTHOR for count in msgs_per_author.values()):
            stats["min_msgs"] += 1
            continue

        kept.append({**conv, "messages": msgs})
        stats["ok"] += 1

    logger.info(
        "Filtros aplicados — removidas: %d vazias, %d não-alnum, "
        "%d ≠2 autores, %d <6 msgs/autor | restantes: %d",
        stats["vazia"],
        stats["nao_alnum"],
        stats["autores"],
        stats["min_msgs"],
        stats["ok"],
    )
    return kept


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def load_pan2012(
    xml_path: str,
    predators_path: str,
    n_msgs_list: list[int] | None = None,
) -> pd.DataFrame:
    """
    Carrega, filtra e processa a base PAN 2012.

    Para cada conversa válida e cada valor de n em n_msgs_list, gera uma
    linha no DataFrame com as primeiras n mensagens concatenadas e limpas.

    Args:
        xml_path:       Caminho para o XML PAN 2012.
        predators_path: Caminho para o arquivo .txt com IDs de predadores.
        n_msgs_list:    Lista de pontos de corte. Default: [5,10,15,20,24,50].

    Returns:
        DataFrame com colunas [conversation_id, n_msgs, text, label].
        label=1 predatória, label=0 normal.
    """
    if n_msgs_list is None:
        n_msgs_list = DEFAULT_N_MSGS

    predator_ids = _load_predator_ids(predators_path)
    raw_convs = _parse_xml(xml_path)
    filtered = _apply_filters(raw_convs)

    stopwords = _ensure_nltk_stopwords()

    rows: list[dict] = []
    for conv in filtered:
        msgs = conv["messages"]
        authors = {m["author"] for m in msgs}
        label = int(bool(authors & predator_ids))  # 1 se algum autor é predador

        for n in n_msgs_list:
            window = msgs[:n]
            if not window:
                continue

            combined = " ".join(m["text"] for m in window)
            cleaned = basic_clean(combined, stopwords=stopwords)

            rows.append({
                "conversation_id": conv["id"],
                "n_msgs": n,
                "text": cleaned,
                "label": label,
            })

    df = pd.DataFrame(rows, columns=["conversation_id", "n_msgs", "text", "label"])
    logger.info(
        "DataFrame gerado: %d linhas | %d conversas únicas | label 1: %d, label 0: %d",
        len(df),
        df["conversation_id"].nunique(),
        (df["label"] == 1).sum(),
        (df["label"] == 0).sum(),
    )
    return df


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Pré-processamento PAN 2012 — Panzariello (2022)"
    )
    parser.add_argument(
        "--train",
        required=True,
        metavar="XML",
        help="Caminho para o XML do corpus PAN 2012",
    )
    parser.add_argument(
        "--predators",
        required=True,
        metavar="TXT",
        help="Caminho para o arquivo .txt de IDs de predadores",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PARQUET",
        help="Caminho de saída para o arquivo .parquet",
    )
    parser.add_argument(
        "--n-msgs",
        nargs="+",
        type=int,
        default=DEFAULT_N_MSGS,
        metavar="N",
        help=f"Pontos de corte de mensagens (default: {DEFAULT_N_MSGS})",
    )
    args = parser.parse_args()

    df = load_pan2012(
        xml_path=args.train,
        predators_path=args.predators,
        n_msgs_list=args.n_msgs,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Salvo em '%s' (%d linhas)", out_path, len(df))
