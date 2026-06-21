"""
Mineração de features discriminativas para o corpus de conversas.

Implementa T1a da SPEC-05: extração de termos mais discriminativos entre
conversas predatórias e normais via TF-IDF (uni+bigramas) + LinearSVC,
usando os coeficientes do hiperplano de separação como ranking.

Saída: CSV em reports/ com colunas [termo, coeficiente, classe]
para consumo pelo KNIME e rastreabilidade do vocabulario_tipico das personas.

⚠️ KNIME SIGNAL (T6):
    Este módulo gera reports/mining_discriminative_features.csv — arquivo NOVO.
    O workflow KNIME atual NÃO tem um CSV Reader para este arquivo.
    Ação manual necessária: adicionar um nó "CSV Reader" apontando para
    reports/mining_discriminative_features.csv no workflow
    BMT_personas-sinteticas-validacao.knwf antes de usar os dados no KNIME.
    Colunas: termo (String), coeficiente (Double), classe (String).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
REPORTS_DIR = Path("reports")
DEFAULT_CSV = REPORTS_DIR / "mining_discriminative_features.csv"


def extract_discriminative_features(
    predatory_texts: list[str],
    normal_texts: list[str],
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Extrai os termos mais discriminativos entre conversas predatórias e normais.

    Treina TF-IDF (uni+bigramas) + LinearSVC nos textos fornecidos e usa os
    coeficientes do hiperplano para rankear features. Os top_n termos de maior
    coeficiente (associados à classe predatória) e os top_n de menor coeficiente
    (associados à classe normal) são retornados.

    Parâmetros
    ----------
    predatory_texts : list[str]
        Lista de textos de conversas predatórias (classe 1).
    normal_texts : list[str]
        Lista de textos de conversas normais/neutras (classe 0).
    top_n : int
        Número de termos mais discriminativos a retornar por classe (default: 50).

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas [termo, coeficiente, classe].
        classe = "predatorio" para coeficientes positivos (SVM),
                 "normal" para coeficientes negativos.
        Ordenado por coeficiente decrescente (mais predatório primeiro).
    """
    if not predatory_texts or not normal_texts:
        raise ValueError("Ambas as listas de textos devem ser não-vazias.")

    texts = predatory_texts + normal_texts
    labels = np.array([1] * len(predatory_texts) + [0] * len(normal_texts))

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
    X = vec.fit_transform(texts)

    clf = LinearSVC(C=1.0, max_iter=10_000, random_state=RANDOM_STATE)
    clf.fit(X, labels)

    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]

    top_pred_idx = np.argsort(coefs)[::-1][:top_n]
    top_norm_idx = np.argsort(coefs)[:top_n]

    rows = []
    for idx in top_pred_idx:
        rows.append({
            "termo": feature_names[idx],
            "coeficiente": float(coefs[idx]),
            "classe": "predatorio",
        })
    for idx in top_norm_idx:
        rows.append({
            "termo": feature_names[idx],
            "coeficiente": float(coefs[idx]),
            "classe": "normal",
        })

    df = pd.DataFrame(rows, columns=["termo", "coeficiente", "classe"])
    logger.info(
        "Features discriminativas extraídas: %d predatórias + %d normais (top_n=%d)",
        top_n, top_n, top_n,
    )
    return df


def save_mining_csv(
    df: pd.DataFrame,
    path: str | Path = DEFAULT_CSV,
) -> Path:
    """
    Salva o DataFrame de features discriminativas em CSV.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame retornado por extract_discriminative_features.
    path : str | Path
        Caminho de saída (default: reports/mining_discriminative_features.csv).

    Retorna
    -------
    Path
        Caminho absoluto do arquivo salvo.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info("CSV de features discriminativas salvo em: %s", out)
    return out.resolve()
