"""
Estratégia 1 de Panzariello (2022): classificação binária de conversas
predatórias com SVM linear e Naive Bayes, TF-IDF uni+bigramas,
cross-validation estratificada, com e sem undersampling.

Experimentos disponíveis:
  sem_balanceamento — corpus original
  undersampling     — RandomUnderSampler(random_state=42)

Métricas reportadas: F0.5, F1, Precisão, Recall (média e std sobre 5-folds).

Uso:
    python -m src.classifier \\
        --input  data/processed/pan2012_train.parquet \\
        --experiment baseline
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.db import MongoDBClient, estimate_cost

logger = logging.getLogger(__name__)

# Pontos de corte de Panzariello (2022)
N_MSGS_LIST: list[int] = [5, 10, 15, 20, 24, 50]

# Configurações fixas
TFIDF_PARAMS = dict(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
CV_FOLDS = 5
RANDOM_STATE = 42

REPORTS_DIR = Path("reports")
RESULTS_CSV = REPORTS_DIR / "baseline_results.csv"


# ---------------------------------------------------------------------------
# Métrica F-beta (β=0.5 prioriza precisão)
# ---------------------------------------------------------------------------

def _fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calcula F-beta score. Retorna 0.0 se precisão+recall == 0."""
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def _fbeta_scorer(beta: float = 0.5):
    """Retorna scorer compatível com sklearn para F-beta."""
    def _score(estimator, X, y):
        y_pred = estimator.predict(X)
        p = precision_score(y, y_pred, zero_division=0)
        r = recall_score(y, y_pred, zero_division=0)
        return _fbeta(p, r, beta)
    return _score


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def _make_pipeline_svm(balanced: bool) -> ImbPipeline:
    """
    Pipeline LinearSVC com TF-IDF.
    Se balanced=True, aplica RandomUnderSampler antes da classificação.
    """
    steps = [("tfidf", TfidfVectorizer(**TFIDF_PARAMS))]
    if balanced:
        steps.append(("sampler", RandomUnderSampler(random_state=RANDOM_STATE)))
    steps.append(("clf", LinearSVC(C=1.0, class_weight=None, max_iter=10_000, random_state=RANDOM_STATE)))
    return ImbPipeline(steps)


def _make_pipeline_nb(balanced: bool) -> ImbPipeline:
    """
    Pipeline MultinomialNB com TF-IDF.
    MultinomialNB requer features não-negativas — sublinear_tf mantém isso.
    """
    steps = [("tfidf", TfidfVectorizer(**TFIDF_PARAMS))]
    if balanced:
        steps.append(("sampler", RandomUnderSampler(random_state=RANDOM_STATE)))
    steps.append(("clf", MultinomialNB(alpha=0.1)))
    return ImbPipeline(steps)


# ---------------------------------------------------------------------------
# Cross-validation para um único (pipeline, X, y)
# ---------------------------------------------------------------------------

def _run_cv(pipeline: ImbPipeline, X: pd.Series, y: pd.Series) -> dict[str, float]:
    """
    Executa CV estratificada de FOLDS e retorna médias e desvios.

    Returns:
        {f05_mean, f05_std, f1_mean, precision_mean, recall_mean}
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "precision": make_scorer(precision_score, zero_division=0),
        "recall":    make_scorer(recall_score,    zero_division=0),
        "f1":        make_scorer(f1_score,         zero_division=0),
        "f05":       _fbeta_scorer(beta=0.5),
    }

    cv_results = cross_validate(
        pipeline, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    return {
        "f05_mean":       float(np.mean(cv_results["test_f05"])),
        "f05_std":        float(np.std(cv_results["test_f05"])),
        "f1_mean":        float(np.mean(cv_results["test_f1"])),
        "precision_mean": float(np.mean(cv_results["test_precision"])),
        "recall_mean":    float(np.mean(cv_results["test_recall"])),
    }


# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------

def run_experiment(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    """
    Executa a Estratégia 1 de Panzariello (2022) para todos os pontos de corte.

    Para cada n_msgs em N_MSGS_LIST e cada classificador (SVM, NB),
    roda CV estratificada com e sem undersampling, salva no MongoDB e
    acumula resultados em um DataFrame.

    Args:
        df:              DataFrame com colunas [conversation_id, n_msgs, text, label].
                         Gerado por src.preprocess.load_pan2012().
        experiment_name: Prefixo usado no experiment_id (ex: "baseline").

    Returns:
        DataFrame com todas as métricas por configuração.
    """
    rows: list[dict] = []

    try:
        db = MongoDBClient()
        db_ok = db.ping()
    except Exception as exc:
        logger.warning("MongoDB indisponível, resultados não serão persistidos: %s", exc)
        db_ok = False
        db = None  # type: ignore[assignment]

    classifiers = {
        "LinearSVC": (_make_pipeline_svm, "SVM linear, C=1.0"),
        "MultinomialNB": (_make_pipeline_nb, "Naive Bayes Multinomial, α=0.1"),
    }

    experiments = {
        "sem_balanceamento": False,
        "undersampling":     True,
    }

    total_configs = len(N_MSGS_LIST) * len(classifiers) * len(experiments)
    done = 0

    for n_msgs in N_MSGS_LIST:
        subset = df[df["n_msgs"] == n_msgs].copy()
        if subset.empty:
            logger.warning("Nenhum dado para n_msgs=%d, pulando.", n_msgs)
            continue

        X = subset["text"]
        y = subset["label"]

        n_pred = int((y == 1).sum())
        n_norm = int((y == 0).sum())
        n_total = len(y)

        logger.info(
            "n_msgs=%d → %d amostras (pred=%d, norm=%d)",
            n_msgs, n_total, n_pred, n_norm,
        )

        for exp_type, balanced in experiments.items():
            for clf_name, (pipeline_fn, _) in classifiers.items():
                done += 1
                logger.info(
                    "[%d/%d] n_msgs=%d | %s | %s",
                    done, total_configs, n_msgs, exp_type, clf_name,
                )

                pipeline = pipeline_fn(balanced=balanced)
                metrics = _run_cv(pipeline, X, y)

                timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                experiment_id = f"{experiment_name}_{exp_type}_{clf_name}_{n_msgs}msgs_{timestamp_str}"

                row = {
                    "experiment_id":   experiment_id,
                    "experiment_name": experiment_name,
                    "experiment_type": exp_type,
                    "n_msgs":          n_msgs,
                    "classifier":      clf_name,
                    "f05_mean":        metrics["f05_mean"],
                    "f05_std":         metrics["f05_std"],
                    "f1_mean":         metrics["f1_mean"],
                    "precision_mean":  metrics["precision_mean"],
                    "recall_mean":     metrics["recall_mean"],
                    "n_total":         n_total,
                    "n_predatory":     n_pred,
                    "n_normal":        n_norm,
                }
                rows.append(row)

                if db_ok:
                    try:
                        db.save_experiment({
                            "experiment_id":   experiment_id,
                            "timestamp":       datetime.utcnow(),
                            "experiment_type": f"{experiment_name}_{exp_type}",
                            "n_msgs":          n_msgs,
                            "classifier":      clf_name,
                            "results": {
                                "f05":       metrics["f05_mean"],
                                "f1":        metrics["f1_mean"],
                                "precision": metrics["precision_mean"],
                                "recall":    metrics["recall_mean"],
                                "std_f05":   metrics["f05_std"],
                            },
                            "dataset_sizes": {
                                "total":     n_total,
                                "predatory": n_pred,
                                "normal":    n_norm,
                            },
                        })
                    except Exception as exc:
                        logger.error("Erro ao salvar experimento no MongoDB: %s", exc)

    results_df = pd.DataFrame(rows)
    _save_csv(results_df)
    _print_table(results_df)
    return results_df


# ---------------------------------------------------------------------------
# Persistência CSV
# ---------------------------------------------------------------------------

def _save_csv(df: pd.DataFrame) -> None:
    """Salva o DataFrame de resultados em reports/baseline_results.csv."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(RESULTS_CSV, index=False, float_format="%.4f")
    logger.info("Resultados salvos em '%s' (%d linhas totais)", RESULTS_CSV, len(df))


# ---------------------------------------------------------------------------
# Tabela terminal
# ---------------------------------------------------------------------------

def _print_table(df: pd.DataFrame) -> None:
    """Imprime tabela formatada de resultados no terminal."""
    if df.empty:
        print("Nenhum resultado para exibir.")
        return

    col_order = [
        "n_msgs", "experiment_type", "classifier",
        "f05_mean", "f05_std", "f1_mean", "precision_mean", "recall_mean",
        "n_total", "n_predatory", "n_normal",
    ]
    display = df[[c for c in col_order if c in df.columns]].copy()

    float_cols = ["f05_mean", "f05_std", "f1_mean", "precision_mean", "recall_mean"]
    for col in float_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.4f}")

    print("\n" + "=" * 100)
    print(f"{'RESULTADOS — ESTRATÉGIA 1 (PANZARIELLO 2022)':^100}")
    print("=" * 100)
    print(display.to_string(index=False))
    print("=" * 100)

    # Destaca a melhor configuração por F0.5
    best = df.loc[df["f05_mean"].idxmax()]
    print(
        f"\nMelhor F0.5: {best['f05_mean']:.4f} ± {best['f05_std']:.4f} "
        f"| {best['classifier']} | {best['experiment_type']} | n_msgs={best['n_msgs']}"
    )


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Estratégia 1 de Panzariello (2022) — classificação binária"
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PARQUET",
        help="Caminho para o parquet gerado por src.preprocess",
    )
    parser.add_argument(
        "--experiment",
        default="baseline",
        metavar="NOME",
        help="Nome do experimento usado como prefixo no experiment_id (default: baseline)",
    )
    parser.add_argument(
        "--n-msgs",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Subconjunto de pontos de corte (default: todos)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {args.input}")

    logger.info("Carregando dados de '%s'...", args.input)
    df = pd.read_parquet(input_path)
    logger.info("Dados carregados: %d linhas, colunas=%s", len(df), list(df.columns))

    if args.n_msgs:
        df = df[df["n_msgs"].isin(args.n_msgs)]
        logger.info("Filtrado para n_msgs=%s: %d linhas", args.n_msgs, len(df))

    run_experiment(df, experiment_name=args.experiment)
