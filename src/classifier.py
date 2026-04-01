"""
Classificadores para detecção de conversas predatórias.
Baseline: SVM linear + Naive Bayes Multinomial com TF-IDF.
Baseado na Estratégia 1 de Panzariello (2022).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np


def create_svm_pipeline() -> Pipeline:
    """Cria pipeline SVM linear com TF-IDF (uni+bigramas)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(
            class_weight="balanced",
            max_iter=10000,
        )),
    ])


def create_nb_pipeline() -> Pipeline:
    """Cria pipeline Naive Bayes Multinomial com TF-IDF."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
        )),
        ("clf", MultinomialNB(alpha=1.0)),
    ])


def evaluate_cross_validation(pipeline, X, y, cv=10):
    """Executa cross-validation estratificada e retorna métricas."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1"]

    results = cross_validate(
        pipeline, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
    )

    metrics = {}
    for metric in scoring:
        key = f"test_{metric}"
        metrics[metric] = {
            "mean": np.mean(results[key]),
            "std": np.std(results[key]),
        }

    return metrics


def fbeta_score(precision, recall, beta=0.5):
    """Calcula F-beta score (padrão: F0.5 que prioriza precisão)."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


if __name__ == "__main__":
    print("Módulo de classificação.")
    print("Use: python classifier.py --train <dados_treino> --test <dados_teste>")
    print("Pipeline em desenvolvimento.")
