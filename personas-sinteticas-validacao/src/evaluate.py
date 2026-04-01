"""
Módulo de avaliação da qualidade das personas.
Mede 4 dimensões: utilidade classificatória, representatividade
linguística, consistência interna e diversidade.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calcula coeficiente de Jaccard entre dois conjuntos."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def top_features(texts: list[str], n: int = 50) -> set:
    """Extrai as top-N features mais discriminativas por TF-IDF."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=n)
    vectorizer.fit_transform(texts)
    return set(vectorizer.get_feature_names_out())


def feature_overlap(real_texts: list[str], synthetic_texts: list[str], n: int = 50) -> dict:
    """
    Compara features entre corpus real e sintético.
    Retorna Jaccard e listas de features compartilhadas/exclusivas.
    """
    real_features = top_features(real_texts, n)
    synth_features = top_features(synthetic_texts, n)

    return {
        "jaccard": jaccard_similarity(real_features, synth_features),
        "shared": real_features & synth_features,
        "only_real": real_features - synth_features,
        "only_synthetic": synth_features - real_features,
    }


def persona_diversity(texts_by_persona: dict[str, str]) -> dict:
    """
    Mede diversidade entre personas via distância coseno.
    texts_by_persona: {persona_id: texto_concatenado}
    """
    ids = list(texts_by_persona.keys())
    texts = [texts_by_persona[pid] for pid in ids]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    sim_matrix = cosine_similarity(tfidf_matrix)

    # Distância média (excluindo diagonal)
    n = len(ids)
    if n < 2:
        return {"mean_cosine_distance": 0.0, "pairs": []}

    distances = []
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = 1 - sim_matrix[i, j]
            distances.append(dist)
            pairs.append((ids[i], ids[j], dist))

    return {
        "mean_cosine_distance": np.mean(distances),
        "min_distance": np.min(distances),
        "max_distance": np.max(distances),
        "pairs": sorted(pairs, key=lambda x: x[2]),
    }


if __name__ == "__main__":
    print("Módulo de avaliação de qualidade das personas.")
    print("Funções: jaccard_similarity, feature_overlap, persona_diversity")
    print("Pipeline em desenvolvimento.")
