"""
Experimentos E2-E5 + análise descritiva do OSAEBA — Panzariello (2022).

E2 — Cross-domain:   treina no PAN 2012, avalia no corpus sintético.
E3 — Leave-one-out:  validação cruzada LOO no corpus sintético + IC 95% bootstrap.
E4 — Jaccard:        sobreposição de features TF-IDF entre os dois corpora.
E5 — Augmentation:   treina com PAN + sintético, avalia no PAN test.
OSAEBA — análise descritiva do survey sociológico (sem conversas).

Resultados persistidos no MongoDB e exportados como CSVs para o KNIME.

Uso:
    python -m src.evaluate --all --export-knime

    python -m src.evaluate \\
        --pan    data/processed/pan2012_train.parquet \\
        --synth  data/processed/synthetic.parquet \\
        --e2 --e3 --e4 --e5 --osaeba --export-knime
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.db import MongoDBClient

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
N_MSGS_LIST: list[int] = [5, 10, 15, 20, 24, 50]
RANDOM_STATE = 42
BOOTSTRAP_N = 1_000

TFIDF_PARAMS = dict(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)


# ---------------------------------------------------------------------------
# Helpers compartilhados
# ---------------------------------------------------------------------------

def _fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return 0.0 if denom == 0.0 else (1 + b2) * precision * recall / denom


def _scores(y_true, y_pred) -> dict[str, float]:
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    return {
        "f05":       _fbeta(p, r, 0.5),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": p,
        "recall":    r,
    }


def _svm_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   LinearSVC(C=1.0, max_iter=10_000, random_state=RANDOM_STATE)),
    ])


def _get_db() -> tuple[MongoDBClient | None, bool]:
    try:
        db = MongoDBClient()
        ok = db.ping()
        return (db, ok)
    except Exception as exc:
        logger.warning("MongoDB indisponível: %s", exc)
        return (None, False)


def _save_experiment(db: MongoDBClient | None, db_ok: bool, doc: dict) -> None:
    if not db_ok or db is None:
        return
    try:
        db.save_experiment(doc)
    except Exception as exc:
        logger.error("Erro ao salvar no MongoDB: %s", exc)


def _subset(df: pd.DataFrame, n_msgs: int) -> tuple[pd.Series, pd.Series]:
    """Retorna (X, y) para um ponto de corte específico."""
    s = df[df["n_msgs"] == n_msgs]
    return s["text"], s["label"]


# ---------------------------------------------------------------------------
# E2 — Cross-domain
# ---------------------------------------------------------------------------

def experiment_e2_cross_domain(
    pan_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Treina SVM no PAN 2012 completo e avalia no corpus sintético.

    Para cada n_msgs em N_MSGS_LIST, treina um classificador com todos
    os dados PAN disponíveis e testa no corpus sintético correspondente.
    Avalia a capacidade de generalização cross-domain do corpus sintético.

    Args:
        pan_df:       DataFrame PAN 2012 (colunas: n_msgs, text, label).
        synthetic_df: DataFrame corpus sintético (mesmas colunas).

    Returns:
        DataFrame com [n_msgs, f05, f1, precision, recall].
    """
    db, db_ok = _get_db()
    rows: list[dict] = []

    for n_msgs in N_MSGS_LIST:
        X_train, y_train = _subset(pan_df, n_msgs)
        X_test,  y_test  = _subset(synthetic_df, n_msgs)

        if X_train.empty or X_test.empty:
            logger.warning("E2: dados insuficientes para n_msgs=%d, pulando.", n_msgs)
            continue

        model = _svm_pipeline()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        m = _scores(y_test, y_pred)
        logger.info(
            "E2 n_msgs=%d | F0.5=%.4f | F1=%.4f | P=%.4f | R=%.4f",
            n_msgs, m["f05"], m["f1"], m["precision"], m["recall"],
        )

        experiment_id = f"e2_cross_domain_{n_msgs}msgs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        _save_experiment(db, db_ok, {
            "experiment_id":   experiment_id,
            "timestamp":       datetime.utcnow(),
            "experiment_type": "cross_domain",
            "n_msgs":          n_msgs,
            "classifier":      "LinearSVC",
            "results": {
                "f05":       m["f05"],
                "f1":        m["f1"],
                "precision": m["precision"],
                "recall":    m["recall"],
                "std_f05":   0.0,
            },
            "dataset_sizes": {
                "total":     int(len(y_train) + len(y_test)),
                "predatory": int((y_train == 1).sum() + (y_test == 1).sum()),
                "normal":    int((y_train == 0).sum() + (y_test == 0).sum()),
            },
        })

        rows.append({"n_msgs": n_msgs, **m})

    df_result = pd.DataFrame(rows, columns=["n_msgs", "f05", "f1", "precision", "recall"])
    logger.info("E2 concluído: %d configurações avaliadas.", len(df_result))
    return df_result


# ---------------------------------------------------------------------------
# E3 — Leave-one-out + bootstrap IC 95 %
# ---------------------------------------------------------------------------

def experiment_e3_leave_one_out(
    synthetic_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Leave-one-out CV no corpus sintético com IC 95% por bootstrap.

    O LOO é calculado em nível de conversa: cada conversa é removida
    uma vez do treino e usada como teste. O IC 95% é estimado por
    bootstrap (n=1000) sobre as predições LOO acumuladas.

    Args:
        synthetic_df: DataFrame corpus sintético (colunas: n_msgs, text, label).

    Returns:
        DataFrame com [n_msgs, f05, ic_lower, ic_upper].
    """
    db, db_ok = _get_db()
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict] = []

    for n_msgs in N_MSGS_LIST:
        X, y = _subset(synthetic_df, n_msgs)

        if len(X) < 2:
            logger.warning("E3: amostras insuficientes para n_msgs=%d, pulando.", n_msgs)
            continue

        X_arr = X.values
        y_arr = y.values

        # LOO — acumula todas as predições
        loo = LeaveOneOut()
        y_true_all: list[int] = []
        y_pred_all: list[int] = []

        for train_idx, test_idx in loo.split(X_arr):
            model = _svm_pipeline()
            model.fit(X_arr[train_idx], y_arr[train_idx])
            y_pred_all.extend(model.predict(X_arr[test_idx]).tolist())
            y_true_all.extend(y_arr[test_idx].tolist())

        y_true_arr = np.array(y_true_all)
        y_pred_arr = np.array(y_pred_all)

        # F0.5 pontual sobre todas as predições LOO
        p_point = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        r_point = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        f05_point = _fbeta(p_point, r_point)

        # Bootstrap para IC 95%
        n_samples = len(y_true_arr)
        boot_f05: list[float] = []
        for _ in range(BOOTSTRAP_N):
            idx = rng.integers(0, n_samples, size=n_samples)
            p_b = float(precision_score(y_true_arr[idx], y_pred_arr[idx], zero_division=0))
            r_b = float(recall_score(y_true_arr[idx], y_pred_arr[idx], zero_division=0))
            boot_f05.append(_fbeta(p_b, r_b))

        ic_lower = float(np.percentile(boot_f05, 2.5))
        ic_upper = float(np.percentile(boot_f05, 97.5))

        logger.info(
            "E3 n_msgs=%d | F0.5=%.4f | IC95=[%.4f, %.4f]",
            n_msgs, f05_point, ic_lower, ic_upper,
        )

        experiment_id = f"e3_loo_{n_msgs}msgs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        _save_experiment(db, db_ok, {
            "experiment_id":   experiment_id,
            "timestamp":       datetime.utcnow(),
            "experiment_type": "loo",
            "n_msgs":          n_msgs,
            "classifier":      "LinearSVC",
            "results": {
                "f05":       f05_point,
                "f1":        float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
                "precision": p_point,
                "recall":    r_point,
                "std_f05":   float(np.std(boot_f05)),
                "ic_lower":  ic_lower,
                "ic_upper":  ic_upper,
            },
            "dataset_sizes": {
                "total":     n_samples,
                "predatory": int((y_true_arr == 1).sum()),
                "normal":    int((y_true_arr == 0).sum()),
            },
        })

        rows.append({
            "n_msgs":   n_msgs,
            "f05":      f05_point,
            "ic_lower": ic_lower,
            "ic_upper": ic_upper,
        })

    df_result = pd.DataFrame(rows, columns=["n_msgs", "f05", "ic_lower", "ic_upper"])
    logger.info("E3 concluído: %d configurações avaliadas.", len(df_result))
    return df_result


# ---------------------------------------------------------------------------
# E4 — Jaccard de features TF-IDF
# ---------------------------------------------------------------------------

def experiment_e4_jaccard(
    pan_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    top_n: int = 50,
) -> tuple[float, list[str]]:
    """
    Calcula sobreposição Jaccard entre as top-N features TF-IDF de cada corpus.

    Usa o ponto de corte padrão n_msgs=10 (conforme Panzariello 2022) para
    a comparação principal, mas calcula e loga os valores para todos os pontos.

    Args:
        pan_df:       DataFrame PAN 2012.
        synthetic_df: DataFrame corpus sintético.
        top_n:        Número de features top a comparar.

    Returns:
        Tupla (jaccard_score, lista de features comuns) para n_msgs=10.
    """
    db, db_ok = _get_db()

    def _top_features(texts: pd.Series, n: int) -> set[str]:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=n)
        vec.fit(texts)
        return set(vec.get_feature_names_out())

    main_jaccard = 0.0
    main_common: list[str] = []

    for n_msgs in N_MSGS_LIST:
        pan_texts,   _ = _subset(pan_df,       n_msgs)
        synth_texts, _ = _subset(synthetic_df, n_msgs)

        if pan_texts.empty or synth_texts.empty:
            continue

        pan_feats   = _top_features(pan_texts,   top_n)
        synth_feats = _top_features(synth_texts, top_n)

        common   = pan_feats & synth_feats
        union    = pan_feats | synth_feats
        jaccard  = len(common) / len(union) if union else 0.0
        only_pan = pan_feats   - synth_feats
        only_syn = synth_feats - pan_feats

        logger.info(
            "E4 n_msgs=%d | Jaccard=%.4f | comuns=%d | só-PAN=%d | só-sintético=%d",
            n_msgs, jaccard, len(common), len(only_pan), len(only_syn),
        )

        if n_msgs == 10:
            main_jaccard = jaccard
            main_common  = sorted(common)

        experiment_id = f"e4_jaccard_{n_msgs}msgs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        _save_experiment(db, db_ok, {
            "experiment_id":   experiment_id,
            "timestamp":       datetime.utcnow(),
            "experiment_type": "jaccard",
            "n_msgs":          n_msgs,
            "classifier":      "TF-IDF",
            "results": {
                "f05":            jaccard,   # campo reutilizado como métrica principal
                "f1":             0.0,
                "precision":      0.0,
                "recall":         0.0,
                "std_f05":        0.0,
                "jaccard":        jaccard,
                "n_common":       len(common),
                "n_only_pan":     len(only_pan),
                "n_only_synth":   len(only_syn),
                "common_features": sorted(common),
                "top_n":          top_n,
            },
            "dataset_sizes": {
                "total":     int(len(pan_texts) + len(synth_texts)),
                "predatory": 0,
                "normal":    0,
            },
        })

    logger.info("E4 concluído | Jaccard principal (n_msgs=10)=%.4f", main_jaccard)
    return main_jaccard, main_common


# ---------------------------------------------------------------------------
# E5 — Data augmentation
# ---------------------------------------------------------------------------

def experiment_e5_augmentation(
    pan_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """
    Compara baseline (só PAN train) vs. augmentation (PAN train + sintético).

    Divide o PAN em treino/teste estratificado. Treina dois modelos:
    - baseline: apenas PAN train
    - augmented: PAN train + corpus sintético completo

    Avalia ambos no PAN test. Retorna delta F0.5 em pontos percentuais.

    Args:
        pan_df:       DataFrame PAN 2012.
        synthetic_df: DataFrame corpus sintético.
        test_size:    Proporção do PAN reservada para teste (default: 0.2).

    Returns:
        DataFrame com [n_msgs, f05_baseline, f05_augmented, delta_pp].
    """
    db, db_ok = _get_db()
    rows: list[dict] = []

    for n_msgs in N_MSGS_LIST:
        X_pan, y_pan = _subset(pan_df, n_msgs)
        X_syn, y_syn = _subset(synthetic_df, n_msgs)

        if X_pan.empty:
            logger.warning("E5: PAN vazio para n_msgs=%d, pulando.", n_msgs)
            continue

        # Divide PAN em treino e teste estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X_pan, y_pan,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y_pan,
        )

        # Baseline — só PAN train
        baseline_model = _svm_pipeline()
        baseline_model.fit(X_train, y_train)
        y_pred_base = baseline_model.predict(X_test)
        m_base = _scores(y_test, y_pred_base)

        # Augmentation — PAN train + sintético
        if not X_syn.empty:
            X_aug = pd.concat([X_train, X_syn], ignore_index=True)
            y_aug = pd.concat([y_train, y_syn], ignore_index=True)
        else:
            logger.warning("E5: corpus sintético vazio para n_msgs=%d; augmentation = baseline.", n_msgs)
            X_aug, y_aug = X_train, y_train

        aug_model = _svm_pipeline()
        aug_model.fit(X_aug, y_aug)
        y_pred_aug = aug_model.predict(X_test)
        m_aug = _scores(y_test, y_pred_aug)

        delta_pp = (m_aug["f05"] - m_base["f05"]) * 100.0

        logger.info(
            "E5 n_msgs=%d | F0.5 base=%.4f aug=%.4f Δ=%+.2f p.p.",
            n_msgs, m_base["f05"], m_aug["f05"], delta_pp,
        )

        for exp_type, m in [("augmentation_baseline", m_base), ("augmentation_aug", m_aug)]:
            experiment_id = f"e5_{exp_type}_{n_msgs}msgs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            _save_experiment(db, db_ok, {
                "experiment_id":   experiment_id,
                "timestamp":       datetime.utcnow(),
                "experiment_type": "augmentation",
                "n_msgs":          n_msgs,
                "classifier":      "LinearSVC",
                "results": {
                    "f05":       m["f05"],
                    "f1":        m["f1"],
                    "precision": m["precision"],
                    "recall":    m["recall"],
                    "std_f05":   0.0,
                    "variant":   exp_type,
                    "delta_pp":  delta_pp,
                },
                "dataset_sizes": {
                    "total":     int(len(y_aug)),
                    "predatory": int((y_aug == 1).sum()),
                    "normal":    int((y_aug == 0).sum()),
                },
            })

        rows.append({
            "n_msgs":          n_msgs,
            "f05_baseline":    m_base["f05"],
            "f05_augmented":   m_aug["f05"],
            "delta_pp":        delta_pp,
        })

    df_result = pd.DataFrame(
        rows, columns=["n_msgs", "f05_baseline", "f05_augmented", "delta_pp"]
    )
    logger.info("E5 concluído: %d configurações avaliadas.", len(df_result))
    return df_result


# ---------------------------------------------------------------------------
# OSAEBA — análise descritiva do survey
# ---------------------------------------------------------------------------

# Mapeamento de nomes de colunas candidatos → chave interna padronizada.
# Cobre variações de codificação que o dataset OSAEBA pode apresentar.
_OSAEBA_COL_MAP: dict[str, list[str]] = {
    # Demográficos
    "idade":         ["idade", "age", "p_idade", "q_idade"],
    "genero":        ["genero", "gênero", "sexo", "gender", "p_genero", "q_genero"],
    "raca_cor":      ["raca_cor", "raça_cor", "cor_raca", "raca", "race", "p_raca"],
    "tipo_escola":   ["tipo_escola", "escola_tipo", "tipo_de_escola", "school_type", "p_escola"],
    # Comportamento digital
    "horas_online":  ["horas_online", "tempo_online", "horas_dia", "h_online", "p_horas"],
    "dispositivo":   ["dispositivo", "device", "aparelho", "p_dispositivo"],
    "plataformas":   ["plataformas", "plataforma", "redes", "apps", "p_plataforma"],
    # Violência online
    "sofreu_violencia":    ["sofreu_violencia", "violencia_online", "foi_vitima", "p_violencia"],
    "interagiu_agressor":  ["interagiu_agressor", "interacao_agressor", "contato_agressor"],
    "forma_interacao":     ["forma_interacao", "tipo_interacao", "canal_interacao", "p_forma"],
    "agressor_desconhecido": ["agressor_desconhecido", "desconhecido", "conhecido_agressor"],
}

# Referência bibliográfica para valores esperados (usados como fallback documentado)
_OSAEBA_REF = {
    "n_respondentes":            8_436,
    "media_idade":               15.0,
    "pct_feminino":              54.7,
    "pct_masculino":             44.9,
    "pct_nao_binario":           0.4,
    "pct_pardo":                 60.0,
    "pct_branco":                21.0,
    "pct_negro":                 15.0,
    "pct_escola_publica":        95.0,
    "media_horas_online_dia":    4.0,
    "pct_celular":               93.0,
    "pct_interagiu_agressor":    20.0,
    "pct_agressor_desconhecido": 76.0,
    "forma_mais_frequente":      "videochamada",
    "fonte":                     "OSAEBA doi:10.17632/vfcrsdsmmh.4",
}

# Arquivos aceitos (em ordem de preferência)
_OSAEBA_CANDIDATES = ["RawData CSV.csv", "RawData.csv", "rawdata.csv", "RawData.xlsx"]


def _find_osaeba_file(osaeba_dir: Path) -> Path:
    """
    Localiza o arquivo de dados do OSAEBA dentro de ``osaeba_dir``.

    Raises:
        FileNotFoundError: Com mensagem orientando o download se nenhum arquivo for encontrado.
    """
    for name in _OSAEBA_CANDIDATES:
        candidate = osaeba_dir / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Nenhum arquivo OSAEBA encontrado em '{osaeba_dir}'.\n"
        f"Arquivos esperados: {_OSAEBA_CANDIDATES}\n\n"
        "Para obter o dataset:\n"
        "  1. Acesse: https://data.mendeley.com/datasets/vfcrsdsmmh/4\n"
        "  2. Baixe o arquivo 'RawData CSV.csv' ou 'RawData.xlsx'\n"
        f"  3. Coloque-o em '{osaeba_dir}/'\n\n"
        "O OSAEBA é um survey com 8436 estudantes brasileiros e NÃO contém conversas.\n"
        "É usado apenas para calibração demográfica das fichas de vítimas."
    )


def _resolve_col(df: pd.DataFrame, key: str) -> str | None:
    """Retorna o primeiro nome de coluna do DataFrame que bate com os candidatos para ``key``."""
    candidates = _OSAEBA_COL_MAP.get(key, [])
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _pct_value(series: pd.Series, values: list) -> float:
    """Retorna o percentual de linhas cujo valor está em ``values``."""
    mask = series.str.lower().isin([str(v).lower() for v in values])
    return round(100.0 * mask.sum() / len(series), 1) if len(series) else 0.0


def analyze_osaeba(osaeba_dir: str = "data/osaeba") -> dict:
    """
    Lê os CSVs/XLSX do OSAEBA e extrai estatísticas descritivas para validação das personas.

    O OSAEBA é um survey sociológico — NÃO contém conversas. Os dados são usados
    exclusivamente para confirmar que as fichas de vítimas refletem a realidade
    demográfica dos adolescentes brasileiros expostos à violência online.

    Indicadores extraídos:
    - Perfil demográfico: idade, gênero, raça/cor, tipo de escola
    - Comportamento digital: horas online, dispositivo, plataformas
    - Violência online: % com interação, formas, % agressores desconhecidos

    Se o arquivo for encontrado mas as colunas não coincidirem com os nomes esperados,
    retorna os valores de referência bibliográfica documentados e loga um aviso.

    Args:
        osaeba_dir: Diretório contendo os arquivos do OSAEBA.

    Returns:
        Dicionário com os indicadores extraídos (ou valores de referência).

    Raises:
        FileNotFoundError: Se nenhum arquivo OSAEBA for encontrado em osaeba_dir.
    """
    db, db_ok = _get_db()
    out_dir = REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    osaeba_path = _find_osaeba_file(Path(osaeba_dir))
    logger.info("OSAEBA: lendo '%s'", osaeba_path)

    # Leitura — suporta CSV e XLSX
    if osaeba_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(osaeba_path)
    else:
        # Tenta UTF-8, fallback para latin-1 (codificação comum em exports BR)
        try:
            df = pd.read_csv(osaeba_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(osaeba_path, encoding="latin-1")

    logger.info("OSAEBA: %d registros, %d colunas", len(df), len(df.columns))
    logger.debug("OSAEBA colunas: %s", list(df.columns))

    # Resolve nomes de colunas
    col = {key: _resolve_col(df, key) for key in _OSAEBA_COL_MAP}
    missing = [k for k, v in col.items() if v is None]
    if missing:
        logger.warning(
            "OSAEBA: colunas não encontradas: %s. "
            "Usando valores de referência bibliográfica para indicadores ausentes.",
            missing,
        )

    result: dict = {"n_respondentes": len(df), "fonte": _OSAEBA_REF["fonte"]}

    # ------------------------------------------------------------------
    # a) Perfil demográfico
    # ------------------------------------------------------------------

    # Idade
    if col["idade"]:
        idade_s = pd.to_numeric(df[col["idade"]], errors="coerce").dropna()
        result["media_idade"]   = round(float(idade_s.mean()), 1)
        result["mediana_idade"] = round(float(idade_s.median()), 1)
        result["dist_idade"]    = idade_s.value_counts().sort_index().to_dict()
    else:
        result["media_idade"] = _OSAEBA_REF["media_idade"]
        result["_ref_idade"]  = "valor bibliográfico (coluna não mapeada)"

    # Gênero
    if col["genero"]:
        gen_counts = df[col["genero"]].value_counts(normalize=True).mul(100).round(1)
        result["dist_genero_pct"] = gen_counts.to_dict()
        # Tenta calcular os percentuais canônicos
        gen_str = df[col["genero"]].str.lower().fillna("")
        result["pct_feminino"]   = round(100.0 * gen_str.str.contains("fem|f$|mulher").sum() / len(df), 1)
        result["pct_masculino"]  = round(100.0 * gen_str.str.contains("mas|m$|homem").sum() / len(df), 1)
    else:
        result["pct_feminino"]  = _OSAEBA_REF["pct_feminino"]
        result["pct_masculino"] = _OSAEBA_REF["pct_masculino"]
        result["_ref_genero"]   = "valor bibliográfico"

    # Raça/cor
    if col["raca_cor"]:
        raca_counts = df[col["raca_cor"]].value_counts(normalize=True).mul(100).round(1)
        result["dist_raca_pct"] = raca_counts.to_dict()
        raca_s = df[col["raca_cor"]].str.lower().fillna("")
        result["pct_pardo"]  = _pct_value(raca_s.rename("x").to_frame()["x"], ["pard", "parda", "pardo"])
        result["pct_branco"] = _pct_value(raca_s.rename("x").to_frame()["x"], ["bran", "branca", "branco"])
        result["pct_negro"]  = _pct_value(raca_s.rename("x").to_frame()["x"], ["negr", "negra", "negro"])
    else:
        result["pct_pardo"]   = _OSAEBA_REF["pct_pardo"]
        result["pct_branco"]  = _OSAEBA_REF["pct_branco"]
        result["pct_negro"]   = _OSAEBA_REF["pct_negro"]
        result["_ref_raca"]   = "valor bibliográfico"

    # Tipo de escola
    if col["tipo_escola"]:
        escola_s = df[col["tipo_escola"]].str.lower().fillna("")
        result["pct_escola_publica"] = round(
            100.0 * escola_s.str.contains("públ|publ|estadual|municipal|federal").sum() / len(df), 1
        )
    else:
        result["pct_escola_publica"] = _OSAEBA_REF["pct_escola_publica"]
        result["_ref_escola"]        = "valor bibliográfico"

    # ------------------------------------------------------------------
    # b) Comportamento digital
    # ------------------------------------------------------------------

    if col["horas_online"]:
        h_s = pd.to_numeric(df[col["horas_online"]], errors="coerce").dropna()
        result["media_horas_online_dia"]   = round(float(h_s.mean()), 1)
        result["mediana_horas_online_dia"] = round(float(h_s.median()), 1)
    else:
        result["media_horas_online_dia"] = _OSAEBA_REF["media_horas_online_dia"]
        result["_ref_horas"]             = "valor bibliográfico"

    if col["dispositivo"]:
        disp_s = df[col["dispositivo"]].str.lower().fillna("")
        result["pct_celular"] = round(
            100.0 * disp_s.str.contains("celular|smartphone|phone|móvel|movel").sum() / len(df), 1
        )
    else:
        result["pct_celular"] = _OSAEBA_REF["pct_celular"]
        result["_ref_dispositivo"] = "valor bibliográfico"

    if col["plataformas"]:
        plat_counts = df[col["plataformas"]].value_counts().head(10).to_dict()
        result["top_plataformas"] = plat_counts
    else:
        result["top_plataformas"] = {
            "WhatsApp": "1º", "Instagram": "2º", "jogos online": "3º", "Telegram": "4º"
        }
        result["_ref_plataformas"] = "valor bibliográfico (ordem de frequência OSAEBA)"

    # ------------------------------------------------------------------
    # c) Indicadores de violência online
    # ------------------------------------------------------------------

    if col["interagiu_agressor"]:
        inter_s = df[col["interagiu_agressor"]].str.lower().fillna("")
        result["pct_interagiu_agressor"] = round(
            100.0 * inter_s.str.contains("sim|yes|1|s$").sum() / len(df), 1
        )
    else:
        result["pct_interagiu_agressor"] = _OSAEBA_REF["pct_interagiu_agressor"]
        result["_ref_interacao"]         = "valor bibliográfico"

    if col["forma_interacao"]:
        forma_counts = df[col["forma_interacao"]].value_counts(normalize=True).mul(100).round(1)
        result["dist_forma_interacao_pct"] = forma_counts.head(5).to_dict()
        vcall_s = df[col["forma_interacao"]].str.lower().fillna("")
        result["pct_videochamada"] = round(
            100.0 * vcall_s.str.contains("video|vídeo|call|chamada").sum() / len(df), 1
        )
    else:
        result["pct_videochamada"]   = 22.0
        result["forma_mais_frequente"] = _OSAEBA_REF["forma_mais_frequente"]
        result["_ref_forma"]         = "valor bibliográfico"

    if col["agressor_desconhecido"]:
        desc_s = df[col["agressor_desconhecido"]].str.lower().fillna("")
        result["pct_agressor_desconhecido"] = round(
            100.0 * desc_s.str.contains("sim|des|desconhec|yes|1").sum() / len(df), 1
        )
    else:
        result["pct_agressor_desconhecido"] = _OSAEBA_REF["pct_agressor_desconhecido"]
        result["_ref_agressor"]             = "valor bibliográfico"

    # ------------------------------------------------------------------
    # Salva CSV para KNIME
    # ------------------------------------------------------------------
    profile_rows = [{"indicador": k, "valor": str(v)} for k, v in result.items()
                    if not k.startswith("dist_") and not k.startswith("top_")]
    # Adiciona distribuições como linhas separadas
    for dist_key in ["dist_idade", "dist_genero_pct", "dist_raca_pct",
                     "dist_forma_interacao_pct", "top_plataformas"]:
        if dist_key in result and isinstance(result[dist_key], dict):
            for sub_k, sub_v in result[dist_key].items():
                profile_rows.append({"indicador": f"{dist_key}.{sub_k}", "valor": str(sub_v)})

    profile_df = pd.DataFrame(profile_rows)
    csv_path = out_dir / "osaeba_profile.csv"
    profile_df.to_csv(csv_path, index=False)
    logger.info("OSAEBA: perfil salvo em '%s' (%d indicadores)", csv_path, len(profile_rows))

    # ------------------------------------------------------------------
    # Salva no MongoDB
    # ------------------------------------------------------------------
    _save_experiment(db, db_ok, {
        "experiment_id":   f"osaeba_profile_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "timestamp":       datetime.utcnow(),
        "experiment_type": "osaeba_profile",
        "n_msgs":          0,
        "classifier":      "descritiva",
        "results": {k: v for k, v in result.items()
                    if not isinstance(v, dict)},
        "dataset_sizes": {
            "total":     result.get("n_respondentes", 0),
            "predatory": 0,
            "normal":    0,
        },
    })

    logger.info("OSAEBA: análise concluída.")
    return result


# ---------------------------------------------------------------------------
# Export para KNIME
# ---------------------------------------------------------------------------

def export_to_knime(output_dir: str = "reports/") -> None:
    """
    Exporta todos os resultados do MongoDB como CSVs separados para o KNIME.

    Arquivos gerados:
        - reports/e1_baseline.csv
        - reports/e2_cross_domain.csv
        - reports/e3_loo.csv
        - reports/e4_jaccard.csv
        - reports/e5_augmentation.csv

    Args:
        output_dir: Diretório de saída (criado se não existir).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    db, db_ok = _get_db()
    if not db_ok or db is None:
        logger.error("MongoDB indisponível — export_to_knime abortado.")
        return

    export_map = {
        "e1_baseline":      ["baseline_sem_balanceamento", "baseline_undersampling"],
        "e2_cross_domain":  ["cross_domain"],
        "e3_loo":           ["loo"],
        "e4_jaccard":       ["jaccard"],
        "e5_augmentation":  ["augmentation"],
        "osaeba_profile":   ["osaeba_profile"],
    }

    for filename, exp_types in export_map.items():
        rows: list[dict] = []
        for exp_type in exp_types:
            docs = db.get_experiment_results(exp_type)
            rows.extend(docs)

        if not rows:
            logger.warning("Nenhum resultado para '%s', CSV não gerado.", filename)
            continue

        df = pd.DataFrame(rows)

        # Expande o subdocumento "results" para colunas planas
        if "results" in df.columns:
            results_expanded = pd.json_normalize(df["results"])
            df = df.drop(columns=["results"]).join(results_expanded)

        # Expande "dataset_sizes"
        if "dataset_sizes" in df.columns:
            sizes_expanded = pd.json_normalize(df["dataset_sizes"])
            df = df.drop(columns=["dataset_sizes"]).join(sizes_expanded)

        # Remove campos internos do MongoDB
        df = df.drop(columns=["_id"], errors="ignore")

        csv_path = out / f"{filename}.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")
        logger.info("Exportado: '%s' (%d linhas)", csv_path, len(df))

    logger.info("export_to_knime concluído em '%s'.", out)


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
        description="Experimentos E2-E5 — avaliação do corpus sintético"
    )
    parser.add_argument(
        "--pan",
        default="data/processed/pan2012_train.parquet",
        metavar="PARQUET",
        help="Parquet PAN 2012 (gerado por src.preprocess)",
    )
    parser.add_argument(
        "--synth",
        default="data/processed/synthetic.parquet",
        metavar="PARQUET",
        help="Parquet corpus sintético",
    )
    parser.add_argument("--all",          action="store_true", help="Executa E2, E3, E4, E5 e análise OSAEBA")
    parser.add_argument("--e2",           action="store_true", help="Executa E2 (cross-domain)")
    parser.add_argument("--e3",           action="store_true", help="Executa E3 (leave-one-out)")
    parser.add_argument("--e4",           action="store_true", help="Executa E4 (Jaccard)")
    parser.add_argument("--e5",           action="store_true", help="Executa E5 (augmentation)")
    parser.add_argument("--osaeba",       action="store_true", help="Executa análise descritiva do OSAEBA")
    parser.add_argument("--export-knime", action="store_true", help="Exporta CSVs para o KNIME")
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Top-N features para E4 Jaccard (default: 50)",
    )
    parser.add_argument(
        "--osaeba-dir",
        default="data/osaeba",
        metavar="DIR",
        help="Diretório com os arquivos do OSAEBA (default: data/osaeba)",
    )
    args = parser.parse_args()

    run_e2     = args.all or args.e2
    run_e3     = args.all or args.e3
    run_e4     = args.all or args.e4
    run_e5     = args.all or args.e5
    run_osaeba = args.all or args.osaeba

    pan_df    = pd.DataFrame()
    synth_df  = pd.DataFrame()

    needs_pan   = run_e2 or run_e4 or run_e5
    needs_synth = run_e2 or run_e3 or run_e4 or run_e5

    if needs_pan:
        pan_path = Path(args.pan)
        if not pan_path.exists():
            raise FileNotFoundError(f"PAN parquet não encontrado: {args.pan}")
        pan_df = pd.read_parquet(pan_path)
        logger.info("PAN carregado: %d linhas", len(pan_df))

    if needs_synth:
        synth_path = Path(args.synth)
        if not synth_path.exists():
            raise FileNotFoundError(f"Sintético parquet não encontrado: {args.synth}")
        synth_df = pd.read_parquet(synth_path)
        logger.info("Sintético carregado: %d linhas", len(synth_df))

    if run_e2:
        df_e2 = experiment_e2_cross_domain(pan_df, synth_df)
        print("\n=== E2 — Cross-domain ===")
        print(df_e2.to_string(index=False, float_format="%.4f"))

    if run_e3:
        df_e3 = experiment_e3_leave_one_out(synth_df)
        print("\n=== E3 — Leave-one-out ===")
        print(df_e3.to_string(index=False, float_format="%.4f"))

    if run_e4:
        score, common = experiment_e4_jaccard(pan_df, synth_df, top_n=args.top_n)
        print(f"\n=== E4 — Jaccard (n_msgs=10) ===")
        print(f"Score: {score:.4f}")
        print(f"Features comuns ({len(common)}): {', '.join(common[:20])}{'...' if len(common) > 20 else ''}")

    if run_e5:
        df_e5 = experiment_e5_augmentation(pan_df, synth_df)
        print("\n=== E5 — Augmentation ===")
        print(df_e5.to_string(index=False, float_format="%.4f"))

    if run_osaeba:
        osaeba_result = analyze_osaeba(osaeba_dir=args.osaeba_dir)
        print("\n=== OSAEBA — Perfil demográfico ===")
        for k, v in osaeba_result.items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")

    if args.export_knime:
        export_to_knime()
        print(f"\nCSVs exportados para '{REPORTS_DIR}/'")
