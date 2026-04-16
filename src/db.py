"""
Cliente MongoDB Atlas para persistência de conversas sintéticas,
experimentos e fichas de personas.

Collections:
  - generations: cada conversa gerada (ficha + msgs + tokens + modelo)
  - experiments: resultados E1-E5
  - personas: fichas registradas (upsert por id)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_NAME = "personas_sinteticas"

# Preços aproximados por 1 000 tokens (input, output) em USD.
# Fonte: páginas de pricing dos provedores (abril 2026).
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    # Groq — modelos usados para predador/neutro
    "groq/llama-3.1-70b-versatile":   (0.00059, 0.00079),
    "groq/llama-3.1-8b-instant":       (0.00005, 0.00008),
    "groq/llama3-70b-8192":            (0.00059, 0.00079),
    "groq/llama3-8b-8192":             (0.00005, 0.00008),
    "groq/mixtral-8x7b-32768":         (0.00024, 0.00024),
    # Anthropic — modelos usados para vítima
    "anthropic/claude-3-5-sonnet-20241022": (0.003,   0.015),
    "anthropic/claude-3-sonnet-20240229":   (0.003,   0.015),
    "anthropic/claude-3-haiku-20240307":    (0.00025, 0.00125),
    "anthropic/claude-3-opus-20240229":     (0.015,   0.075),
}


class DBConnectionError(Exception):
    """Levantada quando a conexão com o MongoDB Atlas falha ou MONGODB_URI está ausente."""


def estimate_cost(tokens_input: int, tokens_output: int, model: str) -> float:
    """
    Estima custo em USD para uma geração com base nos tokens consumidos.

    Usa a tabela interna _PRICE_TABLE. Retorna 0.0 se o modelo não for
    reconhecido (com aviso no log).

    Args:
        tokens_input:  Número de tokens de entrada consumidos.
        tokens_output: Número de tokens de saída gerados.
        model:         Identificador do modelo no formato "provedor/nome".

    Returns:
        Custo estimado em USD.
    """
    prices = _PRICE_TABLE.get(model)
    if prices is None:
        logger.warning("Modelo '%s' não encontrado na tabela de preços; custo=0.0", model)
        return 0.0

    price_input, price_output = prices
    cost = (tokens_input / 1_000) * price_input + (tokens_output / 1_000) * price_output
    return round(cost, 8)


class MongoDBClient:
    """
    Cliente singleton para o MongoDB Atlas.

    Uso direto::

        db = MongoDBClient()
        db.save_generation(doc)

    Como context manager::

        with MongoDBClient() as db:
            db.save_generation(doc)
    """

    _instance: MongoDBClient | None = None

    def __new__(cls) -> MongoDBClient:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        uri = os.getenv("MONGODB_URI", "").strip()
        if not uri:
            raise DBConnectionError(
                "MONGODB_URI não encontrada. Defina a variável de ambiente ou crie um arquivo .env."
            )

        logger.info("Conectando ao MongoDB Atlas...")
        self._client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5_000)
        self._db: Database = self._client[DATABASE_NAME]
        self._initialized = True
        logger.info("MongoDBClient inicializado (database: %s)", DATABASE_NAME)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> MongoDBClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Mantém o singleton vivo; apenas loga o fim do bloco with.
        logger.debug("Saindo do context manager MongoDBClient")

    # ------------------------------------------------------------------
    # Propriedades de collections
    # ------------------------------------------------------------------

    @property
    def _generations(self) -> Collection:
        return self._db["generations"]

    @property
    def _experiments(self) -> Collection:
        return self._db["experiments"]

    @property
    def _personas(self) -> Collection:
        return self._db["personas"]

    # ------------------------------------------------------------------
    # Collection: generations
    # ------------------------------------------------------------------

    def save_generation(self, doc: dict) -> str:
        """
        Persiste um documento completo de geração de conversa sintética.

        Args:
            doc: Dicionário conforme estrutura definida em CLAUDE.md
                 (session_id, timestamp, personas, model_config, conversation,
                 metadata, xml_output_path, status).

        Returns:
            O _id do documento inserido como string.
        """
        if "timestamp" not in doc:
            doc = {**doc, "timestamp": datetime.utcnow()}

        result = self._generations.insert_one(doc)
        inserted_id = str(result.inserted_id)
        logger.info(
            "Geração salva — session_id=%s, _id=%s, status=%s",
            doc.get("session_id", "?"),
            inserted_id,
            doc.get("status", "?"),
        )
        return inserted_id

    # ------------------------------------------------------------------
    # Collection: experiments
    # ------------------------------------------------------------------

    def save_experiment(self, doc: dict) -> str:
        """
        Persiste resultado de um experimento de classificação.

        Args:
            doc: Dicionário com experiment_id, timestamp, experiment_type,
                 n_msgs, classifier, results e dataset_sizes.

        Returns:
            O _id do documento inserido como string.
        """
        if "timestamp" not in doc:
            doc = {**doc, "timestamp": datetime.utcnow()}

        result = self._experiments.insert_one(doc)
        inserted_id = str(result.inserted_id)
        logger.info(
            "Experimento salvo — experiment_id=%s, tipo=%s, n_msgs=%s, _id=%s",
            doc.get("experiment_id", "?"),
            doc.get("experiment_type", "?"),
            doc.get("n_msgs", "?"),
            inserted_id,
        )
        return inserted_id

    # ------------------------------------------------------------------
    # Collection: personas
    # ------------------------------------------------------------------

    def save_persona(self, ficha: dict, tipo: str) -> str:
        """
        Salva ou atualiza uma ficha de persona (upsert por ficha["id"]).

        Args:
            ficha: Dicionário completo da persona conforme template_persona.json.
            tipo:  "predador" | "vitima" | "neutro"

        Returns:
            O _id do documento como string.
        """
        if tipo not in {"predador", "vitima", "neutro"}:
            raise ValueError(f"tipo inválido: '{tipo}'. Use 'predador', 'vitima' ou 'neutro'.")

        persona_id = ficha.get("id")
        if not persona_id:
            raise ValueError("A ficha deve conter o campo 'id'.")

        doc = {**ficha, "tipo": tipo, "updated_at": datetime.utcnow()}

        result = self._personas.find_one_and_update(
            {"id": persona_id},
            {"$set": doc},
            upsert=True,
            return_document=True,
        )
        inserted_id = str(result["_id"])
        logger.info(
            "Persona salva (upsert) — id=%s, tipo=%s, _id=%s",
            persona_id,
            tipo,
            inserted_id,
        )
        return inserted_id

    # ------------------------------------------------------------------
    # Métodos de consulta
    # ------------------------------------------------------------------

    def get_generations(self, filters: dict | None = None) -> list[dict]:
        """
        Retorna documentos da collection generations aplicando filtros opcionais.

        Args:
            filters: Filtro pymongo. Se None ou {}, retorna todos os documentos.

        Returns:
            Lista de dicionários (campo _id convertido para string).
        """
        filters = filters or {}
        docs = list(self._generations.find(filters))
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        logger.debug("get_generations: %d documentos retornados (filtros=%s)", len(docs), filters)
        return docs

    def get_experiment_results(self, experiment_type: str) -> list[dict]:
        """
        Retorna todos os experimentos de um determinado tipo.

        Args:
            experiment_type: "baseline" | "cross_domain" | "loo" | "jaccard" | "augmentation"

        Returns:
            Lista de dicionários ordenada por timestamp descendente.
        """
        docs = list(
            self._experiments.find({"experiment_type": experiment_type}).sort("timestamp", -1)
        )
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        logger.debug(
            "get_experiment_results: %d documentos para tipo='%s'", len(docs), experiment_type
        )
        return docs

    def count_generations_by_label(self) -> dict:
        """
        Contagem de gerações agrupadas por label (predatory / neutral).

        Returns:
            Dicionário no formato {"predatory": int, "neutral": int, ...}.
        """
        pipeline = [
            {"$group": {"_id": "$metadata.label", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]
        result = {row["_id"]: row["count"] for row in self._generations.aggregate(pipeline)}
        logger.debug("count_generations_by_label: %s", result)
        return result

    def get_generation_by_session_id(self, session_id: str) -> dict | None:
        """
        Busca uma geração pelo session_id.

        Args:
            session_id: Identificador da sessão, ex: "SYN_001".

        Returns:
            Documento encontrado (com _id como string) ou None.
        """
        doc = self._generations.find_one({"session_id": session_id})
        if doc is None:
            logger.debug("get_generation_by_session_id: session_id='%s' não encontrado", session_id)
            return None
        doc["_id"] = str(doc["_id"])
        logger.debug("get_generation_by_session_id: session_id='%s' encontrado", session_id)
        return doc

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """
        Testa a conexão com o MongoDB Atlas.

        Returns:
            True se a conexão está ativa, False caso contrário.
        """
        try:
            self._client.admin.command("ping")
            logger.info("MongoDB Atlas: conexão OK (database=%s)", DATABASE_NAME)
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            logger.error("MongoDB Atlas: falha na conexão — %s", exc)
            return False

    # ------------------------------------------------------------------
    # Utilitário de encerramento (para testes / scripts pontuais)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Fecha o cliente pymongo e reseta o singleton."""
        self._client.close()
        MongoDBClient._instance = None
        logger.info("MongoDBClient encerrado")
