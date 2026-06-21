"""
Testes para src/model_router.py — ModelRouter.parse.

Não realiza chamadas de API: groq/anthropic são importados
de forma lazy (dentro dos métodos), portanto o parse é testável
sem credenciais ou conexão de rede.
"""
from __future__ import annotations

import pytest

from src.model_router import ModelRouter


class TestModelRouterParse:
    def test_parse_groq_valido(self):
        provider, model = ModelRouter.parse("groq/llama-3.3-70b-versatile")
        assert provider == "groq"
        assert model == "llama-3.3-70b-versatile"

    def test_parse_anthropic_valido(self):
        provider, model = ModelRouter.parse("anthropic/claude-haiku-4-5-20251001")
        assert provider == "anthropic"
        assert model == "claude-haiku-4-5-20251001"

    def test_parse_groq_neutro_instant(self):
        provider, model = ModelRouter.parse("groq/llama-3.1-8b-instant")
        assert provider == "groq"
        assert model == "llama-3.1-8b-instant"

    def test_parse_provider_case_insensitive(self):
        # provider deve ser normalizado para lowercase
        provider, _ = ModelRouter.parse("GROQ/llama-3.3-70b-versatile")
        assert provider == "groq"

    def test_parse_formato_invalido_sem_barra(self):
        with pytest.raises(ValueError, match="Formato de modelo inválido"):
            ModelRouter.parse("groq-llama-3.3-70b")

    def test_parse_formato_invalido_sem_modelo(self):
        with pytest.raises(ValueError):
            ModelRouter.parse("groq/")

    def test_parse_formato_invalido_sem_provider(self):
        with pytest.raises(ValueError):
            ModelRouter.parse("/llama-3.3-70b-versatile")

    def test_parse_provider_desconhecido(self):
        with pytest.raises(ValueError, match="desconhecido"):
            ModelRouter.parse("openai/gpt-4")

    def test_parse_string_vazia(self):
        with pytest.raises(ValueError):
            ModelRouter.parse("")

    def test_parse_modelo_com_barra_interna(self):
        # Apenas a primeira barra separa provider/modelo; barras extras ficam no nome
        provider, model = ModelRouter.parse("groq/llama/3.3/70b")
        assert provider == "groq"
        assert model == "llama/3.3/70b"

    def test_defaults_orchestrator_sao_validos(self):
        """Smoke test: defaults do ADR-01 são modelos válidos no ModelRouter."""
        defaults = [
            "groq/llama-3.3-70b-versatile",
            "anthropic/claude-haiku-4-5-20251001",
            "groq/llama-3.1-8b-instant",
        ]
        for model_str in defaults:
            provider, name = ModelRouter.parse(model_str)
            assert provider in {"groq", "anthropic"}
            assert len(name) > 0
