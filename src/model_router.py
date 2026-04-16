"""
Roteador de modelos LLM para os agentes do pipeline multi-agente.

Abstrai a diferença de interface entre Groq e Anthropic, expondo
uma API unificada via ModelRouter.generate().

Formato do identificador de modelo: "provedor/nome-do-modelo"
  Exemplos:
    groq/llama-3.1-70b-versatile
    groq/mixtral-8x7b-32768
    anthropic/claude-3-sonnet-20240229
    anthropic/claude-3-haiku-20240307
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import anthropic
import groq
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

load_dotenv()

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {"groq", "anthropic"}

# Delay mínimo entre chamadas por provider (segundos)
_INTER_CALL_DELAY: dict[str, float] = {
    "groq": 0.5,
    "anthropic": 0.3,
}


@dataclass
class GenerationResult:
    """Resultado normalizado de uma chamada a qualquer LLM."""

    text: str
    tokens_input: int
    tokens_output: int
    model: str
    provider: str


class ModelRouter:
    """
    Roteador de modelos LLM com cache de clientes e retry automático.

    Uso::

        router = ModelRouter()
        result = router.generate(
            "groq/llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": "Olá!"}],
            system="Você é um personagem fictício.",
        )
        print(result.text, result.tokens_input, result.tokens_output)
    """

    def __init__(self) -> None:
        self._clients: dict[str, anthropic.Anthropic | groq.Groq] = {}

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    @staticmethod
    def parse(model_str: str) -> tuple[str, str]:
        """
        Separa o identificador de modelo em (provider, model_name).

        Args:
            model_str: String no formato "provider/model-name".

        Returns:
            Tupla (provider, model_name), ex: ("groq", "llama-3.1-70b-versatile").

        Raises:
            ValueError: Se o formato for inválido ou o provider desconhecido.
        """
        parts = model_str.split("/", maxsplit=1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Formato de modelo inválido: '{model_str}'. "
                "Use 'provider/model-name', ex: 'groq/llama-3.1-70b-versatile'."
            )

        provider, model_name = parts[0].lower(), parts[1]
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Provider '{provider}' desconhecido. "
                f"Suportados: {sorted(SUPPORTED_PROVIDERS)}."
            )

        return provider, model_name

    # ------------------------------------------------------------------
    # Client cache
    # ------------------------------------------------------------------

    def get_client(self, model_str: str) -> anthropic.Anthropic | groq.Groq:
        """
        Retorna (ou cria e cacheia) o cliente SDK do provider correspondente.

        Args:
            model_str: Identificador no formato "provider/model-name".

        Returns:
            Instância de anthropic.Anthropic ou groq.Groq.

        Raises:
            ValueError: Se as variáveis de ambiente de API key estiverem ausentes.
        """
        provider, _ = self.parse(model_str)

        if provider in self._clients:
            return self._clients[provider]

        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY não encontrada nas variáveis de ambiente.")
            client: anthropic.Anthropic | groq.Groq = anthropic.Anthropic(api_key=api_key)
            logger.info("Cliente Anthropic criado e cacheado")

        else:  # groq
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if not api_key:
                raise ValueError("GROQ_API_KEY não encontrada nas variáveis de ambiente.")
            client = groq.Groq(api_key=api_key)
            logger.info("Cliente Groq criado e cacheado")

        self._clients[provider] = client
        return client

    # ------------------------------------------------------------------
    # Generate (com retry interno por provider)
    # ------------------------------------------------------------------

    def generate(
        self,
        model_str: str,
        messages: list[dict],
        system: str,
        max_tokens: int = 300,
        temperature: float = 0.8,
    ) -> GenerationResult:
        """
        Chama o LLM indicado e retorna um GenerationResult normalizado.

        Aplica retry com backoff exponencial (máx 3 tentativas) em caso de
        rate limiting ou erros transientes. Aguarda o delay mínimo configurado
        por provider antes de retornar.

        Args:
            model_str:    Identificador no formato "provider/model-name".
            messages:     Lista de mensagens no formato OpenAI
                          [{"role": "user"|"assistant", "content": "..."}].
                          Não inclua a mensagem de system aqui.
            system:       Instrução de sistema (system prompt) da persona.
            max_tokens:   Limite de tokens na resposta.
            temperature:  Temperatura de amostragem.

        Returns:
            GenerationResult com text, tokens_input, tokens_output, model, provider.
        """
        provider, model_name = self.parse(model_str)
        client = self.get_client(model_str)

        if provider == "anthropic":
            result = self._generate_anthropic(
                client,  # type: ignore[arg-type]
                model_name,
                messages,
                system,
                max_tokens,
                temperature,
            )
        else:
            result = self._generate_groq(
                client,  # type: ignore[arg-type]
                model_name,
                messages,
                system,
                max_tokens,
                temperature,
            )

        time.sleep(_INTER_CALL_DELAY[provider])
        return result

    # ------------------------------------------------------------------
    # Chamadas específicas por provider
    # ------------------------------------------------------------------

    def _generate_anthropic(
        self,
        client: anthropic.Anthropic,
        model_name: str,
        messages: list[dict],
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        @retry(
            retry=retry_if_exception_type(
                (anthropic.RateLimitError, anthropic.APIStatusError)
            ),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _call() -> anthropic.types.Message:
            return client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                temperature=temperature,
            )

        response = _call()

        text = response.content[0].text if response.content else ""
        return GenerationResult(
            text=text,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            model=model_name,
            provider="anthropic",
        )

    def _generate_groq(
        self,
        client: groq.Groq,
        model_name: str,
        messages: list[dict],
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        # Groq usa a convenção OpenAI: system como primeira mensagem com role="system"
        full_messages = [{"role": "system", "content": system}, *messages]

        @retry(
            retry=retry_if_exception_type(
                (groq.RateLimitError, groq.APIStatusError)
            ),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _call() -> groq.types.chat.ChatCompletion:
            return client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        response = _call()

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage

        return GenerationResult(
            text=text,
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            model=model_name,
            provider="groq",
        )
