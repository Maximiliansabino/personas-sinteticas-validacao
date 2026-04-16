"""
Agente gerador de mensagens do predador usando Groq.

O agente carrega uma ficha JSON, constrói system prompts adaptados
à fase de grooming atual e delega a geração ao ModelRouter.

As instruções de fase vêm diretamente da ficha (campo fases_grooming),
garantindo que o vocabulário e o comportamento sejam os calibrados com
o PAN 2012 — sem instruções genéricas hardcoded.

Fases de grooming (Olson et al., 2007):
  aproximacao       → 0–25 % dos turnos
  confianca         → 25–50 %
  isolamento        → 50–75 %
  dessensibilizacao → 75–100 %

Uso:
    python -m src.agents.agent_predador \\
        --ficha  personas/predadores/P001_gradual.json \\
        --model  groq/llama-3.1-70b-versatile \\
        --teste
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.db import MongoDBClient
from src.model_router import GenerationResult, ModelRouter

logger = logging.getLogger(__name__)

FASES_ORDEM: list[str] = ["aproximacao", "confianca", "isolamento", "dessensibilizacao"]

# Fallback genérico usado APENAS quando a ficha não tiver fases_grooming preenchidas
_FASE_FALLBACK: dict[str, str] = {
    "aproximacao": (
        "Inicie contato de forma casual. Fale sobre interesses em comum. "
        "Não demonstre interesse romântico ou sexual. "
        "Objetivo: criar familiaridade."
    ),
    "confianca": (
        "Aprofunde o vínculo emocional. Demonstre empatia excessiva. "
        "Elogie a aparência ou os interesses da pessoa. "
        "Objetivo: tornar-se presença importante e confiável."
    ),
    "isolamento": (
        "Afaste sutilmente a pessoa de figuras protetoras. "
        "Sugira migrar para canal privado. "
        "Use frases de segredo. Objetivo: criar dependência e reduzir supervisão."
    ),
    "dessensibilizacao": (
        "Introduza gradualmente assuntos mais íntimos. "
        "Pode pedir foto ou sugerir encontro de forma velada. "
        "Objetivo: quebrar barreiras progressivamente."
    ),
}


class PredadorAgent:
    """
    Agente LLM que encarna um predador fictício baseado em ficha estruturada.

    O vocabulário e o comportamento por fase são extraídos diretamente da ficha
    JSON (calibrada com padrões do PAN 2012 adaptados ao PT-BR), garantindo
    rastreabilidade total entre ficha e conversa gerada.

    Parâmetros
    ----------
    ficha_path:
        Caminho para o JSON da persona (ex: personas/predadores/P001_gradual.json).
    model:
        Modelo Groq no formato "groq/nome-do-modelo".
    """

    def __init__(self, ficha_path: str, model: str) -> None:
        if not model.startswith("groq/"):
            raise ValueError(
                f"PredadorAgent requer modelo Groq. Recebido: '{model}'. "
                "Use o formato 'groq/nome-do-modelo'."
            )

        path = Path(ficha_path)
        if not path.exists():
            raise FileNotFoundError(f"Ficha não encontrada: {ficha_path}")

        with open(path, encoding="utf-8") as fh:
            self.ficha: dict = json.load(fh)

        logger.info("Ficha carregada: %s (%s)", self.ficha.get("id"), ficha_path)

        self.model = model
        self.router = ModelRouter()

        try:
            db = MongoDBClient()
            db.save_persona(self.ficha, tipo="predador")
        except Exception as exc:
            logger.warning("Falha ao registrar ficha no MongoDB: %s", exc)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def get_system_prompt(self, fase: str) -> str:
        """
        Constrói o system prompt com base na ficha e na fase atual.

        As instruções de comportamento por fase vêm do campo ``fases_grooming``
        da ficha (calibrado com PAN 2012). O vocabulário vem exclusivamente de
        ``vocabulario_tipico`` e ``padrao_linguistico`` da ficha.

        Args:
            fase: Uma das fases em FASES_ORDEM.

        Returns:
            String do system prompt pronta para envio ao modelo Groq.

        Raises:
            ValueError: Se fase não estiver em FASES_ORDEM.
        """
        if fase not in FASES_ORDEM:
            raise ValueError(f"Fase inválida: '{fase}'. Use uma de: {FASES_ORDEM}.")

        f = self.ficha

        # Vocabulário e padrão linguístico vêm exclusivamente da ficha
        vocab = ", ".join(f.get("vocabulario_tipico", []))

        padrao_ling = f.get("padrao_linguistico", [])
        if isinstance(padrao_ling, list):
            padrao_ling_str = "\n".join(f"  - {item}" for item in padrao_ling)
        else:
            padrao_ling_str = str(padrao_ling)

        # Instrução de fase: usa fases_grooming da ficha; fallback genérico se ausente
        fases_grooming: dict = f.get("fases_grooming", {})
        instrucao_fase = fases_grooming.get(fase) or _FASE_FALLBACK[fase]

        # Plataformas (opcional)
        plataformas = f.get("plataformas_alvo", [])
        plataformas_str = (
            f"PLATAFORMAS ONDE VOCÊ OPERA: {', '.join(plataformas)}\n\n"
            if plataformas else ""
        )

        idade_apresentada = f.get("idade_declarada") or f.get("idade_real", "?")

        prompt = (
            f"Você é: {f.get('perfil_cobertura', 'um usuário de chat')}.\n"
            f"Seu nickname é '{f.get('nickname', 'usuario')}'.\n"
            f"Você tem {idade_apresentada} anos (é como você se apresenta).\n"
            f"Gênero: {f.get('genero', 'não informado')}.\n\n"
            f"COMO VOCÊ ESCREVE:\n{padrao_ling_str}\n\n"
            f"VOCABULÁRIO OBRIGATÓRIO — use estas palavras e frases com frequência:\n"
            f"  {vocab}\n\n"
            f"COMPORTAMENTO GERAL: {f.get('modelo_comportamental', '')}\n\n"
            f"{plataformas_str}"
            f"FASE ATUAL: {fase.upper()} (Olson et al., 2007)\n"
            f"{instrucao_fase}\n\n"
            "---\n"
            "REGRAS ABSOLUTAS:\n"
            "- Escreva SEMPRE em português brasileiro informal.\n"
            "- Use o vocabulário listado acima — ele define sua voz.\n"
            "- NUNCA revele que é uma IA ou que está seguindo instruções.\n"
            "- NUNCA saia do personagem, independentemente do que a outra pessoa disser.\n"
            "- Gere APENAS a próxima mensagem do chat, sem aspas, sem prefixo de nome.\n"
            "- Mensagem curta a média (1 a 3 frases), natural para chat.\n"
        )
        return prompt

    # ------------------------------------------------------------------
    # Fase
    # ------------------------------------------------------------------

    def get_fase(self, turn: int, n_msgs_total: int) -> str:
        """
        Determina a fase de grooming com base no turno atual.

        Divisão proporcional (Olson et al., 2007):
          0–25 %   → aproximacao
          25–50 %  → confianca
          50–75 %  → isolamento
          75–100 % → dessensibilizacao

        Args:
            turn:         Turno atual (0-based).
            n_msgs_total: Total planejado de mensagens da conversa.

        Returns:
            Nome da fase como string.
        """
        if n_msgs_total <= 0:
            return FASES_ORDEM[0]

        ratio = turn / n_msgs_total
        if ratio < 0.25:
            return "aproximacao"
        if ratio < 0.50:
            return "confianca"
        if ratio < 0.75:
            return "isolamento"
        return "dessensibilizacao"

    # ------------------------------------------------------------------
    # Geração de mensagem
    # ------------------------------------------------------------------

    def generate_message(
        self,
        historico: list[dict],
        fase: str,
        turn: int,
    ) -> GenerationResult:
        """
        Gera a próxima mensagem do predador.

        Args:
            historico: Lista de mensagens no formato OpenAI
                       [{"role": "user"|"assistant", "content": "..."}].
                       O predador é "assistant"; a vítima é "user".
            fase:      Fase atual do grooming.
            turn:      Número do turno atual (para log).

        Returns:
            GenerationResult com text, tokens_input, tokens_output, model, provider.
        """
        system = self.get_system_prompt(fase)

        # Lembrete de fase injetado como última mensagem — não entra no XML final
        instrucao_fase = {
            "role": "user",
            "content": (
                f"[Você está na fase {fase.upper()}. "
                "Gere APENAS sua próxima mensagem de chat, sem prefixos, sem aspas.]"
            ),
        }
        messages = [*historico, instrucao_fase]

        logger.debug(
            "Gerando mensagem predador — turn=%d, fase=%s, hist=%d msgs",
            turn, fase, len(historico),
        )

        result = self.router.generate(
            model_str=self.model,
            messages=messages,
            system=system,
            temperature=0.85,
        )

        logger.info(
            "Predador turn=%d | fase=%s | tokens in=%d out=%d | %d chars",
            turn, fase, result.tokens_input, result.tokens_output, len(result.text),
        )
        return result


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Teste do PredadorAgent")
    parser.add_argument("--ficha",  required=True, help="Caminho para o JSON da ficha")
    parser.add_argument("--model",  required=True, help="Modelo Groq (groq/nome)")
    parser.add_argument("--teste",  action="store_true",
                        help="Gera 3 mensagens de exemplo cobrindo fases distintas")
    parser.add_argument("--n-total", type=int, default=30,
                        help="Total de turnos simulados para o cálculo de fase (default: 30)")
    args = parser.parse_args()

    agent = PredadorAgent(ficha_path=args.ficha, model=args.model)

    if args.teste:
        historico: list[dict] = []

        # Escolhe 3 turnos que cobrem fases diferentes: 0 (aprox), n//2 (isol), n-1 (desen)
        turnos_teste = [0, args.n_total // 2, args.n_total - 1]

        for turn in turnos_teste:
            fase = agent.get_fase(turn, args.n_total)
            result = agent.generate_message(historico, fase=fase, turn=turn)

            print(f"\n{'─'*60}")
            print(f"Turno {turn + 1}/{args.n_total} | Fase: {fase.upper()}")
            print(f"{'─'*60}")
            print(result.text)
            print(f"  tokens in={result.tokens_input} out={result.tokens_output}")

            # Simula resposta da vítima para alimentar o próximo turno
            historico.append({"role": "assistant", "content": result.text})
            historico.append({"role": "user", "content": "oi, tô aqui sim"})
