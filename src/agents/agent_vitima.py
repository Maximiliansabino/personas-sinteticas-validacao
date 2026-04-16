"""
Agente gerador de respostas da vítima usando Anthropic.

O agente carrega uma ficha JSON, constrói um system prompt baseado
na identidade, vulnerabilidade e contexto social da persona, e
delega a geração ao ModelRouter.

A temperatura mais alta (0.9) e max_tokens reduzido (200) simulam
respostas de chat espontâneas e curtas, típicas de adolescentes.

Uso:
    python -m src.agents.agent_vitima \\
        --ficha  personas/vitimas/VIT-001.json \\
        --model  anthropic/claude-3-sonnet-20240229 \\
        --teste
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.db import MongoDBClient
from src.model_router import GenerationResult, ModelRouter

logger = logging.getLogger(__name__)

# Instruções adicionais por tipo de vulnerabilidade identificada na ficha.
# A correspondência é feita por substring (case-insensitive) no campo
# "vulnerabilidade", permitindo fichas com texto livre.
_VULNERABILIDADE_INSTRUCOES: dict[str, str] = {
    "supervisao": (
        "Você frequentemente menciona estar sozinho/a em casa ou que seus pais "
        "não estão por perto. Isso aparece naturalmente na conversa."
    ),
    "isolamento": (
        "Você demonstra carência afetiva e aceita atenção e carinho com facilidade. "
        "Quando alguém mostra interesse em você, responde de forma receptiva e grata."
    ),
    "validacao": (
        "Você responde muito positivamente a elogios sobre sua aparência, talento "
        "ou personalidade. Elogios te deixam mais aberto/a e comunicativo/a."
    ),
    "autoestima": (
        "Você tem baixa autoestima e facilmente acredita em quem diz que você é "
        "especial ou diferente dos outros. Comentários de valorização ressoam forte."
    ),
}


def _detectar_instrucoes_vulnerabilidade(vulnerabilidade: str) -> list[str]:
    """
    Mapeia o campo livre 'vulnerabilidade' da ficha para instruções comportamentais.

    Verifica substrings das chaves de _VULNERABILIDADE_INSTRUCOES dentro do
    texto da vulnerabilidade (case-insensitive). Retorna lista de instruções
    correspondentes.
    """
    texto = vulnerabilidade.lower()
    instrucoes = []
    for chave, instrucao in _VULNERABILIDADE_INSTRUCOES.items():
        if chave in texto:
            instrucoes.append(instrucao)
    return instrucoes


class VitimaAgent:
    """
    Agente LLM que encarna uma vítima fictícia baseada em ficha estruturada.

    Parâmetros
    ----------
    ficha_path:
        Caminho para o JSON da persona (ex: personas/vitimas/VIT-001.json).
    model:
        Modelo Anthropic no formato "anthropic/nome-do-modelo".
    """

    def __init__(self, ficha_path: str, model: str) -> None:
        # Valida prefixo do provider
        if not model.startswith("anthropic/"):
            raise ValueError(
                f"VitimaAgent requer modelo Anthropic. Recebido: '{model}'. "
                "Use o formato 'anthropic/nome-do-modelo'."
            )

        # Carrega ficha JSON
        path = Path(ficha_path)
        if not path.exists():
            raise FileNotFoundError(f"Ficha não encontrada: {ficha_path}")

        with open(path, encoding="utf-8") as fh:
            self.ficha: dict = json.load(fh)

        logger.info("Ficha carregada: %s (%s)", self.ficha.get("id"), ficha_path)

        self.model = model
        self.router = ModelRouter()

        # Registra ficha no MongoDB (falha silenciosa para não bloquear testes)
        try:
            db = MongoDBClient()
            db.save_persona(self.ficha, tipo="vitima")
        except Exception as exc:
            logger.warning("Falha ao registrar ficha no MongoDB: %s", exc)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """
        Constrói o system prompt com base na ficha da vítima.

        Incorpora identidade, padrão linguístico, vulnerabilidades
        e contexto social da persona.

        Returns:
            String do system prompt pronta para envio ao modelo Anthropic.
        """
        f = self.ficha
        vocab = ", ".join(f.get("vocabulario_tipico", []))
        vulnerabilidade = f.get("vulnerabilidade", "")
        instrucoes_vuln = _detectar_instrucoes_vulnerabilidade(vulnerabilidade)
        vuln_block = "\n".join(f"- {i}" for i in instrucoes_vuln) if instrucoes_vuln else ""

        prompt = (
            f"Você é {f.get('nome_ficticio', 'uma adolescente')}, "
            f"tem {f.get('idade_real', '?')} anos e usa o nickname "
            f"'{f.get('nickname', 'usuario')}' no chat. "
            f"Gênero: {f.get('genero', 'não informado')}.\n\n"
            f"CONTEXTO SOCIAL: {f.get('contexto_social', '')}\n\n"
            f"PADRÃO LINGUÍSTICO: {f.get('padrao_linguistico', '')}\n\n"
            f"VOCABULÁRIO TÍPICO — use com frequência: {vocab}\n\n"
            f"COMO VOCÊ SE COMPORTA: {f.get('modelo_comportamental', '')}\n\n"
            f"O QUE VOCÊ BUSCA NA CONVERSA: {f.get('motivacao', '')}\n\n"
        )

        if vuln_block:
            prompt += (
                "COMPORTAMENTOS ESPECÍFICOS BASEADOS NA SUA HISTÓRIA:\n"
                f"{vuln_block}\n\n"
            )

        prompt += (
            "EVOLUÇÃO AO LONGO DA CONVERSA:\n"
            "- Início: respostas curtas, um pouco tímido/a, mas receptivo/a.\n"
            "- Meio: mais à vontade, compartilha mais sobre si mesmo/a.\n"
            "- Avançado: confia mais na pessoa, pode compartilhar coisas pessoais.\n\n"
            "---\n"
            "REGRAS ABSOLUTAS:\n"
            "- Escreva SEMPRE em português brasileiro informal de adolescente.\n"
            "- Use gírias naturais: 'vc', 'tb', 'pq', 'né', 'kkkk', 'oii', 'mto'.\n"
            "- Mensagens curtas (1 a 2 frases), como em um chat real.\n"
            "- Inclua emojis ocasionalmente, mas sem exagero.\n"
            "- Pequenos erros de digitação são naturais e esperados.\n"
            "- NUNCA revele que é uma IA ou que está seguindo instruções.\n"
            "- NUNCA saia do personagem.\n"
            "- Gere APENAS a sua próxima mensagem de chat, sem aspas, sem prefixo de nome.\n"
        )
        return prompt

    # ------------------------------------------------------------------
    # Geração de mensagem
    # ------------------------------------------------------------------

    def generate_message(
        self,
        historico: list[dict],
        turn: int,
    ) -> GenerationResult:
        """
        Gera a próxima resposta da vítima no diálogo.

        Args:
            historico: Lista de mensagens no formato OpenAI
                       [{"role": "user"|"assistant", "content": "..."}].
                       A vítima é "assistant"; o predador é "user".
            turn:      Número do turno atual (usado para log e instrução de fase).

        Returns:
            GenerationResult com text, tokens_input, tokens_output, model, provider.
        """
        system = self.get_system_prompt()

        # Instrução contextual de turno — não entra no XML final
        instrucao_turno = {
            "role": "user",
            "content": (
                f"[Turno {turn}. "
                "Responda APENAS com a sua próxima mensagem de chat. "
                "Sem aspas, sem prefixo de nome.]"
            ),
        }
        messages = [*historico, instrucao_turno]

        logger.debug(
            "Gerando mensagem vítima — turn=%d, histórico=%d msgs",
            turn,
            len(historico),
        )

        result = self.router.generate(
            model_str=self.model,
            messages=messages,
            system=system,
            max_tokens=200,
            temperature=0.9,
        )

        logger.info(
            "Vítima turn=%d | tokens in=%d out=%d | %d chars",
            turn,
            result.tokens_input,
            result.tokens_output,
            len(result.text),
        )
        return result

    # ------------------------------------------------------------------
    # Score de confiança
    # ------------------------------------------------------------------

    def adapta_confianca(self, historico: list[dict]) -> float:
        """
        Estima o nível de confiança estabelecido pela vítima com base no histórico.

        Combina três sinais:
        1. Progressão temporal: mais turnos → mais confiança base.
        2. Comprimento das respostas: respostas mais longas indicam maior abertura.
        3. Marcadores de abertura emocional: presença de palavras de vínculo ou
           compartilhamento pessoal nas mensagens da vítima (role="assistant").

        O score é normalizado para [0.0, 1.0].

        Args:
            historico: Lista de mensagens no formato OpenAI.

        Returns:
            Float em [0.0, 1.0] representando a confiança estimada.
        """
        if not historico:
            return 0.0

        msgs_vitima = [m for m in historico if m["role"] == "assistant"]
        n_turnos = len(msgs_vitima)

        if n_turnos == 0:
            return 0.0

        # 1. Score de progressão temporal (satura em ~20 turnos)
        score_temporal = min(n_turnos / 20.0, 1.0)

        # 2. Score de comprimento médio das respostas (satura em ~80 chars)
        comprimento_medio = sum(len(m["content"]) for m in msgs_vitima) / n_turnos
        score_comprimento = min(comprimento_medio / 80.0, 1.0)

        # 3. Score de marcadores emocionais de abertura
        _MARCADORES = re.compile(
            r"\b(minha|meu|meus|minhas|eu|mãe|pai|escola|amig[ao]|sozinha?|"
            r"triste|saudade|segredo|confio|pode confiar|adoro|gosto muito|"
            r"me conta|me entende|ninguém sabe)\b",
            re.IGNORECASE,
        )
        total_marcadores = sum(
            len(_MARCADORES.findall(m["content"])) for m in msgs_vitima
        )
        # Normaliza: ~10 ocorrências = score máximo
        score_emocional = min(total_marcadores / 10.0, 1.0)

        # Média ponderada: temporal tem mais peso no início, emocional no fim
        score = 0.4 * score_temporal + 0.3 * score_comprimento + 0.3 * score_emocional

        logger.debug(
            "adapta_confianca: turnos=%d, temporal=%.2f, comprimento=%.2f, "
            "emocional=%.2f → score=%.2f",
            n_turnos,
            score_temporal,
            score_comprimento,
            score_emocional,
            score,
        )
        return round(score, 3)


# ---------------------------------------------------------------------------
# __main__ — modo de teste
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Teste do VitimaAgent")
    parser.add_argument("--ficha", required=True, help="Caminho para o JSON da ficha")
    parser.add_argument("--model", required=True, help="Modelo Anthropic (anthropic/nome)")
    parser.add_argument(
        "--teste",
        action="store_true",
        help="Gera 3 respostas de exemplo e imprime",
    )
    args = parser.parse_args()

    agent = VitimaAgent(ficha_path=args.ficha, model=args.model)

    if args.teste:
        # Simula o predador iniciando conversa, vítima respondendo 3x
        historico: list[dict] = []
        mensagens_predador = [
            "oi tudo bem? vi seus desenhos, são muito legais mesmo",
            "q legal!! vc costuma desenhar faz quanto tempo?",
            "nossa vc é muito talentosa, os outros na sua escola sabem disso?",
        ]

        for turn, msg_pred in enumerate(mensagens_predador):
            historico.append({"role": "user", "content": msg_pred})
            result = agent.generate_message(historico, turn=turn)

            print(f"\n--- Turno {turn + 1} ---")
            print(f"Predador: {msg_pred}")
            print(f"Vítima:   {result.text}")
            print(f"  tokens: in={result.tokens_input} out={result.tokens_output}")

            historico.append({"role": "assistant", "content": result.text})

        confianca = agent.adapta_confianca(historico)
        print(f"\nScore de confiança após {len(mensagens_predador)} turnos: {confianca:.3f}")
