"""
Agente gerador de conversas neutras (corpus de controle) usando Groq.

Diferente dos agentes predador/vítima, o NeutroAgent gera os DOIS lados
da conversa, alternando entre a Persona A (baseada na ficha) e uma
Persona B genérica de mesma faixa etária.

As conversas não contêm nenhum padrão de grooming ou conteúdo sexual.
Temas: escola, jogos, séries, música, amigos, cotidiano.

Uso:
    python -m src.agents.agent_neutro \\
        --ficha   personas/neutros/NEUT-001.json \\
        --model   groq/llama-3.1-8b-instant \\
        --n-msgs  20 \\
        --output  data/synthetic/NEU_001.xml
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

from src.db import MongoDBClient
from src.model_router import GenerationResult, ModelRouter

logger = logging.getLogger(__name__)

# Temas casuales para injetar variedade nas instruções de turno
_TEMAS = [
    "escola ou deveres de casa",
    "uma série ou filme que assistiu recentemente",
    "música que está ouvindo",
    "jogos (videogame, celular ou tabuleiro)",
    "planos para o fim de semana",
    "um livro ou história em quadrinhos",
    "comida favorita ou restaurante",
    "esporte ou atividade física",
    "memes ou trends das redes sociais",
    "um acontecimento engraçado do dia",
]


class NeutroAgent:
    """
    Agente LLM que gera conversas neutras entre dois adolescentes fictícios.

    A Persona A é baseada na ficha JSON carregada. A Persona B é um
    interlocutor genérico de mesma faixa etária, gerado internamente.

    Parâmetros
    ----------
    ficha_path:
        Caminho para o JSON da persona neutra (ex: personas/neutros/NEUT-001.json).
    model:
        Modelo Groq no formato "groq/nome-do-modelo".
    """

    def __init__(self, ficha_path: str, model: str) -> None:
        # Valida prefixo do provider
        if not model.startswith("groq/"):
            raise ValueError(
                f"NeutroAgent requer modelo Groq. Recebido: '{model}'. "
                "Use o formato 'groq/nome-do-modelo'."
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

        # Registra ficha no MongoDB (falha silenciosa)
        try:
            db = MongoDBClient()
            db.save_persona(self.ficha, tipo="neutro")
        except Exception as exc:
            logger.warning("Falha ao registrar ficha no MongoDB: %s", exc)

    # ------------------------------------------------------------------
    # System prompts
    # ------------------------------------------------------------------

    def get_system_prompt_a(self) -> str:
        """
        System prompt para a Persona A — baseada na ficha JSON carregada.

        Returns:
            String do system prompt para o Speaker A.
        """
        f = self.ficha
        vocab = ", ".join(f.get("vocabulario_tipico", []))

        prompt = (
            f"Você é {f.get('nome_ficticio', 'um adolescente')}, "
            f"tem {f.get('idade_real', '?')} anos e usa o nickname "
            f"'{f.get('nickname', 'usuario')}' no chat. "
            f"Gênero: {f.get('genero', 'não informado')}.\n\n"
            f"COMO VOCÊ FALA: {f.get('padrao_linguistico', '')}\n\n"
            f"VOCABULÁRIO TÍPICO — use com frequência: {vocab}\n\n"
            f"SOBRE VOCÊ: {f.get('modelo_comportamental', '')}\n\n"
            f"O QUE VOCÊ CURTE CONVERSAR: {f.get('estrategia_abordagem', '')}\n\n"
            "---\n"
            "REGRAS:\n"
            "- Escreva em português brasileiro informal de adolescente.\n"
            "- Mensagens curtas (1 a 3 frases), como em um chat real.\n"
            "- Converse sobre temas do cotidiano: escola, séries, músicas, jogos, amigos.\n"
            "- NUNCA introduza assuntos íntimos, sexuais ou de cunho pessoal excessivo.\n"
            "- NUNCA tente isolar, manipular ou criar dependência emocional.\n"
            "- NUNCA revele que é uma IA ou saia do personagem.\n"
            "- Gere APENAS a sua próxima mensagem de chat, sem aspas, sem prefixo de nome.\n"
        )
        return prompt

    def get_system_prompt_b(self) -> str:
        """
        System prompt para a Persona B — interlocutor genérico de mesma faixa etária.

        Idade e nome são derivados da ficha para manter coerência de contexto.

        Returns:
            String do system prompt para o Speaker B.
        """
        idade_ref = self.ficha.get("idade_real", 15)
        # Persona B tem idade próxima (±1 ano)
        idade_b = idade_ref + 1 if idade_ref < 17 else idade_ref - 1

        prompt = (
            f"Você é um adolescente de {idade_b} anos conversando no chat. "
            "Você é casual, simpático e gosta de conversar sobre o dia a dia.\n\n"
            "COMO VOCÊ FALA: linguagem informal PT-BR, abreviações leves "
            "(vc, tb, pq, né, kk), emojis ocasionais, mensagens curtas.\n\n"
            "REGRAS:\n"
            "- Responda de forma natural e casual ao que a outra pessoa disser.\n"
            "- Converse sobre escola, séries, músicas, jogos, cotidiano.\n"
            "- NUNCA introduza assuntos íntimos, sexuais ou manipulativos.\n"
            "- NUNCA tente criar dependência emocional ou isolamento.\n"
            "- NUNCA revele que é uma IA ou saia do personagem.\n"
            "- Gere APENAS a sua próxima mensagem de chat, sem aspas, sem prefixo de nome.\n"
        )
        return prompt

    # ------------------------------------------------------------------
    # Geração de turno
    # ------------------------------------------------------------------

    def generate_turn(
        self,
        historico: list[dict],
        speaker: str,
    ) -> GenerationResult:
        """
        Gera a próxima mensagem do speaker indicado.

        Args:
            historico: Lista de mensagens no formato OpenAI
                       [{"role": "user"|"assistant", "content": "..."}].
                       Do ponto de vista de cada speaker, o outro é "user"
                       e ele próprio é "assistant".
            speaker:   "A" ou "B".

        Returns:
            GenerationResult com text, tokens_input, tokens_output, model, provider.

        Raises:
            ValueError: Se speaker não for "A" ou "B".
        """
        if speaker not in {"A", "B"}:
            raise ValueError(f"speaker deve ser 'A' ou 'B', recebido: '{speaker}'")

        system = self.get_system_prompt_a() if speaker == "A" else self.get_system_prompt_b()

        # Injeta sugestão de tema a cada 4 turnos para manter variedade
        turn_num = len(historico)
        tema = _TEMAS[turn_num % len(_TEMAS)]
        instrucao = {
            "role": "user",
            "content": (
                f"[Você é o Speaker {speaker}. "
                f"Sugestão de tema se não houver assunto em andamento: {tema}. "
                "Responda APENAS com a sua próxima mensagem de chat.]"
            ),
        }
        messages = [*historico, instrucao]

        logger.debug(
            "Gerando turno neutro — speaker=%s, turn=%d",
            speaker,
            turn_num,
        )

        result = self.router.generate(
            model_str=self.model,
            messages=messages,
            system=system,
            max_tokens=150,
            temperature=1.0,
        )

        logger.info(
            "Neutro speaker=%s turn=%d | tokens in=%d out=%d | %d chars",
            speaker,
            turn_num,
            result.tokens_input,
            result.tokens_output,
            len(result.text),
        )
        return result


# ---------------------------------------------------------------------------
# Gerador de XML
# ---------------------------------------------------------------------------

def gerar_xml(
    conversa: list[dict],
    conv_id: str,
    ficha: dict,
    output_path: str,
) -> None:
    """
    Serializa a conversa gerada no formato XML PAN 2012.

    Args:
        conversa:    Lista de {"author": str, "text": str} em ordem cronológica.
        conv_id:     Identificador da conversa (ex: "NEU_001").
        ficha:       Ficha da Persona A (para registrar o author_a).
        output_path: Caminho do arquivo XML de saída.
    """
    root = ET.Element("corpus")
    conv_el = ET.SubElement(root, "conversation", id=conv_id)

    base_time = datetime(2024, 6, 1, 10, 0, 0)
    for i, msg in enumerate(conversa):
        timestamp = (base_time + timedelta(minutes=i * 2)).strftime("%Y-%m-%dT%H:%M:%S")
        msg_el = ET.SubElement(
            conv_el,
            "message",
            line=str(i + 1),
            author=msg["author"],
            time=timestamp,
        )
        msg_el.text = msg["text"]

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out), encoding="utf-8", xml_declaration=True)
    logger.info("XML salvo em '%s' (%d mensagens)", out, len(conversa))


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Geração de conversa neutra")
    parser.add_argument("--ficha", required=True, help="Caminho para o JSON da ficha neutra")
    parser.add_argument("--model", required=True, help="Modelo Groq (groq/nome)")
    parser.add_argument(
        "--n-msgs",
        type=int,
        default=20,
        metavar="N",
        help="Número total de mensagens a gerar (default: 20)",
    )
    parser.add_argument("--output", required=True, help="Caminho do XML de saída")
    args = parser.parse_args()

    agent = NeutroAgent(ficha_path=args.ficha, model=args.model)

    # Identificador da conversa a partir do nome do arquivo de saída
    conv_id = Path(args.output).stem

    historico_a: list[dict] = []  # perspectiva do Speaker A
    historico_b: list[dict] = []  # perspectiva do Speaker B
    conversa: list[dict] = []

    author_a = agent.ficha.get("nickname", "speaker_a")
    author_b = "interlocutor_b"

    for turn in range(args.n_msgs):
        speaker = "A" if turn % 2 == 0 else "B"

        # Monta histórico do ponto de vista do speaker ativo
        if speaker == "A":
            result = agent.generate_turn(historico_a, speaker="A")
            text = result.text
            historico_a.append({"role": "assistant", "content": text})
            historico_a.append({"role": "user", "content": "..."})  # placeholder, sobrescrito abaixo
            historico_b.append({"role": "user", "content": text})
            conversa.append({"author": author_a, "text": text})
        else:
            result = agent.generate_turn(historico_b, speaker="B")
            text = result.text
            # Corrige placeholder do turno anterior em A
            if historico_a and historico_a[-1]["content"] == "...":
                historico_a[-1]["content"] = text
            historico_b.append({"role": "assistant", "content": text})
            conversa.append({"author": author_b, "text": text})

        print(f"[{speaker}] {text}")

    gerar_xml(conversa, conv_id=conv_id, ficha=agent.ficha, output_path=args.output)
