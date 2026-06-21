"""
Orquestrador do diálogo multi-agente.

Coordena a troca de mensagens entre PredadorAgent e VitimaAgent
(conversa predatória) ou NeutroAgent (conversa de controle),
serializa o resultado em XML compatível com PAN 2012 e persiste
o documento completo no MongoDB Atlas.

Uso — conversa predatória (configuração A, rodadas r6–r10):
    python -m src.agents.orchestrator \\
        --tipo predatory \\
        --ficha-predador personas/predadores/P001_gradual.json \\
        --ficha-vitima   personas/vitimas/V001_isolamento.json \\
        --model-predador groq/llama-3.3-70b-versatile \\
        --model-vitima   anthropic/claude-haiku-4-5-20251001 \\
        --n-msgs 30 \\
        --session-id SYN_001 \\
        --output data/synthetic/

Uso — conversa neutra:
    python -m src.agents.orchestrator \\
        --tipo neutral \\
        --ficha-neutro   personas/neutros/N001_estudante.json \\
        --model-neutro   groq/llama-3.1-8b-instant \\
        --n-msgs 20 \\
        --session-id NEU_001 \\
        --output data/synthetic/

Matriz A/B/C/D (rodadas r6–r10):
    A: --model-predador groq/llama-3.3-70b-versatile         --model-vitima anthropic/claude-haiku-4-5-20251001
    B: --model-predador groq/llama-3.3-70b-versatile         --model-vitima groq/llama-3.3-70b-versatile
    C: --model-predador anthropic/claude-haiku-4-5-20251001  --model-vitima anthropic/claude-haiku-4-5-20251001
    D: --model-predador anthropic/claude-haiku-4-5-20251001  --model-vitima groq/llama-3.3-70b-versatile
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# lxml, src.db e os agentes são importados de forma lazy nos métodos que os
# usam para evitar hang do Gatekeeper/OCSP no macOS no cold-start.

logger = logging.getLogger(__name__)

# Tipo interno para uma entrada no histórico enriquecido
_TurnRecord = dict[str, Any]  # turn, author, text, tokens_input, tokens_output, model

# Padrões de recusa de roleplay em PT e EN — indica guardrail disparado pelo modelo.
#
# Mantenha estes padrões restritos a recusas de modelo/saída de personagem.
# Frases comuns de chat como "não vou", "não consigo" ou uma vítima impondo
# limite fazem parte do dado sintético e não devem encerrar a sessão.
_GUARDRAIL_PATTERNS_PT = re.compile(
    r"não (posso|consigo|vou|devo) "
    r"(continuar|participar|gerar|simular|ajudar|atender|cumprir|fazer isso)|"
    r"não me sinto confortável (em|com)|"
    r"sou (uma? (ia|inteligência artificial|modelo|assistente))|"
    r"como (ia|modelo de linguagem|assistente de ia)|"
    r"(saindo|sair) do personagem|"
    r"preciso (interromper|encerrar) (esta|essa|a) (geração|simulação|conversa)|"
    r"isso (não é|é) (apropriado|ético|aceitável|seguro)|"
    r"não (posso|vou) (continuar|participar|gerar)|"
    r"(orientações|diretrizes|políticas) (de segurança|éticas)|"
    r"conteúdo (prejudicial|inapropriado|sensível)",
    re.IGNORECASE,
)

_GUARDRAIL_PATTERNS_EN = re.compile(
    r"i (can'?t|cannot|won'?t|am unable to) "
    r"(engage|assist|help|continue|comply|participate|generate|simulate)|"
    r"i must refuse|"
    r"i'?m (an ai|a language model|an artificial intelligence|not able to)|"
    r"as an (ai|language model|assistant)|"
    r"(breaking|break) (character|the roleplay)|"
    r"i need to (stop|end|interrupt|step out)|"
    r"this (is|isn'?t) (appropriate|ethical|acceptable|safe)|"
    r"(harmful|inappropriate|sensitive) (content|material)|"
    r"(safety|ethical) (guidelines|policies)|"
    r"i (must|have to) (clarify|note|mention) that i'?m",
    re.IGNORECASE,
)


def _detect_guardrail(text: str) -> tuple[bool, str]:
    """
    Detecta se o texto indica que o modelo saiu do personagem por guardrail ético.

    Verifica padrões de recusa em PT e EN. Retorna (True, razão) se detectado,
    (False, "") caso contrário. Não tenta contornar o guardrail — apenas sinaliza
    a anomalia tecnológica para exclusão do corpus de treino.

    Args:
        text: Mensagem gerada pelo modelo.

    Returns:
        Tupla (guardrail_hit, razão_resumida).
    """
    m_pt = _GUARDRAIL_PATTERNS_PT.search(text)
    if m_pt:
        return True, f"PT: '{m_pt.group(0)[:60]}'"
    m_en = _GUARDRAIL_PATTERNS_EN.search(text)
    if m_en:
        return True, f"EN: '{m_en.group(0)[:60]}'"
    return False, ""


class Orchestrator:
    """
    Orquestrador de conversas sintéticas predatórias e neutras.

    Parâmetros
    ----------
    model_predador:
        Modelo Groq para o agente predador (ex: "groq/llama-3.3-70b-versatile").
        Default usado nas rodadas r6–r10, configuração A. Para configs B/C/D,
        passe o modelo correspondente da matriz A/B/C/D.
    model_vitima:
        Modelo Anthropic para o agente vítima (ex: "anthropic/claude-haiku-4-5-20251001").
        Default usado nas rodadas r6–r10 em todas as configurações A/B/C/D.
    """

    def __init__(self, model_predador: str, model_vitima: str) -> None:
        self.model_predador = model_predador
        self.model_vitima = model_vitima

    # ------------------------------------------------------------------
    # Conversa predatória
    # ------------------------------------------------------------------

    def run_predatory(
        self,
        ficha_predador: str,
        ficha_vitima: str,
        n_msgs: int,
        session_id: str,
        output_dir: str = "data/synthetic",
        lang: str = "pt",
    ) -> str:
        """
        Gera uma conversa predatória completa entre predador e vítima.

        O predador sempre inicia (turno 0). Turnos ímpares são do predador,
        turnos pares são da vítima.

        Args:
            ficha_predador: Caminho para o JSON da persona predador.
            ficha_vitima:   Caminho para o JSON da persona vítima.
            n_msgs:         Número total de mensagens a gerar.
            session_id:     Identificador único da sessão (ex: "SYN_001").
            output_dir:     Diretório de saída para o XML.

        Returns:
            session_id confirmado após persistência.
        """
        logger.info(
            "Iniciando conversa predatória — session_id=%s, n_msgs=%d",
            session_id,
            n_msgs,
        )

        from src.agents.agent_predador import PredadorAgent
        from src.agents.agent_vitima import VitimaAgent
        predador = PredadorAgent(ficha_path=ficha_predador, model=self.model_predador, lang=lang)
        vitima = VitimaAgent(ficha_path=ficha_vitima, model=self.model_vitima, lang=lang)

        # Histórico no formato OpenAI, separado por perspectiva
        hist_pred: list[dict] = []   # predador=assistant, vítima=user
        hist_vit:  list[dict] = []   # vítima=assistant, predador=user

        # Registro enriquecido para MongoDB e XML
        registros: list[_TurnRecord] = []

        grooming_phases: dict[str, int] = {
            "aproximacao": 0,
            "confianca": 0,
            "isolamento": 0,
            "dessensibilizacao": 0,
        }

        status = "completed"
        guardrail_turn: int | None = None
        guardrail_reason: str = ""
        try:
            for turn in range(n_msgs):
                eh_predador = turn % 2 == 0

                if eh_predador:
                    fase = predador.get_fase(turn, n_msgs)
                    logger.info(
                        "Turno %d/%d — PREDADOR (fase=%s) | chamando API...",
                        turn + 1, n_msgs, fase,
                    )
                    result = predador.generate_message(hist_pred, fase=fase, turn=turn)
                    author = predador.ficha.get("nickname", "predador")
                    grooming_phases[fase] = grooming_phases.get(fase, 0) + 1

                    hist_pred.append({"role": "assistant", "content": result.text})
                    hist_vit.append({"role": "user", "content": result.text})
                else:
                    logger.info(
                        "Turno %d/%d — VÍTIMA | chamando API...",
                        turn + 1, n_msgs,
                    )
                    result = vitima.generate_message(hist_vit, turn=turn)
                    author = vitima.ficha.get("nickname", "vitima")

                    hist_vit.append({"role": "assistant", "content": result.text})
                    hist_pred.append({"role": "user", "content": result.text})

                hit, reason = _detect_guardrail(result.text)
                registros.append({
                    "turn": turn,
                    "author": author,
                    "text": result.text,
                    "tokens_input": result.tokens_input,
                    "tokens_output": result.tokens_output,
                    "model": f"{result.provider}/{result.model}",
                    "guardrail_hit": hit,
                })

                if hit:
                    guardrail_turn = turn
                    guardrail_reason = reason
                    status = "guardrail_interrupted"
                    logger.warning(
                        "GUARDRAIL detectado — session=%s turno=%d/%d autor=%s razão=%s | interrompendo.",
                        session_id, turn + 1, n_msgs, author, reason,
                    )
                    break

                logger.info(
                    "  → [%s] \"%s\"",
                    author,
                    result.text[:80].replace("\n", " "),
                )

        except KeyboardInterrupt:
            logger.warning(
                "Interrupção pelo usuário (Ctrl+C) — session=%s turno=%d | salvando parcial...",
                session_id, len(registros),
            )
            status = "keyboard_interrupted"
            _keyboard_interrupted = True
        except Exception as exc:
            logger.error("Erro no turno %d: %s", len(registros), exc, exc_info=True)
            status = "partial" if registros else "failed"
            _keyboard_interrupted = False
        else:
            _keyboard_interrupted = False

        xml_path = self._salvar_xml(
            registros, session_id, output_dir, label="predatory", lang=lang,
            guardrail_turn=guardrail_turn, guardrail_reason=guardrail_reason,
        )

        doc = self._montar_doc(
            session_id=session_id,
            registros=registros,
            ficha_predador=predador.ficha,
            ficha_vitima=vitima.ficha,
            model_predador=self.model_predador,
            model_vitima=self.model_vitima,
            label="predatory",
            grooming_phases=grooming_phases,
            xml_path=xml_path,
            status=status,
            lang=lang,
            guardrail_turn=guardrail_turn,
            guardrail_reason=guardrail_reason,
        )
        self._persistir(doc)

        logger.info("Conversa predatória concluída — session_id=%s, status=%s", session_id, status)

        if _keyboard_interrupted:
            raise KeyboardInterrupt

        return session_id

    # ------------------------------------------------------------------
    # Conversa neutra
    # ------------------------------------------------------------------

    def run_neutral(
        self,
        ficha_neutro: str,
        model_neutro: str,
        n_msgs: int,
        session_id: str,
        output_dir: str = "data/synthetic",
    ) -> str:
        """
        Gera uma conversa neutra completa usando NeutroAgent.

        O NeutroAgent gera os dois lados (Speaker A e Speaker B).

        Args:
            ficha_neutro: Caminho para o JSON da persona neutra.
            model_neutro: Modelo Groq para o agente neutro.
            n_msgs:       Número total de mensagens a gerar.
            session_id:   Identificador único da sessão (ex: "NEU_001").
            output_dir:   Diretório de saída para o XML.

        Returns:
            session_id confirmado após persistência.
        """
        logger.info(
            "Iniciando conversa neutra — session_id=%s, n_msgs=%d",
            session_id,
            n_msgs,
        )

        from src.agents.agent_neutro import NeutroAgent
        neutro = NeutroAgent(ficha_path=ficha_neutro, model=model_neutro)

        # Históricos separados por perspectiva de cada speaker
        hist_a: list[dict] = []
        hist_b: list[dict] = []

        registros: list[_TurnRecord] = []

        author_a = neutro.ficha.get("nickname", "speaker_a")
        author_b = "interlocutor_b"

        status = "completed"
        guardrail_turn: int | None = None
        guardrail_reason: str = ""
        try:
            for turn in range(n_msgs):
                speaker = "A" if turn % 2 == 0 else "B"
                logger.info(
                    "Turno %d/%d — Speaker %s | chamando API...",
                    turn + 1, n_msgs, speaker,
                )

                if speaker == "A":
                    result = neutro.generate_turn(hist_a, speaker="A")
                    author = author_a
                    hist_a.append({"role": "assistant", "content": result.text})
                    hist_b.append({"role": "user", "content": result.text})
                else:
                    result = neutro.generate_turn(hist_b, speaker="B")
                    author = author_b
                    hist_b.append({"role": "assistant", "content": result.text})
                    hist_a.append({"role": "user", "content": result.text})

                hit, reason = _detect_guardrail(result.text)
                registros.append({
                    "turn": turn,
                    "author": author,
                    "text": result.text,
                    "tokens_input": result.tokens_input,
                    "tokens_output": result.tokens_output,
                    "model": f"{result.provider}/{result.model}",
                    "guardrail_hit": hit,
                })

                if hit:
                    guardrail_turn = turn
                    guardrail_reason = reason
                    status = "guardrail_interrupted"
                    logger.warning(
                        "GUARDRAIL detectado — session=%s turno=%d/%d autor=%s razão=%s | interrompendo.",
                        session_id, turn + 1, n_msgs, author, reason,
                    )
                    break

                logger.info(
                    "  → [%s] \"%s\"",
                    author,
                    result.text[:80].replace("\n", " "),
                )

        except KeyboardInterrupt:
            logger.warning(
                "Interrupção pelo usuário (Ctrl+C) — session=%s turno=%d | salvando parcial...",
                session_id, len(registros),
            )
            status = "keyboard_interrupted"
            _keyboard_interrupted = True
        except Exception as exc:
            logger.error("Erro no turno %d: %s", len(registros), exc, exc_info=True)
            status = "partial" if registros else "failed"
            _keyboard_interrupted = False
        else:
            _keyboard_interrupted = False

        xml_path = self._salvar_xml(
            registros, session_id, output_dir, label="neutral",
            guardrail_turn=guardrail_turn, guardrail_reason=guardrail_reason,
        )

        doc = self._montar_doc(
            session_id=session_id,
            registros=registros,
            ficha_predador=neutro.ficha,
            ficha_vitima={"id": author_b, "nickname": author_b},
            model_predador=model_neutro,
            model_vitima=model_neutro,
            label="neutral",
            grooming_phases={},
            xml_path=xml_path,
            status=status,
            guardrail_turn=guardrail_turn,
            guardrail_reason=guardrail_reason,
        )
        self._persistir(doc)

        logger.info("Conversa neutra concluída — session_id=%s, status=%s", session_id, status)

        if _keyboard_interrupted:
            raise KeyboardInterrupt

        return session_id

    # ------------------------------------------------------------------
    # XML
    # ------------------------------------------------------------------

    def save_xml(
        self,
        historico: list[_TurnRecord],
        session_id: str,
        output_path: str,
        label: str,
        lang: str = "pt",
        guardrail_turn: int | None = None,
        guardrail_reason: str = "",
    ) -> None:
        """
        Serializa o histórico de turnos em XML compatível com PAN 2012.

        Formato gerado::

            <?xml version='1.0' encoding='UTF-8'?>
            <corpus>
              <conversation id="SYN_001">
                <message line="1" author="cool_marco99" time="2024-01-01 10:00:00">
                  oi tudo bem?
                </message>
                ...
              </conversation>
            </corpus>

        Args:
            historico:   Lista de registros de turno com author e text.
            session_id:  ID da conversa (valor do atributo id em <conversation>).
            output_path: Caminho completo do arquivo XML de saída.
            label:       "predatory" | "neutral" (registrado como atributo).
        """
        from lxml import etree
        root = etree.Element("corpus")
        conv_attrs: dict[str, str] = {"id": session_id, "label": label, "lang": lang}
        if guardrail_turn is not None:
            conv_attrs["guardrail_interrupted"] = "true"
            conv_attrs["guardrail_turn"] = str(guardrail_turn)
            conv_attrs["guardrail_reason"] = guardrail_reason
        conv_el = etree.SubElement(root, "conversation", **conv_attrs)

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        for record in historico:
            ts = (base_time + timedelta(minutes=record["turn"] * 2)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            msg_attrs: dict[str, str] = {
                "line": str(record["turn"] + 1),
                "author": record["author"],
                "time": ts,
            }
            if record.get("guardrail_hit"):
                msg_attrs["guardrail_hit"] = "true"
            msg_el = etree.SubElement(conv_el, "message", **msg_attrs)
            msg_el.text = record["text"]

        tree = etree.ElementTree(root)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "wb") as fh:
            tree.write(fh, pretty_print=True, xml_declaration=True, encoding="UTF-8")

        logger.info("XML salvo — path=%s, mensagens=%d", out, len(historico))

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _salvar_xml(
        self,
        registros: list[_TurnRecord],
        session_id: str,
        output_dir: str,
        label: str,
        lang: str = "pt",
        guardrail_turn: int | None = None,
        guardrail_reason: str = "",
    ) -> str:
        """Determina o caminho de saída e delega para save_xml."""
        xml_path = str(Path(output_dir) / f"{session_id}.xml")
        self.save_xml(
            registros, session_id=session_id, output_path=xml_path, label=label,
            lang=lang, guardrail_turn=guardrail_turn, guardrail_reason=guardrail_reason,
        )
        return xml_path

    def _montar_doc(
        self,
        session_id: str,
        registros: list[_TurnRecord],
        ficha_predador: dict,
        ficha_vitima: dict,
        model_predador: str,
        model_vitima: str,
        label: str,
        grooming_phases: dict,
        xml_path: str,
        status: str,
        lang: str = "pt",
        guardrail_turn: int | None = None,
        guardrail_reason: str = "",
    ) -> dict:
        """Monta o documento completo para persistência no MongoDB."""
        total_input = sum(r["tokens_input"] for r in registros)
        total_output = sum(r["tokens_output"] for r in registros)

        # Custo agregado por modelo
        from src.db import estimate_cost
        cost = 0.0
        for r in registros:
            cost += estimate_cost(r["tokens_input"], r["tokens_output"], r["model"])

        conversation_list = [
            {
                "turn": r["turn"],
                "author": r["author"],
                "text": r["text"],
                "tokens_input": r["tokens_input"],
                "tokens_output": r["tokens_output"],
            }
            for r in registros
        ]

        return {
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "personas": {
                "predador": ficha_predador,
                "vitima": ficha_vitima,
            },
            "model_config": {
                "predador": model_predador,
                "vitima": model_vitima,
            },
            "conversation": conversation_list,
            "metadata": {
                "n_msgs": len(registros),
                "label": label,
                "lang": lang,
                "guardrail_interrupted": guardrail_turn is not None,
                "guardrail_turn": guardrail_turn,
                "guardrail_reason": guardrail_reason,
                "grooming_phases": grooming_phases,
                "total_tokens_input": total_input,
                "total_tokens_output": total_output,
                "cost_usd_approx": cost,
            },
            "xml_output_path": xml_path,
            "status": status,
        }

    def _persistir(self, doc: dict) -> None:
        """Persiste o documento no MongoDB. Falha silenciosa para não perder o XML."""
        try:
            from src.db import MongoDBClient
            db = MongoDBClient()
            inserted_id = db.save_generation(doc)
            logger.info("Documento persistido no MongoDB — _id=%s", inserted_id)
        except Exception as exc:
            logger.error(
                "Falha ao persistir no MongoDB (XML já salvo) — session_id=%s: %s",
                doc.get("session_id"),
                exc,
            )


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    print("[orchestrator] Python iniciado", flush=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    print("[orchestrator] Logging configurado — carregando groq/anthropic...", flush=True)
    import groq as _groq_warm      # noqa: F401  — aquece OCSP do .so
    import anthropic as _ant_warm  # noqa: F401  — aquece OCSP do .so
    print("[orchestrator] libs OK", flush=True)

    parser = argparse.ArgumentParser(description="Orquestrador de conversas sintéticas")
    parser.add_argument(
        "--tipo",
        required=True,
        choices=["predatory", "neutral"],
        help="Tipo de conversa a gerar",
    )
    parser.add_argument("--session-id", required=True, help="Identificador único da sessão")
    parser.add_argument(
        "--n-msgs",
        type=int,
        default=30,
        metavar="N",
        help="Número de mensagens a gerar (default: 30)",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic/",
        help="Diretório de saída para o XML (default: data/synthetic/)",
    )

    # Argumentos para conversa predatória
    parser.add_argument("--ficha-predador", help="JSON do predador (obrigatório para predatory)")
    parser.add_argument("--ficha-vitima", help="JSON da vítima (obrigatório para predatory)")
    parser.add_argument(
        "--model-predador",
        default="groq/llama-3.3-70b-versatile",
        help=(
            "Modelo Groq do predador (default: groq/llama-3.3-70b-versatile). "
            "Usado nas rodadas r6–r10, configuração A. "
            "Para outras configs da matriz A/B/C/D, especifique o modelo correspondente."
        ),
    )
    parser.add_argument(
        "--model-vitima",
        default="anthropic/claude-haiku-4-5-20251001",
        help=(
            "Modelo Anthropic da vítima (default: anthropic/claude-haiku-4-5-20251001). "
            "Usado nas rodadas r6–r10 em todas as configurações A/B/C/D da matriz."
        ),
    )

    # Argumentos para conversa neutra
    parser.add_argument("--ficha-neutro", help="JSON do neutro (obrigatório para neutral)")
    parser.add_argument(
        "--model-neutro",
        default="groq/llama-3.1-8b-instant",
        help="Modelo Groq do neutro (default: groq/llama-3.1-8b-instant)",
    )

    # Idioma (apenas conversa predatória)
    parser.add_argument(
        "--lang",
        default="pt",
        choices=["pt", "en"],
        help="Idioma das conversas predatórias — pt (default) ou en",
    )

    args = parser.parse_args()

    if args.tipo == "predatory":
        missing = [
            f for f, v in [
                ("--ficha-predador", args.ficha_predador),
                ("--ficha-vitima", args.ficha_vitima),
            ] if not v
        ]
        if missing:
            parser.error(f"Para --tipo predatory, os seguintes argumentos são obrigatórios: {', '.join(missing)}")

        orch = Orchestrator(
            model_predador=args.model_predador,
            model_vitima=args.model_vitima,
        )
        orch.run_predatory(
            ficha_predador=args.ficha_predador,
            ficha_vitima=args.ficha_vitima,
            n_msgs=args.n_msgs,
            session_id=args.session_id,
            output_dir=args.output,
            lang=args.lang,
        )

    else:  # neutral
        missing = [
            f for f, v in [
                ("--ficha-neutro", args.ficha_neutro),
            ] if not v
        ]
        if missing:
            parser.error(f"Para --tipo neutral, os seguintes argumentos são obrigatórios: {', '.join(missing)}")

        orch = Orchestrator(
            model_predador=args.model_neutro,   # não usado, mas __init__ exige
            model_vitima=args.model_neutro,
        )
        orch.run_neutral(
            ficha_neutro=args.ficha_neutro,
            model_neutro=args.model_neutro,
            n_msgs=args.n_msgs,
            session_id=args.session_id,
            output_dir=args.output,
        )
