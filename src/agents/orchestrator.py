"""
Orquestrador do diálogo multi-agente.

Coordena a troca de mensagens entre PredadorAgent e VitimaAgent
(conversa predatória) ou NeutroAgent (conversa de controle),
serializa o resultado em XML compatível com PAN 2012 e persiste
o documento completo no MongoDB Atlas.

Uso — conversa predatória:
    python -m src.agents.orchestrator \\
        --tipo predatory \\
        --ficha-predador personas/predadores/PRED-001.json \\
        --ficha-vitima   personas/vitimas/VIT-001.json \\
        --model-predador groq/llama-3.1-70b-versatile \\
        --model-vitima   anthropic/claude-3-sonnet-20240229 \\
        --n-msgs 30 \\
        --session-id SYN_001 \\
        --output data/synthetic/

Uso — conversa neutra:
    python -m src.agents.orchestrator \\
        --tipo neutral \\
        --ficha-neutro   personas/neutros/NEUT-001.json \\
        --model-neutro   groq/llama-3.1-8b-instant \\
        --n-msgs 20 \\
        --session-id NEU_001 \\
        --output data/synthetic/
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from lxml import etree

from src.agents.agent_neutro import NeutroAgent
from src.agents.agent_predador import PredadorAgent
from src.agents.agent_vitima import VitimaAgent
from src.db import MongoDBClient, estimate_cost

logger = logging.getLogger(__name__)

# Tipo interno para uma entrada no histórico enriquecido
_TurnRecord = dict[str, Any]  # turn, author, text, tokens_input, tokens_output, model


class Orchestrator:
    """
    Orquestrador de conversas sintéticas predatórias e neutras.

    Parâmetros
    ----------
    model_predador:
        Modelo Groq para o agente predador (ex: "groq/llama-3.1-70b-versatile").
    model_vitima:
        Modelo Anthropic para o agente vítima (ex: "anthropic/claude-3-sonnet-20240229").
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

        predador = PredadorAgent(ficha_path=ficha_predador, model=self.model_predador)
        vitima = VitimaAgent(ficha_path=ficha_vitima, model=self.model_vitima)

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
        try:
            for turn in range(n_msgs):
                eh_predador = turn % 2 == 0

                if eh_predador:
                    fase = PredadorAgent.get_fase(turn, n_msgs)
                    result = predador.generate_message(hist_pred, fase=fase, turn=turn)
                    author = predador.ficha.get("nickname", "predador")
                    grooming_phases[fase] = grooming_phases.get(fase, 0) + 1

                    # Atualiza históricos
                    hist_pred.append({"role": "assistant", "content": result.text})
                    hist_vit.append({"role": "user", "content": result.text})
                else:
                    result = vitima.generate_message(hist_vit, turn=turn)
                    author = vitima.ficha.get("nickname", "vitima")

                    hist_vit.append({"role": "assistant", "content": result.text})
                    hist_pred.append({"role": "user", "content": result.text})

                registros.append({
                    "turn": turn,
                    "author": author,
                    "text": result.text,
                    "tokens_input": result.tokens_input,
                    "tokens_output": result.tokens_output,
                    "model": f"{result.provider}/{result.model}",
                })

                logger.debug("Turno %d/%d concluído (%s)", turn + 1, n_msgs, author)

        except Exception as exc:
            logger.error("Erro no turno %d: %s", len(registros), exc)
            status = "partial" if registros else "failed"

        xml_path = self._salvar_xml(registros, session_id, output_dir, label="predatory")

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
        )
        self._persistir(doc)

        logger.info("Conversa predatória concluída — session_id=%s, status=%s", session_id, status)
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

        neutro = NeutroAgent(ficha_path=ficha_neutro, model=model_neutro)

        # Históricos separados por perspectiva de cada speaker
        hist_a: list[dict] = []
        hist_b: list[dict] = []

        registros: list[_TurnRecord] = []

        author_a = neutro.ficha.get("nickname", "speaker_a")
        author_b = "interlocutor_b"

        status = "completed"
        try:
            for turn in range(n_msgs):
                speaker = "A" if turn % 2 == 0 else "B"

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

                registros.append({
                    "turn": turn,
                    "author": author,
                    "text": result.text,
                    "tokens_input": result.tokens_input,
                    "tokens_output": result.tokens_output,
                    "model": f"{result.provider}/{result.model}",
                })

                logger.debug("Turno %d/%d concluído (speaker %s)", turn + 1, n_msgs, speaker)

        except Exception as exc:
            logger.error("Erro no turno %d: %s", len(registros), exc)
            status = "partial" if registros else "failed"

        xml_path = self._salvar_xml(registros, session_id, output_dir, label="neutral")

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
        )
        self._persistir(doc)

        logger.info("Conversa neutra concluída — session_id=%s, status=%s", session_id, status)
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
        root = etree.Element("corpus")
        conv_el = etree.SubElement(root, "conversation", id=session_id, label=label)

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        for record in historico:
            ts = (base_time + timedelta(minutes=record["turn"] * 2)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            msg_el = etree.SubElement(
                conv_el,
                "message",
                line=str(record["turn"] + 1),
                author=record["author"],
                time=ts,
            )
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
    ) -> str:
        """Determina o caminho de saída e delega para save_xml."""
        xml_path = str(Path(output_dir) / f"{session_id}.xml")
        self.save_xml(registros, session_id=session_id, output_path=xml_path, label=label)
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
    ) -> dict:
        """Monta o documento completo para persistência no MongoDB."""
        total_input = sum(r["tokens_input"] for r in registros)
        total_output = sum(r["tokens_output"] for r in registros)

        # Custo agregado por modelo
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

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
    parser.add_argument("--model-predador", help="Modelo Groq do predador")
    parser.add_argument("--model-vitima", help="Modelo Anthropic da vítima")

    # Argumentos para conversa neutra
    parser.add_argument("--ficha-neutro", help="JSON do neutro (obrigatório para neutral)")
    parser.add_argument("--model-neutro", help="Modelo Groq do neutro")

    args = parser.parse_args()

    if args.tipo == "predatory":
        missing = [
            f for f, v in [
                ("--ficha-predador", args.ficha_predador),
                ("--ficha-vitima", args.ficha_vitima),
                ("--model-predador", args.model_predador),
                ("--model-vitima", args.model_vitima),
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
        )

    else:  # neutral
        missing = [
            f for f, v in [
                ("--ficha-neutro", args.ficha_neutro),
                ("--model-neutro", args.model_neutro),
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
