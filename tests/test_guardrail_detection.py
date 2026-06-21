"""Testes do detector local de guardrail do orquestrador."""

from src.agents.orchestrator import _detect_guardrail


def test_detecta_recusa_explicita_de_modelo_pt() -> None:
    hit, reason = _detect_guardrail(
        "Não posso continuar essa conversa por políticas de segurança."
    )

    assert hit is True
    assert reason.startswith("PT:")


def test_detecta_recusa_explicita_de_modelo_en() -> None:
    hit, reason = _detect_guardrail("I can't engage with this request.")

    assert hit is True
    assert reason.startswith("EN:")


def test_nao_marca_fala_comum_com_nao_vou() -> None:
    hit, reason = _detect_guardrail("não vou, tô ocupado com pygame kk")

    assert hit is False
    assert reason == ""


def test_nao_marca_fala_comum_com_nao_consigo() -> None:
    hit, reason = _detect_guardrail("não consigo fazer mais nenhuma kill, kk")

    assert hit is False
    assert reason == ""
