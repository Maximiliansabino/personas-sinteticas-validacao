"""
Testes para src/preprocess.py — limpeza de texto e pipeline PAN 2012.

Usa dados sintéticos em XML mínimo, sem dependência de arquivos PAN reais.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

lxml = pytest.importorskip("lxml", reason="lxml não disponível — testes de preprocess ignorados")

from src.preprocess import basic_clean, _parse_xml, _apply_filters  # noqa: E402


# ---------------------------------------------------------------------------
# basic_clean
# ---------------------------------------------------------------------------

def test_basic_clean_lowercase():
    assert basic_clean("HELLO WORLD") == "hello world"


def test_basic_clean_remove_url():
    result = basic_clean("visita http://example.com agora")
    assert "http" not in result
    assert "example" not in result


def test_basic_clean_remove_www_url():
    result = basic_clean("veja www.example.com aqui")
    assert "www" not in result


def test_basic_clean_normaliza_repeticoes():
    result = basic_clean("noooo lolll yessss")
    assert "noo" in result
    assert "nooo" not in result
    assert "lol" in result


def test_basic_clean_remove_stopwords():
    result = basic_clean("the cat is on the mat", stopwords={"the", "is", "on"})
    assert "the" not in result.split()
    assert "cat" in result
    assert "mat" in result


def test_basic_clean_sem_stopwords_nao_remove():
    result = basic_clean("the cat", stopwords=None)
    assert "the" in result


def test_basic_clean_texto_vazio():
    assert basic_clean("") == ""


def test_basic_clean_espacos_extras():
    result = basic_clean("  hello   world  ")
    assert result == "hello world"


# ---------------------------------------------------------------------------
# _parse_xml
# ---------------------------------------------------------------------------

_SAMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<conversations>
  <conversation id="conv-001">
    <message line="1">
      <author>alice</author>
      <time>10:00</time>
      <text>hello there</text>
    </message>
    <message line="2">
      <author>bob</author>
      <time>10:01</time>
      <text>hey alice</text>
    </message>
  </conversation>
  <conversation id="conv-002">
    <message line="1">
      <author>charlie</author>
      <time>11:00</time>
      <text>anyone here?</text>
    </message>
  </conversation>
</conversations>
"""


def test_parse_xml_numero_de_conversas():
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", encoding="utf-8", delete=False) as f:
        f.write(_SAMPLE_XML)
        path = f.name
    convs = _parse_xml(path)
    assert len(convs) == 2


def test_parse_xml_id_correto():
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", encoding="utf-8", delete=False) as f:
        f.write(_SAMPLE_XML)
        path = f.name
    convs = _parse_xml(path)
    ids = {c["id"] for c in convs}
    assert "conv-001" in ids
    assert "conv-002" in ids


def test_parse_xml_mensagens_ordenadas():
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", encoding="utf-8", delete=False) as f:
        f.write(_SAMPLE_XML)
        path = f.name
    convs = _parse_xml(path)
    conv = next(c for c in convs if c["id"] == "conv-001")
    linhas = [m["line"] for m in conv["messages"]]
    assert linhas == sorted(linhas)


def test_parse_xml_texto_das_mensagens():
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", encoding="utf-8", delete=False) as f:
        f.write(_SAMPLE_XML)
        path = f.name
    convs = _parse_xml(path)
    conv = next(c for c in convs if c["id"] == "conv-001")
    textos = [m["text"] for m in conv["messages"]]
    assert "hello there" in textos
    assert "hey alice" in textos


def test_parse_xml_arquivo_inexistente_levanta_erro():
    with pytest.raises(FileNotFoundError):
        _parse_xml("/nao/existe.xml")


# ---------------------------------------------------------------------------
# _apply_filters
# ---------------------------------------------------------------------------

def _make_conv(conv_id: str, authors_msgs: dict[str, list[str]]) -> dict:
    """Constrói um dict de conversa no formato esperado por _apply_filters."""
    messages = []
    line = 1
    for author, texts in authors_msgs.items():
        for text in texts:
            messages.append({"line": line, "author": author, "text": text})
            line += 1
    return {"id": conv_id, "messages": messages}


def test_apply_filters_mantem_conversa_valida():
    # 2 autores, cada um com 6+ msgs, textos curtos
    msgs_a = [f"hello {i}" for i in range(8)]
    msgs_b = [f"world {i}" for i in range(7)]
    conv = _make_conv("ok", {"alice": msgs_a, "bob": msgs_b})
    result = _apply_filters([conv])
    assert len(result) == 1


def test_apply_filters_remove_conversa_com_um_autor():
    msgs = [f"solo {i}" for i in range(10)]
    conv = _make_conv("mono", {"alice": msgs})
    result = _apply_filters([conv])
    assert len(result) == 0


def test_apply_filters_remove_conversa_com_poucos_msgs_por_autor():
    # alice com 5 msgs, bob com 7 — alice está abaixo de 6
    conv = _make_conv("poucos", {
        "alice": [f"a {i}" for i in range(5)],
        "bob": [f"b {i}" for i in range(7)],
    })
    result = _apply_filters([conv])
    assert len(result) == 0


def test_apply_filters_remove_mensagens_vazias():
    # alice e bob com 6+ msgs, mas algumas vazias; vazias devem ser descartadas
    conv = _make_conv("vazios", {
        "alice": ["ok"] * 4 + ["", "  "] + ["ok"] * 4,
        "bob": ["msg"] * 7,
    })
    result = _apply_filters([conv])
    # alice tem 8 msgs não-vazias → conversa deve passar
    assert len(result) == 1


def test_apply_filters_remove_msgs_com_muitos_nao_alnum():
    # Mensagem com mais de 8 caracteres não-alfanuméricos é descartada
    msgs_a = ["normal msg"] * 6
    msgs_b = ["normal ok!"] * 6
    msgs_a[0] = "!@#$%^&*()+"  # 11 não-alnum → deve ser removida
    conv = _make_conv("nao_alnum", {"alice": msgs_a, "bob": msgs_b})
    result = _apply_filters([conv])
    # a conversa pode ainda passar se cada autor tiver 5+ msgs restantes:
    # alice tem 5 válidas → abaixo de 6 → filtrada
    assert len(result) == 0
