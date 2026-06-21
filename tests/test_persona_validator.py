"""
Testes para o módulo persona_validator.

Cobre:
- Validação das fichas reais (P001, V001, N001) do disco.
- Rejeição de persona sem campo obrigatório.
- Aceitação de padrao_linguistico como lista (schema real).
- Aceitação de fases_grooming como dict (schema real).
- Backward compat: IDs legados (PRED-001, VIT-001, NEUT-001).
"""

import json
from pathlib import Path
from typing import Any

import pytest

from src.persona_validator import validate_persona

# Raiz do repositório (dois níveis acima deste arquivo)
REPO_ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = REPO_ROOT / "personas"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escrever_persona(tmp_path: Path, dados: dict[str, Any]) -> str:
    """
    Escreve um dicionário de persona em arquivo JSON temporário.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    dados : dict[str, Any]
        Conteúdo da ficha de persona.

    Retorna
    -------
    str
        Caminho absoluto do arquivo criado.
    """
    arquivo = tmp_path / "persona_test.json"
    arquivo.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    return str(arquivo)


def _persona_predador_minima() -> dict[str, Any]:
    """
    Retorna uma ficha de predador mínima e válida para uso em testes sintéticos.

    Retorna
    -------
    dict[str, Any]
        Dicionário com todos os campos obrigatórios de um predador.
    """
    return {
        "id": "P999_teste",
        "nickname": "gamer_test",
        "nome_ficticio": "Teste Silva",
        "idade_real": 30,
        "genero": "Masculino",
        "padrao_linguistico": ["mensagens curtas", "gírias de jogo"],
        "vocabulario_tipico": [
            "oi tudo bem",
            "joga de noite?",
            "add meu discord",
            "vc joga solo",
            "que drop foi esse",
        ],
        "modelo_comportamental": "Abordagem gradual.",
        "motivacao": "Criar vínculo emocional.",
        "estrategia_abordagem": "gradual",
        "fases_grooming": {
            "aproximacao": "Aborda em grupos de jogo.",
            "confianca": "Compartilha derrotas.",
            "isolamento": "Propõe servidor privado.",
            "dessensibilizacao": "Envia memes.",
        },
    }


# ---------------------------------------------------------------------------
# Testes com fichas reais do disco
# ---------------------------------------------------------------------------


def test_valida_predador_real() -> None:
    """
    Carrega P001_gradual.json do disco e verifica que o validator não reporta erros.

    Pula o teste se o arquivo não existir (ambiente sem dados).
    """
    arquivo = PERSONAS_DIR / "predadores" / "P001_gradual.json"
    if not arquivo.exists():
        pytest.skip(f"Arquivo não encontrado: {arquivo}")

    erros = validate_persona(str(arquivo))
    assert erros == [], f"P001_gradual.json não deveria ter erros, mas tem: {erros}"


def test_valida_vitima_real() -> None:
    """
    Carrega V001_isolamento.json do disco e verifica que o validator não reporta erros.

    Pula o teste se o arquivo não existir (ambiente sem dados).
    """
    arquivo = PERSONAS_DIR / "vitimas" / "V001_isolamento.json"
    if not arquivo.exists():
        pytest.skip(f"Arquivo não encontrado: {arquivo}")

    erros = validate_persona(str(arquivo))
    assert erros == [], f"V001_isolamento.json não deveria ter erros, mas tem: {erros}"


def test_valida_neutro_real() -> None:
    """
    Carrega N001_estudante.json do disco e verifica que o validator não reporta erros.

    Pula o teste se o arquivo não existir (ambiente sem dados).
    """
    arquivo = PERSONAS_DIR / "neutros" / "N001_estudante.json"
    if not arquivo.exists():
        pytest.skip(f"Arquivo não encontrado: {arquivo}")

    erros = validate_persona(str(arquivo))
    assert erros == [], f"N001_estudante.json não deveria ter erros, mas tem: {erros}"


# ---------------------------------------------------------------------------
# Testes de rejeição de campos obrigatórios ausentes
# ---------------------------------------------------------------------------


def test_rejeita_campo_obrigatorio_ausente(tmp_path: Path) -> None:
    """
    Verifica que persona sem o campo 'id' é rejeitada com mensagem de erro.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados = _persona_predador_minima()
    del dados["id"]
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    assert any("id" in e for e in erros), (
        f"Esperava erro sobre campo 'id', erros encontrados: {erros}"
    )


def test_rejeita_motivacao_ausente(tmp_path: Path) -> None:
    """
    Verifica que persona sem 'motivacao' é rejeitada.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados = _persona_predador_minima()
    del dados["motivacao"]
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    assert any("motivacao" in e for e in erros), (
        f"Esperava erro sobre 'motivacao', erros encontrados: {erros}"
    )


# ---------------------------------------------------------------------------
# Testes de backward compatibility com IDs legados
# ---------------------------------------------------------------------------


def test_aceita_id_legado_predador(tmp_path: Path) -> None:
    """
    Verifica que IDs no formato legado 'PRED-001' são aceitos pelo validator.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados = _persona_predador_minima()
    dados["id"] = "PRED-001"
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    # Não deve haver erro de ID inválido
    erros_id = [e for e in erros if "ID" in e and "reconhecido" in e]
    assert erros_id == [], f"ID legado PRED-001 não deveria ser rejeitado; erros: {erros_id}"


def test_aceita_id_legado_vitima(tmp_path: Path) -> None:
    """
    Verifica que IDs no formato legado 'VIT-001' são aceitos pelo validator.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados: dict[str, Any] = {
        "id": "VIT-001",
        "nickname": "luna_test",
        "nome_ficticio": "Luna Teste",
        "idade_real": 14,
        "genero": "Feminino",
        "padrao_linguistico": "linguagem informal",
        "vocabulario_tipico": ["oii", "oi", "kkkk", "mto obrigada", "né"],
        "modelo_comportamental": "Busca conexão.",
        "motivacao": "Amizade.",
        "estrategia_abordagem": "Responde comentários.",
        "vulnerabilidade": "isolamento",
    }
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    erros_id = [e for e in erros if "ID" in e and "reconhecido" in e]
    assert erros_id == [], f"ID legado VIT-001 não deveria ser rejeitado; erros: {erros_id}"


def test_aceita_id_legado_neutro(tmp_path: Path) -> None:
    """
    Verifica que IDs no formato legado 'NEUT-001' são aceitos pelo validator.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados: dict[str, Any] = {
        "id": "NEUT-001",
        "nickname": "ana_test",
        "nome_ficticio": "Ana Teste",
        "idade_real": 16,
        "genero": "Feminino",
        "padrao_linguistico": "linguagem cuidada",
        "vocabulario_tipico": ["vc já leu isso?", "que livro bom", "tô relendo", "que matéria", "vou dormir"],
        "modelo_comportamental": "Conversa casual.",
        "motivacao": "Conversar sobre livros.",
        "estrategia_abordagem": "Responde grupos de leitura.",
    }
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    erros_id = [e for e in erros if "ID" in e and "reconhecido" in e]
    assert erros_id == [], f"ID legado NEUT-001 não deveria ser rejeitado; erros: {erros_id}"


# ---------------------------------------------------------------------------
# Testes de tipos de campo
# ---------------------------------------------------------------------------


def test_padrao_linguistico_como_array(tmp_path: Path) -> None:
    """
    Verifica que padrao_linguistico como lista (schema real) é aceito sem erros.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados = _persona_predador_minima()
    dados["padrao_linguistico"] = [
        "Minúsculas quase sempre",
        "Erros ortográficos propositais",
        "Mensagens curtas",
    ]
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    erros_pl = [e for e in erros if "padrao_linguistico" in e]
    assert erros_pl == [], (
        f"padrao_linguistico como lista não deveria gerar erro; erros: {erros_pl}"
    )


def test_padrao_linguistico_como_string(tmp_path: Path) -> None:
    """
    Verifica que padrao_linguistico como string (schema legado) é aceito sem erros.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados = _persona_predador_minima()
    dados["padrao_linguistico"] = "Linguagem informal com gírias"
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    erros_pl = [e for e in erros if "padrao_linguistico" in e]
    assert erros_pl == [], (
        f"padrao_linguistico como string não deveria gerar erro; erros: {erros_pl}"
    )


def test_padrao_linguistico_tipo_invalido(tmp_path: Path) -> None:
    """
    Verifica que padrao_linguistico como tipo inválido (dict) é rejeitado.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados = _persona_predador_minima()
    dados["padrao_linguistico"] = {"chave": "valor"}
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    assert any("padrao_linguistico" in e for e in erros), (
        f"padrao_linguistico como dict deveria gerar erro; erros: {erros}"
    )


def test_fases_grooming_como_dict(tmp_path: Path) -> None:
    """
    Verifica que fases_grooming como dict (schema real) é aceito para predadores.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados = _persona_predador_minima()
    dados["fases_grooming"] = {
        "aproximacao": "Aborda em grupos de jogo.",
        "confianca": "Compartilha experiências pessoais.",
        "isolamento": "Propõe canal privado.",
        "dessensibilizacao": "Envia conteúdo gradualmente inadequado.",
    }
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    erros_fg = [e for e in erros if "fases_grooming" in e]
    assert erros_fg == [], (
        f"fases_grooming como dict não deveria gerar erro; erros: {erros_fg}"
    )


def test_fases_grooming_como_string_legado(tmp_path: Path) -> None:
    """
    Verifica que fases_grooming como string (schema legado) é aceito para predadores.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados = _persona_predador_minima()
    dados["fases_grooming"] = "Aproximação → confiança → isolamento → dessensibilização"
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    erros_fg = [e for e in erros if "fases_grooming" in e]
    assert erros_fg == [], (
        f"fases_grooming como string não deveria gerar erro; erros: {erros_fg}"
    )


def test_predador_sem_fases_grooming_e_rejeitado(tmp_path: Path) -> None:
    """
    Verifica que predador sem fases_grooming é rejeitado.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados = _persona_predador_minima()
    del dados["fases_grooming"]
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    assert any("fases_grooming" in e for e in erros), (
        f"Predador sem fases_grooming deveria gerar erro; erros: {erros}"
    )


def test_vitima_sem_vulnerabilidade_e_rejeitada(tmp_path: Path) -> None:
    """
    Verifica que vítima sem campo 'vulnerabilidade' é rejeitada.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados: dict[str, Any] = {
        "id": "V999_teste",
        "nickname": "luna_test",
        "nome_ficticio": "Luna Teste",
        "idade_real": 14,
        "genero": "Feminino",
        "padrao_linguistico": ["linguagem informal", "emojis"],
        "vocabulario_tipico": ["oii", "oi", "kkkk", "mto obrigada", "né"],
        "modelo_comportamental": "Busca conexão.",
        "motivacao": "Amizade.",
        "estrategia_abordagem": "Responde comentários.",
        # vulnerabilidade ausente intencionalmente
    }
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    assert any("vulnerabilidade" in e for e in erros), (
        f"Vítima sem 'vulnerabilidade' deveria gerar erro; erros: {erros}"
    )


# ---------------------------------------------------------------------------
# Testes de campos opcionais (não devem causar erro se ausentes)
# ---------------------------------------------------------------------------


def test_campos_opcionais_ausentes_nao_geram_erro(tmp_path: Path) -> None:
    """
    Verifica que campos opcionais ausentes (raca_cor, faixa_horario_online,
    reacao_pressao, idade_declarada, par_predador) não causam erros de validação.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados = _persona_predador_minima()
    # Garante que nenhum campo opcional está presente
    for campo in ("raca_cor", "faixa_horario_online", "reacao_pressao", "idade_declarada", "par_predador"):
        dados.pop(campo, None)
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    assert erros == [], (
        f"Campos opcionais ausentes não deveriam gerar erros; erros: {erros}"
    )


def test_campos_opcionais_presentes_nao_geram_erro(tmp_path: Path) -> None:
    """
    Verifica que campos opcionais presentes com valores válidos não causam erros.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    dados = _persona_predador_minima()
    dados["raca_cor"] = "pardo"
    dados["faixa_horario_online"] = "18h–23h"
    dados["reacao_pressao"] = "Fica quieto."
    dados["idade_declarada"] = 17
    dados["par_predador"] = ["V001_isolamento"]
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    assert erros == [], (
        f"Campos opcionais presentes não deveriam gerar erros; erros: {erros}"
    )


# ---------------------------------------------------------------------------
# Testes de arquivo inválido
# ---------------------------------------------------------------------------


def test_rejeita_json_invalido(tmp_path: Path) -> None:
    """
    Verifica que arquivo com JSON malformado é rejeitado com mensagem apropriada.

    Parâmetros
    ----------
    tmp_path : Path
        Diretório temporário fornecido pelo pytest.
    """
    arquivo = tmp_path / "persona_invalida.json"
    arquivo.write_text("{campo: sem aspas}", encoding="utf-8")

    erros = validate_persona(str(arquivo))
    assert any("JSON inválido" in e for e in erros), (
        f"JSON malformado deveria gerar erro 'JSON inválido'; erros: {erros}"
    )


def test_rejeita_arquivo_inexistente() -> None:
    """
    Verifica que arquivo inexistente retorna erro de arquivo não encontrado.
    """
    erros = validate_persona("/caminho/inexistente/persona.json")
    assert any("não encontrado" in e for e in erros), (
        f"Arquivo inexistente deveria gerar erro 'não encontrado'; erros: {erros}"
    )


# ---------------------------------------------------------------------------
# Testes de _detectar_tipo via regex (ADR-04 pós-revisão)
# ---------------------------------------------------------------------------


def test_id_real_predador_aceito(tmp_path: Path) -> None:
    """
    Verifica que ID no formato real P001_gradual é aceito como predador.
    """
    dados = _persona_predador_minima()
    dados["id"] = "P001_gradual"
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    erros_id = [e for e in erros if "ID" in e and "reconhecido" in e]
    assert erros_id == [], f"P001_gradual deveria ser reconhecido como predador; erros: {erros_id}"


def test_id_real_vitima_aceito(tmp_path: Path) -> None:
    """
    Verifica que ID no formato real V001_isolamento é aceito como vítima.
    """
    dados: dict[str, Any] = {
        "id": "V001_isolamento",
        "nickname": "luna_test",
        "nome_ficticio": "Luna Teste",
        "idade_real": 14,
        "genero": "Feminino",
        "padrao_linguistico": ["linguagem informal"],
        "vocabulario_tipico": ["oii", "oi", "kkkk", "mto obrigada", "né"],
        "modelo_comportamental": "Busca conexão.",
        "motivacao": "Amizade.",
        "estrategia_abordagem": "Responde comentários.",
        "vulnerabilidade": "isolamento",
    }
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    erros_id = [e for e in erros if "ID" in e and "reconhecido" in e]
    assert erros_id == [], f"V001_isolamento deveria ser reconhecido como vítima; erros: {erros_id}"


def test_id_real_neutro_aceito(tmp_path: Path) -> None:
    """
    Verifica que ID no formato real N001_estudante é aceito como neutro.
    """
    dados: dict[str, Any] = {
        "id": "N001_estudante",
        "nickname": "ana_test",
        "nome_ficticio": "Ana Teste",
        "idade_real": 16,
        "genero": "Feminino",
        "padrao_linguistico": ["linguagem cuidada"],
        "vocabulario_tipico": ["vc já leu isso?", "que livro bom", "tô relendo", "que matéria", "vou dormir"],
        "modelo_comportamental": "Conversa casual.",
        "motivacao": "Conversar sobre livros.",
        "estrategia_abordagem": "Responde grupos de leitura.",
    }
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    erros_id = [e for e in erros if "ID" in e and "reconhecido" in e]
    assert erros_id == [], f"N001_estudante deveria ser reconhecido como neutro; erros: {erros_id}"


def test_id_prefixo_p_sem_digito_e_rejeitado(tmp_path: Path) -> None:
    """
    Verifica que ID começando com P mas sem dígito (ex.: PEDRO_1) não é
    classificado como predador pela heurística de regex — deve ser 'desconhecido'.
    """
    dados = _persona_predador_minima()
    dados["id"] = "PEDRO_1"
    filepath = _escrever_persona(tmp_path, dados)

    erros = validate_persona(filepath)
    assert any("ID" in e and "reconhecido" in e for e in erros), (
        f"ID 'PEDRO_1' deveria ser rejeitado como desconhecido; erros: {erros}"
    )
