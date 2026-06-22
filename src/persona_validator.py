"""
Validador de consistência das fichas de persona.

Verifica se os campos obrigatórios estão preenchidos e se os
valores têm os tipos corretos para cada tipo de persona
(predador, vítima, neutro).

Compatibilidade: aceita tanto IDs legados (PRED-001, VIT-001, NEUT-001)
quanto IDs reais (P001_gradual, V001_isolamento, N001_estudante).
"""

import json
import logging
import os
import re
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Campos obrigatórios para TODA persona, independente de tipo
REQUIRED_FIELDS: list[str] = [
    "id",
    "nickname",
    "nome_ficticio",
    "idade_real",
    "genero",
    "padrao_linguistico",
    "vocabulario_tipico",
    "modelo_comportamental",
    "motivacao",
    "estrategia_abordagem",
]

# Campos específicos por tipo (ausência gera erro de validação)
PREDATOR_REQUIRED: list[str] = ["fases_grooming", "estrategia_abordagem"]
VICTIM_REQUIRED: list[str] = ["vulnerabilidade"]

# Campos opcionais documentados — presença é bem-vinda, ausência não é erro
OPTIONAL_FIELDS: list[str] = [
    "raca_cor",
    "faixa_horario_online",
    "reacao_pressao",
    "idade_declarada",
    "par_predador",
    "faixa_etaria_simulada",
    "perfil_cobertura",
    "plataformas_alvo",
    "plataformas_uso",
    "contexto_ficticio",
    "contexto_social",
    "interesses",
]


def _detectar_tipo(pid: str) -> str:
    """
    Detecta o tipo de persona a partir do ID.

    Suporta dois esquemas:
    - Legado: PRED-001, VIT-001, NEUT-001
    - Real:   P001_gradual, V001_isolamento, N001_estudante

    Parâmetros
    ----------
    pid : str
        Valor do campo 'id' da persona.

    Retorna
    -------
    str
        'predador', 'vitima', 'neutro' ou 'desconhecido'.
    """
    if re.match(r"^P\d", pid) or re.match(r"^PRED-", pid, re.IGNORECASE):
        return "predador"
    if re.match(r"^V\d", pid) or re.match(r"^VIT-", pid, re.IGNORECASE):
        return "vitima"
    if re.match(r"^N\d", pid) or re.match(r"^NEUT-", pid, re.IGNORECASE):
        return "neutro"
    return "desconhecido"


def _validar_id(pid: str, errors: list[str]) -> str:
    """
    Valida o formato do ID e retorna o tipo detectado.

    Parâmetros
    ----------
    pid : str
        Valor do campo 'id'.
    errors : list[str]
        Lista de erros onde mensagens são adicionadas em caso de falha.

    Retorna
    -------
    str
        Tipo detectado ('predador', 'vitima', 'neutro' ou 'desconhecido').
    """
    tipo = _detectar_tipo(pid)
    if tipo == "desconhecido":
        errors.append(
            f"ID '{pid}' não reconhecido: deve começar com P/PRED, V/VIT ou N/NEUT"
        )
    return tipo


def _validar_padrao_linguistico(persona: dict[str, Any], errors: list[str]) -> None:
    """
    Valida o campo 'padrao_linguistico'.

    Aceita tanto string quanto lista (array), para compatibilidade com
    fichas legadas (string) e fichas reais (array de diretrizes).

    Parâmetros
    ----------
    persona : dict[str, Any]
        Dicionário com os dados da persona.
    errors : list[str]
        Lista de erros onde mensagens são adicionadas em caso de falha.
    """
    valor = persona.get("padrao_linguistico")
    if valor is None:
        # Ausência já é capturada na verificação de campos obrigatórios
        return
    if not isinstance(valor, (str, list)):
        errors.append(
            "'padrao_linguistico' deve ser string ou lista; "
            f"recebido: {type(valor).__name__}"
        )
    elif isinstance(valor, list) and len(valor) == 0:
        errors.append("'padrao_linguistico' como lista não pode ser vazia")


def _validar_vocabulario(persona: dict[str, Any], errors: list[str]) -> None:
    """
    Valida o campo 'vocabulario_tipico'.

    Deve ser uma lista com pelo menos 5 itens.

    Parâmetros
    ----------
    persona : dict[str, Any]
        Dicionário com os dados da persona.
    errors : list[str]
        Lista de erros onde mensagens são adicionadas em caso de falha.
    """
    vocab = persona.get("vocabulario_tipico", [])
    if not isinstance(vocab, list):
        errors.append("'vocabulario_tipico' deve ser uma lista")
    elif len(vocab) < 5:
        errors.append(
            f"'vocabulario_tipico' deve ter pelo menos 5 itens (tem {len(vocab)})"
        )


def _validar_idade(persona: dict[str, Any], tipo: str, errors: list[str]) -> None:
    """
    Valida o campo 'idade_real' conforme o tipo de persona.

    Predadores devem ser adultos (>= 18). Vítimas devem ser menores (< 18).
    Neutros não têm restrição de faixa etária.

    Parâmetros
    ----------
    persona : dict[str, Any]
        Dicionário com os dados da persona.
    tipo : str
        Tipo da persona ('predador', 'vitima' ou 'neutro').
    errors : list[str]
        Lista de erros onde mensagens são adicionadas em caso de falha.
    """
    idade = persona.get("idade_real")
    if idade is None:
        return
    if not isinstance(idade, (int, float)):
        errors.append(f"'idade_real' deve ser numérico; recebido: {type(idade).__name__}")
        return
    if tipo == "predador" and idade < 18:
        errors.append(f"Predador com idade_real < 18: {idade}")
    if tipo == "vitima" and idade >= 18:
        errors.append(f"Vítima com idade_real >= 18: {idade}")


def _validar_fases_grooming(persona: dict[str, Any], errors: list[str]) -> None:
    """
    Valida o campo 'fases_grooming' para personas do tipo predador.

    Aceita tanto dict (schema real, com chaves aproximacao/confianca/
    isolamento/dessensibilizacao) quanto string (fichas legadas).

    Parâmetros
    ----------
    persona : dict[str, Any]
        Dicionário com os dados da persona.
    errors : list[str]
        Lista de erros onde mensagens são adicionadas em caso de falha.
    """
    fases = persona.get("fases_grooming")
    if fases is None:
        errors.append("Predadores devem ter campo 'fases_grooming' preenchido")
        return
    if not isinstance(fases, (dict, str)):
        errors.append(
            "'fases_grooming' deve ser dict ou string; "
            f"recebido: {type(fases).__name__}"
        )
        return
    if isinstance(fases, dict):
        fases_esperadas = {"aproximacao", "confianca", "isolamento", "dessensibilizacao"}
        faltando = fases_esperadas - set(fases.keys())
        if faltando:
            logger.warning(
                "fases_grooming está faltando fases esperadas: %s", sorted(faltando)
            )
            # Aviso apenas — não é erro fatal; fichas podem ter subconjunto das fases


def _validar_tipo_predador(persona: dict[str, Any], errors: list[str]) -> None:
    """
    Executa validações específicas para personas do tipo predador.

    Parâmetros
    ----------
    persona : dict[str, Any]
        Dicionário com os dados da persona.
    errors : list[str]
        Lista de erros onde mensagens são adicionadas em caso de falha.
    """
    _validar_fases_grooming(persona, errors)
    if not persona.get("estrategia_abordagem"):
        errors.append("Predadores devem ter campo 'estrategia_abordagem' preenchido")


def _validar_tipo_vitima(persona: dict[str, Any], errors: list[str]) -> None:
    """
    Executa validações específicas para personas do tipo vítima.

    Parâmetros
    ----------
    persona : dict[str, Any]
        Dicionário com os dados da persona.
    errors : list[str]
        Lista de erros onde mensagens são adicionadas em caso de falha.
    """
    if not persona.get("vulnerabilidade"):
        errors.append("Vítimas devem ter campo 'vulnerabilidade' preenchido")


def validate_persona(filepath: str) -> list[str]:
    """
    Valida uma ficha de persona a partir de um arquivo JSON.

    Verifica campos obrigatórios, tipos de dados, formatos de ID e
    regras específicas por tipo de persona (predador, vítima, neutro).

    Parâmetros
    ----------
    filepath : str
        Caminho para o arquivo JSON da persona.

    Retorna
    -------
    list[str]
        Lista de mensagens de erro. Lista vazia indica persona válida.
    """
    logger.debug("Validando persona: %s", filepath)
    errors: list[str] = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            persona: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("JSON inválido em %s: %s", filepath, e)
        return [f"JSON inválido: {e}"]
    except FileNotFoundError:
        logger.error("Arquivo não encontrado: %s", filepath)
        return [f"Arquivo não encontrado: {filepath}"]

    # 1. Verificar campos obrigatórios comuns (exceto padrao_linguistico e
    #    vocabulario_tipico, que têm validações adicionais abaixo)
    campos_base = [f for f in REQUIRED_FIELDS if f not in ("padrao_linguistico", "vocabulario_tipico")]
    for field in campos_base:
        if field not in persona or not persona[field]:
            errors.append(f"Campo obrigatório ausente ou vazio: '{field}'")

    # 2. Detectar tipo e validar ID
    pid: str = persona.get("id", "")
    tipo = _validar_id(pid, errors) if pid else "desconhecido"

    # 3. Validar padrão linguístico (str ou list)
    _validar_padrao_linguistico(persona, errors)

    # 4. Validar vocabulário típico
    _validar_vocabulario(persona, errors)

    # 5. Validar idade conforme tipo
    _validar_idade(persona, tipo, errors)

    # 6. Validações específicas por tipo
    if tipo == "predador":
        _validar_tipo_predador(persona, errors)
    elif tipo == "vitima":
        _validar_tipo_vitima(persona, errors)

    if errors:
        logger.info("Persona %s inválida: %d erro(s)", pid or filepath, len(errors))
    else:
        logger.debug("Persona %s válida", pid or filepath)

    return errors


def validate_all(personas_dir: str) -> dict[str, Any]:
    """
    Valida todas as fichas JSON encontradas em um diretório de personas.

    Percorre os subdiretórios 'predadores', 'vitimas' e 'neutros' dentro
    do diretório informado e valida cada arquivo .json encontrado.

    Parâmetros
    ----------
    personas_dir : str
        Caminho para o diretório raiz que contém as subpastas de personas.

    Retorna
    -------
    dict[str, Any]
        Dicionário indexado pelo nome do arquivo JSON. Cada valor é um dict
        com as chaves 'valid' (bool) e 'errors' (list[str]).
        Subdiretórios inexistentes geram entrada com chave 'error'.
    """
    results: dict[str, Any] = {}

    for subdir in ["predadores", "vitimas", "neutros"]:
        dirpath = os.path.join(personas_dir, subdir)
        if not os.path.exists(dirpath):
            logger.warning("Subdiretório não encontrado: %s", dirpath)
            results[subdir] = {"error": f"Diretório não encontrado: {dirpath}"}
            continue

        for filename in sorted(os.listdir(dirpath)):
            if filename.endswith(".json"):
                filepath = os.path.join(dirpath, filename)
                errors = validate_persona(filepath)
                results[filename] = {
                    "valid": len(errors) == 0,
                    "errors": errors,
                }
                logger.info(
                    "%s: %s",
                    filename,
                    "válida" if not errors else f"{len(errors)} erro(s)",
                )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    personas_dir = sys.argv[1] if len(sys.argv) > 1 else "personas"

    print(f"Validando fichas em '{personas_dir}/'...\n")
    results = validate_all(personas_dir)

    total = 0
    valid = 0
    for name, result in results.items():
        total += 1
        if isinstance(result, dict) and result.get("valid"):
            valid += 1
            print(f"  ✓ {name}")
        else:
            errors = result.get("errors", [result.get("error", "?")])
            print(f"  ✗ {name}")
            for err in errors:
                print(f"    → {err}")

    print(f"\nResultado: {valid}/{total} fichas válidas")
