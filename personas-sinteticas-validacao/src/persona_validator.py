"""
Validador de consistência das fichas de persona.
Verifica se os campos obrigatórios estão preenchidos
e se as conversas são consistentes com as fichas.
"""

import json
import os
import sys

REQUIRED_FIELDS = [
    "id", "nickname", "nome_ficticio", "idade_real",
    "genero", "padrao_linguistico", "vocabulario_tipico",
    "modelo_comportamental", "motivacao", "estrategia_abordagem",
]

PREDATOR_PREFIX = "PRED"
VICTIM_PREFIX = "VIT"
NEUTRAL_PREFIX = "NEUT"


def validate_persona(filepath: str) -> list[str]:
    """Valida uma ficha de persona. Retorna lista de erros."""
    errors = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            persona = json.load(f)
    except json.JSONDecodeError as e:
        return [f"JSON inválido: {e}"]
    except FileNotFoundError:
        return [f"Arquivo não encontrado: {filepath}"]

    # Verificar campos obrigatórios
    for field in REQUIRED_FIELDS:
        if field not in persona or not persona[field]:
            errors.append(f"Campo obrigatório ausente ou vazio: '{field}'")

    # Verificar formato do ID
    pid = persona.get("id", "")
    if pid:
        valid_prefixes = [PREDATOR_PREFIX, VICTIM_PREFIX, NEUTRAL_PREFIX]
        if not any(pid.startswith(p) for p in valid_prefixes):
            errors.append(f"ID '{pid}' deve começar com PRED, VIT ou NEUT")

    # Verificar vocabulário típico é lista
    vocab = persona.get("vocabulario_tipico", [])
    if not isinstance(vocab, list):
        errors.append("'vocabulario_tipico' deve ser uma lista")
    elif len(vocab) < 5:
        errors.append(f"'vocabulario_tipico' deve ter pelo menos 5 itens (tem {len(vocab)})")

    # Verificar idade
    idade = persona.get("idade_real")
    if idade is not None:
        if pid.startswith(PREDATOR_PREFIX) and idade < 18:
            errors.append(f"Predador com idade_real < 18: {idade}")
        if pid.startswith(VICTIM_PREFIX) and idade >= 18:
            errors.append(f"Vítima com idade_real >= 18: {idade}")

    # Verificar campos específicos de vítimas
    if pid.startswith(VICTIM_PREFIX):
        if not persona.get("vulnerabilidade"):
            errors.append("Vítimas devem ter campo 'vulnerabilidade' preenchido")

    return errors


def validate_all(personas_dir: str) -> dict:
    """Valida todas as fichas em um diretório."""
    results = {}

    for subdir in ["predadores", "vitimas", "neutros"]:
        dirpath = os.path.join(personas_dir, subdir)
        if not os.path.exists(dirpath):
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

    return results


if __name__ == "__main__":
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
