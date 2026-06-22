"""Testes placeholder — substitua pelos testes reais de cada módulo."""


def test_estrutura_criada():
    """Verifica que a estrutura do repositório foi criada corretamente."""
    from pathlib import Path

    dirs_esperados = [
        "data/synthetic",
        "personas/framework",
        "personas/predadores",
        "personas/vitimas",
        "personas/neutros",
        "src/agents",
        "reports",
        "knime",
    ]
    for d in dirs_esperados:
        assert Path(d).is_dir(), f"Diretório ausente: {d}"
