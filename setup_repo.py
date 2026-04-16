#!/usr/bin/env python3
"""
setup_repo.py
=============
Script de criação da estrutura do repositório:
  Personas Sintéticas via LLM — COS738 PESC/COPPE/UFRJ 2026
  github.com/Maximiliansabino/personas-sinteticas-validacao

Uso:
    python setup_repo.py                  # cria na pasta atual
    python setup_repo.py --root /caminho  # cria em outro diretório
    python setup_repo.py --dry-run        # só mostra o que seria criado
"""

import argparse
import json
import os
import sys
import hashlib
import shutil
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional

# ──────────────────────────────────────────────────────────────────
#  ESTRUTURA COMPLETA DO REPOSITÓRIO
# ──────────────────────────────────────────────────────────────────

DIRS = [
    # dados (todos gitignored exceto synthetic)
    "data/pan2012",
    "data/osaeba",
    "data/synthetic",
    "data/processed",

    # personas
    "personas/framework",
    "personas/predadores",
    "personas/vitimas",
    "personas/neutros",

    # código-fonte
    "src/agents",

    # análise (KNIME e relatórios)
    "knime",
    "notebooks",
    "reports/figures",

    # testes
    "tests",
]

# ──────────────────────────────────────────────────────────────────
#  CONTEÚDO DOS ARQUIVOS
# ──────────────────────────────────────────────────────────────────

FILES = {}

# ── .gitignore ────────────────────────────────────────────────────
# Linhas que DEVEM estar no .gitignore.
# Se o arquivo já existe, o script faz MERGE — só adiciona o que falta.
GITIGNORE_LINES = [
    "# Dados reais (nunca commitar)",
    "data/pan2012/",
    "data/osaeba/",
    "data/processed/",
    "",
    "# Variáveis de ambiente",
    ".env",
    ".env.*",
    "",
    "# Claude Code — nunca commitar",
    "CLAUDE.md",
    "CLAUDE.local.md",
    ".claude/",
    "",
    "# Python",
    "__pycache__/",
    "*.py[cod]",
    "*.pyo",
    ".pytest_cache/",
    ".mypy_cache/",
    "dist/",
    "build/",
    "*.egg-info/",
    ".eggs/",
    "",
    "# Notebooks checkpoints",
    ".ipynb_checkpoints/",
    "",
    "# IDEs",
    ".vscode/",
    ".idea/",
    "*.swp",
    "*.swo",
    "",
    "# macOS",
    ".DS_Store",
    "",
    "# Logs",
    "*.log",
    "logs/",
    "",
    "# Relatórios grandes (PDFs gerados)",
    "reports/figures/*.pdf",
]

# ── .env.example ─────────────────────────────────────────────────
FILES[".env.example"] = """\
# Copie este arquivo para .env e preencha com suas credenciais
# NUNCA commite o arquivo .env

ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
MONGODB_URI=mongodb+srv://usuario:senha@cluster.mongodb.net/personas_sinteticas
"""

# ── requirements.txt ─────────────────────────────────────────────
FILES["requirements.txt"] = """\
# LLM providers
anthropic>=0.25.0
groq>=0.9.0

# MongoDB
pymongo>=4.7.0

# Machine Learning
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
numpy>=1.26.0
pandas>=2.2.0

# NLP / parsing
nltk>=3.8.0
lxml>=5.2.0

# Utilities
python-dotenv>=1.0.0
tenacity>=8.3.0

# Dev / tests
pytest>=8.2.0
pytest-cov>=5.0.0
"""

# ── CLAUDE.md ─────────────────────────────────────────────────────
FILES["CLAUDE.md"] = """\
# Personas Sintéticas via LLM — COS738 PESC/COPPE/UFRJ 2026
Dupla: Maximilian Sabino Ribeiro + João Pedro
GitHub: github.com/Maximiliansabino/personas-sinteticas-validacao
Overleaf: overleaf.com/project/69cd99693080fb4fc88cf732

## Objetivo
Metodologia de geração de personas sintéticas via LLM para domínios
com restrição de dados reais. Caso de validação: detecção precoce
de predadores sexuais (grooming) em conversas online.

## Stack técnica
- Python 3.10+
- scikit-learn, pandas, numpy, lxml, imbalanced-learn
- anthropic SDK — agente vítima
- groq SDK — agentes predador e neutro
- pymongo — persistência MongoDB Atlas
- pytest para testes unitários

## Datasets
- PAN 2012: data/pan2012/ (gitignored) — XML inglês, 66927 conversas treino
- OSAEBA: data/osaeba/ (gitignored) — PT-BR, calibração de personas
- Corpus sintético: data/synthetic/ — XMLs gerados pelos agentes

## Análise de dados
Feita no KNIME (knime/ — workflows separados).
Python gera CSVs em reports/ → KNIME lê e visualiza.

## Arquitetura multi-agente
src/agents/
  orchestrator.py     — coordena diálogo, salva XML + MongoDB
  agent_predador.py   — Groq (modelo via --model-predador)
  agent_vitima.py     — Anthropic (modelo via --model-vitima)
  agent_neutro.py     — Groq (modelo via --model-neutro)
src/db.py             — cliente MongoDB Atlas, todas as collections
src/model_router.py   — resolve prefixo groq/ ou anthropic/
src/preprocess.py     — pipeline PAN 2012 (Panzariello 2022)
src/classifier.py     — SVM + NB + TF-IDF baseline
src/evaluate.py       — experimentos E1-E5, exporta CSV para KNIME

## Convenções de código
- Docstrings em português em todas as funções públicas
- Type hints obrigatórios
- Logging com módulo logging (nível INFO por padrão)
- Seed reprodutível: random_state=42 em todo código ML
- Nenhum dado real comitado (pan2012/, osaeba/ gitignored)
- requirements.txt sempre atualizado após instalar pacotes

## Variáveis de ambiente (.env — nunca commitar)
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/personas_sinteticas

## Referência metodológica
Panzariello (2022): Estratégia 1, SVM + TF-IDF
Pontos de corte N = {5, 10, 15, 20, 24, 50} mensagens
Baseline esperado: F0.5=85.96% sem balanceamento, F0.5=99.89% undersampling

## Collections MongoDB
- generations : cada conversa gerada (ficha + msgs + tokens + modelo)
- experiments : resultados E1-E5
- personas    : fichas registradas

## Comandos principais
# Pré-processamento PAN 2012
python -m src.preprocess --train data/pan2012/train/corpus.xml \\
  --predators data/pan2012/train/predators.txt

# Baseline
python -m src.classifier --input data/processed/pan2012_train.parquet

# Gerar conversa predatória
python -m src.agents.orchestrator \\
  --tipo predatory \\
  --ficha-predador personas/predadores/P001_gradual.json \\
  --ficha-vitima personas/vitimas/V001_isolamento.json \\
  --model-predador groq/llama-3.1-70b-versatile \\
  --model-vitima anthropic/claude-3-sonnet-20240229 \\
  --n-msgs 30 --session-id SYN_001

# Gerar conversa neutra
python -m src.agents.orchestrator \\
  --tipo neutral \\
  --ficha-neutro personas/neutros/N001_estudante.json \\
  --model-neutro groq/llama-3.1-8b-instant \\
  --n-msgs 20 --session-id NEU_001

# Experimentos completos + exportar para KNIME
python -m src.evaluate --all --export-knime
"""

# ── README.md ─────────────────────────────────────────────────────
FILES["README.md"] = f"""\
# Personas Sintéticas via LLM
## COS738 Busca e Mineração de Texto — PESC/COPPE/UFRJ 2026/01
**Dupla:** Maximilian Sabino Ribeiro · João Pedro

> Metodologia de geração de personas sintéticas via LLM para domínios
> com restrição de dados reais. Caso de validação: detecção precoce de
> predadores sexuais (grooming) em conversas virtuais.

**Overleaf:** https://www.overleaf.com/project/69cd99693080fb4fc88cf732

---

## Estrutura do repositório

```
├── data/
│   ├── pan2012/        ← PAN 2012 (gitignored)
│   ├── osaeba/         ← OSAEBA (gitignored)
│   ├── synthetic/      ← corpus sintético (XMLs gerados)
│   └── processed/      ← parquets pré-processados
├── personas/
│   ├── framework/      ← templates JSON
│   ├── predadores/     ← 6 fichas JSON
│   ├── vitimas/        ← 6 fichas JSON
│   └── neutros/        ← 4 fichas JSON
├── src/
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── agent_predador.py   ← Groq
│   │   ├── agent_vitima.py     ← Anthropic
│   │   └── agent_neutro.py     ← Groq
│   ├── db.py           ← MongoDB Atlas
│   ├── model_router.py ← resolve groq/ | anthropic/
│   ├── preprocess.py
│   ├── classifier.py
│   └── evaluate.py
├── knime/              ← workflows KNIME
├── notebooks/          ← Jupyter exploratórios
├── reports/
│   ├── figures/
│   └── *.csv           ← exportados para o KNIME
├── tests/
├── CLAUDE.md           ← contexto para o Claude Code
├── requirements.txt
└── .env.example
```

## Configuração

```bash
# 1. Clonar e instalar dependências
git clone https://github.com/Maximiliansabino/personas-sinteticas-validacao
cd personas-sinteticas-validacao
pip install -r requirements.txt

# 2. Configurar variáveis de ambiente
cp .env.example .env
# editar .env com suas API keys e MONGODB_URI

# 3. Baixar dados NLTK necessários
python -c "import nltk; nltk.download('stopwords')"
```

## Hipóteses

| ID | Hipótese | Limiar |
|---|---|---|
| H1 | Utilidade classificatória | F0.5 ≥ 70% |
| H2 | Representatividade linguística | Jaccard ≥ 60% |
| H3 | Data augmentation | +2 p.p. em F0.5 |

## Experimentos

| ID | Tipo | Descrição |
|---|---|---|
| E1 | Baseline | Replicação Panzariello (2022) no PAN 2012 |
| E2 | Cross-domain | Treino PAN 2012 → teste corpus sintético |
| E3 | Leave-one-out | Consistência interna do corpus sintético |
| E4 | Jaccard | Sobreposição de features discriminativas |
| E5 | Augmentation | Dados mistos real + sintético |

---
*Repositório criado em: {datetime.now().strftime("%Y-%m-%d")}*
"""

# ── src/__init__.py ───────────────────────────────────────────────
FILES["src/__init__.py"] = '"""Pacote principal — Personas Sintéticas via LLM."""\n'

# ── src/agents/__init__.py ────────────────────────────────────────
FILES["src/agents/__init__.py"] = '"""Agentes de geração de conversas."""\n'

# ── tests/__init__.py ─────────────────────────────────────────────
FILES["tests/__init__.py"] = ""

# ── tests/test_placeholder.py ────────────────────────────────────
FILES["tests/test_placeholder.py"] = """\
\"\"\"Testes placeholder — substitua pelos testes reais de cada módulo.\"\"\"


def test_estrutura_criada():
    \"\"\"Verifica que a estrutura do repositório foi criada corretamente.\"\"\"
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
"""

# ── data/synthetic/.gitkeep ───────────────────────────────────────
FILES["data/synthetic/.gitkeep"] = ""
FILES["data/processed/.gitkeep"] = ""
FILES["reports/figures/.gitkeep"] = ""
FILES["knime/.gitkeep"] = ""
FILES["notebooks/.gitkeep"] = ""

# ── Template predador ─────────────────────────────────────────────
FILES["personas/framework/template_predador.json"] = json.dumps({
    "id": "P000",
    "nickname": "string — apelido usado no chat",
    "idade_real": 0,
    "genero": "M | F | NB",
    "perfil_cobertura": "string — identidade falsa apresentada à vítima",
    "padrao_linguistico": [
        "lista de características do estilo de escrita",
        "ex: uso frequente de diminutivos",
        "ex: frases curtas e diretas"
    ],
    "vocabulario_tipico": [
        "lista de 20 palavras ou expressões características",
        "ex: 'ei', 'oi sumida', 'tô aqui', 'manda foto'"
    ],
    "modelo_comportamental": "texto descrevendo padrão geral de abordagem",
    "motivacao": "string — motivação principal do predador",
    "estrategia_abordagem": "agressiva | gradual | paternal | romantica | financeira | digital",
    "fases_grooming": {
        "aproximacao": "como inicia o contato — ex: comentário em post, jogo online",
        "confianca": "como constrói vínculo — ex: elogios, escuta ativa, segredos",
        "isolamento": "como separa a vítima — ex: 'não conta pra ninguém'",
        "dessensibilizacao": "como introduz temas sexuais progressivamente"
    }
}, indent=2, ensure_ascii=False)

# ── Template vítima ───────────────────────────────────────────────
FILES["personas/framework/template_vitima.json"] = json.dumps({
    "id": "V000",
    "nickname": "string — apelido usado no chat",
    "idade_real": 0,
    "faixa_etaria_simulada": "13-14 | 15-16 | 17",
    "genero": "M | F | NB",
    "perfil_cobertura": "string — como se apresenta online",
    "padrao_linguistico": [
        "lista de características do estilo de escrita",
        "ex: uso de 'kkkk', 'né', 'vc', 'tb', 'pq'"
    ],
    "vocabulario_tipico": [
        "lista de 20 palavras ou expressões características"
    ],
    "modelo_comportamental": "texto descrevendo como reage a abordagens",
    "motivacao": "string — o que busca nas interações online",
    "vulnerabilidade": "baixa_supervisao | isolamento | validacao | dificuldades_afetivas | curiosidade | dependencia",
    "contexto_social": "string — situação familiar, escolar e social",
    "faixa_horario_online": "ex: noite após 22h quando pais dormem"
}, indent=2, ensure_ascii=False)

# ── Template neutro ───────────────────────────────────────────────
FILES["personas/framework/template_neutro.json"] = json.dumps({
    "id": "N000",
    "nickname": "string",
    "idade_real": 0,
    "genero": "M | F | NB",
    "perfil_cobertura": "string — como se apresenta online",
    "padrao_linguistico": [
        "lista de características do estilo de escrita"
    ],
    "vocabulario_tipico": [
        "lista de 20 palavras ou expressões"
    ],
    "modelo_comportamental": "texto — conversa sobre temas cotidianos, sem padrão predatório",
    "motivacao": "string — ex: fazer amigos, falar de jogos, música",
    "interesses": [
        "lista de interesses — ex: jogos, k-pop, futebol, séries"
    ]
}, indent=2, ensure_ascii=False)

# ── Exemplo persona predador P001 ─────────────────────────────────
FILES["personas/predadores/P001_gradual.json"] = json.dumps({
    "id": "P001",
    "nickname": "Rafa_gamer",
    "idade_real": 34,
    "genero": "M",
    "perfil_cobertura": "Adolescente de 17 anos, gosta de jogos online e animes. Diz morar em São Paulo, estudar no 3º ano do ensino médio.",
    "padrao_linguistico": [
        "frases curtas típicas de chat de jogo",
        "uso frequente de abreviações: 'vc', 'tb', 'pq', 'msm'",
        "emojis e emoticons moderados",
        "erros ortográficos leves intencionais para parecer jovem",
        "perguntas sobre a vida pessoal da vítima disfarçadas de curiosidade casual"
    ],
    "vocabulario_tipico": [
        "ei", "oi", "boa", "bora", "msm", "vc", "tb", "kkkk",
        "zuera", "partiu", "trampo", "mano", "cara", "tipo",
        "jogar", "ranked", "drop", "server", "stream", "clip"
    ],
    "modelo_comportamental": "Inicia contato em jogos online ou redes sociais. Progride lentamente, construindo confiança por semanas antes de introduzir temas pessoais. Nunca demonstra urgência nas primeiras interações.",
    "motivacao": "Busca controle e satisfação sexual através de manipulação emocional progressiva.",
    "estrategia_abordagem": "gradual",
    "fases_grooming": {
        "aproximacao": "Comenta em posts de jogos, oferece ajuda em missões, compartilha dicas. Usa linguagem de gamer para criar identificação.",
        "confianca": "Torna-se 'melhor amigo virtual'. Compartilha segredos falsos para criar reciprocidade. Elogia a vítima frequentemente.",
        "isolamento": "Sutilmente desencoraja outras amizades: 'os outros não te entendem como eu'. Pede para manter a amizade em segredo dos pais.",
        "dessensibilizacao": "Introduz temas românticos gradualmente, depois sexuais, usando o contexto de jogos como disfarce inicial."
    }
}, indent=2, ensure_ascii=False)

# ── Exemplo persona vítima V001 ───────────────────────────────────
FILES["personas/vitimas/V001_isolamento.json"] = json.dumps({
    "id": "V001",
    "nickname": "Mari_15",
    "idade_real": 15,
    "faixa_etaria_simulada": "15-16",
    "genero": "F",
    "perfil_cobertura": "Adolescente de 15 anos, gosta de animes e jogos. Usa redes sociais à noite.",
    "padrao_linguistico": [
        "uso intenso de 'kkkk', 'kkk', 'hahaha'",
        "abreviações: 'vc', 'tb', 'pq', 'né', 'msm', 'tô'",
        "frases curtas e respostas rápidas",
        "uso de reticências para indicar hesitação: '...'",
        "erros ortográficos comuns de adolescente digitando rápido"
    ],
    "vocabulario_tipico": [
        "oi", "boa", "kkkk", "né", "tô", "vc", "tb", "pq",
        "saudade", "chato", "nossa", "cara", "tipo assim",
        "não sei", "que legal", "sério?", "mds", "ai que"
    ],
    "modelo_comportamental": "Responde positivamente a atenção e elogios. Compartilha sentimentos quando se sente ouvida. Demonstra carência de conexão emocional.",
    "motivacao": "Busca amizade genuína e validação emocional online, especialmente à noite quando se sente sozinha.",
    "vulnerabilidade": "isolamento",
    "contexto_social": "Pais separados. Mora com a mãe que trabalha até tarde. Poucas amigas próximas na escola. Passa as noites no quarto com o celular.",
    "faixa_horario_online": "21h às 01h, após a mãe dormir"
}, indent=2, ensure_ascii=False)

# ── Exemplo persona neutro N001 ───────────────────────────────────
FILES["personas/neutros/N001_estudante.json"] = json.dumps({
    "id": "N001",
    "nickname": "Gui_gamer",
    "idade_real": 16,
    "genero": "M",
    "perfil_cobertura": "Adolescente de 16 anos, fã de jogos e séries.",
    "padrao_linguistico": [
        "linguagem casual de adolescente",
        "muitos emojis",
        "abreviações comuns: 'vc', 'tb', 'pq'",
        "entusiasmo ao falar de jogos e séries favoritas"
    ],
    "vocabulario_tipico": [
        "cara", "mano", "boa", "partiu", "bora", "saudade",
        "jogo", "série", "aula", "prova", "férias", "rolê",
        "kkkk", "vlw", "tmj", "flw", "até", "sumiu"
    ],
    "modelo_comportamental": "Conversa sobre assuntos do dia a dia: escola, jogos, séries. Sem qualquer padrão de manipulação ou interesse sexual.",
    "motivacao": "Socializar e falar sobre interesses comuns.",
    "interesses": [
        "Free Fire", "FIFA", "animes", "séries Netflix",
        "futebol", "escola", "amigos", "música funk e trap"
    ]
}, indent=2, ensure_ascii=False)

# ── Placeholders para as demais fichas ───────────────────────────
for pid, strat in [("P002", "paternal"), ("P003", "romantica"),
                   ("P004", "digital"), ("P005", "financeira"),
                   ("P006", "agressiva")]:
    FILES[f"personas/predadores/{pid}_{strat}.json"] = json.dumps({
        "id": pid,
        "_status": "TODO — preencher com Claude Code (prompt fichas JSON)",
        "estrategia_abordagem": strat
    }, indent=2, ensure_ascii=False)

for vid, vuln in [("V002", "baixa_supervisao"), ("V003", "validacao"),
                  ("V004", "dificuldades_afetivas"), ("V005", "curiosidade"),
                  ("V006", "dependencia")]:
    FILES[f"personas/vitimas/{vid}_{vuln}.json"] = json.dumps({
        "id": vid,
        "_status": "TODO — preencher com Claude Code (prompt fichas JSON)",
        "vulnerabilidade": vuln
    }, indent=2, ensure_ascii=False)

for nid, tema in [("N002", "fas_musica"), ("N003", "gamer"),
                  ("N004", "esportista")]:
    FILES[f"personas/neutros/{nid}_{tema}.json"] = json.dumps({
        "id": nid,
        "_status": "TODO — preencher com Claude Code (prompt fichas JSON)",
        "tema": tema
    }, indent=2, ensure_ascii=False)

# ── src placeholders ─────────────────────────────────────────────
for modulo in ["db", "model_router", "preprocess", "classifier", "evaluate"]:
    FILES[f"src/{modulo}.py"] = f'''\
"""
{modulo}.py
{"=" * (len(modulo) + 3)}
TODO — gerar com Claude Code (ver prompts no guia).
"""
'''

for agente in ["orchestrator", "agent_predador", "agent_vitima", "agent_neutro"]:
    FILES[f"src/agents/{agente}.py"] = f'''\
"""
{agente}.py
{"=" * (len(agente) + 3)}
TODO — gerar com Claude Code (ver prompts no guia).
"""
'''

# ── Makefile ──────────────────────────────────────────────────────
FILES["Makefile"] = """\
.PHONY: install deps nltkdata test preprocess baseline evaluate clean

install: deps nltkdata

deps:
\tpip install -r requirements.txt

nltkdata:
\tpython -c "import nltk; nltk.download(\'stopwords\')"

test:
\tpytest tests/ -v --cov=src

preprocess:
\tpython -m src.preprocess \\
\t  --train data/pan2012/train/corpus.xml \\
\t  --predators data/pan2012/train/predators.txt

baseline:
\tpython -m src.classifier \\
\t  --input data/processed/pan2012_train.parquet

evaluate:
\tpython -m src.evaluate --all --export-knime

clean:
\tfind . -type f -name "*.pyc" -delete
\tfind . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
"""

# ──────────────────────────────────────────────────────────────────
#  CRIAÇÃO DA ESTRUTURA
# ──────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────
#  DOWNLOAD DOS DATASETS
# ──────────────────────────────────────────────────────────────────

# Tamanho esperado em bytes (para validação pós-download)
PAN2012_EXPECTED_SIZE = 91_163_031
PAN2012_ZIP_NAME = "pan12-sexual-predator-identification-test-and-training.zip"
PAN2012_URL = (
    "https://zenodo.org/api/records/3713280/files/"
    "pan12-sexual-predator-identification-test-and-training.zip/content"
)

MENDELEY_FILES = [
    {
        "filename": "RawData CSV.csv",
        "size": 4_039_772,
        "url": "https://data.mendeley.com/datasets/vfcrsdsmmh/4",
    },
    {
        "filename": "RawData.xlsx",
        "size": 1_273_996,
        "url": "https://data.mendeley.com/datasets/vfcrsdsmmh/4",
    },
    {
        "filename": "Structured data CSV.csv",
        "size": 642_199,
        "url": "https://data.mendeley.com/datasets/vfcrsdsmmh/4",
    },
    {
        "filename": "Codebook.docx",
        "size": 15_940,
        "url": "https://data.mendeley.com/datasets/vfcrsdsmmh/4",
    },
]


def _fmt_size(n_bytes: int) -> str:
    """Formata bytes em MB ou KB."""
    if n_bytes >= 1_000_000:
        return f"{n_bytes / 1_000_000:.1f} MB"
    return f"{n_bytes / 1_000:.0f} KB"


def _progress_bar(downloaded: int, total: int, width: int = 40) -> str:
    """Retorna string com barra de progresso."""
    if total <= 0:
        return f"{_fmt_size(downloaded)} baixados"
    pct = min(downloaded / total, 1.0)
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:.1f}%  {_fmt_size(downloaded)}/{_fmt_size(total)}"


def _download_file(url: str, dest: Path, expected_size: int = 0,
                   label: str = "") -> bool:
    """
    Faz download de um arquivo com barra de progresso.
    Retorna True se bem-sucedido.
    """
    tmp = dest.with_suffix(".tmp")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; setup_repo/1.0; "
            "github.com/Maximiliansabino/personas-sinteticas-validacao)"
        )
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", expected_size))
            downloaded = 0
            chunk_size = 65_536  # 64 KB

            with tmp.open("wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    bar = _progress_bar(downloaded, total)
                    print(f"\r   {label}  {bar}", end="", flush=True)

        print()  # nova linha após progresso

        # Valida tamanho mínimo
        actual = tmp.stat().st_size
        if expected_size > 0 and actual < expected_size * 0.95:
            print(f"   AVISO: tamanho inesperado ({_fmt_size(actual)} vs "
                  f"{_fmt_size(expected_size)} esperado)")

        shutil.move(str(tmp), str(dest))
        return True

    except urllib.error.HTTPError as e:
        print(f"\n   ERRO HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        print(f"\n   ERRO de conexão: {e.reason}")
    except Exception as e:
        print(f"\n   ERRO: {e}")
    finally:
        if tmp.exists():
            tmp.unlink()
    return False


def _extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extrai ZIP com progresso."""
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            members = z.namelist()
            total = len(members)
            print(f"   Extraindo {total} arquivos...")
            for i, member in enumerate(members, 1):
                z.extract(member, dest_dir)
                if i % 100 == 0 or i == total:
                    bar = _progress_bar(i, total)
                    print(f"\r   {bar}", end="", flush=True)
        print()
        return True
    except zipfile.BadZipFile as e:
        print(f"   ERRO ao extrair ZIP: {e}")
        return False


def download_pan2012(root: Path, dry_run: bool = False) -> None:
    """
    Baixa e extrai o dataset PAN 2012 do Zenodo.
    Origem: https://zenodo.org/records/3713280
    Destino: data/pan2012/
    """
    dest_dir = root / "data" / "pan2012"
    zip_dest = dest_dir / PAN2012_ZIP_NAME

    print("\n📥 PAN 2012 — Zenodo (download automático)")
    print(f"   Destino: {dest_dir}")
    print(f"   Tamanho: ~{_fmt_size(PAN2012_EXPECTED_SIZE)}")

    # Verifica se já foi extraído
    xml_files = list(dest_dir.glob("**/*.xml"))
    if xml_files:
        print(f"   ─  Já extraído ({len(xml_files)} arquivos XML encontrados — mantido)")
        return

    # Verifica se o ZIP já existe
    if zip_dest.exists() and zip_dest.stat().st_size >= PAN2012_EXPECTED_SIZE * 0.95:
        print(f"   ─  ZIP já baixado ({_fmt_size(zip_dest.stat().st_size)}) — extraindo...")
    else:
        if dry_run:
            print(f"   [dry-run] Baixaria {PAN2012_URL[:70]}...")
            return
        print(f"   Baixando de: {PAN2012_URL[:70]}...")
        ok = _download_file(PAN2012_URL, zip_dest, PAN2012_EXPECTED_SIZE, "PAN 2012")
        if not ok:
            print("   FALHA no download. Baixe manualmente em:")
            print(f"   https://zenodo.org/records/3713280")
            print(f"   e coloque o ZIP em: {dest_dir}/")
            return

    # Extrai
    if not dry_run:
        print(f"   Extraindo em {dest_dir}/...")
        ok = _extract_zip(zip_dest, dest_dir)
        if ok:
            print(f"   ✓  PAN 2012 extraído com sucesso")
            zip_dest.unlink()  # remove ZIP após extração
        else:
            print("   FALHA na extração. O ZIP pode estar corrompido.")


def download_mendeley(root: Path, dry_run: bool = False) -> None:
    """
    Informa sobre o Mendeley dataset — download requer autenticação.
    Origem: https://data.mendeley.com/datasets/vfcrsdsmmh/4
    Destino: data/osaeba/ (dataset brasileiro de abuso sexual online)
    """
    dest_dir = root / "data" / "osaeba"

    print("\n📋 Mendeley Dataset — requer download manual")
    print(f"   URL: https://data.mendeley.com/datasets/vfcrsdsmmh/4")
    print(f"   Destino: {dest_dir}")
    print()
    print("   O Mendeley requer autenticação para download de arquivos.")
    print("   Passos para baixar manualmente:")
    print("   1. Acesse https://data.mendeley.com/datasets/vfcrsdsmmh/4")
    print("   2. Clique em 'Download All' (ou baixe individualmente)")
    print("   3. Coloque os arquivos em: data/osaeba/")
    print()
    print("   Arquivos esperados:")
    for f in MENDELEY_FILES:
        exists = (dest_dir / f["filename"]).exists()
        status = "✓ já existe" if exists else f"  {_fmt_size(f['size'])}"
        print(f"   {'✓' if exists else '○'}  {f['filename']}  ({status})")


def download_osaeba(root: Path, dry_run: bool = False) -> None:
    """
    Informa sobre o OSAEBA dataset — requer login IEEE DataPort.
    Destino: data/osaeba/ (junto com Mendeley — são complementares)
    """
    dest_dir = root / "data" / "osaeba"

    print("\n📋 OSAEBA (IEEE DataPort) — requer download manual")
    print(f"   URL: https://ieee-dataport.org/open-access/online-sexual-abuse-exploitation-brazilian-adolescents")
    print(f"   Destino: {dest_dir}")
    print()
    print("   O IEEE DataPort requer conta gratuita para download.")
    print("   Passos para baixar manualmente:")
    print("   1. Crie conta gratuita em https://ieee-dataport.org")
    print("   2. Acesse a URL do dataset acima")
    print("   3. Clique em 'Download Dataset'")
    print("   4. Coloque os arquivos em: data/osaeba/")


def baixar_datasets(root: Path, dry_run: bool = False,
                    skip_pan: bool = False) -> None:
    """Orquestra o download de todos os datasets."""

    print("\n" + "=" * 56)
    print("  DOWNLOAD DOS DATASETS")
    print("=" * 56)

    # PAN 2012 — automático
    if not skip_pan:
        download_pan2012(root, dry_run)

    # Mendeley — manual (instrução)
    download_mendeley(root, dry_run)

    # OSAEBA — manual (instrução)
    download_osaeba(root, dry_run)

    print()
    print("=" * 56)
    print("  Datasets: PAN 2012 automático | Mendeley + OSAEBA: manual")
    print("=" * 56)

def _gitignore_merge(path: Path, lines: list[str], dry_run: bool) -> str:
    """Adiciona ao .gitignore apenas as linhas que ainda não existem."""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        existing_lines = set(existing.splitlines())
        missing = [l for l in lines if l not in existing_lines]
        if not missing:
            return "já atualizado"
        if not dry_run:
            with path.open("a", encoding="utf-8") as f:
                f.write("\n# Adicionado pelo setup_repo.py\n")
                f.write("\n".join(missing) + "\n")
        return f"merge: +{len(missing)} linhas"
    else:
        if not dry_run:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return "criado"


def _dir_has_content(path: Path) -> bool:
    """Retorna True se o diretório existe e tem arquivos reais (ignora .gitkeep)."""
    if not path.exists():
        return False
    entries = [e for e in path.iterdir() if e.name != ".gitkeep"]
    return len(entries) > 0


def criar_estrutura(root: Path, dry_run: bool = False) -> None:
    """Cria a estrutura do repositório sem deletar nem sobrescrever conteúdo existente."""

    prefix = "[DRY-RUN] " if dry_run else ""
    print(f"\n{prefix}Criando estrutura em: {root.resolve()}\n")

    criados = 0
    mantidos = 0
    merged = 0

    # ── Diretórios ────────────────────────────────────────────────
    print("📁 Diretórios:")
    for d in DIRS:
        path = root / d
        if dry_run:
            status = "já existe com conteúdo — mantido" if _dir_has_content(path) else "criar"
            print(f"   {status:35s}  {d}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            if _dir_has_content(path):
                print(f"   ─  {d}  (tem conteúdo — mantido)")
                mantidos += 1
            else:
                print(f"   ✓  {d}")
                criados += 1

    print()

    # ── .gitignore (merge) ────────────────────────────────────────
    print("📄 Arquivos:")
    gi_path = root / ".gitignore"
    if dry_run:
        print(f"   merge   .gitignore  (Claude, dados reais, env)")
    else:
        result = _gitignore_merge(gi_path, GITIGNORE_LINES, dry_run)
        if "merge" in result or result == "criado":
            print(f"   ✓  .gitignore  ({result})")
            criados += 1
        else:
            print(f"   ─  .gitignore  ({result})")
            mantidos += 1

    # ── Demais arquivos (nunca sobrescrever se existir) ──────────
    for rel_path, file_content in FILES.items():
        path = root / rel_path

        # .gitkeep: só cria se o diretório estiver vazio
        if path.name == ".gitkeep":
            if dry_run:
                action = "criar" if not _dir_has_content(path.parent) else "omitir (dir tem conteúdo)"
                print(f"   {action:35s}  {rel_path}")
            else:
                if not _dir_has_content(path.parent) and not path.exists():
                    path.write_text("", encoding="utf-8")
                    print(f"   ✓  {rel_path}")
                    criados += 1
                elif path.exists():
                    pass  # silencioso
            continue

        if dry_run:
            exists = path.exists()
            action = "manter (já existe)" if exists else "criar"
            print(f"   {action:35s}  {rel_path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(file_content, encoding="utf-8")
                print(f"   ✓  {rel_path}")
                criados += 1
            else:
                print(f"   ─  {rel_path}  (já existe — mantido)")
                mantidos += 1

    print()

    if not dry_run:
        print("=" * 56)
        print(f"  Estrutura: {criados} criados  |  {mantidos} mantidos")
        print("=" * 56)
        print()
        print("Próximos passos:")
        print("  1. cp .env.example .env  →  preencha suas API keys")
        print("  2. pip install -r requirements.txt")
        print("  3. Abra no Claude Code e use os prompts do CLAUDE.md")
        print()



# ──────────────────────────────────────────────────────────────────
#  LIMPEZA DE ITENS FORA DA ESTRUTURA DO PROJETO
# ──────────────────────────────────────────────────────────────────

# Diretórios que pertencem ao projeto (definidos em DIRS)
# Inclui os diretórios-pai implícitos
VALID_DIRS: set[str] = {
    # raiz do projeto — nunca tocar
    ".git",
    # pais implícitos
    "data",
    "personas",
    "src",
    "reports",
    # estrutura definida em DIRS
    "data/pan2012",
    "data/osaeba",
    "data/synthetic",
    "data/processed",
    "personas/framework",
    "personas/predadores",
    "personas/vitimas",
    "personas/neutros",
    "src/agents",
    "knime",
    "notebooks",
    "reports/figures",
    "tests",
}

# Arquivos raiz válidos (definidos em FILES ou gerados pelo fluxo)
VALID_ROOT_FILES: set[str] = {
    ".gitignore",
    ".env",
    ".env.example",
    "requirements.txt",
    "CLAUDE.md",
    "CLAUDE.local.md",
    "README.md",
    "Makefile",
    "setup_repo.py",   # este script em si
}

# Diretórios que NUNCA devem ser deletados, independente de onde estejam
NEVER_DELETE_DIRS: set[str] = {
    ".git",
    ".github",
    ".vscode",
    ".idea",
}

# Padrões de nomes que indicam caches/temporários — deletar sem perguntar
# (mas ainda listados para o usuário ver)
AUTO_CLEAN_PATTERNS: set[str] = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".eggs",
    "*.egg-info",
    ".ipynb_checkpoints",
}


def _is_auto_clean(name: str) -> bool:
    """Retorna True se o nome corresponde a um padrão de cache/temporário."""
    for pat in AUTO_CLEAN_PATTERNS:
        if pat.startswith("*"):
            if name.endswith(pat[1:]):
                return True
        elif name == pat:
            return True
    return False


def _rel(path: Path, root: Path) -> str:
    """Retorna o caminho relativo como string."""
    return str(path.relative_to(root))


def _fmt_size_dir(path: Path) -> str:
    """Tamanho total de um diretório (somando arquivos recursivamente)."""
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if total >= 1_000_000:
            return f"{total / 1_000_000:.1f} MB"
        if total >= 1_000:
            return f"{total / 1_000:.0f} KB"
        return f"{total} B"
    except Exception:
        return "? B"


def _fmt_size_file(path: Path) -> str:
    """Tamanho de um arquivo."""
    try:
        s = path.stat().st_size
        if s >= 1_000_000:
            return f"{s / 1_000_000:.1f} MB"
        if s >= 1_000:
            return f"{s / 1_000:.0f} KB"
        return f"{s} B"
    except Exception:
        return "? B"


def _confirmar(prompt: str) -> bool:
    """Pede confirmação ao usuário. Retorna True se confirmado."""
    while True:
        try:
            resp = input(f"{prompt} [s/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if resp in ("s", "sim", "y", "yes"):
            return True
        if resp in ("n", "nao", "não", "no", ""):
            return False
        print("   Responda 's' para sim ou 'n' para não.")


def auditar_estrutura(root: Path, dry_run: bool = False) -> None:
    """
    Varre o repositório, identifica itens fora da estrutura esperada,
    lista para o usuário e pede confirmação antes de deletar.

    Regras:
    - Diretórios em VALID_DIRS (relativos à root) → mantidos
    - NEVER_DELETE_DIRS → nunca tocados
    - Arquivos na raiz em VALID_ROOT_FILES → mantidos
    - Arquivos dentro de VALID_DIRS → mantidos (são do projeto)
    - Todo o resto → candidato a deleção
    - AUTO_CLEAN_PATTERNS → listados como [cache] (deletados juntos na confirmação)
    """

    print("\n" + "=" * 56)
    print("  AUDITORIA DA ESTRUTURA DO REPOSITÓRIO")
    print("=" * 56)
    print(f"  Raiz: {root}\n")

    candidatos_dirs: list[Path] = []
    candidatos_files: list[Path] = []

    # ── Escanear diretórios de primeiro nível ─────────────────────
    for item in sorted(root.iterdir()):
        rel = _rel(item, root)

        # Nunca tocar
        if item.name in NEVER_DELETE_DIRS:
            continue

        if item.is_dir():
            if rel in VALID_DIRS:
                continue  # pasta válida — não entra no scan interno
            # Verifica se é sub-diretório de alguma pasta válida
            is_inside_valid = any(
                rel.startswith(vd + "/") for vd in VALID_DIRS
            )
            if is_inside_valid:
                continue
            candidatos_dirs.append(item)

        elif item.is_file():
            if item.name in VALID_ROOT_FILES:
                continue
            # Arquivos de ambiente (.env.local, etc.) — manter
            if item.name.startswith(".env"):
                continue
            candidatos_files.append(item)

    # ── Escanear sub-diretórios indesejados DENTRO das pastas válidas ─
    # Ex: src/venv/, notebooks/node_modules/, etc.
    for vd in VALID_DIRS:
        vd_path = root / vd
        if not vd_path.exists() or not vd_path.is_dir():
            continue
        for item in sorted(vd_path.iterdir()):
            if not item.is_dir():
                continue
            if item.name in NEVER_DELETE_DIRS:
                continue
            rel = _rel(item, root)
            # Sub-dir válido?
            if rel in VALID_DIRS:
                continue
            is_inside_valid = any(
                rel.startswith(vd2 + "/") for vd2 in VALID_DIRS
            )
            if is_inside_valid:
                continue
            # Cache/temporário dentro de pasta válida
            if _is_auto_clean(item.name):
                candidatos_dirs.append(item)

    if not candidatos_dirs and not candidatos_files:
        print("  Nenhum item fora da estrutura encontrado.")
        print("  O repositório está limpo.\n")
        return

    # ── Separar: caches (auto) vs. outros (precisam de confirmação) ──
    caches   = [p for p in candidatos_dirs  if _is_auto_clean(p.name)]
    dirs_ext = [p for p in candidatos_dirs  if not _is_auto_clean(p.name)]
    files_ext = candidatos_files

    # ── Exibir relatório ──────────────────────────────────────────
    total_items = len(caches) + len(dirs_ext) + len(files_ext)
    print(f"  Encontrados {total_items} item(ns) fora da estrutura:\n")

    if dirs_ext:
        print("  📁 Diretórios não reconhecidos:")
        for p in dirs_ext:
            size = _fmt_size_dir(p)
            n_files = sum(1 for _ in p.rglob("*") if _.is_file())
            rel = _rel(p, root)
            print(f"     {rel}/  ({size}, {n_files} arquivo(s))")

    if files_ext:
        print()
        print("  📄 Arquivos na raiz não reconhecidos:")
        for p in files_ext:
            rel = _rel(p, root)
            print(f"     {rel}  ({_fmt_size_file(p)})")

    if caches:
        print()
        print("  🗑  Caches / temporários (gerados automaticamente pelo Python):")
        for p in caches:
            rel = _rel(p, root)
            size = _fmt_size_dir(p)
            print(f"     {rel}/  ({size})  [cache]")

    print()

    if dry_run:
        print("  [dry-run] Nenhuma deleção realizada.\n")
        return

    # ── Confirmar por grupo ───────────────────────────────────────
    to_delete: list[Path] = []

    if dirs_ext:
        print("  ─" * 28)
        if _confirmar(f"  Deletar os {len(dirs_ext)} diretório(s) não reconhecido(s)?"):
            to_delete.extend(dirs_ext)
        else:
            print("  → Diretórios mantidos.")

    if files_ext:
        print()
        if _confirmar(f"  Deletar os {len(files_ext)} arquivo(s) da raiz não reconhecido(s)?"):
            to_delete.extend(files_ext)
        else:
            print("  → Arquivos mantidos.")

    if caches:
        print()
        if _confirmar(f"  Deletar os {len(caches)} cache(s) Python (seguros de remover)?"):
            to_delete.extend(caches)
        else:
            print("  → Caches mantidos.")

    if not to_delete:
        print()
        print("  Nenhum item deletado.\n")
        return

    # ── Executar deleção ──────────────────────────────────────────
    print()
    print("  Deletando...")
    deletados = 0
    erros = 0
    for p in to_delete:
        rel = _rel(p, root)
        try:
            if p.is_dir():
                shutil.rmtree(p)
                print(f"  ✓  {rel}/  removido")
            else:
                p.unlink()
                print(f"  ✓  {rel}  removido")
            deletados += 1
        except Exception as e:
            print(f"  ✗  {rel}  ERRO: {e}")
            erros += 1

    print()
    print(f"  Concluído: {deletados} deletado(s)"
          + (f"  |  {erros} erro(s)" if erros else "") + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cria a estrutura do repositório Personas Sintéticas via LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python setup_repo.py                        # estrutura + baixar PAN 2012
  python setup_repo.py --all                  # estrutura + datasets + limpeza
  python setup_repo.py --only-structure       # só cria pastas e arquivos
  python setup_repo.py --only-datasets        # só baixa/instrui os datasets
  python setup_repo.py --clean                # lista e remove itens fora do projeto
  python setup_repo.py --skip-pan             # estrutura sem baixar PAN 2012
  python setup_repo.py --dry-run              # mostra o que seria feito sem executar
  python setup_repo.py --root /outro/caminho  # diretório customizado
        """,
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Diretório raiz (padrão: pasta atual)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria feito sem executar nada",
    )
    parser.add_argument(
        "--only-structure",
        action="store_true",
        help="Cria apenas a estrutura de pastas/arquivos, sem baixar datasets",
    )
    parser.add_argument(
        "--only-datasets",
        action="store_true",
        help="Apenas inicia/instrui o download dos datasets (estrutura já existe)",
    )
    parser.add_argument(
        "--skip-pan",
        action="store_true",
        help="Pula o download automático do PAN 2012 (baixe manualmente depois)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Lista itens fora da estrutura e pede confirmação para deletar",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Executa tudo: estrutura + datasets + limpeza",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.exists():
        print(f"Erro: diretório '{root}' não existe.", file=sys.stderr)
        sys.exit(1)

    run_structure = not args.only_datasets and not args.clean
    run_datasets  = not args.only_structure and not args.clean
    run_clean     = args.clean or args.all

    if run_structure:
        criar_estrutura(root, dry_run=args.dry_run)

    if run_datasets:
        baixar_datasets(root, dry_run=args.dry_run, skip_pan=args.skip_pan)

    if run_clean:
        auditar_estrutura(root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()