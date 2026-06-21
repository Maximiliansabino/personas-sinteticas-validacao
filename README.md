# Personas Sinteticas via LLM para Validacao em Deteccao de Grooming Online

Repositorio de codigo do artigo sobre uma metodologia de geracao de personas
sinteticas via LLM para dominios com restricao de dados reais. O estudo de
validacao usa deteccao precoce de grooming em conversas online como caso de
uso.

Todo o conteudo de personas e conversas sinteticas deste projeto e ficticio e
foi produzido para pesquisa academica. O repositorio nao inclui dados reais
sensiveis, chaves de API, arquivos `.env` ou a base PAN 2012.

## Estrutura do repositorio

```text
.
├── src/                         # Codigo Python do pipeline
│   ├── agents/                  # Agentes de geracao de conversas sinteticas
│   ├── preprocess.py            # Pre-processamento PAN 2012
│   ├── classifier.py            # Baseline TF-IDF + LinearSVC/NB
│   ├── evaluate.py              # Experimentos E2-E5 e exportacao KNIME
│   ├── model_router.py          # Roteamento provider/model
│   ├── persona_validator.py     # Validacao das fichas de personas
│   └── summarize_results.py     # Consolidacao dos CSVs em Markdown
├── personas/                    # Fichas JSON ficticias e templates
├── scripts/
│   └── run_article_pipeline.sh  # Script principal de reexecucao do artigo
├── reports/final/               # CSVs e resumo usados como resultado final
├── knime/                       # Workflow KNIME de visualizacao
├── tests/                       # Testes automatizados do codigo publico
├── data/                        # Estrutura esperada para dados locais
├── requirements.txt
└── Makefile
```

## Requisitos

- Python 3.10 ou superior.
- Dependencias listadas em `requirements.txt`.
- Stopwords do NLTK disponiveis localmente para reproduzir o
  pre-processamento do PAN 2012.

Instalacao:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

Se o ambiente tiver o comando `rtk`, ele pode ser usado como prefixo dos
comandos. O repositorio tambem funciona com Python diretamente.

## Dados

Os dados brutos nao sao versionados.

Estrutura esperada para reexecucao completa:

```text
data/
├── pan2012/train/
│   ├── pan12-sexual-predator-identification-training-corpus-2012-05-01.xml
│   └── pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt
├── synthetic/
│   └── *.xml
└── processed/
```

A base PAN 2012 deve ser obtida pelos canais oficiais do PAN/CLEF e colocada
localmente em `data/pan2012/`. Os XMLs sinteticos tambem nao sao incluidos
quando contiverem resultados de execucoes locais.

## Execucao principal

O script publico de reexecucao esta em:

```bash
bash scripts/run_article_pipeline.sh
```

Por padrao, o script:

1. valida as fichas em `personas/`;
2. pre-processa o PAN 2012 se os arquivos locais existirem;
3. cria o parquet sintetico se houver XMLs em `data/synthetic/`;
4. executa E1-E5 quando os parquets necessarios estiverem disponiveis;
5. consolida os CSVs em `reports/final/summary.md`.

O script nao dispara geracao LLM por padrao e nao exige credenciais para as
etapas offline. Para proteger reproducibilidade e custo, geracao com APIs deve
ser executada separadamente e apenas com autorizacao explicita.

Variaveis uteis:

```bash
PAN_XML=data/pan2012/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml
PAN_PRED=data/pan2012/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt
PAN_PARQUET=data/processed/pan2012_train.parquet
SYNTH_DIR=data/synthetic
SYNTH_PARQUET=data/processed/synthetic.parquet
RESULTS_DIR=reports/final
```

## Comandos manuais

Validar personas:

```bash
python src/persona_validator.py personas
```

Pre-processar PAN 2012:

```bash
python -m src.preprocess \
  --train data/pan2012/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml \
  --predators data/pan2012/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt \
  --output data/processed/pan2012_train.parquet
```

Executar baseline E1:

```bash
python -m src.classifier \
  --input data/processed/pan2012_train.parquet \
  --experiment baseline_full

cp reports/baseline_results.csv reports/final/baseline_results.csv
```

Criar parquet sintetico a partir dos XMLs locais:

```bash
python -m src.evaluate \
  --create-synth data/synthetic \
  --synth data/processed/synthetic.parquet \
  --split-synth \
  --synth-train data/processed/synthetic_train.parquet \
  --synth-test data/processed/synthetic_test.parquet
```

Executar E2-E5:

```bash
python -m src.evaluate \
  --pan data/processed/pan2012_train.parquet \
  --synth data/processed/synthetic.parquet \
  --synth-train data/processed/synthetic_train.parquet \
  --synth-test data/processed/synthetic_test.parquet \
  --e2 --e3 --e4 --e5 \
  --top-n 20 \
  --export-knime \
  --reports-dir reports/final
```

Consolidar resumo:

```bash
python -m src.summarize_results \
  --results-dir reports/final \
  --baseline-csv reports/final/baseline_results.csv \
  --output reports/final/summary.md
```

## Experimentos

- E1: baseline PAN 2012 com TF-IDF uni+bigramas, LinearSVC e MultinomialNB.
- E2: treino em PAN 2012 e avaliacao no corpus sintetico.
- E3: leave-one-out no corpus sintetico.
- E4: sobreposicao de features por Jaccard.
- E5: avaliacao com augmentation PAN + sintetico.

## Resultados

Os resultados entregues ficam em `reports/final/`:

- `baseline_results.csv`
- `e2_cross_domain.csv`
- `e3_loo.csv`
- `e4_jaccard.csv`
- `e5_augmentation.csv`
- `summary.md`

O workflow KNIME em `knime/BMT_personas-sinteticas-validacao/` consome os
CSVs de resultado para visualizacao.

## Consideracoes eticas

Este projeto e estritamente academico. As personas sao ficticias, os dados
reais sensiveis nao sao versionados e o codigo deve ser usado para pesquisa,
seguranca e deteccao. O repositorio evita exemplos operacionais explicitos de
abuso e nao inclui credenciais ou dados privados.
