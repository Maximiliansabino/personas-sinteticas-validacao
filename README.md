# Geração de Personas Sintéticas para Domínios com Restrição de Dados Reais

**Metodologia e Validação via Detecção de Predadores Sexuais em Conversas Virtuais**

---

## Objetivo

Este projeto propõe uma **metodologia sistemática de geração de personas fictícias** para produzir dados textuais sintéticos em domínios onde dados reais são inacessíveis por razões legais, éticas ou práticas.

Como estudo de caso para validação, utilizamos o domínio de **detecção precoce de predadores sexuais em conversas virtuais**, replicando parcialmente o trabalho de [Panzariello (2022)](https://www.cos.ufrj.br/index.php/pt-BR/publicacoes-pesquisa/details/15/3062).

> **Nota importante:** Todos os dados gerados neste projeto são **100% fictícios**. Nenhuma persona corresponde a uma pessoa real. O projeto tem finalidade exclusivamente acadêmica.

---

## Estrutura do Repositório

```
personas-sinteticas-validacao/
├── README.md                  # Este arquivo
├── requirements.txt           # Dependências Python
├── .gitignore                 # Arquivos ignorados pelo Git
│
├── data/
│   ├── pan2012/               # Base PAN 2012 (não versionada — ver instruções)
│   ├── synthetic/             # Corpus sintético gerado (XMLs das conversas)
│   └── processed/             # Dados pré-processados para classificação
│
├── personas/
│   ├── framework/             # Template de ficha + guia de preenchimento
│   ├── predadores/            # Fichas JSON dos 6 predadores fictícios
│   ├── vitimas/               # Fichas JSON das 6 vítimas fictícias
│   └── neutros/               # Fichas JSON dos 4 perfis neutros
│
├── src/
│   ├── preprocessing.py       # Pipeline de pré-processamento textual
│   ├── classifier.py          # SVM + Naive Bayes com TF-IDF
│   ├── evaluate.py            # Cálculo de métricas e análise de features
│   └── persona_validator.py   # Validação de consistência das fichas
│
├── notebooks/
│   └── 01_eda_exploratoria.ipynb  # Análise exploratória inicial
│
└── reports/
    ├── figures/               # Figuras geradas (wordclouds, gráficos)
    └── proposta.pdf           # Proposta do projeto (Entrega 1)
```

---

## Problema

Em diversas áreas de NLP e Mineração de Textos, os melhores resultados dependem de dados reais que são inacessíveis:
- **Predadores sexuais:** conversas reais são sigilosas e protegidas por lei
- **Saúde mental:** dados de pacientes protegidos por sigilo médico
- **Fraude/engenharia social:** registros contêm dados pessoais das vítimas
- **Assédio no trabalho:** denúncias internas são confidenciais

Este projeto propõe uma solução: **personas fictícias estruturadas** que geram dados sintéticos preservando padrões linguísticos e comportamentais do domínio.

---

## Hipótese

**H1:** Personas fictícias construídas com fichas estruturadas geram conversas sintéticas capazes de produzir **F0.5 ≥ 70%** na classificação binária (conversa predatória vs. normal) com as primeiras 10 mensagens, avaliadas por um SVM treinado em dados reais (PAN 2012).

---

## Instruções de Execução

### Pré-requisitos

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Obter a base PAN 2012

A base PAN 2012 não está incluída no repositório por questões de licença. Para obtê-la:

1. Acesse [https://pan.webis.de/clef12/pan12-web/sexual-predator-identification.html](https://pan.webis.de/clef12/pan12-web/sexual-predator-identification.html)
2. Solicite acesso aos dados para fins de pesquisa
3. Coloque os arquivos XML em `data/pan2012/`

### Executar o pipeline (preliminar)

```bash
# Pré-processamento
python src/preprocessing.py --input data/synthetic/ --output data/processed/

# Classificação
python src/classifier.py --train data/processed/train.csv --test data/processed/test.csv

# Avaliação
python src/evaluate.py --predictions results/predictions.csv --output reports/figures/
```

> **Status:** O pipeline está em desenvolvimento. As instruções acima serão atualizadas conforme o projeto avança.

---

## Metodologia de Personas

Cada persona é definida por uma **ficha estruturada** com campos obrigatórios:

| Campo | Descrição |
|-------|-----------|
| ID | Identificador único (ex: PRED-001) |
| Nickname | Nome usado no chat |
| Idade real / declarada | Idade verdadeira e idade que declara ter |
| Padrão linguístico | Como escreve (gírias, formalidade, erros) |
| Vocabulário típico | Palavras e expressões frequentes |
| Modelo comportamental | Fases de grooming que executa |
| Estratégia de abordagem | Como inicia e escala o contato |

Os templates completos estão em `personas/framework/`.

---

## Links

- **Overleaf:** [https://www.overleaf.com/project/[ID_DO_PROJETO]](https://www.overleaf.com/project/[ID_DO_PROJETO])
- **Referência base:** Panzariello, M. (2022). *Estratégias para Detecção Precoce de Predadores Sexuais em Conversas realizadas na Internet.* Dissertação de Mestrado, COPPE/UFRJ.

---

## Equipe

| Nome | Matrícula |
|------|-----------|
| [Maximilian Sabino Ribeiro] | [DRE/Matrícula] |
| [João Pedro] | [DRE/Matrícula] |

---

## Licença

Este projeto é de uso acadêmico. Os dados sintéticos gerados são fictícios e livres de restrições legais. A base PAN 2012 está sujeita à licença do PAN/CLEF.
