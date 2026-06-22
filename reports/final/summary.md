# Resultados consolidados dos experimentos do artigo

_Relatório final gerado automaticamente por `src/summarize_results.py` a partir dos CSVs consolidados em `reports/final/`._

## E1 — Baseline PAN 2012 (`baseline_full`)

### Sem subamostragem

| N msgs | Classificador | F0.5 | Precisão | Revocação |
|--------|--------------|------|----------|-----------|
| 5 | LinearSVC | 83.31% | 86.52% | 72.57% |
| 5 | MultinomialNB | 83.67% | 93.47% | 58.97% |
| 10 | LinearSVC | 90.70% | 91.74% | 86.87% |
| 10 | MultinomialNB | 88.78% | 96.80% | 66.82% |
| 15 | LinearSVC | 94.11% | 95.30% | 89.68% |
| 15 | MultinomialNB | 91.34% | 96.95% | 74.32% |
| 20 | LinearSVC | 95.45% | 96.56% | 91.33% |
| 20 | MultinomialNB | 93.42% | 97.59% | 79.94% |
| 24 | LinearSVC | 95.87% | 96.84% | 92.26% |
| 24 | MultinomialNB | 94.03% | 97.40% | 82.64% |
| 50 | LinearSVC | 96.70% | 97.17% | 94.96% |
| 50 | MultinomialNB | 96.45% | 97.44% | 92.73% |

### Com undersampling

| N msgs | Classificador | F0.5 | Precisão | Revocação |
|--------|--------------|------|----------|-----------|
| 5 | LinearSVC | 50.83% | 45.74% | 91.80% |
| 5 | MultinomialNB | 51.00% | 45.76% | 94.26% |
| 10 | LinearSVC | 63.62% | 58.47% | 98.36% |
| 10 | MultinomialNB | 65.15% | 60.21% | 97.07% |
| 15 | LinearSVC | 69.39% | 64.56% | 99.06% |
| 15 | MultinomialNB | 73.50% | 69.14% | 98.48% |
| 20 | LinearSVC | 73.94% | 69.51% | 99.41% |
| 20 | MultinomialNB | 77.92% | 74.01% | 98.94% |
| 24 | LinearSVC | 77.54% | 73.48% | 99.53% |
| 24 | MultinomialNB | 80.19% | 76.60% | 98.83% |
| 50 | LinearSVC | 86.55% | 83.76% | 99.88% |
| 50 | MultinomialNB | 85.82% | 82.91% | 99.88% |

## E2 — Cross-domain (treina PAN, avalia sintético)

_Corrida com maior corpus: 8053 amostras (874 predatórias, 7179 normais)_

| N msgs | Classificador | F0.5 | Precisão | Revocação |
|--------|--------------|------|----------|-----------|
| 5 | LinearSVC | 20.00% | 100.00% | 4.76% |
| 10 | LinearSVC | 20.00% | 100.00% | 4.76% |
| 15 | LinearSVC | 45.45% | 100.00% | 14.29% |
| 20 | LinearSVC | 34.48% | 100.00% | 9.52% |
| 24 | LinearSVC | 34.48% | 100.00% | 9.52% |
| 50 | LinearSVC | 0.00% | 0.00% | 0.00% |

## E3 — Leave-one-out no corpus sintético (IC 95% bootstrap)

_Corrida de referência: total=41 (21 predatórias, 20 normais)_

| N msgs | Classificador | F0.5 | IC inf | IC sup |
|--------|--------------|------|--------|--------|
| 5 | LinearSVC | 100.00% | 100.00% | 100.00% |
| 10 | LinearSVC | 100.00% | 100.00% | 100.00% |
| 15 | LinearSVC | 100.00% | 100.00% | 100.00% |
| 20 | LinearSVC | 100.00% | 100.00% | 100.00% |
| 24 | LinearSVC | 100.00% | 100.00% | 100.00% |
| 50 | LinearSVC | 100.00% | 100.00% | 100.00% |

## E4 — Sobreposição de features TF-IDF (Jaccard)

_Corpus de referência: 8053 amostras PAN 2012_

| N msgs | Jaccard | Comuns | Só PAN | Só Sintético |
|--------|---------|--------|--------|--------------|
| 5 | 0.0000 | 0 | 20 | 20 |
| 10 | 0.0000 | 0 | 20 | 20 |
| 15 | 0.0000 | 0 | 20 | 20 |
| 20 | 0.0000 | 0 | 20 | 20 |
| 24 | 0.0000 | 0 | 20 | 20 |
| 50 | 0.0000 | 0 | 20 | 20 |

> **Diagnóstico**: Jaccard ≈ 0 indica vocabulário não sobreposto entre PAN 2012 e corpus sintético nessa corrida.

## E5 — Augmentation (PAN + sintético → avalia PAN test)

### Corrida de referência (total=6450)

| N msgs | Classificador | delta_pp (pp) | F0.5 |
|--------|--------------|---------------|------|
| 5 | LinearSVC | -0.1747 | 78.81% |
| 10 | LinearSVC | 0.0000 | 88.16% |
| 15 | LinearSVC | 0.1609 | 91.90% |
| 20 | LinearSVC | 0.0000 | 93.25% |
| 24 | LinearSVC | -0.3026 | 94.18% |
| 50 | LinearSVC | -0.4583 | 95.21% |

### Faixa de delta_pp por tamanho do corpus de teste

| Total | delta_pp mín | delta_pp máx | N linhas |
|-------|-------------|-------------|---------|
| 6450 | -0.4583 | 0.1609 | 6 |

> **Nota (inconclusivo)**: A faixa de delta_pp varia amplamente entre tamanhos de corpus de teste (de negativos a positivos), indicando que o efeito do augmentation é instável nessa rodada. Resultado não conclusivo.

## Síntese

### O que sustenta

- **E3 (LOO)**: F0.5 entre 100.00% e 100.00% na corrida final de referência (total=41), sugerindo alta discriminabilidade interna do corpus sintético; não implica generalização externa.
- **E1 (baseline PAN)**: LinearSVC sem subamostragem atinge F0.5 > 95% em N ≥ 20 mensagens, confirmando a replicação do resultado de Panzariello (2022).

### O que é diagnóstico

- **E2 (cross-domain)**: F0.5 baixo/nulo para o corpus maior (8034 amostras) confirma lacuna de domínio — vocabulário PAN não transfere para o sintético.
- **E4 (Jaccard)**: Sobreposição de features TF-IDF ≈ 0 explica o resultado do E2; os dois corpora compartilham pouquíssimas features relevantes.

### O que está em aberto

- **E5 (augmentation)**: delta_pp oscila entre negativo e positivo dependendo do tamanho do corpus de teste. Resultado inconclusivo — requer mais rodadas com corpus sintético maior e balanceamento controlado.
- Cross-domain com corpus sintético em português precisa de modelo calibrado para PT-BR; o PAN 2012 é em inglês.
