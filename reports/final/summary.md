# Resultados consolidados dos experimentos do artigo

_Relatório gerado automaticamente por `src/summarize_results.py` a partir dos CSVs em `reports/final/`._

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

_Corrida com maior corpus: 8034 amostras (859 predatórias, 7175 normais)_

| N msgs | Classificador | F0.5 | Precisão | Revocação |
|--------|--------------|------|----------|-----------|
| 5 | LinearSVC | 0.00% | 0.00% | 0.00% |
| 10 | LinearSVC | 50.00% | 100.00% | 16.67% |
| 15 | LinearSVC | 50.00% | 100.00% | 16.67% |
| 20 | LinearSVC | 50.00% | 100.00% | 16.67% |
| 24 | LinearSVC | 50.00% | 100.00% | 16.67% |
| 50 | LinearSVC | 0.00% | 0.00% | 0.00% |

## E3 — Leave-one-out no corpus sintético (IC 95% bootstrap)

_Corrida de referência: total=23 (8 predatórias, 15 normais)_

| N msgs | Classificador | F0.5 | IC inf | IC sup |
|--------|--------------|------|--------|--------|
| 5 | LinearSVC | 93.75% | 76.92% | 100.00% |
| 10 | LinearSVC | 93.75% | 78.95% | 100.00% |
| 15 | LinearSVC | 93.75% | 76.92% | 100.00% |
| 20 | LinearSVC | 93.75% | 76.92% | 100.00% |
| 24 | LinearSVC | 93.75% | 76.92% | 100.00% |
| 50 | LinearSVC | 97.22% | 87.49% | 100.00% |

## E4 — Sobreposição de features TF-IDF (Jaccard)

_Corpus de referência: 8034 amostras PAN 2012_

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

### Corrida de referência (total=6431)

| N msgs | Classificador | delta_pp (pp) | F0.5 |
|--------|--------------|---------------|------|
| 5 | LinearSVC | -0.6693 | 78.31% |
| 10 | LinearSVC | -0.1826 | 87.98% |
| 15 | LinearSVC | 0.4480 | 92.19% |
| 20 | LinearSVC | -0.1566 | 93.09% |
| 24 | LinearSVC | -0.1505 | 94.33% |
| 50 | LinearSVC | -0.6017 | 95.07% |

### Faixa de delta_pp por tamanho do corpus de teste

| Total | delta_pp mín | delta_pp máx | N linhas |
|-------|-------------|-------------|---------|
| 400 | 0.0000 | 0.0000 | 14 |
| 409 | -8.7413 | -8.7413 | 1 |
| 410 | 0.0000 | 8.9744 | 6 |
| 412 | -8.7413 | -8.7413 | 1 |
| 414 | 0.0000 | 0.0000 | 6 |
| 415 | 0.0000 | 12.2378 | 6 |
| 418 | 0.0000 | 12.2378 | 6 |
| 420 | -8.7413 | 0.0000 | 6 |
| 423 | 0.0000 | 12.2378 | 6 |
| 430 | -8.7413 | 0.0000 | 12 |
| 510 | 0.0000 | 0.0000 | 5 |
| 512 | -9.0909 | 0.0000 | 5 |
| 515 | 0.0000 | 28.5714 | 5 |
| 6431 | -0.6693 | 0.4480 | 6 |

> **Nota (inconclusivo)**: A faixa de delta_pp varia amplamente entre tamanhos de corpus de teste (de negativos a positivos), indicando que o efeito do augmentation é instável nessa rodada. Resultado não conclusivo.

## Síntese

### O que sustenta

- **E3 (LOO)**: F0.5 entre 93–97% com IC 95% bem acima de 0 na corrida de referência (total=23), sugerindo que o classificador generaliza dentro do corpus sintético.
- **E1 (baseline PAN)**: LinearSVC sem subamostragem atinge F0.5 > 95% em N ≥ 20 mensagens, confirmando a replicação do resultado de Panzariello (2022).

### O que é diagnóstico

- **E2 (cross-domain)**: F0.5 baixo/nulo para o corpus maior (8034 amostras) confirma lacuna de domínio — vocabulário PAN não transfere para o sintético.
- **E4 (Jaccard)**: Sobreposição de features TF-IDF ≈ 0 explica o resultado do E2; os dois corpora compartilham pouquíssimas features relevantes.

### O que está em aberto

- **E5 (augmentation)**: delta_pp oscila entre negativo e positivo dependendo do tamanho do corpus de teste. Resultado inconclusivo — requer mais rodadas com corpus sintético maior e balanceamento controlado.
- Cross-domain com corpus sintético em português precisa de modelo calibrado para PT-BR; o PAN 2012 é em inglês.
