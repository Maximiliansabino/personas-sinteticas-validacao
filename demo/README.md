# Demo minima do pipeline

Esta pasta contem uma demonstracao minima e offline do fluxo analitico E1-E5.
Ela usa dados pequenos e ficticios em CSV para evitar versionar derivados do PAN
2012 ou qualquer dado real sensivel.

## Conteudo

- `notebooks/demonstracao_pipeline.ipynb`: notebook interativo da demo.
- `data/processed/pan2012_train.csv`: amostra ficticia no formato esperado pelo
  pipeline PAN pre-processado.
- `data/processed/synthetic.csv`: amostra ficticia no formato esperado pelo
  corpus sintetico pre-processado.
- `requirements.txt`: dependencias minimas para executar o notebook.

## Como executar

Execute os comandos a partir desta pasta:

```bash
cd demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/demonstracao_pipeline.ipynb
```

O notebook demonstra carregamento dos dados, cortes temporais, TF-IDF,
LinearSVC, transferencia entre dominios, Leave-One-Out, Jaccard e augmentation.
Os resultados cientificos finais do artigo continuam em `../reports/final/`.
