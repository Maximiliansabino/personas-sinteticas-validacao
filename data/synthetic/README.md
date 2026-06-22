# Corpus sintetico

Este diretorio recebe XMLs sinteticos gerados localmente pelo pipeline de
personas. Os XMLs nao sao versionados por padrao.

Quando os XMLs estiverem presentes, o parquet usado nos experimentos pode ser
gerado com:

```bash
python -m src.evaluate \
  --create-synth data/synthetic \
  --synth data/processed/synthetic.parquet \
  --split-synth \
  --synth-train data/processed/synthetic_train.parquet \
  --synth-test data/processed/synthetic_test.parquet
```

