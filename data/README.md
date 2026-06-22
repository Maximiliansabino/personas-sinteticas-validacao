# Dados locais

Este diretorio define a estrutura esperada para reexecutar os experimentos.
Os dados brutos e processados nao sao versionados.

Use:

```text
data/
├── pan2012/
├── synthetic/
└── processed/
```

- `pan2012/`: arquivos locais da base PAN 2012 obtidos pelos canais oficiais.
- `synthetic/`: XMLs sinteticos gerados localmente.
- `processed/`: parquets e CSVs intermediarios gerados pelo pipeline.

Nao inclua dados sensiveis, credenciais ou bases restritas no repositorio.

