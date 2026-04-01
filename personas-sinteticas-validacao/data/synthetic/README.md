# Corpus Sintético

Esta pasta contém as conversas sintéticas geradas a partir das personas fictícias.

## Formato

Cada arquivo segue o formato XML compatível com a base PAN 2012:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<conversations>
  <conversation id="conv-001">
    <message line="1" author="PRED-001" time="2024-01-15T14:30:00">
      texto da mensagem
    </message>
  </conversation>
</conversations>
```

## Arquivos

- `conversas_predatorias.xml` — conversas entre predadores e vítimas (a ser gerado)
- `conversas_neutras.xml` — conversas normais de controle (a ser gerado)
- `ground_truth.txt` — IDs dos predadores no corpus sintético (a ser gerado)
