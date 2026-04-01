# Guia de Criação de Personas Fictícias

## Visão Geral

Este guia descreve a metodologia para criação de personas fictícias que geram dados textuais sintéticos para domínios com restrição de dados reais.

## Princípios Fundamentais

1. **Nenhum dado real:** Todos os nomes, idades, nicknames e conteúdos devem ser inteiramente fictícios.
2. **Base na literatura:** Os comportamentos devem ser fundamentados em pesquisas publicadas (ex: Olson et al., 2007 para grooming).
3. **Diversidade:** As personas devem cobrir diferentes perfis, estratégias e padrões linguísticos.
4. **Rastreabilidade:** Cada decisão de escrita deve poder ser justificada pela ficha da persona.
5. **Revisão cruzada:** Cada conversa deve ser revisada por um integrante diferente do que a escreveu.

## Processo de Criação

### Passo 1: Preencher a Ficha
- Use o template em `template_persona.json`
- Todos os campos marcados como obrigatórios devem ser preenchidos
- Para predadores: definir quais fases de grooming a persona executa
- Para vítimas: definir vulnerabilidades e como reagem às abordagens
- Para neutros: definir temas de conversa e interesses

### Passo 2: Definir o Guia de Estilo Linguístico
Para cada persona, documente:
- **Nível de formalidade:** escala de 1 (muito informal) a 5 (formal)
- **Uso de abreviações:** quais e com que frequência (ex: vc, tb, blz)
- **Emojis:** com que frequência e quais tipos
- **Erros ortográficos:** intencionais ou não, quais tipos
- **Comprimento de mensagem:** curtas (1-5 palavras), médias (5-15), longas (15+)
- **Pontuação:** usa ou não, tipos preferidos

### Passo 3: Gerar as Conversas
- Cada conversa deve ter entre 10 e 50 mensagens
- Conversas predatórias devem seguir a progressão das fases de grooming
- Manter consistência com a ficha durante toda a conversa
- Incluir variações naturais (nem toda mensagem precisa ser "perfeita")

### Passo 4: Revisão e Validação
- Verificar consistência com a ficha (usar `src/persona_validator.py`)
- Revisão cruzada entre integrantes da dupla
- Checar se as fases de grooming estão presentes nas conversas predatórias
- Verificar que não há dados identificáveis reais

## Fases do Grooming (Olson et al., 2007)

| Fase | Nome | Descrição | Sinais Linguísticos |
|------|------|-----------|---------------------|
| 1 | Seleção | Escolha da vítima, abordagem inicial | Perguntas sobre idade, localização, situação familiar |
| 2 | Ganho de confiança | Criar vínculo emocional | Elogios, interesses em comum, empatia excessiva |
| 3 | Isolamento | Afastar de figuras protetoras | "Não conta pra ninguém", migrar para chat privado |
| 4 | Dessensibilização | Introduzir conteúdo sexual gradualmente | Assuntos íntimos, pedidos de fotos, normalização |
| 5 | Manutenção do segredo | Garantir silêncio da vítima | Ameaças veladas, chantagem emocional, culpabilização |

## Formato XML das Conversas

```xml
<?xml version="1.0" encoding="UTF-8"?>
<conversations>
  <conversation id="conv-001">
    <message line="1" author="PRED-001" time="2024-01-15T14:30:00">
      oi tudo bem? vi seus desenhos, são muito legais
    </message>
    <message line="2" author="VIT-001" time="2024-01-15T14:31:00">
      oii obrigada!! vc desenha tb?
    </message>
    <!-- ... -->
  </conversation>
</conversations>
```

## Checklist de Qualidade

- [ ] Ficha completa com todos os campos obrigatórios
- [ ] Guia de estilo linguístico documentado
- [ ] Conversa com 10-50 mensagens
- [ ] Fases de grooming presentes (para predatórias)
- [ ] Consistência com a ficha em todas as mensagens
- [ ] Nenhum dado real identificável
- [ ] Revisão cruzada realizada
- [ ] Formato XML válido
