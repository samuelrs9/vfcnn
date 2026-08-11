# Comparação de Predições - Documentação

## Visão Geral

O script `compare_predictions.py` compara múltiplas predições, gerando um relatório com:
- Métricas de classificação (recall, precision, tnr, combined_metric, f1_score, matthews_coeff)
- Tempos de inferência (coarse_prediction, fine_prediction, total)
- Identificação do ground-truth usado para cada predição
- Sistema de cache para evitar recálculos desnecessários

## Formato de Configuração

### Formato Atual (Recomendado)

Usa `comparisons` para agrupar predições por ground-truth:

```yaml
comparisons:
  # Comparação 1: Múltiplas predições contra o mesmo GT
  - gt_config_file: /caminho/para/gt_config.yaml
    pred_config_files:
      - /caminho/para/pred1/pred_config.yaml
      - /caminho/para/pred2/pred_config.yaml
      - /caminho/para/pred3/pred_config.yaml
  
  # Comparação 2 (opcional): Mesmas predições contra GT alternativo
  - gt_config_file: /caminho/para/gt_alternativo.yaml
    pred_config_files:
      - /caminho/para/pred1/pred_config.yaml
      - /caminho/para/pred2/pred_config.yaml

report:
  output_path: ./reports/predictions_comparison_report
  format: csv  # ou 'npz'
  plot_metrics: false
  print_metrics: true
```

**Vantagens:**
- Agrupa predições que devem ser comparadas contra o mesmo ground-truth
- Permite comparar as mesmas predições contra múltiplos GTs em um único relatório
- Formato mais limpo e organizado

## Sistema de Cache

O script implementa um sistema inteligente de cache para evitar recálculos:

### Métricas (`metrics_report.csv`)
- **Existe**: Usa o cache, exibe mensagem "Using cached metrics"
- **Não existe**: Calcula usando o `gt_config_file` especificado

### Tempos (`time_report.csv`)
- **Existe**: Lê os tempos do arquivo
- **Não existe**: Usa zeros e exibe warning (tempos são gerados durante predição)

**Benefícios:**
- Execução mais rápida quando métricas já foram calculadas
- Evita processamento desnecessário de arquivos grandes
- Permite recalcular apenas quando necessário (delete o CSV para forçar recálculo)

O relatório CSV contém as seguintes colunas:

| Coluna | Descrição |
|--------|-----------|
| simulation | Nome da simulação |
| ground_truth | GT usado ("pre-calculated" ou nome do arquivo) |
| prediction | Nome da predição |
| recall | Taxa de verdadeiros positivos |
| precision | Valor preditivo positivo |
| tnr | Taxa de verdadeiros negativos |
| combined_metric | Métrica combinada |
| f1_score | F1 score |
| matthews_coeff | Coeficiente de Matthews |
| coarse_prediction_time | Tempo de predição grosseira (s) |
| fine_prediction_time | Tempo de predição fina (s) |
| total_time | Tempo total (s) |

## Exemplos de Uso

### Exemplo 1: Comparar predições contra mesmo ground-truth
```yaml
comparisons:
  - gt_config_file: /caminho/para/gt_config.yaml
    pred_config_files:
      - /caminho/pred1/pred_config.yaml
      - /caminho/pred2/pred_config.yaml
```

### Exemplo 2: Comparar mesmas predições contra múltiplos ground-truths
```yaml
comparisons:
  # GT versão 1
  - gt_config_file: /caminho/gt_v1.yaml
    pred_config_files:
      - /caminho/pred1/pred_config.yaml
      - /caminho/pred2/pred_config.yaml
  
  # GT versão 2 (ex: diferente hdp)
  - gt_config_file: /caminho/gt_v2.yaml
    pred_config_files:
      - /caminho/pred1/pred_config.yaml
      - /caminho/pred2/pred_config.yaml
```

Executar:
```bash
python scripts/reports/compare_predictions.py scripts/configs/reports/compare_predictions.yaml
```

## Casos de Uso

1. **Validação Cruzada**: Comparar diferentes folds de treinamento contra mesmo GT
2. **Comparação de GTs**: Avaliar mesmas predições contra diferentes ground-truths
   - Exemplo: comparar com diferentes valores de `hdp` (smoothing length)
   - Avaliar sensibilidade das predições a diferentes definições de GT
3. **Análise de Performance**: Comparar tempos de inferência entre modelos
4. **Ablation Studies**: Comparar variantes do mesmo modelo (ex: com/sem coarse prediction)
5. **Múltiplas Simulações**: Comparar predições de diferentes simulações em um único relatório

## Notas Importantes

- O `sim_config_file` é usado para inicializar o DataReader e obter o nome da simulação
- O `gt_config_file` é usado para calcular/recalcular as métricas de classificação
- Métricas são sempre calculadas/lidas do diretório de cada predição individual
- Sistema de cache: delete `metrics_report.csv` para forçar recálculo com novo GT
- `time_report.csv` deve existir (gerado durante predição), caso contrário usa zeros
- Todas as predições em uma comparação são agrupadas com o mesmo `gt_config_file`

## Migração de Formatos Antigos

Se você tem configurações antigas, converta para o novo formato:

**Antigo:**
```yaml
sim_config_file: /caminho/sim_config.yaml
pred_config_files:
  - /caminho/pred1/pred_config.yaml
  - /caminho/pred2/pred_config.yaml
```

**Novo:**
```yaml
comparisons:
  - gt_config_file: /caminho/gt_config.yaml
    pred_config_files:
      - /caminho/pred1/pred_config.yaml
      - /caminho/pred2/pred_config.yaml
```

**Nota:** O nome da simulação é extraído automaticamente do nome do diretório onde o `gt_config_file` está localizado.
