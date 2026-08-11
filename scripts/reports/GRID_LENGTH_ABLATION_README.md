# Grid Length Ablation Study

Este diretório contém scripts para executar um estudo de ablação do parâmetro `grid_length` na rede neural sparse para predição de fluidos.

## Visão Geral

O experimento testa diferentes valores de `grid_length` [0.025, 0.05, 0.1, 0.2, 0.4] em 5 simulações diferentes, medindo:

1. **Tempo de inferência** (normalizado por milhão de partículas)
2. **Utilização de GPU** (média e máxima)
3. **Métricas de acurácia** (comparando predições com ground truth):
   - Recall (TPR - True Positive Rate)
   - Precision (PPV - Positive Predictive Value)
   - TNR (True Negative Rate)
   - F1 Score
   - MCC (Matthews Correlation Coefficient)
   - Combined Metric

## Arquivos

### Scripts Principais

- **`run_grid_length_ablation.sh`**: Script bash para executar o experimento completo
  - Itera sobre 5 simulações × 5 valores de grid_length (25 experimentos)
  - Executa predições com monitoramento de GPU
  - Calcula métricas de acurácia automaticamente
  - Gera CSVs com timing e métricas

- **`compute_accuracy_metrics.py`**: Script Python para calcular métricas de acurácia
  - Compara predições com ground truth (gt_config.yaml)
  - Calcula métricas de classificação para boundary
  - Calcula métricas de regressão para normal
  - Gera relatórios CSV por predição

- **`analyze_grid_length_results.py`**: Script Python para análise e visualização
  - Gera gráficos comparativos (tempo, GPU, acurácia)
  - Cria tabelas de métricas
  - Produz relatório markdown completo
  - Identifica configurações ótimas

### Templates

- **`configs/prediction/ablation_grid_length_template.yaml`**: Template YAML para predições
  - Contém placeholders para parâmetros variáveis
  - Configuração de tasks (boundary + normal)
  - Parâmetro `grid_length` parametrizado

## Estrutura de Dados

### Ground Truth Configuration

Cada simulação deve ter um arquivo `gt_config.yaml` no diretório da simulação:

```yaml
boundary:
  labels:
  - interior
  - boundary
  dir: gt_hdp=1.73
  base_name: gt
  extension: dat
normal:
  dir: normal_sph
  base_name: normal
  extension: csv
  columns: nx ny nz
  method: pca
  initial_distance: 0.008
  search_radius: 2.0
```

### Simulações

O experimento utiliza as seguintes simulações:

1. **fountain_3d_big_res** (fold 0)
   - Frames: 301-302
   - Path: `/work1/voxel-fluid-net/data/3D/simulations/fountain_3d_big_res`

2. **inlet_vortex_3d_big_res** (fold 1)
   - Frames: 199-200
   - Path: `/work1/voxel-fluid-net/data/3D/simulations/inlet_vortex_3d_big_res`

3. **db_blocks_3d_big_res** (fold 2)
   - Frames: 199-200
   - Path: `/work1/voxel-fluid-net/data/3D/simulations/db_blocks_3d_big_res`

4. **inlet_collision_3d_big_res** (fold 3)
   - Frames: 199-200
   - Path: `/work1/voxel-fluid-net/data/3D/simulations/inlet_collision_3d_big_res`

5. **ddb_3d_big_res** (fold 4)
   - Frames: 199-200
   - Path: `/work1/voxel-fluid-net/data/3D/simulations/ddb_3d_big_res`

## Como Executar

### 1. Executar o Experimento

```bash
cd /work1/voxel-fluid-net
./scripts/run_grid_length_ablation.sh
```

Isso irá:
- Criar diretório de output: `outputs/ablation_grid_length_YYYYMMDD_HHMMSS/`
- Executar 25 experimentos (5 simulações × 5 grid_length valores)
- **Usar cache**: Se uma predição já existir, reutiliza os resultados
- Gerar `timing_results.csv` com dados de performance
- Gerar `accuracy_metrics.csv` com métricas de acurácia
- Criar log detalhado: `ablation_study.log`

#### Mecanismo de Cache

O script automaticamente detecta predições existentes para evitar recomputação:

- ✓ **Cache HIT**: Se a pasta de predição existe (`pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_{fold}/{predict_id}/`), o script:
  - Reutiliza os arquivos de predição existentes
  - Extrai timing do `time_report.csv` (se disponível)
  - Calcula ou reutiliza métricas de acurácia
  - Adiciona resultados ao CSV de output
  
- ✗ **Cache MISS**: Se a predição não existe, executa normalmente

#### Forçar Recomputação

Para ignorar o cache e recomputar todas as predições:

```bash
./scripts/run_grid_length_ablation.sh --force-recompute
```

Isso é útil quando:
- Você atualizou o modelo
- Quer refazer as medições de timing com GPU limpa
- Suspeita que predições antigas estão corrompidas

### 2. Analisar Resultados

```bash
python scripts/analyze_grid_length_results.py outputs/ablation_grid_length_YYYYMMDD_HHMMSS/timing_results.csv
```

Ou especificando o arquivo de métricas explicitamente:

```bash
python scripts/analyze_grid_length_results.py \
    outputs/ablation_grid_length_YYYYMMDD_HHMMSS/timing_results.csv \
    --metrics-csv outputs/ablation_grid_length_YYYYMMDD_HHMMSS/accuracy_metrics.csv \
    --output-dir outputs/ablation_grid_length_YYYYMMDD_HHMMSS
```

### 3. Calcular Métricas Manualmente (Opcional)

Se você quiser calcular métricas para uma predição específica:

```bash
python scripts/compute_accuracy_metrics.py \
    /path/to/simulation \
    /path/to/pred_config_v2.yaml \
    --gt-config /path/to/gt_config.yaml
```

## Arquivos de Output

### timing_results.csv

Contém dados de performance para cada experimento:

```csv
simulation,grid_length,model_fold,start_time,end_time,duration_seconds,predict_id,num_particles,frames_processed,avg_gpu_util,max_gpu_util,avg_gpu_mem,max_gpu_mem
fountain_3d_big_res,0.025,0,2026-06-21 10:00:00,2026-06-21 10:00:25,25,gl0.025_fountain_3d_big_res_fold0_20260621,2500000,2,45.2,98.0,1024.5,2048.0
...
```

### accuracy_metrics.csv

Contém métricas de acurácia para cada experimento:

```csv
simulation,grid_length,model_fold,predict_id,avg_recall,avg_precision,avg_tnr,avg_f1,avg_mcc,avg_combined_metric
fountain_3d_big_res,0.025,0,gl0.025_fountain_3d_big_res_fold0_20260621,0.9850,0.9820,0.9950,0.9835,0.9730,0.9873
...
```

### Visualizações Geradas

1. **`grid_length_comparison.png`**
   - 4 subplots: tempo/partícula, GPU util, F1 score, MCC
   - Eixo X logarítmico para grid_length
   - Barras de erro (std)

2. **`simulation_comparison.png`**
   - Linhas separadas por simulação
   - Comparação de tempo e GPU entre simulações

3. **`heatmap.png`**
   - Heatmaps de tempo e GPU utilization
   - Simulações × grid_length

### Relatório Markdown

`grid_length_ablation_report.md` contém:
- Estatísticas resumidas
- Tabelas de performance (tempo, GPU)
- Tabelas de acurácia (F1, MCC, Recall, Precision)
- Análise de key findings
- Recomendações de grid_length ótimo

## Parâmetros Configuráveis

### No Script Bash

Edite `run_grid_length_ablation.sh`:

```bash
# Valores de grid_length a testar
GRID_LENGTHS=(0.025 0.05 0.1 0.2 0.4)

# Frame ranges por simulação
FRAME_RANGES[fountain_3d_big_res]="301:302"
FRAME_RANGES[inlet_vortex_3d_big_res]="199:200"

# Skip steps por simulação
SKIP_STEPS[fountain_3d_big_res]=1
```

### No Template YAML

Edite `configs/prediction/ablation_grid_length_template.yaml`:

- `search_radius`: raio de busca de vizinhos
- `batch_size`: tamanho do batch
- `device`: cpu ou cuda

## Métricas Explicadas

### Classificação (Boundary)

- **Recall (TPR)**: Proporção de partículas boundary corretamente identificadas
- **Precision (PPV)**: Proporção de predições boundary que são corretas
- **TNR**: Proporção de partículas interior corretamente identificadas
- **F1 Score**: Média harmônica de Recall e Precision
- **MCC**: Matthews Correlation Coefficient (correlação entre predição e ground truth)
- **Combined Metric**: Métrica combinada customizada

### Regressão (Normal)

- **MAE**: Mean Absolute Error do ângulo entre normais
- **Std Angle**: Desvio padrão do erro angular

## Interpretação dos Resultados

### Trade-offs

- **Grid length pequeno (0.025)**:
  - ✓ Maior resolução espacial
  - ✓ Potencialmente maior acurácia
  - ✗ Mais lento
  - ✗ Mais uso de memória

- **Grid length grande (0.4)**:
  - ✓ Mais rápido
  - ✓ Menor uso de memória
  - ✗ Menor resolução espacial
  - ✗ Potencialmente menor acurácia

### Métricas Importantes

1. **Time per Million Particles**: Normaliza tempo por quantidade de partículas
2. **F1 Score**: Balanceia precisão e recall
3. **MCC**: Métrica robusta para datasets desbalanceados
4. **GPU Utilization**: Indica eficiência de uso do hardware

## Troubleshooting

### Cache Issues

Se você quer reexecutar um experimento específico:

```bash
# Opção 1: Remover a pasta de predição
rm -rf /path/to/simulation/pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_{fold}/{predict_id}

# Opção 2: Usar --force-recompute para ignorar cache
./scripts/run_grid_length_ablation.sh --force-recompute
```

**Observação**: Com cache habilitado, o timing extraído do `time_report.csv` pode não incluir estatísticas de GPU, pois essas são medidas durante a execução real.

### Ground Truth não encontrado

Se `gt_config.yaml` não existir:
```bash
# O script continua mas não calcula métricas de acurácia
WARNING: Ground truth config not found, skipping metrics
```

### Predição falhou

```bash
ERROR: Failed prediction: sim_name - grid_length=0.1 (fold 0)
WARNING: Continuing after error
```

O experimento continua com os próximos testes.

### Ambiente Python

O script detecta automaticamente:
1. Conda environment `vfnet`
2. Conda environment `vfnet_tf22_cuda101`
3. Python do sistema

Para forçar um ambiente específico:
```bash
PYTHON_CMD="conda run -n vfnet python" ./scripts/run_grid_length_ablation.sh
```

## Dependências

- Python 3.8+
- TensorFlow 2.5.0
- NumPy 1.19.5
- Pandas
- Matplotlib
- Seaborn
- nvidia-smi (para monitoramento de GPU)

## Autores

Desenvolvido para o projeto voxel-fluid-net.
