# Ablation Study: Sparse Regionwise Tasks

Este diretório contém scripts para executar um estudo de ablação comparando o desempenho de inferência das tarefas `boundary`, `normal` e `boundary+normal` em diferentes simulações.

## Visão Geral

O estudo de ablação testa 3 configurações de tarefas × 5 simulações = 15 experimentos:

### Tarefas
1. **boundary_only**: Apenas classificação de boundary (interior/boundary)
2. **normal_only**: Apenas regressão de normais (3 outputs)
3. **boundary_normal**: Ambas as tarefas simultaneamente

### Simulações
0. `fountain_3d_big_res` (fold 0)
1. `inlet_vortex_3d_big_res` (fold 1)
2. `db_blocks_3d_big_res` (fold 2)
3. `inlet_collision_3d_big_res` (fold 3)
4. `ddb_3d_big_res` (fold 4)

## Arquivos Criados

### Templates de Configuração
- `scripts/configs/prediction/ablation_boundary_only_template.yaml`
- `scripts/configs/prediction/ablation_normal_only_template.yaml`
- `scripts/configs/prediction/ablation_boundary_normal_template.yaml`

### Scripts
- `scripts/run_ablation_study.sh`: Script principal que executa todos os experimentos
- `scripts/analyze_ablation_results.py`: Script de análise e visualização dos resultados

## Como Usar

### 1. Executar o Estudo de Ablação

```bash
cd /work1/voxel-fluid-net
./scripts/run_ablation_study.sh
```

O script irá:
- Criar um diretório de saída em `outputs/ablation_study_YYYYMMDD_HHMMSS/`
- Gerar configs específicos para cada combinação simulação+tarefa
- Executar todas as predições com medição de tempo
- Salvar logs detalhados e resultados de timing em CSV

**Configuração dos Frames:**
Por padrão, o script executa a predição nos frames 0-9 de cada simulação (10 frames totais).
Para modificar, edite as linhas `initial_step` e `final_step` nos templates YAML:
```yaml
prediction:
  initial_step: 0    # primeiro frame
  final_step: 9      # último frame
  skip_steps: 1      # intervalo entre frames
```

### 2. Analisar os Resultados

Após a conclusão do estudo:

```bash
python scripts/analyze_ablation_results.py outputs/ablation_study_YYYYMMDD_HHMMSS/timing_results.csv
```

O script de análise irá gerar:
- **summary_statistics.csv**: Estatísticas resumidas (média, desvio padrão, min, max)
- **detailed_results.csv**: Tabela detalhada com tempos por simulação e tarefa
- **ablation_study_report.md**: Relatório completo em Markdown
- **task_comparison.png**: Gráfico comparando tempo médio por tarefa
- **simulation_comparison.png**: Gráfico comparando todas as combinações
- **heatmap.png**: Mapa de calor dos tempos de inferência

### 3. Visualizar os Resultados

Os resultados são salvos em CSV e podem ser facilmente importados para análise adicional:

```python
import pandas as pd

# Carregar resultados
df = pd.read_csv('outputs/ablation_study_*/timing_results.csv')

# Ver estatísticas por tarefa
print(df.groupby('task')['duration_seconds'].describe())

# Ver estatísticas por simulação
print(df.groupby('simulation')['duration_seconds'].describe())
```

## Estrutura dos Resultados

O diretório de saída contém:

```
outputs/ablation_study_YYYYMMDD_HHMMSS/
├── ablation_study.log              # Log completo da execução
├── timing_results.csv              # Resultados brutos de timing
├── summary_statistics.csv          # Estatísticas resumidas
├── detailed_results.csv            # Resultados detalhados
├── ablation_study_report.md        # Relatório final
├── task_comparison.png             # Visualização: comparação de tarefas
├── simulation_comparison.png       # Visualização: por simulação
├── heatmap.png                     # Visualização: heatmap
└── config_*.yaml                   # Configs gerados para cada experimento
```

## Formato do CSV de Timing

O arquivo `timing_results.csv` contém:

| Campo | Descrição |
|-------|-----------|
| `simulation` | Nome da simulação |
| `task` | Tipo de tarefa (boundary_only, normal_only, boundary_normal) |
| `model_fold` | Fold do modelo k-fold (0-4) |
| `start_time` | Timestamp de início |
| `end_time` | Timestamp de término |
| `duration_seconds` | Duração em segundos |
| `predict_id` | ID único da predição |

## Customização

### Modificar Número de Frames

Edite os templates YAML em `scripts/configs/prediction/ablation_*_template.yaml`:

```yaml
prediction:
  initial_step: 0     # Primeiro frame a processar
  final_step: 49      # Último frame a processar (ex: 50 frames)
  skip_steps: 5       # Processar a cada N frames
```

### Adicionar Novas Simulações

Edite `scripts/run_ablation_study.sh` e adicione à seção de definições:

```bash
SIMULATIONS[nova_simulacao]="/path/to/sim"
MODEL_PATHS[nova_simulacao]="/path/to/model"
FOLD_NUMBERS[nova_simulacao]=5
```

### Modificar Parâmetros de Predição

Edite os templates YAML para ajustar:
- `batch_size`: Tamanho do batch para inferência
- `search_radius`: Raio de busca de vizinhos
- `grid_length`: Tamanho da célula do grid
- `extract_mesh`: Se deve extrair mesh após predição

## Métricas Principais

O estudo foca em **tempo de inferência**, mas os scripts de predição também geram:
- Arquivos de predição em `outputs/pred_*/`
- Relatórios de tempo em cada diretório de predição
- Logs detalhados para debug

## Dependências

Certifique-se de que o ambiente está ativo:

```bash
conda activate vfnet  # ou o nome do seu ambiente
```

Dependências necessárias:
- TensorFlow (configurado no ambiente)
- pandas
- matplotlib
- seaborn

Para instalar dependências de análise:
```bash
pip install pandas matplotlib seaborn tabulate
```

## Troubleshooting

### Erro de GPU/Memória
Se houver problemas de memória GPU, reduza o `batch_size` nos templates:
```yaml
prediction:
  batch_size: 50000  # reduzir de 100000
```

### Erro de Path
Verifique se todos os paths existem:
```bash
# Verificar simulações
ls /work1/voxel-fluid-net/data/3D/simulations/*/sim_config.yaml

# Verificar modelos
ls /work1/voxel-fluid-net/models/kfold3/sparse_regionwise_approach/models/*/model_config_v2.yaml
```

### Continuar Após Falha
O script continua mesmo se uma predição falhar. Verifique o log para detalhes:
```bash
grep "ERROR" outputs/ablation_study_*/ablation_study.log
```

## Exemplo de Uso Completo

```bash
# 1. Executar estudo (pode demorar várias horas)
cd /work1/voxel-fluid-net
./scripts/run_ablation_study.sh

# 2. Após conclusão, analisar resultados
python scripts/analyze_ablation_results.py outputs/ablation_study_20260621_143022/timing_results.csv

# 3. Ver relatório
cat outputs/ablation_study_20260621_143022/ablation_study_report.md

# 4. Abrir visualizações
xdg-open outputs/ablation_study_20260621_143022/task_comparison.png
```

## Notas

- O script usa `date +%s` para medir tempo real de execução
- Cada experimento é independente e pode ser rodado separadamente
- Os resultados são acumulativos - se interromper, pode continuar editando o script
- O tempo de execução total depende do número de frames e complexidade das simulações
