# Ablation Study: Coarse Prediction Impact Analysis

## 📋 Visão Geral

Este estudo de ablação avalia o impacto da etapa de **coarse prediction** (predição grosseira) no desempenho da rede neural esparsa para simulações de fluidos. O objetivo é quantificar o trade-off entre **tempo de inferência** e **acurácia** ao comparar duas estratégias:

### Estratégias Comparadas

#### 1️⃣ **WITH Coarse Prediction** (Processo de 2 Estágios)
- **Coarse Prediction**: Primeira passagem rápida para identificar a região de interesse
- **Fine Prediction**: Segunda passagem refinada apenas na região identificada
- **Vantagem Esperada**: Menor tempo de processamento ao focar apenas em regiões relevantes
- **Desvantagem Esperada**: Overhead de dois estágios + possível perda de acurácia

#### 2️⃣ **WITHOUT Coarse Prediction** (Processo de 1 Estágio)
- **Fine Prediction Direta**: Predição refinada em toda a região desde o início
- **Vantagem Esperada**: Potencialmente maior acurácia por processar tudo diretamente
- **Desvantagem Esperada**: Maior tempo de processamento

---

## 📂 Estrutura de Arquivos

```
voxel-fluid-net/
├── ablation_coarse_prediction_analysis.py    # Script principal de análise
├── ABLATION_COARSE_PREDICTION_README.md      # Esta documentação
├── reports/
│   ├── predictions_comparison_report/
│   │   └── predictions_comparison.csv        # Dados de entrada
│   └── ablation_coarse_prediction/           # Saída gerada
│       ├── coarse_prediction_comparison.csv  # Tabela detalhada
│       ├── coarse_prediction_summary.csv     # Tabela resumida
│       ├── time_vs_f1_scatter.png           # Gráfico 1
│       ├── time_vs_combined_metric_scatter.png  # Gráfico 2
│       ├── time_comparison_bars.png         # Gráfico 3
│       ├── speedup_factor.png               # Gráfico 4
│       ├── f1_difference.png                # Gráfico 5
│       ├── time_breakdown.png               # Gráfico 6
│       └── tradeoff_analysis.png            # Gráfico 7
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.x
- Bibliotecas: `pandas`, `numpy`, `matplotlib`, `seaborn`
- Arquivo de entrada: `reports/predictions_comparison_report/predictions_comparison.csv`

### Instalação das Dependências

```bash
# Se usando ambiente conda (recomendado)
conda activate vfnet

# Ou instalar manualmente (se necessário)
pip install pandas numpy matplotlib seaborn
```

### Execução

```bash
# Opção 1: Com ambiente conda
conda run -n vfnet python ablation_coarse_prediction_analysis.py

# Opção 2: Com ambiente ativado
conda activate vfnet
python ablation_coarse_prediction_analysis.py

# Opção 3: Python direto (se dependências instaladas globalmente)
python ablation_coarse_prediction_analysis.py
```

### Saída Esperada

```
================================================================================
ABLATION STUDY: COARSE PREDICTION IMPACT
================================================================================

Total rows: 10
Unique simulations: 5

================================================================================
SUMMARY STATISTICS
================================================================================

Number of simulations compared: 5

Time Analysis:
  Average total time WITH coarse: 4.100s
  Average total time WITHOUT coarse: 2.468s
  Average time difference: -1.633s (-39.6%)
  Average speedup factor: 1.771x

Accuracy Analysis (F1-Score):
  Average F1 WITH coarse: 0.9555
  Average F1 WITHOUT coarse: 0.9532
  Average F1 difference: -0.0023

...
```

---

## 📊 Dados de Entrada

### Formato do CSV

O arquivo `predictions_comparison.csv` deve conter as seguintes colunas:

| Coluna | Descrição | Exemplo |
|--------|-----------|---------|
| `simulation` | Nome da simulação | `db_blocks_3d_big_res` |
| `prediction` | Nome do modelo/predição | `pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_2_kfold3` |
| `recall` | Taxa de recall (0-1) | `0.9836` |
| `precision` | Precisão (0-1) | `0.9852` |
| `tnr` | True Negative Rate | `0.9987` |
| `f1_score` | F1-Score | `0.9844` |
| `combined_metric` | Métrica combinada customizada | `0.9823` |
| `matthews_coeff` | Coeficiente de Matthews | `0.983` |
| `coarse_prediction_time` | Tempo da predição grosseira (s) | `3.791` |
| `fine_prediction_time` | Tempo da predição fina (s) | `1.490` |
| `total_time` | Tempo total (s) | `5.281` |

### Identificação de Modelos

- **Com Coarse**: Modelos sem sufixo especial
- **Sem Coarse**: Modelos com sufixo `_no_coarse`

**Exemplo de Pareamento:**
```
pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_2_kfold3
pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_2_kfold3_no_coarse
```

---

## 📈 Saídas Geradas

### 1. Tabelas CSV

#### `coarse_prediction_comparison.csv`
Tabela detalhada com todas as métricas:
- Métricas individuais com e sem coarse
- Tempos detalhados (coarse, fine, total)
- Diferenças absolutas e relativas
- Fator de speedup

#### `coarse_prediction_summary.csv`
Tabela resumida focada nas métricas principais:
- Tempos totais
- Percentual de economia
- F1-Score e Combined Metric
- Diferenças calculadas

### 2. Visualizações (7 Gráficos)

#### **Gráfico 1: `time_vs_f1_scatter.png`**
- **Tipo**: Scatter plot com setas
- **Eixo X**: Tempo total de inferência (segundos)
- **Eixo Y**: F1-Score
- **Interpretação**: Setas mostram a transição de cada simulação (com → sem coarse)
  - ⬅️ Setas para esquerda = economia de tempo
  - ⬆️ Setas para cima = ganho de acurácia
  - ⬇️ Setas para baixo = perda de acurácia

#### **Gráfico 2: `time_vs_combined_metric_scatter.png`**
- Similar ao Gráfico 1, mas usando Combined Metric ao invés de F1-Score

#### **Gráfico 3: `time_comparison_bars.png`**
- **Tipo**: Gráfico de barras lado a lado
- **Cores**: 
  - 🔵 Azul = Com coarse prediction
  - 🟣 Roxo = Sem coarse prediction
- **Interpretação**: Comparação direta de tempo por simulação

#### **Gráfico 4: `speedup_factor.png`**
- **Tipo**: Gráfico de barras
- **Métrica**: Speedup Factor = `Time_WITH / Time_WITHOUT`
- **Cores**:
  - 🟢 Verde: Speedup < 1 (WITHOUT é mais rápido)
  - 🔴 Vermelho: Speedup > 1 (WITH é mais rápido)
- **Linha de Referência**: Linha tracejada em y=1 (sem diferença)

#### **Gráfico 5: `f1_difference.png`**
- **Tipo**: Gráfico de barras
- **Métrica**: Diferença F1 = `F1_WITHOUT - F1_WITH`
- **Cores**:
  - 🟢 Verde: Diferença positiva (sem coarse é melhor)
  - 🔴 Vermelho: Diferença negativa (com coarse é melhor)

#### **Gráfico 6: `time_breakdown.png`**
- **Tipo**: Gráficos de barras empilhadas (2 painéis)
- **Painel Esquerdo** (COM Coarse):
  - 🟠 Laranja: Tempo coarse prediction
  - 🔵 Azul: Tempo fine prediction
- **Painel Direito** (SEM Coarse):
  - 🟣 Roxo: Tempo fine prediction only
- **Interpretação**: Decomposição temporal dos dois processos

#### **Gráfico 7: `tradeoff_analysis.png`** 🔥 **Mais Importante**
- **Tipo**: Scatter plot com quadrantes
- **Eixo X**: Economia de tempo (%) - valores negativos = mais rápido sem coarse
- **Eixo Y**: Mudança no F1-Score (%) - valores positivos = melhor sem coarse
- **Cor dos pontos**: Fator de speedup (escala de cores)
- **Quadrantes**:
  ```
  ┌────────────────┬────────────────┐
  │ Mais rápido    │ Mais lento     │
  │ E melhor       │ MAS melhor     │
  │ ✅ IDEAL       │ ⚠️ Indesejável │
  ├────────────────┼────────────────┤
  │ Mais rápido    │ Mais lento     │
  │ MAS pior       │ E pior         │
  │ ⚡ Trade-off   │ ❌ Pior caso   │
  └────────────────┴────────────────┘
  ```

---

## 🔍 Interpretação dos Resultados

### Métricas Chave

#### 1. **Speedup Factor**
```
Speedup = Time_WITH_coarse / Time_WITHOUT_coarse
```
- **< 1**: Remover coarse é mais rápido
- **= 1**: Sem diferença
- **> 1**: Manter coarse é mais rápido

#### 2. **Time Difference (Percentage)**
```
Time_diff (%) = ((Time_WITHOUT - Time_WITH) / Time_WITH) × 100
```
- **Negativo**: Economia de tempo ao remover coarse
- **Positivo**: Aumento de tempo ao remover coarse

#### 3. **F1 Difference**
```
F1_diff = F1_WITHOUT - F1_WITH
```
- **Positivo**: Ganho de acurácia ao remover coarse
- **Negativo**: Perda de acurácia ao remover coarse

### Cenários de Decisão

#### ✅ **Remover Coarse Prediction se:**
1. `time_diff_percentage < -20%` (economia significativa de tempo)
2. `|f1_diff| < 0.01` (impacto desprezível na acurácia)
3. Todos os pontos no quadrante inferior esquerdo do gráfico 7

#### ⚠️ **Manter Coarse Prediction se:**
1. `f1_diff < -0.05` (perda significativa de acurácia)
2. `time_diff_percentage > 0` (aumento de tempo ao remover)
3. Pontos no quadrante superior direito do gráfico 7

#### 🤔 **Avaliar Trade-off se:**
- Economia de tempo moderada com perda moderada de acurácia
- Depende dos requisitos da aplicação (tempo vs acurácia)

---

## 📊 Exemplo de Resultados Reais

```
================================================================================
PRINCIPAIS DESCOBERTAS (Baseado nos Resultados Atuais)
================================================================================

⏱️ TEMPO DE INFERÊNCIA:
   • COM coarse:    4.10s (média)
   • SEM coarse:    2.47s (média)
   • Economia:      -39.6% (1.77x mais rápido)

🎯 ACURÁCIA (F1-Score):
   • COM coarse:    0.9555
   • SEM coarse:    0.9532
   • Diferença:     -0.0023 (perda de apenas 0.23%)

📈 MÉTRICA COMBINADA:
   • COM coarse:    0.9670
   • SEM coarse:    0.9731
   • Diferença:     +0.0061 (melhora de 0.61%!)

💡 CONCLUSÃO:
   Remover a coarse prediction é VANTAJOSO:
   ✅ 40% mais rápido
   ✅ Impacto mínimo no F1 (0.23% de perda)
   ✅ Melhora na métrica combinada
   
   Recomendação: Usar WITHOUT coarse prediction
```

---

## 🛠️ Personalização

### Modificar Métricas de Comparação

Para adicionar ou modificar métricas, edite o dicionário `result` no loop:

```python
result = {
    'simulation': sim,
    # Adicione suas métricas aqui
    'nova_metrica_with_coarse': wc['nova_metrica'],
    'nova_metrica_without_coarse': woc['nova_metrica'],
    'nova_metrica_diff': woc['nova_metrica'] - wc['nova_metrica'],
}
```

### Adicionar Novos Gráficos

Adicione após a seção de visualizações:

```python
# Seu novo gráfico
fig, ax = plt.subplots(figsize=(12, 8))
# ... código do gráfico ...
plt.savefig(output_dir / "novo_grafico.png", dpi=300, bbox_inches='tight')
```

### Alterar Diretório de Saída

Modifique a variável `output_dir`:

```python
output_dir = Path("seu/novo/caminho/output")
```

---

## 🐛 Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'pandas'"

**Solução:**
```bash
conda activate vfnet  # ou pip install pandas matplotlib seaborn
```

### Erro: "File not found: predictions_comparison.csv"

**Verificações:**
1. O arquivo existe em `reports/predictions_comparison_report/`?
2. O caminho no script está correto?
3. Execute a partir do diretório raiz do projeto

### Avisos: "FixedFormatter should only be used..."

**Status:** ⚠️ Aviso inofensivo
- Não afeta os resultados
- Gráficos são gerados corretamente
- Relacionado ao matplotlib; pode ser ignorado

### Gráficos não aparecem

**Solução:**
- Gráficos são salvos automaticamente em PNG
- Não abrem em janelas interativas
- Verifique o diretório: `reports/ablation_coarse_prediction/`

---

## 📚 Referências

### Métricas de Avaliação

- **F1-Score**: Média harmônica de precisão e recall
  - `F1 = 2 × (precision × recall) / (precision + recall)`

- **Combined Metric**: Métrica customizada do projeto
  - Combina múltiplos aspectos de desempenho

- **Matthews Coefficient**: Correlação entre predições e valores reais
  - Valores entre -1 (pior) e +1 (melhor)

### Estratégia Coarse-to-Fine

A estratégia coarse-to-fine é comum em visão computacional:
1. **Coarse stage**: Identificação rápida de regiões relevantes
2. **Fine stage**: Processamento detalhado apenas nas regiões identificadas

**Referências na Literatura:**
- Cascade networks (Viola-Jones, 2001)
- Hierarchical feature pyramids
- Region-based CNNs (R-CNN family)

---

## 👥 Autor e Manutenção

- **Projeto**: VoxelFluidNet
- **Data de Criação**: 21 de Junho de 2026
- **Última Atualização**: 21 de Junho de 2026

---

## 📄 Licença

Este script faz parte do projeto VoxelFluidNet. Consulte a licença do projeto principal.

---

## 🙏 Contribuições

Para contribuir com melhorias nesta análise:
1. Adicione novos gráficos ou métricas
2. Melhore a documentação
3. Otimize o código
4. Reporte bugs ou comportamentos inesperados

---

**Nota:** Este README documenta especificamente o estudo de ablação de coarse prediction. Para outras análises de ablação (ex: grid length), consulte os respectivos READMEs.
