# 📊 Scripts de Análise e Estudos de Ablação

Este documento lista todos os scripts de análise disponíveis no projeto VoxelFluidNet.

---

## 🔬 Estudos de Ablação

### 1. Coarse Prediction Analysis
**Arquivo:** [`ablation_coarse_prediction_analysis.py`](ablation_coarse_prediction_analysis.py)  
**Documentação:** [`ABLATION_COARSE_PREDICTION_README.md`](ABLATION_COARSE_PREDICTION_README.md)

**Objetivo:** Comparar o desempenho da rede neural esparsa com e sem a etapa de coarse prediction.

**Entrada:**
- `reports/predictions_comparison_report/predictions_comparison.csv`

**Saída:**
- `reports/ablation_coarse_prediction/`
  - 2 tabelas CSV (comparação detalhada e resumida)
  - 7 gráficos de visualização

**Como Executar:**
```bash
conda run -n vfnet python ablation_coarse_prediction_analysis.py
```

**Principais Métricas Comparadas:**
- ⏱️ Tempo de inferência (coarse + fine vs fine only)
- 🎯 F1-Score
- 📈 Combined Metric
- 🚀 Speedup Factor

**Resultado Resumido:**
- **Tempo:** 40% mais rápido sem coarse prediction
- **Acurácia:** Impacto mínimo (0.23% de perda no F1)
- **Recomendação:** Usar sem coarse prediction

---

## 📋 Lista Completa de Scripts

### Scripts de Análise
| Script | Descrição | Documentação |
|--------|-----------|--------------|
| `ablation_coarse_prediction_analysis.py` | Análise de ablação: coarse prediction | [README](ABLATION_COARSE_PREDICTION_README.md) |
| `scripts/analyze_grid_length_results.py` | Análise de grid length | Ver `GRID_LENGTH_ABLATION_README.md` |

### Scripts de Utilidade
| Script | Descrição |
|--------|-----------|
| `fix_model_config_paths.py` | Corrige paths em configs de modelo |
| `fix_pred_config_paths.py` | Corrige paths em configs de predição |
| `update_pred_config_format.py` | Atualiza formato de configs |

### Scripts Principais
| Script | Descrição |
|--------|-----------|
| `main_tutorials.py` | Tutoriais principais do projeto |

---

## 🚀 Quick Start

### 1. Ativar Ambiente
```bash
conda activate vfnet
```

### 2. Executar Análise de Coarse Prediction
```bash
python ablation_coarse_prediction_analysis.py
```

### 3. Visualizar Resultados
```bash
# Listar arquivos gerados
ls -lh reports/ablation_coarse_prediction/

# Abrir gráficos
xdg-open reports/ablation_coarse_prediction/tradeoff_analysis.png
xdg-open reports/ablation_coarse_prediction/time_vs_f1_scatter.png
```

---

## 📊 Estrutura de Saídas

```
reports/
├── ablation_coarse_prediction/
│   ├── coarse_prediction_comparison.csv
│   ├── coarse_prediction_summary.csv
│   ├── time_vs_f1_scatter.png
│   ├── time_vs_combined_metric_scatter.png
│   ├── time_comparison_bars.png
│   ├── speedup_factor.png
│   ├── f1_difference.png
│   ├── time_breakdown.png
│   └── tradeoff_analysis.png
├── predictions_comparison_report/
│   └── predictions_comparison.csv
└── ...
```

---

## 🔧 Requisitos

### Ambiente Python
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn

### Ativação do Ambiente
```bash
# Opção 1: Conda (recomendado)
conda activate vfnet

# Opção 2: Verificar dependências
python -c "import pandas, numpy, matplotlib, seaborn; print('OK')"
```

---

## 📖 Documentação Adicional

- **Ablation Study - Coarse Prediction:** [ABLATION_COARSE_PREDICTION_README.md](ABLATION_COARSE_PREDICTION_README.md)
- **Ablation Study - Grid Length:** [GRID_LENGTH_ABLATION_README.md](GRID_LENGTH_ABLATION_README.md)
- **Main Project README:** [README.md](README.md)

---

## 🐛 Troubleshooting

### Erro: "ModuleNotFoundError"
```bash
conda activate vfnet  # Certifique-se de ativar o ambiente correto
```

### Erro: "File not found"
```bash
# Execute a partir do diretório raiz do projeto
cd /work1/voxel-fluid-net
python ablation_coarse_prediction_analysis.py
```

### Gráficos não aparecem
Os gráficos são salvos automaticamente como arquivos PNG no diretório `reports/`, não abrem em janelas interativas.

---

## 👥 Contribuindo

Para adicionar novos scripts de análise:

1. Crie o script Python na raiz ou em `scripts/`
2. Adicione documentação detalhada no cabeçalho do script
3. Crie um README específico se a análise for complexa
4. Atualize este arquivo `ANALYSIS_SCRIPTS.md`

---

## 📄 Licença

Parte do projeto VoxelFluidNet. Consulte a licença principal do projeto.

---

**Última Atualização:** 21 de Junho de 2026
