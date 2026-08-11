# Sumário da Migração dos Tutoriais 7.x

## Objetivo

Extrair os tutoriais 7.x do arquivo `main_tutorials.py` e criar scripts específicos para cada um, com parâmetros carregados de arquivos de configuração YAML.

## Estrutura Criada

```
voxel-fluid-net/
├── scripts/
│   ├── reports/                                    # Scripts de relatórios
│   │   ├── README.md                              # Documentação principal
│   │   ├── classification_metrics.py              # Tutorial 7.1
│   │   ├── classification_times.py                # Tutorial 7.2
│   │   ├── accuracy_by_curvatures.py              # Tutorial 7.4
│   │   ├── compare_models.py                      # Tutorial 7.31
│   │   └── run_reports.sh                         # Script interativo para executar relatórios
│   │
│   └── configs/
│       └── reports/                                # Configurações YAML
│           ├── EXAMPLES.md                        # Exemplos de configurações personalizadas
│           ├── classification_metrics.yaml        # Config para Tutorial 7.1
│           ├── classification_times.yaml          # Config para Tutorial 7.2
│           ├── accuracy_by_curvatures.yaml        # Config para Tutorial 7.4
│           └── compare_models.yaml                # Config para Tutorial 7.31
│
└── README.md                                       # Atualizado com seção de Reports
```

## Scripts Criados

### 1. classification_metrics.py (Tutorial 7.1)
**Função:** Gera relatório de classificação com métricas de acurácia

**Método chamado:** `Reports.classification_metrics()`

**Parâmetros principais:**
- `working_dir`: Diretório de trabalho base
- `data_dir`: Diretório de dados
- `gt_config_file`: Configuração do ground-truth
- `pred_config_path`: Caminho para as predições
- `sections`: Seções a serem comparadas
- `plot_metrics`, `print_metrics`, `return_metrics`: Opções de saída

**Uso:**
```bash
python scripts/reports/classification_metrics.py scripts/configs/reports/classification_metrics.yaml
```

---

### 2. classification_times.py (Tutorial 7.2)
**Função:** Gera relatório de tempo de classificação

**Método chamado:** `Reports.classification_times()`

**Parâmetros principais:**
- `pred_configs`: Lista de configurações de predição (suporta múltiplos modelos)
- `output_dir`: Diretório de saída
- `extension`: Formato de saída (csv, etc.)
- `plot_times`, `print_times`: Opções de visualização

**Uso:**
```bash
python scripts/reports/classification_times.py scripts/configs/reports/classification_times.yaml
```

---

### 3. accuracy_by_curvatures.py (Tutorial 7.4)
**Função:** Análise de acurácia por intervalos de curvatura

**Método chamado:** `Reports.accuracy_by_curvatures()`

**Parâmetros principais:**
- Similar ao classification_metrics
- Foca em análise por intervalos de curvatura
- Suporta diferentes métodos (CNN, Marrone, BPART, etc.)

**Uso:**
```bash
python scripts/reports/accuracy_by_curvatures.py scripts/configs/reports/accuracy_by_curvatures.yaml
```

---

### 4. compare_models.py (Tutorial 7.31)
**Função:** Comparação entre múltiplos modelos

**Método chamado:** `Reports.compare_models()`

**Parâmetros principais:**
- `pred_configs`: Lista de modelos para comparar
- `output_dir`: Diretório de saída
- `plot_metrics`, `print_metrics`: Opções de visualização

**Uso:**
```bash
python scripts/reports/compare_models.py scripts/configs/reports/compare_models.yaml
```

---

### 5. run_reports.sh
**Função:** Script bash interativo para facilitar a execução dos relatórios

**Características:**
- Menu interativo
- Verifica se o ambiente conda está ativado
- Executa os scripts com seus configs padrão

**Uso:**
```bash
bash scripts/reports/run_reports.sh
```

## Arquivos de Configuração (YAML)

Todos os arquivos YAML seguem uma estrutura consistente:

### Elementos Comuns:
- `working_dir`: Diretório base do projeto
- `data_dir`: Caminho relativo para os dados
- `sim_config_file`: Arquivo de configuração da simulação (padrão: "sim_config.ini")
- `gt_config_file`: Arquivo de configuração do ground-truth (padrão: "gt_config.ini")

### Formatos de Caminho:
```yaml
# Formato 1: Lista de diretórios
pred_config_path:
  - "regionwise_approach"
  - "predictions"
  - "pred_model_name"

# Formato 2: String simples
pred_config_path: "regionwise_approach/predictions/pred_model_name"
```

### Múltiplas Predições:
```yaml
pred_configs:
  - path: ["dir1", "dir2", "model1"]
    file: "pred_config.ini"
  - path: ["dir1", "dir2", "model2"]
    file: "pred_config.ini"
```

## Documentação Adicional

### README.md (scripts/reports/)
- Visão geral dos scripts
- Como usar cada script
- Estrutura de configuração
- Exemplos práticos
- Notas sobre formatos de caminho

### EXAMPLES.md (scripts/configs/reports/)
- Exemplos completos de configurações personalizadas
- Cenários de uso comuns:
  - Múltiplas simulações
  - Comparação de 4 modelos
  - Diferentes métodos (CNN, Marrone, BPART)
  - K-Fold Cross-Validation
  - Configurações mínimas vs completas
- Dicas e boas práticas

## Melhorias Implementadas

### 1. Modularização
- Cada tutorial agora é um script independente
- Facilita manutenção e reutilização
- Reduz complexidade do main_tutorials.py

### 2. Configuração Flexível
- Parâmetros em YAML ao invés de hard-coded
- Fácil criação de múltiplas configurações
- Documentação inline nos arquivos YAML

### 3. Reutilização
- Mesmo script para diferentes experimentos
- Apenas troca o arquivo de configuração
- Suporta múltiplos modelos e comparações

### 4. Documentação
- README completo em scripts/reports/
- Exemplos práticos em EXAMPLES.md
- Comentários nos arquivos YAML
- Documentação inline nos scripts

### 5. Facilidade de Uso
- Script interativo (run_reports.sh)
- Verificação de ambiente
- Mensagens claras de erro e sucesso
- Scripts executáveis (chmod +x)

## Como Começar

### 1. Ativar o Ambiente
```bash
conda activate vfnet
```

### 2. Opção A: Usar Configs Padrão
```bash
# Edite o config desejado
vim scripts/configs/reports/classification_metrics.yaml

# Execute
python scripts/reports/classification_metrics.py scripts/configs/reports/classification_metrics.yaml
```

### 3. Opção B: Criar Config Personalizado
```bash
# Copie um config existente
cp scripts/configs/reports/classification_metrics.yaml my_experiment.yaml

# Edite conforme necessário
vim my_experiment.yaml

# Execute
python scripts/reports/classification_metrics.py my_experiment.yaml
```

### 4. Opção C: Usar Menu Interativo
```bash
bash scripts/reports/run_reports.sh
```

## Tutoriais Migrados

| Tutorial Original | Script Criado | Método Reports | Config YAML |
|-------------------|---------------|----------------|-------------|
| 7.1 | classification_metrics.py | classification_metrics() | classification_metrics.yaml |
| 7.2 | classification_times.py | classification_times() | classification_times.yaml |
| 7.4 | accuracy_by_curvatures.py | accuracy_by_curvatures() | accuracy_by_curvatures.yaml |
| 7.31 | compare_models.py | compare_models() | compare_models.yaml |

## Tutoriais Não Migrados

- **Tutorial 7.3**: Não estava implementado no main_tutorials.py
- **Tutorial 7.32**: Usa método `compare_models_bkp()` que parece ser uma versão antiga
- **Tutorial 8.x**: São tutoriais de análise de curvatura (não eram 7.x)

## Próximos Passos (Opcional)

Se necessário, os seguintes tutoriais também podem ser migrados:
1. Tutorial 8.1: `distribution_per_curvatures()`
2. Tutorial 8.2: `accuracy_per_curvatures()`
3. Tutorial 8.3: `compare_accuracy_per_curvatures()`

## Benefícios da Migração

1. **Organização**: Código mais organizado e modular
2. **Manutenibilidade**: Mais fácil de manter e atualizar
3. **Flexibilidade**: Configs YAML permitem experimentos rápidos
4. **Documentação**: Melhor documentada e com exemplos
5. **Reutilização**: Scripts podem ser usados em diferentes contextos
6. **Produção**: Pronto para uso em pipelines automatizados
7. **Colaboração**: Mais fácil para outros entenderem e usarem

## Referências

- **Código original**: `/work1/voxel-fluid-net/main_tutorials.py`
- **Scripts criados**: `/work1/voxel-fluid-net/scripts/reports/`
- **Configs YAML**: `/work1/voxel-fluid-net/scripts/configs/reports/`
- **Documentação**: `/work1/voxel-fluid-net/scripts/reports/README.md`
- **Exemplos**: `/work1/voxel-fluid-net/scripts/configs/reports/EXAMPLES.md`
