#!/bin/bash
# Script de exemplo para executar os relatórios de métricas
# Certifique-se de que o ambiente conda está ativado: conda activate vfnet

echo "=========================================="
echo "Scripts de Relatórios - Tutoriais 7.x"
echo "=========================================="
echo ""

# Diretórios
REPORTS_DIR="scripts/reports"
CONFIGS_DIR="scripts/configs/reports"

# Verifica se o ambiente está ativado
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "AVISO: Ambiente conda não detectado. Execute: conda activate vfnet"
    exit 1
fi

echo "Ambiente ativo: ${CONDA_DEFAULT_ENV}"
echo ""

# Menu de opções
echo "Escolha o relatório que deseja gerar:"
echo "1) Classification Metrics (Tutorial 7.1)"
echo "2) Classification Times (Tutorial 7.2)"
echo "3) Accuracy by Curvatures (Tutorial 7.4)"
echo "4) Compare Models (Tutorial 7.31)"
echo "5) Sair"
echo ""
read -p "Digite sua opção [1-5]: " option

case $option in
    1)
        echo ""
        echo "Executando Classification Metrics..."
        echo "-----------------------------------"
        python ${REPORTS_DIR}/classification_metrics.py ${CONFIGS_DIR}/classification_metrics.yaml
        ;;
    2)
        echo ""
        echo "Executando Classification Times..."
        echo "----------------------------------"
        python ${REPORTS_DIR}/classification_times.py ${CONFIGS_DIR}/classification_times.yaml
        ;;
    3)
        echo ""
        echo "Executando Accuracy by Curvatures..."
        echo "------------------------------------"
        python ${REPORTS_DIR}/accuracy_by_curvatures.py ${CONFIGS_DIR}/accuracy_by_curvatures.yaml
        ;;
    4)
        echo ""
        echo "Executando Compare Models..."
        echo "----------------------------"
        python ${REPORTS_DIR}/compare_models.py ${CONFIGS_DIR}/compare_models.yaml
        ;;
    5)
        echo "Saindo..."
        exit 0
        ;;
    *)
        echo "Opção inválida!"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Execução concluída!"
echo "=========================================="
