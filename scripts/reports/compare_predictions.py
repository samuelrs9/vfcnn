#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para comparação de múltiplas predições.
Baseado no Tutorial 7.31 do main_tutorials.py
"""

import os
import sys
import yaml
import argparse

# Adiciona o diretório raiz ao path para permitir imports
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from vfnet.report import Reports
from sim_reader.data import DataReader


def load_config(config_file):
    """Carrega as configurações do arquivo YAML."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_file):
    """Executa comparação entre múltiplas predições."""
    
    # Carrega configurações
    config = load_config(config_file)
    
    # Valida formato
    if 'comparisons' not in config:
        raise ValueError("Config must contain 'comparisons' field")
    
    comparisons = config['comparisons']
    if not isinstance(comparisons, list):
        comparisons = [comparisons]
    
    # Processa cada comparação
    all_predictions = []
    data_reader = None
    
    for comparison in comparisons:
        gt_config = comparison.get('gt_config_file')
        pred_list = comparison.get('pred_config_files', [])
        
        if not isinstance(pred_list, list):
            pred_list = [pred_list]
        
        # Deriva sim_config_file do diretório do gt_config_file
        if gt_config:
            gt_dir = os.path.dirname(gt_config)
            sim_config = os.path.join(gt_dir, 'sim_config.yaml')
        else:
            sim_config = None
        
        # Inicializa DataReader (usa primeiro sim_config encontrado)
        if data_reader is None and sim_config and os.path.exists(sim_config):
            if not os.path.isabs(sim_config):
                config_dir = os.path.dirname(os.path.abspath(config_file))
                sim_config = os.path.join(config_dir, sim_config)
            data_reader = DataReader(sim_config)
        
        # Adiciona cada predição com seu gt_config
        for pred_config_file in pred_list:
            all_predictions.append({
                'pred_config_file': pred_config_file,
                'gt_config_file': gt_config
            })
    
    if data_reader is None:
        raise ValueError("No sim_config_file found in comparisons")
    
    # Parâmetros do relatório
    report_config = config.get('report', {})
    output_dir = report_config.get('output_path', None)
    extension = report_config.get('format', 'csv')
    plot_metrics = report_config.get('plot_metrics', True)
    print_metrics = report_config.get('print_metrics', True)
    
    # Gera o relatório
    report = Reports(data_reader)
    report.compare_predictions(
        predictions=all_predictions,
        output_dir=output_dir,
        extension=extension,
        plot_metrics=plot_metrics,
        print_metrics=print_metrics
    )
    
    print("\nComparação de predições concluída!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comparação de múltiplas predições'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Caminho para o arquivo de configuração YAML'
    )
    
    args = parser.parse_args()
    main(args.config)
