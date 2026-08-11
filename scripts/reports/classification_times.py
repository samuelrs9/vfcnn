#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para gerar relatório de tempo de classificação.
Baseado no Tutorial 7.2 do main_tutorials.py
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
    """Executa o relatório de tempos de classificação."""
    
    # Carrega configurações
    config = load_config(config_file)
    
    # Extrai sim_config_file (pode ser absoluto ou relativo)
    sim_config_file = config['sim_config_file']
    if not os.path.isabs(sim_config_file):
        # Se for relativo, assume relativo ao diretório do config
        config_dir = os.path.dirname(os.path.abspath(config_file))
        sim_config_file = os.path.join(config_dir, sim_config_file)
    
    # Inicializa DataReader
    data_reader = DataReader(sim_config_file)
    
    # Configurações de predição (lista de caminhos absolutos)
    pred_config_files = config.get('pred_config_files', [])
    
    if not isinstance(pred_config_files, list):
        pred_config_files = [pred_config_files]
    
    # Parâmetros do relatório
    report_config = config.get('report', {})
    output_dir = report_config.get('output_path', None)
    extension = report_config.get('format', 'csv')
    plot_times = report_config.get('plot_times', True)
    print_times = report_config.get('print_times', True)
    
    # Gera o relatório
    report = Reports(data_reader)
    report.classification_times(
        pred_configs=pred_config_files,
        output_dir=output_dir,
        extension=extension,
        plot_times=plot_times,
        print_times=print_times
    )
    
    print("\nRelatório de tempos de classificação concluído!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gera relatório de tempo de classificação'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Caminho para o arquivo de configuração YAML'
    )
    
    args = parser.parse_args()
    main(args.config)
