#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para gerar relatório de classificação com métricas de acurácia.
Baseado no Tutorial 7.1 do main_tutorials.py
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
    """Executa o relatório de métricas de classificação."""
    
    # Carrega configurações
    config = load_config(config_file)
    
    # Extrai sim_config_file (caminho absoluto)
    sim_config_file = config['sim_config_file']
    if not os.path.isabs(sim_config_file):
        config_dir = os.path.dirname(os.path.abspath(config_file))
        sim_config_file = os.path.join(config_dir, sim_config_file)
    
    # Inicializa DataReader
    data_reader = DataReader(sim_config_file)
    
    # Configurações do ground-truth (caminho absoluto)
    gt_config_file = config['gt_config_file']
    if not os.path.isabs(gt_config_file):
        config_dir = os.path.dirname(os.path.abspath(config_file))
        gt_config_file = os.path.join(config_dir, gt_config_file)
    
    # Configurações de predição (caminho absoluto)
    pred_config_file = config['pred_config_file']
    if not os.path.isabs(pred_config_file):
        config_dir = os.path.dirname(os.path.abspath(config_file))
        pred_config_file = os.path.join(config_dir, pred_config_file)
    
    # Parâmetros do relatório
    sections = tuple(config.get('sections', ['boundary', 'boundary']))
    report_config = config.get('report', {})
    output_dir = report_config.get('output_path', None)
    plot_metrics = report_config.get('plot_metrics', False)
    print_metrics = report_config.get('print_metrics', True)
    return_metrics = report_config.get('return_metrics', False)
    
    # Gera o relatório
    report = Reports(data_reader)
    report.classification_metrics(
        pred_configs=(gt_config_file, pred_config_file),
        sections=sections,
        output_dir=output_dir,
        plot_metrics=plot_metrics,
        print_metrics=print_metrics,
        return_metrics=return_metrics
    )
    
    print("\nRelatório de métricas de classificação concluído!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gera relatório de classificação com métricas de acurácia'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Caminho para o arquivo de configuração YAML'
    )
    
    args = parser.parse_args()
    main(args.config)
