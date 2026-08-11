import os 
import glob
import pandas as pd
from vfnet.preprocessing import DataReader
from vfnet.report import Reports
import matplotlib.pyplot as plt

def main(sim_config_file, gt_config_file, pred_config_file):
    """
    Gera o relatório de classificação com as métricas de acurácia para malhas estáticas.
    """
    data_reader = DataReader(sim_config_file)    
    report = Reports(data_reader)
    metrics = report.classification_metrics(
        pred_configs=(gt_config_file, pred_config_file),
        sections=('boundary', 'boundary'),
        #output_dir = output_dir,
        plot_metrics=False,
        print_metrics=False,
        return_metrics=True
    )
    return metrics
    
if __name__=='__main__':
    simulations = [
        'ddb_3d_big_res', 
        'inlet_collision_3d_big_res', 
        'db_blocks_3d_big_res', 
        'inlet_vortex_3d_big_res', 
        'fountain_3d_big_res'
    ]
    tag = '_hdp=1.73'
    pred_set = f'other_predictions'
    mesh_results = {}
    data_dir = "data/3D/big"
    output_dir = "/work1/Doutorado/data/3D/big"
    for name in simulations:
        print(name)
        gt_config = f'{data_dir}/{name}/gt_config{tag}.ini'
        sim_config = f'{data_dir}/{name}/sim_config{tag}.ini'
        pred_configs = glob.glob(f'{output_dir}/{name}/{pred_set}/*/pred_config.ini')
        method_results = []
        for pred_config in pred_configs:
            method = pred_config.split('/')[-2]
            print(f"method: {method}")
            metrics = main(sim_config, gt_config, pred_config)
            #print(metrics)
            method_results.append([method, metrics['recall'], metrics['precision'], metrics['f1_score'],  metrics['matthews_coefficient']])            
        df = pd.DataFrame(method_results, columns=['method', 'rec', 'pre', 'f1', 'mcc'])
        df.to_csv(f"{data_dir}/{name}/{pred_set}/metrics_report.csv", sep=";", index=False)
        print(df)
        

