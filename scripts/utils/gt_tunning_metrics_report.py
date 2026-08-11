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
    static_names = ['armadillo', 'bunny', 'dragon', 'happy', 'rocker-arm']
    pred_set = 'gt-tunning'
    mesh_results = {}
    for name in static_names:
        print(name)
        gt_config = f'data/3D/static/{name}/gt_config.ini'
        sim_config = f'data/3D/static/{name}/sim_config_hdp=2.0.ini'
        pred_configs = glob.glob(f'data/3D/static/{name}/{pred_set}/gt_config*.ini')
        hdp_results = []
        for pred_config in pred_configs:
            hdp = pred_config.split('/')[-1].split('=')[-1].replace('.ini','')
            print(f"Hdp: {hdp}")
            metrics = main(sim_config, gt_config, pred_config)
            #print(metrics)
            hdp_results.append([hdp, metrics['recall'], metrics['precision'], metrics['f1_score'],  metrics['matthews_coefficient']])
        df = pd.DataFrame(hdp_results, columns=['hdp', 'rec', 'pre', 'f1', 'mcc'])
        df = df.sort_values(by='hdp')
        df['hdp'] = df['hdp'].astype(float)
        print(df)
        plt.plot(df['hdp'], df['mcc'])
        plt.show()
            

