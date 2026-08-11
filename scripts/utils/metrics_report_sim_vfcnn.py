import os 
import glob
import datetime
import pandas as pd
from vfnet.preprocessing import DataReader
from vfnet.report import Reports
import matplotlib.pyplot as plt

def create_report(sim_config_file, gt_config_file, pred_config_file):
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
        print_metrics=True,
        return_metrics=True
    )
    return metrics
    
def simple_report_test():
    simulations = [
        ('ddb_3d_big_res/test',4),
        ('inlet_collision_3d_big_res/test',3), 
        ('db_blocks_3d_big_res/test',2), 
        ('inlet_vortex_3d_big_res/test',1), 
        ('fountain_3d_big_res/test',0)
    ]
    #model_name = "sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.2_0"
    #model_name = "sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.05_1"
    #model_name = "sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0"
    model_name = "sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.1_0"
    data_dir = "data/3D/big"    
    #config_tag = "_hdp=1.73"
    config_tag = "_hdp=1.73"
    pred_dir = f'sparse_regionwise_approach/predictions/kfold3_test{config_tag}_20240901'
    results = []
    for sim_name,model_id in simulations:
        print(f"Mesh: {sim_name}")
        mesh_config = f'{data_dir}/{sim_name}/sim_config{config_tag}.ini'
        gt_config = f'{data_dir}/{sim_name}/gt_config{config_tag}.ini'                
        pred_config = f'{data_dir}/{sim_name}/{pred_dir}/pred_{model_name}_{model_id}_kfold3_no_coarse/pred_config_v2.ini'

        metrics = create_report(mesh_config, gt_config, pred_config)
        
        results.append([sim_name, metrics['recall'], metrics['precision'], metrics['f1_score'],  metrics['matthews_coefficient']])

    df = pd.DataFrame(results, columns=['mesh', 'rec', 'pre', 'f1', 'mcc'])
    df.to_csv(f"{data_dir}/metrics_report_sparse_vfcnn.csv", index=False, sep=";")
    print(df)

def simple_report():
    data_dir = "data/3D/big"    
    output_dir = "/work1/Doutorado/data/3D/big"
    simulations = [
        ('ddb_3d_big_res', 4, 94),
        ('inlet_collision_3d_big_res', 3, 70), 
        ('db_blocks_3d_big_res', 2, 83), 
        ('inlet_vortex_3d_big_res', 1, 45), 
        ('fountain_3d_big_res', 0, 76)
    ]
    model_name = "sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.1_0"
    config_tag = "_hdp=1.73"
    pred_dir = f'sparse_regionwise_approach/predictions_20240907/kfold3{config_tag}*'
    results = []
    for sim_name,model_id,checkpoint in simulations:
        print(f"Mesh: {sim_name}")
        mesh_config = f'{data_dir}/{sim_name}/sim_config{config_tag}.ini'
        gt_config = f'{data_dir}/{sim_name}/gt_config{config_tag}.ini'                
        pred_config = glob.glob(f'{output_dir}/{sim_name}/{pred_dir}/pred_{model_name}_{model_id}_kfold3_no_coarse/pred_config_v2.ini')[0]

        metrics = create_report(mesh_config, gt_config, pred_config)
        
        results.append([sim_name, metrics['recall'], metrics['precision'], metrics['f1_score'],  metrics['matthews_coefficient']])

    df = pd.DataFrame(results, columns=['mesh', 'rec', 'pre', 'f1', 'mcc'])
    df.to_csv(f"{output_dir}/metrics_report_sparse_vfcnn_{datetime.date.today().strftime('%Y%m%d')}.csv", index=False, sep=";")
    print(df)

if __name__=='__main__':
    #hdp_analysis_report()
    #simple_report_test()
    simple_report()