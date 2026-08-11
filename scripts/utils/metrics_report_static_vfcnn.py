import os 
import glob
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
        print_metrics=False,
        return_metrics=True
    )
    return metrics
    
def hdp_analysis_report():
    #static_names = ['armadillo', 'bunny', 'dragon', 'happy', 'rocker-arm']
    static_names = ['armadillo']
    pred_set = 'predictions/kfold3_hdp'
    pred_ids = {
        'model0': 'pred_sparse_voxelized_fluid_cnn_v3_4_1000_1.50_0.10_0_0_kfold3_no_coarse',
        'model1': 'pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_1_kfold3_no_coarse',
        'model2': 'pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_2_kfold3_no_coarse',
        'model3': 'pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_3_kfold3_no_coarse',
        'model4': 'pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_4_kfold3_no_coarse'
    }
    #pred_ids = {
    #    'model0': 'pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_0_kfold3_no_coarse'        
    #}    
    mesh_results = {}
    for name in static_names:
        print(name)
        gt_config = f'data/3D/static/{name}/gt_config_v2.yaml'
        sim_config = f'data/3D/static/{name}/sim_config_v2.yaml'
        for pred in pred_ids:
            pred_configs = glob.glob(f'data/3D/static/{name}/sparse_regionwise_approach/{pred_set}/*/{pred_ids[pred]}/pred_config_v2.yaml')
            hdp_results = []
            for pred_config in pred_configs:
                hdp = pred_config.split('/')[-3]
                print(f"Hdp: {hdp}")
                metrics = create_report(sim_config, gt_config, pred_config)
                #print(metrics)
                hdp_results.append([hdp, metrics['recall'], metrics['precision'], metrics['f1_score'],  metrics['matthews_coefficient']])
            print(pred)                
            df = pd.DataFrame(hdp_results, columns=['hdp', 'rec', 'pre', 'f1', 'mcc'])
            df = df.sort_values(by='hdp')
            df['hdp'] = df['hdp'].astype(float)
            #plt.plot(df['hdp'], df['mcc'])
            #plt.show()
            print(df)
            
def simple_report():
    meshes = [('armadillo',4), ('bunny',3), ('dragon',2), ('happy',1), ('rocker-arm',0)]
    #static_names = ['armadillo']
    model_name = "sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0"
    data_dir = "data/3D/static"    
    config_tag = "_hdp=2.0"
    pred_dir = f'sparse_regionwise_approach/predictions/kfold3_static{config_tag}'
    results = []
    for mesh in meshes:
        print(f"Mesh: {mesh[0]}")
        mesh_dir = f'{data_dir}/{mesh[0]}'
        mesh_config = f'{data_dir}/{mesh[0]}/sim_config{config_tag}.yaml'
        gt_config = f'data/3D/static/{mesh[0]}/gt_config.yaml'                
        pred_config = f'{data_dir}/{mesh[0]}/{pred_dir}/pred_{model_name}_{mesh[1]}_kfold3_no_coarse/pred_config_v2.yaml'

        metrics = create_report(mesh_config, gt_config, pred_config)
        
        results.append([mesh[0], metrics['recall'], metrics['precision'], metrics['f1_score'],  metrics['matthews_coefficient']])

    df = pd.DataFrame(results, columns=['mesh', 'rec', 'pre', 'f1', 'mcc'])
    df.to_csv(f"{data_dir}/metrics_report_sparse_vfcnn.csv", index=False, sep=";")
    print(df)

if __name__=='__main__':
    #hdp_analysis_report()
    simple_report()