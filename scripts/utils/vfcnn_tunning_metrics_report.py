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

def tune_hpd_fluid():
    sim_names = ['inlet_collision_3d_big_res']
    pred_set = 'vfcnn-tunning'
    tag = '_hdp=1.73'
    pred_set = 'sparse_regionwise_approach/predictions/kfold3_hdp=1.73/hdp'
    mesh_results = {}
    for name in sim_names:
        print(name)
        gt_config = f'data/3D/big/{name}/gt_config{tag}.yaml'
        sim_config = f'data/3D/big/{name}/sim_config{tag}.yaml'
        pred_configs = glob.glob(f"data/3D/big/{name}/{pred_set}/*/pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.1_0_3_kfold3_no_coarse/pred_config_v2.yaml")
        hdp_results = []
        for pred_config in pred_configs:
            hdp = pred_config.split('/')[-3]
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


def tune_hpd_static():
    sim_names = ['rocker-arm']    
    tag = '_hdp=2.0'
    pred_set = 'sparse_regionwise_approach/predictions/hdp_tunning'
    data_dir = '/work1/Doutorado/data/3D/static'
    pred_name = 'pred_sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0_0_kfold3_no_coarse'
    mesh_results = {}
    for name in sim_names:
        print(name)
        gt_config = f'{data_dir}/{name}/gt_config.yaml'
        sim_config = f'{data_dir}/{name}/sim_config{tag}.yaml'
        pred_configs = glob.glob(f"{data_dir}/{name}/{pred_set}/*/{pred_name}/pred_config_v2.yaml")
        hdp_results = []
        for pred_config in pred_configs:
            hdp = pred_config.split('/')[-3]
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

def tune_decision_threshold_static():
    sim_names = ['rocker-arm']
    tag = '_hdp=2.0'
    pred_set = 'sparse_regionwise_approach/predictions/decision_threshold_tunning'
    data_dir = '/work1/Doutorado/data/3D/static'
    pred_name = 'pred_sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0_0_kfold3_no_coarse'
    mesh_results = {}
    for name in sim_names:
        print(name)
        gt_config = f'{data_dir}/{name}/gt_config.yaml'
        sim_config = f'{data_dir}/{name}/sim_config{tag}.yaml'
        pred_configs = glob.glob(f"{data_dir}/{name}/{pred_set}/*/{pred_name}/pred_config_v2.yaml")
        threshold_results = []
        for pred_config in pred_configs:
            threshold = pred_config.split('/')[-3]
            print(f"threshold: {threshold}")
            metrics = main(sim_config, gt_config, pred_config)
            #print(metrics)
            threshold_results.append([threshold, metrics['recall'], metrics['precision'], metrics['f1_score'],  metrics['matthews_coefficient']])
        df = pd.DataFrame(threshold_results, columns=['threshold', 'rec', 'pre', 'f1', 'mcc'])
        df = df.sort_values(by='threshold')
        df['threshold'] = df['threshold'].astype(float)
        print(df)
        plt.plot(df['threshold'], df['mcc'])
        plt.show()    


def tune_checkpoint_fluid():
    data_dir = 'data/3D/big'
    sim_names = [
        ('ddb_3d_big_res/test',4),
        ('inlet_collision_3d_big_res/test',3),
        ('db_blocks_3d_big_res/test',2),
        #('inlet_vortex_3d_big_res/test',1),
        ('fountain_3d_big_res/test',0)
    ]
    tag = '_hdp=1.73'
    pred_set = 'sparse_regionwise_approach/predictions/kfold3_test_hdp=1.73_checkpoints'    
    model_tag = 'sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.1_0'    
    for name in sim_names:
        print(name)
        gt_config = f'{data_dir}/{name[0]}/gt_config{tag}.yaml'
        sim_config = f'{data_dir}/{name[0]}/sim_config{tag}.yaml'
        pred_tag = f'pred_{model_tag}_{name[1]}_kfold3_no_coarse'
        pred_configs = glob.glob(f"{data_dir}/{name[0]}/{pred_set}/*/{pred_tag}/pred_config_v2.yaml")
        checkpoint_results = []
        for pred_config in pred_configs:
            epoch = pred_config.split('/')[-3]
            print(f"Epoch: {epoch}")
            metrics = main(sim_config, gt_config, pred_config)
            #print(metrics)
            checkpoint_results.append([epoch, metrics['recall'], metrics['precision'], metrics['f1_score'],  metrics['matthews_coefficient']])
        df = pd.DataFrame(checkpoint_results, columns=['epoch', 'rec', 'pre', 'f1', 'mcc'])
        df['epoch'] = df['epoch'].astype(int)
        df = df.sort_values(by='epoch')
        print(df)
        df.to_csv(f'{data_dir}/{name[0]}/{pred_set}.csv', index=False, sep=';')
        plt.plot(df['epoch'], df['mcc'])
        plt.show()    

def tune_checkpoint_static():
    data_dir = '/work1/Doutorado/data/3D/static'
    #sim_names = [('armadillo',4),('bunny',3),('dragon',2),('happy',1),('rocker-arm',0)]
    sim_names = [('rocker-arm',0)]
    tag = '_hdp=2.0'
    pred_set = 'sparse_regionwise_approach/predictions/kfold3__hdp=2.0_checkpoints'
    model_tag = 'sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0'
    for name in sim_names:
        print(name)
        gt_config = f'{data_dir}/{name[0]}/gt_config.yaml'
        sim_config = f'{data_dir}/{name[0]}/sim_config{tag}.yaml'
        pred_tag = f'pred_{model_tag}_{name[1]}_kfold3_no_coarse'
        pred_configs = glob.glob(f"{data_dir}/{name[0]}/{pred_set}/*/{pred_tag}/pred_config_v2.yaml")
        checkpoint_results = []
        for pred_config in pred_configs:
            epoch = pred_config.split('/')[-3]
            print(f"Epoch: {epoch}")
            metrics = main(sim_config, gt_config, pred_config)
            #print(metrics)
            checkpoint_results.append([epoch, metrics['recall'], metrics['precision'], metrics['f1_score'],  metrics['matthews_coefficient']])
        df = pd.DataFrame(checkpoint_results, columns=['epoch', 'rec', 'pre', 'f1', 'mcc'])
        df['epoch'] = df['epoch'].astype(int)
        df = df.sort_values(by='epoch')
        print(df)
        df.to_csv(f'{data_dir}/{name[0]}/{pred_set}.csv', index=False, sep=';')
        plt.plot(df['epoch'], df['mcc'])
        plt.show()    

if __name__=='__main__':#
    
    #fluid_report()
    #tune_checkpoint_static()
    tune_checkpoint_fluid()
    #tune_hpd_static()
    #tune_decision_threshold_static()
    
            

