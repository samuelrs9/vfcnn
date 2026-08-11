import os
import pandas as pd
from vfnet.report import Reports
from sim_reader.data import DataReader

def create_report(sim_config, gt_config_file, pred_config_file):
  """ 
  Gera relatório de métricas de acurácia para a estimativa de normal de simulação 3D
  """        
  data_reader = DataReader(sim_config)

  report = Reports(data_reader)
  errors = report.compute_regression_metrics(
      gt_config_file,pred_config_file,
      section='normal',
      comparative_label=1,
      initial_step=-1,
      final_step=-1,
      device='cpu',
      return_metrics=True)

  return errors

def simple_vfcnn_dilation_report():
    simulations = [
        ('ddb_3d_big_res', 4, 94),
        ('inlet_collision_3d_big_res', 3, 70), 
        ('db_blocks_3d_big_res', 2, 83), 
        ('inlet_vortex_3d_big_res', 1, 45), 
        ('fountain_3d_big_res', 0, 76)
    ]
    #static_names = ['armadillo']
    config_tag = "_hdp=1.73"
    model_name = "sparse_voxelized_fluid_cnn_100000_1.50_0.10_1"
    data_dir = "data/3D/big"    
    results = []
    for simulation in simulations:
        print(f"Mesh: {simulation[0]}")
        sim_config = f'{data_dir}/{simulation[0]}/sim_config{config_tag}.ini'
        gt_config = f'{data_dir}/{simulation[0]}/gt_config{config_tag}.ini'        
        pred_dir = f'{data_dir}/{simulation[0]}/sparse_regionwise_approach/predictions'
        pred_config = f'{pred_dir}/pred_{model_name}_{simulation[1]}_kfold4_dilation/pred_config_v2.ini'
        
        if os.path.exists(pred_config):
            print(f"Found {pred_config}")

        errors = create_report(sim_config, gt_config, pred_config)
        
        results.append([simulation[0], errors['mean_mae'], errors['std_mae'], errors['mean_mse'], errors['std_mse'], errors['mean_cos'], errors['std_cos'], errors['mean_angle'], errors['std_angle']])

    df = pd.DataFrame(results, columns=['mesh', 'mean_mae', 'std_mae', 'mean_mse', 'std_mse', 'mean_cos', 'std_cos', 'mean_angle', 'std_angle'])
    df.to_csv(f"{data_dir}/metrics_report_normal_svfcnn_dilation_fluid.csv", index=False, sep=";")
    print(df)

def simple_dense_vfcnn_report():
    results = [
        {
            'sim_name': 'ddb_3d_big_res', 
            'sim_config': 'data/3D/big/ddb_3d_big_res/sim_config_hdp=1.73.ini',
            'gt_config': 'data/3D/big/ddb_3d_big_res/gt_config_hdp=1.73.ini',
            'pred_config': 'data/3D/big/ddb_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/pred_config_v2.ini'
        },
        {
            'sim_name': 'inlet_collision_3d_big_res', 
            'sim_config': 'data/3D/big/inlet_collision_3d_big_res/sim_config_hdp=2.0.ini',
            'gt_config': 'data/3D/big/inlet_collision_3d_big_res/gt_config_hdp=2.0.ini',
            'pred_config': 'data/3D/big/inlet_collision_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/pred_config_v2.ini'
        },    
        {
            'sim_name': 'db_blocks_3d_big_res', 
            'sim_config': 'data/3D/big/db_blocks_3d_big_res/sim_config_hdp=1.73.ini',
            'gt_config': 'data/3D/big/db_blocks_3d_big_res/gt_config_hdp=1.73.ini',
            'pred_config': 'data/3D/big/db_blocks_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/pred_config_v2.ini'
        },               
        {
            'sim_name': 'inlet_vortex_3d_big_res', 
            'sim_config': 'data/3D/big/inlet_vortex_3d_big_res/sim_config_hdp=1.73.ini',
            'gt_config': 'data/3D/big/inlet_vortex_3d_big_res/gt_config_hdp=1.73.ini',
            'pred_config': 'data/3D/big/inlet_vortex_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/pred_config_v2.ini'
        },                       
        {
            'sim_name': 'fountain_3d_big_res', 
            'sim_config': 'data/3D/big/fountain_3d_big_res/sim_config_hdp=2.0.ini',
            'gt_config': 'data/3D/big/fountain_3d_big_res/gt_config_hdp=2.0.ini',
            'pred_config': 'data/3D/big/inlet_vortex_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/pred_config_v2.ini'
        }
    ]
    data_dir = "data/3D/big"    
    report = []
    "data/3D/big/ddb_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/pred_config.ini"
    for result in results:
        simulation = result['sim_name']
        print(f"Sim: {simulation}")
        sim_config = result['sim_config']
        gt_config = result['gt_config']
        pred_config = result['pred_config']
        
        if os.path.exists(pred_config):
            print(f"Found {pred_config}")

        errors = create_report(sim_config, gt_config, pred_config)
        
        report.append([simulation, errors['mean_mae'], errors['std_mae'], errors['mean_mse'], errors['std_mse'], errors['mean_cos'], errors['std_cos'], errors['mean_angle'], errors['std_angle']])

    df = pd.DataFrame(report, columns=['sim', 'mean_mae', 'std_mae', 'mean_mse', 'std_mse', 'mean_cos', 'std_cos', 'mean_angle', 'std_angle'])
    df.to_csv(f"{data_dir}/metrics_report_normal_dvfcnn_fluid.csv", index=False, sep=";")
    print(df)


def simple_vfcnn_report():
    simulations = [
        ('ddb_3d_big_res', 4, 94),
        ('inlet_collision_3d_big_res', 3, 70), 
        ('db_blocks_3d_big_res', 2, 83), 
        ('inlet_vortex_3d_big_res', 1, 45), 
        ('fountain_3d_big_res', 0, 76)
    ]
    #static_names = ['armadillo']
    config_tag = "_hdp=1.73"
    model_name = "sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.1_0"
    data_dir = "data/3D/big"    
    output_dir = "/work1/Doutorado/data/3D/big"    
    results = []
    for simulation in simulations:
        print(f"Mesh: {simulation[0]}")
        sim_config = f'{data_dir}/{simulation[0]}/sim_config{config_tag}.ini'
        gt_config = f'{data_dir}/{simulation[0]}/gt_config{config_tag}.ini'        
        pred_dir = f'{output_dir}/{simulation[0]}/sparse_regionwise_approach/predictions_20240907/kfold3{config_tag}_checkpoint_{simulation[2]}'
        pred_config = f'{pred_dir}/pred_{model_name}_{simulation[1]}_kfold3_no_coarse/pred_config_v2.ini'

        errors = create_report(sim_config, gt_config, pred_config)
        
        results.append([simulation[0], errors['mean_mae'], errors['std_mae'], errors['mean_mse'], errors['std_mse'], errors['mean_cos'], errors['std_cos'], errors['mean_angle'], errors['std_angle']])

    df = pd.DataFrame(results, columns=['mesh', 'mean_mae', 'std_mae', 'mean_mse', 'std_mse', 'mean_cos', 'std_cos', 'mean_angle', 'std_angle'])
    df.to_csv(f"{data_dir}/metrics_report_normal_svfcnn_fluid.csv", index=False, sep=";")
    print(df)

def simple_pca_report():
    simulations = [
        ('ddb_3d_big_res', 4, 94),
        ('inlet_collision_3d_big_res', 3, 70), 
        ('db_blocks_3d_big_res', 2, 83), 
        ('inlet_vortex_3d_big_res', 1, 45), 
        ('fountain_3d_big_res', 0, 76)
    ]
    #static_names = ['armadillo']
    model_name = "sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0"
    config_tag = "_hdp=1.73"
    data_dir = "data/3D/big"
    output_dir = "/work1/Doutorado/data/3D/big"
    results = []
    for simulation in simulations:
        print(f"Mesh: {simulation[0]}")
        sim_config = f'{data_dir}/{simulation[0]}/sim_config{config_tag}.ini'
        gt_config = f'{data_dir}/{simulation[0]}/gt_config{config_tag}.ini'
        pred_config = f'{data_dir}/{simulation[0]}/pca_pred_config.ini'

        errors = create_report(sim_config, gt_config, pred_config)
        
        results.append([simulation[0], errors['mean_mae'], errors['std_mae'], errors['mean_mse'], errors['std_mse'], errors['mean_cos'], errors['std_cos'], errors['mean_angle'], errors['std_angle']])

    df = pd.DataFrame(results, columns=['mesh', 'mean_mae', 'std_mae', 'mean_mse', 'std_mse', 'mean_cos', 'std_cos', 'mean_angle', 'std_angle'])
    df.to_csv(f"{data_dir}/metrics_report_normal_pca_fluid.csv", index=False, sep=";")
    print(df)


if __name__=="__main__":
  #simple_vfcnn_report()
  #simple_vfcnn_dilation_report()
  simple_dense_vfcnn_report()
  #simple_pca_report()
  #simple_sph_report()

