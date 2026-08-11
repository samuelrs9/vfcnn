import os
import pandas as pd
from vfnet.report import Reports
from sim_reader.data import DataReader

def create_report(mesh_config, gt_config_file, pred_config_file):
  """ 
  Gera relatório de métricas de acurácia para a estimativa de normal de simulação 3D
  """        
  data_reader = DataReader(mesh_config)

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

def simple_vfcnn_report():
    meshes = [('armadillo',4,73), ('bunny',3,28), ('dragon',2,70), ('happy',1,48), ('rocker-arm',0,98)]
    #static_names = ['armadillo']
    model_name = "sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0"
    data_dir = "/work1/Doutorado/data/3D/static"    
    config_tag = "_hdp=2.0"
    pred_dir = f'sparse_regionwise_approach/predictions/kfold3_{config_tag}_checkpoints'
    results = []
    for mesh in meshes:
        print(f"Mesh: {mesh[0]}")
        mesh_dir = f'{data_dir}/{mesh[0]}'
        mesh_config = f'{data_dir}/{mesh[0]}/sim_config{config_tag}.ini'
        gt_config = f'{data_dir}/{mesh[0]}/gt_config.ini'
        pred_config = f'{data_dir}/{mesh[0]}/{pred_dir}/{mesh[2]}/pred_{model_name}_{mesh[1]}_kfold3_no_coarse/pred_config_v2.ini'

        errors = create_report(mesh_config, gt_config, pred_config)
        
        results.append([mesh[0], errors['mean_mae'], errors['std_mae'], errors['mean_mse'], errors['std_mse'], errors['mean_cos'], errors['std_cos'], errors['mean_angle'], errors['std_angle']])

    df = pd.DataFrame(results, columns=['mesh', 'mean_mae', 'std_mae', 'mean_mse', 'std_mse', 'mean_cos', 'std_cos', 'mean_angle', 'std_angle'])
    df.to_csv(f"{data_dir}/metrics_report_normal_vfcnn_static.csv", index=False, sep=";")
    print(df)

def simple_sph_report():
    meshes = [('armadillo',4,73), ('bunny',3,28), ('dragon',2,70), ('happy',1,48), ('rocker-arm',0,98)]
    #static_names = ['armadillo']
    model_name = "sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0"
    data_dir = "/work1/Doutorado/data/3D/static"    
    config_tag = "_hdp=2.0"
    results = []
    for mesh in meshes:
        print(f"Mesh: {mesh[0]}")
        mesh_dir = f'{data_dir}/{mesh[0]}'
        mesh_config = f'{data_dir}/{mesh[0]}/sim_config{config_tag}.ini'
        gt_config = f'{data_dir}/{mesh[0]}/gt_config.ini'
        pred_config = f'{data_dir}/{mesh[0]}/sph/sph_config.ini'

        errors = create_report(mesh_config, gt_config, pred_config)
        
        results.append([mesh[0], errors['mean_mae'], errors['std_mae'], errors['mean_mse'], errors['std_mse'], errors['mean_cos'], errors['std_cos'], errors['mean_angle'], errors['std_angle']])

    df = pd.DataFrame(results, columns=['mesh', 'mean_mae', 'std_mae', 'mean_mse', 'std_mse', 'mean_cos', 'std_cos', 'mean_angle', 'std_angle'])
    df.to_csv(f"{data_dir}/metrics_report_normal_sph_static.csv", index=False, sep=";")
    print(df)

def simple_pca_report():
    meshes = [('armadillo',4,73), ('bunny',3,28), ('dragon',2,70), ('happy',1,48), ('rocker-arm',0,98)]
    #static_names = ['armadillo']
    model_name = "sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0"
    data_dir = "/work1/Doutorado/data/3D/static"    
    config_tag = "_hdp=2.0"
    results = []
    for mesh in meshes:
        print(f"Mesh: {mesh[0]}")
        mesh_dir = f'{data_dir}/{mesh[0]}'
        mesh_config = f'{data_dir}/{mesh[0]}/sim_config{config_tag}.ini'
        gt_config = f'{data_dir}/{mesh[0]}/gt_config.ini'
        pred_config = f'{data_dir}/{mesh[0]}/pca/pca_config.ini'

        errors = create_report(mesh_config, gt_config, pred_config)
        
        results.append([mesh[0], errors['mean_mae'], errors['std_mae'], errors['mean_mse'], errors['std_mse'], errors['mean_cos'], errors['std_cos'], errors['mean_angle'], errors['std_angle']])

    df = pd.DataFrame(results, columns=['mesh', 'mean_mae', 'std_mae', 'mean_mse', 'std_mse', 'mean_cos', 'std_cos', 'mean_angle', 'std_angle'])
    df.to_csv(f"{data_dir}/metrics_report_normal_pca_static.csv", index=False, sep=";")
    print(df)


if __name__=="__main__":
  #simple_vfcnn_report()
  #simple_pca_report()
  simple_sph_report()

