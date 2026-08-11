import os 
import sys
import glob
from tqdm import tqdm
from vfnet.postprocessing import PostProcessing

sys.path.append("/home/samuel/Doutorado/voxel-fluid-net/sim_reader")


if __name__=="__main__":

  tutorial = 1.1
  """ 
  Cria arquivos csv com dados de previsões. 
  Útil para visualização no paraview.
  """
  if tutorial == 0:
    working_dir = "/home/samuel/Projetos/voxel-fluid-net"
    #data_dir = os.path.join(working_dir, "data/3D/big/inlet_collision_3d_big_res")
    data_dir = os.path.join(working_dir, "data/3D/big/ddb_3d_big_res")
    #data_dir = os.path.join(working_dir, "data/3D/big/toy_dragon_3d_big_res")
    #data_dir = os.path.join(working_dir, "data/3D/big/inlet_vortex_3d_big_res")
    #data_dir = os.path.join(working_dir, "data/3D/big/fountain_3d_big_res")
    #data_dir = os.path.join(working_dir, "data/3D/big/db_blocks_3d_big_res")
    sim_config_file = os.path.join(data_dir, 'sim_config.yaml')
    
    gt_config_file = os.path.join(data_dir, 'gt_config_v2.yaml')
    
    # Configuração de predição
    pred_config_file = os.path.join(
        #data_dir, 'regionwise_approach', 'predictions','kfold2',
        data_dir, 'sparse_regionwise_approach', 'predictions',
        #'pred_sparse_voxelized_fluid_cnn_v3_4_100000_2.00_0.10_0_2_0',
        #'pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_3_kfold3_no_coarse',
        #'pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_3_new_train3_kfold3_no_coarse',
        #'pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_3_new_train3_kfold3_no_coarse_grid=0.11',
        #'pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_1_3_kfold4_no_coarse',
        "kfold4/pred_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_0_kfold4_dilation_train5"

        #'pred_31_9_4.13_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_0',
        #'pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74',
        #'pred_25_7_3.12_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_70',
        #'pred_25_7_4.17_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_0',
        'pred_config.yaml')
    
    pred_config_files = {'gt': gt_config_file, 'pred': pred_config_file}

    process = PostProcessing(sim_config_file)
    process.pred_to_csv(
        pred_config_files = pred_config_files,
        output_dir = os.path.join(data_dir, 'pred_csv'),
        initial_step = -1
    )

  """ 
  Cria arquivos ply com coordenadas de partículas de fronteira e suas normais.
  """
  if tutorial == 1:    
    data_dir = "/work1/Doutorado/data/3D/big/new_db_blocks_3d_big_res"
    
    sim_config_file = os.path.join(data_dir, 'sim_blocks_config_v2.yaml')
    pred_config_file = os.path.join(data_dir, 'gt_blocks_config_v2.yaml')

    process = PostProcessing(sim_config_file)
    process.export_to_ply(
        pred_config_file,
        output_dir = f"{os.path.dirname(pred_config_file)}/gt_blocks_ply",
        initial_step = 0,
        final_step = 500
    )

  """ 
  Cria arquivos ply com coordenadas de partículas de fronteira e suas normais.
  """
  if tutorial == 1.1:
    data_dir = "/work1/Doutorado/data/3D/static"

    #mesh_names = ['armadillo', 'bunny', 'happy', 'dragon', 'rocker-arm']
    mesh_names = ['rocker-arm']

    for mesh_name in tqdm(mesh_names):
      print(f"Mesh: {mesh_name}")
      sim_config_file = f"{data_dir}/{mesh_name}/sim_config_hdp=2.0.yaml"
      gt_config_file = f"{data_dir}/{mesh_name}/gt_config.yaml"
      
      #pred_dir = "sparse_regionwise_approach/predictions/kfold3_static_hdp=2.0"
      pred_dir = "sparse_regionwise_approach/predictions/kfold3__hdp=2.5_checkpoints/*"
      pred_path = f"{data_dir}/{mesh_name}/{pred_dir}/*/pred_config_v2.yaml"
      pred_config_files = glob.glob(pred_path)

      if len(pred_config_files)==0:
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

      for pred_config_file in pred_config_files:
        print(f"Pred: {pred_config_file}")
        process = PostProcessing(sim_config_file)
        process.export_to_ply(
            pred_config_file,
            output_dir = os.path.join(os.path.dirname(pred_config_file), 'ply'),
            initial_step = 0,
            final_step = 0,
            replace=True
        )   


  """ 
  Cria arquivos ply com coordenadas de partículas de fronteira e suas normais para 
  """
  if tutorial == 1.2:
    data_dir = "data/3D/static"

    mesh_names = ['armadillo', 'bunny', 'happy', 'dragon', 'rocker-arm']

    for mesh_name in tqdm(mesh_names):
      print(f"Mesh: {mesh_name}")
      sim_config_file = f"{data_dir}/{mesh_name}/sim_config_hdp=2.0.yaml"
      gt_config_file = f"{data_dir}/{mesh_name}/gt_config.yaml"
      
      pred_dir = "sparse_regionwise_approach/predictions/kfold3__hdp=2.0_checkpoints"
      pred_path = f"{data_dir}/{mesh_name}/{pred_dir}/*/*/pred_config_v2.yaml"
      pred_config_files = glob.glob(pred_path)

      if len(pred_config_files)==0:
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

      for pred_config_file in pred_config_files:
        print(f"Pred: {pred_config_file}")
        process = PostProcessing(sim_config_file)
        process.export_to_ply(
            pred_config_file,
            output_dir = os.path.join(os.path.dirname(pred_config_file), 'ply'),
            initial_step = 0,
            final_step = 0
        )           