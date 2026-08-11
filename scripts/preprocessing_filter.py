import os
import sys
import glob
import argparse
import numpy as np

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_dir))

from sim_reader.data import DataReader
from sim_reader.config import ConfigReader
from vfnet.base import VoxelFluid
from vfnet.report import Reports

def run(sim_config_file,test_type,search_radius,gt_config_file=None):
  """ Run offline coarse test. """
  vfluid = VoxelFluid(
      data_config_file = sim_config_file,
      tasks = ['boundary'],
      search_radius = search_radius,
      enable_plot = False)

  vfluid.coarse_prediction_offline(
      test_type = test_type,
      combined_threshold = 0.7,
      prediction_id = 0,
      initial_step = -1,
      final_step = -1,
      skip_steps = 10,
      device = 'gpu',
      save_outputs = True,
      gt_config_file=gt_config_file)

def fit_threshold(sim_config_file,gt_config_file,coarse_tests_dir):
  """ Carrega dos dados de densidade e centroides de vários testes
    grosseiros e encontra o melhor o melhor limiar de decisão.
  """
  data_reader = DataReader(sim_config_file)
  coarse_config_files = glob.glob(
      os.path.join(coarse_tests_dir, '*', '*.ini'))

  report = Reports(data_reader)

  # Processa saídas dos testes grosseiros
  report.process_coarse_test_outputs(
      gt_config_file, coarse_config_files, output_dir=coarse_tests_dir)

  # Processa thresholds
  report.process_coarse_test_threshold(
      gt_config_file, coarse_config_files, output_dir=coarse_tests_dir)

  # CONTINUAR DAQUI
  # Analise final dos melhores hiperparâmetros do teste grosseiro
  lost_ratio = np.asarray(
      [0, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01])
  results = report.analyze_coarse_test_parameters(
      gt_config_file, coarse_config_files,
      lost_ratio, output_dir=coarse_tests_dir)

def report(sim_config_file,gt_config_file,coarse_tests_dir):

  coarse_config_files = glob.glob(os.path.join(coarse_tests_dir, '*', 'pred_config.ini'))

  data_reader = DataReader(sim_config_file)
  report = Reports(data_reader)

  report.coarse_test_analysis(
      gt_config_file=gt_config_file,
      pred_config_files=coarse_config_files)

if __name__=="__main__":  
  #parser = argparse.ArgumentParser()
  #parser.add_argument('task',type=str,help="task to be performed, this can be run or report")
  #parser.add_argument('-t','--test_type',type=str,default='combined_product',help="coarse test type, this can be density, centroid, combined_logical, combined_product")
  #parser.add_argument('-s','--sim_config_file',type=str,help="simulation config file")
  #parser.add_argument('-g','--gt_config_file',type=str,help="ground-truth config file")
  #parser.add_argument('-r,'--search_radius',type=float,default=1.5,help="particle search radius, this must be a multiplicative factor of kernel length parameter (h)")
  #parser.parse_args()

  task = 'run'
  
  # sim config
  sim_config_file = "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/db_blocks_3d_big_res/sim_config.ini"
  #sim_config_file = "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/ddb_3d_big_res/sim_config.ini"
  #sim_config_file = "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/sim_config.ini"
  #sim_config_file = "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/inlet_collision_3d_big_res/sim_config.ini"
  #sim_config_file = "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/inlet_vortex_3d_big_res/sim_config.ini"  

  data_dir = os.path.dirname(sim_config_file)
  gt_config_file = os.path.join(data_dir,'gt_config.ini')
  coarse_tests_dir = os.path.join(
    data_dir, 
    'coarse_predictions',
    #'compare_radius'
  )

  if task=='run':
    test_type = 'all' 
    search_radius = 2.0 
    if test_type=='all' :
      run(sim_config_file,test_type,search_radius,gt_config_file)
    else:
      run(sim_config_file,test_type,search_radius)

  elif task=='fit':
    fit_threshold(sim_config_file,gt_config_file,coarse_tests_dir)


  elif task=='report':
    report(sim_config_file,gt_config_file,coarse_tests_dir)
