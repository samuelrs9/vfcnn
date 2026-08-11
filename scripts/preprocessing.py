import os
import glob
import sys
from random import random
import numpy as np
import argparse
import yaml
import subprocess
import shutil

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_dir))

from vfnet.base import VoxelFluid
from vfnet.models import VFRWCNN

from vfnet.cnn_models.normal_region_3d import Models31 as NRCNN31
from vfnet.cnn_models.normal_region_3d import Models25 as NRCNN25

from vfnet.preprocessing import Curves2D, DataProcessing, PreprocessSimulation
from vfnet.report import Reports
from vfnet.utils import ressample_simulation

from sim_reader.data import DataReader
from sim_reader.config import ConfigReader

from metrics.classification import Report

from vfnet.plots import Plots2D

def main(data_dir, gencase_file, frames_dir, initial_step, final_step, ressampled_data_dir):
    
    # Paths derivados
    frames_dat_dir = os.path.join(data_dir,'frames_dat')
    gt_dir = os.path.join(data_dir,'gt')
    sim_config_file = os.path.join(data_dir,'sim_config.yaml') 
    gt_config_file = os.path.join(data_dir,'gt_config.yaml')

    # Paths derivados para a simulação reamostrada
    ressampled_frames_dir = os.path.join(ressampled_data_dir,'frames') 
    ressampled_gt_dir = os.path.join(ressampled_data_dir,'gt')
    ressampled_normal_dir = os.path.join(ressampled_data_dir,'normal_sph') 
    ressampled_sim_config_file = os.path.join(ressampled_data_dir, 'sim_config.yaml')  
    ressampled_gt_config_file = os.path.join(ressampled_data_dir,'gt_config.yaml')       

    # 1. Process the DualSPHysics simulation properties file to create the simulation configuration file.
    print("1. Creating simulation configuration file...")
    preprocessing = PreprocessSimulation()
    preprocessing.create_sim_config_from_dualsphysics(
        data_dir,frames_dir,gencase_file, extension='csv',
        initial_step=initial_step,final_step=final_step,origin='dualsphysics')
    print("Done!")

    # 2. Converte as saídas .csv exportadas do DualSPHysics para o formato .dat 
    # usado no código Dilts 3D.
    print("\n2. Converting step files to dat format...")
    preprocessing = PreprocessSimulation(sim_config_file)
    preprocessing.convert_points_data(output_dir=frames_dat_dir,output_extension='dat')
    print("Done!")

    # 3. Roda o ground-truth Dilts e cria o arquivo de configuração.   
    print("3. Running ground-truth...")
    if not os.path.exists(gt_dir):
        #gt_exe = os.path.join(os.getcwd(),"gt3d/build/gt")
        gt_exe = "/home/samuel/Doutorado/gt3d/build/gt"
        
        sim_reader = DataReader(sim_config_file)
        h = sim_reader.properties_info['h']
        
        log_file = os.path.join(frames_dat_dir,'gt','log','log.txt')
        log = subprocess.run([gt_exe,'-g',str(2*h),frames_dat_dir],
            universal_newlines=True,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        #print("stdout:", log.stdout)
        
        os.makedirs(os.path.dirname(log_file),exist_ok=True)
        with open(log_file,'w') as f:
            f.write(log.stdout)

        gt_origin = os.path.join(frames_dat_dir,'gt')
        shutil.move(gt_origin,gt_dir)
        
        gt_config = ConfigReader(gt_config_file,'w')
        section = {
            "labels":"interior boundary",
            "dir": "gt",
            "base_name": "gt",
            "extension": "dat"}
        gt_config.write_section("boundary",section)
    print("Done!")

    # 4. Reamostra a simulação.
    print("4. Running simulation ressampling...")
    preprocessing = PreprocessSimulation(sim_config_file)
    ressample_simulation(
        data_reader = preprocessing.data_reader,
        gt_config_file = gt_config_file,
        section = 'boundary',
        initial_step = initial_step,
        final_step = final_step,
        save = True,
        save_num_particles = True,
        enable_plot = False,
        extension = 'dat',
        output_dir = ressampled_data_dir)    
    print("Done!")

    # 5. Roda o ground-truth Dilts na simulação reamostrada e cria o arquivo de configuração.   
    print("5. Running ground-truth in the ressampled simulation...")    
    if not os.path.exists(ressampled_gt_dir):
        #gt_exe = os.path.join(os.getcwd(),"gt3d/build/gt")        
        gt_exe = "/home/samuel/Doutorado/gt3d/build/gt"
        
        sim_reader = DataReader(ressampled_sim_config_file)
        h = sim_reader.properties_info['h']

        log_file = os.path.join(ressampled_frames_dir,'gt','log','log.txt')
        log = subprocess.run([gt_exe,'-g',str(2*h),ressampled_frames_dir],
            universal_newlines=True,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        #print("stdout:", log.stdout)
        
        os.makedirs(os.path.dirname(log_file))
        with open(log_file,'w') as f:
            f.write(log.stdout)

        gt_origin = os.path.join(ressampled_frames_dir,'gt')
        gt_dest = os.path.join(ressampled_data_dir,'gt')
        shutil.move(gt_origin,gt_dest)        

        gt_config = ConfigReader(ressampled_gt_config_file,'w')
        section = {
            "labels":"interior boundary",
            "dir": "gt",
            "base_name": "gt",
            "extension": "dat"}
        gt_config.write_section("boundary",section)
    print("Done!")  

    # 6. Roda o ground-truth de normal usando o operador SPH
    print("6. Running normal ground-truth in the ressampled simulation...")
    if not os.path.exists(ressampled_normal_dir):
        preprocessing = PreprocessSimulation(ressampled_sim_config_file)
        preprocessing.compute_normals_sph(
            gt_config_file = ressampled_gt_config_file,
            section = 'boundary',
            search_radius = 2.00,
            use_only_boundary=False,
            initial_step = 0,
            final_step = 500,
            enable_plot = False,
            save = True,
            extension = 'csv',
            output_dir = ressampled_normal_dir)
    print("Done!")  

if __name__=="__main__":
    """ Pre-processamento de simulação do dual sphphysics. """
    #parser = argparse.ArgumentParser()
    #parser.add_argument('preprocessing_config',type=str,help="preprocessing config file")
    #parser.parse_args()

    preprocessing_config = "/work1/Doutorado/data/3D/big/originals/07_DamBreakCubes/preprocessing_config.yaml"

    with open(preprocessing_config,'r') as f:
        config = yaml.safe_load(f)

    # Basic args
    args = config['arguments']
    data_dir = os.path.dirname(preprocessing_config)
    gencase_file = os.path.join(data_dir,args['gencase_file'])
    frames_dir = os.path.join(data_dir,args['frames_dir'])
    initial_step = int(args['initial_step'])
    final_step = int(args['final_step'])
    if 'ressampled_data_dir' in args:
        ressampled_data_dir = args['ressampled_data_dir']
    else:
        ressampled_data_dir = os.path.dirname(os.path.dirname(data_dir))
        ressampled_data_dir = os.path.join(ressampled_data_dir,data_dir.split(os.sep)[-1])

    main(data_dir, gencase_file, frames_dir, initial_step, final_step, ressampled_data_dir)