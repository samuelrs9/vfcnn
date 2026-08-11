import os
import shutil
import subprocess
import numpy as np
from sim_reader.data import DataReader
from sim_reader.config import ConfigReader

def run_dilts(frames_dir, gt_dir, gt_config_file, h):
    gt_exe = "/home/samuel/Doutorado/gt3d/build/gt"
    
    log_file = os.path.join(frames_dir,'gt','log','log.txt')
    print('h:',2*h)
    log = subprocess.run([gt_exe, '-g', str(2*h), frames_dir],
                     universal_newlines=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     input=''
                     )
    #print("stdout:", log.stdout)
    
    os.makedirs(os.path.dirname(log_file),exist_ok=True)
    with open(log_file,'w') as f:
        f.write(log.stdout)

    gt_origin = os.path.join(frames_dir,'gt')
    shutil.move(gt_origin,gt_dir)
    
    gt_config = ConfigReader(gt_config_file,'w')
    section = {
        "labels": ['interior', 'boundary'],
        "dir": f"{gt_dir.split('/')[-1]}",
        "base_name": 'gt',
        "extension": 'dat'}
    gt_config.write_section("boundary",section)

# List of meshes
meshes = [
    {'name':'armadillo', 'data_dir': "data/3D/static/armadillo", 'dp': 0.512252},
    {'name':'bunny','data_dir': "data/3D/static/bunny", 'dp': 0.001293},
    {'name':'dragon', 'data_dir': "data/3D/static/dragon", 'dp': 0.655998},
    {'name':'happy','data_dir': "data/3D/static/happy", 'dp': 0.000918},
    {'name':'rocker-arm','data_dir': "data/3D/static/rocker-arm", 'dp': 0.003677}
]

# Loop through each path
for mesh in meshes:
    name = mesh['name']
    data_dir = mesh['data_dir']
    dp = mesh['dp']
    print("=======================================================")
    print(f"Running Dilts method for: {name}")
    if os.path.exists(data_dir):
        Hdp = np.arange(1.1,3.0,0.2)
        for hdp in Hdp:
            h = hdp*dp
            print(f"Hdp: {hdp:.1f}")
            frame_dir = f"{data_dir}/frames_dat"
            gt_dir = f"{data_dir}/gt-tunning/{hdp:.1f}"
            gt_config_file = f"{data_dir}/gt-tunning/gt_config_hdp={hdp:.1f}.ini"
            run_dilts(frame_dir, gt_dir, gt_config_file, h)
    else:
        print(f"{data_dir} does not exist")