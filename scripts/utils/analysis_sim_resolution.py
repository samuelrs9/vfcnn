import os
from sim_reader.data import DataReader
from vfnet.report import Reports


""" 
Analisa a razão entre o parâmetro H do SPH e a distância média entre 
partículas mais próximas.
"""
working_dir = "/home/samuel/Doutorado/voxel-fluid-net"
#data_dir = os.path.join(working_dir,'data','dambreak3d.0_ressampled')
#data_dir = os.path.join(working_dir,'data/3D/big/ddb_3d_big_res') # fold 4 
#data_dir = os.path.join(working_dir,'data/3D/big/inlet_collision_3d_big_res') # fold 3
#data_dir = os.path.join(working_dir,'data/3D/big/db_blocks_3d_big_res') # fold 2
#data_dir = os.path.join(working_dir,'data/3D/big/inlet_vortex_3d_big_res')  # fold 1
data_dir = os.path.join(working_dir,'data/3D/big/fountain_3d_big_res') # fold 0
sim_config_file = os.path.join(data_dir,'sim_config_v2.yaml')

data_reader = DataReader(sim_config_file)

report = Reports(data_reader)
report.ratio_sph_kernel_and_distance_particles()