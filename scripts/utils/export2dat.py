import os
import glob
from vfnet.preprocessing import PreprocessSimulation

sim_config_files = glob.glob("/work1/Doutorado/data/3D/static/rocker-arm/sim_config_hdp=2.0.ini")
for file in sim_config_files:
    print(file)
    # Paths derivados
    data_dir = os.path.dirname(file)
    frames_dir = os.path.join(data_dir,'frames_dat')
    gt_dir = os.path.join(data_dir,'gt')
    gt_config_file = os.path.join(data_dir,'gt_config.ini')

    # 2. Converte as saídas .csv exportadas do DualSPHysics para o formato .dat 
    # usado no código Dilts 3D.
    print("Converting step files to dat format...")
    preprocessing = PreprocessSimulation(file)
    preprocessing.convert_points_data(output_dir=frames_dir,output_extension='dat')
    print("Done!")