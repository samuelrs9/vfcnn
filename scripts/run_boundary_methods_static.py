import os
import subprocess
import shutil
import glob

#meshes = ['armadillo','bunny', 'dragon', 'happy', 'rocker-arm']
meshes = ['rocker-arm']
config_tag = '_hdp=2.0'
bpart = "/home/samuel/BPart/c++/build/BPart_OMP"

data_dir = f"/work1/Doutorado/data/3D/static"

for mesh in meshes:
    print("=======================================================")
    print(f"Mesh: {mesh}")
    mesh_dir = f"/work1/Doutorado/data/3D/static/{mesh}"
    methods = [
        ('hpr', f"{mesh_dir}/frames_dat/simdef_bpart_hpr{config_tag}.ini", f"{data_dir}/configs/pred_config_hpr.ini"),
        ('ia4', f"{mesh_dir}/frames_dat/simdef_bpart_ia{config_tag}.ini", f"{data_dir}/configs/pred_config_ia.ini"),
        ('ss4', f"{mesh_dir}/frames_dat/simdef_bpart_shellsplit{config_tag}.ini", f"{data_dir}/configs/pred_config_ss.ini"),
        ('marrone', f"{mesh_dir}/frames_dat/simdef_marrone{config_tag}.ini", f"{data_dir}/configs/pred_config_marrone.ini")
    ]
    for method,config,pred_config in methods:
        print(f"Running boundary method for: {config}")
        output_dir = f"{mesh_dir}/other_predictions{config_tag}/{method}"
        os.makedirs(f"{output_dir}/pred", exist_ok=True)
        if os.path.isfile(config):
            print(f"{config} exists")
            subprocess.run([bpart, config])
            # move files
            frames_dir = os.path.dirname(config)
            out_files = glob.glob(f"{frames_dir}/out/*")
            for file in out_files:
                shutil.move(file, f"{output_dir}/pred")
            # copy pred config file
            shutil.copy(pred_config, f"{output_dir}/pred_config.ini")
        else:
            print(f"{config} does not exist")