import os
import glob
import argparse
import numpy as np
import pandas as pd

def main(input_dir,output_dir,skip=1, translation=[0,0,0]):
    
    #translation = [-64.72219558, -55.44494758, -58.93945058] # Armadillo
     # Bunny
    # translation =  # Dragon
    # translation =  # Happy
    # translation = [-0.17590323, -0.28162623, -0.52417023] # Rocker arm

    if os.path.exists(output_dir):
        print('Output directory already exists!')
        return  
    else:
        os.mkdir(output_dir)
    frames = glob.glob(input_dir+'/*.dat')
    for file in frames:
        # Read header
        with open(file,'r') as f:
            for l in range(skip):
              print(f.readline())              
        particles = np.loadtxt(file,skiprows=skip)
        particles -= translation
        # Convert to csv
        filename = os.path.splitext(file.split(os.path.sep)[-1])[0]
        output_file = os.path.join(output_dir,filename+'.xyz')
        np.savetxt(output_file,particles,delimiter=' ',fmt='%.9f')

if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('input_dir',type=str,help='Input directory of particle frame coordinates in txt format')
    # parser.add_argument('output_dir',type=str,help='Output diretory of particle frame coordinates in xyz format')
    # parser.add_argument('--skip','-s',type=int,default=1,help='Number of header lines in the frame files')    
    # args = parser.parse_args()    
    #main(args.input_dir, args.output_dir,args.skip)
    
    # static_dir = "/work1/Doutorado/data/3D/static"
    # configs = [
    #     {
    #         'input_dir': f"{static_dir}/armadillo/frames_dat", 
    #         'output_dir': f"{static_dir}/armadillo/frames_xyz", 
    #         "translation": [-64.72219558, -55.44494758, -58.93945058]
    #     },
    #     {
    #         'input_dir': f"{static_dir}/bunny/frames_dat", 
    #         'output_dir': f"{static_dir}/bunny/frames_xyz", 
    #         "translation": [-0.09676097,  0.03091603, -0.06394497]
    #     },        
    #     {
    #         'input_dir': f"{static_dir}/dragon/frames_dat", 
    #         'output_dir': f"{static_dir}/dragon/frames_xyz", 
    #         "translation": [-99.4277232, -42.0996252, -58.0492752]
    #     },
    #     {
    #         'input_dir': f"{static_dir}/happy/frames_dat", 
    #         'output_dir': f"{static_dir}/happy/frames_xyz", 
    #         "translation": [-0.04751173,  0.04835327, -0.04880373]
    #     },                
    #     {
    #         'input_dir': f"{static_dir}/rocker-arm/frames_dat", 
    #         'output_dir': f"{static_dir}/rocker-arm/frames_xyz", 
    #         "translation": [-0.17590323, -0.28162623, -0.52417023]
    #     }
    # ]
    data_dir = "/work1/Doutorado/data/3D/big"
    configs = [
        {
            'input_dir': "/work1/Doutorado/data/3D/big/new_db_blocks_3d_big_res/frames", 
            'output_dir': "/work1/Doutorado/data/3D/big/new_db_blocks_3d_big_res/frames_xyz", 
            "translation": [-0.05, -0.05, -0.05]
        },    
    ]
    for config in configs:
        print("Processing",config['input_dir'])
        main(config['input_dir'], config['output_dir'], 1, config['translation'])

    


