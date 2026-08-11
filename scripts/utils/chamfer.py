import os
import trimesh
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import trimesh.visual

def colorize_mesh(mesh, distances):
    distances = distances.flatten()
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
    mesh.visual.vertex_colors = trimesh.visual.interpolate(normalized_distances, color_map='hot')
    return mesh

def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return 0.5*(np.mean(dist_A) + np.mean(dist_B)), dist_A, dist_B

def main(config_1, config_2, initial_step=0, final_step=300, step_size=1, output_dir='chamfer'):
    """ 
    Compute hausdorff metric between the surface from two simulations.
    """
    os.makedirs(output_dir,exist_ok=True)

    id_1 = config_1['id']
    base_dir_1 = config_1['base_dir']
    base_name_1 = config_1['base_name']
    base_sep_1 = config_1['base_sep']
    
    id_2 = config_2['id']
    base_dir_2 = config_2['base_dir']
    base_name_2 = config_2['base_name']
    base_sep_2 = config_2['base_sep']
    
    output_file = f"{output_dir}/chamfer {id_1} x {id_2}.csv"

    if os.path.exists(output_file):
        distances_df = pd.read_csv(output_file)
        mean_dist = distances_df['distance'].mean()
        print(f"Mean distance: {mean_dist}")
        return mean_dist

    distances = []
    for step in range(initial_step, final_step+1, step_size):
        print('Step',step)

        mesh_file_1 = os.path.join(base_dir_1, f"{base_name_1}{base_sep_1}{step}.ply")
        mesh_file_2 = os.path.join(base_dir_2, f"{base_name_2}{base_sep_2}{step}.ply")


        if not os.path.exists(mesh_file_1):
            print(f"Skipping frame. Meshes file {mesh_file_1} does not exist!")
            continue

        if not os.path.exists(mesh_file_2):
            print(f"Skipping frame. Meshes file {mesh_file_2} does not exist!")
            continue  
        
        mesh1 = trimesh.load_mesh(mesh_file_1)
        mesh2 = trimesh.load_mesh(mesh_file_2)

        mean_dist, dist_A, dist_B = chamfer_distance(mesh1.vertices, mesh2.vertices)

        distances.append([step, mean_dist])        
        print(f"Chamfer distance: {mean_dist}")
        print(f"Chamfer distance (gt to pred): {dist_A.mean()}")
        print(f"Chamfer distance (pred to gt): {dist_B.mean()}")

    if len(distances)>0:        
        distances_df = pd.DataFrame(distances, columns=['step', 'distance'])
        distances_df.to_csv(output_file, index=False)
        mean_dist = np.array(distances)[:,1].mean()
        print(f"Mean distance: {mean_dist}")
        return mean_dist
    else:
        print("[WARN] No frames processed!")
        return None

if __name__=="__main__":

    #sim_configs = [('inlet_collision_3d_big_res',3), ('ddb_3d_big_res',4), ('db_blocks_3d_big_res',2), ('inlet_vortex_3d_big_res',1), ('fountain_3d_big_res',0)]
    sim_configs = [
        # {
        #     'name': 'inlet_collision_3d_big_res',
        #     'kfold': 3,
        #     'initial_step': 0,
        #     'final_step': 300,
        #     'flag1': '',
        #     'flag2': ''
        # },
        {
            'name': 'fountain_3d_big_res',
            'kfold': 0,
            'initial_step': 301,
            'final_step': 500,            
            'flag1': '_new_h',
            'flag2': '_new_h_train5'
        }
    ]

    results = []    
    methods = ['vfcnn', 'hpr', 'ia4', 'ss4', 'marrone']
    for sim_config in sim_configs:
        print("=============================")
        print(f"Simulation {sim_config['name']}")
        sim_result = [sim_config['name']]
        configs = [ 
            {
                'id': 'dilts',
                'base_dir': f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/big/{sim_config['name']}/gt{sim_config['flag1']}_ply",
                'base_name': "boundary",
                'base_sep': "."
            },
            {
                'id': f"vfcnn{sim_config['flag2']}",
                'base_dir': f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/big/{sim_config['name']}/sparse_regionwise_approach/predictions/pred_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_{sim_config['kfold']}_kfold4_dilation{sim_config['flag2']}/ply",
                'base_name': "boundary",
                'base_sep': "."
            },
            {
                'id': 'hpr',
                'base_dir': f"/work1/Doutorado/data/3D/big/{sim_config['name']}/other_predictions{sim_config['flag1']}/bpart_hpr/pred",
                'base_name': "out.hpr",
                'base_sep': "."
            }, 
            {
                'id': 'ia4',
                'base_dir': f"/work1/Doutorado/data/3D/big/{sim_config['name']}/other_predictions{sim_config['flag1']}/bpart_ia_4/pred",
                'base_name': "out.hdia",
                'base_sep': "."
            },            
            {
                'id': 'ss4',
                'base_dir': f"/work1/Doutorado/data/3D/big/{sim_config['name']}/other_predictions{sim_config['flag1']}/bpart_ss_4/pred",
                'base_name': "out.ss",
                'base_sep': "."
            },
            {
                'id': 'marrone',
                'base_dir': f"/work1/Doutorado/data/3D/big/{sim_config['name']}/other_predictions{sim_config['flag1']}/marrone/pred",
                'base_name': "out.marrone",
                'base_sep': "."
            }
        ]
        for i in range(1,len(configs)):
            config = configs[i]
            
            if not os.path.exists(config['base_dir']):
                print(f"[WARN] Path {config['base_dir']} does not exist!")
                sim_result.append(None)
            else:                                
                print(f"[OK] Path {config['base_dir']} exist!")
                mean_dist = main(configs[0], config, initial_step=sim_config['initial_step'], final_step=sim_config['final_step'], step_size=1,
                    output_dir=f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/big/{sim_config['name']}/chamfer")
                sim_result.append(mean_dist)
                #sim_result.append(0)
        
        results.append(sim_result)
        
    results_df = pd.DataFrame(results, columns=['sim']+methods)
    results_df.to_csv(f"chamfer-boundary-metrics{sim_config['flag1']}.csv",sep=';', index=False)
    print("\nCHAMFER METRICS\n",results_df)