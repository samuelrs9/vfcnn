import os
import trimesh
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import date
from sklearn.neighbors import KDTree
import pandas as pd
import matplotlib.pyplot as plt

def batch_nearest_on_surface(mesh, vertices, batch_size=10000):
    distances = []
    for i in tqdm(range(0, len(vertices), batch_size)):
        batch = vertices[i:i + batch_size]
        _, batch_distances, _ = mesh.nearest.on_surface(batch)
        distances.append(batch_distances)
    return np.concatenate(distances)

def hausdorff_distance(mesh1, mesh2):        
    distances21 = batch_nearest_on_surface(mesh1, mesh2.vertices)
    distances12 = batch_nearest_on_surface(mesh2, mesh1.vertices)
    return max(np.max(distances12), np.max(distances21)), distances12, distances21

def hausdorff_distance2(vertices1, vertices2):
    kdtree1 = KDTree(vertices1)
    distances21, _ = kdtree1.query(vertices2)

    kdtree2 = KDTree(vertices2)
    distances12, _ = kdtree2.query(vertices1)

    return max(np.max(distances12), np.max(distances21)), distances12, distances21

def colorize_mesh(mesh, distances):
    # Normalize distances between 0 and 1
    normalized_distances = distances.flatten() / distances.max()

    # Create a color gradient based on distances
    colors = np.zeros((len(normalized_distances), 3))
    colors[:, 0] = normalized_distances  # Red component
    colors[:, 2] = 1 - normalized_distances  # Blue component

    # Assign colors to vertices
    mesh.visual.vertex_colors = trimesh.visual.random_color()

def main(config_1, config_2, initial_step=0, final_step=300, step_size=1, scale=1, output_dir='hausdorff'):
    """ 
    Compute hausdorff metric between the surface from two simulations.
    """
    id_1 = config_1['id']
    base_dir_1 = config_1['base_dir']
    base_name_1 = config_1['base_name']
    base_sep_1 = config_1['base_sep']
    
    id_2 = config_2['id']
    base_dir_2 = config_2['base_dir']
    base_name_2 = config_2['base_name']
    base_sep_2 = config_2['base_sep']

    # if int(id_2) < 25:
    #     raise Exception("Skipping checkpoint")
    
    # if id_2 in ['9','8','6','5','2','11','24']:
    #     raise Exception("Mesh has a bad state")

    # if 'bunny' in base_dir_2:
    #     if id_2 in ['60','74','48','53']:
    #         raise Exception("Mesh has a bad state")            

    os.makedirs(f"{output_dir}/{id_1} x {id_2}/frames",exist_ok=True)

    output_file = os.path.join(f"{output_dir}/{id_1} x {id_2}", f"hausdorff {id_1} x {id_2}.csv")
    detailed_output_file = os.path.join(f"{output_dir}/{id_1} x {id_2}", f"hausdorff {id_1} x {id_2} - detailed.csv")

    if os.path.exists(output_file):
        distances_df = pd.read_csv(output_file)
        mean_dist = distances_df['distance'].mean()
        print(f"Mean distance: {mean_dist}")
        detailed_df = pd.read_csv(detailed_output_file)
        return mean_dist,detailed_df
    hausdorff = {'model': id_2, 'dist12_mean': [], 'dist12_max': [], 'dist21_mean': [], 'dist21_max': [], 'dist':[]}
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

        #if mesh2.faces.shape[0] > 2*mesh1.faces.shape[0]:
            # print("Simpliying mesh to save memory...")
            # mesh2 = mesh2.simplify_quadric_decimation(int(2*mesh1.faces.shape[0]))

        dist, dist12, dist21 = hausdorff_distance(mesh1, mesh2)
        dist, dist12, dist21 = dist/scale, dist12/scale, dist21/scale

        distances.append([step, dist])

        hausdorff['dist12_mean'].append(dist12.mean())
        hausdorff['dist12_max'].append(dist12.max())
        hausdorff['dist21_mean'].append(dist21.mean())
        hausdorff['dist21_max'].append(dist21.max())  
        hausdorff['dist'].append(dist)                
        
        # file distances 1
        distfile1 = os.path.join(f"{output_dir}/{id_1} x {id_2}/frames", f"dist_{id_1}_{id_2}.{step}.txt")
        np.savetxt(distfile1, dist12, fmt="%.9f")

        # file distances 2
        distfile2 = os.path.join(f"{output_dir}/{id_1} x {id_2}/frames", f"dist_{id_2}_{id_1}.{step}.txt")
        np.savetxt(distfile2, dist21, fmt="%.9f")

    print("Hausdorff distances:",hausdorff)
    detailed_df = pd.DataFrame(hausdorff)
    detailed_df.to_csv(output_file.replace('.csv',' - detailed.csv'),index=False)    
    
    if len(distances)>0:        
        distances_df = pd.DataFrame(distances, columns=['step', 'distance'])
        distances_df.to_csv(output_file, index=False)
        mean_dist = np.array(distances)[:,1].mean()
        print(f"Mean distance: {mean_dist}")

        return mean_dist, detailed_df
    else:
        print("[WARN] No frames processed!")
        return None

def hausdorff_report_static():
    data_dir = '/work1/Doutorado/data/3D/static'
    mesh_names = [
        #('armadillo',4,151.3),
        #('bunny',3,0.16), 
        #('dragon',2,201.74), 
        #('happy',1,0.2), 
        ('rocker-arm',0,1.0)
    ]
    #vfcnn_pred_dir = "sparse_regionwise_approach/predictions/kfold3_static_hdp=2.0"
    vfcnn_pred_dir = "sparse_regionwise_approach/predictions/kfold3_static_hdp=2.0"
    pred_dir = "other_predictions_hdp=2.0"
    results = []    
    for mesh_name in mesh_names:
        print("=============================")
        print(f"Simulation {mesh_name[0]}")
        sim_result = [mesh_name[0]]

        configs = [ 
            {
                'id': 'original',
                'base_dir': f"{data_dir}/{mesh_name[0]}",
                'base_name': mesh_name[0],
                'base_sep': "."
            },
            # {
            #     'id': 'vfcnn',
            #     'base_dir': f"{data_dir}/{mesh_name[0]}/{vfcnn_pred_dir}/pred_sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0_{mesh_name[1]}_kfold3_no_coarse/ply",
            #     'base_name': "mesh.boundary",
            #     'base_sep': "."
            # },         
            {
                'id': 'ia4',
                'base_dir': f"{data_dir}/{mesh_name[0]}/{pred_dir}/ia4/pred",
                'base_name': "mesh.out.hdia",
                'base_sep': "."
            },
            {
                'id': 'ss4',
                'base_dir': f"{data_dir}/{mesh_name[0]}/{pred_dir}/ss4/pred",
                'base_name': "mesh.out.ss",
                'base_sep': "."
            },               
            {
                'id': 'hpr',
                'base_dir': f"{data_dir}/{mesh_name[0]}/{pred_dir}/hpr/pred",
                'base_name': "mesh.out.hpr",
                'base_sep': "."
            },
            {
                'id': 'marrone',
                'base_dir': f"{data_dir}/{mesh_name[0]}/{pred_dir}/marrone/pred",
                'base_name': "mesh.out.marrone",
                'base_sep': "."
            }
            # {
            #     'id': 'filomen',
            #     'base_dir': f"{data_dir}/{mesh_name[0]}/surfaces/filomen-fixed",
            #     'base_name': "Frame",
            #     'base_sep': "."
            # },
            # {
            #     'id': 'zhubridson',
            #     'base_dir': f"{data_dir}/{mesh_name[0]}/surfaces/zhubridson-fixed",
            #     'base_name': "Frame",
            #     'base_sep': "."
            # },
            # {
            #     'id': 'yuturk',
            #     'base_dir': f"{data_dir}/{mesh_name[0]}/surfaces/yuturk-fixed",
            #     'base_name': "Frame",
            #     'base_sep': "."
            # }            
        ]
        target_columns = ['model','dist12_max','dist21_max']
        mesh_results = pd.DataFrame(columns=target_columns)
        for i in range(1,len(configs)):
            config = configs[i]
            if not os.path.exists(config['base_dir']):
                print(f"[WARN] Path {config['base_dir']} does not exist!")
                sim_result.append(None)
            else:
                print(f"[OK] Path {config['base_dir']} exist!")
                mean_dist,detaield_result = main(configs[0], config, initial_step=0, final_step=0, step_size=1, scale=mesh_name[2],
                    output_dir=f"{data_dir}/{mesh_name[0]}/hausdorff")
                sim_result.append(mean_dist)
                mesh_results = pd.concat([mesh_results,detaield_result[target_columns]],axis=0)

        results.append(sim_result)
        print(f"--------------------\n{mesh_name[0]}\n{mesh_results}")
        mesh_output_file = f"{data_dir}/{mesh_name[0]}/hausdorf-metrics-static-{date.today().strftime('%Y%m%d')}.csv"
        mesh_results.to_csv(mesh_output_file,sep=';', index=False)
        
    #results_df = pd.DataFrame(results, columns=['sim','vfcnn0', 'ss5', 'ia4', 'hpr', 'marrone', 'filomen', 'zhubridson', 'yuturk'])
    results_df = pd.DataFrame(results, columns=['sim','vfcnn', 'ss4', 'ia4', 'hpr', 'marrone'])
    #results_df = pd.DataFrame(results, columns=['sim','filomen', 'zhubridson', 'yuturk'])
    output_file = f"{data_dir}/hausdorf-metrics-static-{date.today().strftime('%Y%m%d')}.csv"
    results_df.to_csv(output_file,sep=';', index=False)
    print("\nHAUSDORF METRICS\n",results_df)    

def tune_checkpoint_static():
    data_dir = '/work1/Doutorado/data/3D/static'
    mesh_names = [
        #('armadillo',4,151.3), 
        #('bunny',3,0.16), 
        ('dragon',2,201.74), 
        #('happy',1,0.2), 
        #('rocker-arm',0,1.0)
    ]
    tag = '_hdp=2.0'
    pred_set = f'sparse_regionwise_approach/predictions/kfold3_{tag}_checkpoints'
    model_tag = 'sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0'
    for name in mesh_names:
        print(name)
        gt_config = f'{data_dir}/{name[0]}/gt_config.ini'
        sim_config = f'{data_dir}/{name[0]}/sim_config_hdp=2.0.ini'
        pred_tag = f'pred_{model_tag}_{name[1]}_kfold3_no_coarse'
        pred_dirs = glob.glob(f"{data_dir}/{name[0]}/{pred_set}/*/{pred_tag}/ply")
        config_gt = {
            'id': 'original',
            'base_dir': f"{data_dir}/{name[0]}",
            'base_name': name[0],
            'base_sep': "."
        }
        target_columns = ['model','dist12_max','dist21_max']
        checkpoint_results = pd.DataFrame(columns=target_columns)
        for pred_dir in pred_dirs:
            epoch = pred_dir.split('/')[-3]
            print(f"Epoch: {epoch}")
            config_pred = {
                'id': epoch,
                'base_dir': pred_dir,
                'base_name': "mesh.boundary",
                'base_sep': "."
            }          
            if not os.path.exists(config_pred['base_dir']):
                print(f"[WARN] Path {config_pred['base_dir']} does not exist!")
                continue
            else:
                print(f"[OK] Path {config_pred['base_dir']} exist!")
                try:
                    _,detaield_result = main(config_gt, config_pred, initial_step=0, final_step=0, step_size=1, scale=name[2],
                        output_dir=f"{data_dir}/{name[0]}/hausdorff{tag}")
                    checkpoint_results = pd.concat([checkpoint_results,detaield_result[target_columns]],axis=0)
                except Exception as e:
                    checkpoint_results.append({'model':epoch,'dist12_max':'-','dist21_max':'-'},ignore_index=True)
                    print(e)

        df = pd.DataFrame(checkpoint_results, columns=target_columns)
        
        df['model'] = df['model'].astype(int)
        df = df.sort_values(by='model')
        print(df)
        df.to_csv(f"{data_dir}/{name[0]}/{pred_set}/checkpoint-hausdorff-{date.today().strftime('%Y%m%d')}.csv", index=False, sep=';')
        #plt.plot(df['model'], df['mcc'])
        plt.show()    
        
def tune_hdp_static():
    data_dir = '/work1/Doutorado/data/3D/static'
    mesh_names = [
        #('armadillo',4,151.3), 
        #('bunny',3,0.16), 
        #('dragon',2,201.74), 
        #('happy',1,0.2), 
        ('rocker-arm',0,1.0)
    ]
    tag = 'hdp_tunning'
    pred_set = f'sparse_regionwise_approach/predictions/{tag}'
    model_tag = 'sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0'
    for name in mesh_names:
        print(name)
        gt_config = f'{data_dir}/{name[0]}/gt_config.ini'
        sim_config = f'{data_dir}/{name[0]}/sim_config_hdp=2.0.ini'
        pred_tag = f'pred_{model_tag}_{name[1]}_kfold3_no_coarse'
        pred_dirs = glob.glob(f"{data_dir}/{name[0]}/{pred_set}/*/{pred_tag}/ply")
        config_gt = {
            'id': 'original',
            'base_dir': f"{data_dir}/{name[0]}",
            'base_name': name[0],
            'base_sep': "."
        }
        target_columns = ['model','dist12_max','dist21_max']
        checkpoint_results = pd.DataFrame(columns=target_columns)
        for pred_dir in pred_dirs:
            epoch = pred_dir.split('/')[-3]
            print(f"Epoch: {epoch}")
            config_pred = {
                'id': epoch,
                'base_dir': pred_dir,
                'base_name': "mesh.boundary",
                'base_sep': "."
            }          
            if not os.path.exists(config_pred['base_dir']):
                print(f"[WARN] Path {config_pred['base_dir']} does not exist!")
                continue
            else:
                print(f"[OK] Path {config_pred['base_dir']} exist!")
                try:
                    _,detaield_result = main(config_gt, config_pred, initial_step=0, final_step=0, step_size=1, scale=name[2],
                        output_dir=f"{data_dir}/{name[0]}/hausdorff_{tag}")
                    checkpoint_results = pd.concat([checkpoint_results,detaield_result[target_columns]],axis=0)
                except Exception as e:
                    checkpoint_results.append({'model':epoch,'dist12_max':'-','dist21_max':'-'},ignore_index=True)
                    print(e)

        df = pd.DataFrame(checkpoint_results, columns=target_columns)
        
        df['model'] = df['model'].astype(float)
        df = df.sort_values(by='model')
        print(df)
        df.to_csv(f'{data_dir}/{name[0]}/{pred_set}/hausdorff_{tag}.csv', index=False, sep=';')
        #plt.plot(df['model'], df['mcc'])
        plt.show()    
        
def tune_decision_threshold_static():
    data_dir = '/work1/Doutorado/data/3D/static'
    mesh_names = [
        #('armadillo',4,151.3), 
        #('bunny',3,0.16), 
        #('dragon',2,201.74), 
        #('happy',1,0.2), 
        ('rocker-arm',0,1.0)
    ]
    tag = 'decision_threshold_tunning'
    pred_set = f'sparse_regionwise_approach/predictions/{tag}'
    model_tag = 'sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0'
    for name in mesh_names:
        print(name)
        gt_config = f'{data_dir}/{name[0]}/gt_config.ini'
        sim_config = f'{data_dir}/{name[0]}/sim_config_hdp=2.0.ini'
        pred_tag = f'pred_{model_tag}_{name[1]}_kfold3_no_coarse'
        pred_dirs = glob.glob(f"{data_dir}/{name[0]}/{pred_set}/*/{pred_tag}/ply")
        config_gt = {
            'id': 'original',
            'base_dir': f"{data_dir}/{name[0]}",
            'base_name': name[0],
            'base_sep': "."
        }
        target_columns = ['model','dist12_max','dist21_max']
        checkpoint_results = pd.DataFrame(columns=target_columns)
        for pred_dir in pred_dirs:
            epoch = pred_dir.split('/')[-3]
            print(f"Epoch: {epoch}")
            config_pred = {
                'id': epoch,
                'base_dir': pred_dir,
                'base_name': "mesh.boundary",
                'base_sep': "."
            }          
            if not os.path.exists(config_pred['base_dir']):
                print(f"[WARN] Path {config_pred['base_dir']} does not exist!")
                continue
            else:
                print(f"[OK] Path {config_pred['base_dir']} exist!")
                try:
                    _,detaield_result = main(config_gt, config_pred, initial_step=0, final_step=0, step_size=1, scale=name[2],
                        output_dir=f"{data_dir}/{name[0]}/hausdorff_{tag}")
                    checkpoint_results = pd.concat([checkpoint_results,detaield_result[target_columns]],axis=0)
                except Exception as e:
                    checkpoint_results.append({'model':epoch,'dist12_max':'-','dist21_max':'-'},ignore_index=True)
                    print(e)

        df = pd.DataFrame(checkpoint_results, columns=target_columns)
        
        df['model'] = df['model'].astype(float)
        df = df.sort_values(by='model')
        print(df)
        df.to_csv(f'{data_dir}/{name[0]}/{pred_set}/hausdorff_{tag}.csv', index=False, sep=';')
        #plt.plot(df['model'], df['mcc'])
        plt.show()    
        

if __name__=="__main__": 
    hausdorff_report_static()
    #tune_hdp_static()
    #tune_checkpoint_static()
    #tune_decision_threshold_static()