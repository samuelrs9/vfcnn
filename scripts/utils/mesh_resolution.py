import glob
import trimesh
import pandas as pd
import numpy as np
from datetime import date
from scipy.spatial import cKDTree

def load_obj_and_compute_statistics(obj_file):
    mesh = trimesh.load(obj_file)
    if not mesh.is_watertight:
        print("Warning: The mesh is not watertight, edge computation may be inaccurate.")
    edge_lengths = mesh.edges_unique_length

    num_verts = mesh.vertices.shape[0]
    min_coords = mesh.vertices.min(axis=0)
    max_coords = mesh.vertices.max(axis=0)

    #print("Bounding box")
    #print(f"min_x = {min_coords[0]}\nmin_y = {min_coords[1]}\nmin_z = {min_coords[2]}\n")
    #print(f"max_x = {max_coords[0]}\nmax_y = {max_coords[1]}\nmax_z = {max_coords[2]}\n")

    # Calculate the mean edge length
    mean_edge_length = np.mean(edge_lengths)

    return {
        'dp_bound': mean_edge_length, 'num_bound': num_verts, 
        'min_x': min_coords[0],  'min_y': min_coords[1],  'min_z': min_coords[2],
        'max_x': max_coords[0],  'max_y': max_coords[1],  'max_z': max_coords[2]
    }

def load_csv_and_compute_statistics(csv_file):
    df = pd.read_csv(csv_file)
    points = df.to_numpy()    
    kdtree = cKDTree(points)
    distances, indices = kdtree.query(points, k=2)    
    mean_distance = distances[:,1].mean()
    return {'dp_all': mean_distance, 'num_points': points.shape[0]}

if __name__=="__main__":
    #meshes = ['armadillo', 'bunny', 'dragon', 'happy', 'rocker-arm']    
    meshes = ['rocker-arm']
    data_dir = "/work1/Doutorado/data/3D/static"
    #obj_files = glob.glob(f'{data_dir}/*.obj')
    #csv_files = glob.glob(f'{data_dir}/frames/*.csv')
        
    all_meshes = {}
    for mesh in meshes:
        print(mesh)
        all_particles_file = f'{data_dir}/{mesh}-2/frames/{mesh}.msh.0.csv'
        boundary_particles_file = f'{data_dir}/{mesh}/{mesh}.obj'
        all_points = load_csv_and_compute_statistics(all_particles_file)
        bound_points = load_obj_and_compute_statistics(boundary_particles_file)
        all_points.update(bound_points)
        all_meshes[mesh] = all_points

    df = pd.DataFrame(all_meshes).T
    df.to_csv(f"data/3D/static/mesh-statistics-{date.today().strftime('%Y%m%d')}.csv")
    print(df)
        

