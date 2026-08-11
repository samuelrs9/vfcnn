import os
import trimesh
import numpy as np
import pandas as pd
import glob
import csv
from scipy.spatial import cKDTree

def compute_nearest_neighbor_distance(mesh_vertices, points):
    """
    Compute the nearest neighbor distance for each point from the mesh vertices.
    """
    kdtree = cKDTree(points)
    distances, indices = kdtree.query(mesh_vertices, k=1)
    return distances, indices

def labeling_static_mesh_boundary(obj_file, csv_file):
    mesh = trimesh.load(obj_file)
    df = pd.read_csv(csv_file)
    particles = df.to_numpy()    
    distances, indices = compute_nearest_neighbor_distance(mesh.vertices, particles)
    labels = np.zeros((particles.shape[0],1), dtype=int)
    labels[indices] = 1
    normal = np.zeros(particles.shape)
    normal[indices] = mesh.vertex_normals/np.linalg.norm(mesh.vertex_normals, axis=1, keepdims=True)
    normal_df = pd.DataFrame(
        np.hstack([labels, particles, normal]), 
        columns=['labels','x','y','z','nx','ny','nz']
    )
    print("num boundary labels:",labels.sum())
    print("num mesh vertices:",mesh.vertices.shape[0])
    return labels.astype(int), normal_df
    
if __name__=="__main__":
    #data_dir = "/work1/Doutorado/data/3D/static/*"
    data_dir = "/work1/Doutorado/data/3D/static/rocker-arm"
    obj_files = glob.glob(f'{data_dir}/*.obj')
    csv_files = glob.glob(f'{data_dir}/frames/*.csv')

    for file1, file2 in zip(obj_files, csv_files):        
        print("obj_file:", file1)
        print("csv_file:", file2)
        labels, normal_df = labeling_static_mesh_boundary(file1,file2)
        gt_dir = f'{os.path.dirname(file1)}/gt'
        normal_dir = f'{os.path.dirname(file1)}/normal'
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)
        normal_df.to_csv(f'{normal_dir}/normal.0.csv', index=False, sep=';')
        np.savetxt(f'{gt_dir}/gt.0.dat', labels,fmt="%d")
        

        

