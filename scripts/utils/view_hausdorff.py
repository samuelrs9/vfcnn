import os
import trimesh
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd
import open3d as o3d

def load_mesh(mesh_file, distance_file):
    mesh = trimesh.load_mesh(mesh_file)
    distance = np.loadtxt(distance_file)
    mesh.visual.vertex_colors = trimesh.visual.interpolate(distance, color_map='jet')

    bounding_box = mesh.bounding_box.bounds
    min_bound = bounding_box[0]
    max_bound = bounding_box[1]

    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound

    scale = 2.0 / np.max(size)

    mesh.apply_translation(-center)  # Translate the mesh to center it at the origin
    mesh.apply_scale(scale)  # Scale the mesh to fit within [-1, 1]        

    return mesh

def convert_to_o3d(mesh):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.compute_vertex_normals()
    if mesh.visual.vertex_colors is not None:
        vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Ignore alpha channel
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)    
    return mesh_o3d
        

if __name__=="__main__":

    meshes = [
        {'name': 'armadillo', 'exp_id': 4, 'scale': 151.3, 'checkpoint_pred': 73},
        {'name': 'bunny', 'exp_id': 3, 'scale': 0.16, 'checkpoint_pred': 28},
        {'name': 'dragon', 'exp_id': 2, 'scale': 201.74, 'checkpoint_pred': 21},
        {'name': 'happy', 'exp_id': 1, 'scale': 0.2, 'checkpoint_pred': 48},
        {'name': 'rocker-arm', 'exp_id': 0, 'scale': 1.0, 'checkpoint_pred': 98}
    ]
    test_id = 1
    meshes_obj = []
    scene = trimesh.Scene()       
    for i,mesh in enumerate(meshes):
        configs = [
            {
                'id': 'vfcnn',
                'mesh': f"/work1/Doutorado/data/3D/static/{mesh['name']}/sparse_regionwise_approach/predictions/kfold3__hdp=2.0_checkpoints/{mesh['checkpoint_pred']}/pred_sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0_{mesh['exp_id']}_kfold3_no_coarse/ply/mesh.boundary.0.ply",
                'hausdorff': f"/work1/Doutorado/data/3D/static/{mesh['name']}/hausdorff/original x {mesh['checkpoint_pred']}/frames/dist_{mesh['checkpoint_pred']}_original.0.txt",
            }
            # {
            #     'id': 'ia4',
            #     'mesh': f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/{mesh_name[0]}/other_predictions/ia4/mesh.out.hdia.{test_id}.ply",
            #     'hausdorff': f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/{mesh_name[0]}/hausdorff/original x ia4/frames/dist_ia4_original.{test_id}.txt",
            # },
            # {
            #     'id': 'ss4',
            #     'mesh': f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/{mesh_name[0]}/other_predictions/ss4/mesh.out.ss.{test_id}.ply",
            #     'hausdorff': f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/{mesh_name[0]}/hausdorff/original x ss4/frames/dist_ss4_original.{test_id}.txt",
            # },          
            # {
            #     'id': 'hpr',
            #     'mesh': f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/{mesh_name[0]}/other_predictions/hpr/mesh.out.hpr.{test_id}.ply",
            #     'hausdorff': f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/{mesh_name[0]}/hausdorff/original x hpr/frames/dist_hpr_original.{test_id}.txt",
            # },                                
            # {
            #     'id': 'marrone',
            #     'mesh': f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/{mesh_name[0]}/other_predictions/marrone/mesh.out.marrone.{test_id}.ply",
            #     'hausdorff': f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/{mesh_name[0]}/hausdorff/original x marrone/frames/dist_marrone_original.{test_id}.txt",
            # }
        ]


        for j,config in enumerate(configs):
            mesh = load_mesh(config['mesh'], config['hausdorff'])         
            mesh = mesh.apply_translation([j,i,0])
            #mesh_o3d = convert_to_o3d(mesh)
            meshes_obj.append(mesh)
        
        scene.add_geometry(meshes_obj)

    scene.show()
            


        

        