import trimesh
import shutil

meshes1 = [
    {'name': 'armadillo', 'exp_id': 4, 'checkpoint_pred': 73},
    {'name': 'bunny', 'exp_id': 3, 'checkpoint_pred': 28},
    {'name': 'dragon', 'exp_id': 2, 'checkpoint_pred': 21},
    {'name': 'happy', 'exp_id': 1, 'checkpoint_pred': 48},
    {'name': 'rocker-arm', 'exp_id': 0, 'checkpoint_pred': 98}
]

methods = [('ia4','hdia'), ('ss4','ss'), ('hpr','hpr'), ('marrone','marrone')]

for mesh in meshes1:
  #ply_file = f"/work1/Doutorado/data/3D/static/{mesh['name']}/sparse_regionwise_approach/predictions/kfold3__hdp=2.0_checkpoints/{mesh['checkpoint_pred']}/pred_sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0_{mesh['exp_id']}_kfold3_no_coarse/ply/mesh.boundary.0.ply"
  for method in methods:
    ply_file = f"/work1/Doutorado/data/3D/static/{mesh['name']}/other_predictions_hdp=2.0/{method[0]}/pred/mesh.out.{method[1]}.0.ply"
    m = trimesh.load_mesh(ply_file)
    shutil.copy(ply_file, ply_file.replace('.ply','.orig.ply'))
    m.export(ply_file,file_type='ply')