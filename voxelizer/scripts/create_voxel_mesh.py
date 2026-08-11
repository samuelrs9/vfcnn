import os
import numpy as np

res = 0.4
points = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])

script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                 [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
faces_idx = np.array([[3,2,1],[1,4,3],[5,6,7],[7,8,5],
                     [1,2,6],[6,5,1],[7,3,4],[4,8,7],
                     [4,1,5],[5,8,4],[2,3,7],[7,6,2]])
normals_idx = np.array([5,5,6,6,3,3,4,4,1,1,2,2])

normals_str = ('# normals\n'
                'vn -1 0 0\n'
                'vn 1 0 0\n'
                'vn 0 -1 0\n'
                'vn 0 1 0\n'
                'vn 0 0 -1\n'
                'vn 0 0 1\n')
verts_str = '# vertices\n'
faces_str = '# faces\n'
for i in range(points.shape[0]):
    # Cria vértices
    for j in range(verts.shape[0]):  
        v = points[i]+res*verts[j]
        verts_str += f'v {v[0]} {v[1]} {v[2]}\n'
    # Cria faces
    for j in range(faces_idx.shape[0]):
        f = i*8 + faces_idx[j]
        vn = normals_idx[j]
        faces_str += f'f {f[0]}//{vn} {f[1]}//{vn} {f[2]}//{vn}\n'

# Escreve no arquivo obj
with open('gen_cubes.obj','w') as mesh_obj:
    mesh_obj.write(normals_str+'\n')
    mesh_obj.write(verts_str+'\n')
    mesh_obj.write(faces_str)