import os

from sklearn import neighbors
os.chdir(os.path.dirname(__file__))

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from neighbors import KDTree

np.random.seed(0)
fig,axs = plt.subplots(2,2)

num_points = 10000
dim = 2
points = np.random.uniform(size=(num_points,dim)).astype(np.float32)

metric = 'infinity'  # 'euclidian' or 'infinity'

print('knn query')
knn = 200
# CPU
t = time.time()
kdtree_cpu = KDTree(
    points,
    device='cpu',
    metric=metric) 
#neighbors = kdtree.query_radius(points,r=search_radius)
neighbors_cpu,dists_cpu = kdtree_cpu.query(points,knn)
print('--> cpu time: ',time.time()-t)

# GPU
t = time.time()
kdtre_gpu = KDTree(
    points,
    device='gpu',
    metric=metric)
neighbors_gpu,dists_gpu = kdtre_gpu.query(points,knn)
dists_gpu = dists_gpu.numpy()
neighbors_gpu = neighbors_gpu.numpy()
print('--> gpu time): ',time.time()-t)
#print('--> neighbors equal = ',np.array_equal(neighbors_cpu,neighbors_gpu))

n = 0
axs[0,0].scatter(points[:,0],points[:,1])
axs[0,0].scatter(points[neighbors_cpu[n],0],points[neighbors_cpu[n],1])
axs[0,1].scatter(points[:,0],points[:,1])
axs[0,1].scatter(points[neighbors_gpu[n],0],points[neighbors_gpu[n],1])
#plt.show()

print('radius query')
radius = 0.05
# CPU
t = time.time()
kdtree_cpu = KDTree(
    points,
    device='cpu',
    metric=metric) 
#neighbors = kdtree.query_radius(points,r=search_radius)
neighbors_cpu,dists_cpu = kdtree_cpu.query_radius(points,radius)
print('--> cpu time: ',time.time()-t)

# GPU
t = time.time()
kdtre_gpu = KDTree(
    points,
    device='gpu',
    metric=metric)
neighbors_gpu,dists_gpu,count = kdtre_gpu.query_radius(points,radius)
dists_gpu = dists_gpu.numpy()
neighbors_gpu = neighbors_gpu.numpy()
count = count.numpy()
print('--> gpu time): ',time.time()-t)
#print('--> neighbors equal = ',np.array_equal(neighbors_cpu,neighbors_gpu))

n = 0
axs[1,0].scatter(points[:,0],points[:,1])
axs[1,0].scatter(points[neighbors_cpu[n],0],points[neighbors_cpu[n],1])
axs[1,1].scatter(points[:,0],points[:,1])
axs[1,1].scatter(points[neighbors_gpu[n,0:count[n]],0],points[neighbors_gpu[n,0:count[n]],1])
plt.show()