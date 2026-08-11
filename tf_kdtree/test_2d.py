import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from nn_distance import buildKDTree, searchKDTree

nr_refs = 100000
d = 2

np.random.seed(0)
points = np.random.uniform(size=(nr_refs, d)).astype(np.float32)

t0 = time.time()
# Kdtree
t = time.time()
structured_points, part_nr, shuffled_inds = buildKDTree(points, levels=None) #Use maximum available levels
#print('\n\ntf kdtree build time: ',time.time()-t)

# Busca
knn = 256
t = time.time()

dists_knn, inds_knn = searchKDTree(
    points, part_nr[0], 
    nr_nns_searches = knn, 
    metric = 0, 
    shuffled_inds = shuffled_inds)

count = tf.reduce_sum(tf.cast(dists_knn<0.2,dtype=tf.int32),axis=1)

dists_knn = dists_knn.numpy()
inds_knn = inds_knn.numpy()
count = count.numpy()

#print('tf kdtree search time: ',time.time()-t)
print('tf kdtree time: ',time.time()-t0)

t = time.time()
kdtree = KDTree(
    points,
    metric='euclidean',
    #metric='chebyshev',
    ) 
#neighbors = kdtree.query_radius(points,r=search_radius)
neighbors = kdtree.query(points,k=knn)
print('sklearn time: ',time.time()-t)


# print('shape inds_knn: ',inds_knn.shape)
# print('shape dists_knn: ', dists_knn.shape)
# print('shape count: ',count.shape)

#print('\ninds_knn:\n',inds_knn)
#print('\ndists_knn:\n',dists_knn)

n = 0
# plt.scatter(points[:,0],points[:,1])
# plt.scatter(points[inds_knn[n],0],points[inds_knn[n],1])
# plt.show()