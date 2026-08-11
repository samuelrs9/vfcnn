import os
import sys
from sklearn.neighbors import KDTree as KDTreeSklearn
import tensorflow as tf

try:
    sys.path.append(os.path.dirname(__file__))
    import nn_distance
except:
    print('Tf KDTree não foi carregada corretamente!')

class KDTree:

    def __init__(self,points,device,metric='euclidean'):
        self.device = device
        self.metric = metric
        if self.device=='cpu':
            if self.metric=='euclidean':
                self.sklearn_kdtree = KDTreeSklearn(
                    points,metric='euclidean')
            elif self.metric=='infinity':
                self.sklearn_kdtree = KDTreeSklearn(
                    points,metric='chebyshev')

        elif self.device=='gpu':
            structured_points,part_nr,shuffled_inds = nn_distance.buildKDTree(points)
            self.structured_points = structured_points
            self.part_nr = part_nr
            self.shuffled_inds = shuffled_inds

    def query(self,points,knn=1):
        if self.device=='cpu':
            dists,neighbors = self.sklearn_kdtree.query(points,k=knn)
        elif self.device=='gpu':
            if self.metric=='euclidean':
                metric = 0
            elif self.metric=='infinity':
                metric = 1
            dists,neighbors = nn_distance.searchKDTree(
                points,self.part_nr[0],
                nr_nns_searches=knn,
                metric=metric,
                shuffled_inds=self.shuffled_inds)
            neighbors = tf.cast(neighbors,dtype=tf.int32)
            if self.metric=='euclidean':
                dists = tf.sqrt(dists)
        return neighbors,dists

    def query_radius(self,points,radius,max_knn=200,verbose=False):
        if max_knn>512:
             # 512 é o número máximo de vizinhos estipulado na implementação do tf_kdtree,
             # para permitir quantidades maiores é necessário recompilar a lib
            max_knn = 512
        if self.device=='cpu':
            neighbors,dists, = self.sklearn_kdtree.query_radius(
                points,r=radius,return_distance=True,sort_results=True)
            return neighbors,dists
        elif self.device=='gpu':
            if self.metric=='euclidean':
                metric = 0
            elif self.metric=='infinity':
                metric = 1
            with tf.device('gpu'):
                dists,neighbors = nn_distance.searchKDTree(
                    points,self.part_nr[0],
                    nr_nns_searches=max_knn,
                    metric=metric,
                    shuffled_inds=self.shuffled_inds)
                neighbors = tf.cast(neighbors, dtype=tf.int32)
                if self.metric=='euclidean':
                    dists = tf.sqrt(dists)
                count = tf.reduce_sum(tf.cast(dists<radius,dtype=tf.int32),axis=1)                    
                max_count = int(tf.reduce_max(count))
                if verbose:
                    print(f'Warning: maximum knn found is {max_count} and maximum knn allowed is {max_knn}!')            
                    if max_count<0.8*max_knn:
                        print('Warning: you might decrease max knn!!')
                    elif 0.8*max_knn <= max_count < 0.95*max_knn:
                        print('Warning: good choice for max_knn!')
                    elif max_count>0.95*max_knn:
                        print('Warning: you might increase max_knn!')
                
                return neighbors,dists,count
