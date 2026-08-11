import os
from pyexpat import features
import time
import numpy as np
from numpy.core.fromnumeric import ndim
from scipy import sparse
import tensorflow as tf
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

from plots import Plots2D

class SparseVoxelizer:
    
    def __init__(self,limits,res,data_reader=False,enable_plot=False,expand_limits=True):
        """ 
        Construtor.
        Última atualização: 04/09/2021
        
        Args:
            limits: limites da voxelização.
            res:    resolução numérica.
            enable_plot:
        """
        # Atualiza os limites
        if expand_limits:
            limits = limits + [-res,res]
        num_parts = np.ceil((limits[:,1] - limits[:,0])/res).astype(int)
        limits[:,1] = limits[:,0] + num_parts * res
            
        self.res = res
        self.limits = limits
        self.ndim = limits.shape[0]
        self.size = num_parts + 1
        self.global_coord = [
            np.linspace(self.limits[i,0],self.limits[i,1],self.size[i]) 
            for i in range(self.ndim)]
                            
        self.enable_plot = enable_plot
        if self.enable_plot:      
            if data_reader:
                self.plot = Plots2D(data_reader=data_reader,voxelizer=self)
            else:
                self.plot = Plots2D(voxelizer=self)
        
    def set_points(self,points,coord_type=['grid_coord','local_coord'],use_flat_index=False):
        """
        Carrega pontos no grid esparso.
        Última modificação: 01/09/2021.
        
        Args:
            points: coordenadas dos pontos. 
            
        """
        self.points = points
        self.origin_coord = [self.global_coord[i][0] for i in range(self.ndim)]        
        coords = self.compute_coordinates(
            points,self.origin_coord,coord_type=coord_type,use_flat_index=use_flat_index)
        if 'grid_coord' in coord_type:
           self.grid_coord = coords['grid_coord']
        if 'local_coord' in coord_type:
           self.local_coord = coords['local_coord'] 
            
    def compute_coordinates(self,points_coord,origin_coord,
        coord_type=['grid_coord','local_coord'],use_flat_index=False):
        """
        Dado um conjunto de pontos retorna as suas coordenadas de grid.
        Última atualização: 04/09/2021.
        
        Args:
            points_coord: coordenadas dos pontos.
            origin_coord: 
            coord_type:
            use_flat_index:
        
        Returns:
           output: a dict
        """
        coords = (points_coord-origin_coord)/self.res
        local_coord,grid_coord = np.modf(coords)
        grid_coord = np.array(grid_coord).astype(int)
        output = {}
        if 'grid_coord' in coord_type:
            if use_flat_index:
                grid_coord = self.multi_to_flat_index(grid_coord)
            output['grid_coord'] = grid_coord        
        if 'local_coord' in coord_type:
            output['local_coord'] = local_coord
        return output

    def compute_coordinates_bkp(self,points_coord,origin_coord,use_flat_index=False):
        """
        Dado um conjunto de pontos retorna as suas coordenadas de grid.
        Última atualização: 04/09/2021.
        
        Args:
            points_coord: coordenadas dos pontos.
            origin_coord: 
        
        Returns:
           grid_coord: coordenadas dos pontos no grid.
        """
        grid_coord = [np.floor((points_coord[:,i]-origin_coord[i])/
                      self.res).astype(int) for i in range(self.ndim)]
        if use_flat_index:
            return self.multi_to_flat_index(np.asarray(grid_coord))
        else:            
            return np.asarray(grid_coord).transpose()

    def extract_images_pointwise(self,points_idx,image_size,neighbors,
        num_channels=None,channels=['occupancy','local_coords']):
        """ 
        Extrai imagens de ocupância usando kdtree para busca de vizinhos.
        Última modificação: 12/04/2022.
        
        Args:
            points_idx: índices de pontos.
            image_size:  resolução das imagens de saída.
            neighbors:
            num_channels:
            channels:
                
        Returns:
            images:
        """
        n_images = points_idx.shape[0]
        img_radius = int(0.5*image_size)

        if num_channels is None:
            num_channels =  0
            if 'occupancy' in channels:
                num_channels += 1
            if 'local_coords' in channels:
                num_channels += self.ndim

        images = np.zeros((n_images,)+self.ndim*(image_size,)+(num_channels,))

        for i in range(n_images):
            # Vizinhos em coordenadas de grid
            if type(neighbors) is tuple:
                count = neighbors[1][i]
                neighbors_coord = self.grid_coord[neighbors[0][i][0:count]]
                normal = self.local_coord[neighbors[0][i][0:count]]
            else:
                neighbors_coord = self.grid_coord[neighbors[i]] # Vizinhos em coordenadas de grid
                normal = self.local_coord[neighbors[i]]
            #print(neighbors_coord)

            # Converte para coordenadas de imagem
            left_bottom_coord = self.grid_coord[points_idx[i]] - img_radius
            neighbors_image_coord = self.grid_coord_to_image_coord(
                neighbors_coord, left_bottom_coord,
                image_size = image_size, return_flat_coord = True)
            
            channel_idx = 0
            if 'occupancy' in channels:
                np.put(images[i,...,channel_idx],neighbors_image_coord,1)
                channel_idx += 1
            if 'local_coords' in channels:
                np.put(images[i,...,channel_idx],neighbors_image_coord,normal[:,0])
                np.put(images[i,...,channel_idx+1],neighbors_image_coord,normal[:,1])
                if self.ndim==3:
                    np.put(images[i,...,channel_idx+2],neighbors_image_coord,normal[:,2])

            # plt.imshow(images[i,...,0])
            # plt.show()
            # plt.pause(1)
                        
        if self.enable_plot and self.ndim==2:
            radius = self.res * img_radius
            img_length = self.res * image_size            

            self.plot.scatter(self.points[points_idx],clear_axes=True)
            
            p_idx = 0
            p_grid_i = self.grid_coord[points_idx[p_idx]][0]
            p_grid_j = self.grid_coord[points_idx[p_idx]][1]
            p_left_bottom_coord = np.array(
                [self.global_coord[0][p_grid_i-img_radius],
                self.global_coord[1][p_grid_j-img_radius]])
           
            self.plot.draw_image(
                images[p_idx],
                position = p_left_bottom_coord,
                length = [img_length,img_length],
                show_frame = True,
                show_grid = 2)
            
            self.plot.draw_rectangles(
                p_left_bottom_coord.reshape(-1,2),img_length, img_length)
            
            self.plot.draw_circles(
                self.points[points_idx[p_idx]].reshape(-1,2), radius)            
            
        return images

    def extract_images_regionwise(self,full_neighbors,target_neighbors,
        centers_grid_coord,target_labels=None,image_size=31,interior_size=13,
        border_size=9,return_dense=False,return_sparse=False,return_labels=False):
        """ 
        Extrai imagens (2D ou 3D) usando kdtree para busca de vizinhos.
        Última modificação: 15/03/2022.
        
        Args:
            full_neighbors:
            target_neighbors:
            centers_grid_coord:
            target_labels:
            image_size:
            interior_size:
            border_size:
            count_neighbors:
            return_dense:
            return_sparse:
            return_labels:
                
        Returns:
            images:
        """    
        n_images = target_neighbors.shape[0]     
        image_radius = int(0.5*image_size)
                
        # Array de imagens tri-dimensionais
        if return_dense:
            if self.ndim==2:
                images_nn = np.zeros((n_images,image_size,image_size,2))
                if return_labels:
                    labels_mask = np.zeros((n_images,image_size,image_size,2))
            elif self.ndim==3:
                images_nn = np.zeros((n_images,image_size,image_size,image_size,2))
                if return_labels:
                    labels_mask = np.zeros((n_images,image_size,image_size,image_size,2))

        if return_sparse:
            full_neighbors_image_coord = np.empty(n_images,dtype=object)
            target_neighbors_image_coord = np.empty(n_images,dtype=object)
            #if return_labels: 
                #full_labels = np.empty(n_images,dtype=object)
                #target_labels = np.empty(n_images,dtype=object)

        for k in range(n_images):
            if type(full_neighbors) is tuple:
                count = full_neighbors[1][k]
                neighbors_f = full_neighbors[0][k][0:count] # Índices de pontos na imagem 3d completa
                neighbors_f = neighbors_f.astype(int)
            else:
                neighbors_f = full_neighbors[k] # Índices de pontos na imagem 3d completa
            neighbors_t = target_neighbors[k]   # Índices de pontos que serão classificados na imagem 3d
            neighbors_t = neighbors_t.astype(int)

            # Pontos em coordenadas do grid 
            neighbors_coord_f = self.grid_coord[neighbors_f,:]
            neighbors_coord_t = self.grid_coord[neighbors_t,:]

            #t = time.time()
            # Converte para coordenadas de imagem
            left_bottom_grid_coord_f = centers_grid_coord[k] - image_radius
            neighbors_image_coord_f = self.grid_coord_to_image_coord(
                neighbors_coord_f, left_bottom_grid_coord_f,
                image_size = image_size, return_flat_coord = True)                        
            neighbors_image_coord_t = self.grid_coord_to_image_coord(
                neighbors_coord_t, left_bottom_grid_coord_f,
                image_size = image_size, return_flat_coord = True)

            if return_dense:
                # Cria os canais da imagem
                np.put(images_nn[k,...,0], neighbors_image_coord_f, 1)
                np.put(images_nn[k,...,1],neighbors_image_coord_t,
                    np.ones((neighbors_image_coord_t.shape[0])))
                if return_labels:
                    # Interior
                    np.put(labels_mask[k,...,0],neighbors_image_coord_t,
                        (target_labels[k]==0).astype(int))
                    # Fronteira
                    np.put(labels_mask[k,...,1],neighbors_image_coord_t,
                        (target_labels[k]==1).astype(int))
            if return_sparse:
                full_neighbors_image_coord[k] = neighbors_image_coord_f
                target_neighbors_image_coord[k] = neighbors_image_coord_t
            
            #t2 += time.time() - t             
        
        output = {}
        if return_dense:
            output['images'] = images_nn
            if return_labels:
                output['labels_mask'] = labels_mask

        if return_sparse:
            output['full_neighbors_image_coord'] = full_neighbors_image_coord
            output['target_neighbors_image_coord'] = target_neighbors_image_coord
            if return_labels:
                #output['full_labels'] = full_labels
                output['target_labels'] = target_labels

        return output

    def build_image_channels(self,particle_image,dense_binary_channels=False):
        """ 
        
        Args:
            
        Returns:
            
        """              
        idx = particle_image.indices.numpy()
        mainx_idx = (self.border_size <= idx[:,0]) & (idx[:,0] < particle_image.shape[0]-self.border_size)
        mainy_idx = (self.border_size <= idx[:,1]) & (idx[:,1] < particle_image.shape[1]-self.border_size)        
        main_idx = mainx_idx & mainy_idx
        
        channel_1 = tf.sparse.SparseTensor(indices=idx,
                                           values=np.ones(idx.shape[0]),
                                           dense_shape=particle_image.shape)    
        
        channel_2 = tf.sparse.SparseTensor(indices=idx[main_idx,:],
                                           values=np.ones(idx[main_idx,:].shape[0]),
                                           dense_shape=particle_image.shape)                
        
        particle_idx = tf.sparse.SparseTensor(indices=idx[main_idx,:],
                                           values=particle_image.values.numpy()[main_idx],
                                           dense_shape=particle_image.shape)                
        
        channel_1 = tf.sparse.reorder(channel_1)
        channel_2 = tf.sparse.reorder(channel_2)   
        particle_idx = tf.sparse.reorder(particle_idx)
        
        if dense_binary_channels:
            channel_1 = tf.sparse.to_dense(channel_1,validate_indices=False).numpy()
            channel_2 = tf.sparse.to_dense(channel_2,validate_indices=False).numpy()
            return particle_idx,np.concatenate((np.expand_dims(channel_1,axis=-1),
                                   np.expand_dims(channel_2,axis=-1)),axis=2)
        else:
            return particle_idx,[channel_1,channel_2]
                
    def build_image_channels_v2(self,points_coord,center,return_local_coord=False,local_coord_id=1):
        """ 
        
        Args:
            points_coord:       pontos em coordenadas de grid.
            center:             centro em coordenadas de grid.
            return_local_coord: 
            local_coord_id:
            
        Returns:
            
        """
        n_channels = len(points_coord)
        image = np.zeros((1,self.image_size,self.image_size,n_channels))          
        
        for i in range(n_channels):
            # Converte para coordenadas locais de imagem
            points_local_coord = points_coord[i] - center + [self.R,self.R]
            points_local_coord = points_local_coord.transpose()
            
            # Converte para indexação flat
            points_local_coord_flat = np.ravel_multi_index(points_local_coord,
                                                           (self.image_size,self.image_size)) 
            
            # Preenche a imagem binária usando as coordenadas locais
            np.put(image[0,:,:,i],points_local_coord_flat,
                   np.ones((points_coord[i].shape[0])))
            
            if return_local_coord and (i==local_coord_id):
                local_coord_flat = points_local_coord_flat
                
        if return_local_coord:
            return image,local_coord_flat
        else:
            return image
        
    def grid_coord_to_image_coord(
        self,grid_coord,left_bottom_coord,image_size,return_flat_coord=False):
        """ 
        
        Args:
            grid_coord:           pontos em coordenadas de grid.
            left_bottom_coord:    centro em coordenadas de grid.
            image_size:
            return_flat_coord: 
            
        Returns:
            
        """
        # Converte para coordenadas locais de imagem
        image_coord = grid_coord - left_bottom_coord        
        if return_flat_coord:
            return np.ravel_multi_index(image_coord.T,self.ndim*(image_size,))
        else:
            return image_coord
      
    def find_voxelization(self,points_idx,targets=None,res=0.1,
        return_global_coord=False,return_res=False,return_unique=True,
        return_flat=False,return_points_per_voxel=False,
        return_grid_coord=False,debug=False):
        """
        Encontra voxelização de um subconjunto de pontos (2D ou 3D).
        Última atualização: 31/03/2022.
        
        Args:
            points_idx:
            targets:
            res:
            return_global_coord:
            return_res:
            return_unique:
            return_flat:
            return_points_per_voxel:
            return_grid_coord:
            debug:
        Returns: 
            voxelization
        """
        output = {}
        voxelization = SparseVoxelizer(self.limits,res=res,enable_plot=debug)
        voxelization.set_points(self.points[points_idx])

        if return_grid_coord:
            if return_unique:
                output['voxels_grid_coord'] = voxelization.find_unique_voxels(return_flat=return_unique)
            else:
                if return_flat:
                    output['voxels_grid_coord'] = voxelization.multi_to_flat_index(voxelization.grid_coord)
                else:
                    output['voxels_grid_coord'] = voxelization.grid_coord

        if return_global_coord:
            voxels = voxelization.find_unique_voxels(return_flat=False)
            output['voxel_global_coord'] = voxelization.voxel_coord_to_global_coord(voxels).T

        if return_points_per_voxel:
            voxels_flat = voxelization.multi_to_flat_index(voxelization.grid_coord)
            voxel_argsort = voxels_flat.argsort()
            voxel_sort = voxels_flat[voxel_argsort]
            points_sort_per_voxels = points_idx[voxel_argsort]
            unique_voxels_flat,non_empty_voxel_index = np.unique(voxel_sort,return_index=True)
            # usa índices de points_idx
            output['points_idx_per_voxel'] = np.array(np.split(points_sort_per_voxels,non_empty_voxel_index[1:]),dtype=object)
            # usa índices de 1 a points_idx.shape[0]
            output['idx_per_voxel'] = np.array(np.split(voxel_argsort,non_empty_voxel_index[1:]),dtype=object)            
            if targets is not None:
                for key in targets:
                    target_sort_per_voxels = targets[key][voxel_argsort]
                    output[key+'_per_voxel'] = np.array(
                        np.split(target_sort_per_voxels,non_empty_voxel_index[1:]),dtype=object)

        if return_res:
            output['voxel_res'] = voxelization.res

        # Only plot if enabled
        if debug and hasattr(voxelization, 'plot') and voxelization.plot is not None:
            voxelization.plot.draw_voxels(voxels_flat)

        return output

    def find_points_per_voxel(self,voxels=None):
        """ 
        Encontra pontos por voxel. 
        Última atualização: 25/10/2021.
        """
        if voxels is None:
            voxels = self.grid_coord
        voxels = self.multi_to_flat_index(voxels)
        grid_argsort = voxels.argsort()
        grid_sort = voxels[grid_argsort]
        non_empty_cells,non_empty_cells_index = np.unique(grid_sort,return_index=True)        
        points_in_cell = np.asarray(np.split(grid_argsort,non_empty_cells_index[1:]),dtype=object)      
        return points_in_cell
    
    def find_unique_voxels(self,points_idx=None,return_flat=False):
        """
        Encontra voxels únicos de um subconjunto pontos.
        Última atualização: 01/11/2021.
        
        Args:
            points_idx:
            return_flat:
        Returns: 
            unique_voxels:
        """
        if points_idx is None:
            voxels_ij = self.grid_coord
        else:
            voxels_ij = self.grid_coord[points_idx,:]        
        voxels_flat = self.multi_to_flat_index(voxels_ij)        
        unique_voxels_flat = np.unique(voxels_flat)
        if return_flat:
            return unique_voxels_flat
        else:
            return self.flat_to_multi_index(unique_voxels_flat)
                
    def create_voxel_mesh(
        self,mesh_file,voxel_coord=None,points_idx=None,voxel_length=None):
        """ 
        Cria malha de voxels a partir coordenadas de grid.
        Última atualização: 19/10/2021.

        Args:
            mesh_file:
            voxel_coord:
            points_idx:
            voxel_length:
        """
        if voxel_coord is None:
            if points_idx is None:
                voxel_coord = self.find_unique_voxels().T
            else:
                voxel_coord = self.find_unique_voxels(points_idx)

            points = self.voxel_coord_to_global_coord(voxel_coord).T
        else:
            #points = voxel_coord - voxel_coord.mean(axis=0)
            points = voxel_coord

        if voxel_length is None:
            voxel_length = self.res

        if points.ndim==1:
            points = points[np.newaxis,:]

        verts = np.array(
            [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
        faces_idx = np.array(
            [[3,2,1],[1,4,3],[5,6,7],[7,8,5],
            [1,2,6],[6,5,1],[7,3,4],[4,8,7],
            [4,1,5],[5,8,4],[2,3,7],[7,6,2]])
        normals_idx = np.array(
            [5,5,6,6,3,3,4,4,1,1,2,2])

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
                v = points[i]+voxel_length*verts[j]
                verts_str += f'v {v[0]} {v[1]} {v[2]}\n'
            # Cria faces
            for j in range(faces_idx.shape[0]):
                f = i*8 + faces_idx[j]
                vn = normals_idx[j]
                faces_str += f'f {f[0]}//{vn} {f[1]}//{vn} {f[2]}//{vn}\n'

        # Escreve no arquivo obj
        with open(mesh_file,'w') as mesh_obj:
            mesh_obj.write(normals_str+'\n')
            mesh_obj.write(verts_str+'\n')
            mesh_obj.write(faces_str)

    def multi_to_flat_index(self,multi_index):
        """ 
        Faz a conversão de indexação mutidimensional para indexação flat.
        Última modificação: 01/11/2021.
        
        Args:
            multi_index:
            
        Return:
            flat index.
        """             
        return np.ravel_multi_index(multi_index.transpose(),self.size)
    
    def flat_to_multi_index(self,flat_index):
        """ 
        Faz a conversão de indexação flat para indexação multidimensional.
        Última modificação: 01/11/2021.
        
        Args:
            flat index:
            
        Return:
            multi index.
        """        
        return np.array(np.unravel_index(flat_index,self.size))

    def voxel_coord_to_global_coord(self,voxel_coord):
        """ 
        Faz a conversão de coordenadas de voxels para coordenadas globais. 
        Ùltima atualização: 19/10/2021
        
        Args:
            voxel_coord:

        Return:
            global_coord:
        """
        global_coord = np.zeros(voxel_coord.shape)
        for i in range(voxel_coord.shape[1]):
            global_coord[:,i] = [self.global_coord[j][voxel_coord[j,i]] for j in range(self.ndim)]
        
        return global_coord
