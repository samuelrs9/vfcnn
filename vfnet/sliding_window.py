import os
import time
import h5py
import configparser
import multiprocessing as mp
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from numba import cuda
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sim_reader.data import DataReader
from sim_reader.config import ConfigReader

from vfnet.plots import Plots2D
from voxelizer.sparse_voxelizer import SparseVoxelizer
from voxelizer.multi_sparse_voxelizer import MultiSparseVoxelizer

from vfnet.util import trilinear_interpolate

# Try to import TF KDTree, fall back to scipy if not available
USE_TF_KDTREE = False
try:
    from tf_kdtree.neighbors import KDTree
    USE_TF_KDTREE = True
except:
    print('Tf KDTree não foi carregada corretamente!')
    try:
        from scipy.spatial import cKDTree
        print('Using scipy.spatial.cKDTree as fallback')
    except ImportError:
        print('Warning: Neither tf_kdtree nor scipy.spatial is available!')

class Core:
    
    def __init__(self,data_config_file=None,tasks=None,available_tasks=None,
        features=['occupancy','local_coords'],approach=None,ref_length=None,
        search_radius=None,real_data_limits=None,image_size=None,border_size=None,
        spatial_dimensions=None,enable_plot=False,coarse_prediction='combined_product'):
        """ 
        Construtor.
        Última atualização: 01/06/2022.

        Args:             
            data_config_file: caminho do arquivo configuração dos dados.
            tasks: lista de tarefas que serão executadas.
            available_tasks: lista de tarefas disponíveis.
            features: lista de features.
            approach: tipo de abordagem, 'pointwise' ou 'regionwise'.
            ref_length: comprimento de referência (resolução numérica).
            search_radius: comprimento do raio de busca em relação à 'ref_length'.
            grid_length: comprimento da célula (voxel) do grid em relação à 'ref_length'.
            real_grid_length: comprimento real da célula do grid.
            real_grid_limits: limites reais do grid.
            image_size: tamanho das janelas em relação ao 'grid_length'.
            border_size: tamanho da borda das imagens em relação à 'grid_length'.
            image_length: comprimento das imagens em relação à 'ref_length'
            real_image_length: comprimento real das imagens.
            spatial_dimensions: dimensão espacial dos dados.
            enable_plot: variável booleana para habilitar ou desabilitar os plots.
            
        """
        if data_config_file is None:
            raise ValueError("Specify a data configuration file!")

        if not os.path.exists(data_config_file):
            raise FileNotFoundError("Data configuration file not found!")

        if approach is None:
            raise ValueError("Specify the string 'pointwise' or 'regionwise' in the 'approach' keyword argument!")

        if tasks is None:
            raise ValueError(("Specify the list of tasks that will be perfomed. "
                f"The list of available tasks is:  {available_tasks}"))

        for task in tasks:
            if task not in available_tasks:
                raise ValueError(f"Task '{task}' is not available!!!")
                
        self.available_tasks = available_tasks
        self.tasks = tasks
        self.task_dimensions = {}
        self.task_types = {}
        self.labels = []
        self.coarse_prediction = coarse_prediction
        
        self.features = features
        self.data_dir = os.path.dirname(data_config_file)
        self.data_reader = DataReader(data_config_file)

        self.set_free_attributes(
            approach = approach, 
            ref_length = ref_length,
            search_radius = search_radius,
            image_size = image_size,
            border_size = border_size, 
            real_data_limits = real_data_limits, 
            spatial_dimensions = spatial_dimensions, 
            enable_plot = enable_plot)

    def set_free_attributes(self,approach=None,ref_length=None,
        search_radius=None,image_size=None,border_size=None,
        real_data_limits=None,spatial_dimensions=None,enable_plot=None):
        """ 
        Define atributos livres.
        Última modificação: 28/02/2022.
        """
        if approach is not None:
            self.approach = approach

        if ref_length is not None:
            self.ref_length = ref_length

        if search_radius is not None:
            self.search_radius = search_radius

        if image_size is not None:            
            self.image_size = image_size

        if border_size is not None:            
            self.border_size = border_size
        
        if real_data_limits is not None:
            self.real_data_limits = real_data_limits

        if spatial_dimensions is not None:
            self.spatial_dimensions = spatial_dimensions

        if enable_plot is not None:
            self.enable_plot = enable_plot

            if self.enable_plot:
                self.plot = Plots2D(self.data_reader)
            else:
                self.plot = None

    def set_dependent_attributes(self):
        """ 
        Define atributos dependentes.
        Última modificação: 28/02/2022.
        """
        try:
            self.real_search_radius = self.search_radius * self.ref_length
        except ValueError:
            print("'search_radius' and 'ref_length' are required to compute the value of 'real_search_radius'!")

        try:
            #self.grid_length = (2 * self.search_radius) / (self.image_size - 1)
            self.grid_length = 0.1 # tamanho do grid refinado em relação a resolução numérica da simulação (ref_length)
        except ValueError:
            print("'image_size' and 'search_radius' are required to compute the value of 'grid_length'!")

        try:
            self.real_grid_length = self.grid_length * self.ref_length
        except ValueError:
            print("'grid_length' and 'ref_length' are required to compute the value of 'real_grid_length'!")
        
        try:
            self.image_length = self.grid_length * self.image_size
        except ValueError:
            print("'grid_length' and 'image_size' are required to compute the value of 'image_length'!")

        try:
            self.real_image_length = self.image_length * self.ref_length
        except ValueError:
            print("'image_length' and 'ref_length' are required to compute the value of 'real_image_length'!")

        try:
            self.real_grid_limits = self.real_data_limits + self.real_image_length*np.asarray([-1,1])
        except ValueError:
            print("'real_data_limits' and 'real_image_length' are required to compute the value of 'real_grid_limits'!")

    def build_dataset_pointwise(self,points,gt_tasks,selected_idx=None,
        dataset_dir=None,train_name='train',val_name='validation',batch_size=512,
        max_batches=-1,voxelizer=None):
        """ 
        Cria o conjunto de treinamento para a abordagem pontual.
        Última modificação: 03/06/2022.
        
        Args:
            points:
            gt_tasks:
            selected_idx:
            dataset_dir:            
            remove_existing:
            train_name:
            val_name:
            batch_size:
            max_batches:
            voxelizer:
                  
        Returns:
            times:
        """
        t0 = time.time()

        # Cria o diretório de saída caso não exista
        os.makedirs(dataset_dir,exist_ok=True)
        
        # Nomes dos datasets de treino e validação
        trainset_file = os.path.join(dataset_dir,f'{train_name}.hdf5')
        valset_file = os.path.join(dataset_dir,f'{val_name}.hdf5')

        num_channels = 0
        if 'occupancy' in self.features:
            num_channels += 1
        if 'local_coords' in self.features:
            num_channels += self.spatial_dimensions

        initial_shape = [0] + self.spatial_dimensions*[self.image_size] + [num_channels]
        max_shape = [None] + self.spatial_dimensions*[self.image_size] + [num_channels]
        
        # Cria o dataset de treino caso ele não exista
        if not os.path.exists(trainset_file):
            with h5py.File(trainset_file,'a') as f:
                # Attributes
                #f.attrs['shape'] = initial_shape
                f.attrs['num_samples'] = 0
                f.attrs['image_size'] = self.image_size
                f.attrs['image_shape'] = initial_shape[1:]
                f.attrs['spatial_dimensions'] = self.spatial_dimensions
                f.attrs['num_channels'] = num_channels
                f.attrs['tasks'] = self.tasks
                f.attrs['task_dimensions'] = list(self.task_dimensions.values())
                f.attrs['task_types'] = list(self.task_types.values())
                
                # Dataset arrays
                f.create_dataset(
                    'images', dtype='i4', shape = initial_shape,
                    maxshape = max_shape, chunks = True,
                    compression = "gzip", compression_opts = 9)
                
                for task in self.tasks:
                    if self.task_types[task]=='classification':
                        f.create_dataset(
                            task,dtype = 'i4',shape=(0,),
                            maxshape=(None,),chunks=True,
                            compression="gzip",compression_opts=9)
                    elif self.task_types[task]=='regression':
                        f.create_dataset(
                            task,dtype='f',shape=(0,self.task_dimensions[task]),
                            maxshape=(None,self.spatial_dimensions),chunks=True,
                            compression="gzip",compression_opts=9)

        # Cria o dataset de validação caso ele não exista
        if not os.path.exists(valset_file):
            with h5py.File(valset_file,'a') as f:
                # Attributes
                f.attrs['num_samples'] = 0
                f.attrs['image_size'] = self.image_size
                f.attrs['image_shape'] = initial_shape[1:]
                f.attrs['num_spatial_dimensions'] = self.spatial_dimensions
                f.attrs['num_channels'] = num_channels
                f.attrs['tasks'] = self.tasks
                f.attrs['task_dimensions'] = list(self.task_dimensions.values())
                f.attrs['task_types'] = list(self.task_types.values())         
                
                # Dataset arrays
                f.create_dataset(
                    'images',dtype='i4',shape=initial_shape,
                    maxshape=max_shape,chunks=True,
                    compression="gzip",compression_opts=9)            
                for task in self.tasks:
                    if self.task_types[task]=='classification':
                        f.create_dataset(
                            task,dtype = 'i4',shape=(0,),
                            maxshape=(None,),chunks=True,
                            compression="gzip",compression_opts=9)
                    elif self.task_types[task]=='regression':
                        f.create_dataset(
                            task,dtype='f',shape=(0,self.task_dimensions[task]),
                            maxshape=(None,self.spatial_dimensions),chunks=True,
                            compression="gzip",compression_opts=9)

        # Voxelizador
        if voxelizer is None:
            voxelizer = SparseVoxelizer(
                limits = self.real_grid_limits,
                res = self.real_grid_length,
                data_reader=self.data_reader,
                enable_plot = False)
        
        total_train_images = 0
        total_val_images = 0  
        
        # Groundtruth selecionado
        if selected_idx is None:
            gt_selected = gt_tasks
        else:
            gt_selected = {}
            for task in self.tasks:        
                gt_selected[task] = gt_tasks[task][selected_idx]

        kdtree = KDTree(points,device='cpu')
        neighbors,_ = kdtree.query_radius(points[selected_idx],self.real_search_radius)
        
        # Carrega partículas na voxelização
        voxelizer.set_points(points)

        t = time.time()        
        
        # Geração de imagens por batches
        num_selected = selected_idx.shape[0]
        num_batches = 1 + num_selected//batch_size            
        for k,batch in enumerate(np.array_split(range(num_selected),num_batches)):
            if 0 < max_batches and k >= max_batches:
                print(' --> Maximum number of batches reached!')
                break
            print(f' --> Batch {k+1}/{num_batches}')
            t = time.time()
            batch_selected_idx = selected_idx[batch]

            batch_gt_selected = {}
            for task in self.tasks:
                batch_gt_selected[task] = gt_selected[task][batch]
            
            # Extrai imagens
            batch_selected_images = voxelizer.extract_images_pointwise(
                points_idx = batch_selected_idx,
                image_size = self.image_size,
                neighbors = neighbors[batch_selected_idx],
                num_channels = num_channels,
                channels = self.features)
            print(f' ----> images generating time: {time.time()-t:.4f} s')
        
            # 3. Separa imagens de treino e teste
            t = time.time()

            train_idx, val_idx = train_test_split(np.arange(batch_selected_idx.shape[0]),test_size=0.25,random_state=0)
            images_train, images_val = batch_selected_images[train_idx], batch_selected_images[val_idx]

            batch_gt_train = {}
            batch_gt_val = {}
            for task in self.tasks:
                batch_gt_train[task], batch_gt_val[task] = batch_gt_selected[task][train_idx], batch_gt_selected[task][val_idx]
            
            total_train_images +=  images_train.shape[0]
            total_val_images +=  images_val.shape[0]

            # Salva dados de treino
            with h5py.File(trainset_file,'a') as f:
                f.attrs['num_samples'] = total_train_images 
                # Imagens
                f['images'].resize([total_train_images] + initial_shape[1:])
                f['images'][-images_train.shape[0]:] = images_train
                
                for task in self.tasks:
                    if self.task_types[task]=='classification':
                        f[task].resize((total_train_images,))
                        f[task][-batch_gt_train[task].shape[0]:] = batch_gt_train[task]
                    if self.task_types[task]=='regression':
                        f[task].resize((total_train_images,self.task_dimensions[task]))
                        f[task][-batch_gt_train[task].shape[0]:,:] = batch_gt_train[task]
                                    
            # Salva dados de validação
            with h5py.File(valset_file,'a') as f:
                f.attrs['num_samples'] = total_val_images
                # Voxels
                f['images'].resize([total_val_images] + initial_shape[1:])
                f['images'][-images_val.shape[0]:] = images_val                    

                for task in self.tasks:
                    if self.task_types[task]=='classification':
                        f[task].resize((total_val_images,))
                        f[task][-batch_gt_val[task].shape[0]:] = batch_gt_val[task]
                    if self.task_types[task]=='regression':
                        f[task].resize((total_val_images,self.task_dimensions[task]))
                        f[task][-batch_gt_val[task].shape[0]:,:] = batch_gt_val[task]

            print(f' ----> images storage time: {time.time()-t:.4f} s')
        
        total_time = time.time()-t0

        return {'train_images':total_train_images,
            'val_images':total_val_images,'time':total_time}

    def build_dataset_regionwise(self,points,gt_tasks,selected_idx=None,
        dataset_dir=None,train_name='train',val_name='validation',batch_size=512,
        max_batches=-1,voxelizer=None,save_data_to_debug=False):
        """ 
        Cria o conjunto de treinamento para a abordagem regional.
        Última modificação: 03/06/2022.
        
        Args:
            points:
            gt_tasks:
            selected:
            dataset_dir:            
            remove_existing:
            train_name:
            val_name:
            batch_size:
            max_batches:
            voxelizer:
            save_data_to_debug:
                  
        Returns:
            times:
        """
        t0 = time.time()

        # Cria o diretório de saída caso não exista
        os.makedirs(dataset_dir,exist_ok=True)
        
        # Nomes dos datasets de treino e validação
        trainset_file = os.path.join(dataset_dir,f'{train_name}.hdf5')
        valset_file = os.path.join(dataset_dir,f'{val_name}.hdf5')

        image_shape = self.spatial_dimensions*[self.image_size]+[2]

        # Cria o dataset de treino caso ele não exista
        if not os.path.exists(trainset_file):
            with h5py.File(trainset_file,'a') as f:
                # Attributes
                f.attrs['num_samples'] = 0
                f.attrs['image_shape'] = image_shape
                f.attrs['image_size'] = self.image_size
                f.attrs['border_size'] = self.border_size
                f.attrs['spatial_dimensions'] = self.spatial_dimensions
                f.attrs['tasks'] = self.tasks
                f.attrs['task_dimensions'] = list(self.task_dimensions.values())
                f.attrs['task_types'] = list(self.task_types.values())
                
                # Dataset arrays
                f.create_dataset(
                    'full_num_voxels',dtype='i4',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'full_voxels_coord',dtype='i4',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'target_num_voxels',dtype='i4',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'target_voxels_coord',dtype='i4',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
            
                for task in self.tasks:
                    if self.task_types[task]=='classification':
                        f.create_dataset(
                            f'target_{task}',dtype='i4',shape=(0,),maxshape=(None,),
                            chunks=True,compression="gzip",compression_opts=9)
                    if self.task_types[task]=='regression':
                        f.create_dataset(
                            f'target_{task}',dtype='f',shape=(0,self.task_dimensions[task]),
                            maxshape=(None,self.task_dimensions[task]),
                            chunks=True,compression="gzip",compression_opts=9)
                                                
        # Cria o dataset de validação caso ele não exista
        if not os.path.exists(valset_file):
            with h5py.File(valset_file,'a') as f:
                f.attrs['num_samples'] = 0
                f.attrs['image_shape'] = image_shape
                f.attrs['image_size'] = self.image_size
                f.attrs['border_size'] = self.border_size
                f.attrs['spatial_dimensions'] = self.spatial_dimensions
                f.attrs['tasks'] = self.tasks
                f.attrs['task_dimensions'] = list(self.task_dimensions.values())
                f.attrs['task_types'] = list(self.task_types.values())

                # Dataset arrays            
                f.create_dataset(
                    'full_num_voxels',dtype='i4',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'full_voxels_coord',dtype='i4',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)            
                f.create_dataset(
                    'target_num_voxels',dtype='i4',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'target_voxels_coord',dtype='i4',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)            
                
                for task in self.tasks:
                    if self.task_types[task]=='classification':
                        f.create_dataset(
                            f'target_{task}',dtype='i4',shape=(0,),maxshape=(None,),
                            chunks=True,compression="gzip",compression_opts=9)
                    if self.task_types[task]=='regression':
                        f.create_dataset(
                            f'target_{task}',dtype='f',shape=(0,self.task_dimensions[task]),
                            maxshape=(None,self.task_dimensions[task]),
                            chunks=True,compression="gzip",compression_opts=9)           

        # Voxelizador
        if voxelizer is None:
            voxelizer = SparseVoxelizer(
                limits = self.real_grid_limits,
                res = self.real_grid_length,
                data_reader = self.data_reader,
                enable_plot = False)
        
        total_train_images = 0
        total_val_images = 0

        # Full images
        total_full_voxels_train = 0
        total_full_voxels_val = 0

        # Target voxels
        total_target_voxels_train = 0
        total_target_voxels_val = 0
                    
        # Carrega pontos na voxelização
        voxelizer.set_points(points) 

        t = time.time()
        
        # Groundtruth selecionado
        if selected_idx is None:
            gt_selected = gt_tasks
        else:
            gt_selected = {}
            for task in self.tasks:        
                gt_selected[task] = gt_tasks[task][selected_idx]
            
        # voxel_res = border_size + interior_size + border_size
        interior_size = self.image_size - 2*self.border_size
        # Retorna voxels em coordenadas globais
        # voxels,length = voxelizer.find_voxelization(selected_idx,
        #   res=interior_size*voxelizer.res,
        #   return_global_coord=True,
        #   return_res=True)

        # Retorna voxels de partículas em coordenadas de grid                                                    
        v = voxelizer.find_voxelization(
            points_idx = selected_idx,
            targets = gt_selected,
            res = interior_size*voxelizer.res,
            return_grid_coord = True,
            return_flat = True,
            return_unique = False,
            return_global_coord = True,
            return_points_per_voxel = True,                          
            return_res = True)
                                                        
        save_data_to_debug = False
        if save_data_to_debug:
            mesh_file = os.path.join(dataset_dir,'data_debug','voxels_interior.obj')
            voxelizer.create_voxel_mesh(
                mesh_file = mesh_file,
                voxel_coord = v['voxel_global_coord'],
                voxel_length = v['voxel_res'])

        # Centros
        centers = v['voxel_global_coord'] + 0.5*v['voxel_res']

        if save_data_to_debug:
            centers_file = os.path.join(dataset_dir,'data_view','centers.csv')
            self.array_to_csv(centers_file,centers,columns=['X','Y','Z'])

            # Partículas indefinidas
            undef_particles = points[selected_idx,:]
            particles_per_voxel = os.path.join(dataset_dir,'data_view','particles_per_voxel.csv')
            self.array_to_csv(
                particles_per_voxel,
                np.hstack([undef_particles,v['voxels_grid_coord'][:,np.newaxis]]),
                columns=['X','Y','Z','voxel_index'])

        # Busca de vizinhos
        kdtree = KDTree(points,device='cpu',metric='infinity')
        neighbors_R,_ = kdtree.query_radius(
            centers,int(0.5*self.image_size)*voxelizer.res)
        
        centers_grid_coord = voxelizer.compute_coordinates(centers,voxelizer.origin_coord)
        centers_grid_coord = centers_grid_coord['grid_coord']

        # Geração de imagens por batches
        num_voxels = centers.shape[0]
        num_batches = 1 + num_voxels//batch_size
        for k,batch in enumerate(np.array_split(range(num_voxels),num_batches)):
            if 0 < max_batches and k >= max_batches:
                print(' --> Maximum number of batches reached!')
                break
            print(f' --> Batch {k+1}/{num_batches}')
            t = time.time()
            
            batch_centers_grid_coord = centers_grid_coord[batch]
            batch_full_neighbors = neighbors_R[batch]
            batch_target_neighbors = v['points_idx_per_voxel'][batch]

            batch_target = {}
            for task in self.tasks:
                batch_target[task] = v[f'{task}_per_voxel'][batch]
            
            #batch_selected_labels = selected_labels[batch]

            # Extrai imagens
            batch_data = voxelizer.extract_images_regionwise(
                batch_full_neighbors,
                batch_target_neighbors,
                batch_centers_grid_coord,
                #target_labels=batch_target_labels,
                image_size = self.image_size,
                interior_size = interior_size,
                border_size = self.border_size,
                return_dense = False,
                return_sparse = True,
                return_labels = False)

            batch_full_image_coord = batch_data['full_neighbors_image_coord']
            batch_target_image_coord = batch_data['target_neighbors_image_coord']
            #target_labels = batch_data['target_labels']

            print(f' ----> images generating time: {time.time()-t:.4f} s')
        
            # Separa imagens de treino e teste
            t = time.time()

            train_idx,val_idx = train_test_split(np.arange(batch.shape[0]),test_size=0.25,random_state=0)

            # Full
            batch_full_image_coord_train,batch_full_image_coord_val = batch_full_image_coord[train_idx],batch_full_image_coord[val_idx]

            # Target
            batch_target_image_coord_train,batch_target_image_coord_val = batch_target_image_coord[train_idx],batch_target_image_coord[val_idx]

            batch_target_train = {}
            batch_target_val = {}
            for task in self.tasks:
                batch_target_train[task],batch_target_val[task] = batch_target[task][train_idx],batch_target[task][val_idx]
                
            batch_train_images = batch_full_image_coord_train.shape[0]
            batch_val_images = batch_full_image_coord_val.shape[0]

            total_train_images += batch_train_images
            total_val_images += batch_val_images

            # Full
            num_full_voxels_train = [v.shape[0] for v in batch_full_image_coord_train]
            num_full_voxels_val = [v.shape[0] for v in batch_full_image_coord_val]     

            sum_num_full_voxels_train = sum(num_full_voxels_train)
            sum_num_full_voxels_val = sum(num_full_voxels_val)

            total_full_voxels_train += sum_num_full_voxels_train
            total_full_voxels_val += sum_num_full_voxels_val

            # target
            num_target_voxels_train = [v.shape[0] for v in batch_target_image_coord_train]
            num_target_voxels_val = [v.shape[0] for v in batch_target_image_coord_val]

            sum_num_target_voxels_train = sum(num_target_voxels_train)
            sum_num_target_voxels_val = sum(num_target_voxels_val)

            total_target_voxels_train += sum_num_target_voxels_train
            total_target_voxels_val += sum_num_target_voxels_val

            # Salva dados de treino
            with h5py.File(trainset_file,'a') as f:
                # Attributes
                f.attrs['num_samples'] += batch_train_images

                # Full
                cat_batch_full_image_coord_train = np.concatenate(batch_full_image_coord_train)
                f['full_voxels_coord'].resize((f['full_voxels_coord'].shape[0] + sum_num_full_voxels_train,))
                f['full_voxels_coord'][-cat_batch_full_image_coord_train.shape[0]:] = cat_batch_full_image_coord_train
                
                f['full_num_voxels'].resize((f['full_num_voxels'].shape[0] + batch_train_images,))
                f['full_num_voxels'][-len(num_full_voxels_train):] = num_full_voxels_train
                                
                # Target
                cat_batch_target_image_coord_train = np.concatenate(batch_target_image_coord_train)
                f['target_voxels_coord'].resize((f['target_voxels_coord'].shape[0] + sum_num_target_voxels_train,))
                f['target_voxels_coord'][-cat_batch_target_image_coord_train.shape[0]:] = cat_batch_target_image_coord_train
                
                f['target_num_voxels'].resize((f['target_num_voxels'].shape[0] + batch_train_images,))
                f['target_num_voxels'][-len(num_target_voxels_train):] = num_target_voxels_train

                for task in self.tasks:
                    cat_batch_target_train = np.concatenate(batch_target_train[task])
                    if self.task_types[task] == 'classification':
                        cat_batch_target_train = cat_batch_target_train.astype(int)                    
                        f[f'target_{task}'].resize((f[f'target_{task}'].shape[0] + sum_num_target_voxels_train,))
                        f[f'target_{task}'][-cat_batch_target_train.shape[0]:] = cat_batch_target_train
                    if self.task_types[task] == 'regression':
                        cat_batch_target_train = cat_batch_target_train.astype(float)
                        f[f'target_{task}'].resize((f[f'target_{task}'].shape[0] + sum_num_target_voxels_train,self.task_dimensions[task]))
                        f[f'target_{task}'][-cat_batch_target_train.shape[0]:,:] = cat_batch_target_train
                                    
            # Salva dados validação
            with h5py.File(valset_file,'a') as f:
                # Attributes
                f.attrs['num_samples'] += batch_val_images

                # Full
                cat_batch_full_image_coord_val = np.concatenate(batch_full_image_coord_val)
                f['full_voxels_coord'].resize((f['full_voxels_coord'].shape[0] + sum_num_full_voxels_val,))
                f['full_voxels_coord'][-cat_batch_full_image_coord_val.shape[0]:] = cat_batch_full_image_coord_val
                
                f['full_num_voxels'].resize((f['full_num_voxels'].shape[0] + batch_val_images,))
                f['full_num_voxels'][-len(num_full_voxels_val):] = num_full_voxels_val                    
                
                # Target
                cat_batch_target_image_coord_val = np.concatenate(batch_target_image_coord_val)
                f['target_voxels_coord'].resize((f['target_voxels_coord'].shape[0] + sum_num_target_voxels_val,))
                f['target_voxels_coord'][-cat_batch_target_image_coord_val.shape[0]:] = cat_batch_target_image_coord_val
                
                f['target_num_voxels'].resize((f['target_num_voxels'].shape[0] + batch_val_images,))
                f['target_num_voxels'][-len(num_target_voxels_val):] = num_target_voxels_val
                
                for task in self.tasks:
                    cat_batch_target_val = np.concatenate(batch_target_val[task])
                    if self.task_types[task] == 'classification':
                        cat_batch_target_val = cat_batch_target_val.astype(int)
                        f[f'target_{task}'].resize((f[f'target_{task}'].shape[0] + sum_num_target_voxels_val,))
                        f[f'target_{task}'][-cat_batch_target_val.shape[0]:] = cat_batch_target_val
                    if self.task_types[task] == 'regression':                    
                        cat_batch_target_val = cat_batch_target_val.astype(float)
                        f[f'target_{task}'].resize((f[f'target_{task}'].shape[0] + sum_num_target_voxels_val,self.task_dimensions[task]))
                        f[f'target_{task}'][-cat_batch_target_val.shape[0]:,:] = cat_batch_target_val

            print(f' ----> images storage time: {time.time()-t:.4f} s')
        
        total_time = time.time()-t0
        
        return {'train_images':total_train_images,
            'val_images':total_val_images,'time':total_time}

    def build_dataset_sparse_regionwise(self,points,gt_tasks,selected_idx=None,
        dataset_dir=None,train_name='train',val_name='validation',batch_size=512,
        max_batches=-1,voxelizer=None,save_data_to_debug=False):
        """ 
        Cria o conjunto de treinamento para a abordagem regional esparsa.
        Última modificação: 03/06/2022.
        
        Args:
            points:
            gt_tasks:
            selected:
            dataset_dir:            
            remove_existing:
            train_name:
            val_name:
            batch_size:
            max_batches:
            voxelizer:
            save_data_to_debug:
                  
        Returns:
            times:
        """
        # Sort points by x axis
        #points = points[points[:,0].argsort(),:]
    
        t0 = time.time()

        # Cria o diretório de saída caso não exista
        os.makedirs(dataset_dir,exist_ok=True)
        
        # Nomes dos datasets de treino e validação
        trainset_file = os.path.join(dataset_dir,f'{train_name}.hdf5')
        valset_file = os.path.join(dataset_dir,f'{val_name}.hdf5')

        image_shape = self.spatial_dimensions*[self.image_size]+[2]

        # Cria o dataset de treino caso ele não exista
        if not os.path.exists(trainset_file):
            with h5py.File(trainset_file,'a') as f:
                # Attributes
                f.attrs['num_batches'] = 0
                f.attrs['grid_size'] = voxelizer.size
                f.attrs['target_voxels_max'] = 0
                f.attrs['neighbor_voxels_max'] = 0
                #f.attrs['image_dense_size'] = self.image_size
                #f.attrs['border_size'] = self.border_size
                f.attrs['spatial_dimensions'] = self.spatial_dimensions
                f.attrs['tasks'] = self.tasks
                f.attrs['task_dimensions'] = list(self.task_dimensions.values())
                f.attrs['task_types'] = list(self.task_types.values())
                
                # Dataset arrays
                f.create_dataset(
                    'neighbor_voxels_num',dtype='int32',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'neighbor_voxels_coord',dtype='float32',shape=(0,self.spatial_dimensions),maxshape=(None,self.spatial_dimensions),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'target_voxels_num',dtype='int32',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'target_voxels_coord',dtype='float32',shape=(0,self.spatial_dimensions),maxshape=(None,self.spatial_dimensions),
                    chunks=True,compression="gzip",compression_opts=9)
            
                for task in self.tasks:
                    if self.task_types[task]=='classification':
                        f.create_dataset(
                            f'target_{task}',dtype='float32',shape=(0,self.task_dimensions[task]),maxshape=(None,self.task_dimensions[task]),
                            chunks=True,compression="gzip",compression_opts=9)
                    if self.task_types[task]=='regression':
                        f.create_dataset(
                            f'target_{task}',dtype='float32',shape=(0,self.task_dimensions[task]),
                            maxshape=(None,self.task_dimensions[task]),
                            chunks=True,compression="gzip",compression_opts=9)
                                                
        # Cria o dataset de validação caso ele não exista
        if not os.path.exists(valset_file):
            with h5py.File(valset_file,'a') as f:
                f.attrs['num_batches'] = 0
                f.attrs['grid_size'] = voxelizer.size
                f.attrs['target_voxels_max'] = 0
                f.attrs['neighbor_voxels_max'] = 0                
                #f.attrs['image_dense_size'] = self.image_size
                #f.attrs['border_size'] = self.border_size
                f.attrs['spatial_dimensions'] = self.spatial_dimensions
                f.attrs['tasks'] = self.tasks
                f.attrs['task_dimensions'] = list(self.task_dimensions.values())
                f.attrs['task_types'] = list(self.task_types.values())

                # Dataset arrays            
                f.create_dataset(
                    'neighbor_voxels_num',dtype='int32',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'neighbor_voxels_coord',dtype='float32',shape=(0,self.spatial_dimensions),maxshape=(None,self.spatial_dimensions),
                    chunks=True,compression="gzip",compression_opts=9)            
                f.create_dataset(
                    'target_voxels_num',dtype='int32',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'target_voxels_coord',dtype='float32',shape=(0,self.spatial_dimensions),maxshape=(None,self.spatial_dimensions),
                    chunks=True,compression="gzip",compression_opts=9)            
                
                for task in self.tasks:
                    if self.task_types[task]=='classification':
                        f.create_dataset(
                            f'target_{task}',dtype='float32',shape=(0,self.task_dimensions[task]),maxshape=(None,self.task_dimensions[task]),
                            chunks=True,compression="gzip",compression_opts=9)
                    if self.task_types[task]=='regression':
                        f.create_dataset(
                            f'target_{task}',dtype='f',shape=(0,self.task_dimensions[task]),
                            maxshape=(None,self.task_dimensions[task]),
                            chunks=True,compression="gzip",compression_opts=9)           

        # Voxelizador
        if voxelizer is None:
            voxelizer = SparseVoxelizer(
                limits = self.real_grid_limits,
                res = self.real_grid_length,
                data_reader = self.data_reader,
                enable_plot = False)
        
        total_train_batches = 0
        total_val_batches = 0

        # Full images
        total_full_voxels_train = 0
        total_full_voxels_val = 0

        # Target voxels
        total_target_voxels_train = 0
        total_target_voxels_val = 0
                    
        # Carrega pontos na voxelização
        voxelizer.set_points(points, coord_type=['grid_coord','local_coord']) 

        t = time.time()
        
        # Groundtruth selecionado
        if selected_idx is None:
            gt_selected = gt_tasks
        else:
            gt_selected = {}
            for task in self.tasks:        
                gt_selected[task] = gt_tasks[task][selected_idx]
            
        # voxel_res = border_size + interior_size + border_size
        #interior_size = self.image_size - 2*self.border_size
        # Retorna voxels em coordenadas globais
        # voxels,length = voxelizer.find_voxelization(selected_idx,
        #   res=interior_size*voxelizer.res,
        #   return_global_coord=True,
        #   return_res=True)

        selected_points = points[selected_idx]

        batch_selected = np.zeros(points.shape[0])

        kdtree = KDTree(points,device='cpu')

        #selected_points_neighbors,_,inside_radius = kdtree.query_radius(selected_points,self.real_search_radius)

        #selected_points_neighbors = selected_points_neighbors[:,:inside_radius.numpy().max()]
        selected_points_neighbors,_ = kdtree.query_radius(selected_points,self.real_search_radius)
                    

        # Retorna voxels de partículas em coordenadas de grid
        #flat_index = voxelizer.multi_to_flat_index(voxelizer.grid_coord)

        batch_size = min(int(0.5*selected_points.shape[0]), batch_size)

        batch_selected_idx = self.split_indices_into_batches(selected_idx,batch_size)
        batch_selected_points_neighbors = self.split_indices_into_batches(selected_points_neighbors,batch_size) 
        #batch_inside_radius = self.split_indices_into_batches(inside_radius,batch_size) 
        batch_gt_selected = {}
        for task in self.tasks:
            batch_gt_selected[task] = self.split_indices_into_batches(gt_selected[task],batch_size)
        #batch_size = batch_selected_idx[0].shape[0]

        # Split into train and validation batches
        num_batches = len(batch_selected_idx)
        train_idx,val_idx = train_test_split(np.arange(num_batches),test_size=0.25,random_state=0)

        #del selected_points_neighbors
        #del inside_radius

        # Save batches
        for k,batch in enumerate(batch_selected_idx):
            if 0 < max_batches and k >= max_batches:
                print(' --> Maximum number of batches reached!')
                break
            print(f' --> Batch {k+1}/{num_batches}')
            t = time.time()

            batch_nearest_band = self.nearest_points_band(points,target_neighbors = batch_selected_points_neighbors[k])
            batch_nearest_band = batch_nearest_band.astype(bool)

            batch_target = {}
            for task in self.tasks:
                batch_target[task] = batch_gt_selected[task][k]

            batch_full_grid_coord = voxelizer.grid_coord[batch_nearest_band].astype(np.float32)
            batch_target_grid_coord = voxelizer.grid_coord[batch].astype(np.float32)

            use_local_coord = False
            if use_local_coord:
                batch_full_grid_coord += voxelizer.local_coord[batch_nearest_band].astype(np.float32)
                batch_target_grid_coord += voxelizer.local_coord[batch].astype(np.float32)

            print(f' ----> images generating time: {time.time()-t:.4f} s')
        
            # Separa imagens de treino e teste
            t = time.time()

            num_full_voxels = batch_full_grid_coord.shape[0]
            num_target_voxels = batch_target_grid_coord.shape[0]

            # Salva dados de treino
            if k in train_idx:

                total_train_batches += 1
                total_full_voxels_train += num_full_voxels
                total_target_voxels_train += num_target_voxels

                with h5py.File(trainset_file,'a') as f:
                    # Attributes
                    f.attrs['num_batches'] += 1

                    if f.attrs['target_voxels_max'] < num_target_voxels:
                        f.attrs['target_voxels_max'] = num_target_voxels
                    
                    if f.attrs['neighbor_voxels_max'] < num_full_voxels:
                        f.attrs['neighbor_voxels_max'] = num_full_voxels                    

                    # Full
                    f['neighbor_voxels_coord'].resize((f['neighbor_voxels_coord'].shape[0] + num_full_voxels, self.spatial_dimensions))
                    f['neighbor_voxels_coord'][-num_full_voxels:,:] = batch_full_grid_coord
                    
                    f['neighbor_voxels_num'].resize((f['neighbor_voxels_num'].shape[0] + 1,))
                    f['neighbor_voxels_num'][-1:] = num_full_voxels
                                    
                    # Target
                    f['target_voxels_coord'].resize((f['target_voxels_coord'].shape[0] + num_target_voxels, self.spatial_dimensions))
                    f['target_voxels_coord'][-num_target_voxels:,:] = batch_target_grid_coord
                    
                    f['target_voxels_num'].resize((f['target_voxels_num'].shape[0] + 1,))
                    f['target_voxels_num'][-1:] = num_target_voxels

                    for task in self.tasks:
                        if self.task_types[task] == 'classification':
                            batch_target_task = tf.one_hot(batch_target[task].astype(int), depth = self.task_dimensions[task])
                            f[f'target_{task}'].resize((f[f'target_{task}'].shape[0] + num_target_voxels,self.task_dimensions[task]))
                            f[f'target_{task}'][-num_target_voxels:,:] = batch_target_task
                        if self.task_types[task] == 'regression':
                            batch_target_task = batch_target[task].astype(float)
                            f[f'target_{task}'].resize((f[f'target_{task}'].shape[0] + num_target_voxels, self.task_dimensions[task]))
                            f[f'target_{task}'][-num_target_voxels:,:] = batch_target_task

            # Salva dados de validação
            elif k in val_idx:

                total_val_batches += 1
                total_full_voxels_val += num_full_voxels
                total_target_voxels_val += num_target_voxels
                
                with h5py.File(valset_file,'a') as f:
                    # Attributes
                    f.attrs['num_batches'] += 1

                    if f.attrs['target_voxels_max'] < num_target_voxels:
                        f.attrs['target_voxels_max'] = num_target_voxels
                    
                    if f.attrs['neighbor_voxels_max'] < num_full_voxels:
                        f.attrs['neighbor_voxels_max'] = num_full_voxels

                    # Full
                    f['neighbor_voxels_coord'].resize((f['neighbor_voxels_coord'].shape[0] + num_full_voxels, self.spatial_dimensions))
                    f['neighbor_voxels_coord'][-num_full_voxels:,:] = batch_full_grid_coord
                    
                    f['neighbor_voxels_num'].resize((f['neighbor_voxels_num'].shape[0] + 1,))
                    f['neighbor_voxels_num'][-1:] = num_full_voxels
                                    
                    # Target
                    f['target_voxels_coord'].resize((f['target_voxels_coord'].shape[0] + num_target_voxels, self.spatial_dimensions))
                    f['target_voxels_coord'][-num_target_voxels:,:] = batch_target_grid_coord
                    
                    f['target_voxels_num'].resize((f['target_voxels_num'].shape[0] + 1,))
                    f['target_voxels_num'][-1:] = num_target_voxels

                    for task in self.tasks:
                        if self.task_types[task] == 'classification':
                            batch_target_task = tf.one_hot(batch_target[task].astype(int), depth = self.task_dimensions[task])
                            f[f'target_{task}'].resize((f[f'target_{task}'].shape[0] + num_target_voxels,self.task_dimensions[task]))
                            f[f'target_{task}'][-num_target_voxels:,:] = batch_target_task
                        if self.task_types[task] == 'regression':
                            batch_target_task = batch_target[task].astype(float)
                            f[f'target_{task}'].resize((f[f'target_{task}'].shape[0] + num_target_voxels, self.task_dimensions[task]))
                            f[f'target_{task}'][-num_target_voxels:,:] = batch_target_task

            print(f' ----> batches storage time: {time.time()-t:.4f} s')
        
        total_time = time.time()-t0
        
        return {
            'train_batches':total_train_batches,
            'val_batches':total_val_batches,
            'time':total_time
        }

    def split_indices_into_batches(self,indices,batch_size):
        num_batches = np.ceil(indices.shape[0]/batch_size).astype(int)
        return np.array_split(indices,num_batches)

    def predict_pointwise(self,points,model,voxelizer=None,coarse_threshold=0.7,
        batch_size=32,device='cpu',return_times=False):
        """ 
        Classifica partículas com a rede 2d treinada na abordagem 1.
        Última modificação: 31/05/2022.
        
        Args:
            points:
            model:
            voxelization:
            coarse_threshold:
            batch_size:
            device:
            return_times:
                  
        Returns:
            predictions,times.
        """
        # Voxelizador
        if voxelizer is not None:
            voxelizer = SparseVoxelizer(
                limits = self.real_grid_limits,
                res = self.real_grid_length,
                data_reader = self.data_reader,
                enable_plot = False)

        times = {'coarse_prediction':0,'images_generation':0,
            'images_classification':0,'total':0}

        t0 = time.time() # contador para o tempo total
        t1 = time.time() # Contador para o teste grosseiro e voxelização    
                    
        #kdtree = KDTree(points)  # Cria Kdtree
        voxelizer.set_points(points) # Carrega partículas na voxelização
        
        if device in ['cpu','cpu_gpu']:
            coarse_prediction_cpu = self.coarse_prediction_cpu(
                points = points,
                combined_threshold = coarse_threshold,
                test_type = 'combined_product',
                return_neighbors = True,
                return_measures = False )
            selected = coarse_prediction_cpu['pred_combined_product']
            neighbors = coarse_prediction_cpu['neighbors']

        if device in ['gpu','cpu_gpu']:
            coarse_prediction_gpu = self.coarse_prediction_gpu(
                points = points,
                combined_threshold = coarse_threshold,
                test_type = 'combined_product',
                return_neighbors = True,
                return_measures = False)
            selected = coarse_prediction_gpu['pred_combined_product'].numpy()
            neighbors = coarse_prediction_gpu['neighbors'].numpy()
            count = coarse_prediction_gpu['count_neighbors'].numpy()
        
        if device == 'cpu_gpu':
            self.check_coarse_predictions_cpu_gpu(coarse_prediction_cpu,coarse_prediction_gpu)
            return
        
        selected_idx = np.where(selected==1)[0]

        times['coarse_prediction'] = time.time() - t1
        removed = 100*(selected==0).sum()/points.shape[0]
        print(f" --> coarse prediction... ({removed:.2f}% removed): {times['coarse_prediction']:.4f} s")

        # Criação das imagens de partículas selecionadas                     
        num_selected = selected_idx.shape[0]
        num_batches = 1 + num_selected//batch_size

        predictions = {}

        if 'boundary' in self.tasks:
            predictions['labels'] = np.zeros(points.shape[0],dtype=int)
        if 'normal' in self.tasks:
            predictions['normal'] = np.zeros(points.shape,dtype=float)

        for k,batch in enumerate(np.array_split(range(num_selected),num_batches)):
            print(f' --> batch {k}/{num_batches}')

            t2 = time.time()
            batch_selected_idx = selected_idx[batch]
            if device=='cpu':
                batch_neighbors = neighbors[batch]
            elif device=='gpu':
                batch_neighbors = (neighbors[batch],count[batch])
            
            # Extrai imagens
            batch_selected_images = voxelizer.extract_images_pointwise(
                batch_selected_idx,
                self.image_size,
                neighbors = batch_neighbors,
                channels=self.features)

            batch_images_gen_time = time.time() - t2
            times['images_generation'] += batch_images_gen_time
            print(f' ----> images generation time: {batch_images_gen_time:.4f} s')
                            
            # Classifica batch de imagens
            t3 = time.time()
            batch_pred = model.predict(
                dataset = batch_selected_images,
                batch_size = batch_selected_images.shape[0],
                device = device,
                verbose = 0)
        
            if 'boundary' in self.tasks:
                predictions['labels'][batch_selected_idx] = batch_pred['labels']
            if 'normal' in self.tasks:
                predictions['normal'][batch_selected_idx] = batch_pred['normal']

            batch_images_classif_time = time.time() - t3
            times['images_classification'] += batch_images_classif_time
            print(f' ----> imagens classification time: {batch_images_classif_time:.4f} s\n')

            times['total'] = time.time() - t0

        if return_times:
            return predictions,times
        else:
            return predictions

    def predict_regionwise(self,points,model,voxelizer=None,coarse_threshold=0.7,
        batch_size=32,device='cpu',debug=False,return_times=False):
        """
        Classifica pontos com o modelo neural passado como argumento.
        Última modificação: 27/05/2022.
        
        Args:
            points:
            model:
            voxelization:
            coarse_threshold:
            batch_size:
            device:      
            debug:
            return_times:      
                  
        Returns:
            prediction,times.
        """
        # Voxelizador
        if voxelizer is None:
            voxelizer = SparseVoxelizer(
                limits = self.real_grid_limits,
                res = self.real_grid_length,
                data_reader = self.data_reader,
                enable_plot = debug)
        
        times = {'coarse_prediction':0,'images_generation_search':0,
            'images_generation_build':0,'images_classification':0}

        t0 = time.time() # contador para o tempo total
        t1 = time.time() # Contador para o teste grosseiro e voxelização

        # Voxeliza os pontos 
        voxelizer.set_points(points,
            coord_type=['grid_coord','local_coord'])
        
        # Handle device aliases and fallback
        if device == 'cuda':
            device = 'gpu'
        
        if device not in ['cpu', 'gpu', 'cpu_gpu']:
            print(f"Warning: Unknown device '{device}', using 'cpu' as fallback")
            device = 'cpu'
        
        # GPU methods require TF KDTree - fallback to CPU if not available
        if not USE_TF_KDTREE and device in ['gpu', 'cpu_gpu']:
            print(f"Warning: GPU device requires TF KDTree which is not available. Using 'cpu' instead.")
            device = 'cpu'

        if device in ['cpu','cpu_gpu']:
            coarse_prediction_cpu = self.coarse_prediction_cpu(
                points = points,
                combined_threshold = coarse_threshold,
                test_type = 'combined_product',
                return_neighbors = True,
                return_measures = True)
            selected = coarse_prediction_cpu['pred_combined_product']

        if device in ['gpu','cpu_gpu']:
            coarse_prediction_gpu = self.coarse_prediction_gpu(
                points = points,
                combined_threshold = coarse_threshold,
                test_type = 'combined_product',
                return_neighbors = False,
                return_measures = False,
                batch_size=500000)
            selected = coarse_prediction_gpu['pred_combined_product'].numpy()
                    
        if device == 'cpu_gpu':
            self.check_coarse_predictions_cpu_gpu(coarse_prediction_cpu,coarse_prediction_gpu)
            return
                
        selected_idx = np.where(selected==1)[0]

        ressample = False
        if ressample:
            ct_points = points[selected_idx]
            n = 8
            angles = np.linspace(0,2*np.pi,n+1)[:-1]
            #shift = angles[1]*np.random.random((ct_points.shape[0],1))
            shift = np.zeros((ct_points.shape[0],1))
            angles = (angles + shift).flatten()
            
            news_points = 0.5*self.ref_length*np.array([np.cos(angles),np.sin(angles)]).T
            #news_points = self.ref_length*np.array([np.cos(angles),np.sin(angles)]).T
            
            ressample_points = np.repeat(ct_points,n,axis=0) + news_points

            kdtree = KDTree(points,device='gpu',metric='euclidean')
            _,dists = kdtree.query(ressample_points,knn=1)
            good = tf.abs(dists-0.5*self.ref_length)<tf.keras.backend.epsilon()
            good = good.numpy().reshape(-1)

            good_ressample_points = ressample_points[good]

            plt.scatter(points[:,0],points[:,1])
            plt.scatter(ct_points[:,0],ct_points[:,1])
            #plt.scatter(ressample_points[:,0],ressample_points[:,1])
            plt.scatter(good_ressample_points[:,0],good_ressample_points[:,1])
            _ = plt.axis('equal')
            plt.show()

        times['coarse_prediction'] = time.time() - t1
        removed = 100*(selected==0).sum()/points.shape[0]
        print(f" --> coarse prediction... ({removed:.2f}% removed): {times['coarse_prediction']:.4f} s")

        # Criação das imagens de partículas selectionadas
        t2_0 = time.time()

        # voxel_res = border_size + interior_res + border_size
        interior_size = self.image_size - 2*self.border_size
        # Retorna voxels em coordenadas globais
        # voxels,length = voxelizer.find_voxelization(
        #     selected_idx,
        #     res=interior_res*voxelizer.res,
        #     return_global_coord=True,
        #     return_res=True
        # )

        # Retorna voxels de partículas em coordenadas de grid
        v = voxelizer.find_voxelization(
            selected_idx,
            #labels=selected_labels,
            res = interior_size*voxelizer.res,
            return_grid_coord = True,
            return_flat = True,
            return_unique = False,
            return_global_coord = True,
            return_points_per_voxel = True,
            return_res = True,
            debug = debug)
                            
        save_data_view = False
        if save_data_view and self.spatial_dimensions==3:
            mesh_file = os.path.join(self.data_dir,'data_view','voxels_interior.obj')
            voxelizer.create_voxel_mesh(
                mesh_file = mesh_file,
                voxel_coord = v['voxel_global_coord'],
                voxel_length = v['voxel_res'])

        # Centros das células da cobertura de voxels
        centers = v['voxel_global_coord'] + 0.5*v['voxel_res']

        if save_data_view and self.spatial_dimensions==3:
            centers_file = os.path.join(self.data_dir,'data_view','centers.csv')
            self.array_to_csv(centers_file,centers,columns=['X','Y','Z'])

            # Partículas indefinidas
            undef_points = points[selected_idx,:]
            points_per_voxel = os.path.join(
                self.data_dir,'data_view','points_per_voxel.csv')
            self.array_to_csv(
                points_per_voxel,
                np.hstack([undef_points,v['voxels_grid_coord'][:,np.newaxis]]),
                columns=['X','Y','Z','voxel_index'])

        # Segunda busca de vizinhos
        if device=='cpu':
            if USE_TF_KDTREE:
                kdtree2 = KDTree(points,device='cpu',metric='infinity')
                neighbors_R,dists_R = kdtree2.query_radius(
                    centers,radius=int(0.5*self.image_size)*voxelizer.res)
            else:
                # Use scipy cKDTree with Chebyshev metric (infinity norm)
                kdtree2 = cKDTree(points)
                neighbors_R = kdtree2.query_ball_point(
                    centers, int(0.5*self.image_size)*voxelizer.res, p=np.inf)
                # Calculate distances for each neighbor
                dists_R = []
                for i, neighs in enumerate(neighbors_R):
                    if len(neighs) > 0:
                        dists = np.abs(points[neighs] - centers[i]).max(axis=1)
                        dists_R.append(dists)
                    else:
                        dists_R.append(np.array([]))
                # Convert to numpy object arrays for slicing
                neighbors_R = np.array([np.array(n) for n in neighbors_R], dtype=object)
                dists_R = np.array(dists_R, dtype=object)
        elif device=='gpu':
            if self.spatial_dimensions == 2:
                #max_knn =  4*(self.search_radius*self.hdp_suggested)**2
                max_knn =  4*(self.search_radius*self.hdp)**2
            elif self.spatial_dimensions == 3:
                #max_knn = 8*(self.search_radius*self.hdp_suggested)**3
                max_knn = 8*(self.search_radius*self.hdp)**3
            max_knn = int(max_knn*1.25)
            with tf.device('gpu'):
                kdtree2 = KDTree(points,device='gpu',metric='infinity')
                neighbors_R,dists_R,count_R = kdtree2.query_radius(centers,
                    radius = int(0.5*self.image_size)*voxelizer.res,
                    max_knn = max_knn,
                    verbose = True)
            
            neighbors_R = neighbors_R.numpy()
            count_R = count_R.numpy()

        #cuda.get_current_device().reset()

        # Caso only_coarse_test_as_target seja True então apenas os pontos
        # remanescentes do teste grosseiro serão classificadas em cada imagem
        only_coarse_test_as_target = True
        if only_coarse_test_as_target==False:
            target_neighbors = np.empty((neighbors_R.shape[0]),dtype=np.object)
            for i in range(neighbors_R.shape[0]):
                target_neighbors[i] = neighbors_R[i][dists_R[i] < 0.5*interior_size*voxelizer.res]

        centers_grid_coord = voxelizer.compute_coordinates(
            centers,voxelizer.origin_coord)
        centers_grid_coord = centers_grid_coord['grid_coord']

        # Geração de imagens por batches 
        #batch_size = 1000
        num_images = centers.shape[0]
        num_batches = 1 + num_images//batch_size

        predictions = {}

        for task in self.tasks:
            if self.task_types[task] == 'classification':
                predictions[task] = np.zeros((points.shape[0],self.task_dimensions[task]),dtype=float)
                predictions[task][:,0] = 1
            if self.task_types[task] == 'regression':
                predictions[task] = np.zeros((points.shape[0],self.task_dimensions[task]),dtype=float)

        times['images_generation_search'] = time.time() - t2_0
        print(f" --> images generation time (search): {times['images_generation_search']:.4f} s")

        for k,batch in enumerate(np.array_split(range(num_images),num_batches)):
            print(f' --> batch {k}/{num_batches}')
            t2_1 = time.time()
            if device=='cpu':
                batch_full_neighbors = neighbors_R[batch]
            elif device=='gpu':
                batch_full_neighbors = (neighbors_R[batch],count_R[batch])

            if only_coarse_test_as_target:
                batch_target_neighbors = v['points_idx_per_voxel'][batch]
            else:
                batch_target_neighbors = target_neighbors[batch]

            #batch_target_labels = v['labels_per_voxel'][batch]
            batch_centers_grid_coord = centers_grid_coord[batch]

            if save_data_view and self.spatial_dimensions:
                v_idx = 1
                if device=='cpu':
                    p_idx = batch_full_neighbors[v_idx]
                elif device=='cpu':
                    p_idx = batch_full_neighbors[0][v_idx]
                p_idx_target = batch_target_neighbors[v_idx]
                # full_labels = gt_labels[p_idx].reshape(-1,1)
                # target_labels = gt_labels[p_idx_target].reshape(-1,1)
                # Wrong --> target_labels = batch_target_labels[v_idx].reshape(-1,1) 
                
                # points_file = os.path.join(data_dir,'data_view','points_in_voxel.csv')
                # self.array_to_csv(
                #    points_file,
                #   np.hstack([points[p_idx,:],full_labels]),
                #   columns=['X','Y','Z','labels'])

                # points_target_file = os.path.join(data_dir,'data_view','points_target_in_voxel.csv')
                # self.array_to_csv(
                # points_target_file,
                #     np.hstack([points[p_idx_target,:],target_labels]),
                #     columns=['X','Y','Z','labels'])

                interior_voxel_mesh = os.path.join(self.data_dir,'data_view','interior_voxel.obj')
                voxelizer.create_voxel_mesh(
                    mesh_file=interior_voxel_mesh,
                    voxel_coord=v['voxel_global_coord'][batch][v_idx],
                    voxel_length=v['voxel_res'])
                
                full_voxel_mesh = os.path.join(self.data_dir,'data_view','full_voxel.obj')
                full_voxel_coord = v['voxel_global_coord'][batch][v_idx]-voxelizer.res*self.border_size
                voxelizer.create_voxel_mesh(
                    mesh_file=full_voxel_mesh,
                    voxel_coord=full_voxel_coord,
                    voxel_length=self.image_size*voxelizer.res)

                img3d_mesh = os.path.join(self.data_dir,'data_view','image3d.obj')
                #img3d_grid_coord = voxelizer.grid_coord[p_idx]
                #img3d_global_coord = voxelizer.voxel_coord_to_global_coord(img3d_grid_coord)
                voxelizer.create_voxel_mesh(
                    mesh_file=img3d_mesh,
                    points_idx=p_idx,
                    voxel_length=voxelizer.res)

            #batch_selected_labels = selected_labels[batch]

            # Extrai imagens 3D
            batch_data = voxelizer.extract_images_regionwise(
                batch_full_neighbors,
                batch_target_neighbors,
                batch_centers_grid_coord,
                #target_labels=batch_target_labels,
                image_size = self.image_size,
                interior_size = interior_size,
                border_size = self.border_size,
                return_dense = True,
                #return_labels=True,
                return_sparse = True)
    
            batch_images_gen_time = time.time() - t2_1
            times['images_generation_build'] += batch_images_gen_time
            print(f" ----> images generation time (build): {batch_images_gen_time:.4f} s")

            # Classifica batch de imagens
            #batch_data['images'][...,1] = 1 
            t3 = time.time()
            batch_pred = model.predict(
                dataset = batch_data['images'],
                batch_size = batch_data['images'].shape[0],
                device = device,
                verbose = 0)
            #print('  ----> gpu pred time: ',time.time()-t3)
                        
            t4 = time.time()
            # Extrai as predições de cada tarefa usando numpy
            #prediction_interp = False
            for i,target_particle_idx in enumerate(batch_target_neighbors):
                target_image_coord_flat = batch_data['target_neighbors_image_coord'][i]

                for task in self.tasks:
                    # if task=='boundary' and prediction_interp:
                    #     target_image_coord = np.array(
                    #         np.unravel_index(target_image_coord_flat,
                    #         self.spatial_dimensions*(self.image_size,))).T

                    #     target_image_coord = target_image_coord + voxelizer.local_coord[target_particle_idx]
                    #     target_image_coord = target_image_coord[np.newaxis,:].astype(np.float32)
                        
                    #     if self.spatial_dimensions==3:
                    #         pred = trilinear_interpolate(batch_pred[task][i:i+1],target_image_coord)[0]

                    # else:
                    #     pred = batch_pred[task][i].reshape(-1,self.task_dimensions[task])[target_image_coord_flat,:]

                    pred = batch_pred[task][i].reshape(-1,self.task_dimensions[task])[target_image_coord_flat,:]
                    predictions[task][target_particle_idx,:] = pred
            #print('   ----> extração de labels 1: ',time.time()-t4)

            # t41 = time.time()
            # # Extrai as predições de cada tarefa usando tensorflow
            # for i,target_particle_idx in enumerate(batch_target_neighbors):
            #     target_image_coord = tf.convert_to_tensor(batch_data['target_neighbors_image_coord'][i])
            #     for task in self.tasks:                    
            #         batch_pred_reshaped = tf.reshape(batch_pred[task][i],(-1,self.task_dimensions[task]))
            #         predictions2[task][target_particle_idx,:] = tf.gather(batch_pred_reshaped,target_image_coord)
            # print('   ----> extração de labels 2: ',time.time()-t4)

            batch_img_classif_time = time.time() - t3
            times['images_classification'] += batch_img_classif_time
            print(f' ----> images classification time: {batch_img_classif_time:.4f} s\n')

            if self.enable_plot and self.spatial_dimensions==2:
                for i in range(0,batch_pred.shape[0],50):
                    image = np.zeros((self.image_size,self.image_size,3))
                    image[...,0] = batch_data['images'][i,...,0]
                    image[...,1:3] = batch_pred['labels'][i]
                    plt.imshow(image)
                    plt.pause(1)
                    #plt.show()

            times['total'] = time.time() - t0

        if return_times:
            return predictions,times
        else:
            return predictions

    def predict_sparse_regionwise(self,points,model,voxelizer=None,coarse_threshold=0.7,
        batch_size=32,device='cpu',debug=False,return_times=False):
        """
        Classifica pontos com o modelo neural esparso passado como argumento.
        
        Args:
            points:
            model:
            voxelization:
            coarse_threshold:
            batch_size:
            device:      
            debug:
            return_times:      
                  
        Returns:
            prediction,times.
        """
        # Sort points by x axis
        sorted_idx = points[:,0].argsort()
        points = points[sorted_idx,:]
        
        # Voxelizador
        if voxelizer is None:
            voxelizer = SparseVoxelizer(
                limits = self.real_grid_limits,
                res = self.real_grid_length,
                data_reader = self.data_reader,
                enable_plot = debug)
        
        times = {'coarse_prediction':0,'batch_generation_time':0,
                 'batch_prediction_time':0, 'total_time':0}

        t0 = time.time() # contador para o tempo total
        t1 = time.time() # Contador para o teste grosseiro e voxelização

        # Voxeliza os pontos 
        voxelizer.set_points(points,coord_type=['grid_coord','local_coord'])        

        if self.coarse_prediction == None:
            selected = np.ones(points.shape[0],np.bool)
            selected_idx = np.arange(points.shape[0])
        else:
            if device in ['cpu','cpu_gpu']:
                coarse_prediction_cpu = self.coarse_prediction_cpu(
                    points = points,
                    combined_threshold = coarse_threshold,
                    test_type = 'combined_product',
                    return_neighbors = True,
                    return_measures = False,
                    #kdtree = kdtree
                    )
                selected = coarse_prediction_cpu['pred_combined_product']
                selected_points_neighbors = coarse_prediction_cpu['neighbors']

            if device in ['gpu','cpu_gpu']:
                
                coarse_prediction_gpu = self.coarse_prediction_gpu(
                    points = points,
                    combined_threshold = coarse_threshold,
                    test_type = 'combined_product',
                    return_neighbors = True,
                    return_measures = False,
                    batch_size = 100000,
                    #kdtree = kdtree
                )
                """
                import multiprocessing
                multiprocessing.set_start_method('spawn')
                
                p = multiprocessing.Pool()
                coarse_prediction_gpu = p.apply(self.coarse_prediction_gpu, 
                    args=(points,0.8,0.1,coarse_threshold,'combined_product',True,False,False,100000))
                p.close()
                p.join()
                """
                selected = coarse_prediction_gpu['pred_combined_product'].numpy()
                selected_points_neighbors = coarse_prediction_gpu['neighbors']        
                                                    
            if device == 'cpu_gpu':
                self.check_coarse_predictions_cpu_gpu(coarse_prediction_cpu,coarse_prediction_gpu)
                return            
                
            selected_idx = np.where(selected==1)[0]

            times['coarse_prediction'] = time.time() - t1
            removed = 100*(selected==0).sum()/points.shape[0]
            print(f" --> coarse prediction... ({removed:.2f}% removed): {times['coarse_prediction']:.4f} s")


        #np.savetxt('points_selected.txt',points[selected_idx],fmt='%.6f %6f %.6f',header='x y z',comments='')

        ressample = False
        if ressample:
            ct_points = points[selected_idx]
            n = 8
            angles = np.linspace(0,2*np.pi,n+1)[:-1]
            #shift = angles[1]*np.random.random((ct_points.shape[0],1))
            shift = np.zeros((ct_points.shape[0],1))
            angles = (angles + shift).flatten()
            
            news_points = 0.5*self.ref_length*np.array([np.cos(angles),np.sin(angles)]).T
            #news_points = self.ref_length*np.array([np.cos(angles),np.sin(angles)]).T
            
            ressample_points = np.repeat(ct_points,n,axis=0) + news_points

            kdtree = KDTree(points,device='gpu',metric='euclidean')
            _,dists = kdtree.query(ressample_points,knn=1)
            good = tf.abs(dists-0.5*self.ref_length)<tf.keras.backend.epsilon()
            good = good.numpy().reshape(-1)

            good_ressample_points = ressample_points[good]

            plt.scatter(points[:,0],points[:,1])
            plt.scatter(ct_points[:,0],ct_points[:,1])
            #plt.scatter(ressample_points[:,0],ressample_points[:,1])
            plt.scatter(good_ressample_points[:,0],good_ressample_points[:,1])
            _ = plt.axis('equal')
            plt.show()

        # Fine prediction
        batch_indices = self.split_indices_into_batches(np.arange(selected_idx.shape[0]),batch_size)
        
        #batch_selected_idx = self.split_indices_into_batches(selected_idx,batch_size)
        #batch_selected_points_neighbors = self.split_indices_into_batches(selected_points_neighbors,batch_size) 

        # Split into train and validation batches
        num_batches = len(batch_indices)
        # train_idx,val_idx = train_test_split(np.arange(num_batches),test_size=0.25,random_state=0)
        
        # Predict point batches
        selected_predictions = {}
        for task in self.tasks:
            selected_predictions[task] = tf.Variable(tf.zeros((selected_idx.shape[0], self.tasks[task]['outputs']),tf.float32))

        start_idx, end_idx = 0, 0
        for k,batch in enumerate(batch_indices):
            print(f' --> Batch {k+1}/{num_batches}')
            
            start_idx = batch[0]
            end_idx = batch[-1] + 1
            
            batch_selected_idx = selected_idx[start_idx:end_idx]

            if self.coarse_prediction == None:
                min_xcoord, max_xcoord = points[batch_selected_idx[0],0] , points[batch_selected_idx[-1],0]
                batch_nearest_band = np.logical_and(points[:,0] >= min_xcoord-self.real_search_radius,  points[:,0] <= max_xcoord+self.real_search_radius)
            else:
                batch_selected_points_neighbors = selected_points_neighbors[start_idx:end_idx]
                batch_nearest_band = self.nearest_points_band(points,target_neighbors = batch_selected_points_neighbors)
                batch_nearest_band = batch_nearest_band.astype(bool)
            
            t = time.time()

            use_local_coord = False
            if use_local_coord:
                target_voxels_coord = tf.convert_to_tensor(
                    voxelizer.grid_coord[batch_selected_idx] + voxelizer.local_coord[batch_selected_idx], tf.float32
                )
                neighbor_voxels_coord = tf.convert_to_tensor(
                    voxelizer.grid_coord[batch_nearest_band] + voxelizer.local_coord[batch_nearest_band], tf.float32
                )
            else:
                target_voxels_coord = tf.convert_to_tensor(voxelizer.grid_coord[batch_selected_idx], tf.float32)
                neighbor_voxels_coord = tf.convert_to_tensor(voxelizer.grid_coord[batch_nearest_band], tf.float32)

            batch_features = tf.ones((neighbor_voxels_coord.shape[0],1), tf.float32)

            batch_generation_time = time.time()-t
            times['batch_generation_time'] += batch_generation_time
            print(f' ----> Batch generation time: {batch_generation_time:.4f} s')
    
            t2 = time.time()
            batch_pred = model.predict([batch_features, neighbor_voxels_coord, target_voxels_coord])
            for task in self.tasks:
                selected_predictions[task][start_idx:end_idx,:].assign(batch_pred[task])
            batch_prediction_time = time.time() - t2

            times['batch_prediction_time'] += batch_prediction_time
            print(f' ----> Batch predictions time: {batch_prediction_time:.4f} s\n')

            tf.keras.backend.clear_session()
            gc.collect()

        times['total'] = time.time() - t0
        times['fine_prediction'] = times['total'] - times['coarse_prediction']
        
        print(f"\n ----> Coarse prediction time: {times['coarse_prediction']:.4f} s")
        print(f" ----> Fine prediction time: {times['fine_prediction']:.4f} s")
        print(f" ----> Total time: {times['total']:.4f} s\n")

        predictions = {}
        for task in self.tasks:
            predictions[task] = np.zeros((points.shape[0],self.tasks[task]['outputs']))
            predictions[task][selected==1,:] = selected_predictions[task].numpy()
            if task=='boundary':
                predictions[task][selected==0,:] = [1,0]
            if task=='normal':
                predictions[task][selected==0,:] = [0,0,0]
        
        # Restore default indexing
        for task in self.tasks:
            predictions[task] = predictions[task][sorted_idx.argsort()]

        if return_times:
            return predictions, times
        else:
            return predictions


    def coarse_prediction_cpu(self,points,density_threshold=0.8,
        centroid_threshold=0.1,combined_threshold=0.7,test_type='combined_logical',
        return_neighbors=False,return_distances=False,return_measures=False,kdtree=None):
        """
        Predição grosseira com combinação dos testes de contagem e centroidde.
        Última modificação: 15/03/2022.
        
        Args:
            points:         
            density_threshold:
            centroid_threshold:
            combined_threshold:
            test_type: 'density','centroid','combined_logical', 'combined_product' ou 'all'
            return_neighbors:
            return_distances:
            return_measures: True or False
            kdtree:
        
        Returns:
            depende dos argumentos de entrada.
        """        
        if kdtree is None:
            if USE_TF_KDTREE:
                kdtree = KDTree(points,device='cpu',metric='euclidean')
            else:
                kdtree = cKDTree(points)

        # Busca partículas vizinhas em um determinado raio
        if USE_TF_KDTREE:
            neighbors,_ = kdtree.query_radius(points,radius=self.real_search_radius)
        else:
            # scipy cKDTree uses query_ball_point for radius queries
            neighbors = kdtree.query_ball_point(points, self.real_search_radius)
            neighbors = [np.array(n) for n in neighbors]

        count = np.zeros(points.shape[0],dtype=int)
        if test_type in ['density','combined_logical','combined_product','all']:
            density = np.zeros(points.shape[0])
        
        if test_type in ['centroid','combined_logical','combined_product','all']:
            centroid_distance = np.ones(points.shape[0]) 
        
        for i in range(points.shape[0]):
            count[i] = neighbors[i].shape[0]

            if test_type in ['density','combined_logical','combined_product','all']:   
                if self.spatial_dimensions == 2:
                    # Calcula densidade pela área do disco de raio "search_radius * hdp_sim"
                    #density[i] = float(count[i])/(np.pi * (self.search_radius * self.hdp_suggested)**2)
                    density[i] = float(count[i])/(np.pi * (self.search_radius * self.hdp)**2)
                elif self.spatial_dimensions == 3:
                    # Calcula a densidade pelo volume da esfera de raio "search_radius * hdp_sim"
                    #density[i] = float(count[i])/((4/3) * np.pi * (self.search_radius * self.hdp_suggested)**3)
                    density[i] = float(count[i])/((4/3) * np.pi * (self.search_radius * self.hdp)**3)

            if test_type in ['centroid','combined_logical','combined_product','all']:             
                centroid = points[neighbors[i][0:count[i]],:].mean(axis=0)
                centroid_distance[i] = np.linalg.norm(points[i]-centroid) / self.real_search_radius

        # Predição com o teste da densidade
        if test_type in ['density','combined_logical','all']:
            pred_density = (density < density_threshold).astype(int)
    
        # Predição com o teste do centroide
        if test_type in ['centroid','combined_logical','all']:
            pred_centroid = (centroid_distance > centroid_threshold).astype(int)
        
        # Predição de lógica combinada
        if test_type in ['combined_logical','all']:
            pred_combined_logical = np.logical_or(pred_density,pred_centroid).astype(int)

        # Predição de produto combinado
        if test_type in ['combined_product','all']:
            combined_measure = density * (1 - centroid_distance)
            pred_combined_product = (combined_measure < combined_threshold).astype(int)

        # Saídas
        output = {}

        if test_type == 'density':
            output['pred_density'] = pred_density
            if return_measures:
                output['density'] = density
            if return_neighbors:                
                if USE_TF_KDTREE:
                    output['neighbors'] = neighbors[pred_density.astype(bool)]
                    output['count_neighbors'] = count[pred_density.astype(bool)]
                else:
                    selected_mask = pred_density.astype(bool)
                    output['neighbors'] = [neighbors[i] for i in range(len(neighbors)) if selected_mask[i]]
                    output['count_neighbors'] = count[selected_mask]

        elif test_type == 'centroid':
            output['pred_centroid'] = pred_centroid
            if return_measures:
                output['centroid_distances'] = centroid_distance
            if return_neighbors:
                if USE_TF_KDTREE:
                    output['neighbors'] = neighbors[pred_centroid.astype(bool)]
                    output['count_neighbors'] = count[pred_centroid.astype(bool)]
                else:
                    selected_mask = pred_centroid.astype(bool)
                    output['neighbors'] = [neighbors[i] for i in range(len(neighbors)) if selected_mask[i]]
                    output['count_neighbors'] = count[selected_mask]
            
        elif test_type == 'combined_logical':
            output['pred_combined_logical'] = pred_combined_logical
            if return_measures:
                output['density'] = density
                output['centroid_distances'] = centroid_distance
            if return_neighbors:
                if USE_TF_KDTREE:
                    output['neighbors'] = neighbors[pred_combined_logical.astype(bool)]
                    output['count_neighbors'] = count[pred_combined_logical.astype(bool)]
                else:
                    selected_mask = pred_combined_logical.astype(bool)
                    output['neighbors'] = [neighbors[i] for i in range(len(neighbors)) if selected_mask[i]]
                    output['count_neighbors'] = count[selected_mask]
            
        elif test_type == 'combined_product':
            output['pred_combined_product'] = pred_combined_product
            if return_measures:
                output['combined_measures'] = combined_measure
                output['density'] = density
                output['centroid_distances'] = centroid_distance
            if return_neighbors:
                # Handle different neighbor formats (scipy list vs tf array)
                if USE_TF_KDTREE:
                    output['neighbors'] = neighbors[pred_combined_product.astype(bool)]
                    output['count_neighbors'] = count[pred_combined_product.astype(bool)]
                else:
                    selected_mask = pred_combined_product.astype(bool)
                    output['neighbors'] = [neighbors[i] for i in range(len(neighbors)) if selected_mask[i]]
                    output['count_neighbors'] = count[selected_mask]

        elif test_type == 'all':
            output['pred_combined_product'] = pred_combined_product
            output['pred_combined_logical'] = pred_combined_logical
            output['pred_density'] = pred_density
            output['pred_centroid'] = pred_centroid
            if return_measures:
                output['combined_measures'] = combined_measure
                output['density'] = density
                output['centroid_distances'] = centroid_distance
            if return_neighbors:
                output['all_neighbors'] = neighbors
                output['all_count_neighbors'] = count
        
        return output

    def coarse_prediction_gpu(self,points,density_threshold=0.8,
        centroid_threshold=0.1,combined_threshold=0.7,test_type='combined_product',
        return_neighbors=False,return_distances=False,return_measures=False,batch_size=100000,
        kdtree=None):
        """
        Predição grosseira com combinação dos testes de contagem e centroidde.
        Última modificação: 06/06/2022.
        
        Args:
            points:     
            density_threshold:
            centroid_threshold:
            combined_threshold:
            test_type: 'density','centroid','combined_logical', 'combined_product' ou 'all'
            return_neighbors:
            return_distances:
            return_measures: True or False
            kdtree:
        
        Returns:
            depende dos argumentos de entrada.
        """

        import tensorflow as tf

        if self.spatial_dimensions == 2:
            #max_knn = (np.pi * (self.search_radius * self.hdp_suggested)**2)
            max_knn = (np.pi * (self.search_radius * self.hdp)**2)
        elif self.spatial_dimensions == 3:
            #max_knn = ((4/3) * np.pi * (self.search_radius * self.hdp_suggested)**3)
            max_knn = ((4/3) * np.pi * (self.search_radius * self.hdp)**3)
        max_knn = int(max_knn*1.25)

        if batch_size==-1:
            batch_size = points.shape[0]
        num_batches = np.ceil(points.shape[0]/batch_size).astype(int)

        density = np.zeros(points.shape[0])
        centroid_distance = np.zeros(points.shape[0])
        all_count = np.zeros(points.shape[0],np.int32)
        all_neighbors = np.zeros((points.shape[0],max_knn),np.int32)

        with tf.device('gpu'):
            points = tf.convert_to_tensor(points,dtype=tf.float32)
            #points = tf.convert_to_tensor(points,dtype=tf.float16)

            #density = tf.Variable(tf.zeros(points.shape[0]))
            #centroid_distance = tf.Variable(tf.zeros(points.shape[0]))
            #all_count = tf.Variable(tf.zeros(points.shape[0],tf.int32))
            #all_neighbors = tf.Variable(tf.zeros((points.shape[0],max_knn),tf.int32))
            #all_neighbors = np.zeros((points.shape[0],max_knn),np.int32)

            if kdtree is None:
                kdtree = KDTree(points,device='gpu',metric='euclidean')

            for k in tf.range(num_batches):                
                begin_slice,end_slice = k*batch_size,(k+1)*batch_size
                points_batch = points[begin_slice:end_slice]
                
                neighbors,dists,count = kdtree.query_radius(
                    points_batch,radius=self.real_search_radius,max_knn=max_knn)

                # Calcula as densidades
                neighbors_in_radius = tf.cast(dists < self.real_search_radius, dtype=tf.float32)
                if test_type in ['density','combined_logical','combined_product','all']:            
                    if self.spatial_dimensions == 2:
                        #density = tf.cast(count,dtype=tf.float32) / (np.pi * (self.search_radius * self.hdp_suggested)**2)
                        density_batch = tf.cast(count,dtype=tf.float32) / (np.pi * (self.search_radius * self.hdp)**2)
                    elif self.spatial_dimensions == 3:
                        #density = tf.cast(count,dtype=tf.float32) / ((4/3) * np.pi * (self.search_radius * self.hdp_suggested)**3)
                        density_batch = tf.cast(count,dtype=tf.float32) / ((4/3) * np.pi * (self.search_radius * self.hdp)**3)

                # Calcula as distâncias até os centroides
                if test_type in ['centroid','combined_logical','combined_product','all']:
                    neighbors_coord = tf.gather(points,neighbors)
                    weights = tf.expand_dims(neighbors_in_radius, axis=-1)
                    centroides = tf.reduce_sum(neighbors_coord * weights, axis=1) / tf.reduce_sum(weights, axis=1)
                    centroid_distance_batch = tf.sqrt(tf.reduce_sum((centroides - points_batch)**2, axis=1)) / self.real_search_radius

                    # as tensor
                    #density[begin_slice:end_slice].assign(density_batch)
                    #centroid_distance[begin_slice:end_slice].assign(centroid_distance_batch)
                    #all_count[begin_slice:end_slice].assign(count)
                    #all_neighbors[begin_slice:end_slice,:].assign(neighbors)
                    
                    # as array
                    density[begin_slice:end_slice]  = density_batch.numpy()
                    centroid_distance[begin_slice:end_slice] = centroid_distance_batch.numpy()
                    all_count[begin_slice:end_slice] = count.numpy()
                    all_neighbors[begin_slice:end_slice,:] = neighbors.numpy()

                    #del neighbors, points_batch, density_batch, centroid_distance_batch, neighbors_coord, weights, neighbors_in_radius
                    #gc.collect()
                    #tf.keras.backend.clear_session()

        # Saida
        output = {}

        # 1. Predição com o teste da densidade
        if test_type in ['density','combined_logical','all']:
            pred_density = tf.cast(density.read_value() < density_threshold, dtype=tf.int32)
    
        # 2. Predição com o teste do centroide
        if test_type in ['centroid','combined_logical','all']:
            pred_centroid = tf.cast(centroid_distance.read_value() > centroid_threshold, dtype=tf.int32)
        
        # 3. Predição combinada 1
        if test_type in ['combined_logical','all']:
            pred_combined_logical = tf.cast(tf.math.logical_or(pred_density==1,pred_centroid==1), dtype=tf.int32)

        # 3. Predição combinada 2
        if test_type in ['combined_product','all']:
            combined_measure = density * (1 - centroid_distance)
            pred_combined_product = tf.cast(combined_measure < combined_threshold, dtype=tf.int32)

        if test_type == 'density':
            output['pred_density'] = pred_density
            if return_measures:
                #output['density'] = density.read_value()
                output['density'] = density
            #if return_neighbors:                
            #    output['neighbors'] = tf.gather(neighbors,tf.squeeze(tf.where(pred_density==1)))
            #    output['count_neighbors'] = tf.gather(count,tf.squeeze(tf.where(pred_density==1)))

        elif test_type == 'centroid':
            output['pred_centroid'] = pred_centroid
            if return_measures:
                output['centroid_distances'] = centroid_distance.read_value()
            #if return_neighbors:
            #    output['neighbors'] = tf.gather(neighbors,tf.squeeze(tf.where(pred_centroid==1)))
            #    output['count_neighbors'] = tf.gather(count,tf.squeeze(tf.where(pred_centroid==1)))
            
        elif test_type == 'combined_logical':
            output['pred_combined_logical'] = pred_combined_logical
            if return_measures:
                output['density'] = density.read_value()
                output['centroid_distances'] = centroid_distance.read_value()
            #if return_neighbors:
            #    output['neighbors'] = tf.gather(neighbors,tf.squeeze(tf.where(pred_combined_logical==1)))
            #    output['count_neighbors'] = tf.gather(count,tf.squeeze(tf.where(pred_combined_logical==1)))
            
        elif test_type == 'combined_product':
            output['pred_combined_product'] = pred_combined_product
            if return_measures:
                output['combined_measures'] = combined_measure
                #output['density'] = density.read_value()
                #output['centroid_distances'] = centroid_distance.read_value()                
                output['density'] = density
                output['centroid_distances'] = centroid_distance
            if return_neighbors:                
                #all_neighbors = all_neighbors.numpy()
                
                #output['neighbors'] = tf.gather(all_neighbors,tf.squeeze(tf.where(pred_combined_product==1)))
                output['neighbors'] = all_neighbors[pred_combined_product.numpy()==1]
                
                #output['count_neighbors'] = tf.gather(all_count,tf.squeeze(tf.where(pred_combined_product==1)))
                output['count_neighbors'] = all_count[pred_combined_product.numpy()==1]

        elif test_type == 'all':
            output['pred_combined_product'] = pred_combined_product
            output['pred_combined_logical'] = pred_combined_logical
            output['pred_density'] = pred_density
            output['pred_centroid'] = pred_centroid
            if return_measures:
                output['combined_measures'] = combined_measure
                output['density'] = density.read_value()
                output['centroid_distances'] = centroid_distance.read_value()
            #if return_neighbors:
            #    output['all_neighbors'] = neighbors
            #    output['all_count_neighbors'] = count
    

        return output        

    def nearest_points_band(self,points=None,target=None,kdtree=None,
        target_neighbors=None,return_distances=False):
        """
        Encontra a faixa de pontos mais próxima de uma subconjunto de pontos.
        Última atualização: 31/03/2022.
        
        Args:
            points:
            target_labels:    
            kdtree:
            target_neighbors:               
            return_distances: 
                
        Return:
           Índices de pontos mais próximos.
        """
        if target_neighbors is None:
            target_idx = np.where(target==1)[0] 
            if kdtree is None:
                kdtree = KDTree(points,device='cpu')
            target_neighbors,_ = kdtree.query_radius(
                points[target_idx],self.real_search_radius)
                
        nearest_band = np.zeros(points.shape[0],dtype=np.int)
        for neighbors in target_neighbors:
              nearest_band[neighbors] = 1
            
        if return_distances == True:
            _,neighbor_dists = kdtree.query_radius(
                points[nearest_band.astype(bool)],self.real_search_radius)
            return nearest_band,neighbor_dists
        else:
            return nearest_band

    def convert_images_dataset_ap2(self,images_dataset,particle_labels):
        """ 
        Converte o dataset de imagens sem rótulos da abordagem regional para 
        outro tipo com imagens que possuem os rótulos dos pixels codificados 
        nos canais.
        
        Esse formato é usado para salvar as imagens em disco que são usadas 
        como input da rede neural da abordagem regional.
        
        Última atualização: 13/07/2021.
        
        Args:
            images_dataset:
            particle_labels:
                
        Return:
            dataset em formato específico.        
        """
        shape = images_dataset['input_nn'].shape
        images_dataset_2 = np.zeros((shape[0],shape[1],shape[2],3))        
        
        # chanel 1
        images_dataset_2[:,:,:,0] = images_dataset['input_nn'][:,:,:,0]        
        
        for i in range(shape[0]):
            flat_idx = images_dataset['flat_target_idx'][i]            
            p_idx = images_dataset['particle_idx'][i]            
            labels_p = particle_labels[p_idx]
            
            flat_int_idx = flat_idx[labels_p==0]   # índices flat de partículas de interior
            flat_bound_idx = flat_idx[labels_p==1] # índices flat de partículas de borda                                                        
            
            # channel 2
            np.put(images_dataset_2[i,:,:,1],flat_int_idx,np.ones((flat_int_idx.shape[0])))            
            # channel 3
            np.put(images_dataset_2[i,:,:,2],flat_bound_idx,np.ones((flat_bound_idx.shape[0])))
        
        return images_dataset_2

    def convert_sparse_to_dense_dataset(self,sparse_dataset):
        """ 
        Converte um dataset de imagens esparsas da abordagem regional para 
        outro tipo com imagens densas que possuem os rótulos dos pixels codificados
        nos canais.
        
        Esse formato é usado para salvar as imagens em disco que são usadas 
        como input da rede neural da abordagem regional.
        
        Última atualização: 10/02/2022.
        
        Args:
            dataset:            
                
        Return:
            dataset de imagens densas.
        """
        full_neighbors_image_coord = sparse_dataset['full_neighbors_image_coord']
        target_neighbors_image_coord = sparse_dataset['target_neighbors_image_coord']
        target_labels = sparse_dataset['target_labels']

        dense_dataset = np.zeros(
            (full_neighbors_image_coord.shape[0],
            self.image_size,self.image_size,3))
    
        for i in range(dense_dataset.shape[0]):
            # channel 1
            np.put(
                dense_dataset[i,:,:,0], full_neighbors_image_coord[i], 1.0)
            # channel 2
            np.put(
                dense_dataset[i,:,:,1], 
                target_neighbors_image_coord[i],
                (target_labels[i]==0).astype(int)) 
            # channel 3
            np.put(
                dense_dataset[i,:,:,2], 
                target_neighbors_image_coord[i],
                (target_labels[i]==1).astype(int)) 

        return dense_dataset

    def create_voxel_meshes(self,initial_step=0,final_step=-1,skip_steps=10):
        """"
        Cria malhas de voxels.
        Última atualização: 17/02/2022.
        
        Args:
            initial_step:
            final_step:
            skip_steps:
        """
        # Diretórios
        data_dir = self.data_reader.data_dir
        mesh_dir = os.path.join(data_dir,'voxel_mesh')
            
        # Cria os diretório de saída caso não existam
        if not os.path.exists(mesh_dir):
            os.mkdir(mesh_dir)            
            
        # Grid
        voxelization = SparseVoxelizer(
            limits = self.real_grid_limits,
            res = self.real_grid_length,
            image_size = self.image_size,
            data_reader = self.data_reader,
            enable_plot=False)
        
        if final_step == -1:
            final_step = self.data_reader.data_info['final_step']

        for step in range(initial_step,final_step+1,skip_steps):
            print('Step {}\n'.format(step))
            # Carrega as particulas do passo corrente
            particles = self.data_reader.get_step(step)            
            mesh_file = os.path.join(mesh_dir,f'voxels.{step}.obj')
            
            t0 = time.time() 
            voxelizer.set_points(particles)
            #gt_labels = self.data_reader.get_step_labels_config(step,gt_config_file) # Carrega o ground-truth
            voxelizer.create_voxel_mesh(mesh_file)

            print(f' --> Total time: { time.time() - t0:.4f} s\n')
        
    def check_coarse_predictions_cpu_gpu(self,coarse_prediction_cpu,coarse_prediction_gpu):
        """ 
        Verifica se as predições grosseiras realizadas na cpu e gpu são iguais.
        Última atualização: 11/02/2022.

        Args:
            coarse_prediction_cpu:
            coarse_prediction_gpu:

        """        
        print('----------------------------------------------------------------------------------------')
        # pred density
        try:
            pred_density_cpu = coarse_prediction_cpu['pred_density']
            pred_density_gpu = coarse_prediction_gpu['pred_density']
            print('| pred_density sum abs diff: ',np.abs(pred_density_gpu.numpy() - pred_density_cpu).sum())
        except:
            pass
            #print('| "pred_density" não definida!')

        # pred centroid
        try:
            pred_centroid_cpu = coarse_prediction_cpu['pred_centroid']
            pred_centroid_gpu = coarse_prediction_gpu['pred_centroid']
            print('| pred_centroid sum abs diff: ',np.abs(pred_centroid_gpu.numpy() - pred_centroid_cpu).sum())
        except:
            pass
            #print('| "pred_centroid" não definida!')

        # pred combined 1
        try:
            pred_combined_logical_cpu = coarse_prediction_cpu['pred_combined_logical']
            pred_combined_logical_gpu = coarse_prediction_gpu['pred_combined_logical']
            print('| pred_combined_logical sum abs diff: ',np.abs(pred_combined_logical_gpu.numpy() - pred_combined_logical_cpu).sum())
        except:
            pass
            #print('| "pred_combined_logical" não definida!')
        
        # pred combined 2
        try:
            pred_combined_product_cpu = coarse_prediction_cpu['pred_combined_product']
            pred_combined_product_gpu = coarse_prediction_gpu['pred_combined_product']
            print('| pred_combined_product sum abs diff : ',np.abs(pred_combined_product_gpu.numpy() - pred_combined_product_cpu).sum())
        except:
            pass
            #print('| "pred_combined_product" não definida!')
        
        # density
        try:
            density_cpu = coarse_prediction_cpu['density']
            density_gpu = coarse_prediction_gpu['density']  
            print('| density abs max diff: ',np.abs(density_gpu.numpy() - density_cpu).max())
            print('| density_cpu: ',density_cpu)
            print('| density_gpu: ',density_gpu.numpy())
        except:
            pass
            #print('| "density" não definida!')

        # centroid_distance
        try:
            centroid_distance_cpu = coarse_prediction_cpu['centroid_distance']
            centroid_distance_gpu = coarse_prediction_gpu['centroid_distance']
            print('| centroid_distance abs max diff: ',np.abs(centroid_distance_gpu.numpy() - centroid_distance_cpu).max())
            print('| centroid_distance_cpu: ',centroid_distance_cpu)
            print('| centroid_distance_gpu: ',centroid_distance_gpu.numpy())
        except:
            pass
            #print('| "centroid_distance" não definida!')

        # combined_measure
        try:
            combined_measure_cpu = coarse_prediction_cpu['combined_measure']
            combined_measure_gpu = coarse_prediction_gpu['combined_measure']
            print('| combined_measure abs max diff: ',np.abs(combined_measure_gpu.numpy() - combined_measure_cpu).max())
        except:
            pass
            #print('| "combined_measure" não definida!')
        
        # count_neighbors
        try:
            count_neighbors_cpu = coarse_prediction_cpu['count_neighbors']
            count_neighbors_gpu = coarse_prediction_gpu['count_neighbors']
            print('| count_neighbors max (cpu): ',count_neighbors_cpu.max())
            print('| count_neighbors max (gpu): ',count_neighbors_gpu.numpy().max())
            print('| count_neighbors abs max diff: ',np.abs(count_neighbors_gpu.numpy() - count_neighbors_cpu).max())
        except:
            pass
            #print('| "count_neighbors" não definida!')

        # all_count_neighbors
        try:
            all_count_neighbors_cpu = coarse_prediction_cpu['all_count_neighbors']
            all_count_neighbors_gpu = coarse_prediction_gpu['all_count_neighbors']
            print('| all count_neighbors max (cpu): ',all_count_neighbors_cpu.max())
            print('| all count_neighbors max (gpu): ',all_count_neighbors_gpu.numpy().max())
            print('| all count_neighbors abs max diff: ',np.abs(all_count_neighbors_gpu.numpy() - all_count_neighbors_cpu).max())
        except:
            pass
            #print('| "all_count_neighbors" não definida!')

        print('----------------------------------------------------------------------------------------')