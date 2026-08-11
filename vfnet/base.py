import h5py
import numpy as np
import pandas as pd
import os,time,configparser
import gc
import tensorflow as tf
import matplotlib.pyplot as plt
import subprocess
import yaml
from vfnet.sliding_window import Core

import open3d as o3d
from vfnet.models import VFRWCNN
from vfnet.cnn_models.custom_layers import *
from sim_reader.config import ConfigReader
from voxelizer.sparse_voxelizer import SparseVoxelizer

from vfnet.cnn_models.sparse_models import SparseVoxelizedFluidCNN

from vfnet.plots import Plots2D
from vfnet.report import Reports

from sim_reader.data import DataReader
from sim_reader.config import ConfigReader

try:
    from tf_kdtree.neighbors import KDTree
    USE_TF_KDTREE = True
except:
    print('Tf KDTree não foi carregada corretamente!')
    USE_TF_KDTREE = False
    try:
        from scipy.spatial import cKDTree
        print('Usando scipy.spatial.cKDTree como fallback')
    except ImportError:
        from sklearn.neighbors import KDTree as SklearnKDTree
        print('Usando sklearn.neighbors.KDTree como fallback')

class VoxelFluid(Core):

    def __init__(self,data_config_file=None,tasks=['boundary'],
        features=['occupancy'],approach='pointwise',search_radius=2.0,grid_length=0.1,
        image_size=31,border_size=9,enable_plot=False, coarse_prediction='combined_product'):
        """ 
        Construtor.
        
        Args: 
            
        """
        super().__init__(
            data_config_file = data_config_file,
            tasks = tasks,
            approach = approach, 
            search_radius = search_radius,
            image_size = image_size,
            border_size = border_size,
            features = features,
            available_tasks = ['boundary','normal'],
            enable_plot = enable_plot
            )
        
        self.coarse_prediction = coarse_prediction

        # Propriedades da simulação
        self.point_distance = self.data_reader.properties_info['dp'] # distância inicial entre partículas
        self.kernel_length = self.data_reader.properties_info['h'] # kernel length
        self.hdp = self.kernel_length/self.point_distance
        self.ref_length = self.kernel_length
        
        self.spatial_dimensions = self.data_reader.properties_info['dimensions']
        
        #if self.spatial_dimensions==2:
            # Razão h/dp sugerida
        #    self.hdp_suggested = np.sqrt(2)
        #elif self.spatial_dimensions==3:
            # Razão h/dp sugerida
        #    self.hdp_suggested = np.sqrt(3)
        
        #if abs(self.hdp - self.hdp_suggested)<1e-3:
        #    self.ref_length = self.kernel_length
        #else:
        #    self.ref_length = self.hdp_suggested * self.point_distance
                
        self.real_data_limits = np.asarray(self.data_reader.properties_info['limits']).reshape(-1,2)

        self.set_dependent_attributes()

        # Configuração das tarefas
        self.tasks = tasks
        self.task_dimensions = {}
        self.task_types = {}
        if 'boundary' in self.tasks:
            self.labels = ['interior','boundary'] 
            self.task_dimensions['boundary'] = len(self.labels)
            self.task_types['boundary'] = 'classification'
        if 'normal' in self.tasks:
            self.task_dimensions['normal'] = self.spatial_dimensions
            self.task_types['normal'] = 'regression'

    def recompute_resolutions(self, particles, hdp=2.0):
        if USE_TF_KDTREE:
            kdtree = KDTree(particles, device='cpu')
            _, dists = kdtree.query(particles, knn=2)
        else:
            # Use scipy cKDTree as fallback
            kdtree = cKDTree(particles)
            dists, _ = kdtree.query(particles, k=2)
        mean_distance = np.mean(dists[:,1])
        self.ref_length = hdp*mean_distance
        self.set_dependent_attributes()
        print(f"Mean distance: {mean_distance:6f}")
        print(f"Recomputed ref length: {self.ref_length:6f}")
        print(f"Recomputed grid length: {self.real_grid_length:6f}")
        
    def build_dataset(self,gt_config_file,labels=None,based_on=['boundary'],
        dataset_id=0,initial_step=0,final_step=-1,skip_steps=1,train_name='train',
        val_name='validation',batch_size=100,max_batches=-1,random_ratio=0.1,
        vel_threshold=2.0,max_selected_per_bin=10000,balanced_selection=False,
        plot_distribution=False,resolution_based_on_mean_distance=False,hdp=2.0, 
        output_dir=None):
        """ 
        Cria o conjunto de treinamento para a abordagem regional.
        Última modificação: 23/08/2022.
        
        Args:
            gt_config_file: 
            labels=None:
            based_on:
            dataset_id:
            initial_ste:
            final_step:
            skip_steps:
            train_name:
            val_name:
            batch_size:
            max_batches:
            random_ratio:
            vel_threshold:
            max_selected_per_bin:
            balanced_selection:
            plot_distribution:

        """        
        # Diretórios
        str_par = f'{batch_size}_{self.search_radius:.2f}_{self.grid_length}_{dataset_id}'
        data_dir = self.data_reader.data_dir
        approach_dir = os.path.join(data_dir,f'{self.approach}_approach')        
        if output_dir is None:
            dataset_dir = os.path.join(approach_dir,'datasets',f'dataset_{str_par}') 
        else:
            dataset_dir = os.path.join(output_dir,f'dataset_{str_par}') 

        if os.path.exists(dataset_dir):
            raise Exception("Dataset already exists!")

        os.makedirs(dataset_dir)

        if labels == None:
            try:
                labels_config = ConfigReader(gt_config_file)
                labels = labels_config.get_section('boundary',['labels'])
                labels = labels['labels'].split()
            except:
                print('Erro: não foi possível carregar os nomes dos rótulos!')
                return
        num_classes = len(labels)
        
        if self.approach=='pointwise':
            dense_shape = self.spatial_dimensions*[self.image_size] + [1]
        elif self.approach=='regionwise':
            dense_shape = self.spatial_dimensions*[self.image_size] + [2]
    
        
        total_train = 0
        total_val = 0
        total_type_1 = 0
        total_type_2 = 0
        total_time = 0

        if initial_step == -1:
            initial_step = self.data_reader.data_info['initial_step']

        if final_step == -1:
            final_step = self.data_reader.data_info['final_step']
        
        steps = range(initial_step,final_step+1,skip_steps)
       
        idx = range(len(steps))

        steps_idx = dict(zip(steps,idx))

        if 'curvatures' in based_on:
            rep = Reports(self.data_reader)
            curv_report = rep.distribution_per_curvatures(
                gt_config_file,
                sections=['boundary', 'curvatures'],
                enable_plot=False,
                return_report=True)
            bins = curv_report['bins']
            distribution = curv_report['distribution'].astype(int)
            total_per_bin = curv_report['total_per_bin'].astype(int)
            ratios = curv_report['ratios']
                      
            selected_per_bin = max_selected_per_bin*ratios         
            selected_per_bin = selected_per_bin.astype(int)         
            if not balanced_selection:
                selected_per_bin = np.concatenate([selected_per_bin[:,:,np.newaxis],distribution[:,:,np.newaxis]],axis=2).min(axis=2)   
                selected_per_bin = selected_per_bin.astype(int)

            columns = ['bins'] + [f'step-{step}' for step in steps]
            array = np.concatenate([bins[np.newaxis].T,selected_per_bin[list(steps_idx.values()),:].T],axis=1)
            df = pd.DataFrame(array,columns=columns)
            report_file = os.path.join(dataset_dir,'distribution_per_curvatures.csv')
            df.to_csv(report_file,index=False,header=True)          
            
            if plot_distribution:
                res = bins[1]-bins[0]
                #bins_str =  [f'[{bins[i]},{bins[i+1]}]' for i in range(bins.shape[0]-1)]
                plt.figure()
                plt.bar(bins,selected_per_bin.sum(axis=0),res/2)
                plt.ylabel('Total',fontdict={'fontsize':15})
                plt.xlabel('Curvature Intervals',fontdict={'fontsize':15})
                plt.title('Selected particles per curvature intervals',fontdict={'fontsize':15})
                plt.xticks(bins[:-1])
                plt.xlim(-1.0,1.0)
                plt.grid(axis='y')  

        for step in steps:
            print('\nStep {}\n'.format(step))
            # Carrega as particulas do passo corrente
            particles = self.data_reader.get_step(step)
            sorted_idx = particles[:,0].argsort()
            particles = particles[sorted_idx,:]

            if resolution_based_on_mean_distance:
                self.recompute_resolutions(particles, hdp)

            # Voxelizador
            voxelizer = SparseVoxelizer(
                limits = self.real_grid_limits,
                res = self.real_grid_length,
                data_reader = self.data_reader,
                enable_plot = False)

            if 'velocity' in based_on:
                velocity = self.data_reader.get_step(step,'velocity')
                velocity = np.linalg.norm(velocity,axis=1)
            if 'curvatures' in based_on:
                curvatures = self.data_reader.get_step_measures(
                    step,gt_config_file,section='curvatures')
                curvatures = curvatures[:,0]


            gt_tasks = {}
            try:
                for task in self.tasks:
                    if self.task_types[task]=='classification':
                        gt_tasks[task] = self.data_reader.get_step_labels(
                            step,gt_config_file,section=task)
                        gt_tasks[task] = gt_tasks[task][sorted_idx]
                    elif self.task_types[task]=='regression':
                        gt_tasks[task] = self.data_reader.get_step_measures(
                            step,gt_config_file,section=task)
                        
                        gt_tasks[task] = gt_tasks[task][sorted_idx]
            except IndexError as error:
                print("Index Error:", error)
                continue
                

            # Encontra faixa de partículas próximas baseado em um 
            # ground-truth de rótulos especificado
            t = time.time()
            if 'boundary' in based_on:                
                selected = self.nearest_points_band(
                    points = particles,
                    target = gt_tasks['boundary'])
                selected_idx = np.where(selected==1)[0]                
            else:        
                selected_idx = None

            if 'velocity' in based_on:
                velocity = velocity[selected_idx]
                selected_idx_1 = selected_idx[velocity > vel_threshold]
                selected_idx_2 = selected_idx[velocity <= vel_threshold]
                num_random = int(random_ratio*selected_idx_2.shape[0])
                selected_idx_2 = np.random.choice(selected_idx_2,num_random,replace=balanced_selection) 
                selected_idx = np.concatenate([selected_idx_1,selected_idx_2])
                total_type_1 += selected_idx_1.shape[0]
                total_type_2 += selected_idx_2.shape[0]

            if 'curvatures' in based_on:
                res = curv_report['bins'][1]-curv_report['bins'][0]
                limits = np.array([[curv_report['bins'][0],curv_report['bins'][-1]]])

                curv = SparseVoxelizer(limits,res=res,expand_limits=False)
                curv.set_points(curvatures.reshape(-1,1))
                non_empty_bins = curv.find_unique_voxels().reshape(-1)
                indices_per_bin = curv.find_points_per_voxel()

                selected_idx = np.zeros(non_empty_bins.shape[0],dtype=object)
                for i in range(non_empty_bins.shape[0]):
                    selected_idx[i] = np.random.choice(
                        indices_per_bin[i],size=selected_per_bin[steps_idx[step],non_empty_bins[i]],replace=balanced_selection)

                selected_idx = np.concatenate(selected_idx)

            print(f' --> Particle selection... ({100*selected_idx.shape[0]/particles.shape[0]:.2f}% selected): {time.time()-t:.4f} s')
            if 'velocity' in based_on:
                print(f' ----> Type 1: {selected_idx_1.shape[0]}')
                print(f' ----> Type 2: {selected_idx_2.shape[0]}')

            if self.approach=='pointwise':
                report = self.build_dataset_pointwise(
                    points = particles,
                    gt_tasks = gt_tasks,
                    selected_idx = selected_idx,
                    dataset_dir = dataset_dir,
                    batch_size = batch_size,
                    max_batches = max_batches,
                    voxelizer = voxelizer)
            elif self.approach=='regionwise':
                report = self.build_dataset_regionwise(
                    points = particles,
                    gt_tasks = gt_tasks,
                    selected_idx = selected_idx,
                    dataset_dir = dataset_dir,
                    batch_size = batch_size,
                    max_batches = max_batches,
                    voxelizer = voxelizer,
                    save_data_to_debug = False)
            elif self.approach=='sparse_regionwise':
                report = self.build_dataset_sparse_regionwise(
                    points = particles,
                    gt_tasks = gt_tasks,
                    selected_idx = selected_idx,
                    dataset_dir = dataset_dir,
                    batch_size = batch_size,
                    max_batches = max_batches,
                    voxelizer = voxelizer,
                    save_data_to_debug = False)                
            
            total_train += report['train_batches']
            total_val += report['val_batches']
            total_time += report['time']

            tf.keras.backend.clear_session()
               
        print('\nDataset')
        if self.approach=='sparse_regionwise':
            print(f' --> total train batches: {total_train}')
            print(f' --> total validation batches: {total_val}')
        else:
            print(f' --> total train images: {total_train}')
            print(f' --> total validation images: {total_val}')

        if 'velocity' in based_on:
            print(f' --> Particles of type 1: {total_type_1}')
            print(f' --> Particles of type 2: {total_type_2}')
        print(f' --> Total time: {total_time:.4f} s\n')

        # Salva arquivo de configuração
        dataset_config = configparser.ConfigParser()
        dataset_config['general'] = {
            'dataset_id': dataset_id,
            'tasks': str(self.tasks),
            'task_dimensions': str(list(self.task_dimensions.values())),
            'task_types': str(list(self.task_types.values())),
            'labels': str(labels),
            'num_classes': num_classes,
            'approach': f"'{self.approach}'",
            'spatial_dimensions': self.spatial_dimensions,
            'grid_size': str(list(voxelizer.size)),
            #'border_size': self.border_size,
            #'image_length': f'{self.image_length:.2f}',
            #'real_image_length': f'{self.real_image_length:.6f}',
            'search_radius': f'{self.search_radius:.2f}',
            'real_search_radius': f'{self.real_search_radius:.6f}',
            'ref_length': f'{self.ref_length:.6f}',
            'grid_length': f'{self.grid_length:.2f}',
            'real_grid_length': f'{self.real_grid_length:.6f}',
            'used_steps': len(steps),
            'base_on': str(based_on),
            'velocity_threshold': vel_threshold,
            'random_ratio': random_ratio,
            'max_selected_per_bin': max_selected_per_bin}
        
        if self.approach=='pointwise':
            keys = ['images']
            for task in self.tasks:
                keys += f' {task}'
            dataset_config['train_set'] = {
                'name': "'train'",
                'format': "'hdf5'",
                'keys': str(keys),
                'type': "'dense'",
                'num_samples': total_train}
            dataset_config['validation_set'] = {
                'name': "'validation'",
                'format': "'hdf5'",
                'keys': str(keys),
                'type': "'dense'",
                'num_samples': total_val}            
        
        elif self.approach=='regionwise':
            keys = ['full_num_voxels', 'full_voxels_coord', 'target_num_voxels', 'target_voxels_coord']
            for task in self.tasks:
                keys += [f'target_{task}']

            dataset_config['train_set'] = {
                'name': f"'{train_name}'",
                'format': "'hdf5'",
                'keys': str(keys),
                'type': "'sparse'",
                'num_samples': total_train}

            dataset_config['validation_set'] = {
                'name': f"'{val_name}'",
                'format': "'hdf5'",
                'keys': str(keys),
                'type': "'sparse'",
                'num_samples': total_val}
            
        elif self.approach=='sparse_regionwise':
            keys = ['neighbor_voxels_num', 'neighbor_voxels_coord', 'target_voxels_num', 'target_voxels_coord']
            for task in self.tasks:
                keys += [f'target_{task}']

            dataset_config['train_set'] = {
                'name': f"'{train_name}'",
                'format': "'hdf5'",
                'keys': str(keys),
                'type': "'sparse'",
                'num_batches': total_train}

            dataset_config['validation_set'] = {
                'name': f"'{val_name}'",
                'format': "'hdf5'",
                'keys': str(keys),
                'type': "'sparse'",
                'num_batches': total_val}

        dataset_config_file = os.path.join(dataset_dir,'dataset_config_v2.ini')
        with open(dataset_config_file,'w') as configfile:
            dataset_config.write(configfile)

    def predict(self,points,model,voxelizer=None,coarse_threshold=0.7,
        batch_size=32,device='cpu',debug=False,return_times=False):
        """
        Classifica pontos com a abordagem pontual ou regional.
        Última modificação: 31/05/2022.
        
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
        if self.search_radius==2.0:
            coarse_threshold = 0.7
        elif self.search_radius==1.5:
            coarse_threshold = 0.8        

        if self.approach == 'pointwise':
            return self.predict_pointwise(
                points = points,
                model = model,
                voxelizer = voxelizer,
                coarse_threshold = coarse_threshold,
                batch_size = batch_size,
                device = device,
                debug = debug,
                return_times = return_times)
        elif self.approach == 'regionwise':       
            return self.predict_regionwise(
                points = points,
                model = model,
                voxelizer = voxelizer,
                coarse_threshold = coarse_threshold,
                batch_size = batch_size,
                device = device,
                debug = debug,                
                return_times = return_times)
        elif self.approach == 'sparse_regionwise':       
            return self.predict_sparse_regionwise(
                points = points,
                model = model,
                voxelizer = voxelizer,
                coarse_threshold = coarse_threshold,
                batch_size = batch_size,
                device = device,
                debug = debug,
                return_times = return_times)        

    def predict_offline(self,model_config_file=None,predict_id=0,batch_size=512,
        coarse_threshold=0.7,base_name='pred',extension='txt',device='cpu',debug=False,
        report_extension='csv',initial_step=-1,final_step=-1,skip_steps=1,grid_offset=False,
        pred_dir='predictions',return_prediction=False,resolution_based_on_mean_distance=False, 
        hdp=2.0, model_path=None, extract_mesh=False, decision_threshold=0.5, 
        poisson_recon_path=None):
        """
        Classifica partículas com a rede treinada na abordagem 2.
        Última modificação: 18/10/2022.
        
        Args:
            model_config_file:
            predict_id:
            batch_size:
            coarse_threshold:
            base_name:
            extension:
            device:
            debug:
            report_extension:
            initial_step:
            final_step:
            skip_steps:
            grid_offset:            
                  
        Returns:
            predictions:
        """         
        if not os.path.exists(model_config_file):
            raise FileNotFoundError(f'Model configuration file not found: {model_config_file}')
                       
        t0 = time.time()

        if self.approach == 'regionwise':
            bpartcnn = VFRWCNN(tasks=self.tasks)
        elif self.approach == 'sparse_regionwise':   
            if model_path is None:
                model_path = os.path.dirname(model_config_file)
            bpartcnn = tf.keras.models.load_model(model_path, 
                custom_objects={"SparseVoxelizedFluidCNN": SparseVoxelizedFluidCNN}
            )
            bpartcnn.summary()

        if self.approach in ['regionwise']:
            
            config_model = bpartcnn.load_model(
                model_config_file,
                custom_layers = {'Normalize':Normalize,'ArgMax':ArgMax},
                return_model_config=True)
            bpartcnn.model.summary()
            
            pred_config = (
                f"pred_{self.image_size}_{self.border_size}_{self.image_length:.2f}"
                f"_{config_model['name']}_{predict_id}")
        else:
            pred_config = f"pred_{bpartcnn.name}_{predict_id}"
        
        # Diretórios
        data_dir = self.data_reader.data_dir
        if self.approach == 'pointwise':
            approach_dir = os.path.join(data_dir,'pointwise_approach')
        elif self.approach == 'regionwise':
            approach_dir = os.path.join(data_dir,'regionwise_approach')
        elif self.approach == 'sparse_regionwise':
            approach_dir = os.path.join(data_dir,'sparse_regionwise_approach')

        #pred_global_dir = os.path.join(approach_dir,pred_dir)
        pred_config_dir = os.path.join(pred_dir,pred_config)
        #pred_dir = os.path.join(pred_config_dir,'pred')
        #os.makedirs(pred_config_dir)

        if 'boundary' in self.tasks:
            labels_dir = os.path.join(pred_config_dir,'boundary')
            os.makedirs(labels_dir,exist_ok=True)
        if 'normal' in self.tasks:
            normal_dir = os.path.join(pred_config_dir,'normal')
            os.makedirs(normal_dir,exist_ok=True)
        if extract_mesh:
            mesh_dir = os.path.join(pred_config_dir,'ply')
            os.makedirs(mesh_dir,exist_ok=True)
        if (set(self.tasks) == set(self.available_tasks))and extension=='csv':
            all_pred_dir = os.path.join(pred_config_dir,'all_predictions')
            os.makedirs(all_pred_dir,exist_ok=True)            

        # Voxelizador
        if grid_offset:
            np.random.seed(predict_id)
            grid_offset_val = self.real_grid_length * np.random.random((self.spatial_dimensions,1))
        else:
            grid_offset_val = 0
        
        if initial_step == -1:
            initial_step = self.data_reader.data_info['initial_step']

        if final_step == -1:
            final_step = self.data_reader.data_info['final_step']
        
        steps = np.arange(initial_step,final_step+1,skip_steps)

        # Times
        time_report_file = os.path.join(pred_config_dir,f'time_report.{report_extension}')
        if os.path.exists(time_report_file):
            if report_extension=='npz':
                times = dict(np.load(time_report_file, allow_pickle=True))
                for key in times:
                    times[key] = list(times[key])
            elif report_extension=='csv':
                df = pd.read_csv(time_report_file)
                times = {}
                for col in df.columns:
                    times[col] = df[col].tolist()
        else:
            times = {}
            times['coarse_prediction'] = []
            times['total'] = []
            times['steps'] = []
            if self.approach == 'pointwise':
                times['images_classification'] = []
                times['images_generation'] = []
            elif self.approach == 'regionwise':
                times['images_classification'] = []
                times['images_generation_search'] = []
                times['images_generation_build'] = []
            elif self.approach == 'sparse_regionwise':
                times['fine_prediction'] = []
        
        if return_prediction:
            all_predictions = []

        for s,step in enumerate(steps):
            print('Step {}\n'.format(step))
            if step in np.array(times['steps'],dtype=np.int32):
                print(f" Step {step} has already been processed!\n")
                continue
            
            if 'boundary' in self.tasks:
                labels_file = os.path.join(labels_dir,f'labels.{step}.{extension}')
            if 'normal' in self.tasks:
                normal_file = os.path.join(normal_dir,f'normal.{step}.{extension}')
            if (set(self.tasks) == set(self.available_tasks))and extension=='csv':
                pred_file = os.path.join(all_pred_dir,f'pred.{step}.{extension}')            

            particles = self.data_reader.get_step(step)

            if resolution_based_on_mean_distance:
                self.recompute_resolutions(particles,hdp)

            voxelizer = SparseVoxelizer(
                limits = self.real_grid_limits + grid_offset_val,
                res = self.real_grid_length,
                data_reader = self.data_reader,
                enable_plot = debug)

            predictions,step_times = self.predict(
                points = particles,
                model = bpartcnn,
                voxelizer = voxelizer,
                coarse_threshold = coarse_threshold,
                batch_size = batch_size,
                device = device,
                debug = debug,
                return_times = True)

            times['steps'].append(step)
            times['coarse_prediction'].append(step_times['coarse_prediction'])
            times['total'].append(step_times['total'])
            if self.approach in ['pointwise','regionwise']:
                times['images_classification'].append(step_times['images_classification'])                
                if self.approach == 'pointwise':
                    times['images_generation'].append(step_times['images_generation'])
                elif self.approach == 'regionwise':
                    times['images_generation_search'].append(step_times['images_generation_search'])
                    times['images_generation_build'].append(step_times['images_generation_build'])
            elif self.approach == 'sparse_regionwise':
                times['fine_prediction'].append(step_times['fine_prediction'])

            print(f" --> Coarse prediction time: {times['coarse_prediction'][s]:.4f} s")
            if self.approach in ['pointwise','regionwise']:
                if self.approach == 'pointwise':
                    print(f" --> Images generation time: {times['images_generation'][s]:.4f} s")
                elif self.approach == 'regionwise':
                    print(f" --> Images generation time (kdtree search): {times['images_generation_search'][s]:.4f} s")
                    print(f" --> Images generation time (build): {times['images_generation_build'][s]:.4f} s")

                print(f" --> Images classification time: {times['images_classification'][s]:.4f} s")
            elif self.approach == 'sparse_regionwise':
                print(f" --> Fine prediction time : {times['fine_prediction'][s]:.4f} s")
                
            print(f" --> Total time: {times['total'][s]:.4f} s\n")
            
            if return_prediction:
                all_predictions.append(predictions)
            else:
                boundary = None
                if 'boundary' in self.tasks:
                    boundary = predictions['boundary'][:,1] > decision_threshold
                if extract_mesh:
                    if boundary is None or 'normal' not in self.tasks:
                        raise ValueError(
                            "extract_mesh=True requires both 'boundary' and 'normal' tasks."
                        )
                    print("Saving ply prediction file...")
                    pred_ply_file = os.path.join(mesh_dir, f"boundary.{step}.ply")                    
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(particles[boundary])
                    pcd.normals = o3d.utility.Vector3dVector(predictions['normal'][boundary])
                    o3d.io.write_point_cloud(pred_ply_file, pcd)                            
                    print("Extracting ply mesh file...")
                    mesh_ply_file = os.path.join(mesh_dir, f"mesh.boundary.{step}.ply")
                    
                    # Use configured path or default
                    if poisson_recon_path is None:
                        poisson_recon_path = "/home/samuel/Doutorado/SurfaceReconstruction/AdaptiveSolvers/Bin/Linux/PoissonRecon"
                    
                    # Check if PoissonRecon executable exists
                    if not os.path.exists(poisson_recon_path):
                        print(f"Warning: PoissonRecon not found at '{poisson_recon_path}'. Skipping mesh extraction.")
                        print(f"Point cloud saved at: {pred_ply_file}")
                    else:
                        # Create temp directory for PoissonRecon cache files
                        temp_dir = os.path.join(pred_config_dir, 'temp')
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        log = subprocess.run(
                            [poisson_recon_path, "--in", pred_ply_file, "--out", mesh_ply_file, 
                             "--depth", "9", "--tempDir", temp_dir], 
                            universal_newlines=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            input=''
                        )
                        print(log)
                if extension=='npy':
                    if 'boundary' in self.tasks:
                        np.save(labels_file,boundary.astype(int))
                    if 'normal' in self.tasks:
                        np.save(normal_file,predictions['normal'])
                elif extension=='txt':
                    if 'boundary' in self.tasks:
                        np.savetxt(labels_file,boundary.astype(int),fmt='%d')
                    if 'normal' in self.tasks:
                        np.savetxt(normal_file,predictions['normal'],fmt='%.6f')
                elif extension=='csv':        
                    array = particles
                    if self.spatial_dimensions==2:
                        columns = ['x','y']
                        columns_normal = ['pred_nx','pred_ny']
                    elif self.spatial_dimensions==3:
                        columns = ['x','y','z']
                        columns_normal = ['pred_nx','pred_ny','pred_nz']
                    if 'normal' in self.tasks:
                        array_normal = np.concatenate(
                            [particles,predictions['normal']],axis=-1)
                        df_normal = pd.DataFrame(
                            array_normal,columns = columns+columns_normal)
                        df_normal.to_csv(normal_file,index=False,header=True)
                    if 'boundary' in self.tasks:
                        columns_labels = ['pred_labels']
                        array_labels = np.concatenate(
                            [particles,predictions['boundary'][:,np.newaxis]],axis=-1)
                        df_labels = pd.DataFrame(
                            array_labels,columns = columns+columns_labels)
                        df_labels.to_csv(labels_file,index=False,header=True)                        
                    if (set(self.tasks) == set(self.available_tasks)):
                        array = np.concatenate(
                            [particles,predictions['boundary'][:,np.newaxis],predictions['normal']],axis=-1)
                        df = pd.DataFrame(
                            array,columns=columns+columns_labels+columns_normal)
                        df.to_csv(pred_file,index=False,header=True)
                else:
                    raise ValueError(f"A extensão '{extension}' não é suportada!")

                # Salva relatório de tempos
                if report_extension=='npz':
                    if self.approach=='pointwise':
                        np.savez(time_report_file,steps = steps,
                            coarse_prediction = times['coarse_prediction'],
                            images_generation = times['images_generation'],
                            images_classification = times['images_classification'],
                            total = times['total'])
                    elif self.approach=='regionwise':
                        np.savez(time_report_file,steps = steps,
                            coarse_prediction = times['coarse_prediction'],
                            images_generation_search = times['images_generation_search'],
                            images_generation_build = times['images_generation_build'],
                            images_classification = times['images_classification'],
                            total = times['total'])
                    elif self.approach=='sparse_regionwise':                      
                        np.savez(time_report_file, 
                            coarse_prediction = times['coarse_prediction'],
                            fine_prediction = times['fine_prediction'],
                            total = times['total'])
                                
                if report_extension=='csv':
                    if self.approach=='pointwise':
                        columns = ['steps','coarse_prediction','images_generation',
                            'images_classification','total']
                        array = np.array([times['steps'],times['coarse_prediction'],
                            times['images_generation'],times['images_classification'],
                            times['total']]).T
                    elif self.approach=='regionwise':
                        columns = ['steps','coarse_prediction','images_generation_search',
                            'images_generation_build','images_classification','total']
                        array = np.array([times['steps'],times['coarse_prediction'],
                            times['images_generation_search'],times['images_generation_build'],
                            times['images_classification'],times['total']]).T
                    elif self.approach=='sparse_regionwise':   
                        columns = ['steps','coarse_prediction','fine_prediction','total']
                        array = np.array([times['steps'],times['coarse_prediction'],
                            times['fine_prediction'],times['total']]).T

                    df = pd.DataFrame(array.round(3),columns=columns)
                    df.to_csv(time_report_file,index=False,header=True)

                # Salva o arquivo de configuração da predição.
                pred_config = {
                    'general': {
                        'approach': self.approach,
                        'image_size': int(self.image_size),
                        'border_size': int(self.border_size),
                        'image_length': float(f'{self.image_length:.2f}'),
                        'real_image_length': float(f'{self.real_image_length:.6f}'),
                        'search_radius': float(f'{self.search_radius:.2f}'),
                        'real_search_radius': float(f'{self.real_search_radius:.6f}'),
                        'ref_length': float(f'{self.ref_length:.6f}'),
                        'grid_length': float(f'{self.grid_length:.2f}'),
                        'real_grid_length': float(f'{self.real_grid_length:.6f}')
                    }
                }
                
                if self.approach in ['pointwise','regionwise']:
                    pred_config['model'] = {
                        'name': bpartcnn.model.name,
                        'architecture_file': config_model['architecture_file'],
                        'weights_file': config_model['weights_file'],
                        'model_template': config_model['model_template'],
                        'config_file': model_config_file
                    }
                elif self.approach == 'sparse_regionwise':
                    pred_config['model'] = {
                        'name': bpartcnn.name,
                        'config_file': model_config_file
                    } 
                if 'boundary' in self.tasks:
                    pred_config['boundary'] = {
                        'task_type': 'classification',
                        'prediction_id': predict_id,
                        'labels': self.labels,
                        'device': device,
                        'coarse_test': 'combined_product',
                        'combined_threshold': float(coarse_threshold),
                        'dir': 'boundary',
                        'base_name': 'labels',
                        'extension': extension
                    }  
                if 'normal' in self.tasks:
                    pred_config['normal'] = {
                        'task_type': 'regression',
                        'prediction_id': predict_id,
                        'device': device,
                        'coarse_test': 'combined_product',
                        'combined_threshold': float(coarse_threshold),
                        'dir': 'normal',
                        'base_name': 'normal',
                        'extension': extension
                    }
                if (set(self.tasks) == set(self.available_tasks)) and extension=='csv':
                    pred_config['all_predictions'] = {
                        'prediction_id': predict_id,
                        'labels': self.labels,
                        'device': device,
                        'coarse_test': 'combined_product',
                        'combined_threshold': float(coarse_threshold),
                        'dir': 'all_predictions',
                        'base_name': 'pred',
                        'extension': extension
                    }
                pred_config_file = os.path.join(pred_config_dir,'pred_config_v2.yaml')
                with open(pred_config_file,'w') as configfile:
                    yaml.dump(pred_config, configfile, default_flow_style=False, sort_keys=False)


    def coarse_prediction_offline(self,test_type='combined_product',combined_threshold=0.7,
        density_threshold=0.82,centroid_threshold=0.14,prediction_id=0,device='cpu',
        initial_step=-1,final_step=-1,skip_steps=1,gt_config_file=None,base_name='pred',
        extension='txt',save_outputs=False):
        """
        Roda o teste grosseiro que usa contagem de partículas e a distância até 
        o centroide.
        Última modificação: 06/06/2022.
        
        Args:
            test_type:
            combined_threshold:
            density_threshold:
            centroid_threshold:
            prediction_id:
            device:
            initial_step:
            final_step:
            skip_steps:
            gt_config_file:
            base_name:
            extension:
            save_outputs:

        """
        if self.search_radius==1.5:
            combined_threshold = 0.8
        elif self.search_radius==2.0:
            combined_threshold = 0.7

        str_par = f'{self.search_radius}_{prediction_id}'
        
        # Diretórios
        data_dir = self.data_reader.data_dir
        coarse_pred_dir = os.path.join(data_dir,'coarse_predictions') 

        if test_type=='density':
            coarse_pred_dir = os.path.join(coarse_pred_dir,f'pred_density_{str_par}')
        elif test_type=='centroid':
            coarse_pred_dir = os.path.join(coarse_pred_dir,f'pred_centroid_{str_par}')        
        elif test_type=='combined_logical':
            coarse_pred_dir = os.path.join(coarse_pred_dir,f'pred_comb_logical_{str_par}')
        elif test_type=='combined_product':
            coarse_pred_dir = os.path.join(coarse_pred_dir,f'pred_comb_product_{str_par}')
        elif test_type=='all':
            coarse_pred_dir = os.path.join(coarse_pred_dir,f'pred_all_{str_par}')

        if test_type in ['density','all']:
            pred_density_dir = os.path.join(coarse_pred_dir,'pred_density')
            density_dir = os.path.join(coarse_pred_dir,'density')
            os.makedirs(pred_density_dir, exist_ok=True)
            os.makedirs(density_dir, exist_ok=True)
        
        if test_type in ['centroid','all']:
            pred_centroid_dir = os.path.join(coarse_pred_dir,'pred_centroid')
            centroid_distances_dir = os.path.join(coarse_pred_dir,'centroid_distances')
            os.makedirs(pred_centroid_dir, exist_ok=True)
            os.makedirs(centroid_distances_dir, exist_ok=True)

        if test_type in ['combined_logical','all']:
            pred_combined_logical_dir = os.path.join(coarse_pred_dir,'pred_comb_logical')
            os.makedirs(pred_combined_logical_dir, exist_ok=True)                

        if test_type in ['combined_product','all']:
            pred_combined_product_dir = os.path.join(coarse_pred_dir,'pred_comb_product')
            combined_measure_dir = os.path.join(coarse_pred_dir,'combined_measures')
            os.makedirs(pred_combined_product_dir, exist_ok=True)
            os.makedirs(combined_measure_dir, exist_ok=True)
        
        if test_type in ['gt_band','all']:
            gt_band_dir = os.path.join(coarse_pred_dir,'gt_band')
            count_neighbors_dir = os.path.join(coarse_pred_dir,'count_neighbors')
            os.makedirs(count_neighbors_dir, exist_ok=True)
            os.makedirs(gt_band_dir, exist_ok=True)
                                    
        # Inicia classificação grosseira
        time_per_frame = []
        if initial_step==-1:
            initial_step = self.data_reader.data_info['initial_step']                              
        if final_step==-1:
            final_step = self.data_reader.data_info['final_step']               
        for step in range(initial_step,final_step+1,skip_steps):
            print(f'\nStep {step}')
            points = self.data_reader.get_step(step)
                       
            # Teste grosseiro
            t = time.time()

            if test_type in ['density','centroid','combined_logical','combined_product','all']:
                if device in ['cpu','cpu_gpu']:
                    prediction_cpu = self.coarse_prediction_cpu(
                        points,
                        test_type = test_type,
                        combined_threshold = combined_threshold,
                        density_threshold = density_threshold,
                        centroid_threshold = centroid_threshold,
                        return_neighbors = True,
                        return_measures = True)
                    #selected = coarse_prediction_cpu['pred_combined_product']
                    #neighbors = coarse_prediction_cpu['neighbors']
                    predictions = prediction_cpu

                if device in ['gpu','cpu_gpu']:
                    prediction_gpu = self.coarse_prediction_gpu(
                        points,
                        test_type = test_type,
                        combined_threshold = combined_threshold,
                        density_threshold = density_threshold,
                        centroid_threshold = centroid_threshold,
                        return_neighbors = False,
                        return_measures = True)
                    #selected = coarse_prediction_gpu['pred_combined_product'].numpy()
                    predictions = prediction_gpu                    
                
                if device == 'cpu_gpu':
                    self.check_coarse_predictions_cpu_gpu(prediction_cpu,prediction_gpu)
                    continue                                        

            if test_type in ['gt_band','all']:
                gt_labels = self.data_reader.get_step_labels(
                    step, gt_config_file, section='boundary')
                gt_band_labels = self.nearest_points_band(
                    points = points,target = gt_labels)

            pred_time = time.time() - t

            print(f' --> coarse prediction time: {pred_time:.4f} s')
            
            time_per_frame.append(time.time() - t)
            
            if self.enable_plot:
                plt.cla()
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.scatter(predictions['density'][gt_labels==0],predictions['centroid_distances'][gt_labels==0])                
                plt.scatter(predictions['density'][gt_labels==1],predictions['centroid_distances'][gt_labels==1])
                plt.legend(['interior','boundary'])
                plt.title(f'step: {step}')
                plt.pause(0.1)
                     

            if save_outputs:                    
                if test_type in ['density','all']:
                    pred_file = os.path.join(
                        pred_density_dir,f'pred.density.{step}.{extension}')
                    density_file = os.path.join(
                        density_dir,f'density.{step}.{extension}')
                    if extension=='npy':
                        np.save(pred_file,predictions['pred_density'])                
                        np.save(density_file,predictions['density'])
                    if extension=='txt':
                        np.savetxt(pred_file,predictions['pred_density'],fmt='%d')
                        np.savetxt(density_file,predictions['density'],fmt='%.6f')
                    
                if test_type in ['centroid','all']:
                    pred_file = os.path.join(
                        pred_centroid_dir,f'pred.centroid.{step}.{extension}')
                    centroid_file = os.path.join(
                        centroid_distances_dir,f'centroid.distances.{step}.{extension}')
                    if extension=='npy':
                        np.save(pred_file,predictions['pred_centroid'])
                        np.save(centroid_file,predictions['centroid_distances'])
                    if extension=='txt':
                        np.savetxt(pred_file,predictions['pred_centroid'],fmt='%d')
                        np.savetxt(centroid_file,predictions['centroid_distances'],fmt='%.6f')

                if test_type in ['combined_logical','all']:    
                    pred_file = os.path.join(
                        pred_combined_logical_dir,f'pred.comb.logical.{step}.{extension}')
                    if extension=='npy':
                        np.save(pred_file,predictions['pred_combined_logical'])
                    if extension=='txt':                        
                        np.savetxt(pred_file,predictions['pred_combined_logical'],fmt='%d')
                
                if test_type in ['combined_product','all']:
                    pred_file = os.path.join(
                        pred_combined_product_dir,f'pred.comb.product.{step}.{extension}')
                    combined_measure_file = os.path.join(
                        combined_measure_dir,f'combined.measures.{step}.{extension}')                    
                    if extension=='npy':
                        np.save(pred_file,predictions['pred_combined_product'])
                        np.save(combined_measure_file,predictions['combined_measures'])                            
                    if extension=='txt':
                        np.savetxt(pred_file,predictions['pred_combined_product'],fmt='%d')
                        np.savetxt(combined_measure_file,predictions['combined_measures'],fmt='%.6f')                            

                if test_type in ['gt_band','all']:
                    gt_band_file = os.path.join(
                        gt_band_dir,f'gt.band.{step}.{extension}')
                    if extension=='npy':
                        np.save(gt_band_file,gt_band_labels) 
                    elif extension=='txt':
                        np.savetxt(gt_band_file,gt_band_labels,fmt='%d')

                #if test_type in ['all']:
                #    count_file = os.path.join(
                #        count_neighbors_dir,f'count.neighbors.{step}.{extension}')
                #    if extension=='npy':
                #        np.save(count_file,predictions['all_count_neighbors']) 
                #    elif extension=='txt':
                #        np.savetxt(count_file,predictions['all_count_neighbors'],fmt='%d')

            tf.keras.backend.clear_session()                
                                                                                   
        # Salva o arquivo de configuração da predição
        if save_outputs:                    
            pred_config = configparser.ConfigParser()            
            pred_config['general'] = {
                'prediction_id': prediction_id,
                'search_radius': self.search_radius,
                'labels':'interior undefined',
                'pred_sections': '',
                'measure_sections': ''}
            if test_type in ['density','all']:
                pred_config['general']['pred_sections'] +=  'pred_density '
                pred_config['general']['measure_sections'] +=  'density '

                pred_config['pred_density'] = {
                    'dir': 'pred_density',
                    'base_name': 'pred.density',
                    'extension': extension,
                    'density_threshold': density_threshold}
                pred_config['density'] = {
                    'dir': 'density',
                    'base_name': 'density',
                    'extension': extension}

            if test_type in ['centroid','all']:
                pred_config['general']['pred_sections'] +=  'pred_centroid '
                pred_config['general']['measure_sections'] +=  'centroid_distances '

                pred_config['centroid_distances'] = {
                    'dir': 'centroid_distances',
                    'base_name': 'centroid.distances',
                    'extension': extension}
                pred_config['pred_centroid'] = {
                    'dir': 'pred_centroid',
                    'base_name': 'pred.centroid',
                    'extension': extension,
                    'centroid_threshold': centroid_threshold}

            if test_type in ['combined_logical','all']: 
                pred_config['general']['pred_sections'] +=  'pred_combined_logical '

                pred_config['pred_combined_logical'] = {
                    'dir': 'pred_comb_logical',
                    'base_name': 'pred.comb.logical',
                    'extension': extension,
                    'density_threshold': density_threshold,
                    'centroid_threshold': centroid_threshold}

            if test_type in ['combined_product','all']:
                pred_config['general']['pred_sections'] +=  'pred_combined_product '                
                pred_config['general']['measure_sections'] +=  'combined_measures '

                pred_config['pred_combined_product'] = {
                    'dir': 'pred_comb_product',
                    'base_name': 'pred.comb.product',
                    'extension': extension,
                    'combined_threshold': combined_threshold}
                pred_config['combined_measures'] = {
                    'dir': 'combined_measures',
                    'base_name': 'combined.measures',
                    'extension': extension}                

            if test_type in ['gt_band','all']:
                pred_config['general']['pred_sections'] +=  'pred_gt_band '

                pred_config['pred_gt_band'] = {
                    'dir': 'gt_band',
                    'base_name': 'gt.band',
                    'extension': extension}

            if test_type in ['all']:                
                pred_config['general']['measure_sections'] +=  'count_neighobors '
              
                pred_config['count_neighbors'] = {
                    'dir': 'count_neighbors',
                    'base_name': 'count.neighbors',
                    'extension': extension}

            pred_config_file = os.path.join(coarse_pred_dir, 'pred_config.ini')
            with open(pred_config_file,'w') as configfile:
                pred_config.write(configfile)
                
            # Salva outras saídas
            other_outputs_file = os.path.join(coarse_pred_dir,'other_outputs.npz')
            np.savez(other_outputs_file,
                search_radius = self.search_radius,
                time_per_frame = time_per_frame,
                density_threshold = pred_config_file,
                centroid_threshold = centroid_threshold,
                combined_threshold = combined_threshold)

    def particle_band_offline(self,test_id=0,gt_config_file=None,initial_step=0,
        final_step=-1,skip_steps=1,base_name='gt_band',extension='npy',
        save_outputs=False):
        """
        Busca a faixa de partículas próximas do ground-truth.
        o centroide.
        Última modificação: 31/05/2022. 
        
        Args:
            test_id:
            gt_config_file:
            initial_step:
            final_step:
            skip_steps:
            base_name:           
            extension:
            save_outputs:
            
        """
        str_par = f'{self.search_radius}_{test_id}'
                            
        # Diretórios
        data_dir = self.data_reader.data_dir
        coarse_pred_dir = os.path.join(
            data_dir,'coarse_predictions',f'gt_band_{str_par}')
        gt_band_dir = os.path.join(coarse_pred_dir,'gt_band')
        os.makedirs(gt_band_dir, exist_ok=True)
               
        if final_step==-1:
            final_step = self.data_reader.data_info['final_step']      
                    
        for step in range(initial_step,final_step+1,skip_steps):
            print(f'Step {step}\n')
            particles = self.data_reader.get_step(step)
                       
            t = time.time()
            if USE_TF_KDTREE:
                kdtree = KDTree(particles)
            else:
                kdtree = cKDTree(particles)
            
            # Encontra faixa de partículas próximas do ground-truth
            gt_labels = self.data_reader.get_step_labels(step,gt_config_file)
            gt_band_labels = self.nearest_points_band(
                points = particles,
                target = gt_labels,
                kdtree = kdtree)
            band_time = time.time() - t

            print(f' --> gt band... ({100*gt_band_labels.sum()/particles.shape[0]:.2f}%): {band_time:.4f} s')

            if save_outputs:
                gt_band_file = os.path.join(
                    gt_band_dir,f'{base_name}.{step}.{extension}')
                if extension=='npy':
                    np.save(gt_band_file,gt_band_labels)
                elif extension=='txt':
                    np.savetxt(gt_band_file,gt_band_labels,fmt='%d')

        # Salva o arquivo de configuração
        if save_outputs:                    
            pred_config = configparser.ConfigParser()           
            pred_config['general'] = {
                'description':'contains particle labels close to the boundary',
                'labels':'interior gtband',
                'radius':self.search_radius
            }
            pred_config['pred_gt_band'] = {
                'dir': 'gt_band.',
                'base_name': base_name,
                'extension': extension
            }
            pred_config_file = os.path.join(coarse_pred_dir, 'gt_band_config.ini')
            with open(pred_config_file,'w') as configfile:
                pred_config.write(configfile)            
