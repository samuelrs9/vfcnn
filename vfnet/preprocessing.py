import os
import h5py
import time
import configparser
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sim_reader.data import DataReader

from pysph.base.kernels import *
import trimesh

try:
    from tf_kdtree.neighbors import KDTree
except:
    print('Tf KDTree não foi carregada corretamente!')

class PreprocessSimulation:
    
    def __init__(self,config_file=None):
        """ 
        Construtor
        Última modificação: 23/03/2022.
        
        Args:
            data_reader:
        """
        if config_file is not None:
            self.config_file = config_file
            self.data_dir = os.path.dirname(config_file)
            self.data_reader = DataReader(config_file)
    
    def create_sim_config_from_dualsphysics(self,data_dir,steps_dir,
        gencase_file,target_line=None,sim_dim=3,initial_step=0,final_step=300,
        extension='csv',base_name='PartFluid',base_sep='_',origin='dualsphysics'):
        """ 
        Cria o arquivo de configuração da simulação a partir do arquivo de
        propriedades da simulação do DualSPHysics.
        Última modificação: 07/07/2022. 
        
        Args: 
            gencase_file:
            steps_dir:
            target_line:
            sim_dim:
            origin:
        """    
        sim_config_file = os.path.join(data_dir,'sim_config.ini')
        if os.path.exists(sim_config_file):
            print("Warning: simulation configuration file already exists!")
            return
        if target_line is None:          
            target_line = 'List of available variables: '
        found_target_line = False
        with open(gencase_file,'r') as file:
            line = file.readline()
            while line is not '':    
                if target_line in line:
                    found_target_line = True
                    line = line.replace(target_line,'')
                    line = line.replace('[','')
                    line = line.replace(']','')    
                    line = line.split()                    
                    for p in line:                        
                        if 'Dp=' in p:
                            dp = float(p.split('=')[1])
                            #print('dp = ',dp)
                        if 'H=' in p:
                            h = float(p.split('=')[1])
                            #print(' --> h = ',h)
                        if 'MassFluid=' in p:
                            mass = float(p.split('=')[1])
                            #print('mass: ',mass)
                        # Eixo X
                        if 'PosMin_x=' in p:
                            xmin = float(p.split('=')[1])
                            print('xmin: ',xmin)
                        if 'PosMax_x=' in p:
                            xmax = float(p.split('=')[1])
                            #print('xmax: ',xmax)                                            
                        # Eixo Y
                        if 'PosMin_y=' in p:
                            ymin = float(p.split('=')[1])
                            #print('ymin: ',ymin)
                        if 'PosMax_y=' in p:
                            ymax = float(p.split('=')[1])
                            #print('ymax: ',ymax)                 
                        # Eixo Z    
                        if 'PosMin_z=' in p:
                            zmin = float(p.split('=')[1])
                            #print('ymin: ',zmin)
                        if 'PosMax_z=' in p:
                            zmax = float(p.split('=')[1])
                            #print('ymax: ',zmax)   
                    break 
                else:           
                    line = file.readline()  
                
        if not found_target_line:
            raise Exception('Target line not found in gencase file!')

        if sim_dim==2:
            limits = [xmin,xmax,zmin,zmax]
        elif sim_dim==3:
            limits = [xmin,xmax,ymin,ymax,zmin,zmax]

        print('Simulation properties')
        print(' Limits: ',limits)
        print(' dp: ',dp)
        print(' h: ',h)

        if origin=='paraview':
            coords_col = ['Points:0','Points:1','Points:2']                              
            vel_col = ['Vel:0','Vel:1','Vel:2']                              
        elif origin=='dualsphysics':
            coords_col = ['Pos.x [m]','Pos.y [m]','Pos.z [m]']
            vel_col = ['Vel.x [m/s]','Vel.y [m/s]','Vel.z [m/s]']

        # Salva o arquivo de configuração da simulação
        sim_config = configparser.ConfigParser()    
                
        sim_config['simulation_properties'] = {
            'dp':dp,
            'h':h,
            'mass':mass,
            'dimensions':sim_dim,
            'limits':' '.join([str(x) for x in limits])}
        
        sim_config['data'] = {
            'sim_name': data_dir.split(os.sep)[-1],
            'from': origin,
            'frames_dir': steps_dir.replace(data_dir,'')[1:],
            'base_name': base_name,
            'base_sep': base_sep,
            'extension': extension,
            'initial_step': initial_step,
            'final_step': final_step,
            'coords': ' / '.join(coords_col),
            'velocity': ' / '.join(vel_col)
        }        
        with open(sim_config_file,'w') as configfile:
            sim_config.write(configfile)  
    
    def create_sim_config_from_dict(self,sim_properties={},data={}):
        pass

    def create_simdef(self,sim_config_file):
        """ 
        Cria o arquivo de configuração da simulação para o Matlab a partir do arquivo de
        configuração do código Python.
        Última modificação: 21/08/2021. 
        
        Args: 
            sim_config_file:
        """
        data_reader = DataReader(self.data_dir,sim_config_file)
        
        limits = data_reader.properties_info['limits']
        dimensions = data_reader.properties_info['dimensions']
        mass = data_reader.properties_info['mass']
        h = data_reader.properties_info['h']
        dp = data_reader.properties_info['dp']
        
        frames_dir = data_reader.data_info['frames_dir']
        base_name = data_reader.data_info['base_name']
        
        initial_step = data_reader.data_info['initial_step']
        final_step = data_reader.data_info['final_step']        
        
        x_col_id = 1
        y_col_id = 3
        
        simdef_file = os.path.join(self.data_dir,frames_dir,'simdef.m')
        
        with open(simdef_file,'w') as file:
            file.write('dims = [{} {} {} {}];\n'.format(limits[0],limits[1],limits[2],limits[3]))
            file.write('dim = {};\n'.format(dimensions))
            file.write('dp = {};\n'.format(dp))
            file.write('h = {};\n'.format(h))
            file.write('mass = {};\n\n'.format(mass))
            file.write('start_step = {};\n'.format(initial_step))
            file.write('end_step = {};\n'.format(final_step))
            file.write('steps_vec = start_step:end_step;\n')
            file.write('xyzColID = [{} {}];\n\n'.format(x_col_id,y_col_id))
            file.write('basename = "{}";\n'.format(base_name))
            file.write('delimiter = ";";\n')
            file.write('headerlines = 4;')     
    
    def convert_input_2d(self):
        """ 
        Converte as saídas exportadas do DualSPHysics para o formato .csv
        usado no código Dilts 2D.
        """

        self.data_reader.current_step = -1
        while self.data_reader.current_step < self.data_reader.data_info['final_step']:
            particles,step = self.data_reader.get_next_step()
            
            # Escrece arquivo no formato .dat
            file_name = os.path.join(self.data_reader.data_dir,
                                     self.data_reader.data_info['frames_dir'],
                                     'pdata.'+str(step)+'.dat')
            file = open(file_name,'w')
            file.write(str(particles.shape[0])+'\n')
            np.savetxt(file,particles,fmt='%.12f')
            
            file.close()

    def convert_points_data(self,output_dir,output_extension):
        """ 
        Converte dados de pontos 3d para outro formato.
        """
        if os.path.exists(output_dir):
            print("Warning: output dir already exists!")
            return                        
        os.makedirs(output_dir,exist_ok=True)
        spatial_dimensions = self.data_reader.properties_info['dimensions']
        steps = self.data_reader.find_available_steps()
        if len(steps)==0:
            print("No prediction files found!")
            return             
        for step in tqdm(steps, desc='Converting files:'):
            particles = self.data_reader.get_step(step)
            
            frame_file = os.path.join(output_dir,f'pdata.{step}.{output_extension}')      
            if output_extension=='dat' or output_extension=='txt':                
                file = open(frame_file,'w')
                file.write(str(particles.shape[0])+'\n')
                np.savetxt(file,particles,fmt='%.12f')
                
                file.close()
            elif output_extension=='csv':
                    if spatial_dimensions==2:
                        columns = ['x','y']
                    elif spatial_dimensions==3:                        
                        columns = ['x','y','z']
                    df = pd.DataFrame(particles,columns=columns)
                    df.to_csv(frame_file,index=False,header=True)                

    def compute_normals_sph(self,gt_config_file,section,search_radius=2.0,
        use_only_boundary=False,initial_step=0,final_step=-1,skip_steps=1,
        enable_plot=False,pause=0.1,save=False,base_name='normal',extension='txt',
        output_dir=None, kernel_type='wendland'):
        """
        Calcula os vetores normais das partículas de fronteira usando operador de gradiente do método SPH.
        Última modificação: 10/11/2022.

        Args:
            gt_config_file:
            search_radius:
            use_only_boundary:
            initial_step:
            final_step:
            enable_plot:
            save:
            base_name:
            extension:
        """
        if save:
            if  output_dir==None:
                normal_dir = os.path.join(self.data_dir,'normal_sph')
            else:
                normal_dir = output_dir
            os.makedirs(normal_dir,exist_ok=True)

        point_distance = self.data_reader.properties_info['dp']
        kernel_length = self.data_reader.properties_info['h']
        spatial_dimensions = self.data_reader.properties_info['dimensions']
        hdp = kernel_length/point_distance
        
        #ref_length = point_distance
        real_search_radius = search_radius * kernel_length

        if initial_step==-1:
            initial_step = self.data_reader.data_info['initial_step']

        if final_step==-1:
            final_step = self.data_reader.data_info['final_step']

        steps = np.arange(initial_step,final_step+1,skip_steps)
        time_per_step = np.zeros(steps.shape[0])
        
        for k,step in enumerate(steps):
            print('Step',step)

            t = time.time()

            particles = self.data_reader.get_step(step)

            gt_labels = self.data_reader.get_step_labels(
                step,gt_config_file,section=section)
            gt_labels = gt_labels==1

            full_kdtree = KDTree(particles,device='cpu')
            _,dists = full_kdtree.query(particles,knn=2)

            normal = np.zeros(particles.shape)

            boundary_particles = particles[gt_labels]

            if use_only_boundary:
                boundary_kdtree = KDTree(boundary_particles,device='cpu')
                neighbors,_ = boundary_kdtree.query_radius(
                    boundary_particles,real_search_radius)
            else:
                neighbors,_ = full_kdtree.query_radius(
                    boundary_particles,real_search_radius)
            
            boundary_normal_sph = np.zeros(boundary_particles.shape)

            for i in range(neighbors.shape[0]):
                # Normal com operador de gradiente do método SPH
                if kernel_type=='wendland_quintic':
                    W = WendlandQuintic(dim=spatial_dimensions)
                elif kernel_type=='cubic_spline':
                    W = CubicSpline(dim=spatial_dimensions)
                
                ni = np.zeros(3,dtype=float)                
                if spatial_dimensions==2:
                    xi = xj = np.pad(boundary_particles[i],[0,1])
                    vol = np.pi*(0.5*kernel_length)**2
                elif spatial_dimensions==3:
                    vol = 4*np.pi*(0.5*kernel_length)**3/3
                    xi = boundary_particles[i]
                for xj in particles[neighbors[i]]:
                    if spatial_dimensions==2:
                        xj = np.pad(xj,[0,1])
                    grad_Wij = np.zeros(3)
                    W.gradient(xij=xj-xi,rij=np.linalg.norm(xj-xi),h=kernel_length,grad=grad_Wij)
                    ni += vol*grad_Wij*kernel_length
                if spatial_dimensions==2:
                    ni = ni[0:2]
                boundary_normal_sph[i] = ni/np.linalg.norm(ni)

            normal[gt_labels] = boundary_normal_sph

            time_per_step[k] = time.time()-t
            print(f' --> time: {time_per_step[k]} s')

            if enable_plot and spatial_dimensions==2:
                plt.cla()
                plt.scatter(particles[:,0],particles[:,1])
                X = boundary_particles[:,0]
                Y = boundary_particles[:,1]
                U = boundary_normal_sph[:,0]
                V = boundary_normal_sph[:,1]
                plt.quiver(X,Y,U,V)
                plt.axis('equal')
                plt.pause(pause)

            if save:
                normal_file = os.path.join(normal_dir,
                    f'{base_name}.{step}.{extension}')
                if extension=='npy':
                    np.save(normal_file,normal)
                elif extension=='txt':
                    np.savetxt(normal_file,normal,fmt='%.6f')
                elif extension=='csv':
                    if spatial_dimensions==2:
                        col_normal = ['nx','ny']
                        columns = ['label','x','y']+col_normal
                    elif spatial_dimensions==3:
                        col_normal = ['nx','ny','nz']
                        columns = ['label','x','y','z']+col_normal
                    array = np.concatenate(
                        [gt_labels[:,np.newaxis],particles,normal],axis=-1)
                    df = pd.DataFrame(array,columns=columns)
                    df.to_csv(normal_file,index=False,header=True)

        if save:
            times_file = os.path.join(normal_dir,'times_per_step.csv')
            times = np.concatenate(
                [np.array(steps)[:,np.newaxis],time_per_step[:,np.newaxis]],axis=-1)
            df_t = pd.DataFrame(times,columns=['step','time (s)'])
            df_t.to_csv(times_file,index=False,header=True)

            gt_config = configparser.ConfigParser()    
            gt_config.read(gt_config_file)

            if 'normal' not in gt_config:
                gt_config['normal'] = {
                    'dir': 'normal_sph',
                    'base_name': 'normal',
                    'extension':extension,
                    'columns':' '.join(col_normal),
                    'method':'sph',
                    'initial_distance':point_distance,
                    'search_radius':search_radius,
                    'comments':('search radius is a multiplicative factor of the kernel length'
                         'the true search radius is calculated as search_radius*kernel_length')}

                with open(gt_config_file,'w') as configfile:
                    gt_config.write(configfile)  


    def compute_normals_pca_1(self,gt_config_file,search_radius=2.0,
        initial_step=0,final_step=-1,enable_plot=False,save=False,
        base_name='normal',extension='txt',output_dir='normal_pca'):
        """
        Calcula os vetores normais das partículas de fronteira.
        Última modificação: 08/04/2022.

        Args:
            gt_config_file:
            search_radius:
            initial_step:
            final_step:
            enable_plot:
            save:
            base_name:
            extension:
        """
        if save:
            normal_dir = os.path.join(self.data_dir,output_dir)
            os.makedirs(normal_dir,exist_ok=True)

        point_distance = self.data_reader.properties_info['dp']
        kernel_length = self.data_reader.properties_info['h']
        spatial_dimensions = self.data_reader.properties_info['dimensions']
        hdp = kernel_length/point_distance

        if spatial_dimensions==2:
            hdp_suggested = np.sqrt(2)
        elif spatial_dimensions==3:
            hdp_suggested = np.sqrt(3)
        
        if abs(hdp-hdp_suggested)<1e-6:
            ref_length = kernel_length
        else:
            ref_length = hdp_suggested * point_distance
                
        real_search_radius = search_radius * ref_length

        if final_step==-1:
            final_step = self.data_reader.data_info['final_step']

        time_per_step = np.zeros(final_step+1)
        steps = range(initial_step,final_step+1)
        for step in steps:
            print('Step',step)

            t = time.time()

            particles = self.data_reader.get_step(step)

            gt_labels = self.data_reader.get_step_labels(
                step,gt_config_file,section='labels')
            gt_labels = gt_labels==1

            full_kdtree = KDTree(particles,device='cpu')

            normal = np.zeros(particles.shape)
            boundary_particles = particles[gt_labels]
            boundary_kdtree = KDTree(boundary_particles,device='cpu')
            neighbors,_ = boundary_kdtree.query_radius(
                boundary_particles,real_search_radius)
            
            boundary_normal = np.zeros(boundary_particles.shape)
            for i in range(neighbors.shape[0]):
                pca = PCA()
                pca.fit(boundary_particles[neighbors[i]])
                boundary_normal[i] = pca.components_[-1]

                p1 = boundary_particles[i] + real_search_radius*boundary_normal[i]
                p2 = boundary_particles[i] - real_search_radius*boundary_normal[i]

                n_p1p2,_ = full_kdtree.query_radius(
                    np.array([p1,p2]),real_search_radius)

                if len(n_p1p2[0]) > len(n_p1p2[1]):
                    boundary_normal[i] = -1*boundary_normal[i]

            normal[gt_labels] = boundary_normal

            time_per_step[step] = time.time()-t
            print(f' --> time: {time_per_step[step]} s')

            if enable_plot and spatial_dimensions==2:
                plt.cla()
                plt.scatter(particles[:,0],particles[:,1])
                X = boundary_particles[:,0]
                Y = boundary_particles[:,1]
                U = boundary_normal[:,0]
                V = boundary_normal[:,1]
                plt.quiver(X,Y,U,V)
                plt.pause(0.1)

            if save:
                normal_file = os.path.join(normal_dir,
                    f'{base_name}.{step}.{extension}')
                if extension=='npy':
                    np.save(normal_file,normal)
                elif extension=='txt':
                    np.savetxt(normal_file,normal,fmt='%.6f')
                elif extension=='csv':
                    if spatial_dimensions==2:
                        columns = ['label','x','y','nx','ny']
                    elif spatial_dimensions==3:
                        columns = ['label','x','y','z','nx','ny','nz']
                    array = np.concatenate(
                        [gt_labels[:,np.newaxis],particles,normal],axis=-1)
                    df = pd.DataFrame(array,columns=columns)
                    df.to_csv(normal_file,index=False,header=True)

        if save:
            times_file = os.path.join(normal_dir,'times_per_step.csv')
            times = np.concatenate(
                [np.array(steps)[:,np.newaxis],time_per_step[:,np.newaxis]],axis=-1)
            df_t = pd.DataFrame(times,columns=['step','time (s)'])
            df_t.to_csv(times_file,index=False,header=True)

    def generate_particle_labels_3d(self,gt_config_file,search_radius=2.0,
        initial_step=0,final_step=-1,enable_plot=False,save=False,
        base_name='normal',extension='txt',output_dir='normal_pca'):
        """ 
        Gera o ground-truth de rótulos de partículas de acordo o paper
        Boundary particle resampling for surface reconstruction in liquid animation (2019).
        Última modificação: 15/05/2022.

        Args:
            gt_config_file:
            search_radius:
            initial_step:
            final_step:
            enable_plot:
            save:
            base_name:
            extension:
        """
        if save:
            normal_dir = os.path.join(self.data_dir,output_dir)
            os.makedirs(normal_dir,exist_ok=True)

        point_distance = self.data_reader.properties_info['dp']
        kernel_length = self.data_reader.properties_info['h']
        spatial_dimensions = self.data_reader.properties_info['dimensions']
        hdp = kernel_length/point_distance

        if spatial_dimensions==2:
            hdp_suggested = np.sqrt(2)
        elif spatial_dimensions==3:
            hdp_suggested = np.sqrt(3)
        
        if abs(hdp-hdp_suggested)<1e-6:
            ref_length = kernel_length
        else:
            ref_length = hdp_suggested * point_distance
                
        real_search_radius = search_radius * ref_length

        if final_step==-1:
            final_step = self.data_reader.data_info['final_step']

        time_per_step = np.zeros(final_step+1)
        steps = range(initial_step,final_step+1)
        for step in steps:
            print('Step',step)

            t = time.time()

            particles = self.data_reader.get_step(step)

            gt_labels = self.data_reader.get_step_labels(
                step,gt_config_file,section='labels')
            gt_labels = gt_labels==1

            full_kdtree = KDTree(particles,device='cpu')

            normal = np.zeros(particles.shape)

            boundary_particles = particles[gt_labels]
            #particle_labels = 
            
            #boundary_kdtree = KDTree(boundary_particles,device='cpu')
            
            neighbors,_ = full_kdtree.query_radius(
                boundary_particles,real_search_radius)
            
            particle_labels = np.zeros((particles.shape[0],1))
            boundary_particle_labels = np.zeros((boundary_particles.shape[0],1))

            K = 4
            alpha = 0.2
            for i in range(neighbors.shape[0]):
                pca = PCA()
                pca.fit(particles[neighbors[i]])
                #boundary_normal[i] = pca.components_[-1]
                
                # Auto valores
                sigma1 = pca.singular_values_[2]
                sigma2 = pca.singular_values_[1]
                sigma3 = pca.singular_values_[0]

                if neighbors.shape[0]<K:
                    particle_labels[i] = 4
                else:
                    if sigma2 <= alpha*sigma3:
                        particle_labels[i] = 3
                    elif sigma1 <= alpha*sigma3 and sigma2 > alpha*sigma3:
                        particle_labels[i] = 2
                    elif sigma1 > alpha*sigma3:
                        particle_labels[i] = 1

            particle_labels[gt_labels] = boundary_particle_labels

            time_per_step[step] = time.time()-t
            print(f' --> time: {time_per_step[step]} s')

            if save:
                normal_file = os.path.join(normal_dir,
                    f'{base_name}.{step}.{extension}')
                if extension=='npy':
                    np.save(normal_file,normal)
                elif extension=='txt':
                    np.savetxt(normal_file,normal,fmt='%.6f')
                elif extension=='csv':
                    if spatial_dimensions==2:
                        columns = ['label','x','y','nx','ny']
                    elif spatial_dimensions==3:
                        columns = ['label','x','y','z','nx','ny','nz']
                    array = np.concatenate(
                        [gt_labels[:,np.newaxis],particles,normal],axis=-1)
                    df = pd.DataFrame(array,columns=columns)
                    df.to_csv(normal_file,index=False,header=True)

        if save:
            times_file = os.path.join(normal_dir,'times_per_step.csv')
            times = np.concatenate(
                [np.array(steps)[:,np.newaxis],time_per_step[:,np.newaxis]],axis=-1)
            df_t = pd.DataFrame(times,columns=['step','time (s)'])
            df_t.to_csv(times_file,index=False,header=True)

class DataProcessing:
    
    def __init__(self,config_file=None):
        """ 
        Construtor
        Última modificação: 31/11/2021.
        """
        if config_file is not None:
            self.config_file = config_file
            self.data_dir = os.path.dirname(config_file)
            self.data_reader = DataReader(config_file)

    def load_dataset(self,dataset_file,batch=None):
        """
        Carrega um dataset HDF5, detectando automaticamente o formato de
        armazenamento a partir do conteúdo do próprio arquivo:
        - formato denso simples (datasets 'images'/'labels'); ou
        - formato denso codificado a partir de voxels esparsos (datasets
          'full_voxels_coord'/'target_voxels_coord'/'target_labels').
        Unifica os antigos métodos `load_dataset_approach1` e
        `load_dataset_approach2`.
        Última modificação: 02/08/2026.

        Args:
            dataset_file: caminho do arquivo HDF5 do dataset.
            batch: índices (ou máscara) do batch a carregar.

        Return:
            (batch_images, batch_labels): batch_labels é `None` quando os
            rótulos já vêm codificados diretamente nos canais das imagens
            (formato esparso-denso).
        """
        with h5py.File(dataset_file,'r') as f:
            if 'images' in f:
                batch_images = f['images'][batch]
                batch_labels = f['labels'][batch]
                return batch_images,batch_labels

            if 'full_voxels_coord' in f:
                dense_shape = f.attrs['dense_shape']
                if batch is not None:
                    batch_images = np.zeros((batch.shape[0],) + tuple(dense_shape[1:-1]) + (3,))
                else:
                    batch_images = np.zeros(tuple(dense_shape[1:-1]) + (3,))

                # Canais das imagens
                channel_1_idx = self.split_array_into_parts(
                    array = f['full_voxels_coord'],
                    sizes = f['full_num_voxels']
                )
                channel_2_3_idx = self.split_array_into_parts(
                    array = f['target_voxels_coord'],
                    sizes = f['target_num_voxels']
                )
                channel_2_3_values =  self.split_array_into_parts(
                    array = f['target_labels'],
                    sizes = f['target_num_voxels']
                )
                # Pôe os valores nos canais das imagens
                for i in range(batch.shape[0]):
                    # Channel 1
                    batch_images[i,...,0].flat[channel_1_idx[batch[i]]] = 1
                    # Channel 2
                    batch_images[i,...,1].flat[channel_2_3_idx[batch[i]]] = (
                        (channel_2_3_values[batch[i]] == 0).astype(int)
                    )
                    # Channel 3
                    batch_images[i,...,2].flat[channel_2_3_idx[batch[i]]] = (
                        (channel_2_3_values[batch[i]] == 1).astype(int)
                    )
                # Os labels dos pixels já vão codificados nos canais 2 e 3
                # das imagens
                return batch_images,None

            raise ValueError(f"Unrecognized dataset format in {dataset_file}")

    def split_array_into_parts(self,array,sizes):
        """ 
        Divide um array em partes de acordo com os tamanhos informados.
        Usa numpy split.
        Última atualização: 17/02/2022.

        Args:
            array:
            sizes:

        Returns:
            array dividido.
        """
        split_indexes = np.cumsum(sizes)[0:-1]
        return np.split(array,split_indexes)

    def convert_to_numpy_dataset(self,dataset_file,format='dense'):
        """ 
        Converte dataset hdf5 para o formato de array do numpy.
        Última modificação: 16/02/2022.
        """
        with h5py.File(dataset_file,'r') as f:
            numpy_file = dataset_file.replace('.hdf5','.npz')
            if format=='sparse':
                np.savez(
                    numpy_file,
                    dense_shape = f.attrs['dense_shape'], 
                    image_res = f.attrs['image_res'],
                    border_res = f.attrs['border_res'],
                    num_classes = f.attrs['num_classes'],
                    description = f.attrs['description'],
                    full_num_voxels = f['full_num_voxels'], 
                    full_voxels_coord = f['full_voxels_coord'], 
                    target_labels = f['target_labels'], 
                    target_num_voxels = f['target_num_voxels'], 
                    target_voxels_coor = f['target_voxels_coord']
                )
            elif format=='dense':
                np.savez_compressed(
                    numpy_file,
                    image_res = f.attrs['image_res'],
                    num_classes = f.attrs['num_classes'],
                    description = f.attrs['description'],
                    images = f['images'], 
                    labels = f['labels']
                )

    def extract_times_from_logfile_dilts(self,log_file,save=False,plot=True):
        """
        Extrai tempos de execução do arquivo de log do Dilts.

        Args:
            log_file:
            save:
            plot:
        """
        steps = []
        times = []
        with open(log_file,'r') as file:
            for line in file:
                if 'Reading particle file' in line:
                    steps.append(int(line.split('.')[-5]))
                if 'Elapsed time' in line:
                    times.append(float(line.split(' ')[-2])/1000.0)

        steps = np.array(steps)
        times = np.array(times)

        idx_sort = np.argsort(steps)

        steps = steps[idx_sort]
        times = times[idx_sort]

        if save:
            times_file = os.path.join(os.path.dirname(log_file),'time_report.csv')
            df = pd.DataFrame(np.array([steps,times]).T,columns=['steps','times'])
            df.to_csv(times_file,index=False,header=True)

        if plot:
            plt.plot(steps,times)
            plt.title('Time per frame (Dilts)',fontdict={'fontsize': 12})
            plt.axis([0,max(steps),0,1.25*max(times)])
            plt.ylabel('time (s)')
            plt.xlabel('step')
            plt.show()

class Curves2D:

    def __init__(self):
        pass

    def f(self,x,t):
        fx = 10 + 0.4*x*np.cos(0.1*t+np.sin(t+x))
        dfdx = 0.4*np.cos(0.1*t + np.sin(t+x)) - 0.4*x*np.sin(0.1*t+np.sin(t+x))*np.cos(t+x)
        return fx,dfdx

    def generate_simulation(self,data_dir=None,steps=100,pause=1,enable_plot=True,save_results=False):
        """ 
        Gera uma simulação de teste. 
        Última atualização: 18/04/2022.

        Args:
            data_dir:
            steps:
            pause:
            enable_plot:
            save_results:       
        """        
        n = 100
        xmin = 0
        xmax = 6*np.pi
        n_points_curve = int(4.0*n)

        _x = np.linspace(xmin,xmax,n_points_curve)
        
        dp = (xmax-xmin)/(n-1)
        for t in range(steps):
            print(f'Step {t}')
            _y,_dydx = self.f(_x,t)
            
            points = np.array([_x,_y]).T
            
            remove = np.zeros(n_points_curve,dtype=np.bool)            
            i = 0
            j = 1
            while j < n_points_curve-1:
                dist = np.sqrt(np.sum((points[i] - points[j])**2))
                if dist<0.8*dp:
                    remove[j] = True
                    j += 1
                else:                    
                    i = j
                    j += 1

            x = _x[remove==False]
            y = _y[remove==False]
            dydx = _dydx[remove==False]

            length = np.sqrt(dydx**2+1)
            normal_curve =  np.array(
                [-dydx/length,np.ones(length.shape[0])/length]).T
            
            dx = np.linspace(xmin,xmax,n)
            xx,yy = np.meshgrid(dx,dx)
            xx = xx.flatten()
            yy = yy.flatten()

            yy_curve,_ = self.f(xx,t)
            target = (yy_curve - yy) > dp*0.3

            xx = xx[target]
            yy = yy[target] 

            # Ground-truth de fronteira
            gt = np.zeros(xx.shape[0])
            gt[xx==xmin] = 1
            gt[xx==xmax] = 1
            gt[yy==0] = 1

            # Normais das laterais e do fundo
            normal = np.zeros((xx.shape[0],2))
            normal[xx==0] = [-1,0]
            normal[xx==max(x)] = [1,0]
            normal[yy==0] = [0,-1]

            # Normais nas quinas do fundo
            normal[np.logical_and(xx==0,yy==0)] = [-1/np.sqrt(2),-1/np.sqrt(2)]
            normal[np.logical_and(xx==max(x),yy==0)] = [1/np.sqrt(2),-1/np.sqrt(2)]

            xx = np.concatenate([xx,x])
            yy = np.concatenate([yy,y])
            gt = np.concatenate([gt,np.ones(x.shape[0])])
            normal = np.concatenate([normal,normal_curve])
            
            # Normais nas quinas da superfície
            #

            xx[gt==0] = xx[gt==0] + 0.3*dp*(np.random.random(xx[gt==0].shape[0])-0.5)
            yy[gt==0] = yy[gt==0] + 0.3*dp*(np.random.random(yy[gt==0].shape[0])-0.5)

            if enable_plot:
                plt.cla()
                plt.quiver(xx,yy,normal[:,0],normal[:,1],
                    color='r',scale_units='inches',scale=2)
                plt.scatter(xx,yy)
                plt.scatter(xx[gt==1],yy[gt==1])
                plt.plot(x,y,'b')
                #_ = plt.axis('equal')
                plt.xlim([-2,max(x)+2])
                plt.ylim([-2,max(x)+2])
                plt.pause(pause)

            if save_results:
                sim_name = data_dir.split(os.sep)[-1]
                # Frames
                frames_dir = os.path.join(data_dir,'frames')
                os.makedirs(frames_dir,exist_ok=True)                
                frame_file = os.path.join(frames_dir,f'points.{t}.txt')
                points = np.array([xx,yy]).T
                np.savetxt(frame_file,points)

                # Ground-truth
                # Fronteira
                gt_dir = os.path.join(data_dir,'gt')
                os.makedirs(gt_dir,exist_ok=True)                
                gt_file = os.path.join(gt_dir,f'gt.{t}.txt')
                np.savetxt(gt_file,gt,fmt='%d')

                # Normais
                normal_dir = os.path.join(data_dir,'normal')
                os.makedirs(normal_dir,exist_ok=True)
                normal_file = os.path.join(normal_dir,f'normal.{t}.txt')
                np.savetxt(normal_file,normal)

        if save_results:
            # sim config
            sim_config = configparser.ConfigParser()                        
            sim_config['simulation_properties'] = {
                'dp':dp,
                'h':np.sqrt(2)*dp,
                'mass':0,
                'dimensions':2,
                'limits':' '.join(
                    [str(x) for x in [xmin,xmax,0,xmax]])}
            sim_config['data'] = {
                'sim_name': sim_name,
                'from': 'Curves2D',
                'frames_dir': 'frames',
                'base_name': 'points',
                'extension': 'txt',
                'initial_step': 0,
                'final_step': steps-1,
                'x_coord': '',
                'y_coord': '',
                'z_coord': ''}              
            sim_config_file =  os.path.join(data_dir,'sim_config.ini')
            with open(sim_config_file,'w') as configfile:
                sim_config.write(configfile)

            # gt config
            gt_config = configparser.ConfigParser()                        
            gt_config['labels'] = {
                'names':'interior boundary',
                'dir':'gt',
                'base_name':'gt',
                'extension':'txt'}
            gt_config['normal'] = {
                'dir':'normal',
                'base_name':'normal',
                'extension':'txt'}

            gt_config_file =  os.path.join(data_dir,'gt_config.ini')
            with open(gt_config_file,'w') as configfile:
                gt_config.write(configfile)
