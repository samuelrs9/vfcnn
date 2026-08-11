import os 
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sim_reader.data import DataReader
from sim_reader.config import ConfigReader
from metrics.classification import Report

from losses.custom_losses import *
from voxelizer.sparse_voxelizer import SparseVoxelizer

try:
    from tf_kdtree.neighbors import KDTree
except:
    print('Tf KDTree não foi carregada corretamente!')

class Reports:
    
    def __init__(self,data_reader=None):
        """ 
        Construtor.
        """
        if data_reader != None:
            self.data_reader = data_reader
        
    def lost_and_removed_particles(self,gt_config_file=None,pred_config_file=None):
        """ 
        Carrega as medidas de um teste grosseiro e calcula as quantidades de 
        partículas de interior removidas e de partículas de fronteira perdidas 
        usando um determinado threshold.
        Última modificação: 27/08/2021.
        
        Args:
            gt_config_file:
            pred_config_file:
        """        
        gt_config = ConfigReader(gt_config_file)
        gt_config_dict = gt_config.get_labels_config()
        
        pred_config = ConfigReader(pred_config_file)
        pred_general = pred_config.get_section(
            'general',['pred_sections','measure_sections']
        )
        pred_section = pred_general['pred_sections']
        measure_sections = pred_general['measure_sections']

        pred_config_dict = pred_config.get_section(pred_section,['dir','extension'])

        #view = Visualization(self.data_reader)

        # Teste da densidade
        density_threshold_1 = 0.93
        lost_particles_ratio_1 = np.zeros(self.data_reader.data_info['final_step']+1)
        removed_particles_ratio_1 = np.zeros(self.data_reader.data_info['final_step']+1)
        
        # Teste do centroide
        centroid_threshold_1 = 0.02
        lost_particles_ratio_2 = np.zeros(self.data_reader.data_info['final_step']+1)
        removed_particles_ratio_2 = np.zeros(self.data_reader.data_info['final_step']+1)
        
        # Teste combinado 1
        density_threshold_2 = 0.82
        centroid_threshold_2 = 0.14              
        lost_particles_ratio_3 = np.zeros(self.data_reader.data_info['final_step']+1)
        removed_particles_ratio_3 = np.zeros(self.data_reader.data_info['final_step']+1)
        
        # Teste de combinado 2
        combined_threshold = 0.77 
        lost_particles_ratio_4 = np.zeros(self.data_reader.data_info['final_step']+1)
        removed_particles_ratio_4 = np.zeros(self.data_reader.data_info['final_step']+1)                            

        self.data_reader.current_step = -1
        
        while self.data_reader.current_step < self.data_reader.data_info['final_step']:
            # Current step
            particles,step = self.data_reader.get_next_step()
            print('step {}'.format(step))
            
            # ground-truth
            gt_labels = self.data_reader.get_step_labels(step,gt_config_file)

            # predição
            #pred_labels = self.data_reader.get_step_labels_config(step,pred_config_file)


            # Carrega as medidas da predição
            #pred_density = self.data_reader.get_step_labels(pred_config_dict['dir'],
            #                                          'pred.density',step,
            #                                          pred_config_dict['extension'])      

            #pred_centroid = self.data_reader.get_step_labels(pred_config_dict['dir'],
            #                                          'pred.centroid',step,
            #                                          pred_config_dict['extension'])            
            try:
                density = self.data_reader.get_step_measures(
                    step, pred_config_file, section='density'
                )
            except:
                density = None
                print('Warning: density not found!')

            try:
                centroid_distances = self.data_reader.get_step_measures(
                    step, pred_config_file, section='centroid_distances'
                )
            except:
                centroid_distances = None
                print('Warning: centroid_distances not found!')
            
            #lost_labels = np.logical_and(gt_labels==1,pred_labels==0)          

            if density is not None:
                # Teste da densidade
                pred_density = density < density_threshold_1
                
                pred_density_lost = np.logical_and(gt_labels==1,pred_density==0)
                
                lost_particles_ratio_1[step] = pred_density_lost.sum()/(gt_labels==1).sum()
                removed_particles_ratio_1[step] = (pred_density==0).sum()/pred_density.shape[0]
                
                print('\nTeste da densidade')
                print(' --> lost: {:.4f}%'.format(100*lost_particles_ratio_1[step]))
                print(' --> removed: {:.4f}%'.format(100*removed_particles_ratio_1[step])) 
                            
            if centroid_distances is not None:
                # Teste do centroide
                pred_centroid = centroid_distances > centroid_threshold_1
                
                pred_centroid_lost = np.logical_and(gt_labels==1,pred_centroid==0)
                
                lost_particles_ratio_2[step] = pred_centroid_lost.sum()/(gt_labels==1).sum()
                removed_particles_ratio_2[step] = (pred_centroid==0).sum()/pred_centroid.shape[0]
                            
                print('\nTeste do centroide')
                print(' --> lost: {:.4f}%'.format(100*lost_particles_ratio_2[step]))
                print(' --> removed: {:.4f}%'.format(100*removed_particles_ratio_2[step])) 
                
            if centroid_distances is not None and density is not None:    
                # Teste combinado 1 """
                pred_density_2 = density < density_threshold_2
                pred_centroid_2 = centroid_distances > centroid_threshold_2            
                
                pred_combined_1 = np.logical_or(pred_density_2,pred_centroid_2)                        
                
                pred_combined_lost_1 = np.logical_and(gt_labels==1,pred_combined_1==0)
                
                lost_particles_ratio_3[step] = pred_combined_lost_1.sum()/(gt_labels==1).sum()
                removed_particles_ratio_3[step] = (pred_combined_1==0).sum()/pred_combined_1.shape[0]            
                
                print('\nTeste combinado 1')
                print(' --> lost: {:.4f}%'.format(100*lost_particles_ratio_3[step]))
                print(' --> removed: {:.4f}%'.format(100*removed_particles_ratio_3[step]))                    
            
                # Teste combinado 2 """
                pred_combined_2 = (density * (1-centroid_distances)) < combined_threshold
                
                pred_combined_lost_2 = np.logical_and(gt_labels==1,pred_combined_2==0)

                lost_particles_ratio_4[step] = pred_combined_lost_2.sum()/(gt_labels==1).sum()
                removed_particles_ratio_4[step] = (pred_combined_2==0).sum()/pred_combined_2.shape[0]            
                
                print('\nTeste combinado 2')
                print(' --> lost: {:.4f}%'.format(100*lost_particles_ratio_4[step]))
                print(' --> removed: {:.4f}%'.format(100*removed_particles_ratio_4[step]))                    
            
            print()            
            
            #view.scatter(particles,labels=lost_labels,clear_axes=True)
            
            #print('lost particles: {}'.format(lost_labels.sum()))
        
        print('\nMEANS')

        print('\nTeste da densidade')
        print(' --> mean lost: {:.8f}%'.format(100*lost_particles_ratio_1.mean()))
        print(' --> mean removed: {:.8f}%'.format(100*removed_particles_ratio_1.mean()))
              
        print('\nTeste do centroide')
        print(' --> mean lost: {:.8f}%'.format(100*lost_particles_ratio_2.mean()))
        print(' --> mean removed: {:.8f}%'.format(100*removed_particles_ratio_2.mean()))                       
    
        print('\nTeste combinado 1')
        print(' --> mean lost: {:.8f}%'.format(100*lost_particles_ratio_3.mean()))
        print(' --> mean removed: {:.8f}%'.format(100*removed_particles_ratio_3.mean()))         
        
        print('\nTeste combinado 2')
        print(' --> mean lost: {:.8f}%'.format(100*lost_particles_ratio_4.mean()))
        print(' --> mean removed: {:.8f}%'.format(100*removed_particles_ratio_4.mean()))         
        
        # Plota resultados
        fig1, ax1 = plt.subplots(figsize=(10,6))
        ax1.plot(100*lost_particles_ratio_1,label='Teste da densidade',linewidth=2)
        ax1.plot(100*lost_particles_ratio_2,label='Teste do centroide',linewidth=2)
        ax1.plot(100*lost_particles_ratio_3,label='Teste combinado 1',linewidth=2)
        ax1.plot(100*lost_particles_ratio_4,label='Teste combinado 2',linewidth=2)        
        ax1.set_xlabel("frame",fontsize=12)    
        ax1.set_ylabel("percentage (%)",fontsize=12)
        ax1.legend(fontsize=15)
        ax1.set_title('Percentage of lost boundary particles',fontsize=15)
        ax1.grid(True)
        
        fig2, ax2 = plt.subplots(figsize=(10,6))
        ax2.plot(100*removed_particles_ratio_1,label='Teste da densidade',linewidth=2)
        ax2.plot(100*removed_particles_ratio_2,label='Teste do centroide',linewidth=2)
        ax2.plot(100*removed_particles_ratio_3,label='Teste combinado 1',linewidth=2)
        ax2.plot(100*removed_particles_ratio_4,label='Teste combinado 2',linewidth=2)              
        ax2.set_xlabel("frame",fontsize=12)    
        ax2.set_ylabel("percentage (%)",fontsize=12)
        ax2.legend(fontsize=15)
        ax2.set_title('Percentage of removed interior particles',fontsize=15)
        ax2.grid(True)
        
        return {'lost_particles': [lost_particles_ratio_1,lost_particles_ratio_2,lost_particles_ratio_3,lost_particles_ratio_4],
                'removed_particles': [removed_particles_ratio_1,removed_particles_ratio_2,removed_particles_ratio_3,removed_particles_ratio_4]}

    def ratio_sph_kernel_and_distance_particles(self, skip=10):
        """ 
        Razão entre o parâmetro h da simulação SPH e a distância média entre partículas mais próximas.
        Última modificação: 21/08/2021.
        
        Args:
            
        """
        mean_distances_file = os.path.join(self.data_reader.data_dir,'mean_distances.csv')

        if not os.path.exists(mean_distances_file):
            steps = []
            ratios = []
            mean_distances = []
            
            current_step = self.data_reader.data_info['initial_step']-1
            while current_step < self.data_reader.data_info['final_step']:
                current_step += 1
                if current_step % skip != 0:
                    continue
                # Current step
                particles = self.data_reader.get_step(current_step)
                print('Step {}'.format(current_step))
                
                kdtree = KDTree(particles,device='cpu')            
                _,dists = kdtree.query(particles,knn=2)
                
                steps.append(current_step)
                mean_distances.append(np.mean(dists[:,1]))
                ratios.append(self.data_reader.properties_info['h']/mean_distances[-1])

                df = pd.DataFrame(np.array([steps, ratios, mean_distances]).T, columns=['steps', 'ratios', 'mean_distances'])
                df = df.to_csv(mean_distances_file, index=False, sep=';')
    
        df = pd.read_csv(mean_distances_file, sep=';')
        plt.plot(df['steps'],df['ratios'],'b-',linewidth=3)
        plt.ylim([0,1.1*df['ratios'].max()])
        plt.show()
            
    def load_analyze_coarse_test_results(self,data_dir,results_dir,
        radius_id=2,lost_id=3,test_id=1):
        """ 
        Carrega resultados de teste grosseiros de diferentes simulações.
        Última modificação: 22/02/2022.
        
        Args:
            data_dir:
            results_dir:
            radius_id:
            lost_id:
            test_id:
            
        """
        tests = [
            'density_test',
            'centroid_test',
            'combined_test_1',
            'combined_test_2'] 
        
        results = np.empty(len(data_dir),dtype=object) 
        
        removed = np.zeros(len(data_dir))        
        thresholds = np.empty(len(data_dir),dtype=object)
        
        density_threshold_r = np.zeros(len(data_dir))
        centroid_threshold_r = np.zeros(len(data_dir))
        combined_threshold_r = np.zeros(len(data_dir))
        
        for i in range(len(data_dir)):
            results_file = os.path.join(
                data_dir[i],results_dir,'results.npz')                        
            results = dict(np.load(results_file,allow_pickle=True))
            
            lost = results['lost_particles'][lost_id]
            removed[i] = (
                results['removed_particles']
                [0][tests[test_id]][radius_id,lost_id])
            thresholds[i] = (
                list(results['best_thresholds']
                [0][tests[test_id]][radius_id,lost_id].values()))
            
            print('\nSim: ',data_dir[i])
            print(f' --> mean lost particles: {100*lost}%')
            print(f' --> mean removed particles: {100*removed[i]:.2f}%')
            if tests[test_id] == 'density_test':
                print(' --> density threshold: ',thresholds[i][0])
                density_threshold_r[i] = thresholds[i][0].mean()
            if tests[test_id] == 'centroid_test':
                print(' --> centroid threshold: ',thresholds[i][0])
                centroid_threshold_r[i] = thresholds[i][0].mean()
            if tests[test_id] == 'combined_test_1':
                print(' --> density threshold: ',thresholds[i][0])
                print(' --> centroid threshold: ',thresholds[i][1])
                density_threshold_r[i] = thresholds[i][0].mean()
                centroid_threshold_r[i] = thresholds[i][1].mean()
            elif tests[test_id] == 'combined_test_2': 
                print(' --> combined threshold: ',thresholds[i][0])
                combined_threshold_r[i] = thresholds[i][0].mean()
            
        # Thresholds recomendados
        print('\nRecommended threshold for ',end='')
        if tests[test_id] == 'density_test':
            print('density test: ',density_threshold_r.mean().round(2))
        if tests[test_id] == 'centroid_test':
            print('centroid test: ',centroid_threshold_r.mean().round(2))
        if tests[test_id] == 'combined_test_1':
            print('combined test 1: ')
            print(' --> density threshold: ',density_threshold_r.mean().round(2))
            print(' --> centroid threshold: ',centroid_threshold_r.mean().round(2))
        elif tests[test_id] == 'combined_test_2': 
            print('combined test 2: ',combined_threshold_r.mean().round(2))
        
    def analyze_coarse_test_parameters(self,gt_config_file=None,pred_config_files=None,
        output_dir=None,worst_acc_lost = None):
        """ 
        Analisa os parâmetros do teste grosseiro.
        Última modificação: 22/02/2021.
        
        Args:
            gt_config_file:
            pred_config_files:
            output_dir:

        """        
        
        # Carrega as acurácias de predições dos thresholds
        acc_thresholds_file = os.path.join(output_dir,'acc.thresholds.npz')
        if os.path.exists(acc_thresholds_file):
            acc_thresholds = dict(np.load(acc_thresholds_file, allow_pickle=True))
        else:
            print("Warning: thresholds accuracy file don't exists!")

        combined_threshold = acc_thresholds['combined_threshold'] #np.linspace(0.4,1.0,50)
        density_threshold = acc_thresholds['density_threshold']
        distance_threshold = acc_thresholds['distance_threshold']
        
        all_removed_particles_density = acc_thresholds['all_removed_particles_ratio_density']
        all_lost_particles_density = acc_thresholds['all_lost_particles_ratio_density']
        
        all_removed_particles_centroid = acc_thresholds['all_removed_particles_ratio_centroid']
        all_lost_particles_centroid = acc_thresholds['all_lost_particles_ratio_centroid']
        
        all_removed_particles_comb1 = acc_thresholds['all_removed_particles_ratio_comb1']
        all_lost_particles_comb1 = acc_thresholds['all_lost_particles_ratio_comb1']     

        all_removed_particles_comb2 = acc_thresholds['all_removed_particles_ratio_comb2']
        all_lost_particles_comb2 = acc_thresholds['all_lost_particles_ratio_comb2']  
        
        del acc_thresholds
        
        num_predictions = len(pred_config_files)

        if worst_acc_lost==None:     
            worst_acc_lost = np.asarray([0,0.00005,0.0001,0.0005,0.001,0.005,0.01])
        
        plt.close('all')        
        
        search_radius = []

        # Configurações das predições
        for pred_config_file in pred_config_files:
            pred_config = ConfigReader(pred_config_file)
            pred_general = pred_config.get_section('general',['search_radius'])
            search_radius.append(float(pred_general['search_radius']))

        # Teste da densidade
        best_mean_removed_1 = -1*np.ones(
            (num_predictions,len(worst_acc_lost))) 
        best_thresholds_1 = np.empty(
            (num_predictions,len(worst_acc_lost)), dtype=object)
        
        print('Teste da densidade')
        for i in range(0,num_predictions):
            print(' --> {}:'.format(pred_config_files[i]))
            for j in range(len(worst_acc_lost)):  
                # Encontra as combinações de thresholds que satisfazem a condição
                # de que a média de partículas perdidas seja igual ou inferior a worst_acc_lost[j]
                best_ind_thresholds_1 = (
                    all_lost_particles_density[i].mean(axis=0) <= worst_acc_lost[j])
                
                if best_ind_thresholds_1.sum()>0:             
                    # Das combinações acima quais são as melhores no sentido de remover
                    # a maior quantidade média de partículas por frame
                    best_mean_removed_1[i,j] = (
                        all_removed_particles_density[i].mean(axis=0).flat[best_ind_thresholds_1.flatten()].max())
                    ind_best_mean_removed_1 = (
                        all_removed_particles_density[i].mean(axis=0) == best_mean_removed_1[i,j])                    
                    ind_dens_1 = np.where(ind_best_mean_removed_1)
                    best_thresholds_1[i,j] = {
                        'density_threshold':density_threshold[ind_dens_1].round(4)}  
                
        # Teste da distancia para o centroide
        best_mean_removed_2 = -1*np.ones(
            (num_predictions,len(worst_acc_lost))) 
        best_thresholds_2 = np.empty(
            (num_predictions,len(worst_acc_lost)),dtype=object)        
        print('Teste do centroide')
        for i in range(0,num_predictions):
            print(' --> {}:'.format(pred_config_files[i]))                     
            for j in range(len(worst_acc_lost)):  
                # Encontra as combinações de thresholds que satisfazem a condição
                # de que a média de partículas perdidas seja igual ou inferior a worst_acc_lost[j]
                best_ind_thresholds_2 = (
                    all_lost_particles_centroid[i].mean(axis=0) <= worst_acc_lost[j])                
                if best_ind_thresholds_2.sum()>0:             
                    # Das combinações acima quais são as melhores no sentido de remover
                    # a maior quantidade média de partículas por frame
                    best_mean_removed_2[i,j] = (
                        all_removed_particles_centroid[i].mean(axis=0).flat[best_ind_thresholds_2.flatten()].max())
                    ind_best_mean_removed_2 = (
                        all_removed_particles_centroid[i].mean(axis=0) == best_mean_removed_2[i,j])                    
                    ind_dist_2 = np.where(ind_best_mean_removed_2)
                    best_thresholds_2[i,j] = {
                        'distance_threshold':distance_threshold[ind_dist_2].round(4)}        

        # Teste combinado 1
        best_mean_removed_3 = -1*np.ones(
            (num_predictions,len(worst_acc_lost))) 
        best_thresholds_3 = np.empty(
            (num_predictions,len(worst_acc_lost)),dtype=object)        
        # fig = np.empty(num_predictions,dtype=object)
        # axis = np.empty(num_predictions,dtype=object)
        print('Teste combinado 1')
        for i in range(0,num_predictions):
            print(' --> {}:'.format(pred_config_files[i]))
                     
            for j in range(len(worst_acc_lost)):  
                # Encontra as combinações de thresholds que satisfazem a condição
                # de que a média de partículas perdidas seja igual ou inferior a worst_acc_lost[j]
                best_ind_thresholds_3 = (
                    all_lost_particles_comb1[i].mean(axis=0) <= worst_acc_lost[j]
                )                
                if best_ind_thresholds_3.sum()>0:             
                    # Das combinações acima quais são as melhores no sentido de remover
                    # a maior quantidade média de partículas por frame
                    best_mean_removed_3[i,j] = (
                        all_removed_particles_comb1[i].mean(axis=0).flat[best_ind_thresholds_3.flatten()].max())
                    ind_best_mean_removed_3 = (
                        all_removed_particles_comb1[i].mean(axis=0) == best_mean_removed_3[i,j])                    
                    ind_dist_3,ind_dens_3 = np.where(ind_best_mean_removed_3)
                    best_thresholds_3[i,j] = {
                        'density_threshold':density_threshold[ind_dens_3].round(4),'distance_threshold':distance_threshold[ind_dist_3].round(4)}
            
            # fig[i],axis[i] = plt.subplots(2,1,figsize=(10,10))
            
            # plt.figure(fig[i].number)
                        
            # axis[i][0].clear()
            # pcr = axis[i][0].pcolor(density_threshold,distance_threshold,all_removed_particles_ratio_3[i].min(axis=0))
            # fig[i].colorbar(pcr,ax=axis[i][0])
            # axis[i][0].grid(True)
            # axis[i][0].set_xlabel("density threshold",fontsize=12)
            # axis[i][0].set_ylabel("distance threshold",fontsize=12)
            # axis[i][0].set_title('removed particles ratio',fontsize=15)
            
            # axis[i][1].clear()
            # pcl = axis[i][1].pcolor(density_threshold,distance_threshold,all_lost_particles_ratio_3[i].max(axis=0))
            # fig[i].colorbar(pcl,ax=axis[i][1])
            # axis[i][1].grid(True)
            # axis[i][1].set_xlabel("density threshold",fontsize=12)
            # axis[i][1].set_ylabel("distance threshold",fontsize=12)
            # axis[i][1].set_title('removed particles ratio',fontsize=15)
            
            # fig[i].suptitle(pred_config_files[i])
            
            # plt.pause(0.1)
        
        
        # Teste combinado 2
        best_mean_removed_4 = -1*np.ones(
            (num_predictions,len(worst_acc_lost))
        ) 
        best_thresholds_4 = np.empty(
            (num_predictions,len(worst_acc_lost)),dtype=object
        )        
        print('Teste combinado 2')
        for i in range(0,num_predictions):
            print(' --> {}:'.format(pred_config_files[i]))
                     
            for j in range(len(worst_acc_lost)):  
                # Encontra as combinações de thresholds que satisfazem a condição
                # de que a média de partículas perdidas seja igual ou inferior a worst_acc_lost[j]
                best_ind_thresholds_4 = (
                    all_lost_particles_comb2[i].mean(axis=0) <= worst_acc_lost[j]
                )                
                if best_ind_thresholds_4.sum()>0:             
                    # Das combinações acima quais são as melhores no sentido de remover
                    # a maior quantidade média de partículas por frame
                    best_mean_removed_4[i,j] = (
                        all_removed_particles_comb2[i].mean(axis=0).flat[best_ind_thresholds_4.flatten()].max()
                    )
                    ind_best_mean_removed_4 = (
                        all_removed_particles_comb2[i].mean(axis=0) == best_mean_removed_4[i,j]
                    )                    
                    ind_comb_4 = np.where(ind_best_mean_removed_4)
                    best_thresholds_4[i,j] = {
                        'combined_threshold':combined_threshold[ind_comb_4].round(4)
                    }
                                                     
        print('Done!')
        
        # Salva as saídas
        results_output_file = os.path.join(output_dir,f'results.npz')
        
        np.savez(
            results_output_file,
            description = (
                "Rows in 'test removed' arrays represent the",
                "search radius and columns represent the lost particles."
            ),
            search_radius = search_radius,
            lost_particles = worst_acc_lost,            
            removed_particles = [{
                'density_test':best_mean_removed_1,
                'centroid_test':best_mean_removed_2,
                'combined_test_1':best_mean_removed_3,
                'combined_test_2':best_mean_removed_4
            }],
            best_thresholds = [{
                'density_test':best_thresholds_1,
                'centroid_test':best_thresholds_2,
                'combined_test_1':best_thresholds_3,
                'combined_test_2':best_thresholds_4
            }]
        )

        return {
            'rows': [
                'density_test','centroid_test',
                'combined_test_1','combined_test_2'
            ],
            'search_radius': search_radius,
            'lost_particle_threshold': worst_acc_lost,
            'best_mean_removed_particles': [
                best_mean_removed_1,best_mean_removed_2,
                best_mean_removed_3,best_mean_removed_4
            ],
            'best_thresholds': [
                best_thresholds_1,best_thresholds_2,
                best_thresholds_3,best_thresholds_4
            ]
        }
                
    def process_coarse_test_outputs(self,gt_config_file=None,pred_config_files=None,output_dir=None):
        """ 
        Processa as saídas do teste grosseiro.
        Última modificação: 21/02/2022.
        
        Args:
            gt_config_file:
            pred_config_files:
            gt_band_config_file:
        """                       
        process_output_file = os.path.join(output_dir,f'processed.outputs.npz')

        if os.path.exists(process_output_file):
            print('Warning: processed output file already exists!')
            return

        num_predictions = len(pred_config_files)        
        
        #gt_config = ConfigReader(gt_config_file)
        #gt_config_dict = gt_config.get_labels_config()

        #gt_band_config = ConfigReader(gt_band_config_file)

        #view = Visualization(self.data_reader)
        
        all_names = np.empty(num_predictions,dtype=object)
        all_radius = np.zeros(num_predictions)
        
        #particles = self.data_reader.get_step(0)
        #num_particles = particles.shape[0]
        
        num_steps = self.data_reader.data_info['final_step']+1

        all_densities = np.empty((num_predictions,num_steps),dtype=object)
        all_count_neighbors = np.empty((num_predictions,num_steps),dtype=object)
        all_centroid_distances = np.empty((num_predictions,num_steps),dtype=object)

        all_mean_count_gt_boundary = np.zeros((num_predictions,num_steps))
        all_mean_count_interior = np.zeros((num_predictions,num_steps))

        all_mean_density_gt_boundary = np.zeros((num_predictions,num_steps))
        all_mean_density_interior = np.zeros((num_predictions,num_steps))

        all_mean_distances_gt_boundary = np.zeros((num_predictions,num_steps))
        all_mean_distances_interior = np.zeros((num_predictions,num_steps))
        
        initial_step = 0
        final_step = self.data_reader.data_info['final_step']
        for step in tqdm(range(initial_step,final_step+1)):
            #print('step: ',step)
            # ground-truth
            gt_labels = self.data_reader.get_step_labels(step,gt_config_file,section='boundary')

            mean_density_gt_boundary = np.zeros((num_predictions))
            mean_density_interior = np.zeros((num_predictions))

            mean_count_gt_boundary = np.zeros((num_predictions))
            mean_count_interior = np.zeros((num_predictions))

            mean_distances_gt_boundary = np.zeros((num_predictions))
            mean_distances_interior = np.zeros((num_predictions))
            
            # Itera sobre as predições grosseiras
            for i in range(num_predictions):
                pred_config_file = pred_config_files[i]
                
                all_names[i] = pred_config_file.split(os.sep)[-2]
                
                other_pred_time_file = os.path.join(
                    os.path.dirname(pred_config_file),'other_outputs.npz')
                other_outputs = np.load(other_pred_time_file)          
                
                all_radius[i] = float(other_outputs['search_radius'])
    
                # density
                try:
                    density = self.data_reader.get_step_measures(
                        step, pred_config_file, section='density'
                    )
                except:
                    density = None
                    print('Warning: density not found!')
                all_densities[i,step] = density

                # centroid distances
                try:
                    centroid_distances = self.data_reader.get_step_measures(
                        step, pred_config_file, section='centroid_distances'
                    )
                except:
                    centroid_distances = None
                    print('Warning: centroid_distances not found!')
                all_centroid_distances[i,step] = centroid_distances

                # count neighbors
                try:
                    count_neighbors = self.data_reader.get_step_measures(
                        step, pred_config_file, section='count_neighbors'
                    )
                except:
                    count_neighbors = None
                    print('Warning: count_neighbors not found!')
                all_count_neighbors[i,step] = count_neighbors

                # Partículas próximas da fronteira segundo o groundtruth
                try:
                    gt_band_labels = self.data_reader.get_step_labels(
                        step,pred_config_file,section='pred_gt_band'
                    )
                except:
                    print('Warning: gt_band not found!')
                
                # density
                density_gt_bound = density[gt_labels==1]
                density_interior = density[gt_band_labels==0]

                mean_density_gt_boundary[i] = density_gt_bound.mean()
                mean_density_interior[i] = density_interior.mean()

                # Count 
                count_gt_boundary = count_neighbors[gt_labels==1]
                count_interior = count_neighbors[gt_band_labels==0]

                mean_count_gt_boundary[i] = count_gt_boundary.mean()
                mean_count_interior[i] = count_interior.mean()
                
                # distances
                distances_gt_boundary = centroid_distances[gt_labels==1]
                distances_interior = centroid_distances[gt_band_labels==0]                

                mean_distances_gt_boundary[i] = distances_gt_boundary.mean()
                mean_distances_interior[i] = distances_interior.mean()           
                            
            # CONTINUAR DAQUI
            all_mean_density_gt_boundary[:,step] = mean_density_gt_boundary
            all_mean_density_interior[:,step] = mean_density_interior

            all_mean_count_gt_boundary[:,step] = mean_count_gt_boundary
            all_mean_count_interior[:,step] = mean_count_interior          
    
            all_mean_distances_gt_boundary[:,step] = mean_distances_gt_boundary
            all_mean_distances_interior[:,step] = mean_distances_interior
                        
            # print('--> (mean count)/area for interior particles: \n',all_mean_density_interior[:,step].round(4))
            # print('--> (mean count)/area for boundary particles: \n',all_mean_density_gt_boundary[:,step].round(4))
            
            # print('--> mean centroid distances for interior particles: \n',all_mean_distances_interior[:,step].round(4))
            # print('--> mean centroid distances for boundary particles: \n',all_mean_distances_gt_boundary[:,step].round(4))
            # print('\n')        
            
        
        print('\nMEAN COUNT')
        print('--> interior particles: ',all_mean_count_interior.mean(axis=1).round(4))
        print('--> boundary particles: ',all_mean_count_gt_boundary.mean(axis=1).round(4))
        
        print('\nMEAN DENSITY')        
        print('--> interior particles: ',all_mean_density_interior.mean(axis=1).round(4))
        print('--> boundary particles: ',all_mean_density_gt_boundary.mean(axis=1).round(4))
        
        print('\nMEAN CENTROID DISTANCES')
        print('--> interior particles: ',all_mean_distances_interior.mean(axis=1).round(4))
        print('--> boundary particles: ',all_mean_distances_gt_boundary.mean(axis=1).round(4))        
        
        # print('Saída para colar na tabela de contagem:')
        # print("\t".join([str(x)+'\t'+str(y) for x,y in zip(all_mean_count_gt_boundary.mean(axis=1).round(4),all_mean_count_interior.mean(axis=1).round(4))]))
        
        # print('Saída para colar na tabela de distâncias:')
        # print("\t".join([str(x)+'\t'+str(y) for x,y in zip(all_mean_distances_gt_boundary.mean(axis=1).round(4),all_mean_distances_interior.mean(axis=1).round(4))]))                
        
        # fig1, ax1 = plt.subplots(figsize=(12,6))
        # fig2, ax2 = plt.subplots(figsize=(12,6))  
        
        # cmap = plt.cm.Set1   
        
        # plt.figure(fig1.number)
        # for i in range(num_predictions):
        #     ax1.plot(all_mean_density_interior[i],'-',c=cmap(i/num_predictions),linewidth=2)
        # for i in range(num_predictions):
        #     ax1.plot(all_mean_density_gt_boundary[i],'--',c=cmap(i/num_predictions),linewidth=2)
        # ax1.legend(np.hstack([all_radius,all_radius]))
        # ax1.set_title('Densidade',fontsize=15)

        # plt.figure(fig2.number)
        # for i in range(num_predictions):
        #     ax2.plot(all_mean_distances_interior[i],'-',c=cmap(i/num_predictions),linewidth=2)
        # for i in range(num_predictions):
        #     ax2.plot(all_mean_distances_gt_boundary[i],'--',c=cmap(i/num_predictions),linewidth=2)
        # ax2.legend(np.hstack([all_radius,all_radius]))
        # ax2.set_title('Distancia',fontsize=15)    
        
        # Salva as saídas
        np.savez(
            process_output_file,
            pred_names = all_names,
            search_radius = all_radius,
            
            all_count_neighbors = all_count_neighbors,
            all_mean_count_interior = all_mean_count_interior,
            all_mean_count_gt_boundary = all_mean_count_gt_boundary,

            all_densities = all_densities,
            all_mean_density_interior = all_mean_density_interior,
            all_mean_density_gt_boundary =all_mean_density_gt_boundary,

            all_centroid_distances = all_centroid_distances,
            all_mean_distances_interior = all_mean_distances_interior,
            all_mean_distances_gt_boundary = all_mean_distances_gt_boundary
        )
        
    def process_coarse_test_threshold(self,gt_config_file=None,pred_config_files=None,output_dir=None):
        """ 
        Executa a classificação dos testes grosseiros para diferentes thresholds.
        Última modificação: 21/02/2022.
        
        Args:
            bpart_obj:
            gt_config_file:
            outpu_dir:

        """      
        # Carrega as métricas dos testes grosseiros processadas
        processed_outputs_file = os.path.join(output_dir,'processed.outputs.npz')
        if os.path.exists(processed_outputs_file):
            processed_outputs = dict(np.load(processed_outputs_file, allow_pickle=True))
        else:
            print("Warning: processed output file dont't exist!")
            return

        acc_threshold_file = os.path.join(output_dir,'acc.thresholds.npz')
        if os.path.exists(acc_threshold_file):
            print("Warning: thresholds accuracy file already exists!")
            return            

        pred_names = processed_outputs['pred_names']
        search_radius = processed_outputs['search_radius']
        all_count_neighbors = processed_outputs['all_count_neighbors']
        all_densities = processed_outputs['all_densities']
        all_centroid_distances = processed_outputs['all_centroid_distances']
        
        del processed_outputs
                
        # Calcula as predições variando os thresholds
        density_threshold = np.linspace(0.4,1.1,50)
        distance_threshold = np.linspace(0.0,0.5,40)
        
        combined_threshold = np.linspace(0.4,1.0,50)
        
        #fig3,ax3 = plt.subplots(2,1,figsize=(10,10))
        #fig4,ax4 = plt.subplots(2,1,figsize=(10,10))
        #fig5,ax5 = plt.subplots(2,1,figsize=(10,10))
        
        # draw_colorbar = True
        
        num_predictions = pred_names.shape[0]        
        num_frames = self.data_reader.data_info['final_step']+1
        
        # Array para armazenar dados de predição do 
        # teste grosseiro de densidade
        all_lost_particles_ratio_1 = np.zeros(
            (num_predictions,num_frames,density_threshold.shape[0])
        )
        all_removed_particles_ratio_1 = np.zeros(
            (num_predictions,num_frames,density_threshold.shape[0])
        )
        
        # Array para armazenar dados de predição do 
        # teste grosseiro da distancia até o centroide
        all_lost_particles_ratio_2 = np.zeros(
            (num_predictions,num_frames,distance_threshold.shape[0])
        )
        all_removed_particles_ratio_2 = np.zeros(
            (num_predictions,num_frames,distance_threshold.shape[0])
        )     

        # Array para armazenar dados de predição do 
        # teste grosseiro combinado 1
        all_lost_particles_ratio_3 = np.zeros(
            (num_predictions,num_frames,
            distance_threshold.shape[0],density_threshold.shape[0])
        )
        all_removed_particles_ratio_3 = np.zeros(
            (num_predictions,num_frames,
            distance_threshold.shape[0],density_threshold.shape[0])
        )   

        # Array para armazenar dados de predição do 
        # teste grosseiro combinado 2
        all_lost_particles_ratio_4 = np.zeros(
            (num_predictions,num_frames,combined_threshold.shape[0])
        )
        all_removed_particles_ratio_4 = np.zeros(
            (num_predictions,num_frames,combined_threshold.shape[0])
        )
                
        for i in range(0,num_predictions): 
            print(f'Prediction {i+1}/{num_predictions}:', pred_names[i])
            for step in tqdm(range(0,num_frames),desc='processing steps'):
                # Partículas
                #particles = self.data_reader.get_step(step)
                
                # Raio de busca
                radius = search_radius[i]
                
                # Ground-truth
                gt_labels = self.data_reader.get_step_labels(
                    step, gt_config_file
                )
                
                # Métricas
                densities = all_densities[i,step]
                count_neighbors = all_count_neighbors[i,step]
                centroid_distances = all_centroid_distances[i,step]                   
                
                # 1. TESTE DE DENSIDADE
                lost_particles_ratio_1 = -1*np.ones(density_threshold.shape[0])
                removed_particles_ratio_1 = -1*np.ones(density_threshold.shape[0])

                for k in range(density_threshold.shape[0]):
                    den_threshold = density_threshold[k]
                    #if k%10 == 0:
                        #print(f'radius: {radius}\t step: {step}\t density_threshold: {den_threshold:.4f}')
                                        
                    # Predição
                    pred_labels = np.zeros(count_neighbors.shape[0])
                    pred_labels[densities < den_threshold] = 1                
                    
                    # Partículas perdidas
                    lost_particle_bool = np.logical_and(gt_labels==1,pred_labels==0).astype(int)
                    lost_particles_ratio_1[k] = lost_particle_bool.sum()/(gt_labels==1).sum()
                    
                    # Partículas removidas
                    removed_particles_ratio_1[k] = (pred_labels==0).sum()/pred_labels.shape[0]
                
                all_lost_particles_ratio_1[i,step] = lost_particles_ratio_1
                all_removed_particles_ratio_1[i,step] = removed_particles_ratio_1
                    
                # plt.figure(fig3.number)
                # ax3[0].clear()
                # ax3[0].plot(density_threshold,removed_particles_ratio_1)
                # ax3[0].grid(True)
                # ax3[0].set_ylim([0,1])
                # #ax[0].set_xlabel("density threshold",fontsize=12)
                # ax3[0].set_ylabel("ratio",fontsize=12) 
                # ax3[0].set_title('removed particles ratio',fontsize=15)  
                
                # ax3[1].clear()
                # ax3[1].plot(density_threshold,lost_particles_ratio_1)       
                # ax3[1].grid(True)                
                # ax3[1].set_ylim([0,1])
                # ax3[1].set_xlabel("density threshold",fontsize=12)
                # ax3[1].set_ylabel("ratio",fontsize=12) 
                # ax3[1].set_title('lost particles ratio',fontsize=15)  
                
                # plt.pause(0.01)
                            
                # 2. TESTE DO CENTROIDE                
                lost_particles_ratio_2 = -1*np.ones(distance_threshold.shape[0])
                removed_particles_ratio_2 = -1*np.ones(distance_threshold.shape[0])
                for k in range(distance_threshold.shape[0]):
                    dis_threshold = distance_threshold[k]
                    #if k%10 == 0:
                        #print('radius: {}\t step: {}\t distance_threshold: {:.4f}'.format(radius,step,dis_threshold))

                    # Predição
                    pred_labels = np.zeros(centroid_distances.shape[0])
                    pred_labels[centroid_distances > dis_threshold] = 1
                    
                    # Partículas perdidas
                    lost_particle_bool = np.logical_and(gt_labels==1,pred_labels==0).astype(int)
                    lost_particles_ratio_2[k] = lost_particle_bool.sum()/(gt_labels==1).sum()
                                   
                    # Partículas removidas
                    removed_particles_ratio_2[k] = (pred_labels==0).sum()/pred_labels.shape[0]
                
                all_lost_particles_ratio_2[i,step] = lost_particles_ratio_2
                all_removed_particles_ratio_2[i,step] = removed_particles_ratio_2
                    
                # plt.figure(fig4.number)
                # ax4[0].clear()
                # ax4[0].plot(distance_threshold,removed_particles_ratio_2)
                # ax4[0].grid(True)
                # ax4[0].set_ylim([0,1])
                # #ax[0].set_xlabel("density threshold",fontsize=12)
                # ax4[0].set_ylabel("ratio",fontsize=12) 
                # ax4[0].set_title('removed particles ratio',fontsize=15)  
                
                # ax4[1].clear()
                # ax4[1].plot(distance_threshold,lost_particles_ratio_2)            
                # ax4[1].grid(True)                
                # ax4[1].set_ylim([0,1])
                # ax4[1].set_xlabel("distance threshold",fontsize=12)
                # ax4[1].set_ylabel("ratio",fontsize=12)
                # ax4[1].set_title('lost particles ratio',fontsize=15)  
                
                # plt.pause(0.01)

                # 3. TESTE COMBINADO 1
                lost_particles_ratio_3 = -1*np.ones(
                    (distance_threshold.shape[0],density_threshold.shape[0])
                )
                removed_particles_ratio_3 = -1*np.ones(
                    (distance_threshold.shape[0],density_threshold.shape[0])
                )
                for k1 in range(distance_threshold.shape[0]):
                    dis_threshold = distance_threshold[k1]                                        
                    
                    for k2 in range(density_threshold.shape[0]):
                        den_threshold = density_threshold[k2]
                    
                        #if k1%10 == 0 and k2%10 == 0:
                            #print(f'radius: {radius}\t step: {step}\t dis_threshold: {dis_threshold:.4f}\t den_threshold: {den_threshold:.4f}')
    
                        # Predição com a densidade          
                        pred_labels_1 = np.zeros(count_neighbors.shape[0])
                        pred_labels_1[densities < den_threshold] = 1
    
                        # Predição com distancias
                        pred_labels_2 = np.zeros(centroid_distances.shape[0])
                        pred_labels_2[centroid_distances > dis_threshold] = 1
                        
                        # Predição combinada
                        pred_labels = np.logical_or(pred_labels_1,pred_labels_2).astype(int)
                        #pred_labels = np.logical_and(pred_labels_1,pred_labels_2).astype(int)
                        
                        # Partículas perdidas
                        lost_particle_bool = np.logical_and(gt_labels==1,pred_labels==0).astype(int)
                        lost_particles_ratio_3[k1,k2] = lost_particle_bool.sum()/(gt_labels==1).sum()
                                       
                        # Partículas removidas
                        removed_particles_ratio_3[k1,k2] = (pred_labels==0).sum()/pred_labels.shape[0]
                
                all_lost_particles_ratio_3[i,step] = lost_particles_ratio_3
                all_removed_particles_ratio_3[i,step] = removed_particles_ratio_3
                                    
                # 2. TESTE COMBINADO 2
                lost_particles_ratio_4 = -1*np.ones(combined_threshold.shape[0])
                removed_particles_ratio_4 = -1*np.ones(combined_threshold.shape[0])
                for k in range(combined_threshold.shape[0]):
                    comb_threshold = combined_threshold[k]
                    #if k%10 == 0:
                        #print('radius: {}\t step: {}\t combined_threshold: {:.4f}'.format(radius,step,comb_threshold))

                    # Medida combinada
                    combined_measure =  densities * (1 - centroid_distances)
                    
                    # Predição
                    pred_labels = np.zeros(combined_measure.shape[0])
                    pred_labels[combined_measure < comb_threshold] = 1
                    
                    # Partículas perdidas
                    lost_particle_bool = np.logical_and(gt_labels==1,pred_labels==0).astype(int)
                    lost_particles_ratio_4[k] = lost_particle_bool.sum()/(gt_labels==1).sum()
                                   
                    # Partículas removidas
                    removed_particles_ratio_4[k] = (pred_labels==0).sum()/pred_labels.shape[0]
                
                all_lost_particles_ratio_4[i,step] = lost_particles_ratio_4
                all_removed_particles_ratio_4[i,step] = removed_particles_ratio_4
                                                
                # plt.figure(fig5.number)
                # ax5[0].clear()
                # pcr = ax5[0].pcolor(density_threshold,distance_threshold,removed_particles_ratio_3)
                # if draw_colorbar:
                #     fig5.colorbar(pcr,ax=ax5[0])
                # ax5[0].grid(True)
                # ax5[0].set_ylabel("density threshold",fontsize=12)
                # ax5[0].set_xlabel("distance threshold",fontsize=12) 
                # ax5[0].set_title('removed particles ratio',fontsize=15)  
                
                # ax5[1].clear()
                # pcl = ax5[1].pcolor(density_threshold,distance_threshold,lost_particles_ratio_3)    
                # if draw_colorbar:
                #     fig5.colorbar(pcl,ax=ax5[1])
                # ax5[1].grid(True)
                # ax5[1].set_ylabel("density threshold",fontsize=12)
                # ax5[1].set_xlabel("distance threshold",fontsize=12) 
                # ax5[1].set_title('removed particles ratio',fontsize=15)  
                
                # plt.pause(0.01)
                # draw_colorbar = False            
            
            # TESTE DA DENSIDADE
            
            # plt.figure(fig3.number)
            # ax3[0].clear()
            # ax3[0].plot(density_threshold,all_removed_particles_ratio_1.mean(axis=0))
            # ax3[0].plot(density_threshold,all_removed_particles_ratio_1.min(axis=0))
            # ax3[0].legend(['mean','min'],fontsize=15)
            # ax3[0].grid(True)
            # ax3[0].set_ylim([0,1])
            # #ax[0].set_xlabel("density threshold",fontsize=12)
            # ax3[0].set_ylabel("ratio",fontsize=12) 
            # ax3[0].set_title('removed particles ratio',fontsize=15)  
            
            # ax3[1].clear()
            # ax3[1].plot(density_threshold,all_lost_particles_ratio_1.mean(axis=0))       
            # ax3[1].plot(density_threshold,all_lost_particles_ratio_1.max(axis=0))
            # ax3[1].legend(['mean','max'],fontsize=15)
            # ax3[1].grid(True)                
            # ax3[1].set_ylim([0,1])
            # ax3[1].set_xlabel("density threshold",fontsize=12)
            # ax3[1].set_ylabel("ratio",fontsize=12) 
            # ax3[1].set_title('lost particles ratio',fontsize=15)  
            
            # plt.pause(0.01)                
                
            # # TESTE DO CENTROIDE
                
            # plt.figure(fig4.number)
            # ax4[0].clear()
            # ax4[0].plot(distance_threshold,all_removed_particles_ratio_2.mean(axis=0))
            # ax4[0].plot(distance_threshold,all_removed_particles_ratio_2.min(axis=0))
            # ax4[0].legend(['mean','min'],fontsize=15)
            # ax4[0].grid(True)
            # ax4[0].set_ylim([0,1])
            # #ax[0].set_xlabel("density threshold",fontsize=12)
            # ax4[0].set_ylabel("ratio",fontsize=12) 
            # ax4[0].set_title('removed particles ratio',fontsize=15)  
            
            # ax4[1].clear()
            # ax4[1].plot(distance_threshold,all_lost_particles_ratio_2.mean(axis=0))
            # ax4[1].plot(distance_threshold,all_lost_particles_ratio_2.max(axis=0))
            # ax4[1].legend(['mean','max'],fontsize=15)
            # ax4[1].grid(True)                
            # ax4[1].set_ylim([0,1])
            # ax4[1].set_xlabel("distance threshold",fontsize=12)
            # ax4[1].set_ylabel("ratio",fontsize=12) 
            # ax4[1].set_title('lost particles ratio',fontsize=15)  
            
            # plt.pause(0.01)
            
            # plt.figure(fig5.number)
            # ax5[0].clear()
            # pcr = ax5[0].pcolor(density_threshold,distance_threshold,all_removed_particles_ratio_3[i].mean(axis=0))
            # if draw_colorbar:
            #     fig5.colorbar(pcr,ax=ax5[0])
            # ax5[0].grid(True)
            # ax5[0].set_xlabel("density threshold",fontsize=12)
            # ax5[0].set_ylabel("distance threshold",fontsize=12) 
            # ax5[0].set_title('removed particles ratio',fontsize=15)  
            
            # ax5[1].clear()
            # pcl = ax5[1].pcolor(density_threshold,distance_threshold,all_lost_particles_ratio_3[i].mean(axis=0))    
            # if draw_colorbar:
            #     fig5.colorbar(pcl,ax=ax5[1])
            # ax5[1].grid(True)
            # ax5[1].set_xlabel("density threshold",fontsize=12)
            # ax5[1].set_ylabel("distance threshold",fontsize=12) 
            # ax5[1].set_title('removed particles ratio',fontsize=15)  
            
            # plt.pause(5.0)
            # draw_colorbar = False                   
                        
        # Salva os resultados                
        np.savez(
            acc_threshold_file,

            combined_threshold = combined_threshold,
            density_threshold = density_threshold,
            distance_threshold = distance_threshold,

            all_removed_particles_ratio_density = all_removed_particles_ratio_1,
            all_lost_particles_ratio_density = all_lost_particles_ratio_1,

            all_removed_particles_ratio_centroid = all_removed_particles_ratio_2,
            all_lost_particles_ratio_centroid = all_lost_particles_ratio_2,                     

            all_removed_particles_ratio_comb1 = all_removed_particles_ratio_3,
            all_lost_particles_ratio_comb1 = all_lost_particles_ratio_3,

            all_removed_particles_ratio_comb2 = all_removed_particles_ratio_4,
            all_lost_particles_ratio_comb2 = all_lost_particles_ratio_4
        )  

    def classification_metrics(self,pred_configs,sections=None,output_dir=None,
        extension='csv',plot_metrics=False,print_metrics=True,return_metrics=False):
        """
        Gera o relatório de métricas de classificação de partículas por frame.
        Última modificação: 27/05/2022.
        
        Args:            
            pred_configs: caminho completo dos arquivos de configuração do ground-truth e 
                          da predição.            
            sections: nomes das seções nos arquivos de configurações que correspondem as 
                      predições que devem ser avaliadas.
            output_dir: diretório de saída do relatório de métricas de acurácia.
            extension: extensão de saída do relatório métricas de acurácia.
            plot_metrics: se verdadeiro, plota as métricas de acurácia.
            print_metrics: se verdadeiro, imprime os valores das métricas de acurácia.
            return_metrics: se verdadeiro, retorna a média dos valores das métricas de acurácia.
        """
        confusion_matrix = []
        recall,precision,tnr = [],[],[]
        combined_metric,f1_score,matthews_coefficient = [],[],[]
        num_particles = []
        report = {} # Relatório de classificação
        
        name_basic_metric = ['Recall (TPR)','Precision (PPV)','True Negative Rate (TNR)']
        name_advanced_metric = ['Combined Metric','F1 Score','Matthews correlation coefficient']    
        
        gt_config = ConfigReader(pred_configs[0])
        gt_config_dict = gt_config.get_section(sections[0])

        pred_config = ConfigReader(pred_configs[1])
        pred_config_dict = pred_config.get_section(sections[1])        
        try:
            pred_model_dict = pred_config.get_section('model')
            pred_name = f"{pred_model_dict['name']}_{pred_config_dict['prediction_id']}"
        except:
            pred_name = pred_configs[1].split('/')[-1].replace('.yaml','')

        pred_label_dict = pred_config.get_section('boundary')

        pred_dir = os.path.dirname(pred_configs[1])

        if output_dir==None:
            output_dir = os.path.dirname(pred_configs[1])
        if extension=='csv':
            report_file = os.path.join(output_dir,f"{pred_name}_metrics_report.csv")
        elif extension=='npz':
            report_file = os.path.join(output_dir,f"{pred_name}_metrics_report.npz")

        if os.path.exists(report_file):
            if extension=='csv':
                df = pd.read_csv(report_file)
                basic_metric = [df['rec'],df['pre'],df['tnr']]
                advanced_metric = [df['mc'],df['f1'],df['mcc']]
                steps = df['steps']
                num_particles = df['num_particles']
            elif extension=='npz':
                report = np.load(report_file)
                basic_metric = [report['recall'],report['precision'],report['tnr']]
                advanced_metric = [report['combined_metric'],report['f1_score'],report['matthews_coefficient']]
                steps = report['steps']
                num_particles = report['num_particles']
        else:
            steps_path = os.path.join(pred_dir,pred_label_dict['dir'],
                f"{pred_label_dict['base_name']}*.{pred_label_dict['extension']}")
            steps = self.data_reader.find_available_steps(steps_path)
            if len(steps)==0:
                print("No prediction files found!")
                return

            for step in tqdm(steps, desc='load predictions'):
                #print('step {}'.format(step))                
                # Ground-truth
                # Fronteira
                gt = self.data_reader.get_step_labels(
                    step,pred_configs[0],section=sections[0])
                
                # Prediction
                pred = self.data_reader.get_step_labels(
                    step,pred_configs[1],section=sections[1])

                num_particles.append(pred.shape[0])
                            
                # Classification report
                report[step] = Report(gt,pred)
                
                confusion_matrix.append(report[step].confusion_matrix)
                
                recall.append(report[step].recall())
                precision.append(report[step].precision())
                tnr.append(report[step].tnr())
                
                combined_metric.append(report[step].combined_metric())
                f1_score.append(report[step].f1_score())
                matthews_coefficient.append(report[step].matthews_coefficient())
                
            basic_metric = [np.asarray(recall),np.asarray(precision),np.asarray(tnr)]
            advanced_metric = [np.asarray(combined_metric),np.asarray(f1_score),np.asarray(matthews_coefficient)]

            if extension=='csv':
                columns = ['steps','num_particles','rec','pre','tnr','mc','f1','mcc']
                array = np.array([list(steps.keys()),num_particles,recall,
                        precision,tnr,combined_metric,
                        f1_score,matthews_coefficient]).T.round(4)
                df = pd.DataFrame(array,columns=columns)
                df.to_csv(report_file,index=False,header=True)

            elif extension=='npz':
                np.savez(report_file,steps=steps,
                    num_particles=num_particles,
                    confusion_matrix=confusion_matrix,
                    recall=recall,precision=precision,tnr=tnr,
                    matthews_coefficient=matthews_coefficient,
                    f1_score=f1_score,combined_metric=combined_metric)
            
        if plot_metrics: 
            fig1, axs1 = plt.subplots(3,1)
            fig2, axs2 = plt.subplots(3,1)  
            fig3, axs3 = plt.subplots(1,1)       
            
            plt.figure(fig1.number)
            for i in range(3):
                axs1[i].cla()
                axs1[i].plot(steps,basic_metric[i],'r-',linewidth=3)
                axs1[i].set_xlim(min(steps),max(steps))
                axs1[i].set_ylim(0.8,1)
                axs1[i].set_title(name_basic_metric[i],fontdict={'fontsize': 12})
                axs1[i].grid(True)
                plt.pause(0.01)
                
            plt.figure(fig2.number)
            for i in range(3):
                axs2[i].cla()
                axs2[i].plot(steps,advanced_metric[i],'r-',linewidth=3)
                axs2[i].set_xlim(min(steps),max(steps))
                axs2[i].set_ylim(0.8,1)
                axs2[i].set_title(name_advanced_metric[i],fontdict={'fontsize': 12})
                axs2[i].grid(True)
                plt.pause(0.01)

            plt.figure(fig3.number)
            axs3.cla()
            axs3.plot(steps,num_particles,'r-',linewidth=3)
            axs3.set_xlim(min(steps),max(steps))
            #axs3[0].set_ylim(0.8,1)
            axs3.set_title("Number of particles",fontdict={'fontsize': 12})
            axs3.grid(True)

            plt.show()

        num_particles = np.array(num_particles)
        step_weights = num_particles/num_particles.sum()
        
        avg_recall = (basic_metric[0]*step_weights).sum()
        avg_precision = (basic_metric[1]*step_weights).sum()
        avg_tnr = (basic_metric[2]*step_weights).sum()

        avg_comb_metric = (advanced_metric[0]*step_weights).sum()
        avg_f1 = (advanced_metric[1]*step_weights).sum()
        avg_mcc = (advanced_metric[2]*step_weights).sum()

        if print_metrics:
            print(f"\nMetrics Report: {pred_name}")
            print(f' --> Weighted Avg Recall (TPR): {avg_recall:.4f}')
            print(f' --> Weighted Avg Precision (PPV): {avg_precision:.4f}')
            print(f' --> Weighted Avg TNR: {avg_tnr:.4f}')
            
            print(f' --> Weighted Avg Combined Metric: {avg_comb_metric:.4f}')
            print(f' --> Weighted Avg F1-Score: {avg_f1:.4f}')
            print(f' --> Weighted Avg Matthews Corr. Coeff. (MCC): {avg_mcc:.4f}')

        if return_metrics:
            return {'recall':round(avg_recall,4),
                    'precision':round(avg_precision,4),
                    'tnr':round(avg_tnr,4),
                    'combined_metric':round(avg_comb_metric,4),
                    'f1_score':round(avg_f1,4),
                    'matthews_coefficient':round(avg_mcc,4)}            

    def accuracy_per_curvatures(self,pred_configs,sections=None,output_dir=None,
        extension='csv',plot_metrics=False,print_metrics=True,return_metrics=False):
        """
        Gera o relatório de acurácia por intervalos de curvatura.
        Última modificação: 31/08/2022.
        
        Args:            
            pred_configs: caminho completo dos arquivos de configuração do ground-truth e 
                da predição.            
            sections: nomes das seções nos arquivos de configurações que correspondem as 
                predições que devem ser avaliadas.
            output_dir: diretório de saída do relatório de métricas de acurácia.
            extension: extensão de saída do relatório métricas de acurácia.
            plot_metrics: se verdadeiro, plota as métricas de acurácia.
            print_metrics: se verdadeiro, imprime os valores das métricas de acurácia.
            return_metrics: se verdadeiro, retorna a média dos valores das métricas de acurácia.
        """
        gt_config = ConfigReader(pred_configs[0])
        gt_config_dict = gt_config.get_section(sections[0])

        pred_config = ConfigReader(pred_configs[1])
        pred_config_dict = pred_config.get_section(sections[1])        
        pred_model_dict = pred_config.get_section('model')
        pred_name = f"{pred_model_dict['name']}_{pred_config_dict['prediction_id']}"

        pred_label_dict = pred_config.get_section('boundary')

        pred_dir = os.path.dirname(pred_configs[1])

        if output_dir==None:
            output_dir = os.path.dirname(pred_configs[1])
        
        if extension=='csv':
            report_file = os.path.join(output_dir,'accuracy_per_curvatures.csv')
        elif extension=='npz':
            report_file = os.path.join(output_dir,'accuracy_per_curvatures.npz')
        
        if os.path.exists(report_file):      
            if extension=='csv':
                df = pd.read_csv(report_file)
                avg_bins_accuracy = df.to_numpy()
                bins = avg_bins_accuracy[:,0]
                avg_bins_recall = avg_bins_accuracy[:,1]
                avg_bins_precision = avg_bins_accuracy[:,2]
                avg_bins_f1 = avg_bins_accuracy[:,3]
                avg_bins_mcc = avg_bins_accuracy[:,4]

            elif extension=='npz':
                report = np.load(report_file)
                bins = report['bins']
                avg_bins_recall = report['avg_bins_recall']
                avg_bins_precision = report['avg_bins_precision']
                avg_bins_f1 = report['avg_bins_f1']
                avg_bins_mcc = report['avg_bins_mcc']
        else:
            steps_path = os.path.join(pred_dir,pred_label_dict['dir'],
                f"{pred_label_dict['base_name']}*.{pred_label_dict['extension']}")
            steps = self.data_reader.find_available_steps(steps_path)
            if len(steps)==0:
                print("No prediction files found!")
                return

            res = 0.1
            limits = np.array([-1.0,1.0])
            bins = np.arange(limits[0],limits[1]+res,res)
            bins = bins.round(2)

            bins_recall = np.zeros((len(steps),bins.shape[0]))
            bins_precision = np.zeros((len(steps),bins.shape[0]))
            bins_f1 = np.zeros((len(steps),bins.shape[0]))    
            bins_mcc = np.zeros((len(steps),bins.shape[0]))

            bins_num_points = np.zeros((len(steps),bins.shape[0]))

            num_particles = np.zeros((len(steps),))

            for k,step in enumerate(tqdm(steps, desc='load predictions')):
                #print('step {}'.format(step))                                
                # Ground-truth
                # Fronteira
                gt = self.data_reader.get_step_labels(
                    step,pred_configs[0],section=sections[0])
                
                # Prediction
                pred = self.data_reader.get_step_labels(
                    step,pred_configs[1],section=sections[1])
                    
                num_particles[k] = gt.shape[0]

                # Curvatures
                curvatures = self.data_reader.get_step_measures(
                    step,pred_configs[0],section='curvatures')
                curvatures = curvatures[:,0]
                
                # Particiona os dados por intervalo de curvatura
                voxelizer = SparseVoxelizer(limits.reshape(1,2),res=res,expand_limits=False)
                voxelizer.set_points(curvatures.reshape(-1,1))
                non_empty_bins = voxelizer.find_unique_voxels().reshape(-1)
                indices_per_bin = voxelizer.find_points_per_voxel()

                for i in range(non_empty_bins.shape[0]):
                    bins_num_points[k,non_empty_bins[i]] = indices_per_bin[i].shape[0]                
                
                for i in range(non_empty_bins.shape[0]):
                    rep = Report(gt[indices_per_bin[i]],pred[indices_per_bin[i]],labels=[0,1])
                                        
                    bins_num_points[k,non_empty_bins[i]] = indices_per_bin[i].shape[0]
                    bins_recall[k,non_empty_bins[i]] = rep.recall()
                    bins_precision[k,non_empty_bins[i]] = rep.precision()
                    bins_f1[k,non_empty_bins[i]] = rep.f1_score()
                    bins_mcc[k,non_empty_bins[i]] = rep.matthews_coefficient()

                    print(f'Report for curvatures in ({round(bins[non_empty_bins[i]],2)},{round(bins[non_empty_bins[i]+1],2)}):')
                    print(f'--> recall: {rep.recall()}')
                    print(f'--> precision: {rep.precision()}')
                    print(f'--> f1-score: {rep.f1_score()}')
                    print(f'--> mcc: {rep.matthews_coefficient()}')

            # Weighted averages
            step_weights = bins_num_points/bins_num_points.sum(axis=0,keepdims=True)
            avg_bins_recall = np.nansum(bins_recall*step_weights,axis=0).round(4)
            avg_bins_precision = np.nansum(bins_precision*step_weights,axis=0).round(4)
            avg_bins_f1 = np.nansum(bins_f1*step_weights,axis=0).round(4)
            avg_bins_mcc = np.nansum(bins_mcc*step_weights,axis=0).round(4)

            avg_bins_accuracy = np.concatenate(
                [avg_bins_recall[np.newaxis],avg_bins_precision[np.newaxis],
                avg_bins_f1[np.newaxis],avg_bins_mcc[np.newaxis]])

            if extension=='csv':
                columns = ['curvature_bins','avg_recall','avg_precision','avg_f1','avg_mcc']                
                array = np.concatenate([bins[np.newaxis],avg_bins_accuracy]).T
                df = pd.DataFrame(array,columns=columns)
                df.to_csv(report_file,index=False,header=True)                
            elif extension=='npz':
                np.savez(report_file,
                    bins = bins,
                    avg_bins_recall = avg_bins_recall,
                    avg_bins_precision = avg_bins_precision,
                    avg_bins_f1 = avg_bins_f1,
                    avg_bins_mcc = avg_bins_mcc)
            
        res = bins[1]-bins[0]
        bins_str =  [f'[{bins[i]},{bins[i+1]}]' for i in range(bins.shape[0]-1)]
        bins_str = np.array(bins_str)

        if plot_metrics:
            plt.figure()
            plt.bar(bins+res/6, avg_bins_recall,res/3,label='avg_recall')
            plt.bar(bins+res/6+res/3,avg_bins_precision,res/3,label='avg_precision')
            plt.ylabel('Accuracy',fontdict={'fontsize':15})
            plt.xlabel('Curvature Intervals',fontdict={'fontsize':15})
            plt.title('Accuracy metrics per curvature intervals',fontdict={'fontsize':15})
            plt.xticks(bins[1:-1]+res/3, bins_str[1:])
            plt.yticks(np.arange(0,1.2,0.1))   
            plt.xlim(-1.0,1.0)
            plt.ylim(0,1.1)
            plt.grid(axis='y')            
            plt.legend(loc='upper right')

            plt.figure()
            plt.bar(bins+res/6,avg_bins_f1,res/3,label='avg_f1')
            plt.bar(bins+res/6+res/3,avg_bins_mcc,res/3,label='avg_mcc')
            plt.ylabel('Accuracy',fontdict={'fontsize':15})
            plt.xlabel('Curvature Intervals',fontdict={'fontsize':15})
            plt.title('Accuracy metrics per curvature intervals',fontdict={'fontsize':15})
            plt.xticks(bins[1:-1]+res/3, bins_str[1:])               
            plt.yticks(np.arange(0,1.1,0.1))    
            plt.xlim(-1.0,1.0)
            plt.ylim(0,1.1)
            plt.grid(axis='y')           
            plt.legend(loc='upper right')

            plt.show()
                
        if return_metrics:
            return {'bins':bins,
                    'avg_bins_recall':avg_bins_recall.round(4),
                    'avg_bins_precision':avg_bins_precision.round(4),
                    'avg_bins_f1':avg_bins_f1.round(4),
                    'avg_bins_mcc':avg_bins_mcc.round(4)}            

    def compare_accuracy_per_curvatures(self,pred_configs,sections=None,output_dir=None,
        extension='csv',plot_metrics=False,print_metrics=True,return_metrics=False,pred_names=None):
        """
        Gera o relatório de acurácia por intervalos de curvatura.
        Última modificação: 31/08/2022.
        
        Args:            
            pred_configs: caminho completo dos arquivos de configuração do ground-truth e 
                da predição.            
            sections: nomes das seções nos arquivos de configurações que correspondem as 
                predições que devem ser avaliadas.
            output_dir: diretório de saída do relatório de métricas de acurácia.
            extension: extensão de saída do relatório métricas de acurácia.
            plot_metrics: se verdadeiro, plota as métricas de acurácia.
            print_metrics: se verdadeiro, imprime os valores das métricas de acurácia.
            return_metrics: se verdadeiro, retorna a média dos valores das métricas de acurácia.
        """
        num_predictions = len(pred_configs[1])
        avg_bins_recall = []
        avg_bins_precision = []
        avg_bins_f1 = []
        avg_bins_mcc = []
        for k in range(num_predictions):
            pred_report = self.accuracy_per_curvatures(
                pred_configs=(pred_configs[0],pred_configs[1][k]),
                sections=('boundary','boundary'),
                #output_dir = output_dir,
                plot_metrics=False,
                print_metrics=False,
                return_metrics=True)
            
            bins = pred_report['bins']
            avg_bins_recall.append(pred_report['avg_bins_recall'])
            avg_bins_precision.append(pred_report['avg_bins_precision'])
            avg_bins_f1.append(pred_report['avg_bins_f1'])
            avg_bins_mcc.append(pred_report['avg_bins_mcc'])            

        res = bins[1]-bins[0]

        if output_dir==None:
            output_dir = os.path.dirname(pred_configs[0])

        bar_width = res/(num_predictions+1)    
        if plot_metrics:
            # Recall
            plt.figure(figsize=(10,6))
            for i in range(num_predictions):                
                plt.bar(bins+i*bar_width,avg_bins_recall[i],bar_width,label=pred_names[i])
            plt.ylabel('Accuracy',fontdict={'fontsize':15})
            plt.xlabel('Curvature Intervals',fontdict={'fontsize':15})
            plt.title('Avg Recall',fontdict={'fontsize':15})
            plt.xticks(bins)
            plt.yticks(np.arange(0,1.2,0.1))   
            plt.xlim(-1.0,1.0)
            plt.ylim(0,1.1)
            plt.grid(axis='y')            
            plt.legend(loc='upper right',fontsize=15)
            plt.savefig(os.path.join(output_dir,'avg_bins_recall.png'))

            # Precision
            plt.figure(figsize=(10,6))
            for i in range(num_predictions):                
                plt.bar(bins+i*bar_width,avg_bins_precision[i],bar_width,label=pred_names[i])
            plt.ylabel('Accuracy',fontdict={'fontsize':15})
            plt.xlabel('Curvature Intervals',fontdict={'fontsize':15})
            plt.title('Avg Precision',fontdict={'fontsize':15})
            plt.xticks(bins)
            plt.yticks(np.arange(0,1.2,0.1))   
            plt.xlim(-1.0,1.0)
            plt.ylim(0,1.1)
            plt.grid(axis='y')            
            plt.legend(loc='upper right',fontsize=15)
            plt.savefig(os.path.join(output_dir,'avg_bins_precision.png'))

            # F1-Score
            plt.figure(figsize=(10,6))
            for i in range(num_predictions):                
                plt.bar(bins+i*bar_width,avg_bins_f1[i],bar_width,label=pred_names[i])
            plt.ylabel('Accuracy',fontdict={'fontsize':15})
            plt.xlabel('Curvature Intervals',fontdict={'fontsize':15})
            plt.title('Avg F1-Score',fontdict={'fontsize':15})
            plt.xticks(bins)
            plt.yticks(np.arange(0,1.2,0.1))   
            plt.xlim(-1.0,1.0)
            plt.ylim(0,1.1)
            plt.grid(axis='y')            
            plt.legend(loc='upper right',fontsize=15)
            plt.savefig(os.path.join(output_dir,'avg_bins_f1.png'))

            # F1-Score
            plt.figure(figsize=(10,6))
            for i in range(num_predictions):                
                plt.bar(bins+i*bar_width,avg_bins_mcc[i],bar_width,label=pred_names[i])
            plt.ylabel('Accuracy',fontdict={'fontsize':15})
            plt.xlabel('Curvature Intervals',fontdict={'fontsize':15})
            plt.title('Avg Matthews Correlation Coefficient',fontdict={'fontsize':15})
            plt.xticks(bins)
            plt.yticks(np.arange(0,1.2,0.1))   
            plt.xlim(-1.0,1.0)
            plt.ylim(0,1.1)
            plt.grid(axis='y')            
            plt.legend(loc='upper right',fontsize=15)
            plt.savefig(os.path.join(output_dir,'avg_bins_mcc.png'))

            plt.show()


        if return_metrics:
            return {'bins':bins,
                    'avg_bins_recall':avg_bins_recall.round(4),
                    'avg_bins_precision':avg_bins_precision.round(4),
                    'avg_bins_f1':avg_bins_f1.round(4),
                    'avg_bins_mcc':avg_bins_mcc.round(4)}            

    def classification_times(self,pred_configs=None,output_dir=None,
        extension='npz',plot_times=False,print_times=True,return_times=False):
        """
        Gera o relatório de tempos de classificação para diferentes predições.
        Última modificação: 27/05/2021.
        
        Args:            
            pred_configs: arquivo de configuração ou uma lista de dicionários com as chaves 
                          'dir','names', 'base_name' e 'extension' das predições.
            output_dir: diretório de saída do relatório de tempos.
            extension: extensão de saída do relatório de tempos.
            plot_times: se verdadeiro, plota as métricas de acurácia.
            print_times: se verdadeiro, imprime os valores das métricas de acurácia.
            return_times: se verdadeiro, retorna a média dos valores das métricas de acurácia.

        """        
        pred_names = []
        coarse_pred_time = []
        images_gen_time = []
        images_gen_time_search = []
        images_gen_time_build = []
        images_classif_time = []
        total_time = []
        
        if output_dir != None:
            if extension=='csv':
                report_file = os.path.join(output_dir,'time_report.csv')
            elif extension=='npz':
                report_file = os.path.join(output_dir,'time_report.npz')

        #print('Tempos médios por frame')
        for pred_config in pred_configs:
            pred_dir = os.path.dirname(pred_config)
            
            pred_config_r = ConfigReader(pred_config)
            
            pred_config_dict = pred_config_r.get_section('boundary')
            pred_model_dict = pred_config_r.get_section('model')
            pred_general_dict = pred_config_r.get_section('general')

            pred_names.append(pred_config.split(os.sep)[-2])

            # Carrega o arquivo que possui os tempos de classificação

            if extension=='npz':
                pred_time_file = os.path.join(pred_dir,'time_report.npz')
                pred_time = dict(np.load(pred_time_file))
            elif extension=='csv':
                pred_time_file = os.path.join(pred_dir,'time_report.csv')
                pred_time = pd.read_csv(pred_time_file)
            
            coarse_pred_time_p = pred_time['coarse_prediction']
            #coarse_pred_time_p = [x for x in pred_time['coarse_prediction_time'] if x!=0]
            
            if pred_general_dict['approach']=='pointwise':
                images_gen_time_p = pred_time['images_generation']
                #images_gen_time_p = [x for x in outputs['images_generation_time'] if x!=0]
            elif pred_general_dict['approach']=='regionwise':
                images_gen_time_p_0 = pred_time['images_generation_search']
                images_gen_time_p_1 = pred_time['images_generation_build']
                #images_gen_time_p_0 = [x for x in outputs['images_generation_time_search'] if x!=0]
                #images_gen_time_p_1 = [x for x in outputs['images_generation_time_build'] if x!=0]

            # Support both old (images_classification) and new (fine_prediction) formats
            if 'fine_prediction' in pred_time.columns:
                images_classif_time_p = pred_time['fine_prediction']
            else:
                images_classif_time_p = pred_time['images_classification']
            total_time_p = pred_time['total']
            #images_classif_time_p = [x for x in outputs['images_classification_time'] if x!=0]            
            #total_time_p = [x for x in outputs['total_time'] if x!=0]
            
            coarse_pred_time.append(np.mean(coarse_pred_time_p))
            if pred_general_dict['approach']=='pointnwise':
                images_gen_time.append(np.mean(images_gen_time_p))
            elif pred_general_dict['approach']=='regionwise':
                images_gen_time_search.append(np.mean(images_gen_time_p_0))
                images_gen_time_build.append(np.mean(images_gen_time_p_1))
           
            images_classif_time.append(np.mean(images_classif_time_p))            
            total_time.append(np.mean(total_time_p))

            if print_times:
                print(f'Times report: {pred_names[-1]}')
                print(f" --> coarse prediction time: {coarse_pred_time[-1]:.4f}s")
                if pred_general_dict['approach']=='pointwise':
                    print(f" --> images generation time: {images_gen_time[-1]:.4f}s")
                elif pred_general_dict['approach']=='regionwise':
                    print(f" --> images generation time (search): {images_gen_time_search[-1]:.4f}s")
                    print(f" --> images generation time (build): {images_gen_time_build[-1]:.4f}s")
                print(f" --> images classification time: {images_classif_time[-1]:.4f}s")
                print(f" --> total time: {total_time[-1]:.6f}s")

        coarse_pred_time = np.array(coarse_pred_time)
        if pred_general_dict['approach']=='pointwise':
            images_gen_time = np.array(images_gen_time)
        elif pred_general_dict['approach']=='regionwise':
            images_gen_time_search = np.array(images_gen_time_search)
            images_gen_time_build = np.array(images_gen_time_build)
        images_classif_time = np.array(images_classif_time)
        total_time = np.array(total_time)

        # Plot
        if plot_times:
            plt.barh(pred_names,coarse_pred_time)
            if pred_general_dict['approach']=='pointwise':
                plt.barh(pred_names,images_gen_time,left=coarse_pred_time)
                plt.barh(pred_names,images_classif_time,left=coarse_pred_time+images_gen_time)
                plt.legend(['coarse prediction','images generation','images classification'])
            elif pred_general_dict['approach']=='regionwise':
                plt.barh(pred_names,images_gen_time_search,left=coarse_pred_time)
                plt.barh(pred_names,images_gen_time_build,left=coarse_pred_time+images_gen_time_search)
                plt.barh(pred_names,images_classif_time,left=coarse_pred_time+images_gen_time_search+images_gen_time_build)
                plt.legend(['coarse prediction','images generation (search)','images generation (build)','images classification'])
            plt.title('Mean time per frame',fontdict={'fontsize': 12})
            plt.ylabel('modelos')
            plt.xlabel('time (s)')
            plt.subplots_adjust(left=0.35,right=0.95,bottom=0.05,top=0.95)
            plt.show()

        # Save report to file if output_dir is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            if extension == 'csv':
                # Create DataFrame with results
                data = {
                    'model': pred_names,
                    'coarse_prediction': coarse_pred_time,
                    'fine_prediction': images_classif_time,
                    'total': total_time
                }
                
                if pred_general_dict['approach'] == 'pointwise':
                    data['images_generation'] = images_gen_time
                elif pred_general_dict['approach'] == 'regionwise':
                    data['images_generation_search'] = images_gen_time_search
                    data['images_generation_build'] = images_gen_time_build
                
                df = pd.DataFrame(data)
                df.to_csv(report_file, index=False)
                print(f"\nRelatório salvo em: {report_file}")
            
            elif extension == 'npz':
                save_data = {
                    'model': pred_names,
                    'coarse_prediction': coarse_pred_time,
                    'fine_prediction': images_classif_time,
                    'total': total_time
                }
                
                if pred_general_dict['approach'] == 'pointwise':
                    save_data['images_generation'] = images_gen_time
                elif pred_general_dict['approach'] == 'regionwise':
                    save_data['images_generation_search'] = images_gen_time_search
                    save_data['images_generation_build'] = images_gen_time_build
                
                np.savez(report_file, **save_data)
                print(f"\nRelatório salvo em: {report_file}")

        if return_times:
            if pred_general_dict['approach']=='pointwise':
                return {'coarse_pred_time':coarse_pred_time,
                    'images_gen_time':images_gen_time,
                    'images_classif_time':images_classif_time,
                    'total_time':total_time}
            elif pred_general_dict['approach']=='regionwise':
                return {'coarse_pred_time':coarse_pred_time,
                    'images_gen_time_search':images_gen_time_search,
                    'images_gen_time_build':images_gen_time_build,
                    'images_classif_time':images_classif_time,
                    'total_time':total_time}        
    
    def compare_predictions(self,predictions=None,output_dir=None,
        extension='csv',plot_metrics=False,print_metrics=True,return_metrics=False):
        """
        Gera o relatório de métricas de classificação e tempos de inferência para diferentes predições.
        Última modificação: 20/06/2026.
        
        Args:            
            predictions: lista de dicionários com as chaves 'gt_config_file' e 
                        'pred_config_file' para cada predição. Métricas são calculadas 
                        usando cache (se metrics_report.csv existe) ou recalculadas 
                        usando o ground-truth especificado.
            output_dir: diretório de saída do relatório.
            extension: extensão de saída do relatório ('csv' ou 'npz').
            plot_metrics: se verdadeiro, plota as métricas de acurácia.
            print_metrics: se verdadeiro, imprime os valores das métricas de acurácia.
            return_metrics: se verdadeiro, retorna a média dos valores das métricas de acurácia.

        """
        if predictions is None:
            raise ValueError("Parameter 'predictions' is required")
        
        pred_names = []
        recall = []
        precision = []
        tnr = []
        mc = []
        f1 = []
        mcc = []
        
        coarse_pred_time = []
        images_gen_time_search = []
        images_gen_time_build = []
        images_classif_time = []
        total_time = []
        
        # New columns for prediction type and coarse usage
        prediction_types = []  # 'regionwise' or 'sparse_regionwise'
        uses_coarse = []  # 'with_coarse' or 'no_coarse'
        
        sim_name = None
        gt_names = []  # Track which ground-truth was used for each prediction

        metrics = []

        for prediction in predictions:
            pred_config = prediction['pred_config_file']
            gt_config = prediction.get('gt_config_file', None)
            
            pred_dir = os.path.dirname(pred_config)            
            pred_config_r = ConfigReader(pred_config)            
            pred_general_dict = pred_config_r.get_section('general')

            pred_names.append(pred_config.split(os.sep)[-2])
            
            # Store prediction type (approach)
            prediction_types.append(pred_general_dict.get('approach', 'unknown'))
            
            # Determine if coarse prediction is used
            pred_name = pred_config.split(os.sep)[-2]
            if 'no_coarse' in pred_name.lower():
                uses_coarse.append('no_coarse')
            else:
                uses_coarse.append('with_coarse')
            
            # Get simulation name from gt_config directory
            if gt_config is not None:
                # Extract simulation name from gt_config directory
                gt_dir = os.path.dirname(gt_config)
                current_sim_name = os.path.basename(gt_dir)
            else:
                # Fallback to data reader's directory name
                if sim_name is None:
                    sim_name = os.path.basename(self.data_reader.data_dir)
                current_sim_name = sim_name
            
            # Track simulation names for each prediction
            if sim_name is None:
                sim_name = current_sim_name
            
            # Store the simulation name for this specific prediction
            sim_names_list = sim_names_list if 'sim_names_list' in locals() else []
            sim_names_list.append(current_sim_name)
            
            # Track which ground-truth was used
            if gt_config is not None:
                gt_names.append(os.path.basename(gt_config))
            else:
                gt_names.append('pre-calculated')

            # Check if metrics and time reports exist (cache)
            metrics_report_file = os.path.join(pred_dir,'metrics_report.csv')
            time_report_file = os.path.join(pred_dir, 'time_report.csv')
            
            metrics_exist = os.path.exists(metrics_report_file)
            times_exist = os.path.exists(time_report_file)

            # If gt_config_file is provided and metrics don't exist, calculate them
            if gt_config is not None and not metrics_exist:
                print(f"\nCalculating metrics using ground-truth: {gt_config}")
                self.classification_metrics(
                    pred_configs=(gt_config, pred_config),
                    sections=('boundary', 'boundary'),
                    output_dir=pred_dir,
                    extension='csv',
                    plot_metrics=False,
                    print_metrics=False,
                    return_metrics=False
                )
            elif gt_config is not None and metrics_exist:
                print(f"\nUsing cached metrics for: {pred_names[-1]}")
            
            # Check if metrics file exists after potential calculation
            if not os.path.exists(metrics_report_file):
                raise FileNotFoundError(f"Metrics report not found: {metrics_report_file}")

            # Read metrics report
            metrics_report = pd.read_csv(metrics_report_file)
            
            # Select only metric columns
            metric_columns = ['rec', 'pre', 'tnr', 'mc', 'f1', 'mcc']
            metrics.append(metrics_report[metric_columns].mean().to_numpy().round(4))
            
            # Read time report (if it doesn't exist, use zeros)
            if not times_exist:
                print(f"Warning: Time report not found for {pred_names[-1]}, using zeros")
                coarse_pred_time.append(0.0)
                images_gen_time_search.append(0.0)
                images_gen_time_build.append(0.0)
                images_classif_time.append(0.0)
                total_time.append(0.0)
            else:
                time_report = pd.read_csv(time_report_file)
                
                coarse_pred_time.append(np.mean(time_report['coarse_prediction']))
                
                if pred_general_dict['approach'] == 'regionwise':
                    gen_search = np.mean(time_report['images_generation_search'])
                    gen_build = np.mean(time_report['images_generation_build'])
                    images_gen_time_search.append(gen_search)
                    images_gen_time_build.append(gen_build)
                    
                    # For regionwise: fine_prediction includes image generation + classification
                    if 'fine_prediction' in time_report.columns:
                        classif_time = np.mean(time_report['fine_prediction'])
                    else:
                        classif_time = np.mean(time_report['images_classification'])
                    
                    # Aggregate image generation times into fine prediction time
                    images_classif_time.append(gen_search + gen_build + classif_time)
                else:
                    # For non-regionwise approaches, set image generation times to 0
                    images_gen_time_search.append(0.0)
                    images_gen_time_build.append(0.0)
                    
                    # Support both old and new format for fine prediction time
                    if 'fine_prediction' in time_report.columns:
                        images_classif_time.append(np.mean(time_report['fine_prediction']))
                    else:
                        images_classif_time.append(np.mean(time_report['images_classification']))
                
                total_time.append(np.mean(time_report['total']))

            if print_metrics:
                print(f'\n=== Report: {pred_names[-1]} ===')
                print(f'Metrics:')
                print(f"  recall: {metrics_report['rec'].mean().round(4)}")
                print(f"  precision: {metrics_report['pre'].mean().round(4)}")
                print(f"  tnr: {metrics_report['tnr'].mean().round(4)}")
                print(f"  combined metric: {metrics_report['mc'].mean().round(4)}")
                print(f"  f1-score: {metrics_report['f1'].mean().round(4)}")
                print(f"  mattheus coefficent: {metrics_report['mcc'].mean().round(4)}")
                print(f'Times:')
                print(f"  coarse prediction: {coarse_pred_time[-1]:.4f}s")
                if pred_general_dict['approach'] == 'regionwise':
                    print(f"  images generation (search): {images_gen_time_search[-1]:.4f}s")
                    print(f"  images generation (build): {images_gen_time_build[-1]:.4f}s")
                print(f"  fine prediction: {images_classif_time[-1]:.4f}s")
                print(f"  total: {total_time[-1]:.6f}s")

        # Plot
        if plot_metrics:
            if len(metrics) == 2:
                # Original plot for exactly 2 models
                labels = ['recall', 'precision', 'tnr', 'comb-metric', 'f1-score','matthews coeff']

                x = np.arange(len(labels))  # the label locations
                width = 0.35  # the width of the bars

                fig, ax = plt.subplots()
                rects1 = ax.bar(x - width/2, metrics[0], width, label=pred_names[0])
                rects2 = ax.bar(x + width/2, metrics[1], width, label=pred_names[1])

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('Metrics')
                #ax.set_title('Scores by group and gender')
                ax.set_xticks(x, labels)
                ax.legend()

                ax.set_ylim([0,1.1])

                #ax.bar_label(rects1, padding=3)
                #ax.bar_label(rects2, padding=3)

                fig.tight_layout()

                plt.show()
            else:
                print(f"\nWarning: Plotting is only supported for exactly 2 models. Found {len(metrics)} models.")
                print("Skipping plot generation.")
        
        # Save results to file if output_dir is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            if extension == 'csv':
                # Debug: Check array lengths before creating DataFrame
                print(f"\nDebug - Array lengths:")
                print(f"  sim_names_list: {len(sim_names_list)}")
                print(f"  pred_names: {len(pred_names)}")
                print(f"  metrics: {len(metrics)}")
                print(f"  coarse_pred_time: {len(coarse_pred_time)}")
                print(f"  images_classif_time: {len(images_classif_time)}")
                print(f"  total_time: {len(total_time)}")
                if len(images_gen_time_search) > 0:
                    print(f"  images_gen_time_search: {len(images_gen_time_search)}")
                    print(f"  images_gen_time_build: {len(images_gen_time_build)}")
                
                # Create DataFrame with results
                data = {
                    'simulation': sim_names_list,
                    'prediction': pred_names,
                    'prediction_type': prediction_types,
                    'coarse_usage': uses_coarse,
                    'recall': [m[0] for m in metrics],
                    'precision': [m[1] for m in metrics],
                    'tnr': [m[2] for m in metrics],
                    'combined_metric': [m[3] for m in metrics],
                    'f1_score': [m[4] for m in metrics],
                    'matthews_coeff': [m[5] for m in metrics],
                    'coarse_prediction_time': coarse_pred_time,
                    'fine_prediction_time': images_classif_time,
                    'total_time': total_time
                }
                
                # Add regionwise-specific time columns if applicable
                if len(images_gen_time_search) > 0:
                    data['images_generation_search_time'] = images_gen_time_search
                    data['images_generation_build_time'] = images_gen_time_build
                
                df = pd.DataFrame(data)
                report_file = os.path.join(output_dir, 'predictions_comparison.csv')
                df.to_csv(report_file, index=False)
                print(f"\nRelatório salvo em: {report_file}")
            
            elif extension == 'npz':
                save_data = {
                    'simulation': sim_name,
                    'prediction': pred_names,
                    'metrics': metrics,
                    'coarse_prediction_time': coarse_pred_time,
                    'fine_prediction_time': images_classif_time,
                    'total_time': total_time
                }
                
                if len(images_gen_time_search) > 0:
                    save_data['images_generation_search_time'] = images_gen_time_search
                    save_data['images_generation_build_time'] = images_gen_time_build
                
                report_file = os.path.join(output_dir, 'predictions_comparison.npz')
                np.savez(report_file, **save_data)
                print(f"\nRelatório salvo em: {report_file}")

    def coarse_test_analysis(self,gt_config_file,pred_config_files,test_type=0):
        """ 
        Análise do teste grosseiro.     
        Última modificação: 21/02/2022. 
                
        Args:
            gt_config_file:
            pred_config_files:
            test_type:
        """        
        all_percentage_interior_particles = []
        all_percentage_undefined_particles = []         
        all_percentage_lost_boundary_particles = []
        
        all_mean_measure_interior = []
        all_min_measure_interior = []
        all_max_measure_interior = []
    
        all_mean_measure_boundary = []
        all_min_measure_boundary = []
        all_max_measure_boundary = []  
        
        all_time_per_frame = []
        #all_threshold_per_frame = []
        pred_steps = []        
        all_names = []        
        all_radius = []
        
        # Itera sobre as predições grosseiras
        for pred_config_file in pred_config_files:

            pred_config = ConfigReader(pred_config_file)
            #pred_config_dict = pred_config.get_labels_config()
            
            pred_general = pred_config.get_section(
                'general',['pred_sections','measure_sections'])
            pred_section = pred_general['pred_sections']
            measure_section = pred_general['measure_sections']

            pred_config_dict = pred_config.get_section(
                pred_section,['dir','base_name','extension'])

            pred_dir = os.path.dirname(pred_config_file)

            all_names.append(pred_config_file.split(os.sep)[-2])

            pred_config_dir = os.path.dirname(pred_config_file)
            report_file = os.path.join(pred_config_dir,'report.npz')
            
            # Carrega outras saídas do teste grosseiro 
            other_pred_time_file = os.path.join(
                os.path.dirname(pred_config_file),'other_outputs.npz')
            other_outputs = np.load(other_pred_time_file)            
            all_time_per_frame.append(other_outputs['time_per_frame'])            
            all_radius.append(float(other_outputs['search_radius']))
            
            # if test_type == 1 or test_type == 2:
            #     all_radius.append(float(other_outputs['search_radius']))
            # elif test_type == 3:
            #     all_radius.append(float(other_outputs['coarse_grid_rate']))
                
            #all_threshold_per_frame.append(other_outputs['threshold_per_frame'])
                        
            if os.path.exists(report_file):
                
                report = np.load(report_file)
                
                pred_steps.append(report['pred_steps'])
                all_percentage_interior_particles.append(report['percentage_interior_particles'])
                all_percentage_undefined_particles.append(report['percentage_undefined_particles']) 
                all_percentage_lost_boundary_particles.append(report['percentage_lost_boundary_particles'])
                
                all_mean_measure_interior.append(report['mean_measure_interior'])
                all_min_measure_interior.append(report['min_measure_interior'])
                all_max_measure_interior.append(report['max_measure_interior'])
            
                all_mean_measure_boundary.append(report['mean_measure_boundary'])
                all_min_measure_boundary.append(report['min_measure_boundary'])
                all_max_measure_boundary.append(report['max_measure_boundary'])
            else:                
                percentage_interior_particles = []
                percentage_undefined_particles = []         
                percentage_lost_boundary_particles = []
                
                mean_measure_interior = []
                min_measure_interior = []
                max_measure_interior = []
            
                mean_measure_boundary = []
                min_measure_boundary = []
                max_measure_boundary = []   
                
                print('\nCoarse test: ',all_names[-1])            

                pred_files = os.listdir(
                    os.path.join(pred_dir,pred_config_dict['dir']))
                steps = []
                for pred_file in pred_files:
                    if pred_config_dict['base_name'] in pred_file:
                        steps.append(int(pred_file.split('.')[-2]))
                steps.sort()
                if len(steps)==0:
                    print("No prediction files found!")
                    return
                pred_steps.append(steps)
                for step in tqdm(steps, desc='load predictions'):
                    #print('step {}'.format(step))
                    #                     
                    # ground-truth
                    gt_labels = self.data_reader.get_step_labels(
                        step, gt_config_file,section='boundary')

                    # predição
                    pred_labels = self.data_reader.get_step_labels(
                        step, pred_config_file, section=pred_section)

                    # Carrega as medidas da predição
                    try:
                        measures = self.data_reader.get_step_measures(
                            step, pred_config_file, section=measure_section)
                    except:
                        print('Não existem medidas para esta predição!')
                        measures = None
                    
                    percent_int_particles = 100*(pred_labels==0).sum()/pred_labels.shape[0] 
                    percent_undef_particles = 100*(pred_labels==1).sum()/pred_labels.shape[0] 
                    
                    percent_lost_bound_particles = 100*(gt_labels[pred_labels==0]==1).sum()/(gt_labels==1).sum()
                                        
                    # print(' --> interior particles: {}/{} ({:.12f}%)'.format((pred_labels==0).sum(),
                    #                                               pred_labels.shape[0],
                    #                                               percent_int_particles))        
                    # print(' --> undefined particles: {}/{} ({:.12f}%)'.format((pred_labels==1).sum(),
                    #                                                   pred_labels.shape[0],
                    #                                                   percent_undef_particles))
                    # print(' --> lost boundary particles: {}/{} ({:.12f}%)'.format((gt_labels[pred_labels==0]==1).sum(),
                                                                                 # (gt_labels==1).sum(),
                                                                                 # percent_lost_bound_particles))
                           
                    percentage_interior_particles.append(percent_int_particles)
                    percentage_undefined_particles.append(percent_undef_particles)
                    percentage_lost_boundary_particles.append(percent_lost_bound_particles)
                    
                    if measures is not None:
                        # Analisa o ground-truth
                        measure_gt_boundary = measures[gt_labels==1]
                        measure_gt_interior = measures[gt_labels==0]
                        
                        mean_measure_interior.append(measure_gt_interior.mean())
                        min_measure_interior.append(measure_gt_interior.min())
                        max_measure_interior.append(measure_gt_interior.max())
                    
                        mean_measure_boundary.append(measure_gt_boundary.mean())
                        min_measure_boundary.append(measure_gt_boundary.min())
                        max_measure_boundary.append(measure_gt_boundary.max())
                
                # Salva o relatório
                np.savez(
                    report_file,         
                    pred_steps = steps,       
                    time_per_frame = other_outputs['time_per_frame'],
                    #threshold_per_frame = other_outputs['threshold_per_frame'],
                    percentage_interior_particles = percentage_interior_particles,
                    percentage_undefined_particles = percentage_undefined_particles,
                    percentage_lost_boundary_particles = percentage_lost_boundary_particles,
                    mean_measure_interior = mean_measure_interior,
                    min_measure_interior = min_measure_interior,
                    max_measure_interior = max_measure_interior,
                    mean_measure_boundary = mean_measure_boundary,
                    min_measure_boundary = min_measure_boundary,
                    max_measure_boundary = max_measure_boundary)         
                
                all_percentage_interior_particles.append(percentage_interior_particles)
                all_percentage_undefined_particles.append(percentage_undefined_particles) 
                all_percentage_lost_boundary_particles.append(percentage_lost_boundary_particles)
                
                all_mean_measure_interior.append(mean_measure_interior)
                all_min_measure_interior.append(min_measure_interior)
                all_max_measure_interior.append(max_measure_interior)
            
                all_mean_measure_boundary.append(mean_measure_boundary)
                all_min_measure_boundary.append(min_measure_boundary)
                all_max_measure_boundary.append(max_measure_boundary) 

        all_percentage_lost_boundary_particles = np.asarray(all_percentage_lost_boundary_particles)
        all_percentage_interior_particles = np.asarray(all_percentage_interior_particles)
        all_time_per_frame = np.asarray(all_time_per_frame)  

        # 1 figura para as partículas perdidas
        # 1 figura para a porcentagem de partículas removidas
        # 1 figura para os tempos
        
        #1 figura para as médias
        
        # 4 figuras para ilustrar como as medidas se comportam na fronteira e no interior

        # Visualização
        fig1, ax1 = plt.subplots(figsize=(12,6))
        fig2, ax2 = plt.subplots(figsize=(12,6))          
        fig3, ax3 = plt.subplots(figsize=(12,6))        
        
        # fig4, ax4 = plt.subplots(figsize=(6,6))
        # fig5, ax5 = plt.subplots(figsize=(6,6))
        # fig6, ax6 = plt.subplots(figsize=(6,6))
        # fig7, ax7 = plt.subplots(figsize=(12,6))
        
        # figs = [fig4,fig5,fig6,fig7]
        # axs = [ax4,ax5,ax6,ax7]

        for i in range(len(all_names)):
            print('Test:', all_names[i])
            print(f' --> mean removed particles: {np.mean(all_percentage_interior_particles[i]):6f}%')
            print(f' --> mean lost boundary particles: {np.mean(all_percentage_lost_boundary_particles[i]):6f}%')

        # Figura 1
        plt.figure(fig1.number)
        #plt.show()  
        for i in range(len(all_names)):
            ax1.plot(pred_steps[i],all_percentage_lost_boundary_particles[i],label=all_names[i],linewidth=3)
            ax1.grid(True)
            plt.pause(0.01)
        ax1.set_xlabel("frame",fontsize=12)    
        ax1.set_ylabel("percentage (%)",fontsize=12)
        ax1.legend(fontsize=15)
        ax1.set_title('Percentage of lost boundary particles',fontsize=15)
        plt.pause(0.01)

        # Figura 2
        plt.figure(fig2.number)
        for i in range(len(all_names)):            
            ax2.plot(pred_steps[i],all_percentage_interior_particles[i],label=all_names[i],linewidth=3)
            ax2.grid(True)
            plt.pause(0.01)
        ax2.set_xlabel("frame",fontsize=12)
        ax2.set_ylabel("percentage (%)",fontsize=12)
        ax2.legend(fontsize=15)
        ax2.set_title('Percentage of removed interior particles',fontsize=15)
        plt.pause(0.01)

        # Figura 3
        plt.figure(fig3.number)
        for i in range(len(all_names)):            
            ax3.plot(pred_steps[i],all_time_per_frame[i],label=all_names[i],linewidth=3)
            ax3.grid(True)
            plt.pause(0.01)            
        ax3.set_xlabel("frame",fontsize=12)        
        ax3.set_ylabel("time (s)",fontsize=12)        
        ax3.legend(fontsize=15)
        ax3.set_title('Time per frame',fontsize=15)
        plt.pause(0.01)     
        
        plt.show()  

        # Figura 4
        # plt.figure(fig4.number)
        
        # ax4.plot(all_radius,all_percentage_lost_boundary_particles.mean(axis=1),'o-',linewidth=3)
        # ax4.set_xlabel("search radius",fontsize=15)
        # ax4.set_ylabel("percentage (%)",fontsize=15)
        # #ax4[0].legend(fontsize=15)
        # ax4.set_title('Percentage of lost boundary particles',fontsize=15)
        # ax4.set_ylim([0,2.0])
        # ax4.grid(True)
        
        # plt.figure(fig5.number)
        # ax5.plot(all_radius,all_percentage_interior_particles.mean(axis=1),'o-',linewidth=3)
        # ax5.set_xlabel("search radius",fontsize=15)
        # ax5.set_ylabel("percentage (%)",fontsize=15)
        # #ax4[1].legend(fontsize=15)
        # ax5.set_title('Percentage of removed interior particles',fontsize=15)
        # ax5.set_ylim([40,100])
        # ax5.grid(True)
        
        # plt.figure(fig6.number)
        # ax6.plot(all_radius,all_time_per_frame.mean(axis=1),'o-',linewidth=3)
        # ax6.set_xlabel("search radius",fontsize=15)
        # ax6.set_ylabel("time (s)",fontsize=15)
        # #ax4[2].legend(fontsize=15)
        # ax6.set_title('Time per frame',fontsize=15)
        # ax6.set_ylim([0,0.5])
        # ax6.grid(True)
        
        # Figura 4 a 7
        # ylabel = ['average of neighboring particles',
        #           'average of neighboring particles',
        #           'distance to centroid normalized',
        #           'distance to centroid normalized']
        # for i in range(len(figs)):
        #     plt.figure(figs[i].number)
        #     axs[i].plot(all_mean_measure_interior[i],'b-',label='mean_measure_interior')
        #     axs[i].plot(all_min_measure_interior[i],'b:',label='min_measure_interior')
        #     axs[i].plot(all_max_measure_interior[i],'b--',label='max_measure_interior')        
            
        #     axs[i].plot(all_mean_measure_boundary[i],'r-',label='mean_measure_boundary')
        #     axs[i].plot(all_min_measure_boundary[i],'r:',label='mean_measure_boundary')
        #     axs[i].plot(all_max_measure_boundary[i],'r--',label='mean_measure_boundary')        
            
        #     #axs[i].plot(all_threshold_per_frame[i],'k-',label='threshold')
            
        #     axs[i].set_xlabel("frame",fontsize=12)    
        #     axs[i].set_ylabel(ylabel[i],fontsize=12)    
        #     axs[i].legend(fontsize=15)
        #     axs[i].set_title(all_names[i],fontsize=15)

        print('\nFeito!')

    def compute_mean_absolute_error(self,arrays_config_file,section,
        comparative_label=1,initial_step=-1,final_step=-1,pause=0.1):
        """
        Calcula o erro absoluto médio entre arrays de previsões e o ground-truth.
        Última modificação: 18/04/2022.

        Args:
            arrays_config_file:
            section:
            label:
            initial_step:
            final_step:
        """        
        mae_file = os.path.join(self.data_reader.data_dir,'mean_abs_error_normal.txt')
        if os.path.exists(mae_file):
            mae = np.loadtxt(mae_file)            
            if len(mae.shape)==1:
                mae = mae.reshape(-1,1)
        else:
            if initial_step==-1:
                initial_step = self.data_reader.data_info['initial_step']            
            if final_step==-1:
                final_step = self.data_reader.data_info['final_step']
            mae = np.zeros((final_step+1,len(arrays_config_file)-1))
            for step in range(initial_step,final_step+1):
                print('Step',step)
                
                # Ground-truth
                # Fronteira
                gt_bound = self.data_reader.get_step_measures(
                    step,arrays_config_file[0],section='boundary')
                gt_bound = gt_bound==comparative_label
                
                # Normal
                array_0 = self.data_reader.get_step_measures(
                    step,arrays_config_file[0],section=section)

                mae_pw = MeanAbsoluteErrorPW(reduction='none')

                # Previsões
                for i in range(1,len(arrays_config_file)):
                    array_i = self.data_reader.get_step_measures(
                        step,arrays_config_file[i],section=section)
                    mae[step,i-1] = mae_pw(array_0,array_i).numpy()[gt_bound].mean()
            
            np.savetxt(mae_file,mae,fmt='%.6f')

        return mae.T  

    def compute_regression_metrics(self,gt_config_file,pred_config_file,section,
        comparative_label=1,initial_step=-1,final_step=-1,device='cpu',print_metrics=True,
        return_metrics=False):
        """
        Calcula métricas de regressão entre arrays de previsões e o ground-truth.
        Última modificação: 15/12/2022.

        Args:
            gt_config_file:
            pred_config_file:
            section:
            label:
            initial_step:
            final_step:
            print_metrics:
            return_metrics:
        """
        pred_dir = os.path.dirname(pred_config_file)
        metrics_file = os.path.join(pred_dir,'normal_metrics_report.csv')
        if os.path.exists(metrics_file):
            metrics = pd.read_csv(metrics_file)
        else:
            # Descobrir quais frames têm predição disponível
            pred_config = ConfigReader(pred_config_file)
            pred_boundary_section = pred_config.get_section('boundary', 
                ['dir','base_name','extension'], warnings=False)
            
            # Construir padrão para encontrar arquivos de predição
            # Padrão típico: labels.{step}.txt
            pred_boundary_pattern = os.path.join(
                pred_dir,
                pred_boundary_section['dir'],
                f"{pred_boundary_section['base_name']}.*.{pred_boundary_section['extension']}"
            )
            
            # Encontrar frames disponíveis usando glob
            pred_files = glob.glob(pred_boundary_pattern)
            
            if len(pred_files) == 0:
                raise ValueError(f"Nenhum frame de predição encontrado em: {pred_boundary_pattern}")
            
            # Extrair números dos frames dos nomes de arquivo
            available_steps = []
            for pred_file in pred_files:
                # Extrai o número entre o último ponto antes da extensão
                # Ex: labels.301.txt -> 301
                basename = os.path.basename(pred_file)
                parts = basename.split('.')
                if len(parts) >= 3:  # base_name.step.extension
                    try:
                        step = int(parts[-2])
                        available_steps.append(step)
                    except ValueError:
                        continue
            
            if len(available_steps) == 0:
                raise ValueError(f"Não foi possível extrair números de frames de: {pred_boundary_pattern}")
            
            available_steps = sorted(available_steps)
            max_step = max(available_steps)
            
            print(f"Encontrados {len(available_steps)} frames de predição: {min(available_steps)}-{max_step}")
            
            with tf.device(device):
                mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
                mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
                cos = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE) 

            mean_mae = np.zeros((max_step+1,1))
            std_mae = np.zeros((max_step+1,1))
            mean_mse = np.zeros((max_step+1,1))
            std_mse = np.zeros((max_step+1,1))
            mean_cos = np.zeros((max_step+1,1))
            std_cos = np.zeros((max_step+1,1))
            mean_angle = np.zeros((max_step+1,1))
            std_angle = np.zeros((max_step+1,1))
            num_particles = np.zeros((max_step+1,1))
            
            for step in tqdm(available_steps, desc="Processing predictions: "):
               
                # Gt fronteira
                try:
                    gt_bound = self.data_reader.get_step_labels(
                        step,gt_config_file,section='boundary')
                    gt_bound = gt_bound==comparative_label
                except:
                    print(f"Não foi possível carregar o ground truth de fronteira: frame {step}")
                    continue

                # Pred fronteira
                try:
                    pred_bound = self.data_reader.get_step_labels(
                        step,pred_config_file,section='boundary')
                    pred_bound = pred_bound==comparative_label

                    true_positive = np.logical_and(gt_bound,pred_bound)
                except:
                    print(f"Não foi possível carregar a predição de fronteira: frame {step}")
                    continue                
                
                # Pred Normal
                try:
                    pred_normal = self.data_reader.get_step_measures(
                        step,pred_config_file,section=section)
                    if pred_normal is None:
                        print(f"Não foi possível carregar a predição da normal: frame {step} - array é None")
                        continue
                    pred_normal = pred_normal[true_positive]
                except Exception as e:
                    print(f"Não foi possível carregar a predição da normal: frame {step}")
                    print(f"  Erro: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # Gt Normal
                try:
                    gt_normal = self.data_reader.get_step_measures(
                        step,gt_config_file,section=section)
                    print(f"DEBUG: gt_normal carregado para frame {step}, shape={gt_normal.shape if gt_normal is not None else 'None'}")
                    print(f"DEBUG: true_positive sum={np.sum(true_positive)}")
                    if gt_normal is None:
                        print(f"Não foi possível carregar o ground truth de normal: frame {step} - array é None")
                        continue
                    gt_normal = gt_normal[true_positive]
                    if len(gt_normal) == 0:
                        print(f"Não foi possível carregar o ground truth de normal: frame {step} - sem partículas true_positive")
                        continue
                except Exception as e:
                    print(f"Não foi possível carregar o ground truth de normal: frame {step}")
                    print(f"  Erro: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # Métricas            
                with tf.device(device):                    
                    all_mae = mae(gt_normal,pred_normal).numpy()
                    all_mse = mse(gt_normal,pred_normal).numpy()
                    all_cosines = -cos(gt_normal,pred_normal).numpy()
                    all_angles = np.degrees(np.arccos(all_cosines))
                
                mean_mae[step] = all_mae.mean()
                std_mae[step] = all_mae.std()
                mean_mse[step] =  all_mse.mean()
                std_mse[step] =  all_mse.std()
                mean_cos[step] = all_cosines.mean()
                std_cos[step] = all_cosines.std()
                mean_angle[step] = all_angles.mean()
                std_angle[step] = all_angles.std()
                    
                num_particles[step] = gt_normal.shape[0]
            
            metrics = pd.DataFrame(np.hstack([mean_mae,std_mae,mean_mse,std_mse,mean_cos,std_cos,mean_angle,std_angle,num_particles]),
                columns=['mean_mae','std_mae','mean_mse','std_mse','mean_cos','std_cos','mean_angle','std_angle','num_particles']).round(4)
            metrics.to_csv(metrics_file)

        # Métricas ponderadas
        step_weights = metrics['num_particles'].to_numpy()/metrics['num_particles'].to_numpy().sum()
        
        mean_mae = (metrics['mean_mae'].to_numpy()*step_weights).sum()
        std_mae = (metrics['std_mae'].to_numpy()*step_weights).sum()
        mean_mse = (metrics['mean_mse'].to_numpy()*step_weights).sum()        
        std_mse = (metrics['std_mse'].to_numpy()*step_weights).sum()
        mean_cos = (metrics['mean_cos'].to_numpy()*step_weights).sum()
        std_cos = (metrics['std_cos'].to_numpy()*step_weights).sum()
        mean_angle = (metrics['mean_angle'].to_numpy()*step_weights).sum()
        std_angle = (metrics['std_angle'].to_numpy()*step_weights).sum()

        if print_metrics:
            print(f"\nRegression metrics report: {pred_config_file}")
            print(f' --> Mean Absolute Error: {mean_mae:.4f}')
            print(f' --> Std Absolute Error: {std_mae:.4f}')
            print(f' --> Mean Squared Error: {mean_mse:.4f}')
            print(f' --> Std Squared Error: {std_mse:.4f}')
            print(f' --> Mean Cosine Similarity: {mean_cos:.4f}')
            print(f' --> Std Cosine Similarity: {std_cos:.4f}')
            print(f' --> Mean Angle: {mean_angle:.4f}')
            print(f' --> Std Angle: {std_angle:.4f}')

        if return_metrics:
            return {'mean_mae':round(mean_mae,4),
                    'std_mae':round(std_mae,4),
                    'mean_mse':round(mean_mse,4),
                    'std_mse':round(std_mse,4),
                    'mean_cos':round(mean_cos,4),
                    'std_cos':round(std_cos,4),
                    'mean_angle':round(mean_angle,4),
                    'std_angle':round(std_angle,4)
            }

    def compute_ratios(self,name,column,taget_label,initial_step=0,final_step=-1,pause=0.1):
        """
        Calcula a proporção da quantidade de dados com um rótulo específico em 
        releação ao conjunto completo.
        Última modificação: 18/05/2022.

        Args:
            column:
            target_label:
            initial_step:
            final_step:
        """        
        ratio_file = os.path.join(self.data_reader.data_dir,name)
        if os.path.exists(ratio_file):
            mae = np.loadtxt(ratio_file)            
            if len(mae.shape)==1:
                mae = mae.reshape(-1,1)
        else:
            if final_step==-1:
                final_step = self.data_reader.data_info['final_step']
            mae = np.zeros((final_step+1,len(arrays_config_file)-1))
            for step in range(initial_step,final_step+1):
                print('Step',step)
                
                # Ground-truth
                # Fronteira
                gt_bound = self.data_reader.get_step_measures(
                    step,arrays_config_file[0],section='labels')
                gt_bound = gt_bound==comparative_label
                
                # Normal
                array_0 = self.data_reader.get_step_measures(
                    step,arrays_config_file[0],section=section)

                mae_pw = MeanAbsoluteErrorPW(reduction='none')

                # Previsões
                for i in range(1,len(arrays_config_file)):
                    array_i = self.data_reader.get_step_measures(
                        step,arrays_config_file[i],section=section)
                    mae[step,i-1] = mae_pw(array_0,array_i).numpy()[gt_bound].mean()
            
            np.savetxt(mae_file,mae,fmt='%.6f')

        return mae.T          

    def amount_points(self):
        """ 
        Quantidade de pontos por frame.
        """
        steps = self.data_reader.find_available_steps()
        if len(steps)==0:
            print("No prediction files found!")
            return          
        
        amount = []
        for step in tqdm(steps, desc='Processing step files: '):
            particles = self.data_reader.get_step(step)
            amount.append(particles.shape[0])
        
        return np.array(amount).reshape(-1,1)
            
    def distribution_per_curvatures(self,gt_config_file,sections=None,output_dir=None,
        extension='csv',enable_plot=False,return_report=False):
        """
        Calcula a quantidade de pontos que estão em cada intervalo de curvatura.
        Última modificação: 02/09/2022.
        
        Args:            
            gt_config_file: caminho completo do arquivo de configuração do ground-truth.
            section: nomes da seções de fronteira e curvatura no arquivos de configuração do ground-truth.
            output_dir: diretório de saída do relatório de métricas de acurácia.
            extension: extensão de saída do relatório métricas de acurácia.
            enable_plot: se verdadeiro, plota os gráficos de resultados.
            return_report: se verdadeiro, retorna os resultados obtidos.
        """
        gt_config = ConfigReader(gt_config_file)
        gt_config_dict = gt_config.get_section(sections[0])
        
        if output_dir==None:
            output_dir = os.path.dirname(gt_config_file)
        
        if extension=='csv':
            report_file = os.path.join(output_dir,'distribution_per_curvatures.csv')
        elif extension=='npz':
            report_file = os.path.join(output_dir,'distribution_per_curvatures.npz')
        
        if os.path.exists(report_file):      
            if extension=='csv':
                df = pd.read_csv(report_file)
                report = df.to_numpy()
                bins = report[:,0]
                distribution = report[:,1:].T

            elif extension=='npz':
                report = np.load(report_file)
                bins = report['bins']
                distribution = report['distribution']
        else:
            steps_path = os.path.join(output_dir,gt_config_dict['dir'],
                f"{gt_config_dict['base_name']}*.{gt_config_dict['extension']}")
            steps = self.data_reader.find_available_steps(steps_path)
            if len(steps)==0:
                print("No prediction files found!")
                return

            res = 0.1
            limits = [-1.0,1.0]
            bins = np.arange(limits[0],limits[1]+res,res)
            bins = bins.round(2)
            distribution = np.zeros((len(steps),bins.shape[0]),dtype=np.int)            

            num_particles = np.zeros((len(steps),))

            for k,step in enumerate(tqdm(steps, desc='load steps: ')):
                #print('step {}'.format(step))                                
                # Ground-truth
                # Fronteira
                gt = self.data_reader.get_step_labels(
                    step,gt_config_file,section=sections[0])
                
                num_particles[k] = gt.shape[0]

                # Curvatures
                curvatures = self.data_reader.get_step_measures(
                    step,gt_config_file,section=sections[1])
                curvatures = curvatures[:,0]            

                # Particiona os dados por intervalo de curvatura
                limits_ = np.array(limits).reshape(1,2)
                voxelizer = SparseVoxelizer(limits_,res=res,expand_limits=False)
                voxelizer.set_points(curvatures.reshape(-1,1))
                non_empty_bins = voxelizer.find_unique_voxels().reshape(-1)
                indices_per_bin = voxelizer.find_points_per_voxel()

                for i in range(non_empty_bins.shape[0]):
                    distribution[k,non_empty_bins[i]] = indices_per_bin[i].shape[0]                

            distribution[:,0] = 0

            if extension=='csv':
                columns = ['bins'] + [f'step-{step}' for step in steps]
                array = np.concatenate([bins[np.newaxis].T,distribution.T],axis=1)
                df = pd.DataFrame(array,columns=columns)
                df.to_csv(report_file,index=False,header=True)                
            elif extension=='npz':
                np.savez(report_file,bins=bins,distribution=distribution)

        # Total and percentages of points per curvature intervals
        
        total_per_bin = distribution.sum(axis=0,keepdims=True)
        ratios = np.nan_to_num(distribution/total_per_bin)

        res = bins[1]-bins[0]
        bins_str =  [f'[{bins[i]},{bins[i+1]}]' for i in range(bins.shape[0]-1)]
        bins_str = np.array(bins_str)
        #for i,bin in enumerate(bins_str):
        #    print(f'{i}: {bin}')    

        if enable_plot:         
            plt.figure(figsize=(10,6))
            plt.bar(bins,total_per_bin[0],res/2)
            plt.ylabel('Total',fontdict={'fontsize':15})
            plt.xlabel('Curvature Intervals',fontdict={'fontsize':15})
            plt.title('Number of particles per curvature intervals',fontdict={'fontsize':15})
            plt.xticks(bins[:-1])
            plt.xlim(-1.0,1.0)
            plt.grid(axis='y')           
            plt.savefig(os.path.join(output_dir,'distribution_per_curvatures.png'))
            plt.show()

            plt.figure(figsize=(10,6))
            plt.bar(bins,100*total_per_bin[0]/total_per_bin[0].sum(),res/2)
            plt.ylabel('Percentage',fontdict={'fontsize':15})
            plt.xlabel('Curvature Intervals',fontdict={'fontsize':15})
            plt.title('Percentage of particles per curvature intervals',fontdict={'fontsize':15})
            plt.xticks(bins[:-1])
            plt.xlim(-1.0,1.0)
            plt.grid(axis='y')           
            plt.savefig(os.path.join(output_dir,'percentage_per_curvatures.png'))
            plt.show()            
                
        if return_report:
            return {'bins':bins,'distribution':distribution,'total_per_bin':total_per_bin[0],'ratios':ratios}            
