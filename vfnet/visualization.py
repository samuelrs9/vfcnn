import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sim_reader.config import ConfigReader
from losses.custom_losses import *

from voxelizer.visualization import GridView

class SimView(GridView):
        
    def __init__(self,data_reader=None,ndim=None,num_figs=None):
        """ 
        Construtor. 
        Ùltima modificação: 16/07/2021
        
        Args:
            data_reader=None:
            ndim:
            num_figs:
        """
        super().__init__(data_reader,ndim,num_figs) 
 
    def view_simulation(self,labels_config_file=None):
        """ 
        Visualiza a simulação.
        Última modificação: 07/03/2022.
        
        Args:        
            labels_config_file: arquivo de configuração de labels.
        """ 
        #fig, ax = plt.subplots()
        while self.data_reader.current_step < self.data_reader.data_info['final_step']:
            # Clear the current axes.
            self.ax.cla()
            # Carrega as particulas
            particles,step = self.data_reader.get_next_step()           
            # Carrega os rótulos e plota
            if labels_config_file != None:
                labels_config = ConfigReader(labels_config_file)
                labels_config = labels_config.get_labels_config()
                labels = self.data_reader.get_step_labels(step,labels_config_file)
                for label in np.unique(labels):       
                    if self.ndim==2:
                        self.ax.scatter(particles[labels==label,0],particles[labels==label,1])                 
                    elif self.ndim==3:
                        self.ax.scatter(particles[labels==label,0],particles[labels==label,1],particles[labels==label,2])  
                plt.legend(labels_config['names'])
            else:
                if self.ndim==2:
                    self.ax.scatter(particles[:,0],particles[:,1],c='b')
                elif self.ndim==3:
                    self.ax.scatter(particles[:,0],particles[:,1],particles[:,2],c='b')
            self.ax.set_title("frame {}".format(step))
            # Define os limites dos eixos
            
            self.ax.set_xlim(self.limits[0],self.limits[1])
            self.ax.set_ylim(self.limits[2],self.limits[3])                
            if self.ndim==3:
                self.ax.set_zlim(self.limits[4],self.limits[5])             
            #plt.axis('equal')
            # Pausa
            plt.pause(0.1)
        
    def draw_boxplot(self,data,edge_color,fill_color,pos=0,axis=0):
        """ 
        Desenha um boxplot.
        Última atualização: 26/07/2021.
        
        Args:
            data:
            edge_color:
            fill_color:
            pos:
            axis:
        """
        
        bp = self.ax[axis].boxplot(data,positions=[pos],whis=[0,100],
                                   showfliers=False,patch_artist=True)
        
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element],color=edge_color,linewidth=3)
    
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color) 
            
    def draw_text(self,x,y,text,fontsize=12):
        """ 
        Plota um vetor de string.
        Última atualização: 05/08/2021.
        
        Args:
            x:
            y:
            text:
        """
        for i in range(len(text)):
            self.ax.text(x[i],y[i],text[i],fontsize=fontsize)

    def draw_vectors(self,vectors_config_file,gt_config_file=None,initial_step=0,final_step=-1,pause=0.1):
        """
        Desenha vetores 2d. 
        Última modificação: 29/03/2022.

        Args:
            vectors_config_file:
            gt_config_file:
            initial_step:
            final_step:
        """
        if final_step==-1:
            final_step = self.data_reader.data_info['final_step']
        for step in range(initial_step,final_step+1):
            print('Step',step)
            particles = self.data_reader.get_step(step)

            plt.cla()
            plt.scatter(particles[:,0],particles[:,1])
            colors = ['r','b']
            for i,config_file in enumerate(vectors_config_file):
                vectors = self.data_reader.get_step_measures(
                    step,config_file,section='normal')
                if gt_config_file is None:
                    X = particles[:,0]
                    Y = particles[:,1]
                    U = vectors[:,0]
                    V = vectors[:,1]       
                else:
                    gt_labels = self.data_reader.get_step_labels(
                        step,gt_config_file,section='labels')
                    gt_bound = gt_labels==1                    
                    X = particles[gt_bound,0]
                    Y = particles[gt_bound,1]
                    U = vectors[gt_bound,0]
                    V = vectors[gt_bound,1]                                 
                plt.quiver(X,Y,U,V,color=colors[i],scale_units='inches',scale=2)
            #plt.xlim(self.limits[0],self.limits[1])
            #plt.ylim(self.limits[2],self.limits[3])                
            plt.pause(pause)

    def plot_error_histogram(self,arrays_config_file,section,
        comparative_label=1,initial_step=0,final_step=-1,pause=0.1):
        """
        Plota histograma de erro entre dois arrays.
        Última modificação: 07/04/2022.

        Args:
            arrays_config_file:
            section:
            label:
            initial_step:
            final_step:
        """
        if final_step==-1:
            final_step = self.data_reader.data_info['final_step']
        for step in range(initial_step,final_step+1):
            print('Step',step)

            plt.cla()
            gt_bound = self.data_reader.get_step_measures(
                step,arrays_config_file[0],section='labels')
            gt_bound = gt_bound==comparative_label

            array_0 = self.data_reader.get_step_measures(
                step,arrays_config_file[0],section=section)
            array_1 = self.data_reader.get_step_measures(
                step,arrays_config_file[1],section=section)
            
            mae = MeanAbsoluteErrorPW(reduction='none')
            
            bins = np.arange(0,1.05,0.05)

            plt.title(f'Step {step}')
            plt.hist(mae(array_0,array_1).numpy()[gt_bound],bins,range=[0,1])
            plt.xlim(0,1)
            plt.ylim(0,gt_bound.sum())
            plt.xticks(bins)
            plt.yticks(np.arange(0,gt_bound.sum(),20))
            plt.xlabel("Mean Absolute Error")
            plt.ylabel("Quantidade")
            plt.grid(True)
            plt.pause(pause)
        
    def line_chart(self,X=None,Y=None,legend=None,axis_label=None,limits=None,title=None,show_grid=True):
        """ 
        Desenha gráficos de linhas.
        Última atualização: 18/04/2022.

        Args:
            X:
            Y:
            legend:
            axis_label:
            limits:
            title:
            show_grid:
        """
        if X is None:                
            for y in Y:
                plt.plot(y)
        else:
            for x,y in zip(X,Y):
                plt.plot(x,y)
        
        if axis_label is not None:
            plt.xlabel(axis_label[0])
            plt.ylabel(axis_label[1])
        if limits is not None:
            plt.xlim(limits[0])
            plt.ylim(limits[1])    
        if title is not None:
            plt.title(title)
        if legend is not None:
            plt.legend(legend)
        if show_grid:
            plt.grid(True)
        plt.show()