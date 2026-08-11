import os
import glob
from warnings import WarningMessage
import numpy as np
import pandas as pd

from .config import ConfigReader

class DataReader:
    
    def __init__(self,config_file):
        """ 
        Construtor.
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError("Data configuration file not found!")

        if not os.path.isfile(config_file):
            raise Exception("Config file passed as argument is not a file!")

        self.config_file = config_file
        self.data_dir = os.path.dirname(config_file)
        self.config = ConfigReader(self.config_file)
                
        config_sections =  self.config.get_all_sections()
        self.properties_info = config_sections['simulation_properties']
        self.data_info = config_sections['data']

        self.frames_dir = os.path.join(self.data_dir,self.data_info['frames_dir'])
        
        self.current_step = -1
                        
    def get_next_step(self,get_particles=True):
        """ 
        Obtem os dados do próximo frame da simulação. 
        Última modificação: 24/05/2021.
        
        Args:
            get_gt: flag para decidir se método vai carregar o grount truth de
                    partículas de fronteira e interior.
        """        
        self.current_step += 1
        if get_particles:
            particle_coords = self.get_step(self.current_step)
            return particle_coords,self.current_step
        else:
            return self.current_step
                    
    def get_step(self,step,attribute='coords'):
        """ 
        Obtem atributo de pontos de um frame da simulação.
        Última modificação: 23/08/2022.
        
        Args:
            step: um inteiro entre 0 e o índice do último frame da simulação.
            attribute: atributo a ser carregado.
                    
        Returns:
            point_attribute: atributo de pontos.

        """        
        base_sep = self.data_info['base_sep']
        base_name = self.data_info['base_name']
        extension = self.data_info['extension']    
        if 'header_lines' in self.data_info:
            header_lines = self.data_info['header_lines']
        else:
            header_lines = 0
        
        frame_file = os.path.join(
            self.frames_dir,f'{base_name}{base_sep}{step}.{extension}')
        if extension=='csv':
                
            if self.data_info['from'] == 'dualsphysics':
                step_str = '{:4.0f}'.format(step).replace(' ','0')
                frame_file = os.path.join(
                    self.frames_dir,f'{base_name}_{step_str}.{extension}')
                frame_pd = pd.read_csv(frame_file,header=2,sep=';')

            else:
                frame_pd = pd.read_csv(frame_file)          
                              
            attribute = self.data_info[attribute]
            point_attribute = frame_pd[attribute].to_numpy()
            
        elif extension=='txt' or extension=='dat':
            with open(frame_file,'r') as file:
                header = ''
                for i in range(header_lines):
                    header += file.readline()
                point_attribute = np.loadtxt(file)        

        return point_attribute

    def get_step_labels(self,step,config_file,section='labels',
        options=None):
        """ 
        Get the point labels of a step.
        Last modified: 02/21/2022.
        
        Args:
            step:
            config_file:
            section:
            options:
                    
        Returns: 
            step data.

        """
        return self.get_step_array(step,config_file,
            section=section,options=options, dtype=int)

    def get_step_measures(self,step,config_file,section='measures',
        options=None,columns=None):
        """ 
        Get the point labels of a step.
        Last modified: 02/21/2022.
        
        Args:
            step:
            config_file:
            section:
            options:
            columns:
                    
        Returns: 
            step data.

        """        
        return self.get_step_array(step,config_file,
            section=section,options=options,columns=columns,dtype=float)

    def get_step_array(self,step,config_file,section=None,
        options=None,columns=None,dtype=int):
        """ 
        Get the array data from a step.
        Last modified: 02/19/2022.
        
        Args:
            step:
            config_file:
            section:
            options:
            columns:
            dtype:
                    
        Returns: 
            step data.

        """
        config = ConfigReader(config_file)
        config_file_dir = os.path.dirname(config_file)

        if options==None:
            options = ['dir','base_name','extension','columns']

        section_dict = config.get_section(section,options,warnings=False)

        if 'dir' in section_dict:
            dir = os.path.join(config_file_dir,section_dict['dir'])
        if 'base_name' in section_dict:
            base_name = section_dict['base_name']
        if 'extension' in section_dict:
            extension = section_dict['extension']
            if extension == 'csv' and columns is None:
                if 'columns' not in section_dict:
                    raise ValueError("Data extension is 'csv' but 'columns' were not specified!")
            
        file = os.path.join(dir,f'{base_name}.{step}.{extension}')
        array = self.get_array(file,columns=columns,dtype=dtype)

        return array

    def get_array(self,file,columns=None,dtype=int):
        """ 
        Get array data from a specific file.
        Last modified: 02/19/2022.
        
        Args:
            file:
            type:
                    
        Returns: 
            data array.

        """
        extension = os.path.splitext(file)[1]
        if extension == '.txt':
            return np.loadtxt(file,dtype=dtype)
        elif extension == '.dat':
            return np.loadtxt(file,dtype=dtype)
        elif extension == '.npy':
            return np.load(file,allow_pickle=True)
        elif extension == '.csv':
            array = pd.read_csv(file)
            return  array[columns].to_numpy()         
        else:
            print('Error: unsupported extension!')        

    def find_available_steps(self,path=None):
        """" 
        Encontra passos de simulação disponíveis.
        Última atualização: 07/07/2022.

        Returns:
            steps:
        """
        if path is None:
            path = os.path.join(self.data_dir,self.data_info['frames_dir'],
            f"{self.data_info['base_name']}{self.data_info['base_sep']}*.{self.data_info['extension']}")
        step_files =  glob.glob(path)
        steps = []
        for step_file in step_files:
            step = step_file.split('.')[-2]
            step = step.split(self.data_info['base_sep'])[-1]
            try:
                step = int(step)
                steps.append(int(step))
            except:
                    print(f"Warning: {step_file} is not a valid step file!")
        arg_sort = np.argsort(steps)
        step_files = np.array(step_files)[arg_sort]
        steps = np.array(steps)[arg_sort]
        return dict(zip(steps,step_files))

if __name__=='__main__':
    current_dir = os.path.dirname(__file__)
    config_file = os.path.join(current_dir,'sim_config_example.yaml')
    data_reader = DataReader(config_file)
    print('properties info: ',data_reader.properties_info)
    print('data info: ',data_reader.data_info)