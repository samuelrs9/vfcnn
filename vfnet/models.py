import os
import sys

import h5py
import datetime
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
import tensorflow_io as tfio
import tensorflow_addons as tfa

from tqdm import tqdm
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.callbacks import TensorBoard

import tensorflow_graphics.geometry.transformation as tfg

from metrics.classification import Report
from cnn_models import boundary_region_3d
from cnn_models.custom_layers import Normalize, ArgMax
from cnn_models import normal_region_3d

from metrics.custom_metrics import *
from losses.custom_losses import *

from losses.boundary_losses import BoundaryLossRW
from losses.normal_losses import NormalLossRW
from losses.sdf_losses import SDFLoss
from losses.combined_losses import BoundaryNormalLossRW

from cnn_models.sparse_models import SparseVoxelizedFluidCNN

from sim_reader.config import ConfigReader
from scipy.spatial.transform import Rotation


class ModelManager:

    def __init__(self,data_dir=None,dataset_config_file=None):
        """
        Construtor.
        Última modificação: 02/03/2022.
        
        Args:
            data_dir:
            dataset_config_file:

        """
        self.data_dir = data_dir
        self.dataset_config_file = dataset_config_file

        self.dataset_dir = None
        self.general_config = None
        self.train_config = None
        self.val_config = None
        self.approach_dir = None

        self.model = None
        
        self.train_dir = None
        self.val_dir = None

        self.train_set = None
        self.val_set = None

        self.data_augmentation = False
        self.transfer_learning = False
        self.model_template = None
        self.pretrained_model_template = None

        if self.dataset_config_file != None:
            self.dataset_dir = os.path.dirname(dataset_config_file)
            self.dataset_config_file = dataset_config_file

            dataset_config = self.get_dataset_config(dataset_config_file)
            self.general_config = dataset_config[0]
            self.train_config = dataset_config[1]
            self.val_config = dataset_config[2]
        
        if self.data_dir != None and self.dataset_config_file != None:
            if self.general_config['approach']=='regionwise':
                self.approach_dir = os.path.join(self.data_dir,'regionwise_approach') 
            elif self.general_config['approach']=='sparse_regionwise':
                self.approach_dir = os.path.join(self.data_dir,'sparse_regionwise_approach')
                   
            os.makedirs(self.approach_dir,exist_ok=True)

        self.physical_devices = tf.config.list_physical_devices('GPU')
        if len(self.physical_devices)>0:
            tf.config.experimental.set_memory_growth(self.physical_devices[0],True)            
        tf.config.run_functions_eagerly(True)
        self.autotune = tf.data.experimental.AUTOTUNE

    def get_dataset_config(self,dataset_config_file):
        """ 
        Carrega configurações de dataset.
        Última atualização: 24/02/2022.

        Args:
            dataset_config_file:

        """
        if not os.path.exists(dataset_config_file):
            raise FileNotFoundError('Dataset configuration file not found!')
        
        dataset_config = ConfigReader(dataset_config_file)

        # General config
        gen_config = dataset_config.get_section('general')
        gen_config['labels'] = gen_config['labels'].split()
        gen_config['spatial_dimensions'] = int(gen_config['spatial_dimensions'])
        gen_config['num_classes'] = int(gen_config['num_classes'])
        try:
            gen_config['border_size'] = int(gen_config['border_size'])
            gen_config['image_shape'] = [
                int(x) for x in gen_config['image_shape'].split()]
            gen_config['grid_size'] = [
                int(x) for x in gen_config['grid_size'].split()]
            gen_config['image_length'] = float(gen_config['image_length'])
            gen_config['search_radius'] = float(gen_config['search_radius'])
            gen_config['grid_length'] = float(gen_config['grid_length'])
            gen_config['used_steps'] = int(gen_config['used_steps'])

        except:
            print("TO DO: Implement support for multiple simulation specifications!!!")
        gen_config['dir'] = os.path.dirname(dataset_config_file)

        self.general_config = gen_config

        # Train set config
        train_config = dataset_config.get_section('train_set')
        train_config['keys'] = train_config['keys'].split()
        if 'num_samples' in train_config:
            train_config['num_samples'] = int(train_config['num_samples'])
        if 'num_batches' in train_config:
            train_config['num_batches'] = int(train_config['num_batches'])            
        train_config['file'] = os.path.join(
            self.general_config['dir'],
            f"{train_config['name']}.{train_config['format']}")
        self.train_config = train_config

        # Validation set config
        val_config = dataset_config.get_section('validation_set')
        val_config['keys'] = val_config['keys'].split()
        if 'num_samples' in val_config:
            val_config['num_samples'] = int(val_config['num_samples'])
        if 'num_batches' in train_config:
            val_config['num_batches'] = int(val_config['num_batches']) 
        val_config['file'] = os.path.join(
            self.general_config['dir'],
            f"{val_config['name']}.{val_config['format']}")

        return gen_config, train_config, val_config

    def get_model_config(self,model_config_file):
        """ 
        Carrega configurações de modelo.
        Última atualização: 02/03/2022.

        Args:
            model_config_file:

        """
        if not os.path.exists(model_config_file):
            raise FileNotFoundError('Model config file not found!')

        config = ConfigReader(model_config_file)

        # Model config
        model_config = config.get_section(
            'model',['model_id','model_name','architecture_file',
            'weights_file','approach','model_template',
            'pretrained_model_template','input_shape',
            'border_size','image_length','spatial_dimensions',])
        model_config['input_shape'] = [
            int(x) for x in model_config['input_shape'].split()]      
        model_config['image_length'] = float(model_config['image_length'])  
        model_config['border_size'] = int(model_config['border_size'])

        # Training config
        training_config = config.get_section(
            'training',['epochs','learning_rate','optimizer',
            'transfer_learning','transfer_learning_source'])
        training_config['epochs'] = int(training_config['epochs'])
        training_config['learning_rate'] = float(training_config['learning_rate'])

        # Dataset 
        dataset_config = config.get_section(
            'dataset',['config_file','train_file','validation_file'])

        return model_config, training_config, dataset_config  

    def get_dataset_config_from_hdf5(self,dataset_file):
        """
        Carrega configurações do dataset a partir de um arquivo hdf5.
        Última atualização: 03/03/2022.
        
        Args:
            dataset_file:
        """
        if os.path.splitext(dataset_file)[1]!='.hdf5':
            raise FormatError(
                "Invalid dataset format. Only hdf5 format is supported!")

        dataset_config = {}
        dataset_config['file'] = dataset_file
        with h5py.File(dataset_file,'r') as hf:
            for attr in hf.attrs.keys():
                dataset_config[attr] = hf.attrs[attr]
        return dataset_config

    def load_model(self,model_config_file,custom_layers=None,return_model=False,return_model_config=False,):
        """ 
        Carrega um modelo a partir de um arquivo JSON.
        Última modificação: 07/03/2022.
        
        Args:
            model_config_file:
            custom_layers:
            return_model:
            return_model_config:
                
        Returns:
            model e model_config.

        """
        model_dir = os.path.dirname(model_config_file)
        model_config,_,_ = self.get_model_config(model_config_file)

        arch_file = os.path.join(model_dir,model_config['architecture_file'])
        with open(arch_file, 'r') as f:
            arch_json = f.read()
            model = tf.keras.models.model_from_json(arch_json,custom_objects=custom_layers)
        
        weights_file = os.path.join(model_dir,model_config['weights_file'])
        model.load_weights(weights_file)

        print("\nLoaded model from disk")

        self.model = model

        if not return_model and not return_model_config:
            return
        elif return_model and not return_model_config:
            return model
        elif not return_model and return_model_config:
            return model_config
        elif return_model and return_model_config:
            return model,model_config
        
    def save_model(self,model_dir,base_name):
        """ 
        Salva um modelo de rede no formato JSON.
        Última modificação: 22/02/2022.
        
        Args:
            model_dir:
            base_name:
                
        Returns:
            model e model_config.

        """
        # Salva a arquitetura
        arch_file = os.path.join(model_dir,f'{base_name}.json')
        with open(arch_file, 'w') as json_file:
            json_file.write(self.model.to_json())

        # Salva os pesos
        weights_file = os.path.join(model_dir,f'{base_name}.h5')
        self.model.save_weights(weights_file)

        print(f'\nModel {arch_file} has been saved successfully!')


class VFRWCNN(ModelManager):
    
    def __init__(self,tasks=['boundary'],data_dir=None,dataset_config_file=None):
        """
        Construtor.
        Última modificação: 31/03/2022.  
        
        Args:
            tasks:
            data_dir:
            approach_dir:
            dataset_config_file:
            model_config_file:
        """
        super().__init__(data_dir,dataset_config_file)
        self.tasks = tasks
            
    def set_model(self,model=None,pretrained_model_config_file=None,
        from_template=None,transfer_learning=False,trainable_weigths=False):
        """
        Define o modelo de CNN que será treinado.
        Última atualização: 02/03/2022.

        Args:
            model:
            pretrained_model_config_file:
            from_template:
            transfer_weights:
            trainable_weigths:

        """
        if model is None and from_template is None:
            raise ValueError(
                "Specify one model! Use 'model' or 'from_template' keyword "
                "argument for this! Template modules can be found at "
                "boundary_detector.models.")
        if model is not None and from_template is not None:
            raise ValueError(
                "Specify only one model! 'model' and 'from_template'" 
                " are not allowed at the same time!")        

        if model is not None:
            self.model = model

        if from_template is not None:
            self.model_template = from_template
            if transfer_learning is True:
                self.transfer_learning = True
                if pretrained_model_config_file is None:
                    raise ValueError(
                        "'pretrained_model_config_file' is None, specify a "
                        "pre-trained model config file!")
                if 'normal' in self.tasks:
                    custom_layers = {'Normalize':Normalize,'ArgMax':ArgMax}
                else:
                    custom_layers = None
                pretrained_model, model_config = self.load_model(
                    pretrained_model_config_file, 
                    custom_layers = custom_layers,
                    return_model=True,
                    return_model_config = True)

                print('\nModelo pré treinado da abordagem pontual: ')
                pretrained_model.summary()    
                self.pretrained_model_template = model_config['model_template']

                args = (
                    "num_classes = self.general_config['num_classes'],"
                    "input_shape = self.general_config['image_shape'],"
                    "pre_trained_model = pretrained_model,"
                    "transfer_weights = True,"
                    "trainable_weigths = True")
            else:
                args = (
                    "num_classes = self.general_config['num_classes'],"
                    "input_shape = self.general_config['image_shape']")                            
            self.model = eval(f"{from_template}({args})")

    def train_input_pipeline(self,trainset_file=None,batch_size=32,
        buffer_size_factor=10,data_augmentation=False,debug_mode=False):
        """ 
        Pipeline de entrada do dataset de voxels de treino.
        Última modificação: 01/04/2022.
        
        Args:
            trainset_file:
            buffer_size_factor:
            data_augmentation:
            debug_mode:

        """        
        if debug_mode and float(tf.__version__[:-2])>=2.5:
            tf.data.experimental.enable_debug_mode()

        if trainset_file is not None:
            if not os.path.exists(trainset_file):
                raise FileNotFoundError('Training set file not found!')
            else:
                self.train_config = self.get_dataset_config_from_hdf5(trainset_file)
                if self.general_config is None:
                    self.general_config = {}
                    self.general_config['image_shape'] = self.val_config['image_shape'].tolist()
                    self.general_config['num_classes'] = self.train_config['num_classes']
                    self.general_config['spatial_dimensions'] = self.train_config['spatial_dimensions']
                
        self.train_config['batch_size'] = batch_size
        self.train_config['steps'] = 1 + self.train_config['num_samples'] // self.train_config['batch_size']
        self.data_augmentation = data_augmentation

        with h5py.File(self.train_config['file'],'r') as hf:
            full_num_voxels = hf['full_num_voxels'][:]
            full_voxels_coord = hf['full_voxels_coord'][:]
            
            target_num_voxels = hf['target_num_voxels'][:]
            target_voxels_coord = hf['target_voxels_coord'][:]

            if 'boundary' in self.tasks:
                target_labels = hf['target_boundary'][:]
            if 'normal' in self.tasks:
                target_normal = hf['target_normal'][:]

            self.train_size = hf['full_num_voxels'].shape[0]
            print(f'The training dataset contains {self.train_size} images.')
       
        # Faz o split dos arrays
        full_voxels_coord = tf.RaggedTensor.from_row_lengths(
            full_voxels_coord,full_num_voxels)
        target_voxels_coord = tf.RaggedTensor.from_row_lengths(
            target_voxels_coord,target_num_voxels)

        if 'boundary' in self.tasks:
            target_labels = tf.RaggedTensor.from_row_lengths(
                target_labels,target_num_voxels)
        if 'normal' in self.tasks:
            target_normal = tf.RaggedTensor.from_row_lengths(
                target_normal,target_num_voxels)

        # Agrupa os datasets
        full_voxels_coord = tf.data.Dataset.from_tensor_slices(
            full_voxels_coord)
        target_voxels_coord = tf.data.Dataset.from_tensor_slices(
            target_voxels_coord)

        if 'boundary' in self.tasks and 'normal' in self.tasks:
            target_set = tf.data.Dataset.from_tensor_slices(
                (target_labels,target_normal))
        else:
            if 'boundary' in self.tasks:
                target_set = tf.data.Dataset.from_tensor_slices(
                    target_labels)
            if 'normal' in self.tasks:
                target_set = tf.data.Dataset.from_tensor_slices(
                    target_normal)                
        train_set =  tf.data.Dataset.zip(
            (full_voxels_coord,target_voxels_coord,target_set)) 

        if debug_mode:
            for i,sample in enumerate(train_set):
                if i<5:
                    print(f'Sparse sample {i} (Before processing)')
                    print(' --> full_voxels_coord shape: ',sample[0].shape)
                    print(' --> target_voxels_coord shape: ',sample[1].shape)
                    if 'boundary' in self.tasks and 'normal' in self.tasks:
                        print(' --> target_labels shape: ',sample[2][0].shape) 
                        print(' --> target_normal shape: ',sample[2][1].shape) 
                    else:
                        if 'boundary' in self.tasks:
                            print(' --> target_labels shape: ',sample[2].shape)
                        elif 'normal' in self.tasks:
                            print(' --> target_normal shape: ',sample[2].shape) 
                else:
                    break

        # Aplica alguns processamentos
        buffer_size = buffer_size_factor*batch_size
        train_set = train_set.map(self.process_train,num_parallel_calls=self.autotune)
        train_set = train_set.shuffle(buffer_size=buffer_size)
        train_set = train_set.repeat()
        train_set = train_set.batch(batch_size)
        #train_set = train_set.apply(
        #            tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
        train_set = train_set.prefetch(buffer_size=self.autotune)
        
        if debug_mode:
            for i,(images,target_mask) in enumerate(train_set):
                if i<5:
                    print(f'Batch {i}: (After processing)')
                    print(' --> images shape: ',images.shape)
                    print(' --> target mask shape: ',target_mask.shape)
                else:
                    break

        self.train_set = train_set
 
    def val_input_pipeline(self,valset_file=None,batch_size=32,
        debug_mode=False):
        """ 
        Pipeline de entrada do dataset de voxels de treino.
        Última modificação: 01/04/2022.
        
        Args:    
            valset_file:
            batch_size:     
            debug_mode:

        """        
        if debug_mode and float(tf.__version__[:-2])>=2.5:
            tf.data.experimental.enable_debug_mode()

        if valset_file is not None:
            if not os.path.exists(valset_file):
                raise FileNotFoundError('Validation set file not found!')
            else:
                self.val_config = self.get_dataset_config_from_hdf5(valset_file)
                if self.general_config is None:
                    self.general_config = {}
                    self.general_config['image_shape'] = self.val_config['image_shape'].tolist()
                    self.general_config['num_classes'] = self.val_config['num_classes']
                    self.general_config['spatial_dimensions'] = self.val_config['spatial_dimensions']
                
        self.val_config['batch_size'] = batch_size
        self.val_config['steps'] = 1 + self.val_config['num_samples'] // self.val_config['batch_size']

        with h5py.File(self.val_config['file'],'r') as hf:
            full_num_voxels = hf['full_num_voxels'][:]
            full_voxels_coord = hf['full_voxels_coord'][:]
            
            target_num_voxels = hf['target_num_voxels'][:]
            target_voxels_coord = hf['target_voxels_coord'][:]

            if 'boundary' in self.tasks:
                target_labels = hf['target_boundary'][:]
            if 'normal' in self.tasks:
                target_normal = hf['target_normal'][:]

            self.val_size = hf['full_num_voxels'].shape[0]
            print(f'The validation dataset contains {self.val_size} images.')
       
        # Faz o split dos arrays
        full_voxels_coord = tf.RaggedTensor.from_row_lengths(
            full_voxels_coord,full_num_voxels)
        target_voxels_coord = tf.RaggedTensor.from_row_lengths(
            target_voxels_coord,target_num_voxels)

        if 'boundary' in self.tasks:
            target_labels = tf.RaggedTensor.from_row_lengths(
                target_labels,target_num_voxels)
        if 'normal' in self.tasks:
            target_normal = tf.RaggedTensor.from_row_lengths(
                target_normal,target_num_voxels)

        # Agrupa os datasets
        full_voxels_coord = tf.data.Dataset.from_tensor_slices(
            full_voxels_coord)
        target_voxels_coord = tf.data.Dataset.from_tensor_slices(
            target_voxels_coord)

        if 'boundary' in self.tasks and 'normal' in self.tasks:
            target_set = tf.data.Dataset.from_tensor_slices(
                (target_labels,target_normal))
        else:
            if 'boundary' in self.tasks:
                target_set = tf.data.Dataset.from_tensor_slices(
                    target_labels)
            if 'normal' in self.tasks:
                target_set = tf.data.Dataset.from_tensor_slices(
                    target_normal)                
        val_set =  tf.data.Dataset.zip(
            (full_voxels_coord,target_voxels_coord,target_set)) 

        if debug_mode:
            for i,sample in enumerate(val_set):
                if i<5:
                    print(f'Sparse sample {i} (Before processing)')
                    print(' --> full_voxels_coord shape: ',sample[0].shape)
                    print(' --> target_voxels_coord shape: ',sample[1].shape)
                    if 'boundary' in self.tasks and 'normal' in self.tasks:
                        print(' --> target_labels shape: ',sample[2][0].shape) 
                        print(' --> target_normal shape: ',sample[2][1].shape) 
                    else:
                        if 'boundary' in self.tasks:
                            print(' --> target_labels shape: ',sample[2].shape)
                        elif 'normal' in self.tasks:
                            print(' --> target_normal shape: ',sample[2].shape) 
                else:
                    break

        # Aplica alguns processamentos
        val_set = val_set.map(self.process_validation,num_parallel_calls=self.autotune)
        val_set = val_set.batch(batch_size)
        val_set = val_set.prefetch(buffer_size=self.autotune)
        
        if debug_mode:
            for i,(images,target_mask) in enumerate(val_set):
                if i<5:
                    print(f'Batch {i}: (After processing)')
                    print(' --> images shape: ',images.shape)
                    print(' --> target mask shape: ',target_mask.shape)
                else:
                    break

        self.val_set = val_set

    def val_input_pipeline_bkp(self,valset_file=None,batch_size=32,
        val_size=None,debug_mode=False):
        """ 
        Pipeline de entrada do dataset de voxels de validação.        
        Última modificação: 24/02/2022.
        
        Args:    
            valset_file:
            batch_size:            
            val_size:
        """    
        if debug_mode:
            tf.data.experimental.enable_debug_mode()

        if valset_file is not None:
            if not os.path.exists(valset_file):
                raise FileNotFoundError('Validation set file not found!')
            else:
                self.val_config = self.get_dataset_config_from_hdf5(valset_file)           
                if self.general_config is None:
                    self.general_config = {}
                    self.general_config['image_shape'] = self.val_config['image_shape'].tolist()
                    self.general_config['num_classes'] = self.val_config['num_classes']
                    self.general_config['spatial_dimensions'] = self.val_config['spatial_dimensions']

        self.val_config['batch_size'] = batch_size
        self.val_config['steps'] = 1 + self.val_config['num_samples'] // self.val_config['batch_size']

        with h5py.File(self.val_config['file'],'r') as hf:
            full_num_voxels = hf['full_num_voxels'][:]
            full_voxels_coord = hf['full_voxels_coord'][:]
            
            target_labels = hf['target_labels'][:]
            target_num_voxels = hf['target_num_voxels'][:]
            target_voxels_coord = hf['target_voxels_coord'][:]

            self.val_size = hf['full_num_voxels'].shape[0]
            print(f'The validation dataset contains {self.val_size} voxels.')
    
        # Faz o split dos arrays
        full_voxels_coord = tf.RaggedTensor.from_row_lengths(full_voxels_coord,full_num_voxels)
        target_voxels_coord = tf.RaggedTensor.from_row_lengths(target_voxels_coord,target_num_voxels)
        target_labels = tf.RaggedTensor.from_row_lengths(target_labels,target_num_voxels)

        full_voxels_coord = tf.data.Dataset.from_tensor_slices(full_voxels_coord)
        target_voxels_coord = tf.data.Dataset.from_tensor_slices(target_voxels_coord)
        target_labels = tf.data.Dataset.from_tensor_slices(target_labels)
        
        val_set =  tf.data.Dataset.zip((full_voxels_coord,target_voxels_coord,target_labels))

        if debug_mode:
            for i,sample in enumerate(val_set):
                if i<5:
                    print(f'Sparse sample {i}')
                    print(' --> full_voxels_coord shape: ',sample[0].shape)
                    print(' --> target_voxels_coord shape: ',sample[1].shape)
                    print(' --> target_labels shape: ',sample[2].shape)
                else:
                    break            

        # Aplica alguns processamentos
        val_set = val_set.map(self.process_validation,num_parallel_calls=self.autotune)
        #val_set = val_set.repeat()
        val_set = val_set.batch(batch_size)
        val_set = val_set.prefetch(buffer_size=self.autotune)
        
        if debug_mode:
            for i,(images,labels_mask) in enumerate(val_set):
                if i<5:
                    print(f'Batch {i}: ')
                    print(' --> images shape: ',images.shape)
                    print(' --> labels mask shape: ',labels_mask.shape)
                else:
                    break

        self.val_set = val_set
    
    def train_model(self,train_id=0,num_epochs=10,learning_rate=1e-3,
        device='cpu',debug_mode=None):
        """ 
        Treina o modelo.
        Última modificação: 05/04/2022.
        
        Args:
            train_id:
            num_epochs:
            learning_rate: 
            device:                   
            debug_mode:    
    
        """   
        image_res =  self.general_config['image_shape'][0]
        border_size =  self.general_config['border_size']
        image_length = self.general_config['image_length']
        config_str = f'{image_res}_{border_size}_{image_length:.2f}'

        train_config_str = f'{config_str}_{self.model.name}_{train_id}'
        
        model_dir = os.path.join(
            self.approach_dir,'models',f'model_{train_config_str}')
        os.makedirs(model_dir,exist_ok=True)

        class DisplayCallback(tf.keras.callbacks.Callback):
            
            def on_epoch_end(self, epoch, logs=None):
              #show_predictions(dataset['train'],5)
              print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

        log_dir = os.path.join(model_dir,'logs')
        checkpoint_dir = os.path.join(model_dir,'checkpoint','checkpoint')
        exp_log_dir = os.path.join(log_dir,datetime.datetime.now().strftime("%d-%m-%Y [%Hh%Mm%Ss]"),"fit")

        try:
            self.model.load_weights(checkpoint_dir)
        except:
            print('Checkpoint de treino não pôde ser restaurado!')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(exp_log_dir, histogram_freq=1)
        
        Callbacks = [
            #DisplayCallback(),
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_dir, verbose=1, save_best_only=True, save_weights_only=True)]
        
        initial_learning_rate = learning_rate
        lr_schedule = ExponentialDecay(
            initial_learning_rate,decay_steps = 5,decay_rate = 0.9)
        #optimizer = SGD(learning_rate = lr_schedule)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        
        if self.general_config['spatial_dimensions']==2:
            slice_bce = {'begin':[0,0,0,0],'size':[-1,-1,-1,2]}
            slice_mse = {'begin':[0,0,0,2],'size':[-1,-1,-1,2]}
            slice_sdf_mse = {'begin':[0,0,0,4],'size':[-1,-1,-1,3]}
        elif self.general_config['spatial_dimensions']==3:
            slice_bce = {'begin':[0,0,0,0,0],'size':[-1,-1,-1,-1,2]}
            slice_mse = {'begin':[0,0,0,0,2],'size':[-1,-1,-1,-1,3]}

        if 'boundary' in self.tasks and len(self.tasks)==1:
            self.model.compile(
                optimizer = optimizer,
                loss = BoundaryLossRW(slice=slice_bce),
                metrics = [
                    PrecisionRW(slice=slice_bce),
                    RecallRW(slice=slice_bce),
                    F1ScoreRW(slice=slice_bce),
                    MatthewsCoefficientRW(slice=slice_bce)],
                run_eagerly=debug_mode)
        if 'normal' in self.tasks and len(self.tasks)==1:
            self.model.compile(
                optimizer = optimizer, 
                loss = [NormalLossRW()],
                metrics = [MeanSquaredErrorRW()],
                run_eagerly=debug_mode)
        elif 'boundary' in self.tasks and 'normal' in self.tasks and len(self.tasks)==2:
            self.model.compile(                
                optimizer = optimizer, 
                loss = BoundaryNormalLossRW(
                    slice={'bce':slice_bce,'mse':slice_mse},
                    slice_weights=slice_bce),
                metrics = [
                    RecallRW(slice=slice_bce,name='r'),
                    PrecisionRW(slice=slice_bce,name='p'),
                    F1ScoreRW(slice=slice_bce,name='f1'),
                    MatthewsCoefficientRW(slice=slice_bce,name='mcc'),
                    BinaryCrossEntropyRW(
                        slice=slice_bce,slice_weights=slice_bce,name='bce'),
                    MeanSquaredErrorRW(
                        slice=slice_mse,slice_weights=slice_bce,name='mse')],
                run_eagerly=debug_mode)
        elif len(self.tasks)==3:
            self.model.compile(                
                optimizer = optimizer, 
                loss = FullLoss(
                    slice={'bce':slice_bce,'mse':slice_mse,'sdf_mse':slice_sdf_mse},
                    slice_weights=slice_bce),
                metrics = [
                    RecallRW(slice=slice_bce,name='r'),
                    PrecisionRW(slice=slice_bce,name='p'),
                    F1ScoreRW(slice=slice_bce,name='f1'),
                    MatthewsCoefficientRW(slice=slice_bce,name='mcc'),
                    BinaryCrossEntropyRW(
                        slice=slice_bce,slice_weights=slice_bce,name='bce'),
                    MeanSquaredErrorRW(
                        slice=slice_mse,slice_weights=slice_bce,name='mse')],
                run_eagerly=debug_mode)            

        
        #tf.config.run_functions_eagerly(True)
        #tf.data.experimental.enable_debug_mode()

        if device == 'cpu':
            with tf.device("/cpu:0"):
                model_history = self.model.fit(
                    self.train_set,
                    epochs = num_epochs,
                    steps_per_epoch = self.train_config['steps'],
                    validation_steps = self.val_config['steps'],
                    validation_data = self.val_set,
                    callbacks = Callbacks,
                    workers = 28)
        elif device == 'gpu':
            with tf.device("/GPU:0"):
                model_history = self.model.fit(
                    self.train_set,
                    epochs = num_epochs,
                    steps_per_epoch = self.train_config['steps'],
                    validation_steps = self.val_config['steps'],
                    validation_data = self.val_set,
                    callbacks = Callbacks)

        #  Salva o modelo
        if self.general_config['spatial_dimensions'] == 2:
            base_name = 'BRegionCNN2D'
        elif self.general_config['spatial_dimensions'] == 3:
            base_name = 'BRegionCNN3D'
        self.save_model(model_dir,base_name)
        
        # Salva o arquivo de configuração
        model_config = {}
        model_config['model'] = {
            'model_id': train_id,
            'model_name': self.model.name,
            'architecture_file': f'{base_name}.json',
            'weights_file': f'{base_name}.h5',
            'approach': 'regionwise',
            'model_template': str(self.model_template),
            'pretrained_model_template': str(self.pretrained_model_template),
            'transfer_learning': str(self.transfer_learning),
            'input_shape': ' '.join([str(x) for x in self.general_config['image_shape']]),
            'border_size': self.general_config['border_size'],
            'image_length': self.general_config['image_length'],
            'spatial_dimensions': self.general_config['spatial_dimensions']}
        if 'boundary' in self.tasks and 'normal' not in self.tasks:            
            model_config['training'] = {
                'device': device,
                'epochs': num_epochs,
                'learning_rate': learning_rate,
                'optimizer': optimizer._name,
                'loss_function': self.model.loss,
                'metrics': ' '.join(self.model.metrics_names),
                'val_loss': self.model.history.history['val_loss'][-1], 
                'val_precision': self.model.history.history['val_precision'][-1],
                'val_recall': self.model.history.history['val_recall'][-1],
                'val_f1score': self.model.history.history['val_f1score'][-1],
                'val_mcc': self.model.history.history['val_mcc'][-1],
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints',}
        elif 'boundary' not in self.tasks and 'normal' in self.tasks:
            model_config['training'] = {
                'device': device,
                'epochs': num_epochs,
                'batch_size':self.train_config['batch_size'],
                'learning_rate': learning_rate,
                'optimizer': optimizer._name,
                'loss': self.model.loss,
                'metrics': ' '.join(self.model.metrics_names),
                'val_loss': self.model.history.history['val_loss'][-1],
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints'}
        elif 'boundary' in self.tasks and 'normal' in self.tasks:
            model_config['training'] = {
                'device': device,
                'epochs': num_epochs,
                'batch_size':self.train_config['batch_size'],
                'learning_rate': learning_rate,
                'optimizer': optimizer._name,
                'loss': self.model.loss,
                'metrics': ' '.join(self.model.metrics_names),
                'val_loss': self.model.history.history['val_loss'][-1],
                'val_precision': self.model.history.history['val_p'][-1],
                'val_recall': self.model.history.history['val_r'][-1],
                'val_f1score': self.model.history.history['val_f1'][-1],
                'val_mcc': self.model.history.history['val_mcc'][-1],
                'val_bce': self.model.history.history['val_bce'][-1],
                'val_mse': self.model.history.history['val_mse'][-1],
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints'}                
        model_config['dataset'] = {
            'sim_dir': self.data_dir,
            'used_sim_steps':self.general_config['used_steps'],
            'config_file': 
                f"$(sim_dir){self.dataset_config_file.replace(self.data_dir,'')}",
            'train_file': 
                f"$(sim_dir){self.train_config['file'].replace(self.data_dir,'')}",
            'train_size': self.train_size,
            'validation_file': 
                f"$(sim_dir){self.val_config['file'].replace(self.data_dir,'')}",
            'val_size': self.val_size}

        model_config_file = os.path.join(model_dir,'model_config.yaml')
        with open(model_config_file,'w') as configfile:
            yaml.dump(model_config,configfile,default_flow_style=False)

    def custom_train_model(self,train_id=0,num_epochs=10,learning_rate=1e-3,device='cpu'):
        """ 
        Treina o modelo com loop de treino personalizado.
        Última modificação: 16/10/2021.
        
        Args:
            train_id:
            num_epochs:
            learning_rate: 
            device:                       
    
        """ 
        initial_learning_rate = learning_rate
        lr_schedule = ExponentialDecay(
            initial_learning_rate,
            decay_steps = 5,
            decay_rate = 0.9)
        #optimizer = SGD(learning_rate = lr_schedule)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)        

        for epoch in range(num_epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (batch_images_train, batch_labels_mask_train) in enumerate(self.train_set):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    batch_pred_labels_mask_train = self.model(batch_images_train)

                    # Compute the loss value for this minibatch.
                    loss_value = self.custom_loss_function_2(batch_labels_mask_train, batch_pred_labels_mask_train)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value,self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # Log every 5 batches.
                if step % 5 == 0:
                    print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))
                    print("Seen so far: %s samples" % ((step + 1) * self.train_batch_size))        

    def evaluate(self,dataset=None,dataset_file=None,
        batch_size=32,device='cpu',verbose=0):
        """ 
        Avalia a predição de um dataset. 
        Última modificação: 02/03/2021.
        
        Args:
            dataset:
            dataset_file:
            batch_size:
            device: 'cpu' or 'gpu'
        
        Returns:
            report:
        """
        if dataset is None and dataset_file is None:
            raise ValueError(
                "Use 'dataset' or 'dataset_file' "
                "keyword argument to pass de test dataset!")
        if dataset is not None and dataset_file is not None:
            raise ValueError(
                "Specify only one dataset! 'dataset' "
                "and 'dataset_file' are not allowed at the same time!")

        precision = PrecisionMask()
        recall = RecallMask()
        f1score = F1ScoreMask()
        mcc = MatthewsCoefficientMask()
        if dataset is not None:
            with tf.device(device):
                # Ground-truth
                labels_mask_true = dataset[1]
                labels_mask_pred = self.predict(
                    data=dataset[0],batch_size=batch_size,device=device)        

        if dataset_file is not None:
            with tf.device(device):
                self.val_input_pipeline(
                    dataset_file,batch_size,debug_mode=False)
            for k,sample in enumerate(self.val_set):
                images, mask_true = sample
                if verbose:
                    print(f"Batch {k}/{self.val_config['steps']-1}")
                with tf.device(device):
                    mask_pred = self.model.predict(
                        images,verbose=verbose)
                    # Atualiza métricas de avaliação
                    precision.update_state(mask_true,mask_pred)
                    recall.update_state(mask_true,mask_pred)
                    f1score.update_state(mask_true,mask_pred)
                    mcc.update_state(mask_true,mask_pred)
        print('\nAccuracy metrics:')
        print('Precision: ',precision.result().numpy())
        print('Recall: ',recall.result().numpy())
        print('F1-score: ',f1score.result().numpy())
        print('Mcc: ',mcc.result().numpy())
    
    def predict(self,dataset=None,dataset_file=None,batch_size=32,
        device='cpu',max_batches=-1,verbose=0):
        """ 
        Predição de um dataset. 
        Última modificação: 07/03/2022. 
        
        Args:
            dataset:
            dataset_file:
            batch_size:
            device: 'cpu' or 'gpu'
            return_mask_true:
            max_batches:
            verbose:
        
        Returns:
            labels:
        """
        tf.config.run_functions_eagerly(False)
        if dataset is None and dataset_file is None:
            raise ValueError(
                "Use 'dataset' or 'dataset_file' "
                "keyword argument to pass de test dataset!")
        if dataset is not None and dataset_file is not None:
            raise ValueError(
                "Specify only one dataset! 'dataset' "
                "and 'dataset_file' are not allowed at the same time!")

        if dataset is not None:
            with tf.device(device):
                mask_pred = self.model.predict(
                    dataset,
                    #use_multiprocessing = True,
                    verbose=verbose)    

        if dataset_file is not None:
            self.val_input_pipeline(
                dataset_file,batch_size,debug_mode=False)                        
            with tf.device(device):
                mask_pred = tf.zeros(
                    [0]+self.val_config['image_shape'].tolist())
            for k,sample in enumerate(self.val_set):
                if max_batches>0 and k >= max_batches:
                    print('Maximum number of batches reached!')
                    break
                images = sample[0]
                if verbose:
                    print(f"Batch {k}/{self.val_config['steps']-1}")
                with tf.device(device):
                    batch_mask_pred = self.model.predict(
                        images,verbose=verbose)
                    mask_pred = tf.concat(
                        [mask_pred,batch_mask_pred],axis=0)

        pred_dict = {}
        if 'boundary' in self.tasks and 'normal' in self.tasks:
            pred_dict['boundary'] = mask_pred[...,0:2]
            pred_dict['normal'] = mask_pred[...,2:]
        else:
            if 'boundary' in self.tasks:
                pred_dict['boundary'] = mask_pred
            elif 'normal' in self.tasks:
                pred_dict['normal'] = mask_pred
        return pred_dict

    def process_train(self,full_voxels_coord,target_voxels_coord,target):
        """ 
        Processa coordenadas de voxels de uma imagem de treino.
        Última atualização: 01/04/2022.
        
        Args:
            full_voxels_coord:
            target_voxels_coord:
            target:
                
        Returns:
            image,target_mask
        """         
        # print(type(full_voxels_coord))
        # print(type(target_voxels_coord))
        # print(type(target_labels))
        def sparse_to_dense(coords,values):
            #array = np.zeros(3*[self.input_res[0]]+[1])
            array = np.zeros(
                self.general_config['image_shape'][:-1] + [1])
            array[...,0].flat[coords] = values
            return tf.cast(array,tf.float32)

        # Imagem
        image_channel_1 = tf.numpy_function(
            sparse_to_dense,[full_voxels_coord,1],tf.float32)
        image_channel_2 = tf.numpy_function(
            sparse_to_dense,[target_voxels_coord,1],tf.float32)
        image = tf.concat([image_channel_1,image_channel_2],axis=-1)

        if 'boundary' in self.tasks and 'normal' in self.tasks:
            target_labels = target[0]
            target_normal = target[1]
        else:
            if 'boundary' in self.tasks:
                target_labels = target
            elif 'boundary' in self.tasks:
                target_normal = target                                

        target_mask = []
        if 'boundary' in self.tasks:
            onehot_labels = tf.one_hot(
                target_labels,self.general_config['num_classes'])
            # labels channel 1
            labels_channel_1 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                onehot_labels[:,0]],tf.float32)            
            target_mask.append(labels_channel_1)
            # labels channel 2
            labels_channel_2 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                onehot_labels[:,1]],tf.float32)
            target_mask.append(labels_channel_2)
        if 'normal' in self.tasks:
            normal_channel_1 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                target_normal[:,0]],tf.float32)
            target_mask.append(normal_channel_1)
            normal_channel_2 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                target_normal[:,1]],tf.float32)
            target_mask.append(normal_channel_2)
            if self.general_config['spatial_dimensions']==3:
                normal_channel_3 = tf.numpy_function(
                    sparse_to_dense,[target_voxels_coord,
                    target_normal[:,2]],tf.float32)
                target_mask.append(normal_channel_3)
        
        # Concatena channels
        target_mask = tf.concat(target_mask,axis=-1)
                        
        # print('image shape: ',image.shape)
        # print('labels mask shape: ',labels_mask.shape)
        # print('target labels shape: ',target_labels.shape)

        #return voxels,labels,tf.squeeze(target_labels)
        
        if self.data_augmentation:
            return self.apply_data_augmentation(image,target_mask)
        else:
            return image,target_mask

    def process_validation(self,full_voxels_coord,target_voxels_coord,target):
        """ 
        Processa coordenadas de voxels de uma imagem de validação.
        Última atualização: 01/04/2022.
        
        Args:
            full_voxels_coord:
            target_voxels_coord:
            target:
                
        Returns:
            image,target_mask
        """         
        # print(type(full_voxels_coord))
        # print(type(target_voxels_coord))
        # print(type(target_labels))
        def sparse_to_dense(coords,values):
            #array = np.zeros(3*[self.input_res[0]]+[1])
            array = np.zeros(
                self.general_config['image_shape'][:-1] + [1])
            array[...,0].flat[coords] = values
            return tf.cast(array,tf.float32)

        # Imagem
        image_channel_1 = tf.numpy_function(
            sparse_to_dense,[full_voxels_coord,1],tf.float32)
        image_channel_2 = tf.numpy_function(
            sparse_to_dense,[target_voxels_coord,1],tf.float32)
        image = tf.concat([image_channel_1,image_channel_2],axis=-1)

        if 'boundary' in self.tasks and 'normal' in self.tasks:
            target_labels = target[0]
            target_normal = target[1]
        else:
            if 'boundary' in self.tasks:
                target_labels = target
            elif 'boundary' in self.tasks:
                target_normal = target                                

        target_mask = []
        if 'boundary' in self.tasks:
            onehot_labels = tf.one_hot(
                target_labels,self.general_config['num_classes'])
            # labels channel 1
            labels_channel_1 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                onehot_labels[:,0]],tf.float32)            
            target_mask.append(labels_channel_1)
            # labels channel 2
            labels_channel_2 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                onehot_labels[:,1]],tf.float32)
            target_mask.append(labels_channel_2)
        if 'normal' in self.tasks:
            normal_channel_1 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                target_normal[:,0]],tf.float32)
            target_mask.append(normal_channel_1)
            normal_channel_2 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                target_normal[:,1]],tf.float32)
            target_mask.append(normal_channel_2)
            if self.general_config['spatial_dimensions']==3:
                normal_channel_3 = tf.numpy_function(
                    sparse_to_dense,[target_voxels_coord,
                    target_normal[:,2]],tf.float32)
                target_mask.append(normal_channel_3)
        
        # Concatena channels
        target_mask = tf.concat(target_mask,axis=-1)
                        
        return image,target_mask

    def process_validation_bkp(
        self,full_voxels_coord,target_voxels_coord,target_labels):
        """ 
        Processa coordenadas de voxels de uma imagem de validação.
        Última atualização: 24/02/2022.
        
        Args:
            full_voxels_coord:
            target_voxels_coord:
            target_labels:
                
        Returns:
            image,label_mask
        """
        # print(type(full_voxels_coord))
        # print(type(target_voxels_coord))
        # print(type(target_labels))
        def sparse_to_dense(coords,values):
            #array = np.zeros(3*[self.input_res[0]]+[1])
            array = np.zeros(
                self.general_config['image_shape'][:-1] + [1])
            array[...,0].flat[coords] = values
            return tf.cast(array,tf.float32)

        # Imagem
        image_channel_1 = tf.numpy_function(
            sparse_to_dense,[full_voxels_coord,1],tf.float32)
        image_channel_2 = tf.numpy_function(
            sparse_to_dense,[target_voxels_coord,1],tf.float32)
        image = tf.concat([image_channel_1,image_channel_2],axis=-1)

        # Labels mask
        onehot_labels = tf.one_hot(
            target_labels,self.general_config['num_classes'])
        labels_channel_1 = tf.numpy_function(
            sparse_to_dense,[target_voxels_coord,onehot_labels[:,0]],tf.float32)
        labels_channel_2 = tf.numpy_function(
            sparse_to_dense,[target_voxels_coord,onehot_labels[:,1]],tf.float32)
        labels_mask = tf.concat([labels_channel_1,labels_channel_2],axis=-1)
                        
        # print('image shape: ',image.shape)
        # print('labels mask shape: ',labels_mask.shape)
        # print('target labels shape: ',target_labels.shape)

        #return voxels,labels,tf.squeeze(target_labels)
        return image,labels_mask

    def apply_data_augmentation(self,image,labels_mask):
        """ 
        Aplica data augmentation.
        Última modificação: 24/02/2022.

        Args:
            image:
            labels_mask:

        Return;
            transformed image and labels_mask.

        """
        # Data-augmentation
        def random_axis_to_flip():
            if self.general_config['spatial_dimensions'] == 2:
                return np.random.choice([0,1])
            if self.general_config['spatial_dimensions'] == 3:
                return np.random.choice([0,1,2])
        
        def random_angle_to_rotate():
            return np.random.choice([0,1,2,3])

        def random_axes_to_rotate():
            if self.general_config['spatial_dimensions'] == 2:
                return np.array([0,1])
            if self.general_config['spatial_dimensions'] == 3:
                return np.random.choice([0,1,2],size=2,replace=False)

        def flip(array,axis):
            return np.flip(array,axis)
        
        def rotate(array,num_rot90,axes):
            return np.rot90(array,num_rot90,axes)
        
        # Aplica flip
        axis = tf.py_function(random_axis_to_flip,[],tf.int32)
        image = tf.py_function(flip,[image,axis],tf.float32)
        labels_mask = tf.py_function(flip,[labels_mask,axis],tf.float32)

        # Aplica rotação com ângulo múltiplo de 90
        num_rot90 = tf.py_function(random_angle_to_rotate,[],tf.int32)
        axes = tf.py_function(random_axes_to_rotate,[],tf.int32)   
        image = tf.py_function(rotate,[image,num_rot90 ,axes],tf.float32)
        labels_mask = tf.py_function(rotate,[labels_mask,num_rot90 ,axes],tf.float32)

        return image,labels_mask

class SparseVFCNNManager(ModelManager):
    
    def __init__(self,tasks,data_dir=None,dataset_config_file=None,approach_dir=None):
        """
        Construtor.
        Última modificação: 31/03/2022.  
        
        Args:
            tasks:
            data_dir:
            approach_dir:
            dataset_config_file:
            model_config_file:
        """
        super().__init__(data_dir,dataset_config_file,approach_dir)
        self.tasks = tasks
            
    def set_model(self,model=None,pretrained_model_config_file=None,
        from_template=None,transfer_learning=False,trainable_weigths=False):
        """
        Define o modelo de CNN que será treinado.
        Última atualização: 02/03/2022.

        Args:
            model:
            pretrained_model_config_file:
            from_template:
            transfer_weights:
            trainable_weigths:

        """
        if model is None and from_template is None:
            raise ValueError(
                "Specify one model! Use 'model' or 'from_template' keyword "
                "argument for this! Template modules can be found at "
                "boundary_detector.models.")
        if model is not None and from_template is not None:
            raise ValueError(
                "Specify only one model! 'model' and 'from_template'" 
                " are not allowed at the same time!")        

        if model is not None:
            self.model = model

        if from_template is not None:
            self.model_template = from_template
            if transfer_learning is True:
                self.transfer_learning = True
                if pretrained_model_config_file is None:
                    raise ValueError(
                        "'pretrained_model_config_file' is None, specify a "
                        "pre-trained model config file!")
                if 'normal' in self.tasks:
                    custom_layers = {'Normalize':Normalize,'ArgMax':ArgMax}
                else:
                    custom_layers = None
                pretrained_model, model_config = self.load_model(
                    pretrained_model_config_file, 
                    custom_layers = custom_layers,
                    return_model=True,
                    return_model_config = True)

                print('\nModelo pré treinado da abordagem pontual: ')
                pretrained_model.summary()    
                self.pretrained_model_template = model_config['model_template']

                args = (
                    "num_classes = self.general_config['num_classes'],"
                    "input_shape = self.general_config['image_shape'],"
                    "pre_trained_model = pretrained_model,"
                    "transfer_weights = True,"
                    "trainable_weigths = True")
            else:
                args = (
                    "num_classes = self.general_config['num_classes'],"
                    "input_shape = self.general_config['image_shape']")                            
            self.model = eval(f"{from_template}({args})")

    def train_input_pipeline(self,trainset_file=None,batch_size=32,
        buffer_size_factor=10,data_augmentation=False,debug_mode=False):
        """ 
        Pipeline de entrada do dataset de voxels de treino.
        Última modificação: 01/04/2022.
        
        Args:
            trainset_file:
            buffer_size_factor:
            data_augmentation:
            debug_mode:

        """        
        if debug_mode and float(tf.__version__[:-2])>=2.5:
            tf.data.experimental.enable_debug_mode()

        if trainset_file is not None:
            if not os.path.exists(trainset_file):
                raise FileNotFoundError('Training set file not found!')
            else:
                self.train_config = self.get_dataset_config_from_hdf5(trainset_file)
                if self.general_config is None:
                    self.general_config = {}
                    self.general_config['grid_size'] = self.val_config['grid_size'].tolist()
                    self.general_config['num_classes'] = self.train_config['num_classes']
                    self.general_config['spatial_dimensions'] = self.train_config['spatial_dimensions']
        
        self.train_config['batch_size'] = batch_size
        self.train_config['steps'] = 1 + self.train_config['num_batches']//batch_size
        self.data_augmentation = data_augmentation

        with h5py.File(self.train_config['file'],'r') as hf:
            neighbor_voxels_num = hf['neighbor_voxels_num'][:]
            neighbor_voxels_coord = hf['neighbor_voxels_coord'][:]
            
            target_voxels_num = hf['target_voxels_num'][:]
            target_voxels_coord = hf['target_voxels_coord'][:]

            if 'boundary' in self.tasks:
                target_labels = hf['target_boundary'][:]
            if 'normal' in self.tasks:
                target_normal = hf['target_normal'][:]

            self.train_size = hf['neighbor_voxels_num'].shape[0]
            print(f'The training dataset contains {self.train_size} batches.')
       
        # Faz o split dos arrays
        neighbor_voxels_coord = tf.RaggedTensor.from_row_lengths(
            neighbor_voxels_coord,neighbor_voxels_num)
        target_voxels_coord = tf.RaggedTensor.from_row_lengths(
            target_voxels_coord,target_voxels_num)

        if 'boundary' in self.tasks:
            target_labels = tf.RaggedTensor.from_row_lengths(
                target_labels,target_voxels_num)
        if 'normal' in self.tasks:
            target_normal = tf.RaggedTensor.from_row_lengths(
                target_normal,target_voxels_num)

        # Agrupa os datasets
        neighbor_voxels_coord = tf.data.Dataset.from_tensor_slices(neighbor_voxels_coord)
        target_voxels_coord = tf.data.Dataset.from_tensor_slices(target_voxels_coord)

        if 'boundary' in self.tasks and 'normal' in self.tasks:
            target_set = tf.data.Dataset.from_tensor_slices((target_labels,target_normal))
        else:
            if 'boundary' in self.tasks:
                target_set = tf.data.Dataset.from_tensor_slices(target_labels)
            if 'normal' in self.tasks:
                target_set = tf.data.Dataset.from_tensor_slices(target_normal)                
        train_set =  tf.data.Dataset.zip(
            (neighbor_voxels_coord,target_voxels_coord,target_set)) 

        if debug_mode:
            for i,sample in enumerate(train_set):
                if i<5:
                    print(f'Sparse sample {i} (Before processing)')
                    print(' --> neighbor_voxels_coord shape: ',sample[0].shape)
                    print(' --> target_voxels_coord shape: ',sample[1].shape)
                    if 'boundary' in self.tasks and 'normal' in self.tasks:
                        print(' --> target_labels shape: ',sample[2][0].shape) 
                        print(' --> target_normal shape: ',sample[2][1].shape) 
                    else:
                        if 'boundary' in self.tasks:
                            print(' --> target_labels shape: ',sample[2].shape)
                        elif 'normal' in self.tasks:
                            print(' --> target_normal shape: ',sample[2].shape) 
                else:
                    break

        # Aplica alguns processamentos
        buffer_size = buffer_size_factor*batch_size
        train_set = train_set.map(self.process_data)#,num_parallel_calls=self.autotune)
        train_set = train_set.shuffle(buffer_size=buffer_size)
        train_set = train_set.repeat()
        train_set = train_set.batch(batch_size)
        #train_set = train_set.apply(
        #            tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
        train_set = train_set.prefetch(buffer_size=self.autotune)
        
        if debug_mode:
            for i,sample in enumerate(train_set):
                if i<5:
                    print(f'Sparse sample {i} (After processing)')
                    print(' --> neighbor_voxels_coord shape: ',sample[0].shape)
                    print(' --> target_voxels_coord shape: ',sample[1].shape)
                    if 'boundary' in self.tasks and 'normal' in self.tasks:
                        print(' --> target_labels shape: ',sample[2][0].shape) 
                        print(' --> target_normal shape: ',sample[2][1].shape) 
                    else:
                        if 'boundary' in self.tasks:
                            print(' --> target_labels shape: ',sample[2].shape)
                        elif 'normal' in self.tasks:
                            print(' --> target_normal shape: ',sample[2].shape) 
                else:
                    break
        self.train_set = train_set
 
    def val_input_pipeline(self,valset_file=None,batch_size=1,
        buffer_size_factor=10,data_augmentation=False,debug_mode=False):
        """ 
        Pipeline de entrada do dataset de voxels de validação.
        Última modificação: 01/04/2022.
        
        Args:
            valset_file:
            buffer_size_factor:
            data_augmentation:
            debug_mode:

        """        
        if debug_mode and float(tf.__version__[:-2])>=2.5:
            tf.data.experimental.enable_debug_mode()

        if valset_file is not None:
            if not os.path.exists(valset_file):
                raise FileNotFoundError('Validation set file not found!')
            else:
                self.val_config = self.get_dataset_config_from_hdf5(valset_file)
                if self.general_config is None:
                    self.general_config = {}
                    self.general_config['grid_size'] = self.val_config['grid_size'].tolist()
                    self.general_config['num_classes'] = self.val_config['num_classes']
                    self.general_config['spatial_dimensions'] = self.val_config['spatial_dimensions']
        
        self.val_config['batch_size'] = batch_size
        self.val_config['steps'] = 1 + self.val_config['num_batches']//batch_size
        self.data_augmentation = data_augmentation

        with h5py.File(self.val_config['file'],'r') as hf:
            neighbor_voxels_num = hf['neighbor_voxels_num'][:]
            neighbor_voxels_coord = hf['neighbor_voxels_coord'][:]
            
            target_voxels_num = hf['target_voxels_num'][:]
            target_voxels_coord = hf['target_voxels_coord'][:]

            if 'boundary' in self.tasks:
                target_labels = hf['target_boundary'][:]
            if 'normal' in self.tasks:
                target_normal = hf['target_normal'][:]

            self.val_size = hf['neighbor_voxels_num'].shape[0]
            print(f'The validation dataset contains {self.val_size} batches.')
       
        # Faz o split dos arrays
        neighbor_voxels_coord = tf.RaggedTensor.from_row_lengths(
            neighbor_voxels_coord,neighbor_voxels_num)
        target_voxels_coord = tf.RaggedTensor.from_row_lengths(
            target_voxels_coord,target_voxels_num)

        if 'boundary' in self.tasks:
            target_labels = tf.RaggedTensor.from_row_lengths(
                target_labels,target_voxels_num)
        if 'normal' in self.tasks:
            target_normal = tf.RaggedTensor.from_row_lengths(
                target_normal,target_voxels_num)

        # Agrupa os datasets
        neighbor_voxels_coord = tf.data.Dataset.from_tensor_slices(neighbor_voxels_coord)
        target_voxels_coord = tf.data.Dataset.from_tensor_slices(target_voxels_coord)

        if 'boundary' in self.tasks and 'normal' in self.tasks:
            target_set = tf.data.Dataset.from_tensor_slices((target_labels,target_normal))
        else:
            if 'boundary' in self.tasks:
                target_set = tf.data.Dataset.from_tensor_slices(target_labels)
            if 'normal' in self.tasks:
                target_set = tf.data.Dataset.from_tensor_slices(target_normal)                
        val_set =  tf.data.Dataset.zip(
            (neighbor_voxels_coord,target_voxels_coord,target_set)) 

        if debug_mode:
            for i,sample in enumerate(val_set):
                if i<5:
                    print(f'Sparse sample {i} (Before processing)')
                    print(' --> neighbor_voxels_coord shape: ',sample[0].shape)
                    print(' --> target_voxels_coord shape: ',sample[1].shape)
                    if 'boundary' in self.tasks and 'normal' in self.tasks:
                        print(' --> target_labels shape: ',sample[2][0].shape) 
                        print(' --> target_normal shape: ',sample[2][1].shape) 
                    else:
                        if 'boundary' in self.tasks:
                            print(' --> target_labels shape: ',sample[2].shape)
                        elif 'normal' in self.tasks:
                            print(' --> target_normal shape: ',sample[2].shape) 
                else:
                    break

        # Aplica alguns processamentos
        buffer_size = buffer_size_factor*batch_size
        val_set = val_set.map(self.process_data)#,num_parallel_calls=self.autotune)
        val_set = val_set.shuffle(buffer_size=buffer_size)
        val_set = val_set.repeat()
        val_set = val_set.batch(batch_size)
        #val_set = val_set.apply(
        #            tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
        val_set = val_set.prefetch(buffer_size=self.autotune)
        
        if debug_mode:
            for i,sample in enumerate(val_set):
                if i<5:
                    print(f'Sparse sample {i} (After processing)')
                    print(' --> neighbor_voxels_coord shape: ',sample[0].shape)
                    print(' --> target_voxels_coord shape: ',sample[1].shape)
                    if 'boundary' in self.tasks and 'normal' in self.tasks:
                        print(' --> target_labels shape: ',sample[2][0].shape) 
                        print(' --> target_normal shape: ',sample[2][1].shape) 
                    else:
                        if 'boundary' in self.tasks:
                            print(' --> target_labels shape: ',sample[2].shape)
                        elif 'normal' in self.tasks:
                            print(' --> target_normal shape: ',sample[2].shape) 
                else:
                    break
        self.val_set = val_set

    def val_input_pipeline_bkp2(self,valset_file=None,batch_size=32,
        debug_mode=False):
        """ 
        Pipeline de entrada do dataset de voxels de treino.
        Última modificação: 01/04/2022.
        
        Args:    
            valset_file:
            batch_size:     
            debug_mode:

        """        
        if debug_mode and float(tf.__version__[:-2])>=2.5:
            tf.data.experimental.enable_debug_mode()

        if valset_file is not None:
            if not os.path.exists(valset_file):
                raise FileNotFoundError('Validation set file not found!')
            else:
                self.val_config = self.get_dataset_config_from_hdf5(valset_file)
                if self.general_config is None:
                    self.general_config = {}
                    self.general_config['image_shape'] = self.val_config['image_shape'].tolist()
                    self.general_config['num_classes'] = self.val_config['num_classes']
                    self.general_config['spatial_dimensions'] = self.val_config['spatial_dimensions']
                
        self.val_config['batch_size'] = batch_size
        self.val_config['steps'] = 1 + self.val_config['num_samples'] // self.val_config['batch_size']

        with h5py.File(self.val_config['file'],'r') as hf:
            full_num_voxels = hf['full_num_voxels'][:]
            full_voxels_coord = hf['full_voxels_coord'][:]
            
            target_num_voxels = hf['target_num_voxels'][:]
            target_voxels_coord = hf['target_voxels_coord'][:]

            if 'boundary' in self.tasks:
                target_labels = hf['target_boundary'][:]
            if 'normal' in self.tasks:
                target_normal = hf['target_normal'][:]

            self.val_size = hf['full_num_voxels'].shape[0]
            print(f'The validation dataset contains {self.val_size} images.')
       
        # Faz o split dos arrays
        full_voxels_coord = tf.RaggedTensor.from_row_lengths(
            full_voxels_coord,full_num_voxels)
        target_voxels_coord = tf.RaggedTensor.from_row_lengths(
            target_voxels_coord,target_num_voxels)

        if 'boundary' in self.tasks:
            target_labels = tf.RaggedTensor.from_row_lengths(
                target_labels,target_num_voxels)
        if 'normal' in self.tasks:
            target_normal = tf.RaggedTensor.from_row_lengths(
                target_normal,target_num_voxels)

        # Agrupa os datasets
        full_voxels_coord = tf.data.Dataset.from_tensor_slices(
            full_voxels_coord)
        target_voxels_coord = tf.data.Dataset.from_tensor_slices(
            target_voxels_coord)

        if 'boundary' in self.tasks and 'normal' in self.tasks:
            target_set = tf.data.Dataset.from_tensor_slices(
                (target_labels,target_normal))
        else:
            if 'boundary' in self.tasks:
                target_set = tf.data.Dataset.from_tensor_slices(
                    target_labels)
            if 'normal' in self.tasks:
                target_set = tf.data.Dataset.from_tensor_slices(
                    target_normal)                
        val_set =  tf.data.Dataset.zip(
            (full_voxels_coord,target_voxels_coord,target_set)) 

        if debug_mode:
            for i,sample in enumerate(val_set):
                if i<5:
                    print(f'Sparse sample {i} (Before processing)')
                    print(' --> full_voxels_coord shape: ',sample[0].shape)
                    print(' --> target_voxels_coord shape: ',sample[1].shape)
                    if 'boundary' in self.tasks and 'normal' in self.tasks:
                        print(' --> target_labels shape: ',sample[2][0].shape) 
                        print(' --> target_normal shape: ',sample[2][1].shape) 
                    else:
                        if 'boundary' in self.tasks:
                            print(' --> target_labels shape: ',sample[2].shape)
                        elif 'normal' in self.tasks:
                            print(' --> target_normal shape: ',sample[2].shape) 
                else:
                    break

        # Aplica alguns processamentos
        #val_set = val_set.map(self.process_datas,num_parallel_calls=self.autotune)
        val_set = val_set.batch(batch_size)
        val_set = val_set.prefetch(buffer_size=self.autotune)
        
        if debug_mode:
            for i,sample in enumerate(val_set):
                if i<5:
                    print(f'Sparse sample {i} (After processing)')
                    print(' --> full_voxels_coord shape: ',sample[0].shape)
                    print(' --> target_voxels_coord shape: ',sample[1].shape)
                    if 'boundary' in self.tasks and 'normal' in self.tasks:
                        print(' --> target_labels shape: ',sample[2][0].shape) 
                        print(' --> target_normal shape: ',sample[2][1].shape) 
                    else:
                        if 'boundary' in self.tasks:
                            print(' --> target_labels shape: ',sample[2].shape)
                        elif 'normal' in self.tasks:
                            print(' --> target_normal shape: ',sample[2].shape) 
                else:
                    break

        self.val_set = val_set

    def val_input_pipeline_bkp(self,valset_file=None,batch_size=32,
        val_size=None,debug_mode=False):
        """ 
        Pipeline de entrada do dataset de voxels de validação.        
        Última modificação: 24/02/2022.
        
        Args:    
            valset_file:
            batch_size:            
            val_size:
        """    
        if debug_mode:
            tf.data.experimental.enable_debug_mode()

        if valset_file is not None:
            if not os.path.exists(valset_file):
                raise FileNotFoundError('Validation set file not found!')
            else:
                self.val_config = self.get_dataset_config_from_hdf5(valset_file)           
                if self.general_config is None:
                    self.general_config = {}
                    self.general_config['image_shape'] = self.val_config['image_shape'].tolist()
                    self.general_config['num_classes'] = self.val_config['num_classes']
                    self.general_config['spatial_dimensions'] = self.val_config['spatial_dimensions']

        self.val_config['batch_size'] = batch_size
        self.val_config['steps'] = 1 + self.val_config['num_samples'] // self.val_config['batch_size']

        with h5py.File(self.val_config['file'],'r') as hf:
            full_num_voxels = hf['full_num_voxels'][:]
            full_voxels_coord = hf['full_voxels_coord'][:]
            
            target_labels = hf['target_labels'][:]
            target_num_voxels = hf['target_num_voxels'][:]
            target_voxels_coord = hf['target_voxels_coord'][:]

            self.val_size = hf['full_num_voxels'].shape[0]
            print(f'The validation dataset contains {self.val_size} voxels.')
    
        # Faz o split dos arrays
        full_voxels_coord = tf.RaggedTensor.from_row_lengths(full_voxels_coord,full_num_voxels)
        target_voxels_coord = tf.RaggedTensor.from_row_lengths(target_voxels_coord,target_num_voxels)
        target_labels = tf.RaggedTensor.from_row_lengths(target_labels,target_num_voxels)

        full_voxels_coord = tf.data.Dataset.from_tensor_slices(full_voxels_coord)
        target_voxels_coord = tf.data.Dataset.from_tensor_slices(target_voxels_coord)
        target_labels = tf.data.Dataset.from_tensor_slices(target_labels)
        
        val_set =  tf.data.Dataset.zip((full_voxels_coord,target_voxels_coord,target_labels))

        if debug_mode:
            for i,sample in enumerate(val_set):
                if i<5:
                    print(f'Sparse sample {i}')
                    print(' --> full_voxels_coord shape: ',sample[0].shape)
                    print(' --> target_voxels_coord shape: ',sample[1].shape)
                    print(' --> target_labels shape: ',sample[2].shape)
                else:
                    break            

        # Aplica alguns processamentos
        val_set = val_set.map(self.process_validation,num_parallel_calls=self.autotune)
        #val_set = val_set.repeat()
        val_set = val_set.batch(batch_size)
        val_set = val_set.prefetch(buffer_size=self.autotune)
        
        if debug_mode:
            for i,(images,labels_mask) in enumerate(val_set):
                if i<5:
                    print(f'Batch {i}: ')
                    print(' --> images shape: ',images.shape)
                    print(' --> labels mask shape: ',labels_mask.shape)
                else:
                    break

        self.val_set = val_set
    
    def train_model(self,train_id=0,num_epochs=10,learning_rate=1e-3,
        device='cpu',debug_mode=None):
        """ 
        Treina o modelo.
        Última modificação: 05/04/2022.
        
        Args:
            train_id:
            num_epochs:
            learning_rate: 
            device:                   
            debug_mode:    
    
        """   
        image_res =  self.general_config['image_shape'][0]
        border_size =  self.general_config['border_size']
        image_length = self.general_config['image_length']
        config_str = f'{image_res}_{border_size}_{image_length:.2f}'

        train_config_str = f'{config_str}_{self.model.name}_{train_id}'
        
        model_dir = os.path.join(
            self.approach_dir,'models',f'model_{train_config_str}')
        os.makedirs(model_dir,exist_ok=True)

        class DisplayCallback(tf.keras.callbacks.Callback):
            
            def on_epoch_end(self, epoch, logs=None):
              #show_predictions(dataset['train'],5)
              print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

        log_dir = os.path.join(model_dir,'logs')
        checkpoint_dir = os.path.join(model_dir,'checkpoint','checkpoint')
        exp_log_dir = os.path.join(log_dir,datetime.datetime.now().strftime("%d-%m-%Y [%Hh%Mm%Ss]"),"fit")

        try:
            self.model.load_weights(checkpoint_dir)
        except:
            print('Checkpoint de treino não pôde ser restaurado!')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(exp_log_dir, histogram_freq=1)
        
        Callbacks = [
            #DisplayCallback(),
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_dir, verbose=1, save_best_only=True, save_weights_only=True)]
        
        initial_learning_rate = learning_rate
        lr_schedule = ExponentialDecay(
            initial_learning_rate,decay_steps = 5,decay_rate = 0.9)
        #optimizer = SGD(learning_rate = lr_schedule)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        
        if self.general_config['spatial_dimensions']==2:
            slice_bce = {'begin':[0,0,0,0],'size':[-1,-1,-1,2]}
            slice_mse = {'begin':[0,0,0,2],'size':[-1,-1,-1,2]}
            slice_sdf_mse = {'begin':[0,0,0,4],'size':[-1,-1,-1,3]}
        elif self.general_config['spatial_dimensions']==3:
            slice_bce = {'begin':[0,0,0,0,0],'size':[-1,-1,-1,-1,2]}
            slice_mse = {'begin':[0,0,0,0,2],'size':[-1,-1,-1,-1,3]}

        if 'boundary' in self.tasks and len(self.tasks)==1:
            self.model.compile(
                optimizer = optimizer,
                loss =  BinaryCrossentropy(),
                metrics = [MeanSquaredError()],
                    #PrecisionRW(slice=slice_bce),
                    #RecallRW(slice=slice_bce),
                    #F1ScoreRW(slice=slice_bce),
                    #MatthewsCoefficientRW(slice=slice_bce)],
                run_eagerly=debug_mode)
        if 'normal' in self.tasks and len(self.tasks)==1:
            self.model.compile(
                optimizer = optimizer, 
                loss = [NormalLossRW()],
                metrics = [MeanSquaredErrorRW()],
                run_eagerly=debug_mode)
        elif 'boundary' in self.tasks and 'normal' in self.tasks and len(self.tasks)==2:
            self.model.compile(                
                optimizer = optimizer, 
                loss = BoundaryNormalLossRW(
                    slice={'bce':slice_bce,'mse':slice_mse},
                    slice_weights=slice_bce),
                metrics = [RecallRW(slice=slice_bce,name='r'),
                    PrecisionRW(slice=slice_bce,name='p'),
                    F1ScoreRW(slice=slice_bce,name='f1'),
                    MatthewsCoefficientRW(slice=slice_bce,name='mcc'),
                    BinaryCrossEntropyRW(
                        slice=slice_bce,slice_weights=slice_bce,name='bce'),
                    MeanSquaredErrorRW(
                        slice=slice_mse,slice_weights=slice_bce,name='mse')],
                run_eagerly=debug_mode
            )
        elif len(self.tasks)==3:
            self.model.compile(                
                optimizer = optimizer, 
                loss = FullLoss(
                    slice={'bce':slice_bce,'mse':slice_mse,'sdf_mse':slice_sdf_mse},
                    slice_weights=slice_bce),
                metrics = [
                    RecallRW(slice=slice_bce,name='r'),
                    PrecisionRW(slice=slice_bce,name='p'),
                    F1ScoreRW(slice=slice_bce,name='f1'),
                    MatthewsCoefficientRW(slice=slice_bce,name='mcc'),
                    BinaryCrossEntropyRW(
                        slice=slice_bce,slice_weights=slice_bce,name='bce'),
                    MeanSquaredErrorRW(
                        slice=slice_mse,slice_weights=slice_bce,name='mse')],
                run_eagerly=debug_mode)            

        
        #tf.config.run_functions_eagerly(True)
        #tf.data.experimental.enable_debug_mode()

        if device == 'cpu':
            with tf.device("/cpu:0"):
                model_history = self.model.fit(
                    self.train_set,
                    epochs = num_epochs,
                    steps_per_epoch = self.train_config['steps'],
                    validation_steps = self.val_config['steps'],
                    validation_data = self.val_set,
                    callbacks = Callbacks,
                    workers = 28)
        elif device == 'gpu':
            with tf.device("/GPU:0"):
                #  ESCREVER UM LOOP DE TREINO
                model_history = self.model.fit(
                    self.train_set,
                    epochs = num_epochs,
                    steps_per_epoch = self.train_config['steps'],
                    validation_steps = self.val_config['steps'],
                    validation_data = self.val_set,
                    callbacks = Callbacks)

        #  Salva o modelo
        if self.general_config['spatial_dimensions'] == 2:
            base_name = 'BRegionCNN2D'
        elif self.general_config['spatial_dimensions'] == 3:
            base_name = 'BRegionCNN3D'
        self.save_model(model_dir,base_name)
        
        # Salva o arquivo de configuração
        model_config = {}
        model_config['model'] = {
            'model_id': train_id,
            'model_name': self.model.name,
            'architecture_file': f'{base_name}.json',
            'weights_file': f'{base_name}.h5',
            'approach': 'regionwise',
            'model_template': str(self.model_template),
            'pretrained_model_template': str(self.pretrained_model_template),
            'transfer_learning': str(self.transfer_learning),
            'input_shape': ' '.join([str(x) for x in self.general_config['image_shape']]),
            'border_size': self.general_config['border_size'],
            'image_length': self.general_config['image_length'],
            'spatial_dimensions': self.general_config['spatial_dimensions']}
        if 'boundary' in self.tasks and 'normal' not in self.tasks:            
            model_config['training'] = {
                'device': device,
                'epochs': num_epochs,
                'learning_rate': learning_rate,
                'optimizer': optimizer._name,
                'loss_function': self.model.loss,
                'metrics': ' '.join(self.model.metrics_names),
                'val_loss': self.model.history.history['val_loss'][-1], 
                'val_precision': self.model.history.history['val_precision'][-1],
                'val_recall': self.model.history.history['val_recall'][-1],
                'val_f1score': self.model.history.history['val_f1score'][-1],
                'val_mcc': self.model.history.history['val_mcc'][-1],
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints',}
        elif 'boundary' not in self.tasks and 'normal' in self.tasks:
            model_config['training'] = {
                'device': device,
                'epochs': num_epochs,
                'batch_size':self.train_config['batch_size'],
                'learning_rate': learning_rate,
                'optimizer': optimizer._name,
                'loss': self.model.loss,
                'metrics': ' '.join(self.model.metrics_names),
                'val_loss': self.model.history.history['val_loss'][-1],
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints'}
        elif 'boundary' in self.tasks and 'normal' in self.tasks:
            model_config['training'] = {
                'device': device,
                'epochs': num_epochs,
                'batch_size':self.train_config['batch_size'],
                'learning_rate': learning_rate,
                'optimizer': optimizer._name,
                'loss': self.model.loss,
                'metrics': ' '.join(self.model.metrics_names),
                'val_loss': self.model.history.history['val_loss'][-1],
                'val_precision': self.model.history.history['val_p'][-1],
                'val_recall': self.model.history.history['val_r'][-1],
                'val_f1score': self.model.history.history['val_f1'][-1],
                'val_mcc': self.model.history.history['val_mcc'][-1],
                'val_bce': self.model.history.history['val_bce'][-1],
                'val_mse': self.model.history.history['val_mse'][-1],
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints'}                
        model_config['dataset'] = {
            'sim_dir': self.data_dir,
            'used_sim_steps':self.general_config['used_steps'],
            'config_file': 
                f"$(sim_dir){self.dataset_config_file.replace(self.data_dir,'')}",
            'train_file': 
                f"$(sim_dir){self.train_config['file'].replace(self.data_dir,'')}",
            'train_size': self.train_size,
            'validation_file': 
                f"$(sim_dir){self.val_config['file'].replace(self.data_dir,'')}",
            'val_size': self.val_size}

        model_config_file = os.path.join(model_dir,'model_config.yaml')
        with open(model_config_file,'w') as configfile:
            yaml.dump(model_config,configfile,default_flow_style=False)

    def custom_train_model_v0(self,train_id=0,num_epochs=10,learning_rate=1e-3,device='cpu'):
        """ 
        Treina o modelo com loop de treino personalizado.
        Última modificação: 16/10/2021.
        
        Args:
            train_id:
            num_epochs:
            learning_rate: 
            device:                       
    
        """ 
        initial_learning_rate = learning_rate
        lr_schedule = ExponentialDecay(
            initial_learning_rate,
            decay_steps = 5,
            decay_rate = 0.9)
        #optimizer = SGD(learning_rate = lr_schedule)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)        

        #CONTINUARA DAQUI
        for epoch in range(num_epochs):
            print("\nEpoch %d" % epoch)

            # Iterate over the batches of the dataset.
            loss_value = np.inf
            bar = tqdm(total = self.train_config['steps'])
            for step, (full_voxels_coord, target_voxels_coord, target_labels) in enumerate(self.train_set):
                bar.update(1)
                #for full_voxels_coord, target_voxels_coord, target_labels in zip(batch_full_voxels_coord, batch_target_voxels_coord, batch_target_labels):
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    occupancy = self.stack_ragged([tf.ones((full_voxels_coord[i].shape[0],1),dtype=tf.float32) for i in range(full_voxels_coord.shape[0])])

                    full_voxels_coord = self.unravel_index(full_voxels_coord)
                    full_voxels_coord = tf.cast(full_voxels_coord,tf.float32)

                    target_voxels_coord = self.unravel_index(target_voxels_coord)
                    target_voxels_coord = tf.cast(target_voxels_coord,tf.float32)

                    pred_labels,full_voxels_coord = self.model([occupancy,full_voxels_coord,target_voxels_coord])
                    #pred_labels = tf.vectorized_map(tf.argmax,pred_labels)
                    pred_labels = tf.cast(pred_labels,tf.float32)
                    #pred_labels

                    # CONTINUAR DAQUI
                    target_labels = tf.expand_dims(target_labels,axis=-1)
                    target_labels = tf.cast(target_labels,tf.float32)
                    
                    # Compute the loss value for this minibatch.
                    loss_value = self.custom_binary_crossentropy(target_labels,pred_labels)
                    bar.set_postfix(loss=f"{loss_value.numpy():.4f}")

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value,self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            
            bar.close()

    def custom_train_model_v2(self,train_id=0,num_epochs=10,learning_rate=1e-3,patience_max=10,device='cpu'):
        """ 
        Treina o modelo com loop de treino personalizado.
        
        Args:
            train_id:
            num_epochs:
            learning_rate: 
            device:                       
    
        """
        train_config_str = f'{self.model.name}_{train_id}'
        model_dir = os.path.join(
            self.approach_dir,'models',f'model_{train_config_str}')
        os.makedirs(model_dir,exist_ok=True)

        tf.config.run_functions_eagerly(True)

        initial_learning_rate = learning_rate
        # lr_schedule = ExponentialDecay(
        #     initial_learning_rate,
        #     decay_steps = 5,
        #     decay_rate = 0.9)
        #optimizer = SGD(learning_rate = lr_schedule)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        # Losses and metrics
        losses = {}
        metrics = {}
        for task in self.tasks:
            if self.tasks[task]['type']=='classification':
                losses[task] = tf.keras.losses.BinaryCrossentropy(name=f'{task}_loss',reduction=tf.keras.losses.Reduction.AUTO)
                task_metrics = {}
                task_metrics['rec'] = tf.keras.metrics.Recall(name='rec',class_id=1)
                task_metrics['pre'] = tf.keras.metrics.Precision(name='pre',class_id=1)
                task_metrics['f1'] = tfa.metrics.F1Score(name='f1',num_classes=2,threshold=0.5)
                task_metrics['mcc'] = tfa.metrics.MatthewsCorrelationCoefficient(name='mcc',num_classes=2)
                metrics[task] = task_metrics
            if self.tasks[task]['type']=='regression':
                #losses[task] = tf.keras.losses.MeanSquaredError(name=f'{task}_loss',reduction=tf.keras.losses.Reduction.AUTO)
                losses[task] = 1+tf.keras.losses.cosine_similarity(name=f'{task}_loss',reduction=tf.keras.losses.Reduction.AUTO)
                task_metrics = {}
                task_metrics['mse'] = tf.keras.metrics.MeanSquaredError(name='mse')
                task_metrics['cos'] = tf.keras.metrics.CosineSimilarity(name='cos',axis=-1)
                metrics[task] = task_metrics

        self.model.compile(
            optimizer = optimizer,
            losses = losses,
            metrics = metrics
        )

        report = self.model.fit(
            train_set = self.train_set,
            train_steps = self.train_config['steps'],
            val_set = self.val_set,
            val_steps = self.val_config['steps'],
            num_epochs = num_epochs,
            patience_max = patience_max,
            export_dir = model_dir
        )
        best_epoch = report['best_epoch']

        #print('train_loss: ',report['train_loss'])
        #print('val_loss: ',report['val_loss'])
        
        # Salva o arquivo de configuração
        model_config = {}
        model_config['model'] = {
            'model_id': train_id,
            'model_name': self.model.name,
            'approach': 'sparse_regionwise',
            'model_template': str(self.model_template),
            'pretrained_model_template': str(self.pretrained_model_template),
            'transfer_learning': str(self.transfer_learning),
            'grid_size': [str(x) for x in self.model.grid_size],
            'spatial_dimensions': self.general_config['spatial_dimensions']
        }

        model_config['training'] = {
            'device': device,
            'epochs': num_epochs,
            'learning_rate': learning_rate,
            'optimizer': optimizer._name,
            'loss_function': [l.name for l in self.model.losses],
            'metrics': list(self.model.metrics_names), 
            
            'train_loss': float(report['loss']['train']['total'][best_epoch]),
            'train_mcc': float(report['metrics']['train']['boundary']['mcc'][best_epoch]),
            'train_precision': float(report['metrics']['train']['boundary']['pre'][best_epoch]),
            'train_recall': float(report['metrics']['train']['boundary']['rec'][best_epoch]),
            'train_f1score': float(report['metrics']['train']['boundary']['f1'][best_epoch]),

            'val_loss': float(report['loss']['val']['total'][best_epoch]), 
            'val_mcc': float(report['metrics']['val']['boundary']['mcc'][best_epoch]),
            'val_precision': float(report['metrics']['val']['boundary']['pre'][best_epoch]),
            'val_recall': float(report['metrics']['val']['boundary']['rec'][best_epoch]),
            'val_f1score': float(report['metrics']['val']['boundary']['f1'][best_epoch]),            
        }

        model_config['dataset'] = {
            'sim_dir': self.data_dir,
            'used_sim_steps': self.general_config['used_steps'],
            'config_file': self.dataset_config_file,
            'train_file': self.train_config['file'],
            'train_size': self.train_size,
            'validation_file': self.val_config['file'],
            'val_size': self.val_size
        }

        model_config_file = os.path.join(model_dir,'model_config_v2.yaml')
        with open(model_config_file,'w') as configfile:
            yaml.dump(model_config,configfile,default_flow_style=False)

    def custom_binary_crossentropy(self,target_labels,pred_labels):
        def binary_crossentropy(inputs):
            return tf.keras.losses.binary_crossentropy(inputs[0], inputs[1])
        loss = tf.map_fn(
            binary_crossentropy,
            [target_labels,pred_labels],
            fn_output_signature = tf.RaggedTensorSpec(shape=[None],dtype=tf.float32))
        #return tf.reduce_mean(loss,axis=-1)
        return tf.reduce_sum(loss)
        #return loss
        
    def evaluate(self,dataset=None,dataset_file=None,
        batch_size=32,device='cpu',verbose=0):
        """ 
        Avalia a predição de um dataset. 
        Última modificação: 02/03/2021.
        
        Args:
            dataset:
            dataset_file:
            batch_size:
            device: 'cpu' or 'gpu'
        
        Returns:
            report:
        """
        if dataset is None and dataset_file is None:
            raise ValueError(
                "Use 'dataset' or 'dataset_file' "
                "keyword argument to pass de test dataset!")
        if dataset is not None and dataset_file is not None:
            raise ValueError(
                "Specify only one dataset! 'dataset' "
                "and 'dataset_file' are not allowed at the same time!")

        precision = PrecisionMask()
        recall = RecallMask()
        f1score = F1ScoreMask()
        mcc = MatthewsCoefficientMask()
        if dataset is not None:
            with tf.device(device):
                # Ground-truth
                labels_mask_true = dataset[1]
                labels_mask_pred = self.predict(
                    data=dataset[0],batch_size=batch_size,device=device)        

        if dataset_file is not None:
            with tf.device(device):
                self.val_input_pipeline(
                    dataset_file,batch_size,debug_mode=False)
            for k,sample in enumerate(self.val_set):
                images, mask_true = sample
                if verbose:
                    print(f"Batch {k}/{self.val_config['steps']-1}")
                with tf.device(device):
                    mask_pred = self.model.predict(
                        images,verbose=verbose)
                    # Atualiza métricas de avaliação
                    precision.update_state(mask_true,mask_pred)
                    recall.update_state(mask_true,mask_pred)
                    f1score.update_state(mask_true,mask_pred)
                    mcc.update_state(mask_true,mask_pred)
        print('\nAccuracy metrics:')
        print('Precision: ',precision.result().numpy())
        print('Recall: ',recall.result().numpy())
        print('F1-score: ',f1score.result().numpy())
        print('Mcc: ',mcc.result().numpy())
    
    def predict(self,dataset=None,dataset_file=None,batch_size=32,
        device='cpu',max_batches=-1,verbose=0):
        """ 
        Predição de um dataset. 
        Última modificação: 07/03/2022. 
        
        Args:
            dataset:
            dataset_file:
            batch_size:
            device: 'cpu' or 'gpu'
            return_mask_true:
            max_batches:
            verbose:
        
        Returns:
            labels:
        """
        tf.config.run_functions_eagerly(False)
        if dataset is None and dataset_file is None:
            raise ValueError(
                "Use 'dataset' or 'dataset_file' "
                "keyword argument to pass de test dataset!")
        if dataset is not None and dataset_file is not None:
            raise ValueError(
                "Specify only one dataset! 'dataset' "
                "and 'dataset_file' are not allowed at the same time!")

        if dataset is not None:
            with tf.device(device):
                mask_pred = self.model.predict(
                    dataset,
                    #use_multiprocessing = True,
                    verbose=verbose)    

        if dataset_file is not None:
            self.val_input_pipeline(
                dataset_file,batch_size,debug_mode=False)                        
            with tf.device(device):
                mask_pred = tf.zeros(
                    [0]+self.val_config['image_shape'].tolist())
            for k,sample in enumerate(self.val_set):
                if max_batches>0 and k >= max_batches:
                    print('Maximum number of batches reached!')
                    break
                images = sample[0]
                if verbose:
                    print(f"Batch {k}/{self.val_config['steps']-1}")
                with tf.device(device):
                    batch_mask_pred = self.model.predict(
                        images,verbose=verbose)
                    mask_pred = tf.concat(
                        [mask_pred,batch_mask_pred],axis=0)

        pred_dict = {}
        if 'boundary' in self.tasks and 'normal' in self.tasks:
            pred_dict['boundary'] = mask_pred[...,0:2]
            pred_dict['normal'] = mask_pred[...,2:]
        else:
            if 'boundary' in self.tasks:
                pred_dict['boundary'] = mask_pred
            elif 'normal' in self.tasks:
                pred_dict['normal'] = mask_pred
        return pred_dict

    def process_data(self,neighbor_voxels_coord,target_voxels_coord,target):    
        occupancy = tf.ones_like(neighbor_voxels_coord,dtype=tf.float32)
        occupancy = tf.slice(occupancy,[0,0],[-1,1])
        
        neighbor_voxels_coord = tf.cast(neighbor_voxels_coord,tf.float32)
        target_voxels_coord = tf.cast(target_voxels_coord,tf.float32)
        
        return occupancy, neighbor_voxels_coord, target_voxels_coord, target

    def process_data_v2(self,full_voxels_coord,target_voxels_coord,target):

        def unravel_index(flat_indices,dense_shape=(31,31,31)):
            flat_indices = np.array(np.unravel_index(flat_indices,dense_shape))
            #return flat_indices.reshape(1,-1,3)
            return flat_indices

        #target = tf.reshape(target,(1,-1))
        #occupancy = tf.reshape(tf.ones_like(full_voxels_coord),(1,-1))
        occupancy = tf.ones_like(full_voxels_coord)

        full_voxels_coord = tf.py_function(unravel_index,[full_voxels_coord],tf.float32)
        #full_voxels_coord = unravel_index(full_voxels_coord)
        #full_voxels_coord = tf.cast(full_voxels_coord,tf.float32)

        target_voxels_coord = tf.py_function(unravel_index,[target_voxels_coord],tf.float32)
        #target_voxels_coord = unravel_index(target_voxels_coord)
        #target_voxels_coord = tf.cast(target_voxels_coord,tf.float32)

        #def unravel_index(flat_indices,dense_shape=(31,31,31)):
        #    return np.array(np.unravel_index(flat_indices,dense_shape)).T
        
        #full_voxels_coord = tf.py_function(unravel_index,[full_voxels_coord],tf.float32)


        # print('image shape: ',image.shape)
        # print('labels mask shape: ',labels_mask.shape)
        # print('target labels shape: ',target_labels.shape)

        return occupancy,full_voxels_coord,target_voxels_coord,target

    def process_train_bkp(self,full_voxels_coord,target_voxels_coord,target):
        """ 
        Processa coordenadas de voxels de uma imagem de treino.
        Última atualização: 01/04/2022.
        
        Args:
            full_voxels_coord:
            target_voxels_coord:
            target:
                
        Returns:
            image,target_mask
        """         
        # print(type(full_voxels_coord))
        # print(type(target_voxels_coord))
        # print(type(target_labels))
        def sparse_to_dense(coords,values):
            #array = np.zeros(3*[self.input_res[0]]+[1])
            array = np.zeros(
                self.general_config['image_shape'][:-1] + [1])
            array[...,0].flat[coords] = values
            return tf.cast(array,tf.float32)

        # Imagem
        image_channel_1 = tf.numpy_function(
            sparse_to_dense,[full_voxels_coord,1],tf.float32)
        image_channel_2 = tf.numpy_function(
            sparse_to_dense,[target_voxels_coord,1],tf.float32)
        image = tf.concat([image_channel_1,image_channel_2],axis=-1)

        if 'boundary' in self.tasks and 'normal' in self.tasks:
            target_labels = target[0]
            target_normal = target[1]
        else:
            if 'boundary' in self.tasks:
                target_labels = target
            elif 'boundary' in self.tasks:
                target_normal = target                                

        target_mask = []
        if 'boundary' in self.tasks:
            onehot_labels = tf.one_hot(
                target_labels,self.general_config['num_classes'])
            # labels channel 1
            labels_channel_1 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                onehot_labels[:,0]],tf.float32)            
            target_mask.append(labels_channel_1)
            # labels channel 2
            labels_channel_2 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                onehot_labels[:,1]],tf.float32)
            target_mask.append(labels_channel_2)
        if 'normal' in self.tasks:
            normal_channel_1 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                target_normal[:,0]],tf.float32)
            target_mask.append(normal_channel_1)
            normal_channel_2 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                target_normal[:,1]],tf.float32)
            target_mask.append(normal_channel_2)
            if self.general_config['spatial_dimensions']==3:
                normal_channel_3 = tf.numpy_function(
                    sparse_to_dense,[target_voxels_coord,
                    target_normal[:,2]],tf.float32)
                target_mask.append(normal_channel_3)
        
        # Concatena channels
        target_mask = tf.concat(target_mask,axis=-1)
                        
        # print('image shape: ',image.shape)
        # print('labels mask shape: ',labels_mask.shape)
        # print('target labels shape: ',target_labels.shape)

        #return voxels,labels,tf.squeeze(target_labels)
        
        if self.data_augmentation:
            return self.apply_data_augmentation(image,target_mask)
        else:
            return image,target_mask

    def process_validation(self,full_voxels_coord,target_voxels_coord,target):
        """ 
        Processa coordenadas de voxels de uma imagem de validação.
        Última atualização: 01/04/2022.
        
        Args:
            full_voxels_coord:
            target_voxels_coord:
            target:
                
        Returns:
            image,target_mask
        """         
        # print(type(full_voxels_coord))
        # print(type(target_voxels_coord))
        # print(type(target_labels))
        def sparse_to_dense(coords,values):
            #array = np.zeros(3*[self.input_res[0]]+[1])
            array = np.zeros(
                self.general_config['image_shape'][:-1] + [1])
            array[...,0].flat[coords] = values
            return tf.cast(array,tf.float32)

        # Imagem
        image_channel_1 = tf.numpy_function(
            sparse_to_dense,[full_voxels_coord,1],tf.float32)
        image_channel_2 = tf.numpy_function(
            sparse_to_dense,[target_voxels_coord,1],tf.float32)
        image = tf.concat([image_channel_1,image_channel_2],axis=-1)

        if 'boundary' in self.tasks and 'normal' in self.tasks:
            target_labels = target[0]
            target_normal = target[1]
        else:
            if 'boundary' in self.tasks:
                target_labels = target
            elif 'boundary' in self.tasks:
                target_normal = target                                

        target_mask = []
        if 'boundary' in self.tasks:
            onehot_labels = tf.one_hot(
                target_labels,self.general_config['num_classes'])
            # labels channel 1
            labels_channel_1 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                onehot_labels[:,0]],tf.float32)            
            target_mask.append(labels_channel_1)
            # labels channel 2
            labels_channel_2 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                onehot_labels[:,1]],tf.float32)
            target_mask.append(labels_channel_2)
        if 'normal' in self.tasks:
            normal_channel_1 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                target_normal[:,0]],tf.float32)
            target_mask.append(normal_channel_1)
            normal_channel_2 = tf.numpy_function(
                sparse_to_dense,[target_voxels_coord,
                target_normal[:,1]],tf.float32)
            target_mask.append(normal_channel_2)
            if self.general_config['spatial_dimensions']==3:
                normal_channel_3 = tf.numpy_function(
                    sparse_to_dense,[target_voxels_coord,
                    target_normal[:,2]],tf.float32)
                target_mask.append(normal_channel_3)
        
        # Concatena channels
        target_mask = tf.concat(target_mask,axis=-1)
                        
        return image,target_mask

    def process_validation_bkp(
        self,full_voxels_coord,target_voxels_coord,target_labels):
        """ 
        Processa coordenadas de voxels de uma imagem de validação.
        Última atualização: 24/02/2022.
        
        Args:
            full_voxels_coord:
            target_voxels_coord:
            target_labels:
                
        Returns:
            image,label_mask
        """
        # print(type(full_voxels_coord))
        # print(type(target_voxels_coord))
        # print(type(target_labels))
        def sparse_to_dense(coords,values):
            #array = np.zeros(3*[self.input_res[0]]+[1])
            array = np.zeros(
                self.general_config['image_shape'][:-1] + [1])
            array[...,0].flat[coords] = values
            return tf.cast(array,tf.float32)

        # Imagem
        image_channel_1 = tf.numpy_function(
            sparse_to_dense,[full_voxels_coord,1],tf.float32)
        image_channel_2 = tf.numpy_function(
            sparse_to_dense,[target_voxels_coord,1],tf.float32)
        image = tf.concat([image_channel_1,image_channel_2],axis=-1)

        # Labels mask
        onehot_labels = tf.one_hot(
            target_labels,self.general_config['num_classes'])
        labels_channel_1 = tf.numpy_function(
            sparse_to_dense,[target_voxels_coord,onehot_labels[:,0]],tf.float32)
        labels_channel_2 = tf.numpy_function(
            sparse_to_dense,[target_voxels_coord,onehot_labels[:,1]],tf.float32)
        labels_mask = tf.concat([labels_channel_1,labels_channel_2],axis=-1)
                        
        # print('image shape: ',image.shape)
        # print('labels mask shape: ',labels_mask.shape)
        # print('target labels shape: ',target_labels.shape)

        #return voxels,labels,tf.squeeze(target_labels)
        return image,labels_mask

    def apply_data_augmentation(self,image,labels_mask):
        """ 
        Aplica data augmentation.
        Última modificação: 24/02/2022.

        Args:
            image:
            labels_mask:

        Return;
            transformed image and labels_mask.

        """
        # Data-augmentation
        def random_axis_to_flip():
            if self.general_config['spatial_dimensions'] == 2:
                return np.random.choice([0,1])
            if self.general_config['spatial_dimensions'] == 3:
                return np.random.choice([0,1,2])
        
        def random_angle_to_rotate():
            return np.random.choice([0,1,2,3])

        def random_axes_to_rotate():
            if self.general_config['spatial_dimensions'] == 2:
                return np.array([0,1])
            if self.general_config['spatial_dimensions'] == 3:
                return np.random.choice([0,1,2],size=2,replace=False)

        def flip(array,axis):
            return np.flip(array,axis)
        
        def rotate(array,num_rot90,axes):
            return np.rot90(array,num_rot90,axes)
        
        # Aplica flip
        axis = tf.py_function(random_axis_to_flip,[],tf.int32)
        image = tf.py_function(flip,[image,axis],tf.float32)
        labels_mask = tf.py_function(flip,[labels_mask,axis],tf.float32)

        # Aplica rotação com ângulo múltiplo de 90
        num_rot90 = tf.py_function(random_angle_to_rotate,[],tf.int32)
        axes = tf.py_function(random_axes_to_rotate,[],tf.int32)   
        image = tf.py_function(rotate,[image,num_rot90 ,axes],tf.float32)
        labels_mask = tf.py_function(rotate,[labels_mask,num_rot90 ,axes],tf.float32)

        return image,labels_mask

    def unravel_index(self,flat_indices,dense_shape=(31,31,31)):        
        flat_indices = [np.array(np.unravel_index(ind,dense_shape)).T for ind in flat_indices]
        return self.stack_ragged(flat_indices)

    def stack_ragged(self,tensors):
        values = tf.concat(tensors, axis=0)
        lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
        return tf.RaggedTensor.from_row_lengths(values, lens)

    def mix_hdf5_datasets(self,dataset_files,out_dir,dataset_id):
        """
        Merges hdf5 datasets.
        Last modification: 03/28/2022.

        Args:
            dataset_files:
        """ 
        # Loads configurations
        general_configs = np.empty(len(dataset_files),dtype=np.object)
        train_configs = np.empty(len(dataset_files),dtype=np.object)
        val_configs = np.empty(len(dataset_files),dtype=np.object)
        datasets_dirs = np.empty(len(dataset_files),dtype=np.object)
        trainset_files = np.empty(len(dataset_files),dtype=np.object)
        valset_files = np.empty(len(dataset_files),dtype=np.object)
        used_simulations = np.empty(len(dataset_files),dtype=np.object)        

        real_image_length = np.zeros(len(dataset_files),dtype=np.float)
        real_search_radius = np.zeros(len(dataset_files),dtype=np.float)
        ref_length = np.zeros(len(dataset_files),dtype=np.float)
        search_radius = np.zeros(len(dataset_files),dtype=np.float)
        grid_length = np.zeros(len(dataset_files),dtype=np.float)
        real_grid_length = np.zeros(len(dataset_files),dtype=np.float)
        used_steps = np.zeros(len(dataset_files),dtype=np.int)

        num_train_samples = np.zeros(len(dataset_files),dtype=np.int)
        num_val_samples = np.zeros(len(dataset_files),dtype=np.int)

        for i in tqdm(range(len(dataset_files)),desc='Loading dataset specifications'):
            general_configs[i],train_configs[i],val_configs[i] = self.get_dataset_config(dataset_files[i])
            datasets_dirs[i] = general_configs[i]['dir']
            trainset_files[i] = os.path.join(datasets_dirs[i],f"{train_configs[i]['name']}.{train_configs[i]['format']}")
            valset_files[i] = os.path.join(datasets_dirs[i],f"{val_configs[i]['name']}.{val_configs[i]['format']}")
            
            used_simulations[i] = general_configs[i].get('sim_name')

            #real_image_length[i] = general_configs[i]['real_image_length']
            real_search_radius[i] = general_configs[i]['real_search_radius']
            ref_length[i] = general_configs[i]['ref_length']
            search_radius[i] = general_configs[i]['search_radius']
            grid_length[i] = general_configs[i]['grid_length']
            real_grid_length[i] = general_configs[i]['real_grid_length']
            used_steps[i] = general_configs[i]['used_steps']   

            num_train_samples[i] = train_configs[i]['num_batches']
            num_val_samples[i] = val_configs[i]['num_batches']
        
        spatial_dimensions = general_configs[0]['spatial_dimensions']
        tasks = general_configs[0]['tasks']
        task_dimensions = general_configs[0]['task_dimensions']
        task_types = general_configs[0]['task_types']

        # Cria o dataset de treino mesclado caso ele não exista
        os.makedirs(out_dir,exist_ok=True)
        merged_trainset_file = os.path.join(out_dir,'train.hdf5')        
        if not os.path.exists(merged_trainset_file):
            with h5py.File(merged_trainset_file,'a') as f:
                # Attributes
                f.attrs['num_batches'] = 0
                f.attrs['grid_size'] = [2048,2048,2048]
                f.attrs['spatial_dimensions'] = spatial_dimensions
                f.attrs['tasks'] = " ".join(tasks)
                f.attrs['task_dimensions'] = " ".join([str(x) for x in task_dimensions])
                f.attrs['task_types'] = " ".join(task_types)
                
                # Dataset arrays
                f.create_dataset(
                    'neighbor_voxels_num',dtype='int32',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'neighbor_voxels_coord',dtype='float32',shape=(0,3),maxshape=(None,3),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'target_voxels_num',dtype='int32',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'target_voxels_coord',dtype='float32',shape=(0,3),maxshape=(None,3),
                    chunks=True,compression="gzip",compression_opts=9)
            
                for task,task_type,task_dim  in zip(tasks,task_types,task_dimensions):
                    if task_type=='classification':
                        f.create_dataset(
                            f'target_{task}',dtype='float32',shape=(0,task_dim),maxshape=(None,task_dim),
                            chunks=True,compression="gzip",compression_opts=9)
                    if task_type=='regression':
                        f.create_dataset(
                            f'target_{task}',dtype='float32',shape=(0,task_dim),
                            maxshape=(None,task_dim),
                            chunks=True,compression="gzip",compression_opts=9)
                                                
        # Cria o dataset de validação mesclado caso ele não exista
        merged_valset_file = os.path.join(out_dir,'validation.hdf5')
        if not os.path.exists(merged_valset_file):
            with h5py.File(merged_valset_file,'a') as f:
                f.attrs['num_batches'] = 0
                f.attrs['grid_size'] = [2048,2048,2048]
                f.attrs['spatial_dimensions'] = spatial_dimensions
                f.attrs['tasks'] = " ".join(tasks)
                f.attrs['task_dimensions'] = " ".join([str(x) for x in task_dimensions])
                f.attrs['task_types'] = " ".join(task_types)

                f.create_dataset(
                    'neighbor_voxels_num',dtype='int32',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'neighbor_voxels_coord',dtype='float32',shape=(0,3),maxshape=(None,3),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'target_voxels_num',dtype='int32',shape=(0,),maxshape=(None,),
                    chunks=True,compression="gzip",compression_opts=9)
                f.create_dataset(
                    'target_voxels_coord',dtype='float32',shape=(0,3),maxshape=(None,3),
                    chunks=True,compression="gzip",compression_opts=9)
            
                for task,task_type,task_dim  in zip(tasks,task_types,task_dimensions):
                    if task_type=='classification':
                        f.create_dataset(
                            f'target_{task}',dtype='float32',shape=(0,task_dim),maxshape=(None,task_dim),
                            chunks=True,compression="gzip",compression_opts=9)
                    if task_type=='regression':
                        f.create_dataset(
                            f'target_{task}',dtype='float32',shape=(0,task_dim),
                            maxshape=(None,task_dim),
                            chunks=True,compression="gzip",compression_opts=9)

        # Mescla os datasets de treino
        with h5py.File(merged_trainset_file,'a') as mf:
            for i in tqdm(range(len(trainset_files)),desc='Merging training datasets'):
                if not os.path.exists(trainset_files[i]):
                    raise FileNotFoundError('Training set file not found!')
                with h5py.File(trainset_files[i],'r') as f:
                    num_batches = f.attrs['num_batches']
                    neighbor_voxels_num = f['neighbor_voxels_num'][:]
                    neighbor_voxels_coord = f['neighbor_voxels_coord'][:]
                    
                    target_voxels_num = f['target_voxels_num'][:]
                    target_voxels_coord = f['target_voxels_coord'][:]

                    if 'boundary' in tasks:
                        target_boundary = f['target_boundary'][:]
                    if 'normal' in tasks:
                        target_normal = f['target_normal'][:]

                # Attributes
                mf.attrs['num_batches'] += num_batches

                # Full                
                mf['neighbor_voxels_coord'].resize((mf['neighbor_voxels_coord'].shape[0] + neighbor_voxels_coord.shape[0],3))
                mf['neighbor_voxels_coord'][-neighbor_voxels_coord.shape[0]:,:] = neighbor_voxels_coord
                
                mf['neighbor_voxels_num'].resize((mf['neighbor_voxels_num'].shape[0] + neighbor_voxels_num.shape[0],))
                mf['neighbor_voxels_num'][-neighbor_voxels_num.shape[0]:] = neighbor_voxels_num
                                
                # Target
                mf['target_voxels_coord'].resize((mf['target_voxels_coord'].shape[0] + target_voxels_coord.shape[0],3))
                mf['target_voxels_coord'][-target_voxels_coord.shape[0]:,:] = target_voxels_coord
                
                mf['target_voxels_num'].resize((mf['target_voxels_num'].shape[0] + target_voxels_num.shape[0],))
                mf['target_voxels_num'][-target_voxels_num.shape[0]:] = target_voxels_num
            
                if 'boundary' in tasks:
                    mf['target_boundary'].resize((mf['target_boundary'].shape[0] + target_boundary.shape[0],target_boundary.shape[1]))
                    mf['target_boundary'][-target_boundary.shape[0]:,:] = target_boundary
                if 'normal' in tasks:
                    mf['target_normal'].resize((mf['target_normal'].shape[0] + target_normal.shape[0],target_normal.shape[1]))
                    mf['target_normal'][-target_normal.shape[0]:,:] = target_normal
                                    
        # Mescla os datasets de validação
        with h5py.File(merged_valset_file,'a') as mf:
            for i in tqdm(range(len(valset_files)),desc='Merging validation datasets'):
                if not os.path.exists(valset_files[i]):
                    raise FileNotFoundError('Validation set file not found!')
                with h5py.File(valset_files[i],'r') as f:
                    num_batches = f.attrs['num_batches']
                    neighbor_voxels_num = f['neighbor_voxels_num'][:]
                    neighbor_voxels_coord = f['neighbor_voxels_coord'][:]
                    
                    target_voxels_num = f['target_voxels_num'][:]
                    target_voxels_coord = f['target_voxels_coord'][:]

                    if 'boundary' in tasks:
                        target_boundary = f['target_boundary'][:]
                    if 'normal' in tasks:
                        target_normal = f['target_normal'][:]

                # Attributes
                mf.attrs['num_batches'] += num_batches

                # Full                
                mf['neighbor_voxels_coord'].resize((mf['neighbor_voxels_coord'].shape[0] + neighbor_voxels_coord.shape[0],3))
                mf['neighbor_voxels_coord'][-neighbor_voxels_coord.shape[0]:,:] = neighbor_voxels_coord
                
                mf['neighbor_voxels_num'].resize((mf['neighbor_voxels_num'].shape[0] + neighbor_voxels_num.shape[0],))
                mf['neighbor_voxels_num'][-neighbor_voxels_num.shape[0]:] = neighbor_voxels_num
                                
                # Target
                mf['target_voxels_coord'].resize((mf['target_voxels_coord'].shape[0] + target_voxels_coord.shape[0],3))
                mf['target_voxels_coord'][-target_voxels_coord.shape[0]:,:] = target_voxels_coord
                
                mf['target_voxels_num'].resize((mf['target_voxels_num'].shape[0] + target_voxels_num.shape[0],))
                mf['target_voxels_num'][-target_voxels_num.shape[0]:] = target_voxels_num
            
                if 'boundary' in tasks:
                    mf['target_boundary'].resize((mf['target_boundary'].shape[0] + target_boundary.shape[0],target_boundary.shape[1]))
                    mf['target_boundary'][-target_boundary.shape[0]:,:] = target_boundary
                if 'normal' in tasks:
                    mf['target_normal'].resize((mf['target_normal'].shape[0] + target_normal.shape[0],target_normal.shape[1]))
                    mf['target_normal'][-target_normal.shape[0]:,:] = target_normal

        # Expecificações sobre os datasets mesclados
        # Boa parte das expecificações devem ser equivalentes entre todos os datasets mesclados
        dataset_config = ConfigReader(dataset_files[0])
        gen_config = dataset_config.get_section('general')
        train_config = dataset_config.get_section('train_set')
        val_config = dataset_config.get_section('validation_set')
        
        gen_config['dataset_id'] = dataset_id
        gen_config['used_simulations'] = list(used_simulations)
        gen_config['used_steps'] = [int(x) for x in used_steps]
        gen_config['real_image_length'] = [float(x) for x in real_image_length]
        gen_config['real_search_radius'] = [float(x) for x in real_search_radius]
        gen_config['ref_length'] = [float(x) for x in ref_length]
        gen_config['search_radius'] = [float(x) for x in search_radius]
        gen_config['grid_length'] = [float(x) for x in grid_length]
        gen_config['real_grid_length'] = [float(x) for x in real_grid_length]

        train_config['num_batches'] = int(num_train_samples.sum())

        val_config['num_batches'] = int(num_val_samples.sum())

        # Salva arquivo de configuração do dataset mesclado
        dataset_config = {}
        dataset_config['general'] = gen_config
        dataset_config['train_set'] = train_config
        dataset_config['validation_set'] = val_config       

        merge_dataset_config_file = os.path.join(out_dir,'dataset_config_v2.yaml')
        with open(merge_dataset_config_file,'w') as configfile:
            yaml.dump(dataset_config,configfile,default_flow_style=False)
