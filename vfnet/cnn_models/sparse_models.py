import os
import time
import sys
import glob
from tqdm import tqdm
from pkg_resources import split_sections

local_path = os.path.dirname(__file__)
if local_path not in sys.path:
    sys.path.append(local_path)

import trimesh
import numpy as np    
import tensorflow as tf
from tensorflow.keras.layers import *
from custom_layers import Normalize,ArgMax,Split,SampleSDF
from sparse_layers import *

import tensorflow_graphics.geometry.transformation as tfg
from open3d.ml.utils import MODEL


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_input_features=1, input_dense_shape = [1024,1024,1024], **kwargs):
        super().__init__(**kwargs)
        self.num_input_features = num_input_features
        self.input_dense_shape = input_dense_shape
        self.layers = self.build_layers()

    def call(self,inputs):
        
        outputs = []

        features, positions = inputs
        for layer in self.layers:
            if 'sparse_conv' in layer.name:
                features = layer([features,positions,positions])
            if 'sparse_max_pooling' in layer.name:
                features, positions = layer([features,positions])
                outputs.append(positions)
            if 'batch_normalization' in layer.name:
                features = layer(features)
            
        outputs.append(features)

        return outputs[::-1]

    def build_layers(self):
        layers = [
            # Level 0             
            SparseConv3D( # Layer 0
                input_channels = self.num_input_features,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = self.input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 1
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = self.input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 2
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = self.input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 3
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = self.input_dense_shape,
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 4
                pool_size = 2.0,
                input_dense_shape = self.input_dense_shape),
            BatchNormalization(momentum=0.9), # Layer 5
            # Level 1
            SparseConv3D( # Layer 6
                input_channels = 4,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in self.input_dense_shape),
                activation = 'relu'),
            # Level 1
            SparseConv3D( # Layer 7
                input_channels = 8,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in self.input_dense_shape),
                activation = 'relu'),                
            SparseMaxPooling3D( # Layer 8
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.5*size) for size in self.input_dense_shape)),
            BatchNormalization(momentum=0.9), # Layer 9
            # Level 2
            SparseConv3D( # Layer 10
                input_channels = 8,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.25*size) for size in self.input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 11
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.25*size) for size in self.input_dense_shape)),
            BatchNormalization(momentum=0.9), # Layer 12
            # Level 3
            SparseConv3D( # Layer 13
                input_channels = 16,
                output_channels = 32,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.125*size) for size in self.input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 14
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.125*size) for size in self.input_dense_shape)),
            BatchNormalization(momentum=0.9) # Layer 15

        ]
        return layers
    
    def get_config(self):        
        config = super().get_config()
        config.update({'input_dense_shape':self.input_dense_shape,
            'num_input_features':self.num_input_features})
        return config

class BoundaryDecoder(tf.keras.layers.Layer):

    def __init__(self, input_dense_shape=(128,128,128), **kwargs):
        super().__init__(**kwargs)
        self.input_dense_shape = input_dense_shape
        self.layers = self.build_layers()

    def call(self,inputs):        
        features, positions = inputs[0], inputs[1:]
        i = 0
        for layer in self.layers:
            if 'sparse_conv3d_transpose' in layer.name:
                features = layer([features,positions[i],positions[i+1]])
                i += 1
            if 'batch_normalization' in layer.name:
                features = layer(features)
        return features

    def build_layers(self):
        layers = [
            SparseConv3DTranspose(
                input_channels = 32,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = self.input_dense_shape,
                output_dense_shape = tuple(2*size for size in self.input_dense_shape),
                output_scale = 2.0,
                activation = 'relu'),
            BatchNormalization(momentum=0.9),            
            SparseConv3DTranspose(
                input_channels = 16,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(2*size for size in self.input_dense_shape),
                output_dense_shape = tuple(4*size for size in self.input_dense_shape),
                output_scale = 2.0,
                activation = 'relu'),
            BatchNormalization(momentum=0.9),
            SparseConv3DTranspose(
                input_channels = 8,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(4*size for size in self.input_dense_shape),
                output_dense_shape = tuple(8*size for size in self.input_dense_shape),
                output_scale = 2.0,
                activation = 'relu'),
            BatchNormalization(momentum=0.9),
            SparseConv3DTranspose(
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(8*size for size in self.input_dense_shape),
                output_dense_shape =  tuple(16*size for size in self.input_dense_shape),
                output_scale = 2.0,
                activation = 'relu'),
            BatchNormalization(momentum=0.9),
            SparseConv3DTranspose(
                input_channels = 4,
                output_channels = 2,
                kernel_size = [1,1,1],
                input_dense_shape = tuple(16*size for size in self.input_dense_shape),
                output_dense_shape = tuple(16*size for size in self.input_dense_shape),
                output_scale = 1.0,
                activation = 'softmax'),
            ]
        return layers

    def get_config(self):        
        config = super().get_config()
        config.update({'input_dense_shape':self.input_dense_shape})
        return config

class NormalDecoder(tf.keras.layers.Layer):

    def __init__(self, input_dense_shape=(128,128,128), **kwargs):
        super().__init__(**kwargs)
        self.input_dense_shape = input_dense_shape
        self.layers = self.build_layers()

    def call(self,inputs):
        features, positions = inputs[0], inputs[1:]
        i = 0
        for layer in self.layers:
            if 'sparse_conv3d_transpose' in layer.name:
                features = layer([features,positions[i],positions[i+1]])
                i += 1
            if 'batch_normalization' in layer.name:
                features = layer(features)                        
            if 'normalize' in layer.name:
                features = layer(features)
        return features

    def build_layers(self):
        layers = [
            SparseConv3DTranspose(
                input_channels = 32,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = self.input_dense_shape,
                output_dense_shape = tuple(2*size for size in self.input_dense_shape),
                output_scale = 2.0,
                activation = 'linear',
                bias_initializer = tf.keras.initializers.Ones()
                ),
            BatchNormalization(momentum=0.9),            
            SparseConv3DTranspose(
                input_channels = 16,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(2*size for size in self.input_dense_shape),
                output_dense_shape = tuple(4*size for size in self.input_dense_shape),
                output_scale = 2.0,
                activation = 'linear'),
            BatchNormalization(momentum=0.9),
            SparseConv3DTranspose(
                input_channels = 8,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(4*size for size in self.input_dense_shape),
                output_dense_shape = tuple(8*size for size in self.input_dense_shape),
                output_scale = 2.0,
                activation = 'linear'),
            BatchNormalization(momentum=0.9),
            SparseConv3DTranspose(
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(8*size for size in self.input_dense_shape),
                output_dense_shape = tuple(16*size for size in self.input_dense_shape),
                output_scale =2.0,
                activation = 'linear'),
            BatchNormalization(momentum=0.9),
            SparseConv3DTranspose(
                input_channels = 4,
                output_channels = 3,
                kernel_size = [1,1,1],
                input_dense_shape = tuple(16*size for size in self.input_dense_shape),
                output_dense_shape = tuple(16*size for size in self.input_dense_shape),
                output_scale = 1.0,
                activation = 'linear'),
            Normalize(threshold=0.001,name='normalize')
        ]
        return layers
    
    def get_config(self):        
        config = super().get_config()
        config.update({'input_dense_shape':self.input_dense_shape})
        return config

class ProcessOutputs(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arg_max = ArgMax()
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        """
        Args:
            inputs: a list of tensors
                inputs[0]: boundary decoder output
                inputs[1]: normal decoder output
        """
        outputs = {}
        for key in inputs:
            outputs[key] = inputs[key]
        if 'boundary' in inputs:
            boundary_labels = self.arg_max(outputs['boundary'])
            outputs['normal'] = self.multiply([outputs['normal'], boundary_labels])

        return outputs
        #return [boundary_output_features, normal_output_features]
        
    def get_config(self):        
        return super().get_config()

class SparseVoxelizedFluidCNN(tf.keras.Model):

    def __init__(self, tasks: dict = None, num_input_features: int = 1, 
                num_outputs: list = [1], grid_size: tuple = (2048,2048,2048),
                architecture="v3_1", **kwargs):

        super(SparseVoxelizedFluidCNN,self).__init__(**kwargs)
        
        #self.model_manager = ModelManager()

        self.available_tasks = ['boundary','normal']
        self.tasks = tasks
        self.num_input_features = num_input_features
        self.num_outputs = num_outputs
        self.grid_size = grid_size
        self.architecture = architecture

        # Encoder
        self.encoder = Encoder(num_input_features, input_dense_shape=grid_size)

        # Decoders
        self.boundary_decoder = BoundaryDecoder(input_dense_shape=self.encoder.layers[-2].output_dense_shape)
        self.normal_decoder = NormalDecoder(input_dense_shape=self.encoder.layers[-2].output_dense_shape)
        self.process_outputs = ProcessOutputs()
        
        #self.build_model()

    def build_model(self):
        input_features = tf.keras.Input(type_spec=tf.TensorSpec((None,self.num_input_features),tf.float32))
        input_positions = tf.keras.Input(type_spec=tf.TensorSpec((None,3),tf.float32))
        output_positions = tf.keras.Input(type_spec=tf.TensorSpec((None,3),tf.float32))
        self([input_features,input_positions,output_positions])

    def call(self, inputs):

        input_features, input_positions, output_positions = inputs

        # Call encoder
        encoder_outputs = self.encoder([input_features, input_positions])
        encoder_outputs.append(input_positions)
        encoder_outputs.append(output_positions)

        # Call decoders
        decoder_outputs = {}
        if 'boundary' in self.tasks:
            decoder_outputs['boundary'] = self.boundary_decoder(encoder_outputs)
        if 'normal' in self.tasks:
            decoder_outputs['normal'] = self.normal_decoder(encoder_outputs)
        
        return self.process_outputs(decoder_outputs)

    def predict(self, inputs):
        return self(inputs, training=False)

    def compile(self,optimizer=None,losses=None,metrics=None):
        self.optimizer = optimizer
        self.custom_losses = losses
        self.custom_metrics = metrics

    def boundary_regularization(self, inputs, predictions):

        boundary = tf.cast(tf.argmax(predictions['boundary'], axis=1), dtype=tf.bool)

        if tf.reduce_sum(tf.cast(boundary, tf.float32))==0:
            return tf.convert_to_tensor(1.0)

        boundary_coords = tf.boolean_mask(inputs[2], boundary)
        boundary_normal = tf.boolean_mask(predictions['normal'], boundary)

        factor = 5
        new_boundary_coords = boundary_coords + factor*boundary_normal
        new_interior_coords = boundary_coords - factor*boundary_normal

        #first_ortogonal, second_ortogonal = self.compute_ortogonal_vectors(boundary_normal)
        # new_boundary_coords = tf.concat([
        #     boundary_coords - factor*first_ortogonal,
        #     boundary_coords + factor*first_ortogonal,
        #     boundary_coords - factor*second_ortogonal,
        #     boundary_coords + factor*second_ortogonal,
        # ], axis=0)

        # new_interior_coords = tf.concat([
        #     boundary_coords - factor*boundary_normal,
        #     boundary_coords - factor*boundary_normal - factor*first_ortogonal,
        #     boundary_coords - factor*boundary_normal + factor*first_ortogonal, 
        #     boundary_coords - factor*boundary_normal - factor*second_ortogonal,
        #     boundary_coords - factor*boundary_normal + factor*second_ortogonal],
        #     axis=0)

        new_coords = tf.concat([new_boundary_coords, new_interior_coords], axis=0)
        new_predictions = self([inputs[0], inputs[1], new_coords])
        
        new_targets = tf.concat([
            tf.tile([[0.0, 1.0]], [new_boundary_coords.shape[0],1]), 
            tf.tile([[1.0, 0.0]], [new_interior_coords.shape[0],1])],
            axis=0) 

        return self.custom_losses['boundary'](new_targets, new_predictions['boundary'])

        
    def compute_ortogonal_vectors(self, vector):

        first_vector = tf.gather(vector, [1,0,2], axis=-1) * [-1,1,0]
        first_vector = first_vector / tf.norm(first_vector, axis=-1, keepdims=True)

        second_vector = tf.linalg.cross(vector,first_vector)
        second_vector = second_vector / tf.norm(second_vector, axis=-1, keepdims=True)

        return first_vector, second_vector

    def train_step(self, inputs, targets, reg=False):

        with tf.GradientTape(persistent=False) as tape:
            predictions = self(inputs)

            # Sample weights
            if 'boundary' in self.tasks:
                sample_weight = targets['boundary']*(tf.reduce_sum(targets['boundary'],axis=0)/targets['boundary'].shape[0])[::-1]
                sample_weight = tf.reduce_sum(sample_weight,axis=-1)
            
            losses = {}
            for i,task in enumerate(self.tasks):
                losses[task] = self.custom_losses[task](targets[task], predictions[task], sample_weight)
                #losses[task] = self.custom_losses[task](targets[task], predictions[i], sample_weight)
            
            total_loss = losses['boundary'] + losses['normal']
            if reg:
                losses['regularization'] = self.boundary_regularization(inputs, targets)
                total_loss += losses['regularization']
            
            #total_loss = self.custom_losses['total'].call(targets, predictions, sample_weight)
            #total_loss = losses['boundary'] + losses['normal']

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        for i,task in enumerate(self.tasks):
            for metric in self.custom_metrics[task]:                
                if self.custom_metrics[task][metric].name=='cos':
                    self.custom_metrics[task][metric].update_state(targets[task],predictions[task],targets['boundary'][:,1])
                    #self.custom_metrics[task][metric].update_state(targets[task],predictions[task],tf.cast(sample_weight>0.5,tf.float32))
                    #self.custom_metrics[task][metric].update_state(targets[task],predictions[i],tf.cast(sample_weight>0.5,tf.float32))
                else:
                    self.custom_metrics[task][metric].update_state(targets[task],predictions[task])
                    #self.custom_metrics[task][metric].update_state(targets[task],predictions[i])

        losses.update({'total':total_loss})
        metrics = {}
        for task in self.tasks:
            metrics[task] = {self.custom_metrics[task][metric].name: tf.reshape(self.custom_metrics[task][metric].result(),(-1,))[-1] for metric in self.custom_metrics[task]}
                
        return losses, metrics
    
    def validation_step(self, inputs, targets, reg=False):
        predictions = self.predict(inputs)
        
         # Sample weights
        sample_weight = targets['boundary']*(tf.reduce_sum(targets['boundary'],axis=0)/targets['boundary'].shape[0])[::-1]
        sample_weight = tf.reduce_sum(sample_weight,axis=-1)

        losses = {}
        for i,task in enumerate(self.tasks):
            losses[task] = self.custom_losses[task](targets[task],predictions[task], sample_weight)
            #losses[task] = self.custom_losses[task](targets[task],predictions[i], sample_weight)
            #losses[task] = tf.reduce_sum(losses[task])            

        total_loss = losses['boundary'] + losses['normal']        
        if reg:
            losses['regularization'] = self.boundary_regularization(inputs, targets)
            total_loss += losses['regularization']

        #total_loss = 0
        # for loss in losses:
        #     total_loss += losses[loss]

        # Update metrics
        for i,task in enumerate(self.tasks):
            for metric in self.custom_metrics[task]:
                if self.custom_metrics[task][metric].name=='cos':
                    self.custom_metrics[task][metric].update_state(targets[task],predictions[task],targets['boundary'][:,1])
                    #self.custom_metrics[task][metric].update_state(targets[task],predictions[task],tf.cast(sample_weight>0.5,tf.float32))
                    #self.custom_metrics[task][metric].update_state(targets[task],predictions[i],tf.cast(sample_weight>0.5,tf.float32))
                else:
                    self.custom_metrics[task][metric].update_state(targets[task],predictions[task])
                    #self.custom_metrics[task][metric].update_state(targets[task],predictions[i])

    
        losses.update({'total':total_loss})
        metrics = {}
        for task in self.tasks:
            metrics[task] = {self.custom_metrics[task][metric].name: tf.reshape(self.custom_metrics[task][metric].result(),(-1,))[-1] for metric in self.custom_metrics[task]}
                
        return losses, metrics
    
    def fit(self, train_set, train_steps, val_set, val_steps, num_epochs, patience_max, export_dir=None):

        # Restore checkpoint
        checkpoints = [int(checkpoint.split("/")[-1]) for checkpoint in glob.glob(f"{export_dir}/checkpoints/*")]
        if len(checkpoints)>0:
            checkpoints.sort()
            last_epoch = checkpoints[-1]
            print(f"Restoring checkpoint {last_epoch}")
            self.load_weights(f"{export_dir}/checkpoints/{last_epoch}")            
        else:
            last_epoch = 0
            print("No checkpoints found")

        if export_dir!=None:
            report_file = os.path.join(export_dir,"report.npz")

        # Early stop configs
        patience_count = 0
        best_val_loss = np.inf

        if os.path.exists(report_file):
            report = dict(np.load(report_file, allow_pickle=True)['arr_0'].reshape(-1)[0])
        else:
            report = {
                'loss': {
                    'train': {'boundary': [], 'normal': [], 'regularization': [],'total': []}, 
                    'val': {'boundary': [], 'normal': [], 'regularization': [], 'total': []}
                },
                'metrics': {
                    'train': {
                        'boundary': {'rec': [], 'pre': [], 'f1': [], 'mcc': []}, 
                        'normal': {'mse': [], 'cos': []}
                    }, 
                    'val': {
                        'boundary': {'rec': [], 'pre': [], 'f1': [], 'mcc': []}, 
                        'normal': {'mse': [], 'cos': []}
                    }
                },
                'best_epoch': 0
            }

        # Run training loop
        for epoch in range(last_epoch+1,num_epochs):
            print("\nEpoch %d" % epoch)
            print("Train")
            # Iterate over the batches of the dataset.
            train_bar = tqdm(total = train_steps)
            
            batch_train_loss = {'total':[], 'regularization':[]}
            for task in self.tasks:
                batch_train_loss[task] = []

            for step, (occupancy, neighbor_voxels_coord, target_voxels_coord, target) in enumerate(train_set):                
                train_bar.update(1)
                if step > train_steps:
                    break
                
                data_augmentation = False
                if data_augmentation:
                    # Apply random rotation        
                    angles = tf.random.uniform([1,3], minval=0, maxval=2*np.pi)
                    rotation_matrix = tfg.rotation_matrix_3d.from_euler(angles)
                    rot_neighbor_voxels_coord = tfg.rotation_matrix_3d.rotate(neighbor_voxels_coord[0], tf.tile(rotation_matrix, [neighbor_voxels_coord.shape[1],1,1]))
                    rot_target_voxels_coord = tfg.rotation_matrix_3d.rotate(target_voxels_coord[0], tf.tile(rotation_matrix, [target_voxels_coord.shape[1],1,1]))
                    # Update normal vector
                    normal_points = target_voxels_coord[0] + target[1][0]
                    rot_normal_points = tfg.rotation_matrix_3d.rotate(normal_points, tf.tile(rotation_matrix, [normal_points.shape[0],1,1]))
                    rot_normal = rot_normal_points - rot_target_voxels_coord

                    final_target = (target[0][0], rot_normal)

                    #pc_neighbor = trimesh.points.PointCloud(rot_neighbor_voxels_coord.numpy()/2048, colors = np.tile([255,0,0,255], (rot_neighbor_voxels_coord.shape[0], 1)))
                    #pc_target = trimesh.points.PointCloud(rot_target_voxels_coord.numpy()/2048, colors = np.tile([0,0,255,255], (rot_target_voxels_coord.shape[0], 1)))
                    #trimesh.Scene(geometry=[pc_neighbor,pc_target]).show()

                    # Neural network inputs
                    inputs = [occupancy[0], rot_neighbor_voxels_coord, rot_target_voxels_coord]
                    #import pandas as pd
                    #target_points = np.hstack([target[0][0,:,1:2].numpy(), rot_normal.numpy(), rot_target_voxels_coord.numpy()])
                    #df1 = pd.DataFrame(target_points,columns=['label','nx','ny','nz','x','y','z'])
                    #df1.to_csv(f"debug_network_target_points_{step}.csv",index=False)
                    
                    #df2 = pd.DataFrame(neighbor_voxels_coord[0].numpy(),columns=['x','y','z'])
                    #df2.to_csv(f"debug_network_neighbor_points_{step}.csv",index=False)
                    
                    #labels = np.vstack([target[0][0,:,0:1],2*np.ones((neighbor_voxels_coord.shape[1],1))])
                    
                    #all_points = np.vstack([target_voxels_coord[0].numpy(),neighbor_voxels_coord[0].numpy()])
                    #df = pd.DataFrame(np.hstack([labels,points]),columns=['label','x','y','z'])
                    #df.to_csv(f"debug_network_input_{step}.csv",index=False)
                else:                    
                    inputs = [occupancy[0], neighbor_voxels_coord[0], target_voxels_coord[0]]
                    final_target = (target[0][0], target[1][0])

                # Targets
                targets = {}
                for i,task in enumerate(self.tasks):
                    targets[task] = final_target[i]

                # Call train step
                #train_losses, train_metrics = self.train_step(inputs, targets, reg=False)
                train_losses, train_metrics = self.train_step(inputs, targets)
                #train_losses, train_metrics = self.train_step(inputs, targets, epoch=epoch)
                
                for loss in train_losses:
                    batch_train_loss[loss].append(train_losses[loss].numpy())

                train_bar.set_postfix(
                    {'tl':f"{np.mean(batch_train_loss['total']):.4f}",
                    'bl':f"{np.mean(batch_train_loss['boundary']):.4f}",
                    'nl':f"{np.mean(batch_train_loss['normal']):.4f}",
                    'rl':f"{np.mean(batch_train_loss['regularization']):.4f}",
                    'r':f"{train_metrics['boundary']['rec'].numpy():.4f}",
                    'p':f"{train_metrics['boundary']['pre'].numpy():.4f}",
                    'f1':f"{train_metrics['boundary']['f1'].numpy():.4f}",
                    'mcc':f"{train_metrics['boundary']['mcc'].numpy():.4f}",
                    #mse=f"{train_metrics['normal']['mse'].numpy():.4f}",
                    'cos':f"{train_metrics['normal']['cos'].numpy():.4f}"}
                )
            
            train_bar.close()
            
            for k in train_losses:
                report['loss']['train'][k].append(np.mean(batch_train_loss[k]) )
            for k in train_metrics:
                for m in train_metrics[k]:
                    report['metrics']['train'][k][m].append(train_metrics[k][m].numpy())

            #['train_metrics'][].train_metrics
            self.reset_metric_states()            

            # Run a validation loop
            print("Validation")
            val_bar = tqdm(total = val_steps)
            
            batch_val_loss = {'total':[], 'regularization':[]}
            for task in self.tasks:
                batch_val_loss[task] = []

            for step, (occupancy, neighbor_voxels_coord, target_voxels_coord, target) in enumerate(val_set):
                val_bar.update(1)

                if step > val_steps:
                    break

                # Neural network inputs
                inputs = [occupancy[0], neighbor_voxels_coord[0], target_voxels_coord[0]]
                
                # Targets
                targets = {}
                for i,task in enumerate(self.tasks):
                    targets[task] = target[i][0]

                # Call validation step
                val_losses, val_metrics = self.validation_step(inputs, targets, reg=False)
                
                for loss in val_losses:
                    batch_val_loss[loss].append(val_losses[loss].numpy())

                val_bar.set_postfix(
                    {'tl':f"{np.mean(batch_val_loss['total']):.4f}",
                    'bl':f"{np.mean(batch_val_loss['boundary']):.4f}",
                    'nl':f"{np.mean(batch_val_loss['normal']):.4f}",
                    'rl':f"{np.mean(batch_val_loss['regularization']):.4f}",
                    'rec':f"{val_metrics['boundary']['rec'].numpy():.4f}",
                    'pre':f"{val_metrics['boundary']['pre'].numpy():.4f}",
                    #f1=f"{val_metrics['boundary']['f1'].numpy():.4f}",
                    'mcc':f"{val_metrics['boundary']['mcc'].numpy():.4f}",
                    #mse=f"{val_metrics['normal']['mse'].numpy():.4f}",
                    'cos':f"{val_metrics['normal']['cos'].numpy():.4f}"}
                )
            
            val_bar.close()
            
            for k in val_losses:
                report['loss']['val'][k].append(np.mean(batch_val_loss[k]) )
            for k in val_metrics:
                for m in val_metrics[k]:
                    report['metrics']['val'][k][m].append(val_metrics[k][m].numpy())

            self.reset_metric_states()

            # Save checkpoint

            if export_dir!=None:
                np.savez(report_file,report)            
                checkpoint_dir = f"{export_dir}/checkpoints/{epoch}"
                os.makedirs(checkpoint_dir,exist_ok=True)
                self.save(checkpoint_dir)

            patience_count += 1
            if report['loss']['val']['total'][-1] < best_val_loss:
                print(f"Validation loss decrease from {best_val_loss:.6f} to {report['loss']['val']['total'][-1]:.6f}!")
                best_val_loss = report['loss']['val']['total'][-1]
                patience_count = 0
                report['best_epoch'] = epoch
                if export_dir!=None:
                    # https://www.tensorflow.org/guide/keras/save_and_serialize?hl=pt-br
                    #self.save(checkpoint_dir)
                    self.save(export_dir)
            else:
                print(f"Validation loss increase from {best_val_loss:.6f} to {report['loss']['val']['total'][-1]:.6f}!")
            
            if patience_count >= patience_max:
                break      
            
        return report   
        
    def reset_metric_states(self):
        for task in self.tasks:
            for metric in self.custom_metrics[task]:
                self.custom_metrics[task][metric].reset_states()

    def plot_model(self):
        input_features = tf.keras.Input(name='input_features', type_spec=tf.TensorSpec((None,self.num_input_features),tf.float32))
        input_positions = tf.keras.Input(name='input_positions', type_spec=tf.TensorSpec((None,3),tf.float32))
        output_positions = tf.keras.Input(name='output_positions', type_spec=tf.TensorSpec((None,3),tf.float32))
        inputs = [input_features,input_positions,output_positions]
        model_graph = tf.keras.Model(inputs,self.call(inputs) )
        tf.keras.utils.plot_model(model_graph,to_file='model.png',show_shapes=True)

    def get_config(self):        
        config = {
            "tasks": self.tasks,
            "num_input_features": self.num_input_features,
            "num_outputs": self.num_outputs,
            "grid_size": self.grid_size,
            "architecture": self.architecture
        }
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)    