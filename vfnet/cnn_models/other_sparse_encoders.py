import os
import time
import sys
from tqdm import tqdm
from pkg_resources import split_sections

local_path = os.path.dirname(__file__)
if local_path not in sys.path:
    sys.path.append(local_path)

import numpy as np    
import tensorflow as tf
from tensorflow.keras.layers import *
from custom_layers import Normalize,ArgMax,Split,SampleSDF
from sparse_layers import *

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

    def build_layers_v3_4(self):
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
    
    def build_layers_v1(self,input_dense_shape = [1024,1024,1024]):
        """ Sequência de camadas convolucionais esparsas para extração de features. """
        backbone_block = [
            # Level 0
            SparseConv3D(
                input_channels = self.num_input_features,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = input_dense_shape),
            # Level 1
            SparseConv3D(
                input_channels = 4,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape)),
            # Level 2
            SparseConv3D(
                input_channels = 8,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape)),
            # Level 3
            SparseConv3D(
                input_channels = 16,
                output_channels = 32,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.125*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.125*size) for size in input_dense_shape),
                output_dense_shape = tuple(int(0.0625*size) for size in input_dense_shape)),
        ]
        return backbone_block

    def build_layers_v2(self,input_dense_shape = [1024,1024,1024]):
        """ Sequência de camadas convolucionais esparsas para extração de features. """
        backbone_block = [
            # Level 0 - 1
            SparseConv3D(
                input_channels = self.num_input_features,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D(
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D(
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D(
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = input_dense_shape),
            # Level 1
            SparseConv3D(
                input_channels = 4,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape)),
            # Level 2
            SparseConv3D(
                input_channels = 8,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape)),
            # Level 3
            SparseConv3D(
                input_channels = 16,
                output_channels = 32,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.125*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.125*size) for size in input_dense_shape),
                output_dense_shape = tuple(int(0.0625*size) for size in input_dense_shape)),
        ]
        return backbone_block

    def build_layers_v2_1(self,input_dense_shape = [1024,1024,1024]):
        """ Sequência de camadas convolucionais esparsas para extração de features. """
        backbone_block = [
            # Level 0 - 1
            SparseConv3D(
                input_channels = self.num_input_features,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D(
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D(
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D(
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = input_dense_shape),
            BatchNormalization(momentum=0.9),
            # Level 1
            SparseConv3D(
                input_channels = 4,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9),
            # Level 2
            SparseConv3D(
                input_channels = 8,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9),
            # Level 3
            SparseConv3D(
                input_channels = 16,
                output_channels = 32,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.125*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D(
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.125*size) for size in input_dense_shape),
                output_dense_shape = tuple(int(0.0625*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9),
        ]
        return backbone_block

    def build_layers_v3_1(self,input_dense_shape = [1024,1024,1024]):
        """ Sequência de camadas convolucionais esparsas para extração de features. """
        backbone_block = [
            # Level 0             
            SparseConv3D( # Layer 0
                input_channels = self.num_input_features,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 1
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 2
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 3
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 4
                pool_size = 2.0,
                input_dense_shape = input_dense_shape),
            BatchNormalization(momentum=0.9), # Layer 5
            # Level 1
            SparseConv3D( # Layer 6
                input_channels = 4,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape),
                activation = 'relu'),
            # Level 1
            SparseConv3D( # Layer 7
                input_channels = 8,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape),
                activation = 'relu'),                
            SparseMaxPooling3D( # Layer 8
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9), # Layer 9
            # Level 2
            SparseConv3D( # Layer 10
                input_channels = 8,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 11
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9) # Layer 12
        ]
        return backbone_block

    def build_layers_v3_3(self,input_dense_shape = [1024,1024,1024]):
        """ Sequência de camadas convolucionais esparsas para extração de features. """
        backbone_block = [
            # Level 0             
            SparseConv3D( # Layer 0
                input_channels = self.num_input_features,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 1
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 2
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 3
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 4
                pool_size = 2.0,
                input_dense_shape = input_dense_shape),
            BatchNormalization(momentum=0.9), # Layer 5
            # Level 1
            SparseConv3D( # Layer 6
                input_channels = 4,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape),
                activation = 'relu'),
            # Level 1
            SparseConv3D( # Layer 7
                input_channels = 8,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape),
                activation = 'relu'),                
            SparseMaxPooling3D( # Layer 8
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9), # Layer 9
            # Level 2
            SparseConv3D( # Layer 10
                input_channels = 8,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 11
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9), # Layer 12
            # Level 3
            SparseConv3D( # Layer 13
                input_channels = 16,
                output_channels = 32,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.125*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 14
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.125*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9) # Layer 15

        ]
        return backbone_block

    def build_layers_v3_2(self,input_dense_shape = [1024,1024,1024]):
        """ Sequência de camadas convolucionais esparsas para extração de features. """
        backbone_block = [
            # Level 0             
            SparseConv3D( # Layer 0
                input_channels = self.num_input_features,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 1
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 2
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 3
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 4
                pool_size = 2.0,
                input_dense_shape = input_dense_shape),
            BatchNormalization(momentum=0.9), # Layer 5
            # Level 1
            SparseConv3D( # Layer 6
                input_channels = 4,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape),
                activation = 'relu'),
            # Level 1
            SparseConv3D( # Layer 7
                input_channels = 8,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape),
                activation = 'relu'),                
            SparseMaxPooling3D( # Layer 8
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9), # Layer 9
            # Level 2
            SparseConv3D( # Layer 10
                input_channels = 8,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 11
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9) # Layer 12
        ]
        return backbone_block

    def build_layers_v4(self,input_dense_shape = [1024,1024,1024]):
        """ Sequência de camadas convolucionais esparsas para extração de features. """
        backbone_block = [
            # Level 0             
            SparseConv3D( # Layer 0
                input_channels = self.num_input_features,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 1
                input_channels = 8,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 2
                input_channels = 8,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseConv3D( # Layer 3
                input_channels = 8,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = input_dense_shape,
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 4
                pool_size = 2.0,
                input_dense_shape = input_dense_shape),
            BatchNormalization(momentum=0.9), # Layer 5
            # Level 1
            SparseConv3D( # Layer 6
                input_channels = 8,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape),
                activation = 'relu'),
            # Level 1
            SparseConv3D( # Layer 7
                input_channels = 16,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape),
                activation = 'relu'),                
            SparseMaxPooling3D( # Layer 8
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.5*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9), # Layer 9
            # Level 2
            SparseConv3D( # Layer 10
                input_channels = 16,
                output_channels = 32,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 11
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.25*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9), # Layer 12
            # Level 3
            SparseConv3D( # Layer 13
                input_channels = 32,
                output_channels = 64,
                kernel_size = [3,3,3],
                input_dense_shape = tuple(int(0.125*size) for size in input_dense_shape),
                activation = 'relu'),
            SparseMaxPooling3D( # Layer 14
                pool_size = 2.0,
                input_dense_shape = tuple(int(0.125*size) for size in input_dense_shape)),
            BatchNormalization(momentum=0.9) # Layer 15

        ]
        return backbone_block

    def get_config(self):
        config = super().get_config()
        config.update({'input_dense_shape':self.input_dense_shape,
            'num_input_features':self.num_input_features})
        return config
