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

    def get_normal_block_v3_4(self):
        """ Sequência de camadas deconvolucionais para previsão das normais. """
        normal_block = [
            SparseConv3DTranspose(
                input_channels = 32,
                output_channels = 16,
                kernel_size = [3,3,3],
                input_dense_shape = self.backbone_block[14].output_dense_shape,
                output_dense_shape = self.backbone_block[14].input_dense_shape,
                output_scale = 2.0,
                activation = 'linear'),
            BatchNormalization(momentum=0.9),            
            SparseConv3DTranspose(
                input_channels = 16,
                output_channels = 8,
                kernel_size = [3,3,3],
                input_dense_shape = self.backbone_block[11].output_dense_shape,
                output_dense_shape = self.backbone_block[11].input_dense_shape,
                output_scale = 2.0,
                activation = 'linear'),
            BatchNormalization(momentum=0.9),
            SparseConv3DTranspose(
                input_channels = 8,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = self.backbone_block[8].output_dense_shape,
                output_dense_shape = self.backbone_block[8].input_dense_shape,
                output_scale = 2.0,
                activation = 'linear'),
            BatchNormalization(momentum=0.9),
            SparseConv3DTranspose(
                input_channels = 4,
                output_channels = 4,
                kernel_size = [3,3,3],
                input_dense_shape = self.backbone_block[4].output_dense_shape,
                output_dense_shape = self.backbone_block[4].input_dense_shape,
                output_scale =2.0,
                activation = 'linear'),
            BatchNormalization(momentum=0.9),
            SparseConv3DTranspose(
                input_channels = 4,
                output_channels = 3,
                kernel_size = [1,1,1],
                input_dense_shape = self.backbone_block[3].output_dense_shape,
                output_dense_shape = self.backbone_block[3].input_dense_shape,
                output_scale = 1.0,
                activation = 'linear'),
            Normalize(threshold=0.1,name='normalize')
        ]
        return normal_block
