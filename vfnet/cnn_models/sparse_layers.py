import tensorflow as tf
import open3d.ml.tf as ml3d
from open3d.ml.utils import MODEL
from tensorflow.keras import regularizers
from keras import initializers

class PreprocessInputs(tf.keras.layers.Layer):

    def __init__(self,num_output_points=[100,100,10],trainable=False,**kwargs):
        super(PreprocessInputs,self).__init__(kwargs)
        self.num_output_points = tf.constant(num_output_points)
        self.input_idx = tf.Variable(0,trainable=False)
        self.trainable = tf.constant(trainable)
        self.paddings = tf.Variable(tf.zeros((2,2),dtype=tf.int32),trainable=False)

    def call(self,inputs):
        #self.input_idx = tf.constant(0,dtype=tf.int32)
        self.input_idx.assign(0)
        input_features_padded = tf.map_fn(
            self.call_one_batch,
            inputs[0],
            #parallel_iterations=4,
            fn_output_signature = tf.TensorSpec(
              #(self.num_output_points[0],tf.shape(inputs[0])[-1]), dtype=tf.float32)
              (None,None), dtype=tf.float32)
              #(None,None), dtype=tf.float32)
        )
        #self.input_idx = tf.constant(1,dtype=tf.int32)
        self.input_idx.assign(1)
        input_positions_padded = tf.map_fn(
            self.call_one_batch,
            inputs[1],
            #parallel_iterations=4,
            fn_output_signature = tf.TensorSpec(
              #(self.num_output_points[1],tf.shape(inputs[1])[-1]), dtype=tf.float32)
              #(100,3), dtype=tf.float32)
              (None,None), dtype=tf.float32)
        )        
        #self.input_idx = tf.constant(2,dtype=tf.int32)
        self.input_idx.assign(2)
        output_positions_padded = tf.map_fn(
            self.call_one_batch,
            inputs[2],
            #parallel_iterations=4,
            fn_output_signature = tf.TensorSpec(
              #(self.num_output_points[2],tf.shape(inputs[2])[-1]), dtype=tf.float32)
              #(2,3), dtype=tf.float32)
              (None,None), dtype=tf.float32)
        )  
        return [input_features_padded, input_positions_padded, output_positions_padded]

    @tf.function
    def call_one_batch(self,inputs):

        def true_fn():
          return tf.slice(inputs,[0,0],[self.num_output_points[self.input_idx],-1])

        def false_fn():
            self.paddings.scatter_nd_update(
                tf.constant([[0,1]]),
                tf.reshape(self.num_output_points[self.input_idx]-tf.shape(inputs)[0],(1,)))
            return tf.pad(inputs,self.paddings.read_value(),"CONSTANT")        

        return tf.cond(inputs.shape[0] >= self.num_output_points[self.input_idx], true_fn, false_fn)

  
    def call_one_batch_(self,inputs):
      
        if inputs.shape[0] >= self.num_output_points[self.input_idx]:
            outputs = tf.slice(inputs,[0,0],[self.num_output_points[self.input_idx],-1])
        else:
            self.paddings.scatter_nd_update(
                tf.constant([[0,1]]),
                tf.reshape(self.num_output_points[self.input_idx]-tf.shape(inputs)[0],(1,)))
            outputs = tf.pad(inputs,self.paddings.read_value(),"CONSTANT")

        return outputs

class SparseConv3D(ml3d.layers.SparseConv):

    def __init__(self,input_channels: int, output_channels: int, input_dense_shape: tuple,**kwargs) -> None:
        super(SparseConv3D,self).__init__(
            filters=output_channels,
            #kernel_initializer=initializers.GlorotNormal(seed=None),
            #kernel_initializer='zeros',
            #kernel_initializer=initializers.HeUniform(seed=None)
            **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_dense_shape = input_dense_shape
        self.output_dense_shape = input_dense_shape
        self.build([tf.TensorShape((None,input_channels)),tf.TensorShape((None,3)),tf.TensorShape((None,3))])

    def build(self,input_shape):        
        super(SparseConv3D,self).build(input_shape[0])

    def call(self,inputs):
        """
        Args: a list of tensors, where
            inputs[0]: input_features
            inputs[1]: input_positions
            inputs[2]: output_positions
        """        
        output_features = super(SparseConv3D,self).call(
          inputs[0], inputs[1], inputs[2], voxel_size=1.0
        )
        return output_features

    def get_config(self):
        config = super(SparseConv3D,self).get_config()
        config.update({
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "input_dense_shape": self.input_dense_shape,
            "output_dense_shape": self.output_dense_shape
        })
        return config

class SparseMaxPooling3D(ml3d.layers.VoxelPooling):

    def __init__(self,pool_size: float, input_dense_shape: tuple, output_dense_shape: tuple = None, **kwargs) -> None:
        super(SparseMaxPooling3D,self).__init__(position_fn='center',feature_fn='max',**kwargs)
        self.pool_size = pool_size
        self.input_dense_shape = input_dense_shape
        if output_dense_shape is None:
            self.output_dense_shape = tuple(int(size/pool_size) for size in input_dense_shape)
        else:
            self.output_dense_shape = output_dense_shape

    def call(self,inputs):
        """
        Args: a list of tensors, where
            inputs[0]: input_features
            inputs[1]: input_positions
        """         
        outputs = super(SparseMaxPooling3D,self).call(inputs[1],inputs[0], self.pool_size)
        return [outputs.pooled_features, outputs.pooled_positions/self.pool_size]

    def get_config(self):
        config = super(SparseMaxPooling3D,self).get_config()
        config.update({
            "pool_size": self.pool_size,
            "input_dense_shape": list(self.input_dense_shape),
            "output_dense_shape": list(self.output_dense_shape)
        })
        return config

class SparseConv3DTranspose(ml3d.layers.SparseConvTranspose):

    def __init__(self,input_channels: int, output_channels: int, input_dense_shape: tuple, 
        output_dense_shape: tuple = None, output_scale=2.0, return_output_positions: bool = False,**kwargs) -> None:
        super(SparseConv3DTranspose,self).__init__(filters=output_channels,**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_dense_shape = input_dense_shape
        self.output_scale = output_scale
        if output_dense_shape is None:
            self.output_dense_shape = tuple(output_scale * size for size in input_dense_shape)
        else:
            self.output_dense_shape = output_dense_shape
        self.return_output_positions = return_output_positions
        self.build([tf.TensorShape((None,input_channels)), tf.TensorShape((None,3)), tf.TensorShape((None,3))])
    
    def build(self,inputs_shape):
        super(SparseConv3DTranspose,self).build(inputs_shape[0])

    def call(self,inputs):
        
        output_features = super(SparseConv3DTranspose,self).call(
          #inputs[0], inputs[1], inputs[2]/tf.cast(self.kernel_size[0], tf.float32), voxel_size=1
          inputs[0], inputs[1], inputs[2]/tf.cast(self.output_scale, tf.float32), voxel_size=1
        )        
        if self.return_output_positions:
            #return [output_features, inputs[2]*tf.cast(self.kernel_size[0],tf.float32)]
            return [output_features, inputs[2]*tf.cast(self.output_scale,tf.float32)]
        else:
            return output_features
        
    def get_config(self):
        config = super(SparseConv3DTranspose,self).get_config()
        config.update({
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "input_dense_shape": self.input_dense_shape,
            "output_dense_shape": self.output_dense_shape,
            "output_scale": self.output_scale,
            "return_output_positions": self.return_output_positions
        })     
        return config   