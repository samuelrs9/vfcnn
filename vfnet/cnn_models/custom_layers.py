import tensorflow as tf
import open3d.ml.tf as ml3d

class Normalize(tf.keras.layers.Layer):

    def __init__(self,threshold=0.1,trainable=True,name='normalize',**kwargs):
        super().__init__(trainable,name,**kwargs)
        self.threshold = threshold
        self.eps = 1e-6

    def call(self,inputs):
        norm = tf.expand_dims(tf.norm(inputs,axis=-1),axis=-1)
        output = inputs/(norm+self.eps)
        activation = tf.math.greater(norm,tf.constant(self.threshold))
        activation = tf.cast(activation,dtype=tf.float32)
        return tf.multiply(output,activation)

    #def get_config(self):
    #   config = super().get_config()
    #    return config.update({'threshold':int(self.threshold)})

class MultiplyTeste(tf.keras.layers.Layer):

    def __init__(self,trainable=False,name='multiply',**kwargs):
        super().__init__(trainable,name,**kwargs)

    def call(self,inputs):        
        return tf.multiply(inputs[0],inputs[1])

class ArgMax(tf.keras.layers.Layer):

    def __init__(self,trainable=True,name='arg_max',**kwargs):
        super().__init__(trainable,name,**kwargs)

    def call(self,inputs):
        arg_max = tf.expand_dims(tf.argmax(inputs,axis=-1),axis=-1)
        return tf.cast(arg_max,dtype=tf.float32)
    
    def get_config(self):
        return super().get_config()

class Split(tf.keras.layers.Layer):

    def __init__(self,splits=2,axis=-1,trainable=False,
        name='split',**kwargs):
        super().__init__(trainable,name,**kwargs)
        self.splits = splits
        self.axis = axis

    def call(self,inputs):

        return tf.split(inputs,num_or_size_splits=self.splits,axis=self.axis)

class TrilinearInterp(tf.keras.layers.Layer):

    def __init__(self,trainable=False,name='trilinear_interp',**kwargs):
        super().__init__(trainable,name,**kwargs)
        self.eps = 1e-6

    def get_config(self):
        return super().get_config()

    def call(self,inputs):                
        grid_3d = tf.convert_to_tensor(value=inputs[0])
        sampling_points = tf.convert_to_tensor(value=inputs[1])

        voxel_cube_shape = tf.shape(input=grid_3d)[-4:-1]
        sampling_points.set_shape(sampling_points.shape)
        batch_dims = tf.shape(input=sampling_points)[:-2]
        num_points = tf.shape(input=sampling_points)[-2]

        bottom_left = tf.floor(sampling_points)
        top_right = bottom_left + 1
        bottom_left_index = tf.cast(bottom_left, tf.int32)
        top_right_index = tf.cast(top_right, tf.int32)
        x0_index, y0_index, z0_index = tf.unstack(bottom_left_index, axis=-1)
        x1_index, y1_index, z1_index = tf.unstack(top_right_index, axis=-1)
        index_x = tf.concat([x0_index, x1_index, x0_index, x1_index,
                            x0_index, x1_index, x0_index, x1_index], axis=-1)
        index_y = tf.concat([y0_index, y0_index, y1_index, y1_index,
                            y0_index, y0_index, y1_index, y1_index], axis=-1)
        index_z = tf.concat([z0_index, z0_index, z0_index, z0_index,
                            z1_index, z1_index, z1_index, z1_index], axis=-1)
        indices = tf.stack([index_x, index_y, index_z], axis=-1)
        clip_value = tf.convert_to_tensor(
            value=[voxel_cube_shape - 1], dtype=indices.dtype)
        indices = tf.clip_by_value(indices, 0, clip_value)
        content = tf.gather_nd(
            params=grid_3d, indices=indices, batch_dims=tf.size(input=batch_dims))
            
        distance_to_bottom_left = sampling_points - bottom_left
        distance_to_top_right = top_right - sampling_points
        x_x0, y_y0, z_z0 = tf.unstack(distance_to_bottom_left, axis=-1)
        x1_x, y1_y, z1_z = tf.unstack(distance_to_top_right, axis=-1)
        weights_x = tf.concat([x1_x, x_x0, x1_x, x_x0,
                            x1_x, x_x0, x1_x, x_x0], axis=-1)
        weights_y = tf.concat([y1_y, y1_y, y_y0, y_y0,
                            y1_y, y1_y, y_y0, y_y0], axis=-1)
        weights_z = tf.concat([z1_z, z1_z, z1_z, z1_z,
                            z_z0, z_z0, z_z0, z_z0], axis=-1)
        weights = tf.expand_dims(weights_x * weights_y * weights_z, axis=-1)

        interpolated_values = weights * content
        
        return tf.add_n(tf.split(interpolated_values, [num_points] * 8, -2))
        
        #return tf.multiply(output,activation)

class SampleSDF(tf.keras.layers.Layer):

    def __init__(self,shitft=9,border=9,axis_direction=None,
        trainable=False,name='sample_sdf',**kwargs):
        super().__init__(trainable,name,**kwargs)
        self.shift = shitft
        self.border = border

        if axis_direction is not None:
            self.axis_direction = tf.constant(axis_direction,dtype=tf.float32)
        else:
            self.axis_direction = tf.constant([1.0,-1.0,1.0],dtype=tf.float32)
        
    def call(self,inputs):

        if inputs[0].shape[0] is None:
            sdf_gt = tf.zeros_like(inputs[0])
            output = inputs[1] 
        else:
            mask = tf.abs(inputs[0])
            mask = tf.reduce_sum(mask,axis=-1)
            mask = tf.not_equal(mask,0)

            # Valores das normais
            normal = tf.boolean_mask(inputs[0],mask)
            # Corrige o sentido dos eixos
            normal = tf.multiply(normal,self.axis_direction)

            # Inclui a dimensão dos batches
            #zeros = tf.zeros((normal.shape[0],1),dtype=tf.float32)
            #normal = tf.concat([zeros,normal],axis=-1)
            normal = tf.pad(normal,tf.constant([[0,0],[1,0]]),"CONSTANT")

            # Índices de pontos do nível 0
            idx = tf.where(mask)

            # Remove índices fora da região de confiança
            cond0 = tf.cast(idx >= tf.constant([0,self.border,self.border],dtype=tf.int64),dtype=tf.int8)
            cond0 = tf.equal(tf.reduce_sum(cond0,axis=-1),tf.constant([3],dtype=tf.int8))
            cond1 = tf.cast(idx < mask.shape - tf.constant([0,self.border,self.border],dtype=tf.int64),dtype=tf.int8)
            cond1 = tf.equal(tf.reduce_sum(cond1,axis=-1),tf.constant([3],dtype=tf.int8))            
            cond = tf.logical_and(cond0,cond1)
            
            idx = tf.boolean_mask(idx,cond)
            normal = tf.boolean_mask(normal,cond)

            # Sentido da normal
            idx_up1 = idx + tf.cast(0.5*self.shift*normal,dtype=tf.int64)        
            idx_up2 = idx + tf.cast(self.shift*normal,dtype=tf.int64)
            
            # Sentido inverso ao da normal
            idx_down1 = idx - tf.cast(0.5*self.shift*normal,dtype=tf.int64)
            idx_down2 = idx - tf.cast(self.shift*normal,dtype=tf.int64)

            zeros = tf.zeros(idx.shape[0])
            ones = tf.ones(idx.shape[0])
            
            sample_idx = tf.concat(
                [idx,idx_up1,idx_up2,idx_down1,idx_down2],axis=0)
            sample_values = tf.concat(
                [zeros,0.5*ones,ones,-0.5*ones,-ones],axis=0)

            # Atualiza valores
            sample_sdf = tf.scatter_nd(
                sample_idx,sample_values,mask.shape)
            
            mask_sdf = tf.scatter_nd(
                sample_idx,tf.ones(sample_idx.shape[0]),mask.shape)
            mask_sdf = tf.cast(mask_sdf>tf.constant(0,dtype=tf.float32),dtype=tf.float32)
            
            mask_target = tf.pad(
                tf.ones(mask.shape[1:]-tf.constant(2*self.border)),
                tf.constant([[9,9],[9,9]]))

            sample_sdf = tf.multiply(sample_sdf,mask_target)
            mask_sdf = tf.multiply(mask_sdf,mask_target)
            output = tf.multiply(tf.squeeze(inputs[1]),mask_target)
            output = expand_dims(output,axis=-1)

            sdf_gt = tf.concat(
                [tf.expand_dims(sample_sdf,axis=-1),
                tf.expand_dims(mask_sdf,axis=-1)],axis=-1)

        return output,sdf_gt

