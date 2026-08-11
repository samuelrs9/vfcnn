import tensorflow as tf
from tensorflow.keras.losses import *

class CustomLosses:

    @staticmethod
    def loss_function_1(y_true,y_pred,from_logits=False):
        """ 
        Função de perda personalizada 1.        
        Última modificação: 31/10/2021.
        
        Args:
            y_true:
            y_pred:       
                
        Returns:
            Loss.
        """
        bce = BinaryCrossentropy(from_logits=from_logits,
            reduction=tf.keras.losses.Reduction.NONE)
        L = bce(y_true,y_pred) 

        #L = 10000*tf.keras.losses.binary_crossentropy(y_true,y_pred) 
        #L = binary_crossentropy(y_true,y_pred) 
        #L = 10000*tf.keras.losses.mean_squared_error(true_image,pred_image)
        return L   

    @staticmethod
    def binary_cross_entropy_rw(mask_true,mask_pred):
        """ 
        Função de entropia cruzada binária para problemas de classificação
        da abordagem regional.
        Última modificação: 04/04/2022.
        
        Args:
            mask_true:
            mask_pred:
                
        Returns:
            Loss.
        """
        bce_loss = binary_crossentropy(mask_true,mask_pred)

        interior_mask,boundary_mask = tf.split(mask_true,num_or_size_splits=2, axis=-1)
        #interior_mask = tf.reshape(interior_mask,tf.TensorShape(interior_mask.shape[0:-1]))
        interior_mask = tf.squeeze(interior_mask)
        #boundary_mask = tf.reshape(boundary_mask,tf.TensorShape(boundary_mask.shape[0:-1]))
        boundary_mask = tf.squeeze(boundary_mask)

        weighted_interior_mask = interior_mask/tf.reduce_sum(interior_mask)
        weighted_boundary_mask = boundary_mask/tf.reduce_sum(boundary_mask)
        weighted_mask = tf.add(weighted_interior_mask,weighted_boundary_mask)

        weighted_bce_loss = tf.multiply(bce_loss,weighted_mask)

        return tf.reduce_mean(weighted_bce_loss)

    @staticmethod
    def loss_function_3(y_true,y_pred):
        """ 
        Função de perda personalizada 2.        
        Última modificação: 31/10/2021.
        
        Args:
            y_true:
            y_pred:       
                
        Returns:
            Loss.
        """
        y_pred = y_pred[0]

        L0 = 10000*binary_crossentropy(y_true,y_pred)
        int_labels,bound_labels = tf.split(y_true,num_or_size_splits=2, axis=3)
        int_labels = tf.reshape(int_labels,int_labels.shape[0:-1])
        bound_labels = tf.reshape(bound_labels,bound_labels.shape[0:-1])
        L = tf.multiply(L0,int_labels) + 2*tf.multiply(L0,bound_labels)
        #L = 10000*tf.keras.losses.mean_squared_error(true_image,pred_image)
        return L

    @staticmethod
    def loss_function_4(y_true,y_pred):
        """ 
        Função de perda personalizada 4.
        Última modificação: 28/03/2022.
        
        Args:
            y_true:
            y_pred:
                
        Returns:
            Loss.
        """
        labels_dim = 2
        normal_dim = y_true.shape[-1]-labels_dim
        # Separa labels e normais
        labels_true,normal_true = tf.split(y_true,[labels_dim,normal_dim],axis=-1)
        labels_pred,normal_pred = tf.split(y_pred,[labels_dim,normal_dim],axis=-1)
        # Custos
        labels_loss = binary_crossentropy(labels_true,labels_pred)
        normal_loss = mean_squared_error(normal_true,normal_pred)
        return labels_loss + normal_loss

# Pointwise approach
class PointWiseLoss(Loss):

    def __init__(self,slice=None,reduction='none',name='pointwise_loss'):
        super().__init__(reduction,name)
        self.slice = slice

    def slice_targets(self,array,slice=None):
        if slice is not None:
            array = tf.slice(
                array,slice['begin'],slice['size'])
        return array

    def cast_to_tensor(self,y_true,y_pred):
        y_true = tf.cast(y_true,dtype=tf.float32)
        y_pred = tf.cast(y_pred,dtype=tf.float32)
        return y_true,y_pred

    def loss_reduction(self,loss):
        if self.reduction == 'auto':
           return tf.reduce_mean(loss)
        elif self.reduction == 'sum':
           return tf.reduce_sum(loss)
        elif self.reduction == 'none':
            return loss

class MeanAbsoluteErrorPW(PointWiseLoss):

    def __init__(self,slice=None,reduction='auto',name='mae'):
        super().__init__(slice,reduction,name)

    def call(self,y_true,y_pred):
        y_true,y_pred = self.cast_to_tensor(y_true,y_pred)
        y_true = self.slice_targets(y_true,self.slice)
        y_pred = self.slice_targets(y_pred,self.slice)
        mae = mean_absolute_error(y_true,y_pred)
        return self.loss_reduction(mae)

class BinaryCrossEntropyPW(PointWiseLoss):

    def __init__(self,slice=None,reduction='auto',name='bce'):
        super().__init__(slice,reduction,name)

    def call(self,y_true,y_pred):
        y_true,y_pred = self.cast_to_tensor(y_true,y_pred)
        y_true = self.slice_targets(y_true,self.slice)
        y_pred = self.slice_targets(y_pred,self.slice)
        bce = binary_crossentropy(y_true,y_pred)
        return self.loss_reduction(bce)

class MeanSquaredErrorPW(PointWiseLoss):

    def __init__(self,slice=None,reduction='auto',name='mse'):
        super().__init__(slice,reduction,name)

    def call(self,y_true,y_pred):
        y_true,y_pred = self.cast_to_tensor(y_true,y_pred)
        y_true = self.slice_targets(y_true,self.slice)
        y_pred = self.slice_targets(y_pred,self.slice)
        mse = mean_squared_error(y_true,y_pred)
        return self.loss_reduction(mse)

class BinaryCrossEntropyPW(PointWiseLoss):

    def __init__(self,slice=None,reduction='sum',name='bce'):
        super().__init__(slice,reduction,name)

    def call(self,y_true,y_pred):
        y_true,y_pred = self.cast_to_tensor(y_true,y_pred)
        y_true = self.slice_targets(y_true,self.slice)
        y_pred = self.slice_targets(y_pred,self.slice)
        bce = binary_crossentropy(y_true,y_pred)
        return self.loss_reduction(bce)

class SumBinaryCEntropyMeanSErrorPW(PointWiseLoss):

    def __init__(self,slice=None,reduction='auto',name='sum_bce_mse'):
        super().__init__(slice,reduction,name)

    def call(self,y_true,y_pred):
        y_true,y_pred = self.cast_to_tensor(y_true,y_pred)
        y1_true,y1_pred = self.slice_targets(y_true,y_pred,self.slice['bce'])
        y2_true,y2_pred = self.slice_targets(y_true,y_pred,self.slice['mse'])
        bce = binary_crossentropy(y1_true,y1_pred)
        mse = mean_squared_error(y2_true,y2_pred)
        return self.loss_reduction(bce) + self.loss_reduction(mse)

# Regionwise approach
class RegionWiseLoss(Loss):

    def __init__(self,slice=None,slice_weights=None,
        ratio=1,reduction='auto',name='regionwise_loss'):
        super().__init__(reduction,name)
        self.slice = slice
        self.slice_weights = slice_weights
        self.ratio = ratio

    def slice_targets(self,mask,slice=None):
        if slice is not None:
            mask = tf.slice(mask,slice['begin'],slice['size'])
        return mask

    def slice_targets_bkp(self,mask_true,mask_pred=None,slice=None):
        if slice is not None:
            mask_true = tf.slice(
                mask_true,slice['begin'],slice['size'])
            if mask_pred is not None:
                mask_pred = tf.slice(
                    mask_pred,slice['begin'],slice['size'])
        return mask_true,mask_pred

    def cast_to_tensor(self,mask_true,mask_pred):
        mask_true = tf.cast(mask_true,dtype=tf.float32)
        mask_pred = tf.cast(mask_pred,dtype=tf.float32)
        return mask_true,mask_pred

    def loss_reduction(self,loss):
        if self.reduction == 'auto':
           return tf.reduce_mean(loss,axis=-1)
        elif self.reduction == 'sum':
           return tf.reduce_sum(loss,axis=-1)
        elif self.reduction == 'none':
            return loss

class BinaryCrossEntropyRW(RegionWiseLoss):

    def __init__(self,slice=None,slice_weights=None,
        ratio=1,reduction='auto', name='regionwise_loss'):
        super().__init__(slice,slice_weights,ratio,reduction,name)

    def call(self,mask_true,mask_pred):
        mask_true,mask_pred = self.cast_to_tensor(mask_true,mask_pred)
        
        mask_true = self.slice_targets(mask_true,self.slice)
        mask_pred = self.slice_targets(mask_pred,self.slice)
        
        bce = self.ratio*binary_crossentropy(mask_true,mask_pred)

        if self.slice_weights is not None:
            weights_mask = self.slice_targets(mask_true,self.slice_weights)

            interior_mask,boundary_mask = tf.split(weights_mask,num_or_size_splits=2,axis=-1)
            interior_mask = tf.squeeze(interior_mask)
            boundary_mask = tf.squeeze(boundary_mask)

            weighted_interior_mask = interior_mask/tf.reduce_sum(interior_mask)
            weighted_boundary_mask = boundary_mask/tf.reduce_sum(boundary_mask)
            weighted_mask = 0.5*tf.add(weighted_interior_mask,weighted_boundary_mask)

            bce = tf.multiply(bce,weighted_mask)

        return tf.reduce_sum(bce)

class MeanSquaredErrorRW(RegionWiseLoss):

    def __init__(self,slice=None,slice_weights=None,
        ratio=1,reduction='auto',name='mse'):
        super().__init__(slice,slice_weights,ratio,reduction,name)

    def call(self,mask_true,mask_pred,weighted_mask=None):
        mask_true,mask_pred = self.cast_to_tensor(mask_true,mask_pred)
        
        _mask_true = self.slice_targets(mask_true,self.slice)
        _mask_pred = self.slice_targets(mask_pred,self.slice)
        
        mse = self.ratio*mean_squared_error(_mask_true,_mask_pred)

        if self.slice_weights is not None:
            weights_mask = self.slice_targets(mask_true,self.slice_weights)

            if weights_mask.shape[-1] > 1:
                interior_mask,boundary_mask = tf.split(weights_mask,num_or_size_splits=2,axis=-1)
                
                interior_mask = tf.squeeze(interior_mask)
                boundary_mask = tf.squeeze(boundary_mask)

                weighted_interior_mask = interior_mask/tf.reduce_sum(interior_mask)
                weighted_boundary_mask = boundary_mask/tf.reduce_sum(boundary_mask)
                weighted_mask = 0.5*tf.add(weighted_interior_mask,weighted_boundary_mask)
                
                mse = tf.multiply(mse,weighted_mask)

            elif weights_mask.shape[-1] == 1:
                weighted_mask = tf.squeeze(weights_mask)/tf.reduce_sum(weights_mask)
                mse = tf.multiply(mse,weighted_mask)
    
        return tf.reduce_sum(mse)

class SumBinaryCEntropyMeanSErrorRW(RegionWiseLoss):

    def __init__(self,slice=None,slice_weights=None,
        ratio=1,reduction='auto',name='mse'):
        super().__init__(slice,slice_weights,ratio,reduction,name)

    def call(self,mask_true,mask_pred):
        mask_true,mask_pred = self.cast_to_tensor(mask_true,mask_pred)
        
        mask1_true = self.slice_targets(mask_true,self.slice['bce'])
        mask1_pred = self.slice_targets(mask_pred,self.slice['bce'])

        mask2_true = self.slice_targets(mask_true,self.slice['mse'])
        mask2_pred = self.slice_targets(mask_pred,self.slice['mse'])
        
        bce = self.ratio*binary_crossentropy(mask1_true,mask1_pred)
        mse = self.ratio*mean_squared_error(mask2_true,mask2_pred)

        if self.slice_weights is not None:
            weights_mask = self.slice_targets(mask_true,self.slice_weights)

            interior_mask,boundary_mask = tf.split(weights_mask,num_or_size_splits=2,axis=-1)
            interior_mask = tf.squeeze(interior_mask)
            boundary_mask = tf.squeeze(boundary_mask)

            weighted_interior_mask = interior_mask/tf.reduce_sum(interior_mask)
            weighted_boundary_mask = boundary_mask/tf.reduce_sum(boundary_mask)
            weighted_mask = 0.5*tf.add(weighted_interior_mask,weighted_boundary_mask)

            bce = tf.multiply(bce,weighted_mask)
            mse = tf.multiply(mse,weighted_mask)

        bce = tf.reduce_sum(bce)
        mse = tf.reduce_sum(mse)

        return bce + mse

class FullLoss(RegionWiseLoss):

    def __init__(self,slice=None,slice_weights=None,
        ratio=1,reduction='auto',name='mse'):
        super().__init__(slice,slice_weights,ratio,reduction,name)

    def call(self,mask_true,mask_pred):
        mask_true,mask_pred = self.cast_to_tensor(mask_true,mask_pred)

        mask1_true = self.slice_targets(mask_true,self.slice['bce'])
        mask1_pred = self.slice_targets(mask_pred,self.slice['bce'])
        
        mask2_true = self.slice_targets(mask_true,self.slice['mse'])
        mask2_pred = self.slice_targets(mask_pred,self.slice['mse'])
        
        mask3_pred = self.slice_targets(mask_pred,self.slice['sdf_mse'])

        sdf_pred,sdf_gt,sdf_mask = tf.split(mask3_pred,num_or_size_splits=3,axis=-1)
        
        boundary_bce = self.ratio*binary_crossentropy(mask1_true,mask1_pred)
        normal_mse = self.ratio*mean_squared_error(mask2_true,mask2_pred)
        sdf_mse = self.ratio*mean_squared_error(sdf_gt,sdf_pred)

        if self.slice_weights is not None:
            weights_mask = self.slice_targets(mask_true,self.slice_weights)

            interior_mask,boundary_mask = tf.split(weights_mask,num_or_size_splits=2,axis=-1)
            interior_mask = tf.squeeze(interior_mask)
            boundary_mask = tf.squeeze(boundary_mask)

            weighted_interior_mask = interior_mask/tf.reduce_sum(interior_mask)
            weighted_boundary_mask = boundary_mask/tf.reduce_sum(boundary_mask)
            weighted_mask = 0.5*tf.add(weighted_interior_mask,weighted_boundary_mask)

            boundary_bce = tf.multiply(boundary_bce,weighted_mask)
            normal_mse = tf.multiply(normal_mse,weighted_mask)

            sdf_mask = tf.squeeze(sdf_mask)/tf.reduce_sum(sdf_mask)
            sdf_mse = tf.multiply(sdf_mse,sdf_mask)

        boundary_bce = tf.reduce_sum(boundary_bce)
        normal_mse = tf.reduce_sum(normal_mse)
        sdf_mse = tf.reduce_sum(sdf_mse)

        return boundary_bce + normal_mse + sdf_mse

# Sparse regionwise approach