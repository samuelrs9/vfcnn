from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow.keras.losses import *
from tensorflow.keras.regularizers import Regularizer
from vfnet.losses.custom_losses import RegionWiseLoss,MeanSquaredErrorRW
from vfnet.losses.normal_losses import NormalLossRW

class SDFLoss(RegionWiseLoss):

    def __init__(self,slice=None,slice_weights=None,
        ratio=1,reduction='auto',name='shape_loss'):
        super().__init__(slice,slice_weights,ratio,reduction,name)

    def call(self,mask_true,mask_pred,weights_mask=None):
        mask_true,mask_pred = self.cast_to_tensor(mask_true,mask_pred)
        
        normal_mask = self.slice_targets(mask_true,slice=self.slice['normal'])
        sdf_mask = self.slice_targets(mask_pred,slice=self.slice['sdf'])

        weights_mask = tf.cast(tf.greater(tf.reduce_sum(normal_mask,axis=-1),0),dtype=tf.float32)
        weights_mask = weights_mask/tf.reduce_sum(weights_mask,[-2,-1],keepdims=True)

        # COONTINUAR DAQUI

        sdf_grad_mask = self.gradient(sdf_mask)
        
        loss_shape = self.shape_loss(normal_mask,sdf_mask,weights_mask)
        loss_fit = self.fit_loss(self,sdf_mask,weights_mask)
        loss_grad = self.grad_loss(self,sdf_grad_mask)

        return (loss_shape + loss_fit + loss_grad)/3.0
    
    def shape_loss(self,normal_mask,sdf_grad_mask,weights_mask):
        loss = mean_squared_error(normal_mask,sdf_grad_mask)
        return tf.multiply(loss,weights_mask)

    def fit_loss(self,sdf_mask,weights_mask):        
        loss_weighted = tf.multiply(sdf_mask,weights_mask)
        loss_weighted = tf.math.square(loss_weighted)
        return 

    def grad_loss(self,sdf_grad_mask):
        norm2 = tf.reduce_sum(tf.math.square(sdf_grad_mask),axis=-1)
        return tf.reduce_mean(tf.square(norm2-1),axis=[-2,-1])

    def gradient(self,sdf_mask):
        row_var = sdf_mask[:,1:,:,:] - sdf_mask[:,:-1,:,:]
        col_var = sdf_mask[:,:,1:,:] - sdf_mask[:,:,:-1,:]

        row_var = tf.pad(row_var,tf.constant([[0,0],[0,1],[0,0],[0,0]]),"SYMMETRIC")
        col_var = tf.pad(col_var,tf.constant([[0,0],[0,0],[0,1],[0,0]]),"SYMMETRIC")

        return tf.concat([col_var,row_var],axis=-1)

class SDFLoss2(RegionWiseLoss):

    def __init__(self,slice=None,slice_weights=None,
        ratio=1,reduction='auto',name='mse'):
        super().__init__(slice,slice_weights,ratio,reduction,name)

    def call(self,mask_true,mask_pred):
        mask_true,mask_pred = self.cast_to_tensor(mask_true,mask_pred)
        mask1_true,mask1_pred = self.slice_targets(mask_true,mask_pred,self.slice['bce'])
        mask2_true,mask2_pred = self.slice_targets(mask_true,mask_pred,self.slice['mse'])
        
        mask3_pred,_ = self.slice_targets(mask_pred,mask_pred,self.slice['sdf_mse'])
        sdf_pred,sdf_gt,sdf_mask = tf.split(mask3_pred,num_or_size_splits=3,axis=-1)
        
        boundary_bce = self.ratio*binary_crossentropy(mask1_true,mask1_pred)
        normal_mse = self.ratio*mean_squared_error(mask2_true,mask2_pred)
        sdf_mse = self.ratio*mean_squared_error(sdf_gt,sdf_pred)

        if self.slice_weights is not None:
            weights_mask,_ = self.slice_targets(mask_true,mask_pred,self.slice_weights)

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
