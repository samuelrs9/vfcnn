import tensorflow as tf
from tensorflow.keras.losses import *

from losses.boundary_losses import BoundaryLossPW,BoundaryLossRW
from losses.normal_losses import NormalLossPW,NormalLossRW
from losses.sdf_losses import SDFLoss

from losses.custom_losses import PointWiseLoss,RegionWiseLoss

class BoundaryNormalLossPW(PointWiseLoss):

    def __init__(self,slice=None,reduction='auto',name='sum_bce_mse'):
        super().__init__(slice,reduction,name)

    def call(self,y_true,y_pred):
        y_true,y_pred = self.cast_to_tensor(y_true,y_pred)
        
        y1_true = self.slice_targets(y_true,self.slice['bce'])
        y1_pred = self.slice_targets(y_pred,self.slice['bce'])
        
        y2_true = self.slice_targets(y_true,self.slice['mse'])
        y2_pred = self.slice_targets(y_pred,self.slice['mse'])

        bce = binary_crossentropy(y1_true,y1_pred)
        mse = mean_squared_error(y2_true,y2_pred)
        return self.loss_reduction(bce) + self.loss_reduction(mse)

class BoundaryNormalLossRW(RegionWiseLoss):

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
        
        mask3_pred,_ = self.slice_targets(mask_pred,mask_pred,self.slice['sdf_mse'])
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
