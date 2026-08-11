import tensorflow as tf
from tensorflow.keras.losses import *

from vfnet.losses.custom_losses import MeanSquaredErrorPW,MeanSquaredErrorRW

class NormalLossPW(MeanSquaredErrorPW):

    def __init__(self,slice=None,reduction='auto',name='normal_loss'):
        super().__init__(slice,reduction,name)

    def call(self,y_true,y_pred,):
        return super().call(y_true,y_pred)

class NormalLossRW(MeanSquaredErrorRW):

    def __init__(self,slice=None,slice_weights=None,
        ratio=1,reduction='auto',name='mse'):
        super().__init__(slice,slice_weights,ratio,reduction,name)

    def call(self,mask_true,mask_pred,weighted_mask=None):
        return super().call(mask_true,mask_pred,weighted_mask)