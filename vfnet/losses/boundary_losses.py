import tensorflow as tf
from tensorflow.keras.losses import *
from losses.custom_losses import BinaryCrossEntropyPW,BinaryCrossEntropyRW

class BoundaryLossPW(BinaryCrossEntropyPW):

    def __init__(self,slice=None,reduction='auto',name='bound_loss'):
        super().__init__(slice,reduction,name)

    def call(self,y_true,y_pred):
        return super().call(y_true,y_pred)

class BoundaryLossRW(BinaryCrossEntropyRW):

    def __init__(self,slice=None,slice_weights=None,
        ratio=1,reduction='auto', name='bound_loss'):
        super().__init__(slice,slice_weights,ratio,reduction,name)

    def call(self,mask_true,mask_pred):
        return super().call(mask_true,mask_pred)