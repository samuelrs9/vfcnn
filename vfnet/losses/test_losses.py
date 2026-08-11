import numpy as np
import tensorflow as tf
from custom_losses import *
from sdf_losses import *


tutorial = 3.0

if tutorial==0:
    """ Teste de métricas da abordagem pontual caso 2D """
    y_true = tf.constant([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    y_pred = tf.constant([[1,1,0.7,0.7],[1,1,0.4,0.8],[1,1,0.9,0.9]])
    
    mae = MeanAbsoluteErrorPW(
        slice={'begin':[0,2],'size':[-1,2]},reduction='none')
    print('mae: ',mae(y_true,y_pred).numpy())

    mse = MeanSquaredErrorPW(
        slice={'begin':[0,2],'size':[-1,2]},reduction='none')
    print('mse: ',mse(y_true,y_pred).numpy())

    bce = BinaryCrossEntropyPW(
        slice={'begin':[0,0],'size':[-1,2]},
        reduction='auto')
    print('bce: ',bce(y_true,y_pred))

if tutorial==0.1:
    """ Teste de métricas da abordagem regional caso 2D """
    mask_true = np.random.random((4,3,3,5))
    mask_pred = np.random.random((4,3,3,5))    

    mse = MeanSquaredErrorRW(
        reduction='auto',
        slice={'begin':[0,0,0,2],'size':[-1,-1,-1,3]},
        #slice_weights=([0,0,0,0],[-1,-1,-1,2])
    )
    print('mse (with slice): ',mse(mask_true,mask_pred))
    print('mse: ',tf.reduce_mean(mean_squared_error(mask_true[...,2:5],mask_pred[...,2:5])))

    bce = BinaryCrossEntropyRW(
        slice={'begin':[0,0,0,0],'size':[-1,-1,-1,2]},
        slice_weights={'begin':[0,0,0,0],'size':[-1,-1,-1,2]}
    )
    
    print('bce: ',CustomLosses.binary_cross_entropy_rw(
        mask_true[...,0:2],mask_pred[...,0:2]))
    print('bce (with slice)): ',bce(mask_true,mask_pred))    

if tutorial==1.0:
    """ Teste de métricas da abordagem pontual caso 3D """
    y_true = tf.constant(
        [[1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1]])
    y_pred = tf.constant(
        [[1,1,0.7,0.7,0.7],
        [1,1,0.4,0.8,0.8],
        [1,1,0.9,0.9,0.9]])
    
    slice_bce = {'begin':[0,0],'size':[-1,2]}
    slice_mse = {'begin':[0,2],'size':[-1,3]}

    mae = MeanAbsoluteErrorPW(slice=slice_mse,reduction='auto')
    print('mae: ',mae(y_true,y_pred).numpy())

    mse = MeanSquaredErrorPW(slice=slice_mse,reduction='auto')
    print('mse: ',mse(y_true,y_pred).numpy())

    bce = BinaryCrossEntropyPW(
        slice=slice_bce,reduction='auto')
    print('bce: ',bce(y_true,y_pred))

    sum_bce_mse = SumBinaryCEntropyMeanSErrorPW(
        slice={'bce':slice_bce,'mse':slice_mse})
    print('sum_bce_mse: ',sum_bce_mse(y_true,y_pred))    

if tutorial==2.1:
    """ Teste de métricas da abordagem regional caso 2D """
    mask_true = np.zeros((1,2,2,4))
    mask_pred = np.zeros((1,2,2,4))

    mask_true[0,0,0,:] = [1,0,1,1]
    mask_true[0,0,1,:] = [0,1,1,1]
    mask_true[0,1,0,:] = [0,1,1,1]
    mask_true[0,1,1,:] = [0,0,1,1]

    mask_pred[0,0,0,:] = [0.7,0.3,0.9,0.9]
    mask_pred[0,0,1,:] = [0.2,0.8,0.8,0.8]
    mask_pred[0,1,0,:] = [0.8,0.2,0.7,0.7]
    mask_pred[0,1,1,:] = [0,0,1,1]    

    slice_bce = {'begin':[0,0,0,0],'size':[-1,-1,-1,2]}
    slice_mse = {'begin':[0,0,0,2],'size':[-1,-1,-1,2]}

    bce = BinaryCrossEntropyRW(slice=slice_bce,
        slice_weights=slice_bce,name='bce')
    
    mse = MeanSquaredErrorRW(slice=slice_mse,
        slice_weights=slice_bce,name='mse')
    
    sum_bce_mse = SumBinaryCEntropyMeanSErrorRW(
        slice={'bce':slice_bce,'mse':slice_mse},
        slice_weights=slice_bce)

    print('bce: ',bce(mask_true,mask_pred))
    print('mse: ',mse(mask_true,mask_pred))
    print('sum_bce_mse: ',sum_bce_mse(mask_true,mask_pred))

if tutorial==2.1:
    """ Teste de métricas da abordagem regional caso 3D """
    mask_true = np.ones((1,2,2,2,5))
    mask_pred = np.ones((1,2,2,2,5))

    mask_true[0,0,0,0,:] = [1,0,1,1,1]
    mask_true[0,0,1,0,:] = [0,1,1,1,1]
    mask_true[0,1,0,0,:] = [0,1,1,1,1]
    mask_true[0,1,1,0,:] = [0,0,1,1,1]

    mask_pred[0,0,0,0,:] = [0.7,0.3,0.9,0.9,0.9]
    mask_pred[0,0,1,0,:] = [0.2,0.8,0.8,0.8,0.8]
    mask_pred[0,1,0,0,:] = [0.8,0.2,0.7,0.7,0.7]
    mask_pred[0,1,1,0,:] = [0,0,1,1,1]

    slice_bce = {'begin':[0,0,0,0],'size':[-1,-1,-1,2]}
    slice_mse = {'begin':[0,0,0,2],'size':[-1,-1,-1,2]}

    bce = BinaryCrossEntropyRW(slice=slice_bce,
        slice_weights=slice_bce,name='bce')
    
    mse = MeanSquaredErrorRW(slice=slice_mse,
        slice_weights=slice_bce,name='mse')
    
    sum_bce_mse = SumBinaryCEntropyMeanSErrorRW(
        slice={'bce':slice_bce,'mse':slice_mse},
        slice_weights=slice_bce)

    print('bce: ',bce(mask_true,mask_pred))
    print('mse: ',mse(mask_true,mask_pred))
    print('sum_bce_mse: ',sum_bce_mse(mask_true,mask_pred))    


elif tutorial==3:
    """ Testa o erro de estimativa da SDF """ 
    mask_true = np.zeros((3,5,5,2))

    mask_true[0,0,0,:] = [0.6,0.4]
    mask_true[0,0,2,:] = [0.7,0.3]
    mask_true[0,0,4,:] = [0.2,0.7]

    mask_true[1,1,0,:] = [0.67,0.46]
    mask_true[1,1,2,:] = [0.47,0.43]
    mask_true[1,1,4,:] = [0.72,0.47]

    mask_true[2,3,0,:] = [0.96,0.74]
    mask_true[2,3,2,:] = [0.77,0.33]
    mask_true[2,3,4,:] = [0.24,0.87]

    mask_pred = tf.ones((3,5,5,1))
    
    slice_normal = {'begin':[0,0,0,0],'size':[-1,-1,-1,2]}
    slice_sdf = {'begin':[0,0,0,0],'size':[-1,-1,-1,1]}
    sdfloss = SDFLoss(slice={'normal':slice_normal,'sdf':slice_sdf})

    sdfloss(mask_true,mask_pred)