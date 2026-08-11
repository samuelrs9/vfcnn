import numpy as np
import tensorflow as tf
from custom_metrics import *

#tf.config.run_functions_eagerly(True)

tutorial = 1

if tutorial==0:
    y_true = np.array(
        [[0,1,1,1],
        [1,0,0.8,0.8],
        [1,0,0.7,0.6],
        [1,0,0.5,0.6]])
    y_pred = np.array(
        [[0,1,0,1],
        [1,0,0.8,0.8],
        [1,0,0.7,0.6],
        [1,0,0.5,0.6]])
    
    precision = PrecisionPW(slice={'begin':[0,0],'size':[-1,2]})
    precision.update_state(y_true,y_pred)
    print('precision: ',precision.result().numpy())

    recall = RecallPW(slice={'begin':[0,0],'size':[-1,2]})
    recall.update_state(y_true,y_pred)
    print('recall: ',recall.result().numpy())
    
    f1 = F1ScorePW(slice={'begin':[0,0],'size':[-1,2]})
    f1.update_state(y_true,y_pred)
    print('f1 score: ',f1.result().numpy())

    mcc = MatthewsCoefficientPW(slice={'begin':[0,0],'size':[-1,2]})
    mcc.update_state(y_true,y_pred)
    print('mcc: ',mcc.result().numpy())


elif tutorial==1:
    mask_true = np.zeros((2,3,3,4))
    mask_pred = np.zeros((2,3,3,4))

    mask_true[0,0,:,:] = [1,0,1,1]
    mask_true[0,1,:,:] = [0,1,0.5,0.5]
    mask_true[0,:,1,:] = [0,1,0.6,0.7]

    mask_pred[0,0,:,:] = [0.8,0.2,0.9,0.8]
    mask_pred[0,1,:,:] = [0.7,0.3,0.5,0.5]
    mask_pred[0,:,1,:] = [0,1,0.6,0.7]

    metrics = PointwiseMetric(
        slice={'begin':[0,0,0,0],'size':[-1,-1,-1,2]})
    metrics.update_state(mask_true,mask_pred)
    print('confusion matrix:\n',metrics.result().numpy())

    precision = PrecisionRW(
        slice={'begin':[0,0,0,0],'size':[-1,-1,-1,2]})
    precision.update_state(mask_true,mask_pred)
    print('precision: ',precision.result().numpy())

    recall = RecallRW(
        slice={'begin':[0,0,0,0],'size':[-1,-1,-1,2]})
    recall.update_state(mask_true,mask_pred)
    print('recall: ',recall.result().numpy())    

    f1 = F1ScoreRW(
        slice={'begin':[0,0,0,0],'size':[-1,-1,-1,2]})
    f1.update_state(mask_true,mask_pred)
    print('f1 score: ',f1.result().numpy())

    mcc = MatthewsCoefficientRW(
        slice={'begin':[0,0,0,0],'size':[-1,-1,-1,2]})
    mcc.update_state(mask_true,mask_pred)
    print('mcc: ',mcc.result().numpy())
