import tensorflow as tf
from custom_metrics_keras import *

tf.config.run_functions_eagerly(True)

labels_true = [[0,1],[1,0],[1,0],[0,0]]
labels_pred = [[0,1],[1,0],[0,1],[1,0]]

tp = tf.keras.metrics.TruePositives()
tp.update_state(labels_true,labels_pred)
print(tp.result())

f1 = F1Score()
f1.update_state(labels_true,labels_pred)
print('f1 score: ',f1.result())

f1_mask = F1ScoreMask()
f1_mask.update_state(labels_true,labels_pred)
print('f1 score mask: ',f1_mask.result())

mcc_mask = MatthewsCoefficientMask()
mcc_mask.update_state(labels_true,labels_pred)
print('mcc mask: ',mcc_mask.result())

