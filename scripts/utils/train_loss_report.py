import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# Dense Network
dense_losses = []
dense_report_files = [
  ("kfold2/regionwise_approach/models/losses/split1-train-loss.csv","kfold2/regionwise_approach/models/losses/split1-val-loss.csv"),
  ("kfold2/regionwise_approach/models/losses/split2-train-loss.csv","kfold2/regionwise_approach/models/losses/split2-val-loss.csv"),
  ("kfold2/regionwise_approach/models/losses/split3-train-loss.csv","kfold2/regionwise_approach/models/losses/split3-val-loss.csv"),
  ("kfold2/regionwise_approach/models/losses/split4-train-loss.csv","kfold2/regionwise_approach/models/losses/split4-val-loss.csv"),  
  ("kfold2/regionwise_approach/models/losses/split5-train-loss.csv","kfold2/regionwise_approach/models/losses/split5-val-loss.csv")
]
for file in dense_report_files:
    report_train = pd.read_csv(file[0])
    report_val = pd.read_csv(file[1])
    dense_losses.append({'train': report_train['Value'].tolist(), 'val': report_val['Value'].tolist()})

# Sparse Network
# Split 1
sparse_losses = []
sparse_report_files = [
  "kfold4/sparse_regionwise_approach/models/model_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_0_dilation_train5/report.npz",
  "kfold4/sparse_regionwise_approach/models/model_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_1_dilation/report.npz",
  "kfold4/sparse_regionwise_approach/models/model_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_2_dilation/report.npz",
  "kfold4/sparse_regionwise_approach/models/model_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_3_dilation/report.npz",
  "kfold4/sparse_regionwise_approach/models/model_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_4_dilation/report.npz"
]
for file in sparse_report_files:
    report = dict(np.load(file, allow_pickle=True)['arr_0'].reshape(-1)[0])
    sparse_losses.append({'train': report['loss']['train']['total'], 'val': report['loss']['val']['total']})


fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 15))
for i in range(5):
    axes[i,0].plot(dense_losses[i]['train'], label='train loss')
    axes[i,0].plot(dense_losses[i]['val'], label='val loss')
    axes[i,0].set_xlim([0,15])
    axes[i,0].set_ylim([0,0.3])
    axes[i,0].set_ylabel(f"Split {i+1}", rotation=0, labelpad=30)
    if i==4:
        axes[i,0].set_xlabel("Epoch")
    axes[i,0].legend()
axes[0,0].set_title("Losses (Dense VFCNN)")
for i in range(5):
    axes[i,1].plot(sparse_losses[i]['train'], label='train loss')
    axes[i,1].plot(sparse_losses[i]['val'], label='val loss')
    axes[i,1].set_xlim([0,30])
    axes[i,1].set_ylim([0,0.3])
    if i==4:
        axes[i,1].set_xlabel("Epoch")
    axes[i,1].legend()
axes[0,1].set_title("Losses (Sparse VFCNN)")



# Adjust the layout to prevent overlap
#plt.tight_layout()
plt.show()










