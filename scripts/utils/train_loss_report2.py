import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# Sparse Network
# Split 1
sparse_losses = []
sparse_report_files = glob.glob("/home/samuel/Doutorado/voxel-fluid-net/data/3D/big/kfold3_hdp=1.73/sparse_regionwise_approach/models/*/report.npz")
for file in sparse_report_files:
    report = dict(np.load(file, allow_pickle=True)['arr_0'].reshape(-1)[0])
    sparse_losses.append({'train': report['loss']['train']['total'], 'val': report['loss']['val']['total']})


fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 15))
for i in range(5):
    axes[i].plot(sparse_losses[i]['train'], label='train loss')
    axes[i].plot(sparse_losses[i]['val'], label='val loss')
    axes[i].set_xlim([0,30])
    axes[i].set_ylim([0,0.3])
    axes[i].set_ylabel(f"Split {i+1}", rotation=0, labelpad=30)
    if i==4:
        axes[i].set_xlabel("Epoch")
    axes[i].legend()
axes[0].set_title("Losses (Sparse VFCNN)")

# Adjust the layout to prevent overlap
#plt.tight_layout()
plt.show()










