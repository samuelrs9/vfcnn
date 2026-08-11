import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    models_dir = "/work1/Doutorado/data/3D/big/kfold4/sparse_regionwise_approach/models/bests"
    sparse_report_files = [
        f"{models_dir}/model_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_0_dilation_train5/report.npz",
        f"{models_dir}/model_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_1_dilation/report.npz",
        f"{models_dir}/model_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_2_dilation/report.npz",
        f"{models_dir}/model_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_3_dilation/report.npz",
        f"{models_dir}/model_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_4_dilation/report.npz"
    ]
    for i,file in enumerate(sparse_report_files):
      report = dict(np.load(file, allow_pickle=True)['arr_0'].reshape(-1)[0])
      mse = report['metrics']['val']['normal']['mse'][-1]
      cos = report['metrics']['val']['normal']['cos'][-1]
      print(f"Split {i}:\t cos {cos}\t mse {mse}")












