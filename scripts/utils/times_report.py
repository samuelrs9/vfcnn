import pandas as pd

# times = {
#   'inlet_vortex': {
#     "dvfcnn": "data/3D/big/inlet_vortex_3d_big_res/regionwise_approach/predictions/kfold2/pred_31_9_3.10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_74/time_report.csv",
#     "svfcnn_dil": "data/3D/big/inlet_vortex_3d_big_res/sparse_regionwise_approach/predictions/pred_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_1_kfold4_dilation/time_report.csv",
#     #"svfcnn": "data/3D/big/inlet_vortex_3d_big_res/sparse_regionwise_approach/predictions/kfold3_hdp=1.73/pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.1_0_1_kfold3_no_coarse/time_report.csv",
#     "svfcnn": "/work1/Doutorado/data/3D/big/inlet_vortex_3d_big_res/sparse_regionwise_approach/predictions_20240907/kfold3_hdp=1.73_checkpoint_45/pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.1_0_1_kfold3_no_coarse/time_report.csv"
#   }
# }
#times = times['inlet_vortex']

reports = {
  "ddb": "data/3D/big/ddb_3d_big_res/sparse_regionwise_approach/predictions/kfold4/pred_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_4_kfold4_dilation/time_report.csv",
  "inlet_collision": "data/3D/big/inlet_collision_3d_big_res/sparse_regionwise_approach/predictions/pred_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_3_kfold4_dilation/time_report.csv",
  "blocks_dambreak": "data/3D/big/db_blocks_3d_big_res/sparse_regionwise_approach/predictions/pred_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_2_kfold4_dilation/time_report.csv",
  "inlet_vortex": "data/3D/big/inlet_vortex_3d_big_res/sparse_regionwise_approach/predictions/pred_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_1_kfold4_dilation/time_report.csv",
  "fountain": "data/3D/big/fountain_3d_big_res/sparse_regionwise_approach/predictions/pred_sparse_voxelized_fluid_cnn_100000_1.50_0.10_1_0_kfold4_dilation/time_report.csv"
}

all_times = []
for m in reports:
  report = pd.read_csv(reports[m])
  times = report['total']
  min_time = times.min()
  max_time = times.max()
  avg_time = 1000000*times.sum()/report['num_particles'].sum()
  all_times.append([m, min_time, max_time, avg_time])

df = pd.DataFrame(all_times, columns=['methods','min_time', 'max_time', 'avg_time'])
df.to_csv('report_times_svfcnn_dil.csv',index=False)
print(df)
