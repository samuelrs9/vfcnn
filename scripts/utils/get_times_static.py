import glob
from datetime import date
import pandas as pd
import numpy as np

data_dir = 'data/3D/static'
meshes = ['armadillo', 'bunny', 'dragon', 'happy', 'rocker-arm']
other_pred_dir = "other_predictions_hdp=2.0"
frame = 0
all = []
for n in meshes:
  print(n)
  line = [n]
  preds = {
    'vfcnn': glob.glob(f'{data_dir}/{n}/sparse_regionwise_approach/predictions/kfold3_static_hdp=2.0/*/time_report.csv')[0],
    'ss4': glob.glob(f'{data_dir}/{n}/{other_pred_dir}/ss4/pred/summary*.csv')[0],
    'ia4': glob.glob(f'{data_dir}/{n}/{other_pred_dir}/ia4/pred/summary*.csv')[0],
    'hpr': glob.glob(f'{data_dir}/{n}/{other_pred_dir}/hpr/pred/summary*.csv')[0],
    'marrone': glob.glob(f'{data_dir}/{n}/{other_pred_dir}/marrone/pred/summary*.csv')[0]
  }
  for m in preds:
    f = preds[m]
    print(m)
    try:  
      if 'other_predictions' in f:
        times = pd.read_csv(f,header=1, sep=';')
        total_time = times.loc[frame,['timer 1','timer 2','timer 3']].sum()/1000

      else:
        times = pd.read_csv(f)
        total_time = times.loc[frame,'total']
      
      line.append(total_time)

    except Exception as e:
      print("Error:",e)      
      line.append(-1)
    

  all.append(line)


results_df = pd.DataFrame(all, columns=['mesh', 'vfcnn', 'ss4', 'ia4', 'hpr', 'marrone'])
results_df.to_csv(f"/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/times-{date.today().strftime('%Y%m%d')}.csv",sep=';', index=False)
print("\nTimes\n",results_df)
