# Custom Configuration Examples

This file shows examples of how to create your own YAML configurations for different scenarios.

## Important: Absolute Paths

All configuration files now use **absolute paths** for all file references. This makes configurations portable and explicit about data locations.

## 1. Classification Metrics - Multiple Simulations

To compare predictions from different simulations, create a YAML file for each:

### fountain_classification_metrics.yaml
```yaml
sim_config_file: "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/sim_config.ini"
gt_config_file: "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/gt_config.ini"
pred_config_file: "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/sparse_regionwise_approach/predictions/pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_0_kfold3_1/pred_config.ini"
sections: ["boundary", "boundary"]
report:
  plot_metrics: true
  print_metrics: true
```

### toy_dragon_classification_metrics.yaml
```yaml
sim_config_file: "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/toy_dragon_3d_big_res/sim_config.ini"
gt_config_file: "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/toy_dragon_3d_big_res/gt_config.ini"
pred_config_file: "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/toy_dragon_3d_big_res/regionwise_approach/predictions/pred_31_9_4.13_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_0/pred_config.ini"
sections: ["boundary", "boundary"]
report:
  output_path: /home/samuel/Projetos/voxel-fluid-net/data/3D/big/toy_dragon_3d_big_res/regionwise_approach/predictions
  plot_metrics: true
  print_metrics: true
```

## 2. Classification Times - Comparing 4 Models

```yaml
sim_config_file: "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/sim_config.ini"

# Comparing 4 different models
pred_config_files:
  - "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/regionwise_approach/predictions/pred_31_9_4.13_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_0/pred_config.ini"
  - "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/regionwise_approach/predictions/pred_31_9_4.13_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_1/pred_config.ini"
  - "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/regionwise_approach/predictions/pred_25_7_4.17_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_0/pred_config.ini"
  - "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/regionwise_approach/predictions/pred_25_7_4.17_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_1/pred_config.ini"

report:
  output_path: /home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/regionwise_approach/predictions
  format: csv
  plot_times: true
  print_times: true
```

## 3. Accuracy by Curvatures - Different Methods

### CNN Method (Regionwise)
```yaml
sim_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/sim_config.ini"
gt_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/gt_config.ini"
pred_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/regionwise_approach/predictions/pred_25_7_3.12_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_1/pred_config.ini"
sections: ["boundary", "boundary"]
report:
  plot_metrics: true
  print_metrics: true
```

### Marrone Method
```yaml
sim_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/sim_config.ini"
gt_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/gt_config.ini"
pred_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/other_predictions/marrone/pred_config.ini"
sections: ["boundary", "boundary"]
report:
  plot_metrics: true
  print_metrics: true
```

### BPART HPR Method
```yaml
sim_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/sim_config.ini"
gt_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/gt_config.ini"
pred_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/other_predictions/bpart_hpr/pred_config.ini"
sections: ["boundary", "boundary"]
report:
  plot_metrics: true
  print_metrics: true
```

## 4. Compare Models - K-Fold Cross-Validation

Comparing results from different folds:

```yaml
sim_config_file: "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/sim_config.ini"

pred_config_files:
  # Fold 0
  - "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/sparse_regionwise_approach/predictions/pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_0_kfold3_1/pred_config.ini"
  
  # Fold 1
  - "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/sparse_regionwise_approach/predictions/pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_1_kfold3_1/pred_config.ini"
  
  # Fold 2
  - "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/sparse_regionwise_approach/predictions/pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_2_kfold3_1/pred_config.ini"

report:
  output_path: /home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/sparse_regionwise_approach/predictions
  format: csv
  plot_metrics: true
  print_metrics: true
```

## 5. Minimal vs Complete Configuration

### Minimal (uses defaults where possible)
```yaml
sim_config_file: "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/sim_config.ini"
gt_config_file: "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/gt_config.ini"
pred_config_file: "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/regionwise_approach/predictions/pred_25_7_3.12_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_1/pred_config.ini"
```

### Complete (specifies everything)
```yaml
sim_config_file: "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/sim_config.ini"
gt_config_file: "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/gt_config.ini"
pred_config_file: "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/regionwise_approach/predictions/pred_25_7_3.12_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_1/pred_config.ini"
sections: ["boundary", "boundary"]
report:
  output_path: /home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/regionwise_approach/predictions
  plot_metrics: true
  print_metrics: true
  return_metrics: false
```

## Tips

1. **Absolute Paths**: Always use absolute paths for all file references
2. **Multiple Configs**: Create multiple YAML files for different experiments
3. **Comments**: Use `#` to document your configuration choices
4. **Debugging**: Use `print_metrics: true` during development
5. **Visualization**: `plot_metrics: true` generates plots automatically
6. **Portability**: Absolute paths make it clear where data is located

## Running Custom Configs

```bash
# Create your custom config
cp scripts/configs/reports/classification_metrics.yaml my_custom_config.yaml
# Edit as needed
vim my_custom_config.yaml
# Run
python scripts/reports/classification_metrics.py my_custom_config.yaml
```
