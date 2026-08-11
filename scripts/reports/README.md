# Report Scripts - Tutorials 7.x

This directory contains specific scripts to generate classification metrics and time reports, extracted from tutorials 7.x of the `main_tutorials.py` file.

## Structure

```
scripts/reports/
├── classification_metrics.py      # Tutorial 7.1
├── classification_times.py        # Tutorial 7.2
├── accuracy_by_curvatures.py      # Tutorial 7.4
├── compare_models.py              # Tutorial 7.31
└── README.md

scripts/configs/reports/
├── classification_metrics.yaml
├── classification_times.yaml
├── accuracy_by_curvatures.yaml
└── compare_models.yaml
```

## Available Scripts

### 1. Classification Metrics (Tutorial 7.1)

Generates classification report with accuracy metrics.

**Script:** `classification_metrics.py`

**Usage:**
```bash
python scripts/reports/classification_metrics.py scripts/configs/reports/classification_metrics.yaml
```

**Main parameters:**
- `sim_config_file`: Simulation configuration file (absolute path)
- `gt_config_file`: Ground-truth configuration file (absolute path)
- `pred_config_file`: Prediction configuration file (absolute path)
- `sections`: Sections to be compared (gt_section, pred_section)
- `plot_metrics`: Whether to plot metrics
- `print_metrics`: Whether to print metrics to console

**Configuration example:**
```yaml
sim_config_file: "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/sim_config.yaml"
gt_config_file: "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/gt_config.yaml"
pred_config_file: "/home/samuel/Projetos/voxel-fluid-net/data/3D/big/fountain_3d_big_res/sparse_regionwise_approach/predictions/pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_0_kfold3_1/pred_config.yaml"
sections:
  - "boundary"
  - "boundary"
report:
  plot_metrics: false
  print_metrics: true
```

---

### 2. Classification Times (Tutorial 7.2)

Generates classification time report, allowing comparison of multiple models.

**Script:** `classification_times.py`

**Usage:**
```bash
python scripts/reports/classification_times.py scripts/configs/reports/classification_times.yaml
```

**Main parameters:**
- `sim_config_file`: Simulation configuration file (absolute path)
- `pred_config_files`: List of prediction config files (absolute paths)
- `output_dir`: Output directory for the report (optional)
- `extension`: Output format (csv, etc.)
- `plot_times`: Whether to plot times
- `print_times`: Whether to print times to console

**Configuration example:**
```yaml
sim_config_file: "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/sim_config.yaml"
pred_config_files:
  - "/home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/regionwise_approach/predictions/pred_25_7_3.12_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_1/pred_config.yaml"
report:
  output_path: /home/samuel/Projetos/data/paper/ToyDragonBox_ressampled/regionwise_approach/predictions
  format: csv
  plot_times: true
```

---

### 3. Accuracy by Curvatures (Tutorial 7.4)

Accuracy analysis by curvature intervals.

**Script:** `accuracy_by_curvatures.py`

**Usage:**
```bash
python scripts/reports/accuracy_by_curvatures.py scripts/configs/reports/accuracy_by_curvatures.yaml
```

**Main parameters:**
- `sim_config_file`: Simulation configuration file (absolute path)
- `gt_config_file`: Ground-truth configuration file (absolute path)
- `pred_config_file`: Prediction configuration file (absolute path)
- `sections`: Sections to be compared
- `plot_metrics`: Whether to plot metrics
- `print_metrics`: Whether to print metrics to console

**Configuration example:**
```yaml
sim_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/sim_config.yaml"
gt_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/gt_config.yaml"
pred_config_file: "/home/samuel/Projetos/data/paper/fluid_injectors_ressampled/regionwise_approach/predictions/pred_25_7_3.12_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_1/pred_config.yaml"
sections:
  - "boundary"
  - "boundary"
report:
  plot_metrics: true
```

---

### 4. Compare Models (Tutorial 7.31)

Generates comparison report between different models.

**Script:** `compare_models.py`

**Usage:**
```bash
python scripts/reports/compare_models.py scripts/configs/reports/compare_models.yaml
```

**Main parameters:**
- `sim_config_file`: Simulation configuration file (absolute path)
- `pred_config_files`: List of prediction config files to compare (absolute paths)
- `output_dir`: Output directory (optional)
- `extension`: Output format
- `plot_metrics`: Whether to plot metrics
- `print_metrics`: Whether to print metrics to console

**Configuration example:**
```yaml
sim_config_file: "/media/samuel/Meus Arquivos/Doc/Projetos/data/dambreak3d.0_ressampled/sim_config.yaml"
pred_config_files:
  - "/media/samuel/Meus Arquivos/Doc/Projetos/data/dambreak3d.0_ressampled/regionwise_approach/predictions/pred_31_9_4.13_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_0/pred_config.yaml"
  - "/media/samuel/Meus Arquivos/Doc/Projetos/data/dambreak3d.0_ressampled/regionwise_approach/predictions/pred_25_7_4.17_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN_0/pred_config.yaml"
report:
  output_path: /media/samuel/Meus Arquivos/Doc/Projetos/data/dambreak3d.0_ressampled/regionwise_approach/predictions
  format: csv
  plot_metrics: true
```

---

## How to Use

### Prerequisites

1. Activate conda environment:
```bash
conda activate vfnet
```

2. Make sure all required modules are installed:
   - `vfnet.report`
   - `sim_reader.data`
   - `yaml`

### Running a Script

1. Edit the corresponding YAML configuration file in `scripts/configs/reports/`
2. Adjust paths and parameters as needed
3. Run the script:
```bash
python scripts/reports/<script_name>.py scripts/configs/reports/<corresponding_config>.yaml
```

### Customizing Configurations

Each YAML file in `scripts/configs/reports/` contains:
- Comments explaining each parameter
- Examples of alternative configurations
- Default values used in the original tutorials

You can:
- Create multiple YAML files for different experiments
- Modify paths to your data
- Adjust visualization and output parameters

## Notes

- All paths in configuration files should be **absolute paths**
- For single prediction: use a single string path
- For multiple predictions: use a list of string paths
- Reports can be saved in different formats (csv, png for plots, etc.)
- Output directories are optional; if not specified, results are displayed only

## Reference

These scripts were extracted and adapted from tutorials 7.x of the `main_tutorials.py` file, converting hard-coded parameters into flexible YAML-based configurations with absolute paths.
