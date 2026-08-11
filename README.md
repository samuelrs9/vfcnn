# voxel-fluid-net
Convolutional neural network designed to handle voxelized fluid simulation.

## Quick Setup

For a complete automated setup, simply run:

```bash
bash setup.sh
```

This script will:
1. Install system dependencies (cmake, build-essential, gcc-10, g++-10)
2. Create the `vfnet` conda environment from the provided YAML file
3. Install Python dependencies from `requirements.txt`
4. Build the `tf_kdtree` CUDA library
5. Configure the environment variables

After setup completes, activate the environment with:

```bash
conda activate vfnet
source setup.sh  # To set PYTHONPATH
```

## Configuration-driven scripts

The main training and prediction scripts are driven by YAML files under
`scripts/configs`. Before running a long experiment, use `--dry-run` to inspect
paths, model settings, `train_id`/`predict_id`, and the main parameters.

### Prediction

Use `scripts/predict.py` for direct predictions, without parameter sweeps.

```bash
python3 scripts/predict.py scripts/configs/prediction/regionwise_boundary.yaml
python3 scripts/predict.py scripts/configs/prediction/pointwise_boundary_normal.yaml --dry-run
```

Available configs:

```text
scripts/configs/prediction/
  pointwise_boundary.yaml
  pointwise_normal.yaml
  pointwise_boundary_normal.yaml
  regionwise_boundary.yaml
  regionwise_normal.yaml
  regionwise_boundary_normal.yaml
  sparse_regionwise_fluid_best_checkpoint.yaml
```

Use `scripts/tune_prediction.py` for prediction experiments with sweeps, such
as multiple checkpoints, `hdp`, or decision threshold tuning.

```bash
python3 scripts/tune_prediction.py scripts/configs/prediction_tuning/static_hdp_tuning.yaml
python3 scripts/tune_prediction.py scripts/configs/prediction_tuning/checkpoint_sweep_fluid.yaml --dry-run
```

Available configs:

```text
scripts/configs/prediction_tuning/
  checkpoint_sweep_fluid.yaml
  static_hdp_tuning.yaml
  static_threshold_tuning.yaml
```

### Training

Use `scripts/train.py` to train a model from an already configured dataset.

```bash
python3 scripts/train.py scripts/configs/training/pointwise_boundary.yaml
python3 scripts/train.py scripts/configs/training/sparse_regionwise_boundary_normal.yaml --dry-run
```

Available configs:

```text
scripts/configs/training/
  pointwise_boundary.yaml
  pointwise_normal.yaml
  pointwise_boundary_normal.yaml
  regionwise_boundary.yaml
  regionwise_normal.yaml
  regionwise_boundary_normal.yaml
  sparse_regionwise_boundary_normal.yaml
```

Use `scripts/train_kfold.py` when the experiment needs to mix fold datasets
before training.

```bash
python3 scripts/train_kfold.py scripts/configs/training_kfold/sparse_fluid_kfold.yaml
python3 scripts/train_kfold.py scripts/configs/training_kfold/sparse_static_kfold.yaml --dry-run
```

Available configs:

```text
scripts/configs/training_kfold/
  sparse_fluid_kfold.yaml
  sparse_static_kfold.yaml
```

### Reports

Use `scripts/reports/` for generating classification metrics and time reports.
Each script loads parameters from YAML configuration files.

```bash
python3 scripts/reports/classification_metrics.py scripts/configs/reports/classification_metrics.yaml
python3 scripts/reports/classification_times.py scripts/configs/reports/classification_times.yaml
python3 scripts/reports/accuracy_by_curvatures.py scripts/configs/reports/accuracy_by_curvatures.yaml
python3 scripts/reports/compare_models.py scripts/configs/reports/compare_models.yaml
```

Or use the interactive menu:

```bash
bash scripts/reports/run_reports.sh
```

Available report scripts:

```text
scripts/reports/
  classification_metrics.py      # Tutorial 7.1 - Classification accuracy metrics
  classification_times.py        # Tutorial 7.2 - Classification time reports
  accuracy_by_curvatures.py      # Tutorial 7.4 - Accuracy by curvature intervals
  compare_models.py              # Tutorial 7.31 - Model comparison
  run_reports.sh                 # Interactive menu for running reports
```

See [scripts/reports/README.md](scripts/reports/README.md) for detailed documentation.

### YAML Structure

Configs use absolute paths for `data_dir`, `model_dir`, and
`dataset_config_file`.

The `tasks` field uses the explicit format:

```yaml
tasks:
  boundary:
    type: classification
    labels: [interior, boundary]
    outputs: 2
  normal:
    type: regression
    outputs: 3
```

In training configs, training parameters live inside each `jobs` item:

```yaml
jobs:
  - name: sparse_regionwise_boundary_normal
    training:
      method: custom_train_model_v2
      train_id: v3_4_100000_1.50_0.10_0
      num_epochs: 1000
      learning_rate: 0.0005
      device: cuda
```
