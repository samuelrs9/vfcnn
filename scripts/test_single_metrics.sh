#!/bin/bash

# Example script to test accuracy metrics computation for a single prediction
# This helps debug performance issues
# Usage: ./test_single_metrics.sh [--skip-normal]

SKIP_NORMAL=""
if [ "$1" = "--skip-normal" ]; then
    SKIP_NORMAL="--skip-normal"
fi

# Example: Test for fountain simulation with grid_length=0.1
PRED_DIR="/work1/voxel-fluid-net/outputs/pred_sparse_voxelized_fluid_cnn_gl0.1_fountain_3d_big_res_fold0_20260621"
SIM_PATH="/work1/voxel-fluid-net/data/3D/simulations/fountain_3d_big_res"

# Check if directories exist
if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: Prediction directory not found: $PRED_DIR"
    echo ""
    echo "Available predictions:"
    ls -d /work1/voxel-fluid-net/outputs/pred_sparse_voxelized_fluid_cnn_gl* 2>/dev/null | head -5
    exit 1
fi

if [ ! -d "$SIM_PATH" ]; then
    echo "ERROR: Simulation directory not found: $SIM_PATH"
    exit 1
fi

PRED_CONFIG="$PRED_DIR/pred_config_v2.yaml"

echo "=========================================="
echo "Testing Accuracy Metrics Computation"
echo "=========================================="
echo "Prediction: $PRED_DIR"
echo "Simulation: $SIM_PATH"
echo ""

# Check files
if [ ! -f "$PRED_CONFIG" ]; then
    echo "ERROR: pred_config_v2.yaml not found in $PRED_DIR"
    exit 1
fi

echo "Files OK - Starting computation..."
echo ""
echo "TIP: Watch for which step is slow:"
echo "  [1/4] Data reader initialization"
echo "  [2/4] Reports initialization"
echo "  [3/4] Classification metrics (boundary)"
echo "  [4/4] Regression metrics (normal)"
echo ""

# Detect Python command
if command -v conda &> /dev/null; then
    if conda env list | grep -q "^vfnet "; then
        PYTHON_CMD="conda run -n vfnet python"
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python"
fi

# Run with time measurement
time $PYTHON_CMD scripts/compute_accuracy_metrics.py \
    "$SIM_PATH" \
    "$PRED_CONFIG" \
    $SKIP_NORMAL

echo ""
echo "=========================================="
if [ -f "$PRED_DIR/metrics_report.csv" ]; then
    echo "✓ Success! Metrics saved to:"
    echo "  $PRED_DIR/metrics_report.csv"
    echo ""
    echo "Contents:"
    head -5 "$PRED_DIR/metrics_report.csv"
else
    echo "✗ Failed - metrics_report.csv not created"
fi
echo "=========================================="
