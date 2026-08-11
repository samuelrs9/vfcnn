#!/bin/bash

# Script to compute accuracy metrics for all grid_length predictions in parallel
# Usage: ./batch_compute_metrics.sh [--sequential] [--skip-normal]

SEQUENTIAL=false
SKIP_NORMAL=""

for arg in "$@"; do
    if [ "$arg" = "--sequential" ]; then
        SEQUENTIAL=true
    elif [ "$arg" = "--skip-normal" ]; then
        SKIP_NORMAL="--skip-normal"
    fi
done

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

BASE_DIR="/work1/voxel-fluid-net"
SCRIPTS_DIR="$BASE_DIR/scripts"

# Map of simulation names to paths
declare -A SIM_PATHS
SIM_PATHS[fountain_3d_big_res]="/work1/voxel-fluid-net/data/3D/simulations/fountain_3d_big_res"
SIM_PATHS[inlet_vortex_3d_big_res]="/work1/voxel-fluid-net/data/3D/simulations/inlet_vortex_3d_big_res"
SIM_PATHS[db_blocks_3d_big_res]="/work1/voxel-fluid-net/data/3D/simulations/db_blocks_3d_big_res"
SIM_PATHS[inlet_collision_3d_big_res]="/work1/voxel-fluid-net/data/3D/simulations/inlet_collision_3d_big_res"
SIM_PATHS[ddb_3d_big_res]="/work1/voxel-fluid-net/data/3D/simulations/ddb_3d_big_res"

count=0
total=0

echo "Scanning for prediction directories..."
for pred_dir in "$BASE_DIR/outputs"/pred_sparse_voxelized_fluid_cnn_gl*; do
    if [ -d "$pred_dir" ] && [ -f "$pred_dir/pred_config_v2.yaml" ]; then
        total=$((total + 1))
    fi
done

echo "Found $total prediction directories"
echo "Sequential mode: $SEQUENTIAL"
echo ""

for pred_dir in "$BASE_DIR/outputs"/pred_sparse_voxelized_fluid_cnn_gl*; do
    if [ ! -d "$pred_dir" ] || [ ! -f "$pred_dir/pred_config_v2.yaml" ]; then
        continue
    fi
    
    # Extract simulation name from directory name
    dir_name=$(basename "$pred_dir")
    
    # Pattern: pred_sparse_voxelized_fluid_cnn_gl{grid_length}_{sim_name}_fold{fold}_{timestamp}
    sim_name=""
    for sim in "${!SIM_PATHS[@]}"; do
        if [[ "$dir_name" == *"_${sim}_"* ]]; then
            sim_name="$sim"
            break
        fi
    done
    
    if [ -z "$sim_name" ]; then
        echo "WARNING: Could not determine simulation name for $dir_name"
        continue
    fi
    
    sim_path="${SIM_PATHS[$sim_name]}"
    gt_config="$sim_path/gt_config.yaml"
    pred_config="$pred_dir/pred_config_v2.yaml"
    metrics_report="$pred_dir/metrics_report.csv"
    
    count=$((count + 1))
    
    # Skip if metrics already exist
    if [ -f "$metrics_report" ]; then
        echo "[$count/$total] SKIP: Metrics exist for $dir_name"
        continue
    fi
    
    # Check if ground truth exists
    if [ ! -f "$gt_config" ]; then
        echo "[$count/$total] SKIP: No ground truth for $sim_name"
        continue
    fi
    
    echo "[$count/$total] Computing metrics for $dir_name..."
    
    if [ "$SEQUENTIAL" = true ]; then
        # Run sequentially
        $PYTHON_CMD "$SCRIPTS_DIR/compute_accuracy_metrics.py" "$sim_path" "$pred_config" --gt-config "$gt_config" $SKIP_NORMAL 2>&1 | grep -E "ERROR|✓|Simulation"
    else
        # Run in background (parallel)
        $PYTHON_CMD "$SCRIPTS_DIR/compute_accuracy_metrics.py" "$sim_path" "$pred_config" --gt-config "$gt_config" $SKIP_NORMAL > "$pred_dir/metrics_computation.log" 2>&1 &
    fi
done

if [ "$SEQUENTIAL" = false ]; then
    echo ""
    echo "Waiting for all background jobs to complete..."
    wait
    echo "✓ All metrics computation jobs completed!"
    
    # Show summary
    echo ""
    echo "Summary:"
    success=0
    failed=0
    for pred_dir in "$BASE_DIR/outputs"/pred_sparse_voxelized_fluid_cnn_gl*; do
        if [ -f "$pred_dir/metrics_report.csv" ]; then
            success=$((success + 1))
        elif [ -f "$pred_dir/metrics_computation.log" ]; then
            if grep -q "ERROR" "$pred_dir/metrics_computation.log"; then
                failed=$((failed + 1))
                echo "  FAILED: $(basename "$pred_dir")"
            fi
        fi
    done
    echo "  Success: $success"
    echo "  Failed: $failed"
fi

echo ""
echo "Done!"
