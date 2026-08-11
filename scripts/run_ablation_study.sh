#!/bin/bash

# Script to run ablation study on sparse regionwise prediction tasks
# Tests boundary-only, normal-only, and boundary+normal predictions
# across 5 different simulations

set -e  # Exit on error

# Detect Python command (prefer conda environment)
if command -v conda &> /dev/null; then
    # Check if vfnet environment exists
    if conda env list | grep -q "^vfnet "; then
        PYTHON_CMD="conda run -n vfnet python"
    elif conda env list | grep -q "^vfnet_tf22_cuda101 "; then
        PYTHON_CMD="conda run -n vfnet_tf22_cuda101 python"
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python"
fi

# Base directory
BASE_DIR="/work1/voxel-fluid-net"
SCRIPTS_DIR="$BASE_DIR/scripts"
CONFIG_DIR="$SCRIPTS_DIR/configs/prediction"
OUTPUT_DIR="$BASE_DIR/outputs/ablation_study_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/ablation_study.log"
TIMING_CSV="$OUTPUT_DIR/timing_results.csv"

# Initialize timing results CSV
echo "simulation,task,model_fold,start_time,end_time,duration_seconds,predict_id,num_particles,frames_processed,avg_gpu_util,max_gpu_util,avg_gpu_mem,max_gpu_mem" > "$TIMING_CSV"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to generate config from template
generate_config() {
    local template=$1
    local sim_name=$2
    local sim_path=$3
    local model_path=$4
    local fold=$5
    local task_type=$6
    local output_config=$7
    local frame_range=$8
    local skip_steps=$9
    
    local predict_id="${task_type}_${sim_name}_fold${fold}_$(date +%Y%m%d)"
    
    # Extract initial and final steps from frame_range (format: "initial:final")
    local initial_step=$(echo "$frame_range" | cut -d':' -f1)
    local final_step=$(echo "$frame_range" | cut -d':' -f2)
    
    # Replace placeholders in template
    sed -e "s|PLACEHOLDER_NAME|${task_type}_${sim_name}_fold${fold}|g" \
        -e "s|PLACEHOLDER_SIM_PATH|${sim_path}|g" \
        -e "s|PLACEHOLDER_MODEL_PATH|${model_path}|g" \
        -e "s|PLACEHOLDER_PREDICT_ID|${predict_id}|g" \
        -e "s|PLACEHOLDER_INITIAL_STEP|${initial_step}|g" \
        -e "s|PLACEHOLDER_FINAL_STEP|${final_step}|g" \
        -e "s|PLACEHOLDER_SKIP_STEPS|${skip_steps}|g" \
        "$template" > "$output_config"
    
    echo "$predict_id"
}

# Function to run prediction with timing
run_prediction() {
    local config_file=$1
    local sim_name=$2
    local task_type=$3
    local fold=$4
    local predict_id=$5
    local sim_path=$6
    local frame_range=$7
    local skip_steps=$8
    
    log_message "Starting prediction: $sim_name - $task_type (fold $fold)"
    
    # Count particles in the frame range
    local initial_step=$(echo "$frame_range" | cut -d':' -f1)
    local final_step=$(echo "$frame_range" | cut -d':' -f2)
    local sim_config="$sim_path/sim_config.yaml"
    
    local particle_info=$($PYTHON_CMD "$SCRIPTS_DIR/count_particles.py" "$sim_config" "$initial_step" "$final_step" "$skip_steps" 2>&1)
    local particle_exit_code=$?
    
    if [ $particle_exit_code -eq 0 ]; then
        local total_particles=$(echo "$particle_info" | tail -n 1 | cut -d',' -f1)
        local num_frames=$(echo "$particle_info" | tail -n 1 | cut -d',' -f2)
    else
        log_message "WARNING: Could not count particles (exit code: $particle_exit_code)"
        log_message "Error output: $particle_info"
        total_particles=0
        num_frames=0
    fi
    
    if [ -z "$total_particles" ] || [ "$total_particles" = "0" ]; then
        log_message "WARNING: Particle count is 0, check simulation data"
        total_particles=0
        num_frames=0
    fi
    
    log_message "Processing $num_frames frames with total $total_particles particles"
    
    local start_time=$(date +%s)
    local start_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Start GPU monitoring in background
    local gpu_log="$OUTPUT_DIR/gpu_${sim_name}_${task_type}.log"
    > "$gpu_log"  # Clear file
    
    # Monitor GPU usage every 0.5 seconds in background
    (while true; do 
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0,0"
        sleep 0.5
    done) > "$gpu_log" &
    local monitor_pid=$!
    
    # Run prediction
    if $PYTHON_CMD "$SCRIPTS_DIR/predict.py" "$config_file" >> "$LOG_FILE" 2>&1; then
        # Stop GPU monitoring
        kill $monitor_pid 2>/dev/null
        wait $monitor_pid 2>/dev/null
        
        local end_time=$(date +%s)
        local end_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local duration=$((end_time - start_time))
        
        # Calculate GPU stats from log
        local gpu_stats="0.0,0.0,0.0,0.0"
        if [ -f "$gpu_log" ] && [ -s "$gpu_log" ]; then
            # Calculate average and max GPU utilization and memory
            gpu_stats=$(awk -F',' '{
                util+=$1; mem+=$2; count++;
                if ($1 > max_util) max_util=$1;
                if ($2 > max_mem) max_mem=$2;
            } END {
                if (count > 0) {
                    printf "%.2f,%.2f,%.2f,%.2f", util/count, max_util, mem/count, max_mem
                } else {
                    print "0.0,0.0,0.0,0.0"
                }
            }' "$gpu_log")
        fi
        rm -f "$gpu_log"
        
        log_message "Completed prediction: $sim_name - $task_type (fold $fold) in ${duration}s"
        log_message "GPU stats: $gpu_stats"
        
        # Record timing with particle count and GPU stats
        echo "$sim_name,$task_type,$fold,$start_timestamp,$end_timestamp,$duration,$predict_id,$total_particles,$num_frames,$gpu_stats" >> "$TIMING_CSV"
        
        return 0
    else
        # Stop GPU monitoring
        kill $monitor_pid 2>/dev/null
        wait $monitor_pid 2>/dev/null
        rm -f "$gpu_log"
        
        log_message "ERROR: Failed prediction: $sim_name - $task_type (fold $fold)"
        return 1
    fi
}

# Define simulations and models
declare -A SIMULATIONS
SIMULATIONS[fountain_3d_big_res]="/work1/voxel-fluid-net/data/3D/simulations/fountain_3d_big_res"
SIMULATIONS[inlet_vortex_3d_big_res]="/work1/voxel-fluid-net/data/3D/simulations/inlet_vortex_3d_big_res"
SIMULATIONS[db_blocks_3d_big_res]="/work1/voxel-fluid-net/data/3D/simulations/db_blocks_3d_big_res"
SIMULATIONS[inlet_collision_3d_big_res]="/work1/voxel-fluid-net/data/3D/simulations/inlet_collision_3d_big_res"
SIMULATIONS[ddb_3d_big_res]="/work1/voxel-fluid-net/data/3D/simulations/ddb_3d_big_res"

declare -A MODEL_PATHS
MODEL_PATHS[fountain_3d_big_res]="/work1/voxel-fluid-net/models/kfold3/sparse_regionwise_approach/models/model_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_0"
MODEL_PATHS[inlet_vortex_3d_big_res]="/work1/voxel-fluid-net/models/kfold3/sparse_regionwise_approach/models/model_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_1"
MODEL_PATHS[db_blocks_3d_big_res]="/work1/voxel-fluid-net/models/kfold3/sparse_regionwise_approach/models/model_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_2"
MODEL_PATHS[inlet_collision_3d_big_res]="/work1/voxel-fluid-net/models/kfold3/sparse_regionwise_approach/models/model_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_3"
MODEL_PATHS[ddb_3d_big_res]="/work1/voxel-fluid-net/models/kfold3/sparse_regionwise_approach/models/model_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_4"

declare -A FOLD_NUMBERS
FOLD_NUMBERS[fountain_3d_big_res]=0
FOLD_NUMBERS[inlet_vortex_3d_big_res]=1
FOLD_NUMBERS[db_blocks_3d_big_res]=2
FOLD_NUMBERS[inlet_collision_3d_big_res]=3
FOLD_NUMBERS[ddb_3d_big_res]=4

# Frame ranges for each simulation (initial_step:final_step)
# Use -1 for final_step to process until the last frame of the simulation
declare -A FRAME_RANGES
FRAME_RANGES[fountain_3d_big_res]="301:-1"  # fountain starts at frame 301, testing 10 frames
FRAME_RANGES[inlet_vortex_3d_big_res]="0:-1"  # testing first 10 frames
FRAME_RANGES[db_blocks_3d_big_res]="0:-1"     # testing first 10 frames
FRAME_RANGES[inlet_collision_3d_big_res]="0:-1"  # testing first 10 frames
FRAME_RANGES[ddb_3d_big_res]="0:-1"           # testing first 10 frames

# Example: To process ALL frames of a simulation, use -1 as final_step:
# FRAME_RANGES[fountain_3d_big_res]="301:-1"  # from frame 301 to last frame
# FRAME_RANGES[inlet_vortex_3d_big_res]="0:-1"  # from frame 0 to last frame

# Skip steps for each simulation (how many frames to skip between predictions)
declare -A SKIP_STEPS
SKIP_STEPS[fountain_3d_big_res]=10
SKIP_STEPS[inlet_vortex_3d_big_res]=10
SKIP_STEPS[db_blocks_3d_big_res]=10
SKIP_STEPS[inlet_collision_3d_big_res]=10
SKIP_STEPS[ddb_3d_big_res]=10

# Task types and their templates
declare -A TASK_TEMPLATES
TASK_TEMPLATES[boundary_only]="$CONFIG_DIR/ablation_boundary_only_template.yaml"
TASK_TEMPLATES[normal_only]="$CONFIG_DIR/ablation_normal_only_template.yaml"
TASK_TEMPLATES[boundary_normal]="$CONFIG_DIR/ablation_boundary_normal_template.yaml"

log_message "==================================================="
log_message "Starting Ablation Study"
log_message "Python command: $PYTHON_CMD"
log_message "Output directory: $OUTPUT_DIR"
log_message "==================================================="

# Counter for progress
total_runs=$((${#SIMULATIONS[@]} * ${#TASK_TEMPLATES[@]}))
current_run=0

# Main loop: iterate over simulations and tasks
for sim_name in "${!SIMULATIONS[@]}"; do
    sim_path="${SIMULATIONS[$sim_name]}"
    model_path="${MODEL_PATHS[$sim_name]}"
    fold="${FOLD_NUMBERS[$sim_name]}"
    frame_range="${FRAME_RANGES[$sim_name]}"
    skip_steps="${SKIP_STEPS[$sim_name]}"
    
    log_message ""
    log_message "Processing simulation: $sim_name (fold $fold) - Frames: $frame_range - Skip: $skip_steps"
    
    for task_type in "${!TASK_TEMPLATES[@]}"; do
        current_run=$((current_run + 1))
        
        log_message "[$current_run/$total_runs] Task: $task_type"
        
        template="${TASK_TEMPLATES[$task_type]}"
        config_file="$OUTPUT_DIR/config_${sim_name}_${task_type}.yaml"
        
        # Generate config file
        predict_id=$(generate_config "$template" "$sim_name" "$sim_path" "$model_path" "$fold" "$task_type" "$config_file" "$frame_range" "$skip_steps")
        
        # Run prediction with timing
        run_prediction "$config_file" "$sim_name" "$task_type" "$fold" "$predict_id" "$sim_path" "$frame_range" "$skip_steps" || log_message "WARNING: Continuing after error"
    done
done

log_message ""
log_message "==================================================="
log_message "Ablation Study Completed"
log_message "Results saved to: $TIMING_CSV"
log_message "Log file: $LOG_FILE"
log_message "==================================================="

# Display summary
log_message ""
log_message "Timing Summary:"
column -t -s ',' "$TIMING_CSV" | tee -a "$LOG_FILE"
