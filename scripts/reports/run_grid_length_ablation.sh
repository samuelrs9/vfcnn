#!/bin/bash

# Script to run ablation study on grid_length parameter
# Tests different grid_length values: 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2
# across 5 different simulations
#
# Usage: ./run_grid_length_ablation.sh [--force-recompute] [--compute-metrics]
#   --force-recompute: Force recomputation even if predictions exist
#   --compute-metrics: Compute accuracy metrics (slower, disabled by default)

set -e  # Exit on error

# Parse command line arguments
FORCE_RECOMPUTE=false
SKIP_METRICS=true  # Skip metrics by default for speed
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-recompute)
            FORCE_RECOMPUTE=true
            shift
            ;;
        --compute-metrics)
            SKIP_METRICS=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--force-recompute] [--compute-metrics]"
            exit 1
            ;;
    esac
done

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
OUTPUT_DIR="$BASE_DIR/outputs/ablation_grid_length_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/ablation_study.log"
TIMING_CSV="$OUTPUT_DIR/timing_results.csv"

# Initialize timing results CSV
echo "simulation,grid_length,model_fold,start_time,end_time,duration_seconds,predict_id,num_particles,frames_processed,avg_gpu_util,max_gpu_util,avg_gpu_mem,max_gpu_mem" > "$TIMING_CSV"

# Metrics CSV
METRICS_CSV="$OUTPUT_DIR/accuracy_metrics.csv"
echo "simulation,grid_length,model_fold,predict_id,avg_recall,avg_precision,avg_tnr,avg_f1,avg_mcc,avg_combined_metric" > "$METRICS_CSV"

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
    local grid_length=$6
    local output_config=$7
    local frame_range=$8
    local skip_steps=$9
    
    local predict_id="gl${grid_length}_${sim_name}_fold${fold}_$(date +%Y%m%d)"
    
    # Extract initial and final steps from frame_range (format: "initial:final")
    local initial_step=$(echo "$frame_range" | cut -d':' -f1)
    local final_step=$(echo "$frame_range" | cut -d':' -f2)
    
    # Replace placeholders in template
    sed -e "s|PLACEHOLDER_NAME|gl${grid_length}_${sim_name}_fold${fold}|g" \
        -e "s|PLACEHOLDER_SIM_PATH|${sim_path}|g" \
        -e "s|PLACEHOLDER_MODEL_PATH|${model_path}|g" \
        -e "s|PLACEHOLDER_PREDICT_ID|${predict_id}|g" \
        -e "s|PLACEHOLDER_INITIAL_STEP|${initial_step}|g" \
        -e "s|PLACEHOLDER_FINAL_STEP|${final_step}|g" \
        -e "s|PLACEHOLDER_SKIP_STEPS|${skip_steps}|g" \
        -e "s|PLACEHOLDER_GRID_LENGTH|${grid_length}|g" \
        "$template" > "$output_config"
    
    echo "$predict_id"
}

# Function to run prediction with timing
run_prediction() {
    local config_file=$1
    local sim_name=$2
    local grid_length=$3
    local fold=$4
    local predict_id=$5
    local sim_path=$6
    local frame_range=$7
    local skip_steps=$8
    
    # Check if prediction already exists (cache)
    # Search for existing predictions in outputs directory (ignore timestamp)
    local outputs_dir="$BASE_DIR/outputs"
    local pred_pattern="pred_sparse_voxelized_fluid_cnn_gl${grid_length}_${sim_name}_fold${fold}_*"
    local pred_dir=""
    local cached_predict_id=""
    
    # Find existing prediction directory
    if [ "$FORCE_RECOMPUTE" = false ] && [ -d "$outputs_dir" ]; then
        for dir in "$outputs_dir"/$pred_pattern; do
            if [ -d "$dir" ] && [ -f "$dir/pred_config_v2.yaml" ]; then
                pred_dir="$dir"
                cached_predict_id=$(basename "$dir" | sed 's/pred_sparse_voxelized_fluid_cnn_//')
                break
            fi
        done
    fi
    
    if [ -n "$pred_dir" ] && [ -f "$pred_dir/pred_config_v2.yaml" ]; then
        log_message "CACHE HIT: Prediction already exists: $sim_name - grid_length=$grid_length (fold $fold)"
        log_message "Using cached prediction from: $pred_dir"
        log_message "Cached ID: $cached_predict_id"
        
        # Use cached predict_id for consistency
        local predict_id="$cached_predict_id"
        local pred_config="$pred_dir/pred_config_v2.yaml"
        local time_report="$pred_dir/time_report.csv"
        
        # Count particles for consistency
        local initial_step=$(echo "$frame_range" | cut -d':' -f1)
        local final_step=$(echo "$frame_range" | cut -d':' -f2)
        local sim_config="$sim_path/sim_config.yaml"
        
        local particle_info=$($PYTHON_CMD "$SCRIPTS_DIR/count_particles.py" "$sim_config" "$initial_step" "$final_step" "$skip_steps" 2>&1)
        local particle_exit_code=$?
        
        if [ $particle_exit_code -eq 0 ]; then
            local total_particles=$(echo "$particle_info" | tail -n 1 | cut -d',' -f1)
            local num_frames=$(echo "$particle_info" | tail -n 1 | cut -d',' -f2)
        else
            total_particles=0
            num_frames=0
        fi
        
        # Try to extract timing from existing time_report.csv
        local duration=0
        local start_timestamp="N/A"
        local end_timestamp="N/A"
        local gpu_stats="0.0,0.0,0.0,0.0"
        
        if [ -f "$time_report" ]; then
            # Extract total time from time_report.csv (last column is usually total time)
            duration=$(tail -n 1 "$time_report" | awk -F',' '{print $NF}' | grep -o '[0-9.]*' | head -1)
            if [ -z "$duration" ]; then
                duration=0
            fi
            log_message "Extracted duration from time_report: ${duration}s"
        else
            log_message "WARNING: time_report.csv not found, setting duration to 0"
        fi
        
        # Record timing (with N/A timestamps since we're using cached results)
        echo "$sim_name,$grid_length,$fold,$start_timestamp,$end_timestamp,$duration,$predict_id,$total_particles,$num_frames,$gpu_stats" >> "$TIMING_CSV"
        
        # Compute accuracy metrics (if gt_config.yaml exists and not skipped)
        local gt_config="$sim_path/gt_config.yaml"
        
        if [ "$SKIP_METRICS" = true ]; then
            log_message "Skipping accuracy metrics (--skip-metrics enabled)"
            echo "$sim_name,$grid_length,$fold,$predict_id,0,0,0,0,0,0" >> "$METRICS_CSV"
        elif [ -f "$gt_config" ] && [ -f "$pred_config" ]; then
            local metrics_report="$pred_dir/metrics_report.csv"
            
            # Check if metrics already exist
            if [ -f "$metrics_report" ]; then
                log_message "CACHE HIT: Metrics already exist"
                
                # Extract metrics from existing report
                local metrics_summary=$(awk -F',' 'NR>1 {
                    particles+=$2; 
                    rec+=$2*$3; pre+=$2*$4; tnr+=$2*$5; 
                    mc+=$2*$6; f1+=$2*$7; mcc+=$2*$8
                } END {
                    if (particles > 0) {
                        printf "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f", 
                            rec/particles, pre/particles, tnr/particles, 
                            f1/particles, mcc/particles, mc/particles
                    } else {
                        print "0,0,0,0,0,0"
                    }
                }' "$metrics_report")
                
                echo "$sim_name,$grid_length,$fold,$predict_id,$metrics_summary" >> "$METRICS_CSV"
                log_message "Metrics: $metrics_summary"
            else
                # Compute metrics if they don't exist
                log_message "Computing accuracy metrics (not cached)... This may take a while."
                
                # Run with timeout of 120 seconds (skip normal metrics for speed)
                # Note: Not passing --gt-config to allow automatic selection of gt_config_hdp=1.73.yaml
                if timeout 120s $PYTHON_CMD "$SCRIPTS_DIR/compute_accuracy_metrics.py" "$sim_path" "$pred_config" --skip-normal >> "$LOG_FILE" 2>&1; then
                    log_message "Accuracy metrics computed successfully"
                    
                    if [ -f "$metrics_report" ]; then
                        local metrics_summary=$(awk -F',' 'NR>1 {
                            particles+=$2; 
                            rec+=$2*$3; pre+=$2*$4; tnr+=$2*$5; 
                            mc+=$2*$6; f1+=$2*$7; mcc+=$2*$8
                        } END {
                            if (particles > 0) {
                                printf "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f", 
                                    rec/particles, pre/particles, tnr/particles, 
                                    f1/particles, mcc/particles, mc/particles
                            } else {
                                print "0,0,0,0,0,0"
                            }
                        }' "$metrics_report")
                        
                        echo "$sim_name,$grid_length,$fold,$predict_id,$metrics_summary" >> "$METRICS_CSV"
                        log_message "Metrics: $metrics_summary"
                    else
                        echo "$sim_name,$grid_length,$fold,$predict_id,0,0,0,0,0,0" >> "$METRICS_CSV"
                    fi
                else
                    exit_code=$?
                    if [ $exit_code -eq 124 ]; then
                        log_message "WARNING: Accuracy metrics computation timed out (>120s)"
                    else
                        log_message "WARNING: Could not compute accuracy metrics (exit code: $exit_code)"
                    fi
                    echo "$sim_name,$grid_length,$fold,$predict_id,0,0,0,0,0,0" >> "$METRICS_CSV"
                fi
            fi
        else
            echo "$sim_name,$grid_length,$fold,$predict_id,0,0,0,0,0,0" >> "$METRICS_CSV"
        fi
        
        return 0
    fi
    
    # If prediction doesn't exist, compute it
    log_message "CACHE MISS: Computing new prediction: $sim_name - grid_length=$grid_length (fold $fold)"
    
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
    local gpu_log="$OUTPUT_DIR/gpu_${sim_name}_gl${grid_length}.log"
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
        
        log_message "Completed prediction: $sim_name - grid_length=$grid_length (fold $fold) in ${duration}s"
        log_message "GPU stats: $gpu_stats"
        
        # Record timing with particle count and GPU stats
        echo "$sim_name,$grid_length,$fold,$start_timestamp,$end_timestamp,$duration,$predict_id,$total_particles,$num_frames,$gpu_stats" >> "$TIMING_CSV"
        
        # Compute accuracy metrics (if gt_config.yaml exists)
        local gt_config="$sim_path/gt_config.yaml"
        local pred_dir="$sim_path/pred_sparse_voxelized_fluid_cnn_v3_4_100000_1.50_0.10_0_${fold}/${predict_id}"
        local pred_config="$pred_dir/pred_config_v2.yaml"
        
        if [ -f "$gt_config" ] && [ -f "$pred_config" ]; then
            log_message "Computing accuracy metrics..."
            
            if $PYTHON_CMD "$SCRIPTS_DIR/compute_accuracy_metrics.py" "$sim_path" "$pred_config" --gt-config "$gt_config" >> "$LOG_FILE" 2>&1; then
                log_message "Accuracy metrics computed successfully"
                
                # Extract metrics from the generated metrics_report.csv
                local metrics_report="$pred_dir/metrics_report.csv"
                if [ -f "$metrics_report" ]; then
                    # Calculate weighted average metrics using awk
                    local metrics_summary=$(awk -F',' 'NR>1 {
                        particles+=$2; 
                        rec+=$2*$3; pre+=$2*$4; tnr+=$2*$5; 
                        mc+=$2*$6; f1+=$2*$7; mcc+=$2*$8
                    } END {
                        if (particles > 0) {
                            printf "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f", 
                                rec/particles, pre/particles, tnr/particles, 
                                f1/particles, mcc/particles, mc/particles
                        } else {
                            print "0,0,0,0,0,0"
                        }
                    }' "$metrics_report")
                    
                    echo "$sim_name,$grid_length,$fold,$predict_id,$metrics_summary" >> "$METRICS_CSV"
                    log_message "Metrics: $metrics_summary"
                else
                    log_message "WARNING: Metrics report not found at $metrics_report"
                    echo "$sim_name,$grid_length,$fold,$predict_id,0,0,0,0,0,0" >> "$METRICS_CSV"
                fi
            else
                log_message "WARNING: Could not compute accuracy metrics"
                echo "$sim_name,$grid_length,$fold,$predict_id,0,0,0,0,0,0" >> "$METRICS_CSV"
            fi
        else
            log_message "WARNING: Ground truth config not found or prediction config missing, skipping metrics"
            echo "$sim_name,$grid_length,$fold,$predict_id,0,0,0,0,0,0" >> "$METRICS_CSV"
        fi
        
        return 0
    else
        # Stop GPU monitoring
        kill $monitor_pid 2>/dev/null
        wait $monitor_pid 2>/dev/null
        rm -f "$gpu_log"
        
        log_message "ERROR: Failed prediction: $sim_name - grid_length=$grid_length (fold $fold)"
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
declare -A FRAME_RANGES
FRAME_RANGES[fountain_3d_big_res]="301:302"
FRAME_RANGES[inlet_vortex_3d_big_res]="199:200"
FRAME_RANGES[db_blocks_3d_big_res]="199:200"
FRAME_RANGES[inlet_collision_3d_big_res]="199:200"
FRAME_RANGES[ddb_3d_big_res]="199:200"

# Skip steps for each simulation
declare -A SKIP_STEPS
SKIP_STEPS[fountain_3d_big_res]=1
SKIP_STEPS[inlet_vortex_3d_big_res]=1
SKIP_STEPS[db_blocks_3d_big_res]=1
SKIP_STEPS[inlet_collision_3d_big_res]=1
SKIP_STEPS[ddb_3d_big_res]=1

# Grid length values to test
GRID_LENGTHS=(0.025 0.05 0.1 0.2 0.4 0.8 1.6 3.2)

# Template file
TEMPLATE="$CONFIG_DIR/ablation_grid_length_template.yaml"

log_message "==================================================="
log_message "Starting Grid Length Ablation Study"
log_message "Python command: $PYTHON_CMD"
log_message "Grid length values: ${GRID_LENGTHS[@]}"
log_message "Output directory: $OUTPUT_DIR"
log_message "Force recompute: $FORCE_RECOMPUTE"
log_message "Skip metrics: $SKIP_METRICS"
log_message "Cache enabled: $( [ "$FORCE_RECOMPUTE" = false ] && echo 'YES' || echo 'NO' )"
log_message "==================================================="

# Counter for progress
total_runs=$((${#SIMULATIONS[@]} * ${#GRID_LENGTHS[@]}))
current_run=0

# Main loop: iterate over simulations and grid lengths
for sim_name in "${!SIMULATIONS[@]}"; do
    sim_path="${SIMULATIONS[$sim_name]}"
    model_path="${MODEL_PATHS[$sim_name]}"
    fold="${FOLD_NUMBERS[$sim_name]}"
    frame_range="${FRAME_RANGES[$sim_name]}"
    skip_steps="${SKIP_STEPS[$sim_name]}"
    
    log_message ""
    log_message "Processing simulation: $sim_name (fold $fold) - Frames: $frame_range - Skip: $skip_steps"
    
    for grid_length in "${GRID_LENGTHS[@]}"; do
        current_run=$((current_run + 1))
        
        log_message "[$current_run/$total_runs] Grid length: $grid_length"
        
        config_file="$OUTPUT_DIR/config_${sim_name}_gl${grid_length}.yaml"
        
        # Generate config file
        predict_id=$(generate_config "$TEMPLATE" "$sim_name" "$sim_path" "$model_path" "$fold" "$grid_length" "$config_file" "$frame_range" "$skip_steps")
        
        # Run prediction with timing
        run_prediction "$config_file" "$sim_name" "$grid_length" "$fold" "$predict_id" "$sim_path" "$frame_range" "$skip_steps" || log_message "WARNING: Continuing after error"
    done
done

log_message ""
log_message "==================================================="
log_message "Grid Length Ablation Study Completed"
log_message "Results saved to: $TIMING_CSV"
log_message "Log file: $LOG_FILE"
log_message "==================================================="

# Display summary
log_message ""
log_message "Timing Summary:"
column -t -s ',' "$TIMING_CSV" | tee -a "$LOG_FILE"
