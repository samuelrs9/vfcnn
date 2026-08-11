#!/bin/bash

data_dir="/work1/Doutorado/data/3D/static"

#static=("armadillo" "bunny" "dragon" "happy" "rocker-arm")
static=("rocker-arm")
#static=("armadillo")

for s in "${static[@]}"
do
  # List of paths
  paths=$(find "$data_dir/$s/sparse_regionwise_approach/predictions/kfold3__hdp=2.0_checkpoints" -type d -path "*/pred_sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0_*_kfold3_no_coarse/ply")
  
  # Loop through each path
  for path in $paths
  do
      # Run surface_reconstruction.sh with the path as an argument
      echo "======================================================="
      echo "Running surface reconstruction for $path"
      bash $(pwd)/scripts/surface_reconstruction.sh "$path"
  done
done