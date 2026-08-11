#!/bin/bash

#static=("armadillo" "bunny" "dragon" "happy" "rocker-arm")
static=("rocker-arm")
#data_dir = $(pwd)
data_dir="/work1/Doutorado/data/3D/static"
for s in "${static[@]}"
do
  # List of paths
  paths=(
    #"$(pwd)/data/3D/static/$s/sparse_regionwise_approach/predictions/kfold3_static_hdp=2.0/pred_sparse_voxelized_fluid_cnn_v3_4_10000_1.50_0.1_0_1_kfold3_no_coarse/ply"
    "$data_dir/$s/other_predictions_hdp=2.0/hpr/pred" 
    "$data_dir/$s/other_predictions_hdp=2.0/ss4/pred" 
    "$data_dir/$s/other_predictions_hdp=2.0/ia4/pred" 
    "$data_dir/$s/other_predictions_hdp=2.0/marrone/pred"    
  )

  # Loop through each path
  for path in "${paths[@]}"
  do
      # Run surface_reconstruction.sh with the path as an argument
      echo "======================================================="
      echo "Running surface reconstruction for $path"
      bash $(pwd)/scripts/surface_reconstruction.sh "$path"
  done
done