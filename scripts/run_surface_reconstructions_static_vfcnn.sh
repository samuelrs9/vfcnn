#!/bin/bash

#static=("armadillo" "bunny" "dragon" "happy" "rocker-arm")
static=("dragon")
#static=("armadillo")

pred_dir="sparse_regionwise_approach/predictions/kfold3_static_hdp=2.0"

for mesh in "${static[@]}"
do
  pred_paths="$(pwd)/data/3D/static/$mesh/$pred_dir/*/ply"
  # Loop through each path
  for path in $pred_paths
  do
    if [ -e "$path" ]; then
      echo "======================================================="
      echo "Running surface reconstruction for $path"
      bash "$(pwd)/scripts/surface_reconstruction.sh" "$path"
    else
      echo "Path $path does not exist or is not accessible"
    fi
  done
done