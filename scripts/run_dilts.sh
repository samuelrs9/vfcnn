#!/bin/bash

# List of simulations
configs=(
  # "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/armadillo/frames_dat 2.119136816" 
  # "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/bunny/frames_dat 0.00358278"
  # "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/dragon/frames_dat 1.383524602"
  # "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/rocker-arm/frames_dat 0.041814491"
  # "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/happy/frames_dat 0.002440555"
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/big/fountain_3d_big_res/frames_rest 0.027712812"
)

# Loop through each path
for config in "${configs[@]}"
do
    path=$(echo $config | awk '{print $1}')
    res=$(echo $config | awk '{print $2}')
    echo "======================================================="    
    echo "Running Dilts method: $path"
    if [ -d "$path" ]; then
      echo "$path exist"
      /home/samuel/Doutorado/gt3d/build/gt -g "$res" "$path"
    else
      echo "$path does not exist"
    fi    
done