#!/bin/bash

# List of paths
paths=(
  "/work1/Doutorado/data/3D/big/new_db_blocks_3d_big_res/gt_blocks_ply"
  # # ddb_3d_big_res
  # "/work1/Doutorado/data/3D/big/ddb_3d_big_res/other_predictions/bpart_hpr/pred" 
  # "/work1/Doutorado/data/3D/big/ddb_3d_big_res/other_predictions/bpart_ia_4/pred" 
  # "/work1/Doutorado/data/3D/big/ddb_3d_big_res/other_predictions/bpart_ss_5/pred"
  # "/work1/Doutorado/data/3D/big/ddb_3d_big_res/other_predictions/marrone/pred"
  # # inlet collision
  # "/home/samuel/Doutorado/voxel-fluid-net/data/3D/big/inlet_collision_3d_big_res/other_predictions/marrone/mesh"
  # # inlet vortex
  # "/work1/Doutorado/data/3D/big/inlet_vortex_3d_big_res/other_predictions/bpart_hpr/pred" 
  # "/work1/Doutorado/data/3D/big/inlet_vortex_3d_big_res/other_predictions/bpart_ia_4/pred" 
  # "/work1/Doutorado/data/3D/big/inlet_vortex_3d_big_res/other_predictions/bpart_ss_5/pred"
  # "/work1/Doutorado/data/3D/big/inlet_vortex_3d_big_res/other_predictions/marrone/pred"
)

# Loop through each path
for path in "${paths[@]}"
do
    # Run surface_reconstruction.sh with the path as an argument
    echo "======================================================="
    echo "Running surface reconstruction for $path"
    bash surface_reconstruction.sh "$path"
done