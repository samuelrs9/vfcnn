#!/bin/bash

# List of paths
paths=(
  #Armadillo
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/armadillo/frames_dat/simdef_bpart_hpr_hdp=2.0.ini" 
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/armadillo/frames_dat/simdef_bpart_ia_hdp=2.0.ini"
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/armadillo/frames_dat/simdef_bpart_shellsplit_hdp=2.0.ini"
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/armadillo/frames_dat/simdef_marrone_hdp=2.0.ini"
  # Bunny
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/bunny/frames_dat/simdef_bpart_hpr_hdp=2.0.ini" 
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/bunny/frames_dat/simdef_bpart_ia_hdp=2.0.ini"
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/bunny/frames_dat/simdef_bpart_shellsplit_hdp=2.0.ini"
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/bunny/frames_dat/simdef_marrone_hdp=2.0.ini"  
  # Dragon
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/dragon/frames_dat/simdef_bpart_hpr_hdp=2.0.ini" 
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/dragon/frames_dat/simdef_bpart_ia_hdp=2.0.ini"
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/dragon/frames_dat/simdef_bpart_shellsplit_hdp=2.0.ini"
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/dragon/frames_dat/simdef_marrone_hdp=2.0.ini"
  #Rocker arm
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/rocker-arm/frames_dat/simdef_bpart_hpr_hdp=2.0.ini" 
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/rocker-arm/frames_dat/simdef_bpart_ia_hdp=2.0.ini"
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/rocker-arm/frames_dat/simdef_bpart_shellsplit_hdp=2.0.ini"
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/rocker-arm/frames_dat/simdef_marrone_hdp=2.0.ini"      
  #Happy
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/happy/frames_dat/simdef_bpart_hpr_hdp=2.0.ini" 
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/happy/frames_dat/simdef_bpart_ia_hdp=2.0.ini"
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/happy/frames_dat/simdef_bpart_shellsplit_hdp=2.0.ini"
  "/home/samuel/Doutorado/voxel-fluid-net/data/3D/static/happy/frames_dat/simdef_marrone_hdp=2.0.ini"    
)

# Loop through each path
for path in "${paths[@]}"
do
    echo "======================================================="
    echo "Running boundary method: $path"
    if [ -f "$path" ]; then
      echo "$path exist"
      /home/samuel/BPart/c++/build/BPart_OMP "$path"
    else
      echo "$path does not exist"
    fi    
done