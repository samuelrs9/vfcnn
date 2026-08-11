import os
import sys

local_path = os.path.dirname(__file__)
if local_path not in sys.path:
    sys.path.append(local_path)

local_path_up = os.path.dirname(local_path)
if local_path_up not in sys.path:
    sys.path.append(local_path_up)

#local_path_losses = os.path.join(local_path,'losses')
#if local_path_losses not in sys.path:
#    sys.path.append(local_path_losses)

# voxel-cloud-net path
vfnet_path = os.path.join(local_path_up,'voxel-cloud-net')
if vfnet_path not in sys.path:
    sys.path.append(vfnet_path)

# voxelizer path
voxelizer_path = os.path.join(local_path_up,'voxelizer')
if voxelizer_path not in sys.path:
    sys.path.append(voxelizer_path)