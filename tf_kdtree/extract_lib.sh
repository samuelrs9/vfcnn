#!/bin/bash
# Script to extract compiled library from Docker container

set -e

echo "Building Docker image..."
docker build -t tf_kdtree:test .

echo "Extracting compiled library..."
# Create temporary container
CONTAINER_ID=$(docker create tf_kdtree:test)

# Copy library from container to host
docker cp $CONTAINER_ID:/tf_kdtree/libtf_nndistance.so ./libtf_nndistance.so

# Remove temporary container
docker rm $CONTAINER_ID

echo "Library extracted successfully: libtf_nndistance.so"
ls -lh libtf_nndistance.so
