#!/bin/bash

# Check if the number of arguments is equal to 1
if [ $# -ne 1 ]; then
    echo "Usage: $0 directory"
    exit 1
fi

# Directory provided as an argument
directory="$1"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "Directory does not exist: $directory"
    exit 1
fi

# Iterate through files in the directory with the pattern "*.ply"
for file in "$directory"/*.*.ply; do
    if [ -f "$file" ]; then
        # Extract the filename without the path
        filename=$(basename "$file")

        # Define the output filename using the same name with "_mesh" at the end
        output_file="$directory/mesh.${filename%.*}.ply"

        # Check if the file exists
        if [ -f "$output_file" ]; then
            echo "File exists. Skipping loop iteration."
            # Skip rest of the loop iteration using continue
            continue
        fi

        # Run the PoissonRecon command
        /home/samuel/Doutorado/SurfaceReconstruction/AdaptiveSolvers/Bin/Linux/PoissonRecon --in "$file" --out "$output_file" --depth 9

        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "Mesh extraction for $file completed successfully."
        else
            echo "Error extracting mesh for $file."
        fi
    fi
done
