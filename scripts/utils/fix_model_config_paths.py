#!/usr/bin/env python3
"""
Script to fix paths in model_config_v2.yaml files that are in list format instead of string format.
"""
import os
import yaml
from pathlib import Path
import glob


def fix_path_list(path_list):
    """Convert a list of path components to a proper path string."""
    if not isinstance(path_list, list):
        return path_list
    
    # Join the path components, handling special cases
    # Remove quotes from first and last elements if present
    cleaned = []
    for i, part in enumerate(path_list):
        part_str = str(part)
        # Remove leading/trailing quotes
        part_str = part_str.strip("'\"")
        cleaned.append(part_str)
    
    # Join with appropriate separator
    path_str = '/' + '/'.join(cleaned)
    # Clean up any double slashes
    path_str = path_str.replace('//', '/')
    
    return path_str


def fix_config_file(config_path):
    """Fix a single config file."""
    print(f"Processing: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check and fix paths in dataset section
    if 'dataset' in config:
        dataset = config['dataset']
        
        # Fix sim_dir
        if 'sim_dir' in dataset and isinstance(dataset['sim_dir'], list):
            print(f"  Fixing sim_dir: {dataset['sim_dir']}")
            dataset['sim_dir'] = fix_path_list(dataset['sim_dir'])
            print(f"  -> {dataset['sim_dir']}")
        
        # Fix config_file
        if 'config_file' in dataset and isinstance(dataset['config_file'], list):
            print(f"  Fixing config_file: {dataset['config_file']}")
            dataset['config_file'] = fix_path_list(dataset['config_file'])
            print(f"  -> {dataset['config_file']}")
        
        # Fix train_file
        if 'train_file' in dataset and isinstance(dataset['train_file'], list):
            print(f"  Fixing train_file: {dataset['train_file']}")
            dataset['train_file'] = fix_path_list(dataset['train_file'])
            print(f"  -> {dataset['train_file']}")
        
        # Fix validation_file
        if 'validation_file' in dataset and isinstance(dataset['validation_file'], list):
            print(f"  Fixing validation_file: {dataset['validation_file']}")
            dataset['validation_file'] = fix_path_list(dataset['validation_file'])
            print(f"  -> {dataset['validation_file']}")
    
    # Write back to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"  ✓ Fixed successfully\n")


def main():
    """Main function to process all config files."""
    pattern = "/work1/voxel-fluid-net/models/kfold3/sparse_regionwise_approach/models/*/model_config_v2.yaml"
    config_files = glob.glob(pattern)
    
    print(f"Found {len(config_files)} config files to process\n")
    
    for config_file in config_files:
        try:
            fix_config_file(config_file)
        except Exception as e:
            print(f"ERROR processing {config_file}: {e}\n")
    
    print(f"Finished processing {len(config_files)} files")


if __name__ == "__main__":
    main()
