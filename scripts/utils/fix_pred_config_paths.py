#!/usr/bin/env python3
"""
Fix model_config_file paths in pred_config.yaml files.
Converts from list format to string format.
"""

import os
import yaml
from pathlib import Path

def fix_model_config_file(config):
    """Convert model_config_file from list to string path."""
    if 'model' in config and 'model_config_file' in config['model']:
        model_config = config['model']['model_config_file']
        
        # Check if it's a list
        if isinstance(model_config, list):
            # Filter empty strings and join with '/'
            path_parts = [part for part in model_config if part]
            if path_parts:
                # Join parts with '/' and add leading '/' if first part is not empty
                if model_config[0] == '':
                    config['model']['model_config_file'] = '/' + '/'.join(path_parts)
                else:
                    config['model']['model_config_file'] = '/'.join(path_parts)
                return True
    return False

def process_file(file_path):
    """Process a single pred_config.yaml file."""
    try:
        # Read original file to preserve formatting
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse YAML
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'model' in config and 'model_config_file' in config['model']:
            model_config = config['model']['model_config_file']
            
            # Check if it's a list
            if isinstance(model_config, list):
                # Filter empty strings and join with '/'
                path_parts = [part for part in model_config if part]
                if path_parts:
                    # Join parts with '/' and add leading '/' if first part is not empty
                    if model_config[0] == '':
                        fixed_path = '/' + '/'.join(path_parts)
                    else:
                        fixed_path = '/'.join(path_parts)
                    
                    # Replace in the original file
                    in_model_config = False
                    new_lines = []
                    skip_next_lines = 0
                    
                    for i, line in enumerate(lines):
                        if skip_next_lines > 0:
                            skip_next_lines -= 1
                            continue
                            
                        if '  model_config_file:' in line:
                            # Found the line, replace with string version
                            new_lines.append(f'  model_config_file: {fixed_path}\n')
                            # Skip subsequent list items
                            j = i + 1
                            while j < len(lines) and lines[j].startswith('  - '):
                                skip_next_lines += 1
                                j += 1
                        else:
                            new_lines.append(line)
                    
                    # Write back
                    with open(file_path, 'w') as f:
                        f.writelines(new_lines)
                    
                    print(f"✓ Fixed: {file_path}")
                    print(f"  Path: {fixed_path}")
                    return True
        
        print(f"- Skipped (no list found): {file_path}")
        return False
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def main():
    base_path = Path("/work1/voxel-fluid-net/data/3D/big2")
    
    # Find all pred_config.yaml files in sparse_regionwise_approach directories
    pred_config_files = list(base_path.glob("*/sparse_regionwise_approach/predictions/**/pred_config.yaml"))
    
    print(f"Found {len(pred_config_files)} pred_config.yaml files\n")
    
    fixed_count = 0
    for file_path in sorted(pred_config_files):
        if process_file(file_path):
            fixed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Total files processed: {len(pred_config_files)}")
    print(f"Files fixed: {fixed_count}")
    print(f"Files skipped: {len(pred_config_files) - fixed_count}")

if __name__ == "__main__":
    main()
