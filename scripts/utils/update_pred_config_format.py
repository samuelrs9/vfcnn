#!/usr/bin/env python3
"""
Update pred_config.yaml format:
- model.model_name -> model.name
- model.model_config_file -> model.config_file
- boundary/normal.type_problem -> boundary/normal.task_type
"""

import os
import re
from pathlib import Path

def update_file(file_path):
    """Update a single pred_config.yaml file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Replace model_name with name
        content = re.sub(
            r'^(\s+)model_name:',
            r'\1name:',
            content,
            flags=re.MULTILINE
        )
        
        # Replace model_config_file with config_file
        content = re.sub(
            r'^(\s+)model_config_file:',
            r'\1config_file:',
            content,
            flags=re.MULTILINE
        )
        
        # Replace type_problem with task_type
        content = re.sub(
            r'^(\s+)type_problem:',
            r'\1task_type:',
            content,
            flags=re.MULTILINE
        )
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✓ Updated: {file_path}")
            return True
        else:
            print(f"- No changes: {file_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {file_path}: {e}")
        return False

def main():
    base_path = Path("/work1/voxel-fluid-net/data/3D/big2")
    
    # Find all pred_config.yaml files in sparse_regionwise_approach directories
    pred_config_files = list(base_path.glob("*/sparse_regionwise_approach/predictions/**/pred_config.yaml"))
    
    print(f"Found {len(pred_config_files)} pred_config.yaml files\n")
    
    updated_count = 0
    for file_path in sorted(pred_config_files):
        if update_file(file_path):
            updated_count += 1
    
    print(f"\n{'='*60}")
    print(f"Total files processed: {len(pred_config_files)}")
    print(f"Files updated: {updated_count}")
    print(f"Files unchanged: {len(pred_config_files) - updated_count}")

if __name__ == "__main__":
    main()
