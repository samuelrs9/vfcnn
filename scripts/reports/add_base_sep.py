#!/usr/bin/env python3
"""
Add missing base_sep field to sim_config YAML files.
"""
import yaml
import sys
from pathlib import Path


def add_base_sep_to_file(file_path, base_sep_value='.'):
    """Add base_sep field to a sim_config YAML file if it's missing."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            return False
        
        # Check if data section exists and if base_sep is missing
        if 'data' in data and 'base_sep' not in data['data']:
            data['data']['base_sep'] = base_sep_value
            
            # Write back
            class NoAliasDumper(yaml.SafeDumper):
                def ignore_aliases(self, data):
                    return True
            
            with open(file_path, 'w') as f:
                yaml.dump(data, f, 
                         Dumper=NoAliasDumper,
                         default_flow_style=False, 
                         sort_keys=False, 
                         allow_unicode=True,
                         width=float('inf'))
            
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Add base_sep field to sim_config files')
    parser.add_argument('paths', nargs='+', help='Files or directories to process')
    parser.add_argument('-r', '--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--value', default='.', help='Value for base_sep (default: ".")')
    
    args = parser.parse_args()
    
    yaml_files = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and 'sim_config' in path.name and path.suffix in ['.yaml', '.yml']:
            yaml_files.append(path)
        elif path.is_dir():
            if args.recursive:
                yaml_files.extend(path.rglob('sim_config*.yaml'))
                yaml_files.extend(path.rglob('sim_config*.yml'))
            else:
                yaml_files.extend(path.glob('sim_config*.yaml'))
                yaml_files.extend(path.glob('sim_config*.yml'))
    
    if not yaml_files:
        print("No sim_config YAML files found.")
        return
    
    print(f"Found {len(yaml_files)} sim_config files")
    
    updated_count = 0
    for yaml_file in yaml_files:
        if add_base_sep_to_file(yaml_file, args.value):
            updated_count += 1
            print(f"Updated: {yaml_file}")
    
    print(f"\nUpdated {updated_count}/{len(yaml_files)} files")


if __name__ == '__main__':
    main()
