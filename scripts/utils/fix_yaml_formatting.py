#!/usr/bin/env python3
"""
Fix YAML formatting issues from INI to YAML conversion:
1. Convert Python list strings to proper YAML lists
2. Remove excessive quotes
"""

import yaml
import argparse
import sys
from pathlib import Path
import re
import ast


def parse_value(value_str, key=None):
    """Parse string values and convert to proper Python types."""
    if not isinstance(value_str, str):
        return value_str
    
    value_str = value_str.strip()
    
    # Remove surrounding single quotes that are part of the string content
    # YAML interprets '''string''' as the literal string 'string' (with quotes)
    if value_str.startswith("'") and value_str.endswith("'") and len(value_str) >= 2:
        value_str = value_str[1:-1]
    
    # Try to parse as Python literal (lists, tuples, etc.)
    try:
        parsed = ast.literal_eval(value_str)
        # If it's a list, tuple, dict, or number, use the parsed value
        if isinstance(parsed, (list, tuple, dict, int, float)):
            # Convert tuples to lists for YAML
            if isinstance(parsed, tuple):
                return list(parsed)
            return parsed
    except (ValueError, SyntaxError):
        pass
    
    # Try to convert to number
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        pass
    
    # Check if this is a space-separated list for specific fields
    # Common fields that should be lists: labels, coords, etc.
    if key in ['labels', 'coords'] and ' ' in value_str:
        # Split by whitespace and return as list
        parts = value_str.split()
        if len(parts) > 1:
            # Check if all parts are simple strings (not containing special chars)
            if all(part.replace('_', '').replace('-', '').isalnum() for part in parts):
                return parts
    
    # Return as string (without excessive quotes)
    return value_str


def fix_yaml_dict(data, parent_key=None):
    """Recursively fix YAML dictionary values."""
    if isinstance(data, dict):
        return {key: fix_yaml_dict(value, key) for key, value in data.items()}
    elif isinstance(data, list):
        return [fix_yaml_dict(item, parent_key) for item in data]
    elif isinstance(data, str):
        return parse_value(data, parent_key)
    else:
        return data


def fix_yaml_file(file_path, dry_run=False):
    """Fix a single YAML file."""
    try:
        # Read the file
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            return False
        
        # Fix the data
        fixed_data = fix_yaml_dict(data)
        
        if not dry_run:
            # Write back with proper formatting
            # Use a custom representer to avoid excessive quoting
            class NoAliasDumper(yaml.SafeDumper):
                def ignore_aliases(self, data):
                    return True
                    
                def represent_str(self, data):
                    # Use plain style for simple strings without special characters
                    if any(c in data for c in '\n\r\t\'"\\'):
                        return super().represent_str(data)
                    return self.represent_scalar('tag:yaml.org,2002:str', data, style='')
            
            NoAliasDumper.add_representer(str, NoAliasDumper.represent_str)
            
            with open(file_path, 'w') as f:
                yaml.dump(fixed_data, f, 
                         Dumper=NoAliasDumper,
                         default_flow_style=False, 
                         sort_keys=False, 
                         allow_unicode=True,
                         width=float('inf'))
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Fix YAML formatting issues')
    parser.add_argument('paths', nargs='+', help='Files or directories to fix')
    parser.add_argument('-r', '--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    yaml_files = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix in ['.yaml', '.yml']:
            yaml_files.append(path)
        elif path.is_dir():
            if args.recursive:
                yaml_files.extend(path.rglob('*.yaml'))
                yaml_files.extend(path.rglob('*.yml'))
            else:
                yaml_files.extend(path.glob('*.yaml'))
                yaml_files.extend(path.glob('*.yml'))
    
    if not yaml_files:
        print("No YAML files found.")
        return
    
    print(f"Found {len(yaml_files)} YAML files")
    
    success_count = 0
    for yaml_file in yaml_files:
        if fix_yaml_file(yaml_file, args.dry_run):
            success_count += 1
            action = "Would fix" if args.dry_run else "Fixed"
            print(f"{action}: {yaml_file}")
    
    print(f"\nSuccessfully {'would fix' if args.dry_run else 'fixed'}: {success_count}/{len(yaml_files)}")


if __name__ == '__main__':
    main()
