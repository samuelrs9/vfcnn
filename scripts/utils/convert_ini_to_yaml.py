#!/usr/bin/env python3
"""
Convert .ini configuration files to .yaml format.
"""
import configparser
import os
import sys
from pathlib import Path
import yaml


def parse_value(value):
    """Parse a config value, handling different types."""
    value = value.strip()
    
    # Try to convert to number
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    # Check if it's space-separated numbers (like limits)
    parts = value.split()
    if len(parts) > 1:
        try:
            return [float(p) for p in parts]
        except ValueError:
            pass
    
    # Check if it's slash-separated (like coords)
    if '/' in value:
        return [p.strip() for p in value.split('/')]
    
    # Return as string
    return value


def ini_to_dict(ini_file):
    """Convert INI file to dictionary."""
    config = configparser.ConfigParser()
    config.read(ini_file)
    
    result = {}
    for section in config.sections():
        result[section] = {}
        for key, value in config.items(section):
            result[section][key] = parse_value(value)
    
    return result


def convert_file(ini_path, yaml_path=None, dry_run=False):
    """Convert a single INI file to YAML."""
    if yaml_path is None:
        yaml_path = ini_path.replace('.ini', '.yaml')
    
    try:
        data = ini_to_dict(ini_path)
        
        if dry_run:
            print(f"Would convert: {ini_path} -> {yaml_path}")
            print(yaml.dump(data, default_flow_style=False, sort_keys=False))
            print("---")
        else:
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            print(f"Converted: {ini_path} -> {yaml_path}")
        
        return True
    except Exception as e:
        print(f"Error converting {ini_path}: {e}", file=sys.stderr)
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert INI files to YAML')
    parser.add_argument('paths', nargs='+', help='Paths to INI files or directories')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be converted without writing files')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursively search directories')
    args = parser.parse_args()
    
    ini_files = []
    for path in args.paths:
        p = Path(path)
        if p.is_file() and p.suffix == '.ini':
            ini_files.append(str(p))
        elif p.is_dir():
            pattern = '**/*.ini' if args.recursive else '*.ini'
            ini_files.extend(str(f) for f in p.glob(pattern))
    
    if not ini_files:
        print("No .ini files found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(ini_files)} .ini files")
    
    success = 0
    for ini_file in ini_files:
        if convert_file(ini_file, dry_run=args.dry_run):
            success += 1
    
    print(f"\nSuccessfully converted: {success}/{len(ini_files)}")


if __name__ == '__main__':
    main()
