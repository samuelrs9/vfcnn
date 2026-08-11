#!/bin/bash
# Migration script: Convert all .ini config files to .yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONVERTER="$SCRIPT_DIR/convert_ini_to_yaml.py"

echo "==================================="
echo "INI to YAML Migration Script"
echo "==================================="
echo

# Count .ini files
echo "Scanning for .ini files..."
INI_COUNT=$(find "$REPO_DIR/data" "$REPO_DIR/models" -name "*.ini" -type f 2>/dev/null | wc -l)
echo "Found $INI_COUNT .ini files"
echo

# Ask for confirmation
read -p "Convert all $INI_COUNT files? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# Convert all files
echo
echo "Converting files..."
python "$CONVERTER" "$REPO_DIR/data" "$REPO_DIR/models" -r

echo
echo "==================================="
echo "Migration completed!"
echo "==================================="
echo
echo "Next steps:"
echo "1. Update ConfigReader in sim_reader/config.py to read YAML"
echo "2. Update default config filenames from .ini to .yaml"
echo "3. Test with: python scripts/predict.py scripts/configs/prediction/sparse_regionwise_fluid_best_checkpoint.yaml"
