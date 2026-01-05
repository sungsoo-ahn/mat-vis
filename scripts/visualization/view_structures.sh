#!/bin/bash
#
# View static structures from CIF/XYZ files
#
# Usage:
#   bash scripts/visualization/view_structures.sh <config.yaml>
#
# Example:
#   bash scripts/visualization/view_structures.sh configs/visualization/view_catalyst.yaml
#

if [ -z "$1" ]; then
    echo "Error: No config file specified"
    echo "Usage: bash scripts/visualization/view_structures.sh <config.yaml>"
    exit 1
fi

CONFIG="$1"

echo "Running structure visualization with config: $CONFIG"
uv run python src/structure_visualization.py "$CONFIG"
