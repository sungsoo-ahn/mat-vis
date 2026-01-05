#!/bin/bash
#
# Run flow matching visualization for MOF structure
#
# Usage:
#   bash scripts/visualization/run_mof_diffusion.sh
#   bash scripts/visualization/run_mof_diffusion.sh --create-gif
#

CONFIG="configs/visualization/mof_diffusion.yaml"

# Check if --create-gif flag is passed
if [[ "$1" == "--create-gif" ]]; then
    echo "Running with GIF animation enabled..."
    # Temporarily modify config to enable GIF (using sed)
    sed -i.bak 's/create_gif: false/create_gif: true/' "$CONFIG"
    uv run python src/visualization.py "$CONFIG"
    # Restore original config
    mv "${CONFIG}.bak" "$CONFIG"
else
    echo "Running MOF diffusion visualization (static frames only)..."
    uv run python src/visualization.py "$CONFIG"
fi
