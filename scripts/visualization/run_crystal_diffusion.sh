#!/bin/bash
#
# Run unconditional crystal diffusion visualization
#
# Usage:
#   bash scripts/visualization/run_crystal_diffusion.sh
#   bash scripts/visualization/run_crystal_diffusion.sh configs/visualization/custom.yaml
#

CONFIG="${1:-configs/visualization/crystal_diffusion.yaml}"

echo "Running crystal diffusion visualization..."
uv run python src/diffusion_visualization.py "$CONFIG"
