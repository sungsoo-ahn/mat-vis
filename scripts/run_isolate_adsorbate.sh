#!/bin/bash
# Run adsorbate isolation on MOF structure
# Usage: bash scripts/run_isolate_adsorbate.sh [config_path]

CONFIG_PATH="${1:-configs/isolate_adsorbate.yaml}"

echo "Running adsorbate isolation..."
echo "Config: $CONFIG_PATH"
echo ""

uv run python src/isolate_adsorbate.py "$CONFIG_PATH"
