#!/bin/bash
# Run all visualizations (structure and diffusion for MOF and catalyst)

set -e

echo "=============================================="
echo "Running all visualizations"
echo "=============================================="

# Structure visualizations
echo ""
echo "--- MOF Structure ---"
uv run python src/structure_visualization.py configs/visualization/view_mof.yaml

echo ""
echo "--- Catalyst Structure ---"
uv run python src/structure_visualization.py configs/visualization/view_catalyst.yaml

# Diffusion visualizations
echo ""
echo "--- MOF Diffusion ---"
uv run python src/diffusion_visualization.py configs/visualization/mof_diffusion.yaml

echo ""
echo "--- Catalyst Diffusion ---"
uv run python src/diffusion_visualization.py configs/visualization/catalyst_diffusion.yaml

echo ""
echo "=============================================="
echo "All visualizations complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  - data/mof/structure/figures/"
echo "  - data/mof/diffusion/figures/"
echo "  - data/catalyst/structure/figures/"
echo "  - data/catalyst/diffusion/figures/"
