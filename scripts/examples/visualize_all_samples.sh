#!/bin/bash
#
# Visualize all sample structures to showcase capabilities
#
# This script generates visualizations of all sample data:
# - Catalyst structures
# - MOF structures
#
# Each structure type uses its own config with optimized view angles.

echo "============================================================"
echo "Visualizing Sample Structures"
echo "============================================================"
echo ""
echo "This will generate visualizations of all sample data."
echo "Output will be saved to: data/examples_output/"
echo ""

# Catalyst structure
echo "1. Visualizing catalyst structure..."
uv run python src/structure_visualization.py configs/examples/view_catalyst.yaml
echo ""

# MOF structure
echo "2. Visualizing MOF structure..."
uv run python src/structure_visualization.py configs/examples/view_mof.yaml
echo ""

echo "============================================================"
echo "All sample visualizations complete!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - data/examples_output/catalyst/figures/catalyst_*.png"
echo "  - data/examples_output/mof/figures/mof_*.png"
echo ""
echo "Next steps:"
echo "  1. View the generated PNG files"
echo "  2. Customize configs in configs/examples/ for different view angles"
echo "  3. Visualize your own CIF files:"
echo "     bash scripts/visualization/view_cif.sh path/to/your.cif"
echo ""
