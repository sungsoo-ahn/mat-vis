#!/bin/bash
#
# Visualize all sample structures to showcase capabilities
#
# This script generates visualizations of all sample data:
# - Catalyst structures
# - MOF structures
# - Isolated adsorbates
#

echo "============================================================"
echo "Visualizing Sample Structures"
echo "============================================================"
echo ""
echo "This will generate visualizations of all sample data."
echo "Output will be saved to: data/examples_output/"
echo ""

OUTPUT_DIR="data/examples_output"

# Catalyst structure
echo "1. Visualizing catalyst structure..."
bash scripts/visualization/view_cif.sh data/sample/catalyst/1234.cif "$OUTPUT_DIR/catalyst"
echo ""

# MOF structure
echo "2. Visualizing MOF structure..."
bash scripts/visualization/view_cif.sh data/sample/mof.cif "$OUTPUT_DIR/mof"
echo ""

# Adsorbate (if exists)
if [ -f "data/sample/adsorbate.cif" ]; then
    echo "3. Visualizing adsorbate structure..."
    bash scripts/visualization/view_cif.sh data/sample/adsorbate.cif "$OUTPUT_DIR/adsorbate"
    echo ""
fi

echo "============================================================"
echo "All sample visualizations complete!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/catalyst/figures/1234.png"
echo "  - $OUTPUT_DIR/mof/figures/mof.png"
if [ -f "data/sample/adsorbate.cif" ]; then
    echo "  - $OUTPUT_DIR/adsorbate/figures/adsorbate.png"
fi
echo ""
echo "Next steps:"
echo "  1. View the generated PNG files"
echo "  2. Visualize your own CIF files:"
echo "     bash scripts/visualization/view_cif.sh path/to/your.cif"
echo ""
