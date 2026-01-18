#!/bin/bash
#
# Simple script to visualize any CIF file
#
# Usage:
#   bash scripts/visualization/view_cif.sh <path/to/file.cif> [output_dir]
#
# Examples:
#   bash scripts/visualization/view_cif.sh examples/catalyst.cif
#   bash scripts/visualization/view_cif.sh my_structure.cif data/my_viz
#

if [ -z "$1" ]; then
    echo "Error: No CIF file specified"
    echo ""
    echo "Usage: bash scripts/visualization/view_cif.sh <path/to/file.cif> [output_dir]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/visualization/view_cif.sh examples/catalyst.cif"
    echo "  bash scripts/visualization/view_cif.sh my_structure.cif data/my_viz"
    exit 1
fi

CIF_FILE="$1"
OUTPUT_DIR="${2:-data/viz_cif}"

# Check if file exists
if [ ! -f "$CIF_FILE" ]; then
    echo "Error: File not found: $CIF_FILE"
    exit 1
fi

# Create temporary config file
TMP_CONFIG=$(mktemp)
cat > "$TMP_CONFIG" <<EOF
output_dir: "$OUTPUT_DIR"

# Color schemes
framework_scheme: "cool_gray"
adsorbate_scheme: "mint_coral"
boundary_scheme: "thin_dark"

# Input file
input_files:
  - "$CIF_FILE"

# Visualization settings
visualization:
  separate_adsorbate: true
  tiling: [2, 2, 1]
  view_elev: 20
  view_azim: -60
  dpi: 200
EOF

echo "Visualizing: $CIF_FILE"
echo "Output will be saved to: $OUTPUT_DIR/figures/"
echo ""

# Run visualization
uv run python src/structure_visualization.py "$TMP_CONFIG"

# Cleanup
rm "$TMP_CONFIG"

echo ""
echo "Done! Check $OUTPUT_DIR/figures/ for output."
