#!/bin/bash
# Visualize adsorbate, slab, and MOF structures separately with appropriate color schemes

uv run python src/structure_visualization.py configs/visualization/view_adsorbate.yaml
uv run python src/structure_visualization.py configs/visualization/view_slab.yaml
uv run python src/structure_visualization.py configs/visualization/view_mof.yaml
