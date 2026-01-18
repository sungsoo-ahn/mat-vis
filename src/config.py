"""
Shared configuration parsing utilities for mat-vis.

Provides centralized config loading and parsing for both
structure_visualization.py and diffusion_visualization.py.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from src.utils import (
    get_framework_colors,
    get_adsorbate_colors,
    get_boundary_style,
    get_crystal_colors,
)


# ============================================================================
# CONFIG DATACLASSES
# ============================================================================

@dataclass
class VisualizationConfig:
    """Parsed visualization configuration."""
    # Output paths
    output_dir: Path
    figures_dir: Path

    # Color schemes (resolved dicts)
    framework_colors: Dict[int, str]
    adsorbate_colors: Dict[int, str]
    boundary_style: Dict[str, Any]

    # View settings
    view_elev: float
    view_azim: float
    dpi: int

    # Supercell settings
    supercell_matrix: Optional[List[List[int]]]

    # Adsorbate settings
    separate_adsorbate: bool
    plot_separate: bool
    isolation_method: str
    isolation_config: Dict[str, Any]
    adsorbate_shift: Optional[Tuple[float, float, float]]
    adsorbate_index: Optional[int]

    # Input files
    input_files: List[str]
    input_file: Optional[str]

    # Structure type (controls color scheme: "adsorbate", "slab", "mof", "framework")
    structure_type: str

    # Rendering options
    show_shadow: bool
    glossy: bool
    margin: float
    size_scale: float
    figsize: Tuple[int, int]
    transparent: bool
    show_title: bool


@dataclass
class DiffusionConfig(VisualizationConfig):
    """Extended config for diffusion visualization."""
    mode: str
    random_seed: int
    create_gif: bool
    trajectory_config: Dict[str, Any]
    animation_config: Dict[str, Any]
    crystal_type: str
    crystal_colors: Dict[int, str]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def tiling_to_supercell_matrix(tiling: List[int]) -> List[List[int]]:
    """Convert [nx, ny, nz] tiling to 3x3 supercell matrix."""
    nx, ny, nz = tiling
    return [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]


def parse_supercell_config(vis_config: Dict[str, Any]) -> Optional[List[List[int]]]:
    """Extract supercell matrix from visualization config."""
    supercell_matrix = vis_config.get('supercell_matrix')
    tiling = vis_config.get('tiling')

    if tiling is not None and supercell_matrix is None:
        return tiling_to_supercell_matrix(tiling)
    return supercell_matrix


def parse_adsorbate_shift(shift: Optional[List[float]]) -> Optional[Tuple[float, float, float]]:
    """Convert adsorbate shift list to tuple."""
    if shift is not None:
        return tuple(shift)
    return None


# ============================================================================
# CONFIG PARSERS
# ============================================================================

def parse_visualization_config(config: Dict[str, Any]) -> VisualizationConfig:
    """Parse raw config dict into VisualizationConfig dataclass."""
    output_dir = Path(config['output_dir'])
    figures_dir = output_dir / 'figures'

    vis_config = config.get('visualization', {})

    # Parse input files
    input_files = config.get('input_files', [])
    if isinstance(input_files, str):
        input_files = [input_files]

    input_file = config.get('input_file')
    if len(input_files) > 0 and input_file is None:
        input_file = input_files[0]

    return VisualizationConfig(
        output_dir=output_dir,
        figures_dir=figures_dir,
        framework_colors=get_framework_colors(config.get('framework_scheme')),
        adsorbate_colors=get_adsorbate_colors(config.get('adsorbate_scheme')),
        boundary_style=get_boundary_style(config.get('boundary_scheme')),
        view_elev=vis_config.get('view_elev', 20),
        view_azim=vis_config.get('view_azim', -60),
        dpi=vis_config.get('dpi', 200),
        supercell_matrix=parse_supercell_config(vis_config),
        separate_adsorbate=vis_config.get('separate_adsorbate', False),
        plot_separate=vis_config.get('plot_separate', False),
        isolation_method=vis_config.get('isolation_method', 'element'),
        isolation_config=config.get('isolation', {}),
        adsorbate_shift=parse_adsorbate_shift(vis_config.get('adsorbate_shift')),
        adsorbate_index=vis_config.get('adsorbate_index'),
        input_files=input_files,
        input_file=input_file,
        structure_type=vis_config.get('structure_type', 'framework'),
        show_shadow=vis_config.get('show_shadow', True),
        glossy=vis_config.get('glossy', True),
        margin=vis_config.get('margin', 1.0),
        size_scale=vis_config.get('size_scale', 1.0),
        figsize=tuple(vis_config.get('figsize', [10, 10])),
        transparent=vis_config.get('transparent', False),
        show_title=vis_config.get('show_title', True),
    )


def parse_diffusion_config(config: Dict[str, Any]) -> DiffusionConfig:
    """Parse raw config dict into DiffusionConfig dataclass."""
    output_dir = Path(config['output_dir'])
    figures_dir = output_dir / 'figures'

    mode = config.get('mode', 'mof')
    vis_config = config.get('visualization', {})
    mode_config = config.get(mode, {})

    # Parse input files with mode fallback
    input_files = config.get('input_files', [])
    if isinstance(input_files, str):
        input_files = [input_files]

    input_file = config.get('input_file')
    if len(input_files) > 0 and input_file is None:
        input_file = input_files[0]
    if input_file is None:
        input_file = mode_config.get('input_file') or mode_config.get('catalyst_file')

    # View settings with mode-specific defaults
    view_elev = vis_config.get('view_elev')
    view_azim = vis_config.get('view_azim')

    if view_elev is None:
        view_elev = mode_config.get('view_elev', -10 if mode == 'mof' else 35)
    if view_azim is None:
        view_azim = mode_config.get('view_azim', 0 if mode == 'mof' else -50)

    # Supercell with mode fallback
    supercell_matrix = parse_supercell_config(vis_config)
    if supercell_matrix is None:
        mode_tiling = mode_config.get('tiling')
        if mode_tiling:
            supercell_matrix = tiling_to_supercell_matrix(mode_tiling)
        else:
            supercell_matrix = mode_config.get('supercell_matrix')

    # Adsorbate settings with mode fallback
    ads_shift = parse_adsorbate_shift(vis_config.get('adsorbate_shift'))
    if ads_shift is None:
        ads_shift = parse_adsorbate_shift(mode_config.get('adsorbate_shift'))

    ads_index = vis_config.get('adsorbate_index')
    if ads_index is None:
        ads_index = mode_config.get('adsorbate_index')

    # Isolation method with fallback
    isolation_method = vis_config.get('isolation_method')
    if isolation_method is None:
        isolation_method = config.get('isolation_method', 'element')

    # Crystal type for unconditional diffusion
    crystal_type = config.get('crystal_type', 'organic')

    return DiffusionConfig(
        output_dir=output_dir,
        figures_dir=figures_dir,
        framework_colors=get_framework_colors(config.get('framework_scheme')),
        adsorbate_colors=get_adsorbate_colors(config.get('adsorbate_scheme')),
        boundary_style=get_boundary_style(config.get('boundary_scheme')),
        view_elev=view_elev,
        view_azim=view_azim,
        dpi=config.get('dpi', vis_config.get('dpi', 200)),
        supercell_matrix=supercell_matrix,
        separate_adsorbate=vis_config.get('separate_adsorbate', False),
        plot_separate=config.get('plot_separate', vis_config.get('plot_separate', False)),
        isolation_method=isolation_method,
        isolation_config=config.get('isolation', {}),
        adsorbate_shift=ads_shift,
        adsorbate_index=ads_index,
        input_files=input_files,
        input_file=input_file,
        structure_type=vis_config.get('structure_type', 'framework'),
        show_shadow=vis_config.get('show_shadow', True),
        glossy=vis_config.get('glossy', True),
        margin=vis_config.get('margin', 1.0),
        size_scale=vis_config.get('size_scale', 1.0),
        figsize=tuple(vis_config.get('figsize', [10, 10])),
        transparent=vis_config.get('transparent', False),
        show_title=vis_config.get('show_title', True),
        mode=mode,
        random_seed=config.get('random_seed', 42),
        create_gif=config.get('create_gif', False),
        trajectory_config=config.get('trajectory', {}),
        animation_config=config.get('animation', {}),
        crystal_type=crystal_type,
        crystal_colors=get_crystal_colors(crystal_type),
    )
