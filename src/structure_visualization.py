"""
Static structure visualization for MOF and catalyst CIF/XYZ files.

Provides simple viewing and rendering of atomic structures without diffusion.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.build import make_supercell
import yaml

from src.utils import (
    get_framework_colors,
    get_adsorbate_colors,
    get_boundary_style,
)
from src.rendering import render_structure, set_axis_limits_with_margin
from src.isolate_adsorbate import load_and_isolate


# ============================================================================
# DATA LOADING
# ============================================================================

def load_structure(file_path, supercell_matrix=None, separate_adsorbate=False,
                   isolation_method="element", isolation_config=None,
                   adsorbate_shift=None, adsorbate_index=None):
    """
    Load structure from CIF or XYZ file.

    Args:
        file_path: Path to structure file
        supercell_matrix: Optional 3x3 matrix for supercell creation
        separate_adsorbate: Whether to separate adsorbate from framework
        isolation_method: "element" (by atomic number) or "connectivity" (for MOFs)
        isolation_config: Config dict for connectivity isolation
        adsorbate_shift: Periodic shift (na, nb, nc) for adsorbate positioning
        adsorbate_index: Index of adsorbate to keep (None = all, 0 = first, etc.)

    Returns:
        If separate_adsorbate=False: (positions, atomic_numbers, formula)
        If separate_adsorbate=True: (framework_pos, framework_nums, ads_pos, ads_nums, formula)
    """
    if separate_adsorbate:
        return load_and_isolate(
            file_path=file_path,
            method=isolation_method,
            supercell_matrix=supercell_matrix,
            isolation_config=isolation_config,
            center_adsorbate=(isolation_method == "element"),
            adsorbate_shift=adsorbate_shift,
            adsorbate_index=adsorbate_index,
            verbose=False
        )
    else:
        atoms = read(file_path)
        formula = atoms.get_chemical_formula()

        if supercell_matrix is not None:
            P = np.array(supercell_matrix)
            atoms = make_supercell(atoms, P)

        positions = atoms.positions.copy()
        numbers = atoms.numbers.copy()

        # Center at origin
        center = positions.mean(axis=0)
        positions -= center
        return positions, numbers, formula


def load_mof_with_adsorbate(framework_file, adsorbate_file, supercell_matrix=None,
                            adsorbate_shift=(0, 0, 0)):
    """
    Load MOF framework and adsorbate from separate files.

    Args:
        framework_file: Path to framework structure file
        adsorbate_file: Path to adsorbate structure file
        supercell_matrix: Optional 3x3 matrix for framework supercell
        adsorbate_shift: Periodic shift (na, nb, nc) for adsorbate

    Returns:
        Tuple of (framework_pos, framework_nums, ads_pos, ads_nums, formula)
    """
    framework_atoms = read(framework_file)
    ads_atoms = read(adsorbate_file)

    framework_formula = framework_atoms.get_chemical_formula()
    ads_formula = ads_atoms.get_chemical_formula()

    # Apply periodic shift to adsorbate
    cell = ads_atoms.get_cell()
    na, nb, nc = adsorbate_shift
    shift_cartesian = na * cell[0] + nb * cell[1] + nc * cell[2]
    ads_pos = ads_atoms.positions.copy() + shift_cartesian
    ads_nums = ads_atoms.numbers.copy()

    # Create supercell if specified
    if supercell_matrix is not None:
        P = np.array(supercell_matrix)
        framework_atoms = make_supercell(framework_atoms, P)

    framework_pos = framework_atoms.positions.copy()
    framework_nums = framework_atoms.numbers.copy()

    # Center at origin
    all_pos = np.vstack([framework_pos, ads_pos])
    center = all_pos.mean(axis=0)
    framework_pos -= center
    ads_pos -= center

    return framework_pos, framework_nums, ads_pos, ads_nums, f"{framework_formula} + {ads_formula}"


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_structure_file(file_path, output_path=None, supercell_matrix=None,
                             separate_adsorbate=False, view_elev=20, view_azim=-60,
                             framework_scheme=None, adsorbate_scheme=None,
                             boundary_scheme=None, dpi=200, figsize=(10, 10),
                             plot_separate=False, isolation_method="element",
                             isolation_config=None, adsorbate_shift=None,
                             adsorbate_index=None):
    """
    Visualize a structure file (CIF or XYZ).

    Args:
        file_path: Path to structure file
        output_path: Optional output path for PNG
        supercell_matrix: Optional supercell matrix
        separate_adsorbate: Whether to separate and highlight adsorbate
        view_elev: Camera elevation
        view_azim: Camera azimuth
        framework_scheme: Framework color scheme name
        adsorbate_scheme: Adsorbate color scheme name
        boundary_scheme: Boundary scheme name
        dpi: Resolution
        figsize: Figure size (width, height)
        plot_separate: If True, generate separate plots for framework, adsorbate, and combined
        isolation_method: "element" (by atomic number) or "connectivity" (for MOFs)
        isolation_config: Config dict for connectivity isolation
        adsorbate_shift: Periodic shift (na, nb, nc) for adsorbate positioning
        adsorbate_index: Index of adsorbate to keep (None = all, 0 = first, etc.)

    Returns:
        Figure object (or dict of figures if plot_separate=True)
    """
    # Get color schemes
    framework_colors = get_framework_colors(framework_scheme)
    adsorbate_colors = get_adsorbate_colors(adsorbate_scheme)
    boundary_style = get_boundary_style(boundary_scheme)

    # Load structure
    if separate_adsorbate:
        framework_pos, framework_nums, ads_pos, ads_nums, formula = load_structure(
            file_path, supercell_matrix=supercell_matrix, separate_adsorbate=True,
            isolation_method=isolation_method, isolation_config=isolation_config,
            adsorbate_shift=adsorbate_shift, adsorbate_index=adsorbate_index
        )
    else:
        positions, numbers, formula = load_structure(
            file_path, supercell_matrix=supercell_matrix, separate_adsorbate=False
        )
        framework_pos, framework_nums = positions, numbers
        ads_pos, ads_nums = np.array([]).reshape(0, 3), np.array([])

    # Compute shared axis limits for consistent views
    all_pos = np.vstack([framework_pos, ads_pos]) if len(ads_pos) > 0 else framework_pos

    if plot_separate and separate_adsorbate and len(ads_pos) > 0:
        figures = {}
        output_stem = output_path.stem if output_path else "structure"
        output_dir = output_path.parent if output_path else Path(".")

        # 1. Framework only
        fig_fw = plt.figure(figsize=figsize, dpi=dpi)
        ax_fw = fig_fw.add_subplot(111, projection='3d')
        empty_pos = np.array([]).reshape(0, 3)
        empty_nums = np.array([])
        render_structure(ax_fw, framework_pos, framework_nums, empty_pos, empty_nums,
                        title="Framework", view_elev=view_elev, view_azim=view_azim,
                        framework_colors=framework_colors,
                        adsorbate_colors=adsorbate_colors,
                        boundary_style=boundary_style)
        set_axis_limits_with_margin(ax_fw, all_pos, margin=1.0)
        if output_path:
            fw_path = output_dir / f"{output_stem}_framework.png"
            plt.savefig(fw_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Saved: {fw_path}")
        figures['framework'] = fig_fw

        # 2. Adsorbate only
        fig_ads = plt.figure(figsize=figsize, dpi=dpi)
        ax_ads = fig_ads.add_subplot(111, projection='3d')
        render_structure(ax_ads, empty_pos, empty_nums, ads_pos, ads_nums,
                        title="Adsorbate", view_elev=view_elev, view_azim=view_azim,
                        framework_colors=framework_colors,
                        adsorbate_colors=adsorbate_colors,
                        boundary_style=boundary_style)
        set_axis_limits_with_margin(ax_ads, all_pos, margin=1.0)
        if output_path:
            ads_path = output_dir / f"{output_stem}_adsorbate.png"
            plt.savefig(ads_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Saved: {ads_path}")
        figures['adsorbate'] = fig_ads

        # 3. Combined
        fig_combined = plt.figure(figsize=figsize, dpi=dpi)
        ax_combined = fig_combined.add_subplot(111, projection='3d')
        render_structure(ax_combined, framework_pos, framework_nums, ads_pos, ads_nums,
                        title=formula, view_elev=view_elev, view_azim=view_azim,
                        framework_colors=framework_colors,
                        adsorbate_colors=adsorbate_colors,
                        boundary_style=boundary_style)
        set_axis_limits_with_margin(ax_combined, all_pos, margin=1.0)
        if output_path:
            combined_path = output_dir / f"{output_stem}_combined.png"
            plt.savefig(combined_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Saved: {combined_path}")
        figures['combined'] = fig_combined

        return figures

    # Single combined figure (default behavior)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    render_structure(ax, framework_pos, framework_nums, ads_pos, ads_nums,
                    title=formula, view_elev=view_elev, view_azim=view_azim,
                    framework_colors=framework_colors,
                    adsorbate_colors=adsorbate_colors,
                    boundary_style=boundary_style)

    set_axis_limits_with_margin(ax, all_pos, margin=1.0)

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main entry point for config-driven structure visualization."""
    if len(sys.argv) < 2:
        print("Usage: python src/structure_visualization.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup output directory
    output_dir = Path(config['output_dir'])
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Get visualization settings
    vis_config = config.get('visualization', {})
    view_elev = vis_config.get('view_elev', 20)
    view_azim = vis_config.get('view_azim', -60)
    dpi = vis_config.get('dpi', 200)
    separate_adsorbate = vis_config.get('separate_adsorbate', False)
    plot_separate = vis_config.get('plot_separate', False)
    supercell_matrix = vis_config.get('supercell_matrix')

    # Support simple tiling option [nx, ny, nz] as alternative to supercell_matrix
    tiling = vis_config.get('tiling')
    if tiling is not None and supercell_matrix is None:
        nx, ny, nz = tiling
        supercell_matrix = [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]

    # Isolation settings for separating adsorbates
    # "element": by atomic number (for catalysts)
    # "connectivity": by graph connectivity (for MOFs)
    isolation_method = vis_config.get('isolation_method', 'element')
    isolation_config = config.get('isolation', {})

    # Adsorbate shift (periodic cell units)
    ads_shift = vis_config.get('adsorbate_shift')
    if ads_shift is not None:
        ads_shift = tuple(ads_shift)

    # Adsorbate index (which adsorbate to keep, None = all)
    ads_index = vis_config.get('adsorbate_index')

    # Get color schemes
    framework_scheme = config.get('framework_scheme')
    adsorbate_scheme = config.get('adsorbate_scheme')
    boundary_scheme = config.get('boundary_scheme')

    # Process input files
    input_files = config.get('input_files', [])
    if isinstance(input_files, str):
        input_files = [input_files]

    print("=" * 60)
    print("Structure Visualization")
    print("=" * 60)

    for input_file in input_files:
        input_path = Path(input_file)
        print(f"\nProcessing: {input_path}")

        output_name = input_path.stem + '.png'
        output_path = figures_dir / output_name

        visualize_structure_file(
            file_path=input_file,
            output_path=output_path,
            supercell_matrix=supercell_matrix,
            separate_adsorbate=separate_adsorbate,
            view_elev=view_elev,
            view_azim=view_azim,
            framework_scheme=framework_scheme,
            adsorbate_scheme=adsorbate_scheme,
            boundary_scheme=boundary_scheme,
            dpi=dpi,
            plot_separate=plot_separate,
            isolation_method=isolation_method,
            isolation_config=isolation_config,
            adsorbate_shift=ads_shift,
            adsorbate_index=ads_index
        )

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output saved to: {figures_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
