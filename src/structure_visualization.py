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
                             adsorbate_index=None,
                             framework_colors=None, adsorbate_colors=None,
                             boundary_style=None, structure_type="framework",
                             show_shadow=True, glossy=True, margin=1.0,
                             size_scale=1.0, transparent=False, show_title=True):
    """
    Visualize a structure file (CIF or XYZ).

    Args:
        file_path: Path to structure file
        output_path: Optional output path for PNG
        supercell_matrix: Optional supercell matrix
        separate_adsorbate: Whether to separate and highlight adsorbate
        view_elev: Camera elevation
        view_azim: Camera azimuth
        framework_scheme: Framework color scheme name (used if framework_colors not provided)
        adsorbate_scheme: Adsorbate color scheme name (used if adsorbate_colors not provided)
        boundary_scheme: Boundary scheme name (used if boundary_style not provided)
        dpi: Resolution
        figsize: Figure size (width, height)
        plot_separate: If True, generate separate plots for framework, adsorbate, and combined
        isolation_method: "element" (by atomic number) or "connectivity" (for MOFs)
        isolation_config: Config dict for connectivity isolation
        adsorbate_shift: Periodic shift (na, nb, nc) for adsorbate positioning
        adsorbate_index: Index of adsorbate to keep (None = all, 0 = first, etc.)
        framework_colors: Pre-resolved framework color dict (overrides framework_scheme)
        adsorbate_colors: Pre-resolved adsorbate color dict (overrides adsorbate_scheme)
        boundary_style: Pre-resolved boundary style dict (overrides boundary_scheme)
        structure_type: Type of structure ("adsorbate", "slab", "mof", "framework").
                        When "adsorbate" and separate_adsorbate=False, uses adsorbate_colors.
        show_shadow: Whether to render shadows on atoms
        glossy: Whether to render highlight and specular effects (False for flat spheres)
        margin: Margin around structure for axis limits
        size_scale: Scale factor for atom sizes
        figsize: Figure size (width, height)
        transparent: Whether to save with transparent background
        show_title: Whether to display title on figure

    Returns:
        Figure object (or dict of figures if plot_separate=True)
    """
    # Get color schemes (use provided or resolve from scheme names)
    if framework_colors is None:
        framework_colors = get_framework_colors(framework_scheme)
    if adsorbate_colors is None:
        adsorbate_colors = get_adsorbate_colors(adsorbate_scheme)
    if boundary_style is None:
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
        # For adsorbate structure type, render as adsorbate (bright colors)
        if structure_type == "adsorbate":
            framework_pos, framework_nums = np.array([]).reshape(0, 3), np.array([])
            ads_pos, ads_nums = positions, numbers
        else:
            framework_pos, framework_nums = positions, numbers
            ads_pos, ads_nums = np.array([]).reshape(0, 3), np.array([])

    # Determine title based on structure_type
    structure_titles = {
        "adsorbate": "Adsorbate",
        "slab": "Slab",
        "mof": "MOF",
        "framework": formula,
    }
    title = structure_titles.get(structure_type, formula)

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
                        boundary_style=boundary_style,
                        show_shadow=show_shadow,
                        glossy=glossy,
                        size_scale=size_scale)
        set_axis_limits_with_margin(ax_fw, all_pos, margin=margin)
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
                        boundary_style=boundary_style,
                        show_shadow=show_shadow,
                        glossy=glossy,
                        size_scale=size_scale)
        set_axis_limits_with_margin(ax_ads, all_pos, margin=margin)
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
                        title=title, view_elev=view_elev, view_azim=view_azim,
                        framework_colors=framework_colors,
                        adsorbate_colors=adsorbate_colors,
                        boundary_style=boundary_style,
                        show_shadow=show_shadow,
                        glossy=glossy,
                        size_scale=size_scale)
        set_axis_limits_with_margin(ax_combined, all_pos, margin=margin)
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
                    title=title if show_title else "", view_elev=view_elev, view_azim=view_azim,
                    framework_colors=framework_colors,
                    adsorbate_colors=adsorbate_colors,
                    boundary_style=boundary_style,
                    show_shadow=show_shadow,
                    glossy=glossy,
                    size_scale=size_scale)

    set_axis_limits_with_margin(ax, all_pos, margin=margin)

    if output_path:
        facecolor = 'none' if transparent else 'white'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor=facecolor, edgecolor='none', transparent=transparent)
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

    from src.config import load_config, parse_visualization_config

    config = load_config(sys.argv[1])
    cfg = parse_visualization_config(config)

    # Setup output directory
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Structure Visualization")
    print("=" * 60)

    for input_file in cfg.input_files:
        input_path = Path(input_file)
        print(f"\nProcessing: {input_path}")

        output_name = input_path.stem + '.png'
        output_path = cfg.figures_dir / output_name

        visualize_structure_file(
            file_path=input_file,
            output_path=output_path,
            supercell_matrix=cfg.supercell_matrix,
            separate_adsorbate=cfg.separate_adsorbate,
            view_elev=cfg.view_elev,
            view_azim=cfg.view_azim,
            dpi=cfg.dpi,
            plot_separate=cfg.plot_separate,
            isolation_method=cfg.isolation_method,
            isolation_config=cfg.isolation_config,
            adsorbate_shift=cfg.adsorbate_shift,
            adsorbate_index=cfg.adsorbate_index,
            framework_colors=cfg.framework_colors,
            adsorbate_colors=cfg.adsorbate_colors,
            boundary_style=cfg.boundary_style,
            structure_type=cfg.structure_type,
            show_shadow=cfg.show_shadow,
            glossy=cfg.glossy,
            margin=cfg.margin,
            size_scale=cfg.size_scale,
            figsize=cfg.figsize,
            transparent=cfg.transparent,
            show_title=cfg.show_title,
        )

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output saved to: {cfg.figures_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
