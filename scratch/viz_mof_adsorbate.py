"""
Visualize framework.xyz combined with adsorbate_CO2.xyz.
Framework tiled 1x2x2, view with azimuth=0, elevation=0.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgb
from ase.io import read
from ase.build import make_supercell
from pathlib import Path


# Element colors (lighter, pastel tones for better visibility)
ELEMENT_COLORS = {
    1: '#F8F8F8',    # H - white
    6: '#707070',    # C - medium gray
    7: '#6B8EF0',    # N - light blue
    8: '#F06060',    # O - coral red
    26: '#E08050',   # Fe - orange-brown
    46: '#40B0B8',   # Pd - light teal
    48: '#F0D060',   # Cd - light gold
    80: '#B0B0C8',   # Hg - light silver-blue
}


def lighten_color(color, amount=0.5):
    """Lighten a color by blending with white."""
    rgb = to_rgb(color)
    white = (1.0, 1.0, 1.0)
    return tuple(c * (1 - amount) + w * amount for c, w in zip(rgb, white))


def darken_color(color, amount=0.3):
    """Darken a color by reducing brightness."""
    rgb = to_rgb(color)
    return tuple(max(0, c * (1 - amount)) for c in rgb)


def get_element_size(atomic_number, is_adsorbate=False):
    """Get visualization size based on atomic number."""
    base_sizes = {
        1: 90, 6: 150, 7: 140, 8: 140,
        26: 180,  # Fe
        46: 170, 48: 180, 80: 190,
    }
    size = base_sizes.get(atomic_number, 160)
    if is_adsorbate:
        size *= 1.4
    else:
        size *= 0.5
    return size


def setup_clean_3d_axes(ax):
    """Remove grids and clean up 3D axes."""
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.xaxis.line.set_color('none')
    ax.yaxis.line.set_color('none')
    ax.zaxis.line.set_color('none')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def draw_glossy_sphere(ax, pos, atomic_num, size, alpha=1.0, is_adsorbate=False):
    """Draw a glossy sphere with specular highlight."""
    base_color = ELEMENT_COLORS.get(int(atomic_num), '#808080')
    light_offset = np.array([-0.15, 0.15, 0.2])
    base_size = size * alpha

    # Shadow layer
    shadow_color = darken_color(base_color, 0.5)
    shadow_offset = pos + np.array([0.05, -0.05, -0.05]) * (size / 100)
    ax.scatter(*shadow_offset, c=[shadow_color], s=base_size * 1.1,
               alpha=alpha * 0.4, edgecolors='none', depthshade=False)

    # Main sphere
    edge_color = '#333333'
    ax.scatter(*pos, c=[base_color], s=base_size,
               edgecolors=edge_color if alpha > 0.4 else '#888888',
               linewidth=1.8 if is_adsorbate else 1.2,
               alpha=alpha * 0.95, depthshade=False)

    # Highlight
    if alpha > 0.3:
        highlight_color = lighten_color(base_color, 0.3)
        highlight_pos = pos + light_offset * (size / 150)
        ax.scatter(*highlight_pos, c=[highlight_color], s=base_size * 0.5,
                   alpha=alpha * 0.6, edgecolors='none', depthshade=False)

    # Specular
    if alpha > 0.5:
        specular_pos = pos + light_offset * (size / 120)
        specular_size = base_size * 0.15 if is_adsorbate else base_size * 0.12
        ax.scatter(*specular_pos, c='white', s=specular_size,
                   alpha=alpha * 0.8, edgecolors='none', depthshade=False)


def load_mof_and_adsorbate(mof_path, ads_path, supercell_matrix=None, ads_shift=(0, 0, 0)):
    """Load MOF and adsorbate from separate files.

    Preserves the original spatial relationship between framework and adsorbate.
    Both files should share the same coordinate system (e.g., from the same
    isolation process).

    Args:
        mof_path: Path to MOF/framework file
        ads_path: Path to adsorbate file
        supercell_matrix: 3x3 matrix for supercell tiling
        ads_shift: (na, nb, nc) integer multiples of lattice vectors to shift adsorbate
    """
    mof_atoms = read(mof_path)
    ads_atoms = read(ads_path)

    mof_formula = mof_atoms.get_chemical_formula()
    ads_formula = ads_atoms.get_chemical_formula()

    # Get lattice vectors for periodic shift
    cell = ads_atoms.get_cell()

    # Apply periodic shift to adsorbate (integer multiples of lattice vectors)
    na, nb, nc = ads_shift
    shift_cartesian = na * cell[0] + nb * cell[1] + nc * cell[2]
    ads_pos = ads_atoms.positions.copy() + shift_cartesian
    ads_nums = ads_atoms.numbers.copy()

    if supercell_matrix is not None:
        P = np.array(supercell_matrix)
        mof_supercell = make_supercell(mof_atoms, P)
        mof_pos = mof_supercell.positions.copy()
        mof_nums = mof_supercell.numbers.copy()
    else:
        mof_pos = mof_atoms.positions.copy()
        mof_nums = mof_atoms.numbers.copy()

    # Center the entire complex at origin for visualization
    all_pos = np.vstack([mof_pos, ads_pos])
    center = all_pos.mean(axis=0)
    mof_pos -= center
    ads_pos -= center

    return mof_pos, mof_nums, ads_pos, ads_nums, f"{mof_formula} + {ads_formula}"


def visualize_structure(ax, mof_pos, mof_nums, ads_pos, ads_nums,
                        title="", mof_alpha=1.0, view_elev=0, view_azim=0):
    """Plot structure with glossy sphere rendering."""
    ax.clear()

    # Combine all atoms with their properties
    all_atoms = []
    for pos, num in zip(mof_pos, mof_nums):
        all_atoms.append({
            'pos': pos, 'num': num, 'is_adsorbate': False, 'alpha': mof_alpha
        })
    for pos, num in zip(ads_pos, ads_nums):
        all_atoms.append({
            'pos': pos, 'num': num, 'is_adsorbate': True, 'alpha': 1.0
        })

    # Sort by depth for proper rendering
    cam_dir = np.array([
        np.cos(np.radians(view_azim)) * np.cos(np.radians(view_elev)),
        np.sin(np.radians(view_azim)) * np.cos(np.radians(view_elev)),
        np.sin(np.radians(view_elev))
    ])
    all_atoms.sort(key=lambda a: np.dot(a['pos'], cam_dir))

    # Draw atoms back to front
    for atom in all_atoms:
        size = get_element_size(atom['num'], atom['is_adsorbate'])
        draw_glossy_sphere(
            ax, atom['pos'], atom['num'], size,
            alpha=atom['alpha'], is_adsorbate=atom['is_adsorbate']
        )

    setup_clean_3d_axes(ax)
    ax.view_init(elev=view_elev, azim=view_azim)

    if title:
        ax.set_title(title, fontsize=11, pad=10, fontweight='medium')


def main():
    """Main function to visualize framework with adsorbate."""
    framework_path = Path("data/sample/mof_clean.xyz")
    ads_path = Path("data/sample/co2_with_cell.xyz")
    output_dir = Path("data/sample/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading framework: {framework_path}")
    print(f"Loading adsorbate: {ads_path}")

    # 1x2x2 supercell with (0,0,1) periodic shift for adsorbate
    supercell_matrix = [[1, 0, 0], [0, 2, 0], [0, 0, 2]]
    ads_shift = (0, 0, 1)  # Shift by 1 lattice vector in c direction

    mof_pos, mof_nums, ads_pos, ads_nums, formula = load_mof_and_adsorbate(
        framework_path, ads_path, supercell_matrix=supercell_matrix, ads_shift=ads_shift
    )

    print(f"Formula: {formula}")
    print(f"Framework atoms (1x2x2 supercell): {len(mof_pos)}")
    print(f"Adsorbate atoms: {len(ads_pos)}")
    print(f"Adsorbate shift: {ads_shift}")

    # Create visualization with azimuth=0, elevation=0
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    visualize_structure(
        ax, mof_pos, mof_nums, ads_pos, ads_nums,
        title=f"Framework + CO2 (1x2x2 tiling, elev=0, azim=0)",
        mof_alpha=1.0,
        view_elev=0, view_azim=0
    )

    # Set axis limits
    all_pos = np.vstack([mof_pos, ads_pos])
    margin = 1.0
    xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    output_path = output_dir / "framework_with_CO2_1x2x2.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
