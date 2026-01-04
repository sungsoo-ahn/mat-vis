"""
Search over periodic shifts to place CO2 inside the hexagonal pore.
Uses 1x2x2 tiling with integer lattice vector shifts for adsorbate.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgb
from ase.io import read
from ase.build import make_supercell
from pathlib import Path


ELEMENT_COLORS = {
    1: '#F8F8F8', 6: '#707070', 7: '#6B8EF0', 8: '#F06060', 26: '#E08050',
}


def get_element_size(atomic_number, is_adsorbate=False):
    base_sizes = {1: 90, 6: 150, 7: 140, 8: 140, 26: 180}
    size = base_sizes.get(atomic_number, 160)
    return size * (1.4 if is_adsorbate else 0.5)


def setup_clean_3d_axes(ax):
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
        axis.line.set_color('none')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def draw_atom(ax, pos, atomic_num, size, alpha=1.0, is_adsorbate=False):
    base_color = ELEMENT_COLORS.get(int(atomic_num), '#808080')
    ax.scatter(*pos, c=[base_color], s=size * alpha,
               edgecolors='#333333', linewidth=1.5 if is_adsorbate else 0.8,
               alpha=alpha * 0.95, depthshade=False)


def load_and_tile_with_shift(mof_path, ads_path, supercell_matrix, shift_integers):
    """Load structure with integer periodic shift applied to adsorbate.

    Args:
        shift_integers: (na, nb, nc) integer multiples of lattice vectors to shift adsorbate
    """
    mof_atoms = read(mof_path)
    ads_atoms = read(ads_path)

    # Get lattice vectors from the adsorbate file (same as MOF)
    cell = ads_atoms.get_cell()

    # Apply periodic shift to adsorbate (integer multiples of lattice vectors)
    na, nb, nc = shift_integers
    shift_cartesian = na * cell[0] + nb * cell[1] + nc * cell[2]
    ads_pos = ads_atoms.positions.copy() + shift_cartesian
    ads_nums = ads_atoms.numbers.copy()

    # Create supercell of MOF
    P = np.array(supercell_matrix)
    mof_supercell = make_supercell(mof_atoms, P)
    mof_pos = mof_supercell.positions.copy()
    mof_nums = mof_supercell.numbers.copy()

    # Center everything for visualization
    all_pos = np.vstack([mof_pos, ads_pos])
    center = all_pos.mean(axis=0)
    mof_pos -= center
    ads_pos -= center

    return mof_pos, mof_nums, ads_pos, ads_nums


def main():
    framework_path = Path("data/sample/mof_clean.xyz")
    ads_path = Path("data/sample/co2_with_cell.xyz")
    output_dir = Path("data/sample/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1x2x2 tiling
    supercell_matrix = [[1, 0, 0], [0, 2, 0], [0, 0, 2]]

    # Search over integer shifts of lattice vectors
    # For 1x2x2 supercell, valid shifts are:
    # a: 0 (only 1 cell in a direction)
    # b: 0, 1 (2 cells in b direction)
    # c: 0, 1 (2 cells in c direction)
    shift_options = []
    labels = []

    for na in [0]:  # Only 1 cell in a
        for nb in [0, 1]:  # 2 cells in b
            for nc in [0, 1]:  # 2 cells in c
                shift_options.append((na, nb, nc))
                labels.append(f"({na},{nb},{nc})")

    n_options = len(shift_options)
    n_cols = 4
    n_rows = 1

    fig = plt.figure(figsize=(16, 4))

    print(f"Testing {n_options} integer periodic shifts for 1x2x2 tiling...")
    print("Shifts are integer multiples of lattice vectors (a, b, c)")
    print("=" * 60)

    for idx, (shift, label) in enumerate(zip(shift_options, labels)):
        mof_pos, mof_nums, ads_pos, ads_nums = load_and_tile_with_shift(
            framework_path, ads_path, supercell_matrix, shift
        )

        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        # Sort by depth for rendering
        cam_dir = np.array([1, 0, 0])
        all_atoms = []
        for pos, num in zip(mof_pos, mof_nums):
            all_atoms.append((pos, num, False, np.dot(pos, cam_dir)))
        for pos, num in zip(ads_pos, ads_nums):
            all_atoms.append((pos, num, True, np.dot(pos, cam_dir)))
        all_atoms.sort(key=lambda x: x[3])

        for pos, num, is_ads, _ in all_atoms:
            size = get_element_size(num, is_ads)
            draw_atom(ax, pos, num, size, is_adsorbate=is_ads)

        setup_clean_3d_axes(ax)
        ax.view_init(elev=0, azim=0)
        ax.set_title(f"Shift: {label}", fontsize=11, fontweight='bold')

        # Set consistent limits
        all_pos_arr = np.vstack([mof_pos, ads_pos])
        margin = 1.0
        ax.set_xlim(all_pos_arr[:, 0].min() - margin, all_pos_arr[:, 0].max() + margin)
        ax.set_ylim(all_pos_arr[:, 1].min() - margin, all_pos_arr[:, 1].max() + margin)
        ax.set_zlim(all_pos_arr[:, 2].min() - margin, all_pos_arr[:, 2].max() + margin)

        print(f"  Shift {label}: CO2 center at {ads_pos.mean(axis=0)}")

    plt.suptitle('Integer Periodic Shifts (1x2x2 tiling): (na, nb, nc) multiples of lattice vectors',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    output_path = output_dir / "periodic_shift_search_integer.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
