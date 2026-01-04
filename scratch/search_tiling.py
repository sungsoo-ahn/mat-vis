"""
Search over different tiling directions to find one that centers CO2.
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


def lighten_color(color, amount=0.5):
    rgb = to_rgb(color)
    return tuple(c * (1 - amount) + w * amount for c, w in zip(rgb, (1, 1, 1)))


def darken_color(color, amount=0.3):
    rgb = to_rgb(color)
    return tuple(max(0, c * (1 - amount)) for c in rgb)


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


def load_and_tile(mof_path, ads_path, supercell_matrix):
    mof_atoms = read(mof_path)
    ads_atoms = read(ads_path)

    ads_pos = ads_atoms.positions.copy()
    ads_nums = ads_atoms.numbers.copy()

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


def compute_centrality(mof_pos, ads_pos):
    """Compute how centered the adsorbate is within the framework."""
    mof_min = mof_pos.min(axis=0)
    mof_max = mof_pos.max(axis=0)
    mof_center = (mof_min + mof_max) / 2
    ads_center = ads_pos.mean(axis=0)

    # Distance from adsorbate center to framework center
    distance = np.linalg.norm(ads_center - mof_center)

    # Normalized by framework size
    mof_size = np.linalg.norm(mof_max - mof_min)
    normalized_distance = distance / mof_size if mof_size > 0 else distance

    return normalized_distance


def main():
    framework_path = Path("data/sample/mof_clean.xyz")
    ads_path = Path("data/sample/co2_with_cell.xyz")
    output_dir = Path("data/sample/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Different tiling options to search
    tiling_options = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 1x1x1
        [[2, 0, 0], [0, 1, 0], [0, 0, 1]],  # 2x1x1
        [[1, 0, 0], [0, 2, 0], [0, 0, 1]],  # 1x2x1
        [[1, 0, 0], [0, 1, 0], [0, 0, 2]],  # 1x1x2
        [[2, 0, 0], [0, 2, 0], [0, 0, 1]],  # 2x2x1
        [[2, 0, 0], [0, 1, 0], [0, 0, 2]],  # 2x1x2
        [[1, 0, 0], [0, 2, 0], [0, 0, 2]],  # 1x2x2
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]],  # 2x2x2
        [[3, 0, 0], [0, 1, 0], [0, 0, 1]],  # 3x1x1
        [[1, 0, 0], [0, 3, 0], [0, 0, 1]],  # 1x3x1
        [[1, 0, 0], [0, 1, 0], [0, 0, 3]],  # 1x1x3
        [[3, 0, 0], [0, 2, 0], [0, 0, 1]],  # 3x2x1
    ]

    labels = ['1x1x1', '2x1x1', '1x2x1', '1x1x2', '2x2x1', '2x1x2',
              '1x2x2', '2x2x2', '3x1x1', '1x3x1', '1x1x3', '3x2x1']

    # Compute centrality for each tiling
    centralities = []
    for tiling in tiling_options:
        mof_pos, mof_nums, ads_pos, ads_nums = load_and_tile(
            framework_path, ads_path, tiling
        )
        centrality = compute_centrality(mof_pos, ads_pos)
        centralities.append(centrality)

    # Sort by centrality (lower is better - more centered)
    sorted_indices = np.argsort(centralities)

    print("Tiling options sorted by CO2 centrality (best first):")
    print("-" * 50)
    for i, idx in enumerate(sorted_indices):
        print(f"{i+1}. {labels[idx]:8s} - centrality score: {centralities[idx]:.4f}")

    # Create comparison figure
    n_cols = 4
    n_rows = 3
    fig = plt.figure(figsize=(16, 12))

    for plot_idx, idx in enumerate(sorted_indices):
        tiling = tiling_options[idx]
        label = labels[idx]

        mof_pos, mof_nums, ads_pos, ads_nums = load_and_tile(
            framework_path, ads_path, tiling
        )

        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection='3d')

        # Sort by depth
        cam_dir = np.array([1, 0, 0])  # azim=0, elev=0
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

        rank = plot_idx + 1
        ax.set_title(f"#{rank}: {label}\n(score: {centralities[idx]:.3f})",
                    fontsize=10, fontweight='bold' if rank <= 3 else 'normal')

        # Set consistent limits
        all_pos = np.vstack([mof_pos, ads_pos])
        margin = 1.0
        ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
        ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
        ax.set_zlim(all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    plt.suptitle('Tiling Search: CO2 Centrality (lower score = more centered)',
                 fontsize=14, y=0.98)
    plt.tight_layout()

    output_path = output_dir / "tiling_search_centrality.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")

    # Print recommendation
    best_idx = sorted_indices[0]
    print(f"\nRecommendation: Use {labels[best_idx]} tiling for best CO2 centering")


if __name__ == "__main__":
    main()
