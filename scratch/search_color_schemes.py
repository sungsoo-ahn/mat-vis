"""
Search over different color and boundary schemes for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from ase.io import read
from ase.build import make_supercell
from pathlib import Path


# Define multiple color schemes to test

# Framework color schemes (muted, darker tones)
FRAMEWORK_SCHEMES = {
    'dark_muted': {
        1: '#C0C0C0', 6: '#404040', 7: '#4060A0', 8: '#A03030',
        26: '#B06020', 46: '#308080', 48: '#A09030', 80: '#808090',
    },
    'earthy': {
        1: '#D4C4B0', 6: '#5C4033', 7: '#4A6741', 8: '#8B4513',
        26: '#CD853F', 46: '#708090', 48: '#B8860B', 80: '#696969',
    },
    'cool_gray': {
        1: '#B8C4CE', 6: '#3A4A5C', 7: '#4682B4', 8: '#8B0000',
        26: '#A0522D', 46: '#5F9EA0', 48: '#DAA520', 80: '#778899',
    },
    'warm_brown': {
        1: '#D2B48C', 6: '#3E2723', 7: '#1565C0', 8: '#B71C1C',
        26: '#E65100', 46: '#00695C', 48: '#F9A825', 80: '#455A64',
    },
    'slate': {
        1: '#A9A9A9', 6: '#2F4F4F', 7: '#483D8B', 8: '#800000',
        26: '#8B4513', 46: '#2E8B57', 48: '#BDB76B', 80: '#708090',
    },
    'charcoal': {
        1: '#909090', 6: '#1C1C1C', 7: '#191970', 8: '#660000',
        26: '#8B4000', 46: '#006666', 48: '#9B870C', 80: '#4A4A4A',
    },
}

# Adsorbate color schemes (bright, high contrast)
# Default: neon - searching for pleasant alternatives that remain distinguishable
ADSORBATE_SCHEMES = {
    'neon': {  # Default baseline
        1: '#FFFFFF', 6: '#00FF80', 7: '#00FFFF', 8: '#FF00FF', 16: '#FFFF00',
    },
    'soft_neon': {  # Softer version of neon
        1: '#FFFFFF', 6: '#50E090', 7: '#60D0E0', 8: '#E060D0', 16: '#E0E050',
    },
    'mint_coral': {  # Mint green + coral tones
        1: '#FFFFFF', 6: '#3EB489', 7: '#5BC0EB', 8: '#FF6B6B', 16: '#FFE66D',
    },
    'ocean_sunset': {  # Ocean blues + sunset warm
        1: '#FFFFFF', 6: '#2ECC71', 7: '#3498DB', 8: '#E74C3C', 16: '#F39C12',
    },
    'spring': {  # Fresh spring palette
        1: '#FFFFFF', 6: '#00D084', 7: '#00B4D8', 8: '#FF5C8D', 16: '#FFD93D',
    },
    'arctic': {  # Cool arctic tones + warm accent
        1: '#FFFFFF', 6: '#48CAE4', 7: '#90E0EF', 8: '#F72585', 16: '#FFBE0B',
    },
    'forest_berry': {  # Natural greens + berry accents
        1: '#FFFFFF', 6: '#52B788', 7: '#74C69D', 8: '#D64161', 16: '#FFB703',
    },
    'tech': {  # Modern tech palette
        1: '#FFFFFF', 6: '#00F5D4', 7: '#00BBF9', 8: '#F15BB5', 16: '#FEE440',
    },
}

# Boundary schemes
BOUNDARY_SCHEMES = {
    'none': {'edge': 'none', 'linewidth': 0},
    'thin_dark': {'edge': '#333333', 'linewidth': 0.5},
    'thin_black': {'edge': '#000000', 'linewidth': 0.5},
    'medium_dark': {'edge': '#333333', 'linewidth': 1.0},
    'thick_dark': {'edge': '#222222', 'linewidth': 1.5},
    'colored': {'edge': 'auto', 'linewidth': 0.8},  # Use darker version of atom color
}


def lighten_color(color, amount=0.5):
    rgb = to_rgb(color)
    return tuple(c * (1 - amount) + w * amount for c, w in zip(rgb, (1, 1, 1)))


def darken_color(color, amount=0.3):
    rgb = to_rgb(color)
    return tuple(max(0, c * (1 - amount)) for c in rgb)


def get_element_size(atomic_number, is_adsorbate=False):
    covalent_radii = {
        1: 0.31, 6: 0.77, 7: 0.71, 8: 0.66, 16: 1.05,
        26: 1.32, 46: 1.39, 48: 1.44, 80: 1.32,
    }
    radius = covalent_radii.get(atomic_number, 1.0)
    if is_adsorbate:
        return 300 * (radius ** 2)
    else:
        return 80 * (radius ** 2)


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


def draw_atom(ax, pos, atomic_num, size, alpha, is_adsorbate,
              framework_colors, adsorbate_colors, boundary_scheme):
    """Draw atom with specified color and boundary scheme."""
    if is_adsorbate:
        base_color = adsorbate_colors.get(int(atomic_num), '#808080')
    else:
        base_color = framework_colors.get(int(atomic_num), '#808080')

    base_size = size * alpha

    # Determine edge color
    edge = boundary_scheme['edge']
    linewidth = boundary_scheme['linewidth']

    if edge == 'auto':
        edge_color = darken_color(base_color, 0.5)
    elif edge == 'none':
        edge_color = 'none'
    else:
        edge_color = edge

    # Shadow
    shadow_color = darken_color(base_color, 0.5)
    shadow_offset = pos + np.array([0.05, -0.05, -0.05]) * (size / 100)
    ax.scatter(*shadow_offset, c=[shadow_color], s=base_size * 1.1,
               alpha=alpha * 0.3, edgecolors='none', depthshade=False)

    # Main sphere
    ax.scatter(*pos, c=[base_color], s=base_size,
               edgecolors=edge_color, linewidth=linewidth,
               alpha=alpha * 0.95, depthshade=False)

    # Highlight
    if alpha > 0.3:
        light_offset = np.array([-0.15, 0.15, 0.2])
        highlight_color = lighten_color(base_color, 0.3)
        highlight_pos = pos + light_offset * (size / 150)
        ax.scatter(*highlight_pos, c=[highlight_color], s=base_size * 0.4,
                   alpha=alpha * 0.5, edgecolors='none', depthshade=False)


def visualize_with_scheme(ax, slab_pos, slab_nums, ads_pos, ads_nums,
                          framework_colors, adsorbate_colors, boundary_scheme,
                          view_elev, view_azim, title=""):
    """Visualize structure with specified color/boundary scheme."""
    ax.clear()

    all_atoms = []
    for pos, num in zip(slab_pos, slab_nums):
        all_atoms.append({'pos': pos, 'num': num, 'is_adsorbate': False, 'alpha': 1.0})
    for pos, num in zip(ads_pos, ads_nums):
        all_atoms.append({'pos': pos, 'num': num, 'is_adsorbate': True, 'alpha': 1.0})

    cam_dir = np.array([
        np.cos(np.radians(view_azim)) * np.cos(np.radians(view_elev)),
        np.sin(np.radians(view_azim)) * np.cos(np.radians(view_elev)),
        np.sin(np.radians(view_elev))
    ])
    all_atoms.sort(key=lambda a: np.dot(a['pos'], cam_dir))

    for atom in all_atoms:
        size = get_element_size(atom['num'], atom['is_adsorbate'])
        draw_atom(ax, atom['pos'], atom['num'], size, atom['alpha'],
                  atom['is_adsorbate'], framework_colors, adsorbate_colors, boundary_scheme)

    setup_clean_3d_axes(ax)
    ax.view_init(elev=view_elev, azim=view_azim)
    if title:
        ax.set_title(title, fontsize=8, pad=2)


def load_mof_data():
    """Load MOF data for testing."""
    mof_path = Path("data/sample/mof_clean.xyz")
    ads_path = Path("data/sample/co2_with_cell.xyz")

    mof_atoms = read(mof_path)
    ads_atoms = read(ads_path)

    cell = ads_atoms.get_cell()
    ads_shift = (0, 0, 1)
    shift_cartesian = ads_shift[0] * cell[0] + ads_shift[1] * cell[1] + ads_shift[2] * cell[2]
    ads_pos = ads_atoms.positions.copy() + shift_cartesian
    ads_nums = ads_atoms.numbers.copy()

    supercell_matrix = [[1, 0, 0], [0, 2, 0], [0, 0, 2]]
    P = np.array(supercell_matrix)
    mof_supercell = make_supercell(mof_atoms, P)
    mof_pos = mof_supercell.positions.copy()
    mof_nums = mof_supercell.numbers.copy()

    all_pos = np.vstack([mof_pos, ads_pos])
    center = all_pos.mean(axis=0)
    mof_pos -= center
    ads_pos -= center

    return mof_pos, mof_nums, ads_pos, ads_nums


def load_catalyst_data():
    """Load catalyst data for testing."""
    cif_path = Path("data/sample/catalyst/1234.cif")
    atoms = read(cif_path)

    adsorbate_elements = {1, 6, 7, 8, 16}
    numbers = atoms.numbers
    positions = atoms.positions

    slab_mask = np.array([n not in adsorbate_elements for n in numbers])
    ads_mask = ~slab_mask

    ads_pos = positions[ads_mask].copy()
    ads_nums = numbers[ads_mask].copy()
    slab_atoms = atoms[slab_mask]

    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    P = np.array(supercell_matrix)
    slab_supercell = make_supercell(slab_atoms, P)
    slab_pos = slab_supercell.positions.copy()
    slab_nums = slab_supercell.numbers.copy()

    slab_center_xy = slab_pos[:, :2].mean(axis=0)
    ads_center_xy = ads_pos[:, :2].mean(axis=0)
    ads_pos[:, :2] += slab_center_xy - ads_center_xy

    all_pos = np.vstack([slab_pos, ads_pos])
    center = all_pos.mean(axis=0)
    slab_pos -= center
    ads_pos -= center

    return slab_pos, slab_nums, ads_pos, ads_nums


def main():
    output_dir = Path("data/sample/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading MOF data...")
    mof_pos, mof_nums, mof_ads_pos, mof_ads_nums = load_mof_data()

    print("Loading catalyst data...")
    cat_pos, cat_nums, cat_ads_pos, cat_ads_nums = load_catalyst_data()

    # Fixed defaults: cool_gray framework + thin_dark boundary
    # Only search over adsorbate color schemes
    framework_scheme = FRAMEWORK_SCHEMES['cool_gray']
    boundary_scheme = BOUNDARY_SCHEMES['thin_dark']
    adsorbate_names = list(ADSORBATE_SCHEMES.keys())
    n_ads = len(adsorbate_names)

    # Create comparison grid for MOF - adsorbate colors only
    print("\nGenerating MOF adsorbate color comparison (cool_gray + thin_dark base)...")
    n_cols = 4
    n_rows = (n_ads + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))

    for idx, ads_name in enumerate(adsorbate_names):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        visualize_with_scheme(
            ax, mof_pos, mof_nums, mof_ads_pos, mof_ads_nums,
            framework_scheme, ADSORBATE_SCHEMES[ads_name],
            boundary_scheme,
            view_elev=-10, view_azim=0,
            title=ads_name
        )

        all_pos = np.vstack([mof_pos, mof_ads_pos])
        margin = 1.0
        ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
        ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
        ax.set_zlim(all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    plt.suptitle('MOF: Adsorbate Color Search (cool_gray + thin_dark)', fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "adsorbate_color_search_mof.png", dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'adsorbate_color_search_mof.png'}")
    plt.close()

    # Create comparison grid for Catalyst - adsorbate colors only
    print("\nGenerating Catalyst adsorbate color comparison (cool_gray + thin_dark base)...")
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))

    for idx, ads_name in enumerate(adsorbate_names):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        visualize_with_scheme(
            ax, cat_pos, cat_nums, cat_ads_pos, cat_ads_nums,
            framework_scheme, ADSORBATE_SCHEMES[ads_name],
            boundary_scheme,
            view_elev=35, view_azim=-50,
            title=ads_name
        )

        all_pos = np.vstack([cat_pos, cat_ads_pos])
        margin = 1.0
        ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
        ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
        ax.set_zlim(all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    plt.suptitle('Catalyst: Adsorbate Color Search (cool_gray + thin_dark)', fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "adsorbate_color_search_catalyst.png", dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'adsorbate_color_search_catalyst.png'}")
    plt.close()

    print("\nDone! Review the images to select the best adsorbate color scheme.")
    print("Default: neon | Alternatives: soft_neon, mint_coral, ocean_sunset, spring, arctic, forest_berry, tech")


if __name__ == "__main__":
    main()
