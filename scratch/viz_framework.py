"""
Visualize isolated framework from data/isolate_adsorbate/results/framework.xyz
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgb
from ase.io import read
from ase.build import make_supercell
from pathlib import Path


# Element colors
ELEMENT_COLORS = {
    1: '#F8F8F8',    # H - white
    6: '#707070',    # C - medium gray
    7: '#6B8EF0',    # N - light blue
    8: '#F06060',    # O - coral red
    26: '#E08050',   # Fe - orange-brown
}


def lighten_color(color, amount=0.5):
    rgb = to_rgb(color)
    white = (1.0, 1.0, 1.0)
    return tuple(c * (1 - amount) + w * amount for c, w in zip(rgb, white))


def darken_color(color, amount=0.3):
    rgb = to_rgb(color)
    return tuple(max(0, c * (1 - amount)) for c in rgb)


def get_element_size(atomic_number):
    base_sizes = {
        1: 90, 6: 150, 7: 140, 8: 140, 26: 180,
    }
    return base_sizes.get(atomic_number, 160) * 0.6


def setup_clean_3d_axes(ax):
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


def draw_glossy_sphere(ax, pos, atomic_num, size, alpha=1.0):
    base_color = ELEMENT_COLORS.get(int(atomic_num), '#808080')
    light_offset = np.array([-0.15, 0.15, 0.2])
    base_size = size * alpha

    # Shadow
    shadow_color = darken_color(base_color, 0.5)
    shadow_offset = pos + np.array([0.05, -0.05, -0.05]) * (size / 100)
    ax.scatter(*shadow_offset, c=[shadow_color], s=base_size * 1.1,
               alpha=alpha * 0.4, edgecolors='none', depthshade=False)

    # Main sphere
    ax.scatter(*pos, c=[base_color], s=base_size,
               edgecolors='#333333', linewidth=1.2,
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
        ax.scatter(*specular_pos, c='white', s=base_size * 0.12,
                   alpha=alpha * 0.8, edgecolors='none', depthshade=False)


def visualize_structure(ax, positions, numbers, title="", view_elev=0, view_azim=0):
    ax.clear()

    # Sort by depth
    cam_dir = np.array([
        np.cos(np.radians(view_azim)) * np.cos(np.radians(view_elev)),
        np.sin(np.radians(view_azim)) * np.cos(np.radians(view_elev)),
        np.sin(np.radians(view_elev))
    ])

    atoms_data = [(pos, num, np.dot(pos, cam_dir)) for pos, num in zip(positions, numbers)]
    atoms_data.sort(key=lambda x: x[2])

    for pos, num, _ in atoms_data:
        size = get_element_size(num)
        draw_glossy_sphere(ax, pos, num, size)

    setup_clean_3d_axes(ax)
    ax.view_init(elev=view_elev, azim=view_azim)

    if title:
        ax.set_title(title, fontsize=11, pad=10, fontweight='medium')


def main():
    framework_path = Path("data/isolate_adsorbate/results/framework.xyz")
    output_dir = Path("data/isolate_adsorbate/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading framework: {framework_path}")
    atoms = read(framework_path)
    print(f"Loaded {len(atoms)} atoms: {atoms.get_chemical_formula()}")

    # Center at origin
    positions = atoms.positions.copy()
    positions -= positions.mean(axis=0)
    numbers = atoms.numbers

    # Create visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    visualize_structure(
        ax, positions, numbers,
        title=f"Isolated Framework ({atoms.get_chemical_formula()}, elev=0, azim=0)",
        view_elev=0, view_azim=0
    )

    # Set axis limits
    margin = 1.0
    xlim = (positions[:, 0].min() - margin, positions[:, 0].max() + margin)
    ylim = (positions[:, 1].min() - margin, positions[:, 1].max() + margin)
    zlim = (positions[:, 2].min() - margin, positions[:, 2].max() + margin)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    output_path = output_dir / "framework_isolated.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
