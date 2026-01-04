"""
Flow Matching Visualization for MOF and Catalyst structures.

Generates static comparison plots and animations showing conditional generation:
adsorbate stays fixed, framework/slab is generated around it.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from ase.io import read
from ase.build import make_supercell
from pathlib import Path


# Element colors for FRAMEWORK atoms (cool_gray scheme)
ELEMENT_COLORS_FRAMEWORK = {
    1: '#B8C4CE',    # H - cool gray
    6: '#3A4A5C',    # C - slate blue-gray
    7: '#4682B4',    # N - steel blue
    8: '#8B0000',    # O - dark red
    26: '#A0522D',   # Fe - sienna
    46: '#5F9EA0',   # Pd - cadet blue
    48: '#DAA520',   # Cd - goldenrod
    80: '#778899',   # Hg - light slate gray
}

# Element colors for ADSORBATE atoms (mint_coral scheme)
ELEMENT_COLORS_ADSORBATE = {
    1: '#FFFFFF',    # H - white
    6: '#3EB489',    # C - mint green
    7: '#5BC0EB',    # N - sky blue
    8: '#FF6B6B',    # O - coral red
    16: '#FFE66D',   # S - sunny yellow
}

# Boundary styling (thin_dark)
BOUNDARY_STYLE = {
    'edge': '#333333',
    'linewidth': 0.5,
}


def lighten_color(color, amount=0.5):
    """Lighten a color by blending with white."""
    rgb = to_rgb(color)
    return tuple(c * (1 - amount) + w * amount for c, w in zip(rgb, (1, 1, 1)))


def darken_color(color, amount=0.3):
    """Darken a color by reducing brightness."""
    rgb = to_rgb(color)
    return tuple(max(0, c * (1 - amount)) for c in rgb)


def get_element_size(atomic_number, is_adsorbate=False, size_scale=1.0):
    """Get visualization size based on atomic number using covalent radii."""
    covalent_radii = {
        1: 0.31, 6: 0.77, 7: 0.71, 8: 0.66, 16: 1.05,
        26: 1.32, 46: 1.39, 48: 1.44, 80: 1.32,
    }
    radius = covalent_radii.get(atomic_number, 1.0)
    if is_adsorbate:
        return 300 * (radius ** 2) * size_scale
    else:
        return 200 * (radius ** 2) * size_scale


def load_mof_and_adsorbate(mof_path, ads_path, supercell_matrix=None, ads_shift=(0, 0, 0)):
    """Load MOF and adsorbate from separate files."""
    mof_atoms = read(mof_path)
    ads_atoms = read(ads_path)

    mof_formula = mof_atoms.get_chemical_formula()
    ads_formula = ads_atoms.get_chemical_formula()

    # Apply periodic shift to adsorbate
    cell = ads_atoms.get_cell()
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

    # Center at origin
    all_pos = np.vstack([mof_pos, ads_pos])
    center = all_pos.mean(axis=0)
    mof_pos -= center
    ads_pos -= center

    return mof_pos, mof_nums, ads_pos, ads_nums, f"{mof_formula} + {ads_formula}"


def load_and_tile_catalyst(cif_path, supercell_matrix=None):
    """Load catalyst CIF and create supercell with adsorbate centered."""
    atoms = read(cif_path)
    formula = atoms.get_chemical_formula()

    # Adsorbate elements (small molecules)
    adsorbate_elements = {1, 6, 7, 8, 16}

    numbers = atoms.numbers
    positions = atoms.positions

    slab_mask = np.array([n not in adsorbate_elements for n in numbers])
    ads_mask = ~slab_mask

    ads_pos = positions[ads_mask].copy()
    ads_nums = numbers[ads_mask].copy()
    slab_atoms = atoms[slab_mask]

    if supercell_matrix is not None:
        P = np.array(supercell_matrix)
        slab_supercell = make_supercell(slab_atoms, P)
        slab_pos = slab_supercell.positions.copy()
        slab_nums = slab_supercell.numbers.copy()

        # Center adsorbate on supercell
        slab_center_xy = slab_pos[:, :2].mean(axis=0)
        ads_center_xy = ads_pos[:, :2].mean(axis=0)
        ads_pos[:, :2] += slab_center_xy - ads_center_xy
    else:
        slab_pos = positions[slab_mask].copy()
        slab_nums = numbers[slab_mask].copy()

    # Center at origin
    all_pos = np.vstack([slab_pos, ads_pos])
    center = all_pos.mean(axis=0)
    slab_pos -= center
    ads_pos -= center

    return slab_pos, slab_nums, ads_pos, ads_nums, formula


def create_conditional_trajectory(slab_target, ads_target, num_steps=50,
                                   noise_scale=1.5):
    """Create conditional diffusion trajectory.

    Simulates reverse diffusion: noise -> structure.
    t=0: pure noise (prior), t=1: clean structure (target)

    Uses DDPM-style formulation:
      x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    where alpha_bar uses cosine schedule.

    Args:
        noise_scale: Multiplier for noise std relative to target std (default 1.5)
    """
    times = np.linspace(0, 1, num_steps + 1)
    slab_traj = np.zeros((num_steps + 1, *slab_target.shape))
    ads_traj = np.zeros((num_steps + 1, *ads_target.shape))

    # Compute noise std from target's spatial extent (per-axis std)
    slab_std = slab_target.std(axis=0).mean() * noise_scale
    ads_std = ads_target.std(axis=0).mean() * noise_scale if len(ads_target) > 1 else slab_std * 0.5

    # Sample fixed noise vectors (same noise throughout trajectory for consistency)
    slab_noise = np.random.randn(*slab_target.shape) * slab_std
    ads_noise = np.random.randn(*ads_target.shape) * ads_std

    for i, t in enumerate(times):
        # Cosine schedule: alpha_bar goes from 0 (t=0) to 1 (t=1)
        alpha_bar = np.sin(t * np.pi / 2) ** 2

        # DDPM forward process formula (run in reverse for visualization)
        # x_t = sqrt(alpha_bar) * target + sqrt(1 - alpha_bar) * noise
        signal_weight = np.sqrt(alpha_bar)
        noise_weight = np.sqrt(1 - alpha_bar)

        slab_traj[i] = signal_weight * slab_target + noise_weight * slab_noise
        ads_traj[i] = signal_weight * ads_target + noise_weight * ads_noise

    return slab_traj, ads_traj, times


def setup_clean_3d_axes(ax):
    """Remove grids and clean up 3D axes."""
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


def draw_glossy_sphere(ax, pos, atomic_num, size, alpha=1.0, is_adsorbate=False):
    """Draw a glossy sphere with specular highlight and thin dark boundary."""
    if is_adsorbate:
        base_color = ELEMENT_COLORS_ADSORBATE.get(int(atomic_num), '#808080')
    else:
        base_color = ELEMENT_COLORS_FRAMEWORK.get(int(atomic_num), '#808080')

    light_offset = np.array([-0.15, 0.15, 0.2])
    base_size = size * alpha

    # Shadow layer
    shadow_color = darken_color(base_color, 0.5)
    shadow_offset = pos + np.array([0.05, -0.05, -0.05]) * (size / 100)
    ax.scatter(*shadow_offset, c=[shadow_color], s=base_size * 1.1,
               alpha=alpha * 0.3, edgecolors='none', depthshade=False)

    # Main sphere with thin dark boundary
    ax.scatter(*pos, c=[base_color], s=base_size,
               edgecolors=BOUNDARY_STYLE['edge'],
               linewidth=BOUNDARY_STYLE['linewidth'],
               alpha=alpha * 0.95, depthshade=False)

    # Highlight
    if alpha > 0.3:
        highlight_color = lighten_color(base_color, 0.3)
        highlight_pos = pos + light_offset * (size / 150)
        ax.scatter(*highlight_pos, c=[highlight_color], s=base_size * 0.4,
                   alpha=alpha * 0.5, edgecolors='none', depthshade=False)

    # Specular
    if alpha > 0.5:
        specular_pos = pos + light_offset * (size / 120)
        specular_size = base_size * 0.15 if is_adsorbate else base_size * 0.12
        ax.scatter(*specular_pos, c='white', s=specular_size,
                   alpha=alpha * 0.7, edgecolors='none', depthshade=False)


def visualize_structure(ax, slab_pos, slab_nums, ads_pos, ads_nums,
                        title="", slab_alpha=1.0, view_elev=25, view_azim=-60,
                        size_scale=1.0):
    """Plot catalyst structure with glossy sphere rendering."""
    ax.clear()

    # Combine and sort atoms by depth
    all_atoms = []
    for pos, num in zip(slab_pos, slab_nums):
        all_atoms.append({'pos': pos, 'num': num, 'is_adsorbate': False, 'alpha': slab_alpha})
    for pos, num in zip(ads_pos, ads_nums):
        all_atoms.append({'pos': pos, 'num': num, 'is_adsorbate': True, 'alpha': 1.0})

    cam_dir = np.array([
        np.cos(np.radians(view_azim)) * np.cos(np.radians(view_elev)),
        np.sin(np.radians(view_azim)) * np.cos(np.radians(view_elev)),
        np.sin(np.radians(view_elev))
    ])
    all_atoms.sort(key=lambda a: np.dot(a['pos'], cam_dir))

    for atom in all_atoms:
        size = get_element_size(atom['num'], atom['is_adsorbate'], size_scale=size_scale)
        draw_glossy_sphere(ax, atom['pos'], atom['num'], size,
                          alpha=atom['alpha'], is_adsorbate=atom['is_adsorbate'])

    setup_clean_3d_axes(ax)
    ax.view_init(elev=view_elev, azim=view_azim)
    if title:
        ax.set_title(title, fontsize=11, pad=10, fontweight='medium')


def plot_static_comparison(slab_target, slab_nums, ads_target, ads_nums,
                           slab_traj, ads_traj, times, output_path=None,
                           view_elev=0, view_azim=0, dpi=200):
    """Create static comparison plot at key timesteps with legend."""
    # Use square panels (4x4 each) for consistent sizing with GIF (7x7)
    # Scale factor: increased to make atoms fill more of each subplot
    size_scale = 0.3
    fig = plt.figure(figsize=(20, 4))

    timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]
    step_indices = [int(t * (len(times) - 1)) for t in timesteps]

    all_pos = np.vstack([slab_traj.reshape(-1, 3), ads_traj.reshape(-1, 3)])
    margin = -10
    xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    for idx, (t, step_idx) in enumerate(zip(timesteps, step_indices)):
        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
        slab_pos = slab_traj[step_idx]
        ads_pos = ads_traj[step_idx]
        slab_alpha = 0.3 + 0.7 * t

        visualize_structure(ax, slab_pos, slab_nums, ads_pos, ads_nums,
                           title=f't = {t:.2f}', slab_alpha=slab_alpha,
                           view_elev=view_elev, view_azim=view_azim,
                           size_scale=size_scale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    #plt.subplots_adjust(left=0.3, right=1.5, top=0.88, bottom=0.12, wspace=-0.15)
    plt.suptitle('Conditional Diffusion: Framework Generation',
                 fontsize=12, y=0.96, fontweight='medium')

    # Legend
    element_names = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 26: 'Fe', 46: 'Pd', 48: 'Cd', 80: 'Hg'}
    all_slab_elements = set(slab_nums)
    all_ads_elements = set(ads_nums)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#444444',
               markersize=0, label='Adsorbate (fixed):'),
    ]
    for elem in sorted(all_ads_elements):
        if elem in ELEMENT_COLORS_ADSORBATE:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ELEMENT_COLORS_ADSORBATE[elem],
                       markeredgecolor='#333', markersize=10,
                       label=element_names.get(elem, f'Z={elem}')))

    legend_elements.append(Line2D([0], [0], marker='', color='w', markersize=0, label='    '))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#444444',
               markersize=0, label='Framework (generated):'))

    for elem in sorted(all_slab_elements):
        if elem in ELEMENT_COLORS_FRAMEWORK:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ELEMENT_COLORS_FRAMEWORK[elem],
                       markeredgecolor='#333', markersize=10,
                       label=element_names.get(elem, f'Z={elem}')))

    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
               fontsize=9, frameon=False, handletextpad=0.3, columnspacing=0.8)

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig


def save_trajectory_frames(slab_traj, slab_nums, ads_traj, ads_nums, times,
                           output_prefix, view_elev=0, view_azim=0, dpi=200):
    """Save individual PNG frames at key timesteps (same format as GIF frames)."""
    size_scale = 0.45  # 1.5x increase from 0.3
    timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]
    step_indices = [int(t * (len(times) - 1)) for t in timesteps]

    all_pos = np.vstack([slab_traj.reshape(-1, 3), ads_traj.reshape(-1, 3)])
    margin = -10
    xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    saved_files = []
    for t, step_idx in zip(timesteps, step_indices):
        fig = plt.figure(figsize=(7, 7), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        slab_pos = slab_traj[step_idx]
        ads_pos = ads_traj[step_idx]
        slab_alpha = 0.3 + 0.7 * t

        visualize_structure(ax, slab_pos, slab_nums, ads_pos, ads_nums,
                           title=f't = {t:.2f}', slab_alpha=slab_alpha,
                           view_elev=view_elev, view_azim=view_azim,
                           size_scale=size_scale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        output_path = f"{output_prefix}_t{t:.2f}.png"
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        saved_files.append(output_path)
        print(f"Saved: {output_path}")

    return saved_files


def create_animation(slab_traj, slab_nums, ads_traj, ads_nums, times,
                     output_path=None, fps=15, pause_frames=30,
                     view_elev=0, view_azim=0, dpi=200):
    """Create animation of the diffusion process."""
    fig = plt.figure(figsize=(7, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    all_pos = np.vstack([slab_traj.reshape(-1, 3), ads_traj.reshape(-1, 3)])
    margin = -10
    xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    total_frames = len(times) + pause_frames

    def update(frame):
        ax.clear()
        traj_frame = min(frame, len(times) - 1)
        t = times[traj_frame]
        slab_pos = slab_traj[traj_frame]
        ads_pos = ads_traj[traj_frame]
        slab_alpha = 0.3 + 0.7 * t

        visualize_structure(ax, slab_pos, slab_nums, ads_pos, ads_nums,
                           title=f'Diffusion (t = {t:.2f})', slab_alpha=slab_alpha,
                           view_elev=view_elev, view_azim=view_azim,
                           size_scale=0.45)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        return ax,

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)

    if output_path:
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
        print(f"Saved: {output_path}")

    plt.close(fig)
    return anim


def main_mof(output_dir, create_gif=False, dpi=200):
    """Generate visualization for MOF with adsorbate."""
    print("=" * 60)
    print("Diffusion Visualization - MOF with CO2 Adsorbate")
    print("=" * 60)

    np.random.seed(42)

    mof_path = Path("data/sample/mof_clean.xyz")
    ads_path = Path("data/sample/co2_with_cell.xyz")

    print(f"\nLoading MOF: {mof_path}")
    print(f"Loading adsorbate: {ads_path}")

    supercell_matrix = [[1, 0, 0], [0, 2, 0], [0, 0, 2]]
    ads_shift = (0, 0, 1)

    slab_pos, slab_nums, ads_pos, ads_nums, formula = load_mof_and_adsorbate(
        mof_path, ads_path, supercell_matrix=supercell_matrix, ads_shift=ads_shift)

    print(f"  Formula: {formula}")
    print(f"  Framework atoms (1x2x2 supercell): {len(slab_pos)}")
    print(f"  Adsorbate atoms: {len(ads_pos)}")
    print(f"  Adsorbate shift: {ads_shift}")

    view_elev, view_azim = -10, 0

    print("\nGenerating diffusion trajectory...")
    slab_traj, ads_traj, times = create_conditional_trajectory(
        slab_pos, ads_pos, num_steps=60, noise_scale=1.5)

    print(f"\nGenerating trajectory frames (elev={view_elev}, azim={view_azim}, dpi={dpi})...")
    save_trajectory_frames(slab_traj, slab_nums, ads_traj, ads_nums, times,
                          output_prefix=f"{output_dir}/figures/mof_diffusion",
                          view_elev=view_elev, view_azim=view_azim, dpi=dpi)

    if create_gif:
        print(f"\nGenerating animation (dpi={dpi})...")
        create_animation(slab_traj, slab_nums, ads_traj, ads_nums, times,
                        output_path=f"{output_dir}/figures/mof_diffusion.gif",
                        fps=20, pause_frames=40, view_elev=view_elev, view_azim=view_azim,
                        dpi=dpi)

    return formula


def main_catalyst(output_dir, catalyst_id="1234", create_gif=False, dpi=200):
    """Generate visualization for catalyst structure."""
    print("=" * 60)
    print(f"Diffusion Visualization - Catalyst {catalyst_id}")
    print("=" * 60)

    np.random.seed(42)

    cif_path = Path(f"data/sample/catalyst/{catalyst_id}.cif")
    print(f"\nLoading: {cif_path}")

    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]

    slab_pos, slab_nums, ads_pos, ads_nums, formula = load_and_tile_catalyst(
        cif_path, supercell_matrix=supercell_matrix)

    print(f"  Formula: {formula}")
    print(f"  Slab atoms (2x2 supercell): {len(slab_pos)}")
    print(f"  Adsorbate atoms: {len(ads_pos)}")

    view_elev, view_azim = 35, -50

    print("\nGenerating diffusion trajectory...")
    slab_traj, ads_traj, times = create_conditional_trajectory(
        slab_pos, ads_pos, num_steps=60, noise_scale=1.5)

    print(f"\nGenerating trajectory frames (elev={view_elev}, azim={view_azim}, dpi={dpi})...")
    save_trajectory_frames(slab_traj, slab_nums, ads_traj, ads_nums, times,
                          output_prefix=f"{output_dir}/figures/catalyst_{catalyst_id}_diffusion",
                          view_elev=view_elev, view_azim=view_azim, dpi=dpi)

    if create_gif:
        print(f"\nGenerating animation (dpi={dpi})...")
        create_animation(slab_traj, slab_nums, ads_traj, ads_nums, times,
                        output_path=f"{output_dir}/figures/catalyst_{catalyst_id}_diffusion.gif",
                        fps=20, pause_frames=40, view_elev=view_elev, view_azim=view_azim,
                        dpi=dpi)

    return formula


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Diffusion Visualization")
    parser.add_argument("--mode", choices=["mof", "catalyst", "both"], default="mof",
                        help="Visualization mode")
    parser.add_argument("--catalyst-id", default="1234", help="Catalyst ID")
    parser.add_argument("--create-gif", action="store_true", help="Create GIF animation")
    parser.add_argument("--dpi", type=int, default=200, help="Resolution (default: 40)")
    args = parser.parse_args()

    output_dir = "data/flow_matching_viz"
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    outputs = []

    timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]

    if args.mode in ["mof", "both"]:
        main_mof(output_dir, create_gif=args.create_gif, dpi=args.dpi)
        for t in timesteps:
            outputs.append(f"mof_diffusion_t{t:.2f}.png")
        if args.create_gif:
            outputs.append("mof_diffusion.gif")

    if args.mode in ["catalyst", "both"]:
        if args.mode == "both":
            print("\n" + "=" * 60 + "\n")
        main_catalyst(output_dir, catalyst_id=args.catalyst_id, create_gif=args.create_gif, dpi=args.dpi)
        for t in timesteps:
            outputs.append(f"catalyst_{args.catalyst_id}_diffusion_t{t:.2f}.png")
        if args.create_gif:
            outputs.append(f"catalyst_{args.catalyst_id}_diffusion.gif")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output saved to: {output_dir}/figures/")
    for out in outputs:
        print(f"  - {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
