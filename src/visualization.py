"""
Flow Matching Visualization for MOF and Catalyst structures.

This module provides functions for visualizing conditional diffusion processes:
- Adsorbate stays fixed (condition)
- Framework/slab is generated around it

Supports both static plots and animations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from ase.io import read
from ase.build import make_supercell
import yaml

from src.utils import (
    get_framework_colors,
    get_adsorbate_colors,
    get_boundary_style,
    lighten_color,
    darken_color,
    get_element_size,
    setup_clean_3d_axes,
    get_camera_direction,
    ELEMENT_NAMES,
    ADSORBATE_ELEMENTS,
)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_mof_and_adsorbate(mof_path, ads_path, supercell_matrix=None, ads_shift=(0, 0, 0)):
    """
    Load MOF and adsorbate from separate files.

    Args:
        mof_path: Path to MOF structure file
        ads_path: Path to adsorbate structure file
        supercell_matrix: Optional 3x3 matrix for supercell creation
        ads_shift: Periodic shift (na, nb, nc) for adsorbate positioning

    Returns:
        Tuple of (mof_pos, mof_nums, ads_pos, ads_nums, formula)
    """
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

    # Create supercell if specified
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
    """
    Load catalyst CIF and create supercell with adsorbate centered.

    Args:
        cif_path: Path to catalyst CIF file
        supercell_matrix: Optional 3x3 matrix for supercell creation

    Returns:
        Tuple of (slab_pos, slab_nums, ads_pos, ads_nums, formula)
    """
    atoms = read(cif_path)
    formula = atoms.get_chemical_formula()

    numbers = atoms.numbers
    positions = atoms.positions

    # Separate slab and adsorbate
    slab_mask = np.array([n not in ADSORBATE_ELEMENTS for n in numbers])
    ads_mask = ~slab_mask

    ads_pos = positions[ads_mask].copy()
    ads_nums = numbers[ads_mask].copy()
    slab_atoms = atoms[slab_mask]

    # Create supercell if specified
    if supercell_matrix is not None:
        P = np.array(supercell_matrix)
        slab_supercell = make_supercell(slab_atoms, P)
        slab_pos = slab_supercell.positions.copy()
        slab_nums = slab_supercell.numbers.copy()

        # Center adsorbate on supercell (xy plane)
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


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def create_conditional_trajectory(slab_target, ads_target, num_steps=50, noise_scale=1.5):
    """
    Create conditional diffusion trajectory.

    Simulates reverse diffusion: noise -> structure.
    - t=0: pure noise (prior)
    - t=1: clean structure (target)

    Uses DDPM-style formulation with cosine schedule:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

    Args:
        slab_target: Target slab/framework positions
        ads_target: Target adsorbate positions
        num_steps: Number of timesteps
        noise_scale: Multiplier for noise std relative to target std

    Returns:
        Tuple of (slab_traj, ads_traj, times)
    """
    times = np.linspace(0, 1, num_steps + 1)
    slab_traj = np.zeros((num_steps + 1, *slab_target.shape))
    ads_traj = np.zeros((num_steps + 1, *ads_target.shape))

    # Compute noise std from target's spatial extent
    slab_std = slab_target.std(axis=0).mean() * noise_scale
    ads_std = ads_target.std(axis=0).mean() * noise_scale if len(ads_target) > 1 else slab_std * 0.5

    # Sample fixed noise vectors
    slab_noise = np.random.randn(*slab_target.shape) * slab_std
    ads_noise = np.random.randn(*ads_target.shape) * ads_std

    for i, t in enumerate(times):
        # Cosine schedule: alpha_bar goes from 0 (t=0) to 1 (t=1)
        alpha_bar = np.sin(t * np.pi / 2) ** 2

        # DDPM forward process formula
        signal_weight = np.sqrt(alpha_bar)
        noise_weight = np.sqrt(1 - alpha_bar)

        slab_traj[i] = signal_weight * slab_target + noise_weight * slab_noise
        ads_traj[i] = signal_weight * ads_target + noise_weight * ads_noise

    return slab_traj, ads_traj, times


# ============================================================================
# RENDERING
# ============================================================================

def draw_glossy_sphere(ax, pos, atomic_num, size, alpha=1.0, is_adsorbate=False,
                       framework_colors=None, adsorbate_colors=None, boundary_style=None):
    """
    Draw a glossy sphere with specular highlight and boundary.

    Args:
        ax: Matplotlib 3D axes
        pos: Position (x, y, z)
        atomic_num: Atomic number
        size: Sphere size
        alpha: Transparency
        is_adsorbate: Whether atom is part of adsorbate
        framework_colors: Color scheme for framework atoms
        adsorbate_colors: Color scheme for adsorbate atoms
        boundary_style: Boundary style dict
    """
    # Get colors from schemes
    if is_adsorbate:
        base_color = adsorbate_colors.get(int(atomic_num), '#808080')
    else:
        base_color = framework_colors.get(int(atomic_num), '#808080')

    # Determine edge color
    edge = boundary_style['edge']
    linewidth = boundary_style['linewidth']

    if edge == 'auto':
        edge_color = darken_color(base_color, 0.5)
    elif edge == 'none':
        edge_color = 'none'
    else:
        edge_color = edge

    light_offset = np.array([-0.15, 0.15, 0.2])
    base_size = size * alpha

    # Shadow layer
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
                        size_scale=1.0, framework_colors=None, adsorbate_colors=None,
                        boundary_style=None):
    """
    Plot structure with glossy sphere rendering.

    Args:
        ax: Matplotlib 3D axes
        slab_pos: Slab/framework positions
        slab_nums: Slab/framework atomic numbers
        ads_pos: Adsorbate positions
        ads_nums: Adsorbate atomic numbers
        title: Plot title
        slab_alpha: Transparency for slab atoms
        view_elev: Camera elevation angle
        view_azim: Camera azimuth angle
        size_scale: Size scaling factor
        framework_colors: Framework color scheme
        adsorbate_colors: Adsorbate color scheme
        boundary_style: Boundary style dict
    """
    ax.clear()

    # Combine and sort atoms by depth
    all_atoms = []
    for pos, num in zip(slab_pos, slab_nums):
        all_atoms.append({'pos': pos, 'num': num, 'is_adsorbate': False, 'alpha': slab_alpha})
    for pos, num in zip(ads_pos, ads_nums):
        all_atoms.append({'pos': pos, 'num': num, 'is_adsorbate': True, 'alpha': 1.0})

    cam_dir = get_camera_direction(view_elev, view_azim)
    all_atoms.sort(key=lambda a: np.dot(a['pos'], cam_dir))

    for atom in all_atoms:
        size = get_element_size(atom['num'], atom['is_adsorbate'], size_scale=size_scale)
        draw_glossy_sphere(ax, atom['pos'], atom['num'], size,
                          alpha=atom['alpha'], is_adsorbate=atom['is_adsorbate'],
                          framework_colors=framework_colors,
                          adsorbate_colors=adsorbate_colors,
                          boundary_style=boundary_style)

    setup_clean_3d_axes(ax)
    ax.view_init(elev=view_elev, azim=view_azim)
    if title:
        ax.set_title(title, fontsize=11, pad=10, fontweight='medium')


# ============================================================================
# PLOT GENERATION
# ============================================================================

def save_trajectory_frames(slab_traj, slab_nums, ads_traj, ads_nums, times,
                           output_prefix, view_elev=0, view_azim=0, dpi=200,
                           framework_colors=None, adsorbate_colors=None,
                           boundary_style=None):
    """
    Save individual PNG frames at key timesteps.

    Args:
        slab_traj: Slab trajectory array
        slab_nums: Slab atomic numbers
        ads_traj: Adsorbate trajectory array
        ads_nums: Adsorbate atomic numbers
        times: Time array
        output_prefix: Output file prefix
        view_elev: Camera elevation
        view_azim: Camera azimuth
        dpi: Resolution
        framework_colors: Framework color scheme
        adsorbate_colors: Adsorbate color scheme
        boundary_style: Boundary style

    Returns:
        List of saved file paths
    """
    size_scale = 0.45
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
                           size_scale=size_scale,
                           framework_colors=framework_colors,
                           adsorbate_colors=adsorbate_colors,
                           boundary_style=boundary_style)
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
                     view_elev=0, view_azim=0, dpi=200,
                     framework_colors=None, adsorbate_colors=None,
                     boundary_style=None):
    """
    Create animation of the diffusion process.

    Args:
        slab_traj: Slab trajectory array
        slab_nums: Slab atomic numbers
        ads_traj: Adsorbate trajectory array
        ads_nums: Adsorbate atomic numbers
        times: Time array
        output_path: Output GIF path
        fps: Frames per second
        pause_frames: Number of frames to pause at end
        view_elev: Camera elevation
        view_azim: Camera azimuth
        dpi: Resolution
        framework_colors: Framework color scheme
        adsorbate_colors: Adsorbate color scheme
        boundary_style: Boundary style

    Returns:
        Animation object
    """
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
                           size_scale=0.45,
                           framework_colors=framework_colors,
                           adsorbate_colors=adsorbate_colors,
                           boundary_style=boundary_style)
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


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main entry point for config-driven visualization."""
    if len(sys.argv) < 2:
        print("Usage: python src/visualization.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup output directory
    output_dir = Path(config['output_dir'])
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Get color schemes
    framework_colors = get_framework_colors(config.get('framework_scheme'))
    adsorbate_colors = get_adsorbate_colors(config.get('adsorbate_scheme'))
    boundary_style = get_boundary_style(config.get('boundary_scheme'))

    # Set random seed
    np.random.seed(config.get('random_seed', 42))

    # Mode selection
    mode = config.get('mode', 'mof')
    create_gif = config.get('create_gif', False)
    dpi = config.get('dpi', 200)

    if mode == 'mof':
        print("=" * 60)
        print("Diffusion Visualization - MOF with Adsorbate")
        print("=" * 60)

        # Load data
        mof_cfg = config['mof']
        slab_pos, slab_nums, ads_pos, ads_nums, formula = load_mof_and_adsorbate(
            mof_path=mof_cfg['framework_file'],
            ads_path=mof_cfg['adsorbate_file'],
            supercell_matrix=mof_cfg.get('supercell_matrix'),
            ads_shift=tuple(mof_cfg.get('adsorbate_shift', [0, 0, 0]))
        )

        print(f"\nFormula: {formula}")
        print(f"Framework atoms: {len(slab_pos)}")
        print(f"Adsorbate atoms: {len(ads_pos)}")

        view_elev = mof_cfg.get('view_elev', -10)
        view_azim = mof_cfg.get('view_azim', 0)

    elif mode == 'catalyst':
        print("=" * 60)
        print("Diffusion Visualization - Catalyst")
        print("=" * 60)

        # Load data
        cat_cfg = config['catalyst']
        slab_pos, slab_nums, ads_pos, ads_nums, formula = load_and_tile_catalyst(
            cif_path=cat_cfg['catalyst_file'],
            supercell_matrix=cat_cfg.get('supercell_matrix')
        )

        print(f"\nFormula: {formula}")
        print(f"Slab atoms: {len(slab_pos)}")
        print(f"Adsorbate atoms: {len(ads_pos)}")

        view_elev = cat_cfg.get('view_elev', 35)
        view_azim = cat_cfg.get('view_azim', -50)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    # Generate trajectory
    traj_cfg = config.get('trajectory', {})
    print("\nGenerating diffusion trajectory...")
    slab_traj, ads_traj, times = create_conditional_trajectory(
        slab_pos, ads_pos,
        num_steps=traj_cfg.get('num_steps', 60),
        noise_scale=traj_cfg.get('noise_scale', 1.5)
    )

    # Generate frames
    print(f"\nGenerating trajectory frames (elev={view_elev}, azim={view_azim}, dpi={dpi})...")
    output_prefix = str(figures_dir / f"{mode}_diffusion")
    save_trajectory_frames(slab_traj, slab_nums, ads_traj, ads_nums, times,
                          output_prefix=output_prefix,
                          view_elev=view_elev, view_azim=view_azim, dpi=dpi,
                          framework_colors=framework_colors,
                          adsorbate_colors=adsorbate_colors,
                          boundary_style=boundary_style)

    # Generate animation if requested
    if create_gif:
        print(f"\nGenerating animation (dpi={dpi})...")
        anim_cfg = config.get('animation', {})
        create_animation(slab_traj, slab_nums, ads_traj, ads_nums, times,
                        output_path=str(figures_dir / f"{mode}_diffusion.gif"),
                        fps=anim_cfg.get('fps', 20),
                        pause_frames=anim_cfg.get('pause_frames', 40),
                        view_elev=view_elev, view_azim=view_azim, dpi=dpi,
                        framework_colors=framework_colors,
                        adsorbate_colors=adsorbate_colors,
                        boundary_style=boundary_style)

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output saved to: {figures_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
