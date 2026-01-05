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
)
from src.rendering import render_structure, set_axis_limits_with_margin
from src.isolate_adsorbate import load_and_isolate


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def create_conditional_trajectory(slab_target, ads_target, num_steps=50, noise_scale=1.5,
                                   fixed_adsorbate=True):
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
        fixed_adsorbate: If True, adsorbate stays fixed (no diffusion)

    Returns:
        Tuple of (slab_traj, ads_traj, times)
    """
    times = np.linspace(0, 1, num_steps + 1)
    slab_traj = np.zeros((num_steps + 1, *slab_target.shape))
    ads_traj = np.zeros((num_steps + 1, *ads_target.shape))

    # Compute noise std from target's spatial extent
    slab_std = slab_target.std(axis=0).mean() * noise_scale

    # Sample fixed noise vectors
    slab_noise = np.random.randn(*slab_target.shape) * slab_std

    for i, t in enumerate(times):
        # Cosine schedule: alpha_bar goes from 0 (t=0) to 1 (t=1)
        alpha_bar = np.sin(t * np.pi / 2) ** 2

        # DDPM forward process formula
        signal_weight = np.sqrt(alpha_bar)
        noise_weight = np.sqrt(1 - alpha_bar)

        slab_traj[i] = signal_weight * slab_target + noise_weight * slab_noise

        # Adsorbate stays fixed (conditional generation)
        if fixed_adsorbate:
            ads_traj[i] = ads_target
        else:
            ads_std = ads_target.std(axis=0).mean() * noise_scale if len(ads_target) > 1 else slab_std * 0.5
            ads_noise = np.random.randn(*ads_target.shape) * ads_std
            ads_traj[i] = signal_weight * ads_target + noise_weight * ads_noise

    return slab_traj, ads_traj, times


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
    timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]
    step_indices = [int(t * (len(times) - 1)) for t in timesteps]

    # Use final frame positions for consistent axis limits
    all_pos = np.vstack([slab_traj[-1], ads_traj[-1]])

    saved_files = []
    for t, step_idx in zip(timesteps, step_indices):
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        slab_pos = slab_traj[step_idx]
        ads_pos = ads_traj[step_idx]
        slab_alpha = 0.3 + 0.7 * t

        render_structure(ax, slab_pos, slab_nums, ads_pos, ads_nums,
                        title=f't = {t:.2f}', slab_alpha=slab_alpha,
                        view_elev=view_elev, view_azim=view_azim,
                        framework_colors=framework_colors,
                        adsorbate_colors=adsorbate_colors,
                        boundary_style=boundary_style)
        set_axis_limits_with_margin(ax, all_pos, margin=1.0)

        output_path = f"{output_prefix}_t{t:.2f}.png"
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        saved_files.append(output_path)
        print(f"Saved: {output_path}")

    return saved_files


def save_separate_trajectory_frames(slab_traj, slab_nums, ads_traj, ads_nums, times,
                                     output_prefix, view_elev=0, view_azim=0, dpi=200,
                                     framework_colors=None, adsorbate_colors=None,
                                     boundary_style=None):
    """
    Save separate PNG frames for framework, adsorbate, and combined at key timesteps.

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
    timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]
    step_indices = [int(t * (len(times) - 1)) for t in timesteps]

    # Use final frame positions for consistent axis limits
    all_pos = np.vstack([slab_traj[-1], ads_traj[-1]])

    empty_pos = np.array([]).reshape(0, 3)
    empty_nums = np.array([])

    saved_files = []
    for t, step_idx in zip(timesteps, step_indices):
        slab_pos = slab_traj[step_idx]
        ads_pos = ads_traj[step_idx]
        slab_alpha = 0.3 + 0.7 * t

        # 1. Framework only
        fig_fw = plt.figure(figsize=(10, 10), dpi=dpi)
        ax_fw = fig_fw.add_subplot(111, projection='3d')
        render_structure(ax_fw, slab_pos, slab_nums, empty_pos, empty_nums,
                        title=f'Framework (t = {t:.2f})', slab_alpha=slab_alpha,
                        view_elev=view_elev, view_azim=view_azim,
                        framework_colors=framework_colors,
                        adsorbate_colors=adsorbate_colors,
                        boundary_style=boundary_style)
        set_axis_limits_with_margin(ax_fw, all_pos, margin=1.0)
        fw_path = f"{output_prefix}_framework_t{t:.2f}.png"
        plt.savefig(fw_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig_fw)
        saved_files.append(fw_path)
        print(f"Saved: {fw_path}")

        # 2. Adsorbate only
        fig_ads = plt.figure(figsize=(10, 10), dpi=dpi)
        ax_ads = fig_ads.add_subplot(111, projection='3d')
        render_structure(ax_ads, empty_pos, empty_nums, ads_pos, ads_nums,
                        title=f'Adsorbate (t = {t:.2f})', slab_alpha=1.0,
                        view_elev=view_elev, view_azim=view_azim,
                        framework_colors=framework_colors,
                        adsorbate_colors=adsorbate_colors,
                        boundary_style=boundary_style)
        set_axis_limits_with_margin(ax_ads, all_pos, margin=1.0)
        ads_path = f"{output_prefix}_adsorbate_t{t:.2f}.png"
        plt.savefig(ads_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig_ads)
        saved_files.append(ads_path)
        print(f"Saved: {ads_path}")

        # 3. Combined
        fig_combined = plt.figure(figsize=(10, 10), dpi=dpi)
        ax_combined = fig_combined.add_subplot(111, projection='3d')
        render_structure(ax_combined, slab_pos, slab_nums, ads_pos, ads_nums,
                        title=f'Combined (t = {t:.2f})', slab_alpha=slab_alpha,
                        view_elev=view_elev, view_azim=view_azim,
                        framework_colors=framework_colors,
                        adsorbate_colors=adsorbate_colors,
                        boundary_style=boundary_style)
        set_axis_limits_with_margin(ax_combined, all_pos, margin=1.0)
        combined_path = f"{output_prefix}_combined_t{t:.2f}.png"
        plt.savefig(combined_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig_combined)
        saved_files.append(combined_path)
        print(f"Saved: {combined_path}")

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
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Use final frame positions for consistent axis limits
    all_pos = np.vstack([slab_traj[-1], ads_traj[-1]])

    total_frames = len(times) + pause_frames

    def update(frame):
        ax.clear()
        traj_frame = min(frame, len(times) - 1)
        t = times[traj_frame]
        slab_pos = slab_traj[traj_frame]
        ads_pos = ads_traj[traj_frame]
        slab_alpha = 0.3 + 0.7 * t

        render_structure(ax, slab_pos, slab_nums, ads_pos, ads_nums,
                        title=f'Diffusion (t = {t:.2f})', slab_alpha=slab_alpha,
                        view_elev=view_elev, view_azim=view_azim,
                        framework_colors=framework_colors,
                        adsorbate_colors=adsorbate_colors,
                        boundary_style=boundary_style)
        set_axis_limits_with_margin(ax, all_pos, margin=1.0)
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
    """Main entry point for config-driven diffusion visualization."""
    if len(sys.argv) < 2:
        print("Usage: python src/diffusion_visualization.py <config.yaml>")
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

    # Get visualization settings (shared format with structure_visualization)
    vis_config = config.get('visualization', {})

    # Mode selection
    mode = config.get('mode', 'mof')
    create_gif = config.get('create_gif', False)
    plot_separate = config.get('plot_separate', vis_config.get('plot_separate', False))
    dpi = config.get('dpi', vis_config.get('dpi', 200))

    # Get isolation settings (can be in visualization or top-level)
    isolation_method = config.get('isolation_method', vis_config.get('isolation_method', 'element'))
    isolation_config = config.get('isolation', {})

    # Get common visualization settings from visualization section
    view_elev = vis_config.get('view_elev')
    view_azim = vis_config.get('view_azim')

    # Get supercell matrix (support tiling shorthand)
    supercell_matrix = vis_config.get('supercell_matrix')
    tiling = vis_config.get('tiling')
    if tiling is not None and supercell_matrix is None:
        nx, ny, nz = tiling
        supercell_matrix = [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]

    # Get adsorbate settings from visualization section
    ads_shift = vis_config.get('adsorbate_shift')
    if ads_shift is not None:
        ads_shift = tuple(ads_shift)
    ads_index = vis_config.get('adsorbate_index')

    # Get input file (support both formats)
    input_file = config.get('input_file')
    if isinstance(config.get('input_files'), list) and len(config['input_files']) > 0:
        input_file = config['input_files'][0]

    if mode == 'mof':
        print("=" * 60)
        print("Diffusion Visualization - MOF with Adsorbate")
        print("=" * 60)

        # Override with mode-specific settings if present (backwards compatibility)
        mof_cfg = config.get('mof', {})
        if view_elev is None:
            view_elev = mof_cfg.get('view_elev', -10)
        if view_azim is None:
            view_azim = mof_cfg.get('view_azim', 0)
        if supercell_matrix is None:
            tiling = mof_cfg.get('tiling')
            if tiling is not None:
                nx, ny, nz = tiling
                supercell_matrix = [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]
            else:
                supercell_matrix = mof_cfg.get('supercell_matrix')
        if input_file is None:
            input_file = mof_cfg.get('input_file')
        if ads_shift is None:
            ads_shift = mof_cfg.get('adsorbate_shift')
            if ads_shift is not None:
                ads_shift = tuple(ads_shift)
        if ads_index is None:
            ads_index = mof_cfg.get('adsorbate_index')

        if input_file is None:
            raise ValueError("input_file, input_files, or mof.input_file is required")

        slab_pos, slab_nums, ads_pos, ads_nums, formula = load_and_isolate(
            file_path=input_file,
            method=isolation_method,
            supercell_matrix=supercell_matrix,
            isolation_config=isolation_config,
            center_adsorbate=False,
            adsorbate_shift=ads_shift,
            adsorbate_index=ads_index,
            verbose=False
        )

        print(f"\nFormula: {formula}")
        print(f"Framework atoms: {len(slab_pos)}")
        print(f"Adsorbate atoms: {len(ads_pos)}")

    elif mode == 'catalyst':
        print("=" * 60)
        print("Diffusion Visualization - Catalyst")
        print("=" * 60)

        # Override with mode-specific settings if present (backwards compatibility)
        cat_cfg = config.get('catalyst', {})
        if view_elev is None:
            view_elev = cat_cfg.get('view_elev', 35)
        if view_azim is None:
            view_azim = cat_cfg.get('view_azim', -50)
        if supercell_matrix is None:
            tiling = cat_cfg.get('tiling')
            if tiling is not None:
                nx, ny, nz = tiling
                supercell_matrix = [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]
            else:
                supercell_matrix = cat_cfg.get('supercell_matrix')
        if input_file is None:
            input_file = cat_cfg.get('input_file') or cat_cfg.get('catalyst_file')
        if ads_shift is None:
            ads_shift = cat_cfg.get('adsorbate_shift')
            if ads_shift is not None:
                ads_shift = tuple(ads_shift)
        if ads_index is None:
            ads_index = cat_cfg.get('adsorbate_index')

        if input_file is None:
            raise ValueError("input_file, input_files, or catalyst.input_file is required")

        slab_pos, slab_nums, ads_pos, ads_nums, formula = load_and_isolate(
            file_path=input_file,
            method=isolation_method,
            supercell_matrix=supercell_matrix,
            isolation_config=isolation_config,
            center_adsorbate=True,
            adsorbate_shift=ads_shift,
            adsorbate_index=ads_index,
            verbose=False
        )

        print(f"\nFormula: {formula}")
        print(f"Slab atoms: {len(slab_pos)}")
        print(f"Adsorbate atoms: {len(ads_pos)}")

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    # Generate trajectory
    traj_cfg = config.get('trajectory', {})
    fixed_adsorbate = traj_cfg.get('fixed_adsorbate', True)
    print("\nGenerating diffusion trajectory...")
    slab_traj, ads_traj, times = create_conditional_trajectory(
        slab_pos, ads_pos,
        num_steps=traj_cfg.get('num_steps', 30),
        noise_scale=traj_cfg.get('noise_scale', 1.5),
        fixed_adsorbate=fixed_adsorbate
    )

    # Generate frames
    print(f"\nGenerating trajectory frames (elev={view_elev}, azim={view_azim}, dpi={dpi})...")
    output_prefix = str(figures_dir / f"{mode}_diffusion")

    if plot_separate:
        save_separate_trajectory_frames(slab_traj, slab_nums, ads_traj, ads_nums, times,
                                        output_prefix=output_prefix,
                                        view_elev=view_elev, view_azim=view_azim, dpi=dpi,
                                        framework_colors=framework_colors,
                                        adsorbate_colors=adsorbate_colors,
                                        boundary_style=boundary_style)
    else:
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
