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

from src.rendering import render_structure, render_crystal, set_axis_limits_with_margin
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

    # Center all positions at origin for visualization
    all_positions = np.vstack([slab_target, ads_target])
    center = all_positions.mean(axis=0)
    centered_slab = slab_target - center
    centered_ads = ads_target - center

    # Compute noise std from target's spatial extent
    slab_std = centered_slab.std(axis=0).mean() * noise_scale

    # Sample fixed noise vectors centered at origin
    slab_noise = np.random.randn(*centered_slab.shape) * slab_std

    for i, t in enumerate(times):
        # Cosine schedule: alpha_bar goes from 0 (t=0) to 1 (t=1)
        alpha_bar = np.sin(t * np.pi / 2) ** 2

        # DDPM forward process formula
        signal_weight = np.sqrt(alpha_bar)
        noise_weight = np.sqrt(1 - alpha_bar)

        slab_traj[i] = signal_weight * centered_slab + noise_weight * slab_noise

        # Adsorbate stays fixed (conditional generation)
        if fixed_adsorbate:
            ads_traj[i] = centered_ads
        else:
            ads_std = centered_ads.std(axis=0).mean() * noise_scale if len(centered_ads) > 1 else slab_std * 0.5
            ads_noise = np.random.randn(*centered_ads.shape) * ads_std
            ads_traj[i] = signal_weight * centered_ads + noise_weight * ads_noise

    return slab_traj, ads_traj, times


def create_unconditional_trajectory(target_positions, num_steps=50, noise_scale=1.5):
    """
    Create unconditional diffusion trajectory for crystal structures.

    All atoms transition from noise to structure (no fixed condition).

    Args:
        target_positions: Target atom positions (Nx3 array)
        num_steps: Number of timesteps
        noise_scale: Multiplier for noise std relative to target std

    Returns:
        Tuple of (trajectory, times)
    """
    times = np.linspace(0, 1, num_steps + 1)
    trajectory = np.zeros((num_steps + 1, *target_positions.shape))

    # Center target positions at origin for visualization
    center = target_positions.mean(axis=0)
    centered_target = target_positions - center

    # Compute noise std from target's spatial extent
    pos_std = centered_target.std(axis=0).mean() * noise_scale

    # Sample fixed noise vector and center it at origin (same as structure)
    noise = np.random.randn(*centered_target.shape) * pos_std
    noise = noise - noise.mean(axis=0)  # Center noise at origin

    for i, t in enumerate(times):
        # Cosine schedule: alpha_bar goes from 0 (t=0) to 1 (t=1)
        alpha_bar = np.sin(t * np.pi / 2) ** 2

        # DDPM forward process formula
        signal_weight = np.sqrt(alpha_bar)
        noise_weight = np.sqrt(1 - alpha_bar)

        trajectory[i] = signal_weight * centered_target + noise_weight * noise

    return trajectory, times


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
# CRYSTAL DIFFUSION FUNCTIONS
# ============================================================================

def save_crystal_trajectory_frames(trajectory, atomic_nums, times,
                                    output_prefix, view_elev=15, view_azim=-45, dpi=200,
                                    crystal_colors=None, boundary_style=None):
    """
    Save individual PNG frames for crystal diffusion at key timesteps.

    Args:
        trajectory: Position trajectory array (T x N x 3)
        atomic_nums: Atomic numbers
        times: Time array
        output_prefix: Output file prefix
        view_elev: Camera elevation
        view_azim: Camera azimuth
        dpi: Resolution
        crystal_colors: Crystal color scheme
        boundary_style: Boundary style

    Returns:
        List of saved file paths
    """
    timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]
    step_indices = [int(t * (len(times) - 1)) for t in timesteps]

    # Use final frame positions for consistent axis limits
    final_pos = trajectory[-1]

    saved_files = []
    for t, step_idx in zip(timesteps, step_indices):
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        positions = trajectory[step_idx]
        alpha = 0.3 + 0.7 * t  # Fade in from noise to structure

        render_crystal(ax, positions, atomic_nums,
                      title=f't = {t:.2f}', alpha=alpha,
                      view_elev=view_elev, view_azim=view_azim,
                      crystal_colors=crystal_colors,
                      boundary_style=boundary_style)
        set_axis_limits_with_margin(ax, final_pos, margin=1.0)

        output_path = f"{output_prefix}_t{t:.2f}.png"
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        saved_files.append(output_path)
        print(f"Saved: {output_path}")

    return saved_files


def create_crystal_animation(trajectory, atomic_nums, times,
                              output_path=None, fps=20, pause_frames=40,
                              view_elev=15, view_azim=-45, dpi=200,
                              crystal_colors=None, boundary_style=None):
    """
    Create animation of crystal diffusion process.

    Args:
        trajectory: Position trajectory array (T x N x 3)
        atomic_nums: Atomic numbers
        times: Time array
        output_path: Output GIF path
        fps: Frames per second
        pause_frames: Number of frames to pause at end
        view_elev: Camera elevation
        view_azim: Camera azimuth
        dpi: Resolution
        crystal_colors: Crystal color scheme
        boundary_style: Boundary style

    Returns:
        Animation object
    """
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Use final frame positions for consistent axis limits
    final_pos = trajectory[-1]

    total_frames = len(times) + pause_frames

    def update(frame):
        ax.clear()
        traj_frame = min(frame, len(times) - 1)
        t = times[traj_frame]
        positions = trajectory[traj_frame]
        alpha = 0.3 + 0.7 * t

        render_crystal(ax, positions, atomic_nums,
                      title=f'Crystal Diffusion (t = {t:.2f})', alpha=alpha,
                      view_elev=view_elev, view_azim=view_azim,
                      crystal_colors=crystal_colors,
                      boundary_style=boundary_style)
        set_axis_limits_with_margin(ax, final_pos, margin=1.0)
        return ax,

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)

    if output_path:
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
        print(f"Saved: {output_path}")

    plt.close(fig)
    return anim


def run_crystal_pipeline(cfg):
    """
    Run the crystal diffusion visualization pipeline.

    Args:
        cfg: DiffusionConfig object
    """
    from ase.io import read
    from ase.build import make_supercell

    print("=" * 60)
    print(f"Crystal Diffusion Visualization - {cfg.crystal_type.upper()}")
    print("=" * 60)

    # Load structure
    atoms = read(cfg.input_file)

    # Apply supercell if specified
    if cfg.supercell_matrix is not None:
        atoms = make_supercell(atoms, cfg.supercell_matrix)

    positions = atoms.get_positions()
    atomic_nums = atoms.get_atomic_numbers()

    print(f"\nFormula: {atoms.get_chemical_formula()}")
    print(f"Total atoms: {len(positions)}")
    print(f"Crystal type: {cfg.crystal_type}")

    # Generate trajectory
    traj_cfg = cfg.trajectory_config
    print("\nGenerating unconditional diffusion trajectory...")
    trajectory, times = create_unconditional_trajectory(
        positions,
        num_steps=traj_cfg.get('num_steps', 30),
        noise_scale=traj_cfg.get('noise_scale', 1.5)
    )

    # Generate frames
    print(f"\nGenerating trajectory frames (elev={cfg.view_elev}, azim={cfg.view_azim}, dpi={cfg.dpi})...")
    output_prefix = str(cfg.figures_dir / "crystal_diffusion")

    save_crystal_trajectory_frames(
        trajectory, atomic_nums, times,
        output_prefix=output_prefix,
        view_elev=cfg.view_elev, view_azim=cfg.view_azim, dpi=cfg.dpi,
        crystal_colors=cfg.crystal_colors,
        boundary_style=cfg.boundary_style
    )

    # Generate animation if requested
    if cfg.create_gif:
        print(f"\nGenerating animation (dpi={cfg.dpi})...")
        anim_cfg = cfg.animation_config
        create_crystal_animation(
            trajectory, atomic_nums, times,
            output_path=str(cfg.figures_dir / "crystal_diffusion.gif"),
            fps=anim_cfg.get('fps', 20),
            pause_frames=anim_cfg.get('pause_frames', 40),
            view_elev=cfg.view_elev, view_azim=cfg.view_azim, dpi=cfg.dpi,
            crystal_colors=cfg.crystal_colors,
            boundary_style=cfg.boundary_style
        )

    print("\n" + "=" * 60)
    print("Crystal visualization complete!")
    print(f"Output saved to: {cfg.figures_dir}/")
    print("=" * 60)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main entry point for config-driven diffusion visualization."""
    if len(sys.argv) < 2:
        print("Usage: python src/diffusion_visualization.py <config.yaml>")
        sys.exit(1)

    from src.config import load_config, parse_diffusion_config

    config = load_config(sys.argv[1])
    cfg = parse_diffusion_config(config)

    # Setup output directory
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    np.random.seed(cfg.random_seed)

    if cfg.input_file is None:
        raise ValueError("input_file, input_files, or mode.input_file is required")

    # Dispatch to crystal pipeline for unconditional diffusion
    if cfg.mode == "crystal":
        run_crystal_pipeline(cfg)
        return

    print("=" * 60)
    print(f"Diffusion Visualization - {cfg.mode.upper()}")
    print("=" * 60)

    # Load and isolate adsorbates
    slab_pos, slab_nums, ads_pos, ads_nums, formula = load_and_isolate(
        file_path=cfg.input_file,
        method=cfg.isolation_method,
        supercell_matrix=cfg.supercell_matrix,
        isolation_config=cfg.isolation_config,
        center_adsorbate=(cfg.mode == "catalyst"),
        adsorbate_shift=cfg.adsorbate_shift,
        adsorbate_index=cfg.adsorbate_index,
        verbose=False
    )

    print(f"\nFormula: {formula}")
    print(f"Framework atoms: {len(slab_pos)}")
    print(f"Adsorbate atoms: {len(ads_pos)}")

    # Generate trajectory
    traj_cfg = cfg.trajectory_config
    fixed_adsorbate = traj_cfg.get('fixed_adsorbate', True)
    print("\nGenerating diffusion trajectory...")
    slab_traj, ads_traj, times = create_conditional_trajectory(
        slab_pos, ads_pos,
        num_steps=traj_cfg.get('num_steps', 30),
        noise_scale=traj_cfg.get('noise_scale', 1.5),
        fixed_adsorbate=fixed_adsorbate
    )

    # Generate frames
    print(f"\nGenerating trajectory frames (elev={cfg.view_elev}, azim={cfg.view_azim}, dpi={cfg.dpi})...")
    output_prefix = str(cfg.figures_dir / f"{cfg.mode}_diffusion")

    if cfg.plot_separate:
        save_separate_trajectory_frames(
            slab_traj, slab_nums, ads_traj, ads_nums, times,
            output_prefix=output_prefix,
            view_elev=cfg.view_elev, view_azim=cfg.view_azim, dpi=cfg.dpi,
            framework_colors=cfg.framework_colors,
            adsorbate_colors=cfg.adsorbate_colors,
            boundary_style=cfg.boundary_style
        )
    else:
        save_trajectory_frames(
            slab_traj, slab_nums, ads_traj, ads_nums, times,
            output_prefix=output_prefix,
            view_elev=cfg.view_elev, view_azim=cfg.view_azim, dpi=cfg.dpi,
            framework_colors=cfg.framework_colors,
            adsorbate_colors=cfg.adsorbate_colors,
            boundary_style=cfg.boundary_style
        )

    # Generate animation if requested
    if cfg.create_gif:
        print(f"\nGenerating animation (dpi={cfg.dpi})...")
        anim_cfg = cfg.animation_config
        create_animation(
            slab_traj, slab_nums, ads_traj, ads_nums, times,
            output_path=str(cfg.figures_dir / f"{cfg.mode}_diffusion.gif"),
            fps=anim_cfg.get('fps', 20),
            pause_frames=anim_cfg.get('pause_frames', 40),
            view_elev=cfg.view_elev, view_azim=cfg.view_azim, dpi=cfg.dpi,
            framework_colors=cfg.framework_colors,
            adsorbate_colors=cfg.adsorbate_colors,
            boundary_style=cfg.boundary_style
        )

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output saved to: {cfg.figures_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
