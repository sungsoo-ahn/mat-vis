"""
Improved Flow Matching Visualization for Catalyst 1234.

Shows conditional generation: adsorbate stays fixed, slab is generated around it.
- Clean visualization without grids
- Tiled supercell so adsorbate is centered
- Adsorbate slightly perturbed, slab from noise
- Glossy sphere rendering with specular highlights
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgb, to_hex
import colorsys
from ase.io import read
from ase.build import make_supercell
from ase.neighborlist import neighbor_list
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pathlib import Path


# Element colors for FRAMEWORK atoms (muted, darker tones)
ELEMENT_COLORS_FRAMEWORK = {
    1: '#C0C0C0',    # H - gray
    6: '#404040',    # C - dark gray (framework carbon)
    7: '#4060A0',    # N - dark blue
    8: '#A03030',    # O - dark red (framework oxygen)
    26: '#B06020',   # Fe - brown (darker, no clash with adsorbate)
    46: '#308080',   # Pd - dark teal
    48: '#A09030',   # Cd - dark gold
    80: '#808090',   # Hg - dark silver
}

# Element colors for ADSORBATE atoms (bright, high contrast)
ELEMENT_COLORS_ADSORBATE = {
    1: '#FFFFFF',    # H - white
    6: '#00FF80',    # C - bright green/cyan (adsorbate carbon)
    7: '#00FFFF',    # N - cyan
    8: '#FF00FF',    # O - magenta (adsorbate oxygen) - high contrast
    16: '#FFFF00',   # S - bright yellow
}

# Combined for backward compatibility
ELEMENT_COLORS = ELEMENT_COLORS_FRAMEWORK.copy()

# Adsorbate elements (small molecules) - fallback if connectivity fails
ADSORBATE_ELEMENTS = {1, 6, 7, 8, 16}  # H, C, N, O, S

# Covalent radii for bond detection (in Angstroms)
COVALENT_RADII = {
    1: 0.31,   # H
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    16: 1.05,  # S
    26: 1.32,  # Fe
    46: 1.39,  # Pd
    48: 1.44,  # Cd
    80: 1.32,  # Hg
}


def find_connected_components(atoms, cutoff_scale=1.3):
    """Find connected components in an atomic structure.

    Uses neighbor lists based on covalent radii to build a connectivity graph,
    then finds connected components. The largest component is assumed to be
    the framework, smaller components are adsorbates.

    Args:
        atoms: ASE Atoms object
        cutoff_scale: Multiplier for sum of covalent radii to determine bonds

    Returns:
        framework_mask: Boolean array, True for framework atoms
        adsorbate_mask: Boolean array, True for adsorbate atoms
        n_components: Number of connected components
        component_labels: Array of component labels for each atom
    """
    from ase.neighborlist import natural_cutoffs, NeighborList

    n_atoms = len(atoms)
    numbers = atoms.numbers

    # Use ASE's natural cutoffs (based on covalent radii)
    cutoffs = natural_cutoffs(atoms, mult=cutoff_scale)

    # Build neighbor list with periodic boundary conditions
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    # Build adjacency list
    valid_bonds = set()
    for i in range(n_atoms):
        indices, offsets = nl.get_neighbors(i)
        for j in indices:
            if i < j:
                valid_bonds.add((i, j))

    # Build sparse adjacency matrix
    if valid_bonds:
        rows = [b[0] for b in valid_bonds] + [b[1] for b in valid_bonds]
        cols = [b[1] for b in valid_bonds] + [b[0] for b in valid_bonds]
        data = [1] * len(rows)
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_atoms, n_atoms))
    else:
        adj_matrix = csr_matrix((n_atoms, n_atoms))

    # Find connected components
    n_components, component_labels = connected_components(adj_matrix, directed=False)

    # Find the largest component (framework)
    component_sizes = np.bincount(component_labels)
    largest_component = np.argmax(component_sizes)

    framework_mask = component_labels == largest_component
    adsorbate_mask = ~framework_mask

    # Print component info (summarized)
    print(f"  Found {n_components} connected components:")
    for comp_id in range(min(n_components, 10)):  # Show first 10
        comp_size = component_sizes[comp_id]
        comp_atoms = atoms[component_labels == comp_id]
        comp_formula = comp_atoms.get_chemical_formula()
        label = "FRAMEWORK" if comp_id == largest_component else "ADSORBATE"
        print(f"    Component {comp_id}: {comp_formula} ({comp_size} atoms) - {label}")
    if n_components > 10:
        print(f"    ... and {n_components - 10} more components")

    return framework_mask, adsorbate_mask, n_components, component_labels


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
    """Get visualization size based on atomic number.

    Uses covalent radii for non-linear scaling that reflects actual atomic sizes.
    """
    # Covalent radii in Angstroms (for non-linear, physically-based scaling)
    covalent_radii = {
        1: 0.31,   # H
        6: 0.77,   # C
        7: 0.71,   # N
        8: 0.66,   # O  (smaller than C)
        16: 1.05,  # S
        26: 1.32,  # Fe
        46: 1.39,  # Pd
        48: 1.44,  # Cd
        80: 1.32,  # Hg
    }

    radius = covalent_radii.get(atomic_number, 1.0)

    if is_adsorbate:
        # Scale based on covalent radius squared (area scaling)
        # Base size 300 for radius 1.0
        size = 300 * (radius ** 2)
    else:
        # Framework atoms: smaller, also radius-based
        size = 80 * (radius ** 2)

    return size


def load_mof_and_adsorbate(mof_path, ads_path, supercell_matrix=None, ads_shift=(0, 0, 0)):
    """Load MOF and adsorbate from separate files.

    Args:
        mof_path: Path to MOF/framework file
        ads_path: Path to adsorbate file
        supercell_matrix: 3x3 matrix for MOF supercell
        ads_shift: (na, nb, nc) integer multiples of lattice vectors to shift adsorbate

    Returns:
        mof_pos, mof_nums, ads_pos, ads_nums, formula
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
        # Create supercell of MOF
        P = np.array(supercell_matrix)
        mof_supercell = make_supercell(mof_atoms, P)
        mof_pos = mof_supercell.positions.copy()
        mof_nums = mof_supercell.numbers.copy()
    else:
        mof_pos = mof_atoms.positions.copy()
        mof_nums = mof_atoms.numbers.copy()

    # Center the entire complex at origin
    all_pos = np.vstack([mof_pos, ads_pos])
    center = all_pos.mean(axis=0)
    mof_pos -= center
    ads_pos -= center

    return mof_pos, mof_nums, ads_pos, ads_nums, f"{mof_formula} + {ads_formula}"


def load_mof_with_adsorbate_detection(mof_path, supercell_matrix=None):
    """Load MOF CIF and detect adsorbates using connected components.

    The largest connected component is treated as the framework,
    all smaller components are treated as adsorbates.

    Args:
        mof_path: Path to MOF CIF file (may contain adsorbates)
        supercell_matrix: 3x3 matrix for supercell

    Returns:
        framework_pos, framework_nums, ads_pos, ads_nums, formula
    """
    atoms = read(mof_path)
    formula = atoms.get_chemical_formula()

    print(f"  Detecting adsorbates via connected components...")

    # Find connected components to identify framework vs adsorbates
    framework_mask, adsorbate_mask, n_components, component_labels = \
        find_connected_components(atoms)

    # Separate framework and adsorbate atoms BEFORE tiling
    framework_atoms = atoms[framework_mask]
    adsorbate_atoms = atoms[adsorbate_mask]

    # Get adsorbate positions (don't tile adsorbates)
    ads_pos = adsorbate_atoms.positions.copy() if len(adsorbate_atoms) > 0 else np.zeros((0, 3))
    ads_nums = adsorbate_atoms.numbers.copy() if len(adsorbate_atoms) > 0 else np.array([], dtype=int)

    if supercell_matrix is not None:
        # Create supercell of framework only
        P = np.array(supercell_matrix)
        framework_supercell = make_supercell(framework_atoms, P)
        framework_pos = framework_supercell.positions.copy()
        framework_nums = framework_supercell.numbers.copy()

        # Center adsorbate in the supercell
        if len(ads_pos) > 0:
            framework_center = framework_pos.mean(axis=0)
            ads_center = ads_pos.mean(axis=0)
            ads_pos += (framework_center - ads_center)
    else:
        framework_pos = framework_atoms.positions.copy()
        framework_nums = framework_atoms.numbers.copy()

    # Center the entire complex at origin
    if len(ads_pos) > 0:
        all_pos = np.vstack([framework_pos, ads_pos])
    else:
        all_pos = framework_pos
    center = all_pos.mean(axis=0)
    framework_pos -= center
    if len(ads_pos) > 0:
        ads_pos -= center

    return framework_pos, framework_nums, ads_pos, ads_nums, formula


def load_and_tile_catalyst(cif_path, supercell_matrix=None):
    """Load CIF and create supercell with adsorbate centered.

    Args:
        cif_path: Path to CIF file
        supercell_matrix: 3x3 matrix for supercell, e.g., [[2,0,0],[0,2,0],[0,0,1]]

    Returns:
        slab_pos, slab_nums, ads_pos, ads_nums, formula
    """
    atoms = read(cif_path)
    formula = atoms.get_chemical_formula()

    # Separate slab and adsorbate BEFORE tiling
    numbers = atoms.numbers
    positions = atoms.positions

    slab_mask = np.array([n not in ADSORBATE_ELEMENTS for n in numbers])
    ads_mask = ~slab_mask

    # Get adsorbate info (don't tile adsorbate)
    ads_pos = positions[ads_mask].copy()
    ads_nums = numbers[ads_mask].copy()

    # Create slab-only atoms object for tiling
    slab_atoms = atoms[slab_mask]

    if supercell_matrix is not None:
        # Create supercell of slab only
        P = np.array(supercell_matrix)
        slab_supercell = make_supercell(slab_atoms, P)
        slab_pos = slab_supercell.positions.copy()
        slab_nums = slab_supercell.numbers.copy()

        # Center the adsorbate on the supercell
        slab_center_xy = slab_pos[:, :2].mean(axis=0)
        ads_center_xy = ads_pos[:, :2].mean(axis=0)
        offset = slab_center_xy - ads_center_xy
        ads_pos[:, :2] += offset
    else:
        slab_pos = positions[slab_mask].copy()
        slab_nums = numbers[slab_mask].copy()

    # Center the entire complex at origin for better visualization
    all_pos = np.vstack([slab_pos, ads_pos])
    center = all_pos.mean(axis=0)
    slab_pos -= center
    ads_pos -= center

    return slab_pos, slab_nums, ads_pos, ads_nums, formula


def cosine_schedule(t):
    """Cosine noise schedule for diffusion - smoother transition."""
    return np.cos((1 - t) * np.pi / 2) ** 2


def create_conditional_trajectory(slab_target, ads_target, num_steps=50,
                                   slab_noise_std=1.0, ads_noise_std=0.15):
    """Create conditional diffusion-style trajectory.

    Uses cosine schedule for smoother, more realistic denoising.
    Adsorbate: starts from slight perturbation, stays nearly fixed
    Slab: starts from noise, fully reconstructs with diffusion dynamics

    Args:
        slab_target: (N, 3) target slab coordinates
        ads_target: (M, 3) target adsorbate coordinates
        num_steps: Number of interpolation steps
        slab_noise_std: Noise level for slab prior (large)
        ads_noise_std: Noise level for adsorbate prior (small)

    Returns:
        slab_traj, ads_traj, times
    """
    # Adsorbate: slight perturbation from target
    ads_noise = np.random.randn(*ads_target.shape) * ads_noise_std
    ads_prior = ads_target + ads_noise

    # Slab: Gaussian noise centered around each atom's target position
    # This keeps the overall complex centered throughout the trajectory
    slab_prior = slab_target + np.random.randn(*slab_target.shape) * slab_noise_std * 3.0

    # Create trajectories with cosine (diffusion-style) interpolation
    times = np.linspace(0, 1, num_steps + 1)

    slab_traj = np.zeros((num_steps + 1, *slab_target.shape))
    ads_traj = np.zeros((num_steps + 1, *ads_target.shape))

    for i, t in enumerate(times):
        # Diffusion-style interpolation using cosine schedule
        alpha_t = cosine_schedule(t)

        # Interpolate from noisy prior to target
        slab_traj[i] = (1 - alpha_t) * slab_prior + alpha_t * slab_target
        ads_traj[i] = ads_target + ads_noise * (1 - alpha_t)  # Adsorbate noise decays

    return slab_traj, ads_traj, times


def setup_clean_3d_axes(ax):
    """Remove grids and clean up 3D axes."""
    # Remove grid
    ax.grid(False)

    # Make panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Make pane edges transparent
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    # Hide axis lines
    ax.xaxis.line.set_color('none')
    ax.yaxis.line.set_color('none')
    ax.zaxis.line.set_color('none')

    # Hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def draw_glossy_sphere(ax, pos, atomic_num, size, alpha=1.0, is_adsorbate=False):
    """Draw a glossy sphere with specular highlight.

    Creates a 3D-looking sphere using layered scatter points:
    1. Dark base/shadow layer
    2. Main color layer
    3. Lighter gradient layer
    4. Specular highlight (bright spot)
    """
    # Get color from atomic number - use different colors for adsorbate vs framework
    if is_adsorbate:
        base_color = ELEMENT_COLORS_ADSORBATE.get(int(atomic_num), '#808080')
    else:
        base_color = ELEMENT_COLORS_FRAMEWORK.get(int(atomic_num), '#808080')

    # Light direction offset (upper-left)
    light_offset = np.array([-0.15, 0.15, 0.2])

    # Scale sizes
    base_size = size * alpha

    # Layer 1: Dark edge/shadow (slightly offset down-right)
    shadow_color = darken_color(base_color, 0.5)
    shadow_offset = pos + np.array([0.05, -0.05, -0.05]) * (size / 100)
    ax.scatter(*shadow_offset, c=[shadow_color], s=base_size * 1.1,
               alpha=alpha * 0.4, edgecolors='none', depthshade=False)

    # Layer 2: Main sphere body without edges for cleaner overlap
    ax.scatter(*pos, c=[base_color], s=base_size,
               edgecolors='none',
               alpha=alpha * 0.95, depthshade=False)

    # Layer 3: Gradient highlight (lighter top-left area)
    if alpha > 0.3:
        highlight_color = lighten_color(base_color, 0.3)
        highlight_pos = pos + light_offset * (size / 150)
        ax.scatter(*highlight_pos, c=[highlight_color], s=base_size * 0.5,
                   alpha=alpha * 0.6, edgecolors='none', depthshade=False)

    # Layer 4: Specular highlight (bright white spot)
    if alpha > 0.5:
        specular_pos = pos + light_offset * (size / 120)
        specular_size = base_size * 0.15 if is_adsorbate else base_size * 0.12
        ax.scatter(*specular_pos, c='white', s=specular_size,
                   alpha=alpha * 0.8, edgecolors='none', depthshade=False)


def visualize_structure(ax, slab_pos, slab_nums, ads_pos, ads_nums,
                        title="", slab_alpha=1.0, view_elev=25, view_azim=-60):
    """Plot catalyst structure with glossy sphere rendering."""
    ax.clear()

    # Sort atoms by z-coordinate for proper depth rendering (back to front)
    # Combine all atoms with their properties
    all_atoms = []

    for pos, num in zip(slab_pos, slab_nums):
        all_atoms.append({
            'pos': pos,
            'num': num,
            'is_adsorbate': False,
            'alpha': slab_alpha
        })

    for pos, num in zip(ads_pos, ads_nums):
        all_atoms.append({
            'pos': pos,
            'num': num,
            'is_adsorbate': True,
            'alpha': 1.0
        })

    # Sort by z (and slightly by distance from camera for better depth)
    # Camera is roughly at azim=-60, elev=25
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

    # Clean axes
    setup_clean_3d_axes(ax)

    # Set viewing angle (looking down at surface)
    ax.view_init(elev=view_elev, azim=view_azim)

    if title:
        ax.set_title(title, fontsize=11, pad=10, fontweight='medium')


def plot_static_comparison(slab_target, slab_nums, ads_target, ads_nums,
                           slab_traj, ads_traj, times, output_path=None,
                           view_elev=0, view_azim=0):
    """Create static comparison plot at key timesteps with legend."""
    # Tighter figure layout
    fig = plt.figure(figsize=(16, 4))

    timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]
    step_indices = [int(t * (len(times) - 1)) for t in timesteps]

    # Compute axis limits from all positions (tighter margins)
    all_pos = np.vstack([slab_traj.reshape(-1, 3), ads_traj.reshape(-1, 3)])
    margin = 0.5
    xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    for idx, (t, step_idx) in enumerate(zip(timesteps, step_indices)):
        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')

        slab_pos = slab_traj[step_idx]
        ads_pos = ads_traj[step_idx]

        # Slab becomes more solid as t increases
        slab_alpha = 0.3 + 0.7 * t

        visualize_structure(
            ax, slab_pos, slab_nums, ads_pos, ads_nums,
            title=f't = {t:.2f}',
            slab_alpha=slab_alpha,
            view_elev=view_elev, view_azim=view_azim
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    # Tight layout with minimal spacing
    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12, wspace=0.02)

    plt.suptitle('Conditional Flow Matching: Framework Generation',
                 fontsize=12, y=0.96, fontweight='medium')

    # Add colored legend at bottom using scatter markers
    from matplotlib.lines import Line2D

    # Detect which elements are present
    all_slab_elements = set(slab_nums)
    all_ads_elements = set(ads_nums)

    # Element names for legend
    element_names = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 26: 'Fe',
        46: 'Pd', 48: 'Cd', 80: 'Hg'
    }

    # Create legend elements dynamically with distinct colors
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#444444',
               markersize=0, label='Adsorbate (fixed):'),
    ]

    # Adsorbate elements with adsorbate colors
    for elem in sorted(all_ads_elements):
        if elem in ELEMENT_COLORS_ADSORBATE:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ELEMENT_COLORS_ADSORBATE.get(elem, '#808080'),
                       markeredgecolor='#333', markersize=10,
                       label=element_names.get(elem, f'Z={elem}'))
            )

    legend_elements.append(
        Line2D([0], [0], marker='', color='w', markersize=0, label='    ')
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#444444',
               markersize=0, label='Framework (generated):')
    )

    # Framework elements with framework colors (show all, including C and O)
    for elem in sorted(all_slab_elements):
        if elem in ELEMENT_COLORS_FRAMEWORK:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ELEMENT_COLORS_FRAMEWORK.get(elem, '#808080'),
                       markeredgecolor='#333', markersize=10,
                       label=element_names.get(elem, f'Z={elem}'))
            )

    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
               fontsize=9, frameon=False, handletextpad=0.3, columnspacing=0.8)

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig


def create_animation(slab_traj, slab_nums, ads_traj, ads_nums, times,
                     output_path=None, fps=15, pause_frames=30,
                     view_elev=0, view_azim=0):
    """Create smooth animation of the flow matching process.

    Args:
        pause_frames: Number of extra frames to hold at the end
        view_elev: Elevation angle for viewing
        view_azim: Azimuth angle for viewing
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Compute axis limits
    all_pos = np.vstack([slab_traj.reshape(-1, 3), ads_traj.reshape(-1, 3)])
    margin = 0.5
    xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    total_frames = len(times) + pause_frames

    def update(frame):
        ax.clear()

        # Clamp frame to trajectory length (repeat final frame for pause)
        traj_frame = min(frame, len(times) - 1)
        t = times[traj_frame]

        slab_pos = slab_traj[traj_frame]
        ads_pos = ads_traj[traj_frame]

        slab_alpha = 0.3 + 0.7 * t

        visualize_structure(
            ax, slab_pos, slab_nums, ads_pos, ads_nums,
            title=f'Flow Matching (t = {t:.2f})',
            slab_alpha=slab_alpha,
            view_elev=view_elev, view_azim=view_azim
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        return ax,

    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=1000/fps, blit=False)

    if output_path:
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"Saved: {output_path}")

    plt.close(fig)
    return anim


def plot_rotation_tiling_comparison(framework_pos, framework_nums, ads_pos, ads_nums,
                                     formula, output_path=None, structure_type="MOF",
                                     n_rotations=9):
    """Create comparison grid showing different rotations and views.

    Shows the final structure (t=1.0) from various viewing angles
    to help choose the best visualization perspective.

    Args:
        n_rotations: Number of rotations to show (9, 16, 25, 36, etc.)
    """
    if n_rotations <= 9:
        # Original 9 rotations
        rotation_options = [
            (15, -60, "elev=15, azim=-60"),
            (25, -60, "elev=25, azim=-60"),
            (35, -60, "elev=35, azim=-60"),
            (45, -60, "elev=45, azim=-60"),
            (25, -30, "elev=25, azim=-30"),
            (25, -90, "elev=25, azim=-90"),
            (35, -45, "elev=35, azim=-45"),
            (35, -135, "elev=35, azim=-135"),
            (60, -60, "elev=60, azim=-60"),
        ]
        cols, rows = 3, 3
    else:
        # Generate grid of rotations
        # Elevation: from 10 to 70 degrees
        # Azimuth: full 360 degree rotation
        n_elev = int(np.sqrt(n_rotations / 2))  # Fewer elevation steps
        n_azim = int(n_rotations / n_elev)  # More azimuth steps

        elevations = np.linspace(10, 70, n_elev)
        azimuths = np.linspace(0, 360, n_azim, endpoint=False)

        rotation_options = []
        for elev in elevations:
            for azim in azimuths:
                rotation_options.append((int(elev), int(azim), f"e={int(elev)}, a={int(azim)}"))

        cols = n_azim
        rows = n_elev

    n_total = len(rotation_options)
    fig = plt.figure(figsize=(3 * cols, 3 * rows))

    # Compute axis limits
    all_pos = np.vstack([framework_pos, ads_pos])
    margin = 0.5
    xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    for idx, (elev, azim, label) in enumerate(rotation_options):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

        visualize_structure(
            ax, framework_pos, framework_nums, ads_pos, ads_nums,
            title=label,
            slab_alpha=1.0,
            view_elev=elev, view_azim=azim
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    plt.suptitle(f'{structure_type} Rotation Comparison ({n_total} views)',
                 fontsize=14, y=0.995, fontweight='medium')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01,
                        wspace=0.02, hspace=0.12)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig


def plot_elevation_comparison(framework_pos, framework_nums, ads_pos, ads_nums,
                               formula, output_path=None, structure_type="MOF",
                               azim=0, elevations=None):
    """Create comparison grid showing different elevation angles with fixed azimuth.

    Args:
        azim: Fixed azimuth angle
        elevations: List of elevation angles to show (default: 0 to 90 in steps of 10)
    """
    if elevations is None:
        elevations = list(range(0, 91, 10))  # 0, 10, 20, ..., 90

    n_views = len(elevations)
    cols = min(5, n_views)
    rows = (n_views + cols - 1) // cols

    fig = plt.figure(figsize=(3.5 * cols, 3.5 * rows))

    # Compute axis limits
    all_pos = np.vstack([framework_pos, ads_pos])
    margin = 0.5
    xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    for idx, elev in enumerate(elevations):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

        visualize_structure(
            ax, framework_pos, framework_nums, ads_pos, ads_nums,
            title=f"elev={elev}°",
            slab_alpha=1.0,
            view_elev=elev, view_azim=azim
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    plt.suptitle(f'{structure_type} Elevation Comparison (azim={azim}°, {n_views} views)',
                 fontsize=14, y=0.995, fontweight='medium')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01,
                        wspace=0.02, hspace=0.12)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig


def plot_tiling_comparison(load_func, path_or_paths, tiling_options, output_path=None,
                           structure_type="MOF", view_elev=35, view_azim=-50):
    """Create comparison showing different supercell tilings.

    Args:
        load_func: Function to load structure (load_mof_and_adsorbate or load_and_tile_catalyst)
        path_or_paths: Either single path (catalyst) or tuple of paths (mof, ads)
        tiling_options: List of (supercell_matrix, label) tuples
        output_path: Where to save the figure
        structure_type: "MOF" or "Catalyst"
        view_elev: Elevation angle for viewing
        view_azim: Azimuth angle for viewing
    """
    n_options = len(tiling_options)
    cols = min(3, n_options)
    rows = (n_options + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 5 * rows))

    for idx, (supercell, label) in enumerate(tiling_options):
        # Load structure with this tiling
        if isinstance(path_or_paths, tuple):
            # MOF + adsorbate
            framework_pos, framework_nums, ads_pos, ads_nums, formula = load_func(
                path_or_paths[0], path_or_paths[1], supercell_matrix=supercell
            )
        else:
            # Catalyst
            framework_pos, framework_nums, ads_pos, ads_nums, formula = load_func(
                path_or_paths, supercell_matrix=supercell
            )

        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

        # Compute axis limits for this structure
        all_pos = np.vstack([framework_pos, ads_pos])
        margin = 0.5
        xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
        ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
        zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

        n_atoms = len(framework_pos) + len(ads_pos)
        visualize_structure(
            ax, framework_pos, framework_nums, ads_pos, ads_nums,
            title=f"{label}\n({n_atoms} atoms)",
            slab_alpha=1.0,
            view_elev=view_elev, view_azim=view_azim
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    plt.suptitle(f'{structure_type} Tiling Comparison (elev={view_elev}, azim={view_azim})',
                 fontsize=14, y=0.98, fontweight='medium')
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02,
                        wspace=0.05, hspace=0.15)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig


def plot_tiling_comparison_with_detection(load_func, mof_path, tiling_options, output_path=None,
                                          structure_type="MOF", view_elev=35, view_azim=-50):
    """Create comparison showing different supercell tilings with adsorbate detection.

    Args:
        load_func: Function to load structure with adsorbate detection
        mof_path: Path to MOF CIF file
        tiling_options: List of (supercell_matrix, label) tuples
        output_path: Where to save the figure
        structure_type: "MOF" or "Catalyst"
        view_elev: Elevation angle for viewing
        view_azim: Azimuth angle for viewing
    """
    n_options = len(tiling_options)
    # Use 9 columns for 27 options (3x3x3 grid organized by z slices)
    cols = 9
    rows = (n_options + cols - 1) // cols

    fig = plt.figure(figsize=(2.5 * cols, 2.5 * rows))

    # Suppress repeated output during tiling comparison
    import sys
    from io import StringIO

    for idx, (supercell, label) in enumerate(tiling_options):
        # Suppress print output for cleaner logs
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            framework_pos, framework_nums, ads_pos, ads_nums, formula = load_func(
                mof_path, supercell_matrix=supercell
            )
        finally:
            sys.stdout = old_stdout

        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

        # Compute axis limits for this structure
        if len(ads_pos) > 0:
            all_pos = np.vstack([framework_pos, ads_pos])
        else:
            all_pos = framework_pos
        margin = 0.5
        xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
        ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
        zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

        n_framework = len(framework_pos)
        n_ads = len(ads_pos)
        visualize_structure(
            ax, framework_pos, framework_nums, ads_pos, ads_nums,
            title=f"{label}\n(F:{n_framework} A:{n_ads})",
            slab_alpha=1.0,
            view_elev=view_elev, view_azim=view_azim
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    plt.suptitle(f'{structure_type} Tiling Comparison (elev={view_elev}, azim={view_azim})\nF=Framework, A=Adsorbate',
                 fontsize=12, y=0.995, fontweight='medium')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01,
                        wspace=0.02, hspace=0.12)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig


def main_mof(output_dir, create_gif=False):
    """Generate visualization for MOF with adsorbate."""
    print("=" * 60)
    print("Flow Matching Visualization - MOF with CO2 Adsorbate")
    print("=" * 60)

    np.random.seed(42)

    # Use separate framework and adsorbate files
    mof_path = Path("data/sample/mof_clean.xyz")
    ads_path = Path("data/sample/co2_with_cell.xyz")

    print(f"\nLoading MOF: {mof_path}")
    print(f"Loading adsorbate: {ads_path}")

    # Final settings: 1x2x2 supercell with (0,0,1) adsorbate shift
    supercell_matrix = [[1, 0, 0], [0, 2, 0], [0, 0, 2]]
    ads_shift = (0, 0, 1)  # Shift by 1 lattice vector in c direction

    slab_pos, slab_nums, ads_pos, ads_nums, formula = load_mof_and_adsorbate(
        mof_path, ads_path, supercell_matrix=supercell_matrix, ads_shift=ads_shift
    )

    print(f"  Formula: {formula}")
    print(f"  Framework atoms (1x2x2 supercell): {len(slab_pos)}")
    print(f"  Adsorbate atoms: {len(ads_pos)}")
    print(f"  Adsorbate shift: {ads_shift}")

    # Final view settings for MOF
    view_elev = -10
    view_azim = 0

    # Generate trajectory visualization
    print("\nGenerating conditional flow trajectory...")
    slab_traj, ads_traj, times = create_conditional_trajectory(
        slab_pos, ads_pos, num_steps=20, slab_noise_std=1.0, ads_noise_std=0.15
    )

    print(f"\nGenerating static comparison (elev={view_elev}, azim={view_azim})...")
    plot_static_comparison(
        slab_pos, slab_nums, ads_pos, ads_nums,
        slab_traj, ads_traj, times,
        output_path=f"{output_dir}/figures/mof_flow_static.png",
        view_elev=view_elev, view_azim=view_azim
    )
    plt.close()

    if create_gif:
        print("\nGenerating animation...")
        create_animation(
            slab_traj, slab_nums, ads_traj, ads_nums, times,
            output_path=f"{output_dir}/figures/mof_flow_reconstruction.gif",
            fps=2, pause_frames=4,
            view_elev=view_elev, view_azim=view_azim
        )

    return formula


def main_catalyst(output_dir, catalyst_id="1234", create_gif=False):
    """Generate visualization for catalyst structure."""
    print("=" * 60)
    print(f"Flow Matching Visualization - Catalyst {catalyst_id}")
    print("=" * 60)

    np.random.seed(42)

    cif_path = Path(f"data/sample/catalyst/{catalyst_id}.cif")
    print(f"\nLoading: {cif_path}")

    # 2x2 supercell for catalyst slab
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]

    slab_pos, slab_nums, ads_pos, ads_nums, formula = load_and_tile_catalyst(
        cif_path, supercell_matrix=supercell_matrix
    )

    print(f"  Formula: {formula}")
    print(f"  Slab atoms (2x2 supercell): {len(slab_pos)}")
    print(f"  Adsorbate atoms: {len(ads_pos)}")

    # Final view settings for catalyst
    view_elev = 35
    view_azim = -50

    # Generate trajectory visualization
    print("\nGenerating conditional flow trajectory...")
    slab_traj, ads_traj, times = create_conditional_trajectory(
        slab_pos, ads_pos, num_steps=20, slab_noise_std=1.0, ads_noise_std=0.15
    )

    print(f"\nGenerating static comparison (elev={view_elev}, azim={view_azim})...")
    plot_static_comparison(
        slab_pos, slab_nums, ads_pos, ads_nums,
        slab_traj, ads_traj, times,
        output_path=f"{output_dir}/figures/catalyst_{catalyst_id}_flow_static.png",
        view_elev=view_elev, view_azim=view_azim
    )
    plt.close()

    if create_gif:
        print("\nGenerating animation...")
        create_animation(
            slab_traj, slab_nums, ads_traj, ads_nums, times,
            output_path=f"{output_dir}/figures/catalyst_{catalyst_id}_flow_reconstruction.gif",
            fps=2, pause_frames=4,
            view_elev=view_elev, view_azim=view_azim
        )

    return formula


def main():
    """Main entry point with options for MOF or catalyst visualization."""
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Flow Matching Visualization")
    parser.add_argument("--mode", choices=["mof", "catalyst", "both"], default="mof",
                        help="Visualization mode: mof, catalyst, or both")
    parser.add_argument("--catalyst-id", default="1234",
                        help="Catalyst ID (for catalyst mode)")
    parser.add_argument("--create-gif", action="store_true",
                        help="Create GIF animation (disabled by default)")
    args = parser.parse_args()

    output_dir = "data/flow_matching_viz"
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    outputs = []

    if args.mode in ["mof", "both"]:
        formula = main_mof(output_dir, create_gif=args.create_gif)
        outputs.append("mof_flow_static.png")
        if args.create_gif:
            outputs.append("mof_flow_reconstruction.gif")

    if args.mode in ["catalyst", "both"]:
        if args.mode == "both":
            print("\n" + "=" * 60 + "\n")
        formula = main_catalyst(output_dir, catalyst_id=args.catalyst_id,
                               create_gif=args.create_gif)
        outputs.append(f"catalyst_{args.catalyst_id}_flow_static.png")
        if args.create_gif:
            outputs.append(f"catalyst_{args.catalyst_id}_flow_reconstruction.gif")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output saved to: {output_dir}/figures/")
    for out in outputs:
        print(f"  - {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
