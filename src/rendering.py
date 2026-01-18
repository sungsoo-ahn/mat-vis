"""
Shared rendering functions for 3D structure visualization.

Provides glossy sphere rendering with shadows, highlights, and specular effects.
"""

import numpy as np
from src.utils import (
    lighten_color,
    darken_color,
    get_element_size,
    setup_clean_3d_axes,
    get_camera_direction,
)


def draw_glossy_sphere(ax, pos, atomic_num, size, alpha=1.0, is_adsorbate=False,
                       framework_colors=None, adsorbate_colors=None, boundary_style=None,
                       show_shadow=True, glossy=True):
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
        show_shadow: Whether to render shadows
        glossy: Whether to render highlight and specular effects
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

    # Shadow layer (optional)
    if show_shadow:
        shadow_color = darken_color(base_color, 0.5)
        shadow_offset = pos + np.array([0.05, -0.05, -0.05]) * (size / 100)
        ax.scatter(*shadow_offset, c=[shadow_color], s=base_size * 1.1,
                   alpha=alpha * 0.3, edgecolors='none', depthshade=False)

    # Main sphere
    ax.scatter(*pos, c=[base_color], s=base_size,
               edgecolors=edge_color, linewidth=linewidth,
               alpha=alpha, depthshade=False)

    # Highlight (glossy mode only) - positioned within the sphere
    if glossy and alpha > 0.3:
        highlight_color = lighten_color(base_color, 0.35)
        # Small offset with positive z to render in front
        highlight_offset = np.array([-0.06, 0.06, 0.1]) * np.sqrt(base_size) * 0.012
        highlight_pos = pos + highlight_offset
        ax.scatter(*highlight_pos, c=[highlight_color], s=base_size * 0.35,
                   alpha=alpha * 0.6, edgecolors='none', depthshade=False)

    # Specular (glossy mode only) - white shine spot
    if glossy and alpha > 0.5:
        # Small offset with positive z to render in front
        specular_offset = np.array([-0.08, 0.08, 0.15]) * np.sqrt(base_size) * 0.01
        specular_pos = pos + specular_offset
        specular_size = base_size * 0.08 if is_adsorbate else base_size * 0.06
        ax.scatter(*specular_pos, c='white', s=specular_size,
                   alpha=alpha * 0.95, edgecolors='none', depthshade=False)


def render_structure(ax, slab_pos, slab_nums, ads_pos, ads_nums,
                     title="", slab_alpha=1.0, view_elev=25, view_azim=-60,
                     size_scale=1.0, framework_colors=None, adsorbate_colors=None,
                     boundary_style=None, show_shadow=True, glossy=True):
    """
    Render a structure with glossy sphere visualization.

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
        show_shadow: Whether to render shadows
        glossy: Whether to render highlight and specular effects
    """
    ax.clear()

    # Combine and sort atoms by depth for proper rendering order
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
                          boundary_style=boundary_style,
                          show_shadow=show_shadow,
                          glossy=glossy)

    setup_clean_3d_axes(ax)
    ax.view_init(elev=view_elev, azim=view_azim)
    if title:
        ax.set_title(title, fontsize=11, pad=10, fontweight='medium')


def set_axis_limits_with_margin(ax, positions, margin=1.0):
    """
    Set axis limits based on positions with margin.

    Args:
        ax: Matplotlib 3D axes
        positions: Nx3 array of positions
        margin: Margin to add around structure
    """
    xlim = (positions[:, 0].min() - margin, positions[:, 0].max() + margin)
    ylim = (positions[:, 1].min() - margin, positions[:, 1].max() + margin)
    zlim = (positions[:, 2].min() - margin, positions[:, 2].max() + margin)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)


def draw_crystal_sphere(ax, pos, atomic_num, size, alpha=1.0,
                        crystal_colors=None, boundary_style=None,
                        show_shadow=True, glossy=True):
    """
    Draw a glossy sphere for crystal visualization (unified coloring).

    Args:
        ax: Matplotlib 3D axes
        pos: Position (x, y, z)
        atomic_num: Atomic number
        size: Sphere size
        alpha: Transparency
        crystal_colors: Color scheme for crystal atoms
        boundary_style: Boundary style dict
        show_shadow: Whether to render shadows
        glossy: Whether to render highlight and specular effects
    """
    base_color = crystal_colors.get(int(atomic_num), '#808080')

    # Determine edge color
    edge = boundary_style['edge']
    linewidth = boundary_style['linewidth']

    if edge == 'auto':
        edge_color = darken_color(base_color, 0.5)
    elif edge == 'none':
        edge_color = 'none'
    else:
        edge_color = edge

    base_size = size * alpha

    # Shadow layer (optional)
    if show_shadow:
        shadow_color = darken_color(base_color, 0.5)
        shadow_offset = pos + np.array([0.05, -0.05, -0.05]) * (size / 100)
        ax.scatter(*shadow_offset, c=[shadow_color], s=base_size * 1.1,
                   alpha=alpha * 0.3, edgecolors='none', depthshade=False)

    # Main sphere
    ax.scatter(*pos, c=[base_color], s=base_size,
               edgecolors=edge_color, linewidth=linewidth,
               alpha=alpha, depthshade=False)

    # Highlight (glossy mode only)
    if glossy and alpha > 0.3:
        highlight_color = lighten_color(base_color, 0.35)
        highlight_offset = np.array([-0.06, 0.06, 0.1]) * np.sqrt(base_size) * 0.012
        highlight_pos = pos + highlight_offset
        ax.scatter(*highlight_pos, c=[highlight_color], s=base_size * 0.35,
                   alpha=alpha * 0.6, edgecolors='none', depthshade=False)

    # Specular (glossy mode only)
    if glossy and alpha > 0.5:
        specular_offset = np.array([-0.08, 0.08, 0.15]) * np.sqrt(base_size) * 0.01
        specular_pos = pos + specular_offset
        specular_size = base_size * 0.06
        ax.scatter(*specular_pos, c='white', s=specular_size,
                   alpha=alpha * 0.95, edgecolors='none', depthshade=False)


def render_crystal(ax, positions, atomic_nums, title="", alpha=1.0,
                   view_elev=25, view_azim=-60, size_scale=1.0,
                   crystal_colors=None, boundary_style=None,
                   show_shadow=True, glossy=True):
    """
    Render a crystal structure with unified coloring (no framework/adsorbate distinction).

    Args:
        ax: Matplotlib 3D axes
        positions: Atom positions (Nx3 array)
        atomic_nums: Atomic numbers
        title: Plot title
        alpha: Transparency for all atoms
        view_elev: Camera elevation angle
        view_azim: Camera azimuth angle
        size_scale: Size scaling factor
        crystal_colors: Crystal color scheme
        boundary_style: Boundary style dict
        show_shadow: Whether to render shadows
        glossy: Whether to render highlight and specular effects
    """
    ax.clear()

    # Sort atoms by depth for proper rendering order
    cam_dir = get_camera_direction(view_elev, view_azim)
    depth_order = np.argsort([np.dot(pos, cam_dir) for pos in positions])

    for idx in depth_order:
        pos = positions[idx]
        num = atomic_nums[idx]
        size = get_element_size(num, is_adsorbate=False, size_scale=size_scale)
        draw_crystal_sphere(ax, pos, num, size, alpha=alpha,
                           crystal_colors=crystal_colors,
                           boundary_style=boundary_style,
                           show_shadow=show_shadow,
                           glossy=glossy)

    setup_clean_3d_axes(ax)
    ax.view_init(elev=view_elev, azim=view_azim)
    if title:
        ax.set_title(title, fontsize=11, pad=10, fontweight='medium')
