"""
Utility functions and constants for materials visualization.

This module provides:
- Color schemes for framework and adsorbate atoms
- Element size calculations based on covalent radii
- Color manipulation utilities (lighten, darken)
- 3D axes cleanup functions
"""

import numpy as np
from matplotlib.colors import to_rgb


# ============================================================================
# COLOR SCHEMES
# ============================================================================

# Framework color schemes (muted, darker tones for background structures)
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

# Adsorbate color schemes (bright, high contrast for emphasis)
ADSORBATE_SCHEMES = {
    'neon': {
        1: '#FFFFFF', 6: '#00FF80', 7: '#00FFFF', 8: '#FF00FF', 16: '#FFFF00',
    },
    'soft_neon': {
        1: '#FFFFFF', 6: '#50E090', 7: '#60D0E0', 8: '#E060D0', 16: '#E0E050',
    },
    'mint_coral': {
        1: '#FFFFFF', 6: '#3EB489', 7: '#5BC0EB', 8: '#FF6B6B', 16: '#FFE66D',
    },
    'ocean_sunset': {
        1: '#FFFFFF', 6: '#2ECC71', 7: '#3498DB', 8: '#E74C3C', 16: '#F39C12',
    },
    'spring': {
        1: '#FFFFFF', 6: '#00D084', 7: '#00B4D8', 8: '#FF5C8D', 16: '#FFD93D',
    },
    'arctic': {
        1: '#FFFFFF', 6: '#48CAE4', 7: '#90E0EF', 8: '#F72585', 16: '#FFBE0B',
    },
    'forest_berry': {
        1: '#FFFFFF', 6: '#52B788', 7: '#74C69D', 8: '#D64161', 16: '#FFB703',
    },
    'tech': {
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

# Default schemes
DEFAULT_FRAMEWORK_SCHEME = 'cool_gray'
DEFAULT_ADSORBATE_SCHEME = 'mint_coral'
DEFAULT_BOUNDARY_SCHEME = 'thin_dark'


# ============================================================================
# ELEMENT PROPERTIES
# ============================================================================

# Covalent radii in Angstroms
COVALENT_RADII = {
    1: 0.31,   # H
    6: 0.77,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    16: 1.05,  # S
    26: 1.32,  # Fe
    46: 1.39,  # Pd
    48: 1.44,  # Cd
    80: 1.32,  # Hg
}

# Element names for legends
ELEMENT_NAMES = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S',
    26: 'Fe', 46: 'Pd', 48: 'Cd', 80: 'Hg',
}

# Adsorbate element set (common small molecules)
ADSORBATE_ELEMENTS = {1, 6, 7, 8, 16}


# ============================================================================
# COLOR MANIPULATION
# ============================================================================

def lighten_color(color, amount=0.5):
    """
    Lighten a color by blending with white.

    Args:
        color: Color string (hex or named)
        amount: Amount to lighten (0=no change, 1=white)

    Returns:
        RGB tuple
    """
    rgb = to_rgb(color)
    return tuple(c * (1 - amount) + amount for c in rgb)


def darken_color(color, amount=0.3):
    """
    Darken a color by reducing brightness.

    Args:
        color: Color string (hex or named)
        amount: Amount to darken (0=no change, 1=black)

    Returns:
        RGB tuple
    """
    rgb = to_rgb(color)
    return tuple(max(0, c * (1 - amount)) for c in rgb)


# ============================================================================
# SIZE CALCULATIONS
# ============================================================================

def get_element_size(atomic_number, is_adsorbate=False, size_scale=1.0):
    """
    Get visualization size based on atomic number using covalent radii.

    Args:
        atomic_number: Atomic number (Z)
        is_adsorbate: Whether atom is part of adsorbate (larger display)
        size_scale: Additional scaling factor

    Returns:
        Scatter plot size value
    """
    radius = COVALENT_RADII.get(atomic_number, 1.0)
    if is_adsorbate:
        return 300 * (radius ** 2) * size_scale
    else:
        return 200 * (radius ** 2) * size_scale


# ============================================================================
# MATPLOTLIB UTILITIES
# ============================================================================

def setup_clean_3d_axes(ax):
    """
    Remove grids, panes, and tick labels from 3D axes for clean visualization.

    Args:
        ax: Matplotlib 3D axes object
    """
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


def get_camera_direction(view_elev, view_azim):
    """
    Calculate camera direction vector from elevation and azimuth angles.

    Args:
        view_elev: Elevation angle in degrees
        view_azim: Azimuth angle in degrees

    Returns:
        Camera direction vector (3D numpy array)
    """
    return np.array([
        np.cos(np.radians(view_azim)) * np.cos(np.radians(view_elev)),
        np.sin(np.radians(view_azim)) * np.cos(np.radians(view_elev)),
        np.sin(np.radians(view_elev))
    ])


# ============================================================================
# SCHEME GETTERS
# ============================================================================

def get_framework_colors(scheme_name=None):
    """Get framework color scheme by name."""
    if scheme_name is None:
        scheme_name = DEFAULT_FRAMEWORK_SCHEME
    return FRAMEWORK_SCHEMES.get(scheme_name, FRAMEWORK_SCHEMES[DEFAULT_FRAMEWORK_SCHEME])


def get_adsorbate_colors(scheme_name=None):
    """Get adsorbate color scheme by name."""
    if scheme_name is None:
        scheme_name = DEFAULT_ADSORBATE_SCHEME
    return ADSORBATE_SCHEMES.get(scheme_name, ADSORBATE_SCHEMES[DEFAULT_ADSORBATE_SCHEME])


def get_boundary_style(scheme_name=None):
    """Get boundary style by name."""
    if scheme_name is None:
        scheme_name = DEFAULT_BOUNDARY_SCHEME
    return BOUNDARY_SCHEMES.get(scheme_name, BOUNDARY_SCHEMES[DEFAULT_BOUNDARY_SCHEME])
