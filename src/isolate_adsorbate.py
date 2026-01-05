"""
Adsorbate Isolation Module.

Provides unified adsorbate isolation for both MOF and catalyst structures:
- MOF: Connectivity-based isolation using graph analysis
- Catalyst: Element-based isolation using atomic numbers
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

from ase.io import read, write
from ase import Atoms
from ase.build import make_supercell
from ase.neighborlist import natural_cutoffs, NeighborList
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.utils import ADSORBATE_ELEMENTS


@dataclass
class IsolationResult:
    """Result of adsorbate isolation."""
    framework: Atoms
    adsorbates: List[Atoms]
    method: str  # "connectivity" or "element"
    cutoff_multiplier: Optional[float] = None  # Only for connectivity method
    n_components: Optional[int] = None  # Only for connectivity method


# ============================================================================
# CONNECTIVITY-BASED ISOLATION (for MOFs)
# ============================================================================

def find_connected_components(atoms: Atoms, cutoff_mult: float = 1.0) -> Tuple[int, np.ndarray]:
    """Find connected components in an atomic structure.

    Uses neighbor lists based on covalent radii to build a connectivity graph,
    then finds connected components.

    Args:
        atoms: ASE Atoms object
        cutoff_mult: Multiplier for sum of covalent radii to determine bonds

    Returns:
        n_components: Number of connected components
        component_labels: Array of component labels for each atom
    """
    n_atoms = len(atoms)

    if n_atoms == 0:
        return 0, np.array([], dtype=int)

    # Use ASE's natural cutoffs (based on covalent radii)
    cutoffs = natural_cutoffs(atoms, mult=cutoff_mult)

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

    return n_components, component_labels


def search_cutoff_for_n_components(
    atoms: Atoms,
    target_n_components: int,
    cutoff_min: float = 0.5,
    cutoff_max: float = 2.0,
    tolerance: float = 0.001,
    max_iterations: int = 100,
    verbose: bool = True
) -> Tuple[Optional[float], int, np.ndarray]:
    """Search for cutoff multiplier that yields target number of components.

    Uses binary search to find the cutoff multiplier where the structure
    separates into exactly the target number of disconnected components.

    Args:
        atoms: ASE Atoms object
        target_n_components: Desired number of disconnected components
        cutoff_min: Minimum cutoff multiplier to search
        cutoff_max: Maximum cutoff multiplier to search
        tolerance: Convergence tolerance for binary search
        max_iterations: Maximum number of search iterations
        verbose: Print progress information

    Returns:
        cutoff_mult: Found cutoff multiplier (None if not found)
        n_components: Number of components at found cutoff
        component_labels: Component labels for each atom
    """
    if verbose:
        print(f"Searching for cutoff that yields {target_n_components} components...")
        print(f"Search range: [{cutoff_min:.3f}, {cutoff_max:.3f}]")

    # First, scan the range to understand the landscape
    scan_points = np.linspace(cutoff_min, cutoff_max, 20)
    scan_results = []

    for mult in scan_points:
        n_comp, labels = find_connected_components(atoms, mult)
        scan_results.append((mult, n_comp))
        if verbose:
            print(f"  cutoff_mult={mult:.3f}: {n_comp} components")

    # Find range where target might exist
    found_range = None
    for i in range(len(scan_results) - 1):
        mult_low, n_low = scan_results[i]
        mult_high, n_high = scan_results[i + 1]

        # Check if target is in this range (components decrease as cutoff increases)
        if n_low >= target_n_components >= n_high:
            found_range = (mult_low, mult_high)
            break
        elif n_low == target_n_components:
            # Exact match at scan point
            _, labels = find_connected_components(atoms, mult_low)
            return mult_low, n_low, labels

    if found_range is None:
        if verbose:
            print(f"Target {target_n_components} components not achievable in range")
            print(f"Range yields {scan_results[0][1]} to {scan_results[-1][1]} components")
        # Return the closest match
        closest = min(scan_results, key=lambda x: abs(x[1] - target_n_components))
        _, labels = find_connected_components(atoms, closest[0])
        return closest[0], closest[1], labels

    # Binary search within found range
    low, high = found_range
    best_mult = None
    best_labels = None

    if verbose:
        print(f"\nBinary search in range [{low:.3f}, {high:.3f}]")

    for iteration in range(max_iterations):
        mid = (low + high) / 2
        n_comp, labels = find_connected_components(atoms, mid)

        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration}: cutoff={mid:.4f}, components={n_comp}")

        if n_comp == target_n_components:
            best_mult = mid
            best_labels = labels
            # Try to find the largest cutoff that still gives target components
            low = mid
        elif n_comp > target_n_components:
            # Need larger cutoff (more connectivity)
            low = mid
        else:
            # Need smaller cutoff (less connectivity)
            high = mid

        if high - low < tolerance:
            break

    if best_mult is not None:
        if verbose:
            print(f"\nFound cutoff_mult={best_mult:.4f} yielding {target_n_components} components")
        return best_mult, target_n_components, best_labels

    # Return the best we found
    final_mult = (low + high) / 2
    n_comp, labels = find_connected_components(atoms, final_mult)
    if verbose:
        print(f"\nBest found: cutoff_mult={final_mult:.4f} yielding {n_comp} components")
    return final_mult, n_comp, labels


def isolate_by_connectivity(
    atoms: Atoms,
    target_n_components: int = 3,
    cutoff_min: float = 0.5,
    cutoff_max: float = 2.0,
    verbose: bool = False
) -> IsolationResult:
    """Isolate adsorbates using connectivity analysis (for MOFs).

    Searches for distance cutoff where structure separates into target
    number of components. The largest component is assumed to be the
    framework, all others are adsorbates.

    Args:
        atoms: ASE Atoms object containing structure with adsorbates
        target_n_components: Expected number of components (framework + adsorbates)
        cutoff_min: Minimum cutoff multiplier to search
        cutoff_max: Maximum cutoff multiplier to search
        verbose: Print progress information

    Returns:
        IsolationResult with framework and adsorbate Atoms objects
    """
    cutoff_mult, n_components, labels = search_cutoff_for_n_components(
        atoms, target_n_components, cutoff_min, cutoff_max, verbose=verbose
    )

    # Identify components by size
    component_sizes = []
    for comp_id in range(n_components):
        mask = labels == comp_id
        component_sizes.append((comp_id, np.sum(mask)))

    # Sort by size (largest first)
    component_sizes.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"\nComponent analysis:")
        for comp_id, size in component_sizes:
            comp_atoms = atoms[labels == comp_id]
            formula = comp_atoms.get_chemical_formula()
            print(f"  Component {comp_id}: {size} atoms ({formula})")

    # Largest component is framework
    framework_id = component_sizes[0][0]
    framework_mask = labels == framework_id
    framework = atoms[framework_mask]

    # All other components are adsorbates
    adsorbates = []
    for comp_id, size in component_sizes[1:]:
        ads_mask = labels == comp_id
        adsorbate = atoms[ads_mask]
        adsorbates.append(adsorbate)

    return IsolationResult(
        framework=framework,
        adsorbates=adsorbates,
        method="connectivity",
        cutoff_multiplier=cutoff_mult,
        n_components=n_components
    )


# ============================================================================
# ELEMENT-BASED ISOLATION (for Catalysts)
# ============================================================================

def isolate_by_element(atoms: Atoms, verbose: bool = False) -> IsolationResult:
    """Isolate adsorbates using element-based separation (for catalysts).

    Separates atoms based on atomic numbers. Atoms with atomic numbers
    in ADSORBATE_ELEMENTS (C, H, O, N, S) are considered adsorbates.

    Args:
        atoms: ASE Atoms object containing structure with adsorbates
        verbose: Print progress information

    Returns:
        IsolationResult with framework and adsorbate Atoms objects
    """
    numbers = atoms.numbers
    positions = atoms.positions

    # Separate framework and adsorbate by atomic number
    framework_mask = np.array([n not in ADSORBATE_ELEMENTS for n in numbers])
    ads_mask = ~framework_mask

    framework = atoms[framework_mask]
    adsorbate_atoms = atoms[ads_mask]

    # Treat all adsorbate atoms as a single adsorbate group
    adsorbates = [adsorbate_atoms] if len(adsorbate_atoms) > 0 else []

    if verbose:
        print(f"Element-based isolation:")
        print(f"  Framework: {len(framework)} atoms ({framework.get_chemical_formula()})")
        if adsorbates:
            print(f"  Adsorbate: {len(adsorbate_atoms)} atoms ({adsorbate_atoms.get_chemical_formula()})")

    return IsolationResult(
        framework=framework,
        adsorbates=adsorbates,
        method="element"
    )


# ============================================================================
# UNIFIED ISOLATION FUNCTION
# ============================================================================

def isolate_adsorbates(
    atoms: Atoms,
    method: str = "connectivity",
    target_n_components: int = 3,
    cutoff_min: float = 0.5,
    cutoff_max: float = 2.0,
    verbose: bool = False
) -> IsolationResult:
    """Unified adsorbate isolation supporting both MOF and catalyst structures.

    Args:
        atoms: ASE Atoms object containing structure with adsorbates
        method: Isolation method - "connectivity" (MOF) or "element" (catalyst)
        target_n_components: For connectivity method, expected number of components
        cutoff_min: For connectivity method, minimum cutoff multiplier
        cutoff_max: For connectivity method, maximum cutoff multiplier
        verbose: Print progress information

    Returns:
        IsolationResult with framework and adsorbate Atoms objects
    """
    if method == "connectivity":
        return isolate_by_connectivity(
            atoms,
            target_n_components=target_n_components,
            cutoff_min=cutoff_min,
            cutoff_max=cutoff_max,
            verbose=verbose
        )
    elif method == "element":
        return isolate_by_element(atoms, verbose=verbose)
    else:
        raise ValueError(f"Unknown isolation method: {method}. Use 'connectivity' or 'element'.")


def load_and_isolate(
    file_path: str,
    method: str = "connectivity",
    supercell_matrix: Optional[List] = None,
    isolation_config: Optional[dict] = None,
    center_adsorbate: bool = True,
    adsorbate_shift: Optional[Tuple[float, float, float]] = None,
    adsorbate_index: Optional[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load structure file and isolate adsorbates with optional supercell.

    Args:
        file_path: Path to structure file (CIF, XYZ, etc.)
        method: Isolation method - "connectivity" (MOF) or "element" (catalyst)
        supercell_matrix: Optional 3x3 matrix for framework supercell
        isolation_config: Config dict for connectivity isolation
        center_adsorbate: Whether to center adsorbate on supercell (for catalysts)
        adsorbate_shift: Periodic shift (na, nb, nc) for adsorbate positioning
        adsorbate_index: Index of adsorbate to keep (None = all, 0 = first, etc.)
        verbose: Print progress information

    Returns:
        Tuple of (framework_pos, framework_nums, ads_pos, ads_nums, formula)
    """
    atoms = read(file_path)
    formula = atoms.get_chemical_formula()
    cell = atoms.get_cell()

    # Get isolation parameters
    iso_config = isolation_config or {}
    target_components = iso_config.get('target_components', 2)
    cutoff_min = iso_config.get('cutoff_min', 0.5)
    cutoff_max = iso_config.get('cutoff_max', 2.0)

    # Isolate adsorbates
    result = isolate_adsorbates(
        atoms,
        method=method,
        target_n_components=target_components,
        cutoff_min=cutoff_min,
        cutoff_max=cutoff_max,
        verbose=verbose
    )

    framework_atoms = result.framework

    # Select adsorbates to keep
    if result.adsorbates:
        if adsorbate_index is not None:
            # Keep only specified adsorbate
            if 0 <= adsorbate_index < len(result.adsorbates):
                selected_adsorbates = [result.adsorbates[adsorbate_index]]
            else:
                print(f"Warning: adsorbate_index {adsorbate_index} out of range, using first")
                selected_adsorbates = [result.adsorbates[0]]
        else:
            # Keep all adsorbates
            selected_adsorbates = result.adsorbates

        # Combine selected adsorbates
        ads_atoms = selected_adsorbates[0].copy()
        for ads in selected_adsorbates[1:]:
            ads_atoms += ads
        ads_pos = ads_atoms.positions.copy()
        ads_nums = ads_atoms.numbers.copy()
    else:
        ads_pos = np.array([]).reshape(0, 3)
        ads_nums = np.array([])

    # Apply periodic shift to adsorbate
    if adsorbate_shift is not None and len(ads_pos) > 0:
        na, nb, nc = adsorbate_shift
        shift_cartesian = na * cell[0] + nb * cell[1] + nc * cell[2]
        ads_pos += shift_cartesian

    # Create supercell of framework
    if supercell_matrix is not None:
        P = np.array(supercell_matrix)
        framework_atoms = make_supercell(framework_atoms, P)

        # Center adsorbate on supercell (for catalysts)
        if center_adsorbate and len(ads_pos) > 0 and method == "element":
            framework_pos = framework_atoms.positions.copy()
            slab_center_xy = framework_pos[:, :2].mean(axis=0)
            ads_center_xy = ads_pos[:, :2].mean(axis=0)
            ads_pos[:, :2] += slab_center_xy - ads_center_xy

    framework_pos = framework_atoms.positions.copy()
    framework_nums = framework_atoms.numbers.copy()

    # Center at origin
    all_pos = np.vstack([framework_pos, ads_pos]) if len(ads_pos) > 0 else framework_pos
    center = all_pos.mean(axis=0)
    framework_pos -= center
    if len(ads_pos) > 0:
        ads_pos -= center

    return framework_pos, framework_nums, ads_pos, ads_nums, formula


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main(config_path: str = None):
    """Main entry point for adsorbate isolation."""
    import yaml
    import sys

    # Load config
    if config_path is None:
        config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/isolate_adsorbate.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)

    # Load structure
    input_path = Path(config['input_file'])
    print(f"Loading structure from: {input_path}")
    atoms = read(input_path)
    print(f"Loaded {len(atoms)} atoms: {atoms.get_chemical_formula()}")

    # Get isolation parameters
    isolation_config = config.get('isolation', {})
    method = isolation_config.get('method', 'connectivity')
    target_components = isolation_config.get('target_components', 3)
    cutoff_min = isolation_config.get('cutoff_min', 0.5)
    cutoff_max = isolation_config.get('cutoff_max', 2.0)

    # Isolate adsorbates
    result = isolate_adsorbates(
        atoms,
        method=method,
        target_n_components=target_components,
        cutoff_min=cutoff_min,
        cutoff_max=cutoff_max,
        verbose=True
    )

    # Save results
    print(f"\n{'='*60}")
    print("Isolation Results:")
    print(f"{'='*60}")
    print(f"Method: {result.method}")
    if result.cutoff_multiplier is not None:
        print(f"Cutoff multiplier: {result.cutoff_multiplier:.4f}")
    if result.n_components is not None:
        print(f"Number of components: {result.n_components}")
    print(f"Framework: {len(result.framework)} atoms ({result.framework.get_chemical_formula()})")

    # Save framework
    framework_path = output_dir / 'results' / 'framework.cif'
    write(framework_path, result.framework)
    print(f"Saved framework to: {framework_path}")

    # Save adsorbates
    for i, ads in enumerate(result.adsorbates):
        ads_path = output_dir / 'results' / f'adsorbate_{i+1}.cif'
        write(ads_path, ads)
        print(f"Adsorbate {i+1}: {len(ads)} atoms ({ads.get_chemical_formula()}) -> {ads_path}")

    # Save combined adsorbates
    if len(result.adsorbates) > 1:
        combined_ads = result.adsorbates[0].copy()
        for ads in result.adsorbates[1:]:
            combined_ads += ads
        combined_path = output_dir / 'results' / 'adsorbates_combined.cif'
        write(combined_path, combined_ads)
        print(f"Combined adsorbates: {combined_path}")

    # Save summary
    import yaml as yaml_module
    summary = {
        'input_file': str(input_path),
        'method': result.method,
        'framework_atoms': int(len(result.framework)),
        'framework_formula': result.framework.get_chemical_formula(),
        'n_adsorbates': len(result.adsorbates),
        'adsorbate_formulas': [ads.get_chemical_formula() for ads in result.adsorbates]
    }
    if result.cutoff_multiplier is not None:
        summary['cutoff_multiplier'] = float(result.cutoff_multiplier)
    if result.n_components is not None:
        summary['n_components'] = int(result.n_components)

    summary_path = output_dir / 'results' / 'isolation_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml_module.dump(summary, f, default_flow_style=False)
    print(f"\nSummary saved to: {summary_path}")

    return result


if __name__ == "__main__":
    main()
