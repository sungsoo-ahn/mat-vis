# mat-vis: Materials Structure Visualization

A research codebase for visualizing and manipulating MOF (Metal-Organic Framework) and catalyst structures, with a focus on conditional diffusion process visualization.

## Features

- **Adsorbate Isolation**: Graph-based connectivity analysis to separate adsorbates from framework structures
- **Flow Matching Visualization**: Visualize conditional diffusion processes where adsorbates are fixed and frameworks are generated around them
- **Customizable Color Schemes**: Multiple pre-defined color schemes for framework and adsorbate atoms
- **3D Glossy Rendering**: Professional-quality 3D visualizations with shadows, highlights, and specular effects
- **Static & Animated Outputs**: Generate both static PNG frames and animated GIFs

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mat-vis

# Install dependencies using uv
uv sync
```

### Run Visualization

```bash
# MOF diffusion visualization (static frames)
bash scripts/visualization/run_mof_diffusion.sh

# Catalyst diffusion visualization (static frames)
bash scripts/visualization/run_catalyst_diffusion.sh

# Generate with GIF animation
bash scripts/visualization/run_mof_diffusion.sh --create-gif
```

### Run Adsorbate Isolation

```bash
# Isolate adsorbates from MOF structure
bash scripts/run_isolate_adsorbate.sh
```

## Project Structure

```
mat-vis/
├── src/                          # Python source code
│   ├── utils.py                  # Color schemes, element properties, utilities
│   ├── visualization.py          # Flow matching visualization pipeline
│   └── isolate_adsorbate.py      # Adsorbate isolation module
├── configs/                      # YAML configuration files
│   ├── visualization/
│   │   ├── mof_diffusion.yaml    # MOF visualization config
│   │   └── catalyst_diffusion.yaml # Catalyst visualization config
│   └── isolate_adsorbate.yaml    # Adsorbate isolation config
├── scripts/                      # Execution scripts
│   ├── visualization/
│   │   ├── run_mof_diffusion.sh
│   │   └── run_catalyst_diffusion.sh
│   └── run_isolate_adsorbate.sh
├── data/                         # Data and outputs (gitignored)
│   └── sample/                   # Sample structures (git-tracked)
│       ├── mof_clean.xyz
│       ├── co2_with_cell.xyz
│       └── catalyst/             # Sample catalyst structures
├── scratch/                      # Experimental code (gitignored)
└── README.md
```

## Modules

### Adsorbate Isolation (`src/isolate_adsorbate.py`)

Separates adsorbate molecules from framework structures using graph-based connectivity analysis.

**Algorithm:**
1. Uses covalent radii to determine atomic bonds
2. Builds adjacency matrix with ASE neighbor lists
3. Finds connected components using scipy
4. Separates largest component (framework) from smaller ones (adsorbates)

**Usage:**
```bash
bash scripts/run_isolate_adsorbate.sh
```

**Outputs:**
- `framework.xyz` - Isolated framework atoms
- `adsorbate_*.xyz` - Individual adsorbate molecules
- `summary.yaml` - Analysis summary

### Flow Matching Visualization (`src/visualization.py`)

Visualizes conditional diffusion processes for materials generation.

**Features:**
- Conditional generation: adsorbate fixed, framework generated
- DDPM-style trajectory with cosine noise schedule
- Glossy 3D rendering with shadows and highlights
- Customizable color schemes and camera angles

**Usage:**
```bash
# Static frames only
bash scripts/visualization/run_mof_diffusion.sh

# With GIF animation
bash scripts/visualization/run_mof_diffusion.sh --create-gif
```

**Outputs:**
- `*_t0.00.png` through `*_t1.00.png` - Frames at key timesteps
- `*_diffusion.gif` - Animated visualization (if enabled)

## Configuration

All experiments use YAML configuration files in `configs/`. Key parameters:

### Visualization Config

```yaml
output_dir: "data/viz_mof_diffusion"
mode: "mof"  # or "catalyst"

# Color schemes
framework_scheme: "cool_gray"
adsorbate_scheme: "mint_coral"
boundary_scheme: "thin_dark"

# MOF-specific
mof:
  framework_file: "data/sample/mof_clean.xyz"
  adsorbate_file: "data/sample/co2_with_cell.xyz"
  supercell_matrix: [[1, 0, 0], [0, 2, 0], [0, 0, 2]]
  adsorbate_shift: [0, 0, 1]
  view_elev: -10
  view_azim: 0

# Trajectory
trajectory:
  num_steps: 60
  noise_scale: 1.5

# Output
dpi: 200
create_gif: false
```

## Color Schemes

### Framework Schemes (Muted Tones)
- `cool_gray` (default) - Cool gray-blue palette
- `earthy` - Natural earth tones
- `warm_brown` - Warm brown palette
- `dark_muted` - Dark muted colors
- `slate` - Slate gray tones
- `charcoal` - Deep charcoal palette

### Adsorbate Schemes (Bright Tones)
- `mint_coral` (default) - Mint green and coral
- `neon` - High-contrast neon colors
- `soft_neon` - Softer neon palette
- `ocean_sunset` - Ocean blues and sunset warm
- `spring` - Fresh spring colors
- `arctic` - Cool arctic with warm accents
- `forest_berry` - Natural greens and berry
- `tech` - Modern tech palette

### Boundary Schemes
- `thin_dark` (default) - Thin dark boundary
- `none` - No boundary
- `medium_dark` - Medium dark boundary
- `thick_dark` - Thick dark boundary
- `colored` - Boundary matches atom color

## Sample Data

The `data/sample/` directory contains example structures:

- **MOF**: `mof_clean.xyz` (clean framework), `mof.cif` (with adsorbate)
- **Adsorbates**: `co2_with_cell.xyz`, `adsorbate_CO2.xyz`
- **Catalysts**: `catalyst/1234.cif`, `catalyst/3.cif`, etc.

## Development Workflow

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

**Key conventions:**
- All Python code in `src/`
- All configs in `configs/`
- All scripts in `scripts/`
- Use `uv run python` or activate `.venv`
- Add packages with `uv add <package>`

## Dependencies

- **Python 3.12+**
- **Core**: numpy, scipy, matplotlib, pillow
- **Materials**: ase (Atomic Simulation Environment)
- **Analysis**: jupyterlab, ipywidgets, ipympl

Install with: `uv sync`

## Examples

### Visualize MOF Diffusion Process

```bash
# Generate static frames
bash scripts/visualization/run_mof_diffusion.sh

# Output: data/viz_mof_diffusion/figures/
#   - mof_diffusion_t0.00.png (noise)
#   - mof_diffusion_t0.25.png
#   - mof_diffusion_t0.50.png
#   - mof_diffusion_t0.75.png
#   - mof_diffusion_t1.00.png (clean structure)
```

### Custom Configuration

Create a new config file:

```yaml
# configs/visualization/my_custom_viz.yaml
output_dir: "data/my_experiment"
mode: "mof"
framework_scheme: "earthy"
adsorbate_scheme: "neon"
create_gif: true
trajectory:
  num_steps: 100
  noise_scale: 2.0
```

Run with:

```bash
uv run python src/visualization.py configs/visualization/my_custom_viz.yaml
```

### Isolate Adsorbates

```bash
# Edit configs/isolate_adsorbate.yaml to set input file
bash scripts/run_isolate_adsorbate.sh

# Check results in data/isolate_adsorbate/results/
```

## Output Structure

Each run creates organized output:

```
data/<experiment_name>/
├── figures/          # Plots and visualizations
├── results/          # Data files (XYZ, YAML)
└── logs/             # Execution logs
```

## Troubleshooting

**ModuleNotFoundError**
- Ensure you're using `uv run python` or activated `.venv`
- Run `uv sync` to install dependencies

**Import errors**
- Check imports use `from src.module import ...`

**Permission denied**
- Make scripts executable: `chmod +x scripts/**/*.sh`

**Missing visualizations**
- Verify config has correct `output_dir` field
- Check input file paths exist

## Contributing

1. Follow conventions in [CLAUDE.md](CLAUDE.md)
2. Keep code in `src/`, configs in `configs/`, scripts in `scripts/`
3. Use config-driven design for reproducibility
4. Add packages with `uv add`, never `pip install`

## License

[Add license information]

## Citation

If you use this code in your research, please cite:

```
[Add citation information]
```

## Contact

Sungsoo Ahn - sungsoo.ahn@kaist.ac.kr

---

**Note**: This project follows the development guidelines in [CLAUDE.md](CLAUDE.md). See that file for detailed information about project organization, workflow, and conventions.
