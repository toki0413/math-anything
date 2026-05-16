# Changelog

All notable changes to Math-Anything will be documented in this file.

## [2.1.0] - 2026-05-16

### Added

#### Analysis Tools (P1)
- **SINDyDiscoverer** (`tools/sindy.py`): Sparse Identification of Nonlinear Dynamics — ODE/PDE equation discovery from time series data
  - PySINDy backend (optional) with numpy-based STLSQ fallback
  - PDE discovery via finite differences + sparse regression
  - Pipeline: DynamicsAnalyzer detects chaos → SINDy discovers equations → PSRN validates
- **SymmetryAnalyzer** (`tools/symmetry.py`): Group theory analysis for crystal structures
  - Space group detection via spglib
  - Irreducible representations, character tables, selection rules
  - Fallback from space group hint when no lattice data
- **TDAAnalyzer** (`tools/tda.py`): Topological Data Analysis
  - Persistent homology via ripser
  - Cubical complex via gudhi
  - Betti numbers, persistence entropy, bottleneck distance
- **SpectralAnalyzer** (`tools/spectral.py`): Spectral analysis
  - DOS computation with Gaussian broadening
  - Band gap detection (direct/indirect)
  - Z₂ topological invariant via pythtb
- **DynamicsAnalyzer** (`tools/dynamics.py`): Dynamical systems analysis
  - Lyapunov exponents via nolds
  - Fractal dimension, Hurst exponent
  - DMD modal decomposition via pydmd
  - Chaos detection and classification

#### Advanced Features (P3)
- **ProvenanceTracker** (`provenance.py`): Computation provenance tracking with parent-child relationships and confidence decay
- **ReferenceTracker** (`references.py`): Literature reference lookup for VASP/LAMMPS/Abaqus constraints
- **MLPotentialAnalyzer** (`tools/ml_potential.py`): Mathematical structure analysis of ML interatomic potentials (DeepMD, MACE, NequIP)
- **LanglandsAnalyzer** (`tools/langlands.py`): Langlands Program computations — Galois groups, L-functions, representation theory for materials science
- **SandboxExecutor** (`sandbox.py`): Isolated code execution environment
  - E2B cloud sandbox backend (optional)
  - Local subprocess sandbox with import restrictions (Python 3.13 compatible)
  - Constraint validation scripts

#### Visualization (P2)
- **InteractiveVisualizer** (`tools/viz.py`): Plotly-based interactive scientific visualization
  - DOS plots, persistence diagrams, phase portraits
  - SINDy coefficient heatmaps
  - 3D Riemannian manifold surfaces
  - Brillouin zone rendering
- **GeometryPage 3D rendering**: Manifold surface and Brillouin zone visualization in frontend
- **SchemaPage enhancement**: KaTeX formula rendering, constraint severity badges (critical/warning/info), provenance chain display

#### Frontend
- **AnalysisPage**: 5-tab interface (Symmetry / Spectral / Dynamics / SINDy / TDA) with interactive Plotly visualization
- **SandboxPage**: Code editor + output display for constraint validation
- API client methods for all new endpoints

#### Infrastructure
- **Tool Registry auto-discovery**: 8 analysis tools automatically registered via `auto_discover_analysis_tools()`
- **Pipeline integration tests**: End-to-end tests covering extraction → analysis → visualization
- **Optional dependency groups**: `sindy`, `langlands`, `sandbox` added to pyproject.toml

### Fixed
- **Christoffel symbol calculation**: Partial derivatives were always zero due to incorrect computation; now uses numerical differentiation with central finite differences
- **MetricTensor**: Added `at(coords)` method for coordinate-dependent evaluation; inverse computed at evaluation point instead of reference point
- **Advisor discipline filtering**: `DISCIPLINE_STATUS` now correctly marks `statistical_mechanics`, `stochastic_processes`, `information_theory` as implemented
- **PSRN `_expr_to_node()`**: Replaced placeholder with recursive descent parser for expression strings
- **Mermaid visualization**: Replaced `\\n` with `<br/>` for proper newline rendering
- **Advisor engine coverage**: Added COMSOL, SolidWorks, Voxel engine entries
- **spglib API compatibility**: Uses `get_symmetry_dataset` instead of deprecated `get_spacegroup`
- **Plotly Scatter**: Fixed `colorscale` attribute (must be inside `marker` dict, not at trace level)
- **Persistence diagram**: Fixed TypeError when `to_dict()` includes non-list values (description key)
- **SINDy STLSQ**: Fixed zero-coefficient edge case when all features eliminated by threshold
- **SINDy class name**: Fixed `PDEDiscovedResult` → `PDEDiscoveredResult` typo
- **SINDy boundary checks**: Added validation for `dt <= 0` and `threshold < 0`

### Changed
- Version unified to **2.1.0** across all config files (pyproject.toml, package.json, tauri.conf.json, Cargo.toml, version.py)
- Removed conflicting `core/setup.py` (was using different package name `math-anything-core`)
- CI quality gates now enforce failures (removed `|| true` from flake8, mypy, bandit, safety)
- Release workflow now includes Tauri desktop builds for Windows/macOS/Linux
- `computation` and `all` optional dependency groups now include `sindy` and `langlands`

### Tests
- 304 tests passing (up from 257), including:
  - 17 SINDy tests
  - 20 Sandbox tests
  - 10 Pipeline integration tests
  - Updated advisor tests for new discipline status

## [2.0.0] - 2025-12-01

### Added
- Initial Tauri desktop application
- FastAPI server backend
- Mathematical structure extraction for VASP, LAMMPS, Abaqus, Multiwfn
- PSRN symbolic regression
- Geometry module (metric tensor, Christoffel symbols)
- Formal verification tools
- Cross-engine coupling analysis
- Schema extension system

## [1.0.0] - 2025-06-15

### Added
- Core mathematical structure extraction framework
- CLI interface
- Schema serialization/deserialization
- Basic constraint validation
