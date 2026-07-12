<p align="center">
  <h1 align="center">Bourbaki</h1>
  <p align="center">
    <em>Physics, chemistry, mechanics, and materials science are all different instantiations of the same set of mathematical structures</em>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/bourbaki/">
    <img src="https://img.shields.io/pypi/v/bourbaki?color=blue" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/bourbaki/">
    <img src="https://img.shields.io/pypi/pyversions/bourbaki.svg" alt="Python Versions">
  </a>
  <a href="https://github.com/toki0413/math-anything/blob/main/math-anything/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
  <a href="https://github.com/toki0413/math-anything/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/toki0413/math-anything/ci.yml?branch=main" alt="CI">
  </a>
</p>

---

## The Core Proposition

DFT, MD, CFD, FEM — these are not different software tools. They are different **morphism chains** applied to the same **conservation fields**.

```
                    ┌─────────────────────────────────────────────┐
  Layer 3           │  Domain Instantiation                       │
  (Physics/ML)      │  DFT  │  CFD  │  MD  │  FEM  │  EM  │  QC  │  phase_field  │  supervised_learning │
                    │  (different morphism chains, same base)     │
                    ├─────────────────────────────────────────────┤
  Layer 2           │  Mathematical Structures                    │
  (Types)           │  Conservation fields, morphisms, invariants  │
                    ├─────────────────────────────────────────────┤
  Layer 1           │  Foundation                                 │
  (Algorithms)      │  Category theory, type theory, constraints  │
                    └─────────────────────────────────────────────┘
```

What this means: when you run a DFT calculation and your colleague runs an MD simulation, you're both solving different approximations of the same mathematical structure. The conservation laws you share (energy, particle number) are preserved by both morphism chains. The properties you lose (quantum correlations in MD, classical limit in DFT) are determined by which morphisms you apply.

## Example: DFT vs MD

| | DFT | MD |
|---|---|---|
| **Conservation field** | Kohn-Sham equations | Hamiltonian mechanics |
| **Shared invariants** | Energy, particle number | Energy, particle number |
| **Morphism chain** | Born-Oppenheimer → Kohn-Sham → XC approx → Pseudopotential → Basis set → k-points | Classical limit → Force field → Cutoff → Thermostat → Time integration |
| **Lost in DFT** | Nuclear quantum effects, exact XC, core electrons | — |
| **Lost in MD** | — | Quantum tunneling, zero-point energy, electronic structure |
| **Emerged in DFT** | Self-consistency, basis convergence | — |
| **Emerged in MD** | — | Ensemble distribution, ergodic hypothesis |

Both start from the same Schrödinger equation. Both preserve energy and particle number. But the morphism chains diverge: DFT keeps quantum mechanics (in approximate form) while MD drops it entirely.

## Why Bourbaki?

Traditional parsers give you parameter values. Bourbaki gives you the **math** — and tells you what the math *loses*.

```
Traditional:  ENCUT = 520
Bourbaki:     ENCUT = 520
              Constraint: ENCUT > 0 ✓
              Constraint: ENCUT > max(ENMAX) ✓
              Relationship: ENCUT = ENMAX × factor(PREC)
              Semantics: "Plane-wave energy cutoff for basis set expansion"
              Domain: DFT → morphism "plane_wave_basis" → lost: completeness
```

## Features

- **8 physics and ML domains** — DFT, CFD, MD, FEM, EM, QC, phase field, and supervised learning, each as a morphism chain over conservation fields
- **18 conservation matrix field equations** — Navier-Stokes, Euler, Schrödinger, Maxwell, elasticity, MHD, heat, Dirac, Einstein field, and more, each with Noether correspondence
- **5-layer verification pipeline** — Syntax → Semantics → Invariants → Conservation → Completeness
- **12 Rust-accelerated functions** — EML operator, closure computation, Buckingham π groups, category graph traversal, expression simplification, and more via `math_anything_rs`
- **MCP server with 13 tools** — Domain analysis, cross-domain comparison, conservation fields, morphism chains, dimensional analysis, Riemann geometry, topology/loop/homotopy/curvature, ML surrogate, training-trajectory curvature, optimization-landscape homotopy, transfer learning as natural transformation, surrogate backends, structure verification, and more

## Quick Start

```bash
pip install bourbaki[mcp]
bourbaki-mcp
```

### MCP Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "bourbaki": {
      "command": "python",
      "args": ["-m", "server.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/math-anything"
      }
    }
  }
}
```

For **uv** users:

```json
{
  "mcpServers": {
    "bourbaki": {
      "command": "uv",
      "args": ["run", "--with", "mcp", "python", "-m", "server.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/math-anything"
      }
    }
  }
}
```

See [docs/mcp.md](docs/mcp.md) for full configuration options (Cursor, Windsurf, SSE transport).

## Python API

```python
from math_anything import MathAnything, extract, discover_equation

# High-level API
ma = MathAnything()
result = ma.extract("vasp", {"ENCUT": 520, "SIGMA": 0.05, "EDIFF": 1e-6})
print(result.mathematical_structure.canonical_form)
# H[n]ψ = εψ

# One-liner
schema = extract("lammps", {"timestep": 0.001, "fix": "nvt"})

# Equation discovery
discovery = discover_equation(data_columns="T,V,P", max_complexity=5)
```

### Domain Analysis

```python
from math_anything.domains import DOMAIN_REGISTRY

# Analyze DFT domain
dft = DOMAIN_REGISTRY["dft"]({"xc_functional": "PBE", "ecutwfc": 50.0})
analysis = dft.analyze()

print(f"Preserved: {analysis.preserved}")
# ['energy_conservation', 'particle_number_conservation', 'norm_conservation']

print(f"Lost: {analysis.lost}")
# ['nuclear_quantum_effects', 'explicit_electron_correlation', ...]

print(f"Emerged: {analysis.emerged}")
# ['adiabatic_approximation', 'self_consistency', ...]

# Compare DFT vs MD
md = DOMAIN_REGISTRY["md"]({"force_field": "Lennard-Jones"})
comparison = dft.compare_with(md)
print(f"Common preserved: {comparison['common_preserved']}")
# ['energy_conservation', 'particle_number_conservation']
```

### Verification

```python
from math_anything import verify_mathematical_structure

report = verify_mathematical_structure(schema)
# Checks: governing equations, boundary conditions,
# conservation properties, discretization consistency
```

### Cross-Engine Comparison

```python
from math_anything import MathAnything

ma = MathAnything()
vasp_schema = ma.extract("vasp", vasp_params)
qe_schema = ma.extract("qe", qe_params)

diff = ma.diff(vasp_schema, qe_schema)
# Semantic diff: what changed mathematically, not just what lines differ
```

### Machine Learning as Mathematical Structure

```python
from math_anything.mcp_server import analyze_ml_model

report = analyze_ml_model(
    input_dim=2,
    output_dim=1,
    architecture="mlp",
    loss="mse",
    compare_paths=True,
    transfer=True,
    backend="numpy",
)
```

## Architecture

Bourbaki implements a **3-layer architecture**:

```
Layer 1  Foundation     — Algorithms, category theory, type theory, constraint propagation
Layer 2  Structures     — Mathematical structure type system (conservation fields, morphisms)
Layer 3  Domains        — Physics/ML discipline instantiations (DFT, CFD, MD, FEM, EM, QC, phase field, supervised learning)
```

Each layer builds on the one below. Domains are *fibers* over the base of mathematical structures — DFT, CFD, MD, FEM, EM, QC, phase field, and supervised learning are all sections of the same sheaf.

### Morphism Chain

Approximations are modeled as **morphisms** between mathematical structures. Each morphism explicitly records what invariants are kept, lost, and introduced:

```
Born-Oppenheimer → Kohn-Sham → Plane-wave truncation → Pseudopotential
   (kept: total energy)  (lost: exact kinetic)  (lost: high-frequency)
```

This lets you trace exactly where a physical property was sacrificed.

## Supported Domains

| Domain | Equation Type | Morphism Chain Length | Key Invariants |
|--------|--------------|----------------------|----------------|
| DFT | Self-consistent (Kohn-Sham) | 6 | Energy, particle number, norm |
| CFD | Navier-Stokes | 4 | Mass, momentum |
| MD | Hamiltonian | 5 | Energy, momentum, angular momentum |
| FEM | Variational | 4 | Variational consistency, Galerkin orthogonality |
| EM | Maxwell | 2 | Charge conservation |
| QC | Many-electron Schrödinger | 4 | Particle number conservation |
| phase_field | Cahn-Hilliard / Allen-Cahn | 3 | Free-energy dissipation |
| supervised_learning | Function approximation | 5 | Generalization gap, optimization landscape |

## Supported Engines

| Category | Engines |
|----------|---------|
| Quantum Mechanics | VASP, Quantum ESPRESSO, CP2K, Gaussian, GAMESS, NWChem, Multiwfn |
| Molecular Dynamics | LAMMPS, GROMACS, LIGGGHTS |
| Continuum Mechanics | Abaqus, Ansys, COMSOL, SolidWorks |
| CFD | OpenFOAM, Fluent, SU2 |
| Uncertainty Quantification | Dakota |
| Multiscale | Voxel |

## Development

```bash
git clone https://github.com/toki0413/math-anything.git
cd math-anything/math-anything

# Python dev environment
pip install -e ".[dev,mcp]"

# Rust extension (optional, for acceleration)
pip install maturin
maturin develop --release

# Run tests
python -m pytest tests/unit/ -v --tb=short --cov=math_anything
python -m pytest tests/integration/ -v --tb=short

# Lint & typecheck
ruff check math_anything/
mypy math_anything/ --ignore-missing-imports
```

## License

[MIT](LICENSE) © 2024-2025 Bourbaki Project
