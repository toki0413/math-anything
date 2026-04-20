# Math Anything

A mathematical semantic layer for computational materials science. Instead of just pulling parameter values out of simulation inputs, Math Anything extracts the *equations, constraints, and relationships* behind them, so LLMs can reason about physics, not just read numbers.

## Why this exists

If you've ever worked with VASP, LAMMPS, or any other simulation code, you know the drill: the input file has `ENCUT = 520`, but what does that *mean*? Is it enough? Too much? Does it satisfy the mathematical constraints of a plane-wave DFT calculation?

Traditional parsers give you the value. Math Anything gives you the math:

```
Traditional:  ENCUT = 520
Math Anything: ENCUT = 520
               Constraint: ENCUT > 0 ✓
               Constraint: ENCUT > max(ENMAX) ✓
               Relationship: ENCUT = ENMAX × factor(PREC)
               Semantics: "Plane-wave energy cutoff for basis set expansion"
```

This matters because LLMs (and humans new to a code) can't validate what they don't understand. A symbolic constraint like `SIGMA > 0` is something an agent can check. A bare `0.05` is not.

## What it does

- **Extracts mathematical structures**: governing equations, boundary conditions, constitutive relations
- **Reveals mathematical essence**: not just "ENCUT=520" but "this is a nonlinear eigenvalue problem requiring SCF iteration"
- **Validates symbolic constraints**: are the parameters physically/mathematically consistent?
- **Maps parameters across engines**: VASP's `ENCUT` ↔ Quantum ESPRESSO's `ecutwfc`
- **Compares calculations semantically**: what changed mathematically, not just what lines differ

## Supported engines

| Engine | Type | Mathematical Problem | What LLM Understands |
|--------|------|---------------------|---------------------|
| VASP | DFT | H[n]ψ = εψ (nonlinear eigenvalue) | "Needs SCF iteration, V_eff depends on density" |
| LAMMPS | MD | m d²r/dt² = F(r) (initial value ODE) | "Time integration, not iterative solving" |
| Abaqus | FEM | ∇·σ + f = 0 (boundary value) | "FEM solving equilibrium" |
| Ansys | FEM | Kφ = λMφ (eigenvalue) | "Finding natural frequencies" |
| COMSOL | Multiphysics | Coupled PDE system | "Multiple physics coupled together" |
| GROMACS | Biomolecular MD | Stochastic ODE | "Biomolecular dynamics with constraints" |
| Multiwfn | Wavefunction | ∇ρ(r) = 0 (topological) | "Finding critical points in density" |

When you run extraction, you get the mathematical structure, variable dependencies, and the full hierarchy of approximations from physics to numerics.

## Quick start

```bash
# Install
git clone https://github.com/yourusername/math-anything.git
cd math-anything
pip install -r requirements.txt

# Interactive REPL
math-anything repl

# One-shot extraction
math-anything extract vasp INCAR POSCAR KPOINTS --output schema.json

# Compare two calculations
math-anything diff calc1.json calc2.json

# Cross-engine mapping
math-anything cross vasp_schema.json quantum_espresso
```

Python API:

```python
from math_anything.vasp.core.extractor_v2 import VaspExtractor
from math_anything.schemas import extract_vasp_mathematical_precision

extractor = VaspExtractor()
schema = extractor.extract({'incar': 'INCAR', 'poscar': 'POSCAR', 'kpoints': 'KPOINTS'})

# Get mathematical precision
precision = extract_vasp_mathematical_precision(schema.raw_symbols)
print(precision['mathematical_structure']['problem_type'])  # nonlinear_eigenvalue

# Check constraints
for c in schema.symbolic_constraints:
    print(f"{'✓' if 'SATISFIED' in c.description else '✗'} {c.expression}")
```

## Real use cases

**Catch bad inputs early**

```
✓ ENCUT > 0
✓ EDIFF > 0
✗ SIGMA > 0  ← SIGMA = -0.2 is invalid!
```

**Understand what you're computing**

```
VASP isn't just "running DFT":
  Problem Type: nonlinear_eigenvalue
  Canonical Form: H[n]ψ = εψ
  Variable Dependencies: V_eff → n → ψ → V_eff (circular)
  → SCF iteration required
```

**Decrypt black-box calculations**

```
LAMMPS input script:
  Core Problem: initial_value_ode
  Approximations: classical mechanics → force field → cutoff
  Hierarchy: quantum → Born-Oppenheimer → classical → empirical
```

**Guide ML with physics context**

```
ML model predicting formation energies:
  Approximating: DFT total energy calculation
  Missing: explicit physics constraints (symmetry, conservation)
```

**Compare across physics scales**

```
VASP (DFT):     H[n]ψ = εψ              (quantum)
LAMMPS (MD):    m d²r/dt² = F(r)        (classical)
Abaqus (FEM):   ∇·σ + f = 0             (continuum)
→ Different mathematical frameworks, careful upscaling needed
```

## Design principles

**Zero intrusion**: Never modifies your input files. Only reads and reports.

**Zero judgment**: Doesn't tell you "ENCUT=200 is wrong." Reports "ENCUT=200 is outside typical range 200-800 eV." The decision is yours.

**Mathematical precision**: Expresses structures in canonical forms. `H[n]ψ = εψ` means the same thing to any physicist.

## Acknowledgments

Inspired by [CLI-Anything](https://github.com/fzdwx/cli-anything), which showed that CLI tools can be made intelligible to AI agents through structured extraction. We extend this from CLI semantics to mathematical semantics.

## License

MIT
