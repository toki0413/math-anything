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
- **Generates mathematical propositions**: formulates existence, uniqueness, stability theorems
- **Provides tiered analysis**: 5 levels from quick screening to complete mathematical framework

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

### Installation

**From GitHub (International users)**
```bash
git clone https://github.com/toki0413/math-anything.git
cd math-anything
pip install -e .
```

**From Gitee (China users, faster)**
```bash
git clone https://gitee.com/crested-ibis-0413/math-anything.git
cd math-anything
pip install -e .
```

### CLI Usage

```bash
# Interactive REPL
math-anything repl

# One-shot extraction
math-anything extract vasp INCAR POSCAR KPOINTS --output schema.json

# Compare two calculations
math-anything diff calc1.json calc2.json

# Cross-engine mapping
math-anything cross vasp_schema.json quantum_espresso
```

### Python API

#### Basic Extraction

```python
from math_anything import extract, MathAnything

# Simple extraction
result = extract("vasp", {"ENCUT": 520, "SIGMA": 0.05})
print(result.schema["mathematical_structure"]["canonical_form"])
# Output: H[n]ψ = εψ

# With file parsing
ma = MathAnything()
result = ma.extract_file("lammps", "in.file")
print(result.to_mermaid())  # Visualize as diagram
```

#### Tiered Analysis (New)

```python
from math_anything import TieredAnalyzer, AnalysisTier, tiered_analyze

# Auto-detect analysis level based on complexity
result = tiered_analyze("large_simulation.lmp")
print(f"Recommended tier: {result.tier.name}")
# Output: ADVANCED (for large system with long simulation)

# Or specify exact tier
analyzer = TieredAnalyzer()
result = analyzer.analyze("simulation.lmp", tier=AnalysisTier.PROFESSIONAL)

# Get recommendation without running
rec = analyzer.get_recommendation("simulation.lmp")
print(f"Complexity score: {rec.complexity_score.total}")
print(f"Estimated time: {rec.estimated_time}")
print(f"Suitable tiers: {[t.name for t in rec.suitable_tiers]}")
```

#### Mathematical Proposition Generation

```python
from math_anything import PropositionGenerator, MathematicalTask, TaskType

# Generate mathematical propositions from simulation
extractor = PropositionGenerator()
propositions = extractor.generate(
    engine="lammps",
    parameters={"timestep": 0.5, "run": 80000, "fix": "nvt"},
    task_type=TaskType.WELL_POSEDNESS
)

print(propositions)
# Output: Theorem (Well-posedness of MD simulation):
#   Given m·r̈ = F(r) with Lipschitz continuous F,
#   and initial conditions r(0) = r₀, ṙ(0) = v₀,
#   there exists a unique solution for t ∈ [0, T].
```

## Tiered Analysis System

Math Anything provides 5 levels of analysis depth, automatically selected based on system complexity:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Tiered Analysis Levels                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Level 1: Basic        → Quick screening, simple feature extraction     │
│  Level 2: Enhanced     → Detailed parameters and validation             │
│  Level 3: Professional → + Topology analysis (Betti numbers)            │
│  Level 4: Advanced     → + Geometric methods (symplectic integrator)    │
│  Level 5: Complete     → Five-layer unified framework + latent space    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Five-Layer Unified Framework (Level 5)

The complete analysis combines advanced mathematical methods:

1. **Symplectic Integrator**: Preserves energy and phase space structure (O(Δt²) error)
2. **Constrained Manifold**: Reduces dimensionality for systems with constraints
3. **Topology Analysis**: Betti numbers identify connected components, loops, voids
4. **Morse Theory**: Critical points analysis of energy landscape
5. **Latent Space**: E(3)-equivariant neural representations for acceleration

```python
# Level 5: Complete analysis with all methods
result = analyzer.analyze("complex_system.lmp", tier=AnalysisTier.COMPLETE)

print(result.topology_info)    # Betti numbers: [1, 0, 0]
print(result.manifold_info)    # Dimension, metric, symplectic structure
print(result.morse_info)       # Critical points of energy landscape
print(result.latent_info)      # Recommended encoder, speedup estimate
```

## Real use cases

### Catch bad inputs early

```
✓ ENCUT > 0
✓ EDIFF > 0
✗ SIGMA > 0  ← SIGMA = -0.2 is invalid!
```

### Understand what you're computing

```
VASP isn't just "running DFT":
  Problem Type: nonlinear_eigenvalue
  Canonical Form: H[n]ψ = εψ
  Variable Dependencies: V_eff → n → ψ → V_eff (circular)
  → SCF iteration required
```

### Decrypt black-box calculations

```
LAMMPS input script:
  Core Problem: initial_value_ode
  Approximations: classical mechanics → force field → cutoff
  Hierarchy: quantum → Born-Oppenheimer → classical → empirical
```

### Guide ML with physics context

```
ML model predicting formation energies:
  Approximating: DFT total energy calculation
  Missing: explicit physics constraints (symmetry, conservation)
  
Recommendation based on E(3) symmetry:
  → Use SchNet or NequIP (E(3)-equivariant networks)
  → Compatibility proof: message passing rotation-invariant
```

### Compare across physics scales

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

**Tiered complexity**: From quick checks to deep mathematical analysis, match the effort to the problem.

## Documentation

- [QUICK_START.md](QUICK_START.md) - Get started in 5 minutes
- [WORKFLOW.md](WORKFLOW.md) - Complete workflow from file to math proposition
- [TIERED_SYSTEM.md](TIERED_SYSTEM.md) - Tiered analysis system design
- [UNIFIED_MATH_FRAMEWORK.md](UNIFIED_MATH_FRAMEWORK.md) - Five-layer unified framework
- [TOPOLOGY_MANIFOLD_ANALYSIS.md](TOPOLOGY_MANIFOLD_ANALYSIS.md) - Topology and manifold efficiency analysis
- [LATENT_SPACE_ANALYSIS.md](LATENT_SPACE_ANALYSIS.md) - Latent space acceleration analysis

## Acknowledgments

Inspired by [CLI-Anything](https://github.com/fzdwx/cli-anything), which showed that CLI tools can be made intelligible to AI agents through structured extraction. We extend this from CLI semantics to mathematical semantics.

The EML (Exp-Minus-Log) symbolic regression implementation is based on the work by **Andrzej Odrzywołek** and his paper *"All elementary functions from a single binary operator"* (arXiv:2603.21852), which demonstrates that all elementary functions can be constructed from a single binary operator `eml(x,y) = exp(x) - ln(y)`.

## License

MIT
