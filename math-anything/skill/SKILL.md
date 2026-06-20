---
name: bourbaki
description: Mathematical structure modeling for computational science. Reveals the shared mathematical foundation across physics domains (DFT, CFD, MD, FEM, EM, QC, PhaseField) through conservation fields, morphism chains, and type-theoretic verification.
allowed-tools:
  - Read
  - Write
  - Bash
  - RunCommand
when_to_use: |
  Trigger when user mentions:
  - "mathematical structure" of a simulation
  - "domain analysis" for DFT, CFD, MD, FEM, EM, QC, PhaseField
  - "conservation law" or "invariant" tracking
  - "morphism chain" or "approximation hierarchy"
  - "dimensional analysis" or "Buckingham Pi"
  - "equation discovery" or "symbolic regression"
  - "verify" a mathematical structure or equation
  - "translate" engine parameters between VASP, QE, LAMMPS, GROMACS, Abaqus, OpenFOAM, COMSOL, etc.
  - Cross-domain comparison, e.g., "DFT vs MD" or "FEM vs FDTD"
argument-hint: "[domain/action] [parameters]"
arguments:
  - domain_or_action
  - parameters
---

# Bourbaki

A mathematical semantic layer for computational science. It treats physics domains (DFT, CFD, MD, FEM, EM, QC, PhaseField) as instantiations of the same mathematical structures: conservation fields, morphism chains, and type-theoretic types.

## Core Philosophy

**Structure over syntax**: A VASP INCAR and a Quantum ESPRESSO input can express the same mathematical object — a plane-wave DFT calculation. Bourbaki extracts that object.

**Invariants over parameters**: Instead of validating whether `ENCUT=300` is good, Bourbaki asks: does this setup preserve energy, momentum, and particle number through its approximation chain?

**Truthful about loss**: Every approximation loses something. Bourbaki tracks what is preserved, weakened, or lost at each morphism step.

## 3-Layer Architecture

```
Foundation (algorithms)
    ↓
Structures (types: conservation field, morphism, Riemannian geometry, spectral problem)
    ↓
Domains (physics: DFT, CFD, MD, FEM, EM, QC, PhaseField)
```

## 7 Domains

| Domain | Fundamental Equation | Morphism Chain |
|--------|---------------------|----------------|
| DFT | Kohn-Sham | BO → KS → PW → SCF → XC |
| CFD | Navier-Stokes | Incompressibility → Reynolds → Turbulence → LES/LES → Wall |
| MD | Newton/Euler-Lagrange | Classical limit → Force field → Integrator |
| FEM | Variational PDE | Strong form → Weak form → Discretization → Assembly |
| EM | Maxwell | Frequency domain → Quasi-static → FDTD/FEM → PML |
| QC | Many-electron Schrödinger | BO → HF → Basis → Post-HF/DFT → Relativistic |
| PhaseField | Cahn-Hilliard / Allen-Cahn | Sharp interface → Diffuse → CH/AC → Anisotropy → Mechanics |

## Actions

### 1. Analyze a Domain

Understand the mathematical structure of a physics domain.

```python
from math_anything.domains import DOMAIN_REGISTRY

dft = DOMAIN_REGISTRY["dft"]({"ecutwfc": 500, "n_kpoints": [4, 4, 4]})
analysis = dft.analyze()
print(analysis.to_dict())
```

### 2. Compare Two Domains

```python
comparison = DOMAIN_REGISTRY["dft"]().compare_with(DOMAIN_REGISTRY["md"]())
print(comparison)
```

### 3. Build a Conservation Field

```python
from math_anything.structures.conservation_field import ConservationMatrixField

field = ConservationMatrixField()
field.build_from_navier_stokes(mu=0.01)
print(field.conserved_quantities)
print(field.eigenvalues)
```

### 4. Trace a Morphism Chain

```python
from math_anything.domains import DOMAIN_REGISTRY

dom = DOMAIN_REGISTRY["dft"]()
chain = dom.build_morphism_chain()
for step in chain:
    print(step["name"], step["invariants_kept"], step["invariants_lost"])
```

### 5. Dimensional Analysis

```python
from math_anything.dimensional.equation_checker import SymbolicDimensionalAnalyzer

analyzer = SymbolicDimensionalAnalyzer()
result = analyzer.check_equation("rho * v * v", "p")
print(result)
```

### 6. Discover Equations from Data

```python
from math_anything.psrn.sindyc import SINDyC
import numpy as np

X = np.random.randn(100, 2)
y = X[:, 0] ** 2
sindyc = SINDyC()
result = sindyc.discover(X, y, variable_names=["x", "y"])
print(result)
```

### 7. Verify a Mathematical Structure

```python
from math_anything.type_theory.verify import VerificationPipeline

pipeline = VerificationPipeline()
result = pipeline.verify("mass conservation implies div(rho*v) = -drho/dt")
print(result.overall_passed)
```

### 8. Translate Engine Parameters

```python
from math_anything.adapters import translate_params

result = translate_params("vasp", {"ENCUT": 520, "EDIFF": 1e-6, "ISMEAR": 0})
print(result)  # domain-agnostic parameters
```

## MCP Tool Mapping

When Bourbaki is configured as an MCP server, use these tools:

| Tool | Purpose |
|------|---------|
| `list_domains` | Discover available physics domains |
| `analyze_domain` | Get conservation field + morphism chain for a domain |
| `compare_domains` | Cross-domain structural comparison |
| `build_conservation_field` | Build conservation matrix field for an equation |
| `analyze_morphism_chain` | Trace invariant changes through approximations |
| `compute_riemann_geometry` | Compute curvature tensors from metric/Christoffel |
| `solve_numerical` | Symplectic integrator, eigenvalue, SCF, conservation law, FEM, continuum |
| `dimensional_analyze` | Buckingham Pi + dimensional consistency |
| `discover_equations` | Symbolic regression from data |
| `verify_structure` | Multi-layer verification pipeline |
| `translate_engine_params` | Engine-specific → domain-agnostic parameters |

## Usage Patterns

**Understand a VASP input**:
```
> translate_engine_params("vasp", {"ENCUT": 520, "EDIFF": 1e-6})
> analyze_domain("dft", {"ecutwfc": 520, "scf_tol": 1e-6})
> This is a nonlinear eigenvalue problem H[n]ψ = εψ with plane-wave basis and SCF iteration.
```

**Compare DFT and MD**:
```
> compare_domains("dft", {}, "md", {})
< Both conserve energy; DFT additionally conserves particle number in KS space; MD loses quantum phase.
```

**Check if an equation is dimensionally consistent**:
```
> dimensional_analyze(expression_lhs="rho * v * v", expression_rhs="p")
< consistent: True
```

## Guardrails

- Never modifies user input files
- Never judges parameter values as "right" or "wrong"
- Always expresses structures in canonical mathematical forms
- Always reports approximation hierarchies honestly
- Never hides the mathematical consequences of approximations

## Outputs

All outputs are JSON-serializable dictionaries. They can be saved, compared across domains, fed to other tools, or used to guide ML model design with physics context.
