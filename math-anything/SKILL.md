# Bourbaki — Mathematical Structure Modeling for Computational Science

## Overview

Bourbaki reveals that different physics disciplines (DFT, CFD, MD, FEM, EM, QC, PhaseField) are instantiations of the same mathematical structures. It traces how invariants are preserved, lost, or introduced through approximation chains.

**Core proposition**: Physical simulations across disciplines share the same mathematical foundation — conservation fields, morphism chains, and type-theoretic verification.

## Architecture (3 Layers)

```
Foundation (algorithms) → Structures (types) → Domains (physics)
```

- **Foundation**: Buckingham Pi dimensional analysis, PSRN symbolic regression, MLTT type checking, Rust acceleration
- **Structures**: 18 conservation matrix fields, morphism chains, Riemannian geometry, spectral problems, continuum mechanics
- **Domains**: 7 physics domains (DFT, CFD, MD, FEM, EM, QC, PhaseField) as specific configurations of structures

## When to Use

- Analyzing mathematical structure of a simulation setup
- Comparing different physics domains at the structural level
- Checking dimensional consistency of equations
- Discovering equations from data via symbolic regression
- Verifying mathematical structures through type theory
- Computing conservation fields, Riemannian geometry, or numerical solutions
- Translating engine-specific parameters to domain-agnostic parameters

## Quick Reference

### Domain Analysis

```python
from math_anything.domains import DOMAIN_REGISTRY

# Analyze a domain
dft = DOMAIN_REGISTRY["dft"]({"ecutwfc": 500, "n_kpoints": 12})
analysis = dft.analyze()
print(analysis.to_dict())

# Compare two domains
comparison = DOMAIN_REGISTRY["dft"]().compare_with(DOMAIN_REGISTRY["md"]())
print(comparison)

# List all domains
from math_anything.domains import list_domains
print(list_domains())  # ['cfd', 'dft', 'em', 'fem', 'md', 'phase_field', 'qc']
```

### Conservation Fields

```python
from math_anything.structures.conservation_field import ConservationMatrixField

field = ConservationMatrixField()
field.build_from_navier_stokes(mu=0.01)
print(field.conserved_quantities)
print(field.coupling_matrix)
print(field.eigenvalues)
```

### Morphism Chains

```python
from math_anything.morphisms import BornOppenheimerApproximation, KohnShamMapping

# Apply morphisms to a state
state = {"n_electrons": 10, "explicit_correlation": True}
bo = BornOppenheimerApproximation()
state = bo.apply(state)  # Separates electronic/nuclear DOF

ks = KohnShamMapping()
state = ks.apply(state)  # Replaces many-body with non-interacting

# Compose morphisms
composed = ks.compose(bo)
state = composed.apply({"n_electrons": 10})
```

### Numerical Solvers

```python
import numpy as np

# Symplectic integrator (Hamiltonian systems)
from math_anything.structures.evolution import SymplecticIntegrator
H = lambda q, p: 0.5 * np.sum(p**2) + 0.5 * np.sum(q**2)
integrator = SymplecticIntegrator(H, dim=1)
result = integrator.integrate(np.array([1.0]), np.array([0.0]), dt=0.01, n_steps=1000)

# Eigenvalue solver
from math_anything.structures.spectral import EigenvalueSolver
M = np.array([[2, 1], [1, 3]])
solver = EigenvalueSolver(M)
print(solver.eigenvalues())

# SCF solver
from math_anything.structures.spectral import SelfConsistentSolver
def H_builder(density):
    return np.array([[-1, 0.5], [0.5, 1]]) + 0.5 * density
scf = SelfConsistentSolver(H_builder, n_states=1)
result = scf.solve(np.eye(2) * 0.5)

# Variational solver (1D Poisson)
from math_anything.structures.equilibrium import VariationalSolver
vs = VariationalSolver()
result = vs.solve_1d_poisson(n_elements=20)

# Conservation law solver
from math_anything.structures.evolution import ConservationLawSolver
cls = ConservationLawSolver(lambda U: U, n_vars=1)
print(cls.characteristic_speeds(np.array([1.0])))

# Continuum mechanics
from math_anything.structures.geometry_continuum import DeformationGradient
F = np.array([[1.0, 0.5, 0], [0, 1.0, 0], [0, 0, 1.0]])
dg = DeformationGradient(F)
print(dg.green_lagrange_strain())
print(dg.von_mises_stress(100, 50))
```

### Dimensional Analysis

```python
from math_anything.dimensional.equation_checker import SymbolicDimensionalAnalyzer
analyzer = SymbolicDimensionalAnalyzer()
result = analyzer.check_equation("rho * v * v", "p")  # ρv² = p → consistent

from math_anything.dimensional.scaling_group import BuckinghamPiEngine
bpe = BuckinghamPiEngine()
# Compute Pi groups from dimensional matrix
```

### Group Theory

```python
from math_anything.structures.groups import character_table_oh, CharacterTable

ct = character_table_oh()  # Oh point group
print(ct.verify_orthogonality())
print(ct.degeneracy("Eg"))  # 2
print(ct.selection_rules("A1g", "T1u", "T1u"))  # transition allowed?

# Decompose a representation
import numpy as np
chars = np.array([3, 0, -1, 1, -1, -3, -1, 0, 1, 1])
print(ct.decompose_representation(chars))
```

### Type Theory Verification

```python
from math_anything.type_theory.hott import UnivalenceVerifier

verifier = UnivalenceVerifier()
result = verifier.verify_equivalence_instance(
    name="bool_negation",
    f=lambda x: not x,
    g=lambda x: not x,
    test_data_A=[True, False],
    test_data_B=[True, False],
)
print(result["is_equivalence"])  # True

# h-level computation
levels = verifier.compute_h_level(
    equality_test=lambda a, b: a == b,
    elements=[True, False],
)
print(levels)  # h_level: 2 (set)
```

### Engine Parameter Extraction

```python
from math_anything import MathAnything

ma = MathAnything()

# Extract mathematical structure from engine parameters
result = ma.extract("vasp", {"ENCUT": 500, "EDIFF": 1e-6})
print(result.schema["mathematical_structure"]["canonical_form"])
# → "H[n]ψ = εψ" (nonlinear eigenvalue problem)

# Extract from input files
result = ma.extract_file("vasp", "INCAR")
print(result.to_mermaid())  # Visualize as diagram

# List supported engines
print(ma.supported_engines)
```

## Key Concepts

- **Conservation Field**: The mathematical operator dU/dt + div(F(U)) = S(U) with coupling matrix, flux tensor, and eigenvalues
- **Morphism Chain**: Sequence of approximations connecting fundamental equations to computable forms, tracking invariant changes
- **Domain**: A physics discipline as a specific configuration of conservation fields and morphism chains
- **Invariant Tracking**: At each morphism step, invariants are classified as preserved, lost, or introduced

## 7 Domains

| Domain | Equation | Key Morphisms |
|--------|----------|---------------|
| DFT | Kohn-Sham | BO → KS → PW → SCF → XC |
| CFD | Navier-Stokes | Incompressibility → Reynolds → Turbulence → LES |
| MD | Newton/Lagrangian | Classical limit → Force field |
| FEM | Variational | Weak form → Discretization → Assembly |
| EM | Maxwell | Frequency domain → Quasi-static → FDTD/FEM → PML |
| QC | Many-electron Schrödinger | BO → HF → Basis → Post-HF/DFT → Relativistic |
| PhaseField | Cahn-Hilliard/Allen-Cahn | Diffuse interface → CH/AC → Anisotropy → Coupling |
