---
name: math-anything
description: Extract mathematical structures from computational materials science engines (VASP, LAMMPS, Abaqus, Ansys, COMSOL, GROMACS, Multiwfn). Use when analyzing simulation inputs, comparing calculations, validating constraints, or understanding the mathematical essence behind computational setups.
allowed-tools:
  - Read
  - Write
  - Bash
when_to_use: |
  Trigger when user mentions:
  - "extract math structure" or "mathematical structure"
  - "what equations does this solve"
  - "compare VASP and LAMMPS" or cross-engine comparison
  - "validate constraints" or "check parameters"
  - "what is the math behind" any simulation engine
  - "decrypt" or "understand" a calculation setup
  - Any engine name (vasp, lammps, abaqus, ansys, comsol, gromacs, multiwfn)
argument-hint: "[engine] [action] [parameters]"
arguments:
  - engine
  - action
  - parameters
---

# Math Anything

A mathematical semantic layer for computational materials science. Extract equations, constraints, and relationships from simulation inputs so LLMs can reason about physics, not just read numbers.

## Core Philosophy

**Zero intrusion**: Never modifies your input files. Only reads and reports.

**Zero judgment**: Doesn't tell you "ENCUT=200 is wrong." Reports "ENCUT=200 is outside typical range 200-800 eV." The decision is yours.

**Mathematical precision**: Expresses structures in canonical forms. `H[n]ψ = εψ` means the same thing to any physicist.

## Supported Engines

| Engine | Type | Mathematical Problem | Canonical Form |
|--------|------|---------------------|----------------|
| VASP | DFT | Nonlinear eigenvalue | `H[n]ψ = εψ` |
| LAMMPS | MD | Initial value ODE | `m_i d²r_i/dt² = F_i` |
| Abaqus | FEM | Boundary value | `∇·σ + f = 0` |
| Ansys | FEM | Eigenvalue | `Kφ = λMφ` |
| COMSOL | Multiphysics | Coupled PDE system | Multi-physics coupling |
| GROMACS | Biomolecular MD | Stochastic ODE | Langevin dynamics |
| Multiwfn | Wavefunction | Topological analysis | `∇ρ(r) = 0` |

## Actions

### 1. Extract Mathematical Structure

Extract the complete mathematical structure from engine parameters.

**Input**: Engine name + parameter dictionary
**Output**: EnhancedMathSchema with:
- `mathematical_structure`: problem type, canonical form, properties
- `variable_dependencies`: how variables depend on each other
- `discretization_scheme`: numerical approximation method
- `solution_strategy`: solver approach and convergence criteria
- `approximations`: hierarchy of approximations from physics to numerics
- `mathematical_decoding`: decryption of the computational setup

**Example**:
```python
# VASP extraction
extract_mathematical_structure("vasp", {
    "encut": 520,
    "ediff": 1e-6,
    "sigma": 0.05
})
# Returns: nonlinear eigenvalue problem H[n]ψ = εψ
#          with SCF iteration, plane-wave basis

# LAMMPS extraction
extract_mathematical_structure("lammps", {
    "timestep": 0.001,
    "temperature": 300
})
# Returns: initial value ODE m d²r/dt² = F(r)
#          with Velocity Verlet integration
```

### 2. Compare Calculations

Compare two mathematical schemas and report semantic differences.

**Input**: Two EnhancedMathSchema dictionaries
**Output**: Diff report with categories:
- `critical`: Changes affecting physical correctness
- `warning`: Changes requiring attention
- `info`: Informational changes

**Example**:
```python
compare_calculations(schema_v1, schema_v2, critical_only=True)
# Reports: integrator changed, conservation lost, etc.
```

### 3. Validate Constraints

Report symbolic constraints present in a schema without judging satisfaction.

**Input**: EnhancedMathSchema dictionary
**Output**: List of constraints with expressions and descriptions

**Example**:
```python
validate_constraints(schema)
# Returns: ENCUT > 0, EDIFF > 0, SIGMA > 0, etc.
```

### 4. List Supported Engines

Return all supported simulation engines.

## Usage Patterns

**Understand a calculation**:
```
> extract vasp {"encut": 520, "sigma": 0.05}
< This is a nonlinear eigenvalue problem H[n]ψ = εψ
< Requires SCF iteration due to circular dependency V_eff → n → ψ
< Plane-wave basis with cutoff 520 eV
```

**Compare two setups**:
```
> compare schema1 schema2
< Critical: Time integrator changed from symplectic to non-symplectic
< Warning: Time step increased by 10x
```

**Check constraints**:
```
> validate schema
< ENCUT = 520 > 0 ✓
< SIGMA = 0.05 > 0 ✓
```

## Guardrails

- Never modify user input files
- Never judge parameter values as "right" or "wrong"
- Always express structures in canonical mathematical forms
- Always report approximation hierarchies honestly
- Never hide the mathematical consequences of approximations

## Outputs

All outputs are JSON-serializable dictionaries following the EnhancedMathSchema format. They can be:
- Saved to files for later analysis
- Fed to other tools or LLMs
- Compared across engine boundaries
- Used to guide ML model design with physics context
