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

---
name: cli-anything
description: Command-line interface for Math Anything - provides interactive REPL, extraction, diff comparison, and validation tools for computational materials science engines. Use when needing command-line access to math-anything functionality.
allowed-tools:
  - Read
  - Write
  - Bash
when_to_use: |
  Trigger when user mentions:
  - "cli anything" or "cli-anything"
  - "command line" interface for math-anything
  - "terminal" or "shell" access to math-anything
  - "run math-anything from command line"
  - "math-anything repl" or "interactive mode"
  - "extract from command line"
  - "compare schemas via CLI"
argument-hint: "[command] [options]"
arguments:
  - command
  - options
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

## CLI Anything Commands

### Interactive REPL Mode

Start an interactive session for exploring mathematical structures:

```bash
cli-anything repl                    # Start REPL
cli-anything repl --session file     # Load existing session
```

### Extract Mathematical Structure

Extract schema from simulation input files:

```bash
cli-anything extract vasp INCAR POSCAR KPOINTS
cli-anything extract lammps input.in
cli-anything extract ansys model.inp --output schema.json
```

### Compare Calculations

Compare two mathematical schemas:

```bash
cli-anything diff schema1.json schema2.json
cli-anything diff schema1.json schema2.json --format json
```

### Cross-Engine Mapping

Map parameters between different simulation engines:

```bash
cli-anything cross vasp_schema.json quantum_espresso
cli-anything cross lammps_schema.json gromacs
```

### Validate Constraints

Validate symbolic constraints in a schema:

```bash
cli-anything validate schema.json
```

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

### 4. Discover Equations from Data

Discover mathematical equations from simulation output data using PSRN (Parallel Symbolic Regression Network).

**Input**: Data arrays (X, y) from simulation outputs
**Output**: Discovered equation in standard mathematical notation
**Algorithm**: PSRN with compiled evaluation - 50x faster than traditional GP

**Example**:
```python
from math_anything import MathAnything
import numpy as np

ma = MathAnything()

# Discover equation from VASP energy-volume curve
volumes = np.array([50, 55, 60, 65, 70])  # Angstrom^3
energies = np.array([-100, -120, -130, -135, -138])  # eV

equation = ma.discover(volumes.reshape(-1, 1), energies, ['V'])
# Returns: discovered E(V) relationship

# Multi-variable: discover relationship between multiple parameters
# Discover: E = f(T, P, composition)
equation = ma.discover(X, y, ['T', 'P', 'x'])

# Advanced: tune PSRN parameters
equation = ma.discover(X, y, names, n_layers=3, max_iterations=5)
```

### 5. List Supported Engines

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

**CLI usage**:
```bash
# Extract and save
cli-anything extract vasp INCAR POSCAR --output vasp_schema.json

# Compare via CLI
cli-anything diff vasp_schema.json lammps_schema.json

# Interactive exploration
cli-anything repl
```

**Discover equations from data**:
```
> discover E(V) from VASP energy-volume data
< Found: E = -140 + 0.5*(V-65)^2  (MSE: 2.3e-4)
< Physical interpretation: Birch-Murnaghan EOS near equilibrium
< Time: 6.2 seconds (PSRN with compiled evaluation)
```

**Complete workflow**:
```python
# 1. Extract mathematical structure from simulation input
result = ma.extract_file("vasp", "INCAR")

# 2. Translate to LLM-solvable propositions
props = ma.translate(result)

# 3. Run simulation, collect output data
# ... (user's simulation workflow)

# 4. Discover equations from output
equation = ma.discover(X, y, ['V', 'T'])

# 5. Compare with theoretical predictions
# ... (analysis)
```

## Guardrails

- Never modifies user input files
- Never judges parameter values as "right" or "wrong"
- Always expresses structures in canonical mathematical forms
- Always reports approximation hierarchies honestly
- Never hides the mathematical consequences of approximations

## Outputs

All outputs are JSON-serializable dictionaries following the EnhancedMathSchema format. They can be:
- Saved to files for later analysis
- Fed to other tools or LLMs
- Compared across engine boundaries
- Used to guide ML model design with physics context

## Performance

**PSRN Symbolic Regression**:
- 50x faster than traditional GP (compiled evaluation)
- Handles multi-variable problems efficiently
- Typical runtime: 5-10 seconds for 100-1000 data points
- Memory: scales with expression tree depth (default 2 layers)

**Comparison**:
| Method | Time (100 samples) | Time (1000 samples) |
|--------|-------------------|---------------------|
| Legacy GP | 300-500s | 3000-5000s |
| PSRN (optimized) | 6-8s | 8-12s |

**Backward Compatibility**:
- Set `use_psrn=False` to use legacy GP mode
- All existing code continues to work unchanged
