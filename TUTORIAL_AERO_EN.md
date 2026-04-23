# Math Anything Tutorial: Aerospace Engineering

> Follow aerospace engineer Sarah Chen's 14-day journey analyzing composite wing structures

---

## Character Background

**Sarah Chen**, Structural Analysis Engineer at a commercial aircraft company, specializing in composite materials and fatigue analysis.

**Background**:
- Master's in Aerospace Engineering with focus on structural mechanics
- Expert in Abaqus for nonlinear FEA
- Works on certification analysis for new aircraft designs
- Needs to document mathematical models for FAA compliance

**Goal**: Use Math Anything to extract mathematical structures from FEA models and automate documentation generation.

---

## Chapter 1: Static Stress Analysis (Days 1-3)

### 1.1 First Abaqus Model Extraction

Sarah analyzes a composite wing spar under bending loads:

```python
from math_anything import MathAnything

ma = MathAnything()

# Extract from Abaqus input file
result = ma.extract_file("abaqus", "wing_spar.inp")

print("Mathematical Structure:")
print(result.schema["mathematical_structure"]["canonical_form"])
# Output: ∇·σ + f = 0 (Linear momentum balance)

print("\nProblem Type:", result.schema["mathematical_structure"]["problem_type"])
# Output: boundary_value_pde
```

**Her discovery**: "So Abaqus is solving the linear momentum balance equation with material nonlinearity!"

### 1.2 Material Model Analysis

```python
from math_anything import extract

# Analyze composite material properties
result = extract("abaqus", {
    "material": "carbon_fiber_epoxy",
    "elastic_type": "lamina",
    "hashin_damage": True,
    "damage_evolution": "energy",
    "e1": 150000,  # MPa - longitudinal modulus
    "e2": 10000,   # MPa - transverse modulus
    "g12": 5000,   # MPa - shear modulus
    "nu12": 0.3,
})

# Check failure criteria
for constraint in result.schema.get("constraints", []):
    print(f"{'✓' if constraint['satisfied'] else '✗'} {constraint['expression']}")
```

**Output**:
```
✓ E1 > E2: Orthotropic material correctly defined
✓ |nu12| < sqrt(E1/E2): Stability condition satisfied
✓ G12 > 0: Positive shear modulus
! Warning: No temperature-dependent properties defined
```

### 1.3 Load Case Comparison

Sarah compares two critical load cases:

```python
from math_anything import MathAnything, MathDiffer

ma = MathAnything()

# Limit load case
limit_load = ma.extract_file("abaqus", "limit_load.inp")

# Ultimate load case (with safety factor)
ultimate_load = ma.extract_file("abaqus", "ultimate_load.inp")

# Compare
differ = MathDiffer()
report = differ.diff(limit_load.schema, ultimate_load.schema)

print(report.to_text())
```

**Key findings**:
```
Mathematical Changes:
  ✓ Problem type: boundary_value_pde (unchanged)
  ✗ Boundary conditions: Pressure 2.5g → 3.75g (1.5× safety factor)
  ✗ Material model: Linear elastic → Hashin damage initiation
  ✗ Solution procedure: Static → Quasi-static with damage evolution
```

---

## Chapter 2: Buckling Analysis (Days 4-6)

### 2.1 Eigenvalue Buckling Extraction

Sarah extracts the buckling analysis for a stiffened panel:

```python
from math_anything import extract

result = extract("abaqus", {
    "step_type": "buckle",
    "num_eigenvalues": 10,
    "eigensolver": "lanczos",
    "normalization": "displacement",
})

print("Mathematical Formulation:")
print("  (K - λᵢKG)φᵢ = 0")  # Generalized eigenvalue problem
print(f"  Solving for {result.schema['numerical_method']['num_modes']} eigenvalues")

print("\nExtracted Eigenvalues:")
for i, mode in enumerate(result.schema["results"]["eigenvalues"][:5]):
    print(f"  Mode {i+1}: λ = {mode['value']:.3f}, "
          f"Load Factor = {mode['load_factor']:.2f}")
```

### 2.2 Post-Buckling Analysis

```python
# Extract nonlinear post-buckling analysis
result = ma.extract_file("abaqus", "post_buckle.inp")

print("Analysis Features:")
print(f"  Arc length method: {result.schema['numerical_method']['arc_length']}")
print(f"  Imperfection scale: {result.schema['initial_conditions']['imperfection']} mm")
print(f"  Maximum increment: {result.schema['numerical_method']['max_increment']}")

print("\nMathematical Structure:")
print("  K(u) · Δu = R - F_int(u)")  # Nonlinear equilibrium
print("  With arc length constraint: ΔuᵀΔu + Δλ² = Δs²")
```

---

## Chapter 3: Fatigue and Damage (Days 7-10)

### 3.1 Tiered Analysis for Certification

Sarah runs a complex damage tolerance analysis:

```python
from math_anything import TieredAnalyzer, AnalysisTier

analyzer = TieredAnalyzer()

# Get recommendation for fatigue analysis
rec = analyzer.get_recommendation("damage_tolerance.inp")
print(f"Recommended tier: {rec.recommended_tier}")
print(f"Complexity: {rec.complexity_score.total}/100")
print(f"Reasons: {rec.reasons}")
```

**Output**:
```
Recommended tier: COMPLETE
Complexity: 85/100
Reasons: [
    "Cohesive zone modeling detected",
    "VCCT (Virtual Crack Closure Technique) enabled",
    "Cycle-dependent damage evolution",
    "Multi-step fatigue loading spectrum"
]
```

### 3.2 Fracture Mechanics Extraction

```python
result = analyzer.analyze("damage_tolerance.inp", tier=AnalysisTier.COMPLETE)

print("Fracture Mechanics Framework:")
print(f"  Stress intensity factor KI: {result.manifold_info.crack_tip_field}")
print(f"  Energy release rate G: {result.topology_info.energy_release_rate}")

print("\nParis Law Parameters:")
print(f"  da/dN = C(ΔK)ᵐ")
print(f"  C = {result.material_parameters['paris_c']}")
print(f"  m = {result.material_parameters['paris_m']}")
```

### 3.3 Generating Certification Documentation

```python
from math_anything import PropositionGenerator, TaskType

generator = PropositionGenerator()

# Generate damage tolerance theorem
theorem = generator.generate(
    engine="abaqus",
    parameters={
        "analysis_type": "xfem",
        "crack_growth_model": "paris",
        "loading_spectrum": "flight_by_flight",
        "inspection_interval": 1000,  # flights
    },
    task_type=TaskType.WELL_POSEDNESS
)

print(theorem)
```

**Output**:
```
Theorem (Damage Tolerance Analysis):
  Consider a composite panel with initial flaw size a₀
  under spectrum loading Δσᵢ(t).
  
  Given:
    - Paris law: da/dN = C(ΔK)ᵐ with validated constants
    - XFEM enrichment for crack tip fields
    - Cycle-by-cycle damage accumulation
    
  The residual strength after N cycles satisfies:
    σ_res(N) = σ_ult · (1 - D(N))
    
  where D(N) = Σᵢ (da/dN)ᵢ · ΔNᵢ is the accumulated damage.
  
  Safety requirement:
    σ_res(N_inspect) ≥ 1.15 × σ_limit
```

---

## Chapter 4: Multi-Physics Coupling (Days 11-12)

### 4.1 Thermo-Mechanical Analysis

Sarah couples thermal and structural analysis:

```python
from math_anything import CrossEngineSession

session = CrossEngineSession()

# Thermal model
session.add_model("thermal_model", {
    "engine": "abaqus",
    "physics": "heat_transfer",
    "governing_equation": "ρc_p ∂T/∂t = ∇·(k∇T) + Q",
    "boundary_conditions": ["convection", "radiation"],
})

# Structural model
session.add_model("structural_model", {
    "engine": "abaqus",
    "physics": "solid_mechanics",
    "governing_equation": "∇·σ + f = 0",
    "temperature_dependent": True,
})

# Couple them
session.add_interface(
    "thermal_model", "structural_model",
    coupling_type="sequential",
    shared_variables=["temperature", "thermal_strain"]
)
```

---

## Chapter 5: Complete Project Report (Days 13-14)

### 5.1 Automated Report Generation

```python
from math_anything import generate_report

# Compile all analyses
report_data = {
    "static_analysis": static_result.schema,
    "buckling_analysis": buckling_result.schema,
    "fatigue_analysis": fatigue_result.schema,
    "certification_status": "PASSED",
}

report = generate_report(report_data, format="pdf")

# FAA-compliant documentation
with open("wing_spar_certification.pdf", "wb") as f:
    f.write(report)
```

---

## Aerospace-Specific Tips

### Tip 1: Safety Factor Extraction

```python
result = extract("abaqus", input_file)

safety_factors = {
    "limit_load": 1.0,
    "ultimate_load": 1.5,
    "damage_tolerance": 1.15,
}

for check, factor in safety_factors.items():
    if result.schema["loads"]["safety_factor"] >= factor:
        print(f"✓ {check}: Safety factor {factor} satisfied")
```

### Tip 2: Unit Consistency Check

```python
# Aerospace typically uses mm, MPa, tonne
expected_units = {
    "length": "mm",
    "stress": "MPa",
    "mass": "tonne",
    "time": "s",
}

for quantity, unit in expected_units.items():
    actual = result.schema["units"][quantity]
    if actual != unit:
        print(f"! Warning: {quantity} in {actual}, expected {unit}")
```

---

## Summary: Sarah's Achievements

After 14 days, Sarah has:

1. ✅ Automated extraction of mathematical models from Abaqus
2. ✅ Generated FAA-compliant documentation
3. ✅ Established damage tolerance analysis workflow
4. ✅ Created multi-physics (thermal-structural) coupling
5. ✅ Validated safety factors for certification

**Impact**: Reduced documentation time by 70%, improved traceability for certification audits.
