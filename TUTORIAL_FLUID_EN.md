# Math Anything Tutorial: Computational Fluid Dynamics

> Follow CFD engineer Michael Park's 14-day journey simulating turbine blade cooling

---

## Character Background

**Michael Park**, Senior CFD Engineer at a gas turbine manufacturer, specializing in conjugate heat transfer and turbulence modeling.

**Background**:
- PhD in Mechanical Engineering, focus on turbulent flows
- Expert in ANSYS Fluent and OpenFOAM
- Works on film cooling optimization for turbine blades
- Needs to document simulation methodologies for internal review

**Goal**: Use Math Anything to extract mathematical models from CFD simulations and automate best practice validation.

---

## Chapter 1: Turbulent Flow Analysis (Days 1-3)

### 1.1 First CFD Model Extraction

Michael analyzes flow over a turbine blade:

```python
from math_anything import MathAnything

ma = MathAnything()

# Extract from Fluent case
result = ma.extract_file("fluent", "turbine_blade.cas")

print("Governing Equations:")
print(result.schema["mathematical_structure"]["canonical_form"])
# Output: 
#   ∂ρ/∂t + ∇·(ρu) = 0  (Continuity)
#   ∂(ρu)/∂t + ∇·(ρuu) = -∇p + ∇·τ + ρg  (Momentum)
#   ∂(ρE)/∂t + ∇·(u(ρE+p)) = ∇·(k∇T) + ∇·(τ·u)  (Energy)

print("\nTurbulence Model:")
print(result.schema["turbulence"]["model"])
# Output: k-ω SST with curvature correction
```

### 1.2 Turbulence Model Analysis

```python
from math_anything import extract

# Analyze RANS turbulence closure
result = extract("fluent", {
    "solver": "pressure_based",
    "turbulence_model": "k_omega_sst",
    "near_wall_treatment": "automatic",
    "y_plus_target": 1.0,
})

print("Turbulence Modeling Framework:")
print(f"  Model type: {result.schema['turbulence']['closure_type']}")
print(f"  Governing equations: {result.schema['turbulence']['transport_equations']}")

print("\nWall Treatment:")
y_plus = result.schema["near_wall"]["y_plus_max"]
if y_plus < 5:
    print(f"  ✓ y+ = {y_plus:.2f}: Resolving viscous sublayer")
elif y_plus < 30:
    print(f"  ! y+ = {y_plus:.2f}: Buffer layer (avoid)")
else:
    print(f"  ✗ y+ = {y_plus:.2f}: Using wall functions")
```

**Output**:
```
Turbulence Modeling Framework:
  Model type: Eddy viscosity model
  Governing equations: 
    - Transport of k (turbulent kinetic energy)
    - Transport of ω (specific dissipation rate)

Wall Treatment:
  ✓ y+ = 0.85: Resolving viscous sublayer
  ✓ First cell height: 2.3e-6 m
```

### 1.3 Mesh Quality Assessment

```python
# Extract mesh information
result = ma.extract_file("fluent", "mesh_quality.msh")

print("Mesh Quality Metrics:")
metrics = result.schema["mesh"]["quality_metrics"]
for metric, value in metrics.items():
    status = "✓" if value["passed"] else "✗"
    print(f"  {status} {metric}: {value['value']:.3f} (min: {value['threshold']})")
```

---

## Chapter 2: Conjugate Heat Transfer (Days 4-6)

### 2.1 Fluid-Solid Coupling

Michael couples fluid and solid domains for blade cooling:

```python
from math_anything import extract

result = extract("fluent", {
    "solver_type": "pressure_based",
    "energy_equation": True,
    "conjugate_ht": True,
    "domains": ["fluid", "solid"],
})

print("Conjugate Heat Transfer:")
print(f"  Fluid domain: {result.schema['domains']['fluid']}")
print(f"  Solid domain: {result.schema['domains']['solid']}")

print("\nInterface Coupling:")
print("  Heat flux continuity: q_f = q_s")
print("  Temperature continuity: T_f = T_s (at interface)")

print("\nDimensionless Numbers:")
print(f"  Reynolds: {result.schema['nondim']['reynolds']:.2e}")
print(f"  Prandtl: {result.schema['nondim']['prandtl']:.2f}")
print(f"  Nusselt: {result.schema['nondim']['nusselt']:.2f}")
```

### 2.2 Film Cooling Effectiveness

```python
# Analyze film cooling configuration
result = ma.extract_file("fluent", "film_cooling.cas")

print("Film Cooling Parameters:")
print(f"  Blowing ratio: {result.schema['cooling']['blowing_ratio']:.2f}")
print(f"  Density ratio: {result.schema['cooling']['density_ratio']:.2f}")
print(f"  Momentum flux ratio: {result.schema['cooling']['momentum_ratio']:.2f}")

print("\nCooling Effectiveness:")
print("  η = (T_∞ - T_aw) / (T_∞ - T_c)")
print(f"  Peak η: {result.schema['cooling']['peak_effectiveness']:.3f}")
```

### 2.3 Transient Thermal Analysis

```python
# Extract time-dependent simulation
result = extract("fluent", {
    "solver": "density_based",
    "time_type": "unsteady",
    "time_step": 1e-5,
    "max_iter": 100000,
})

print("Unsteady Formulation:")
print(f"  Temporal discretization: {result.schema['time']['scheme']}")
print(f"  Courant number: {result.schema['time']['courant_max']:.2f}")
print(f"  Time step (CFL<1): Δt = {result.schema['time']['dt']:.2e} s")
```

---

## Chapter 3: Tiered Analysis for Certification (Days 7-10)

### 3.1 High-Fidelity LES Analysis

Michael runs Large Eddy Simulation for detailed flow physics:

```python
from math_anything import TieredAnalyzer, AnalysisTier

analyzer = TieredAnalyzer()

rec = analyzer.get_recommendation("les_turbine.cas")
print(f"Recommended tier: {rec.recommended_tier}")
print(f"Complexity: {rec.complexity_score.total}/100")
```

**Output**:
```
Recommended tier: COMPLETE
Complexity: 88/100
Reasons: [
    "LES with dynamic Smagorinsky model",
    "Billion-cell unstructured mesh",
    "Conjugate heat transfer with radiation",
    "Multi-species combustion products"
]
```

### 3.2 Extracting Subgrid Scale Models

```python
result = analyzer.analyze("les_turbine.cas", tier=AnalysisTier.COMPLETE)

print("LES Mathematical Framework:")
print("  Filtered Navier-Stokes:")
print("    ∂ūᵢ/∂t + ūⱖ∂ūᵢ/∂xⱖ = -1/ρ ∂p̄/∂xᵢ + ν ∂²ūᵢ/∂xⱖ² - ∂τᵢⱖˢᴳˢ/∂xⱖ")

print("\nSubgrid Scale Model:")
print(f"  Model: {result.manifold_info.sgs_model}")
print(f"  Cs (Smagorinsky coeff): {result.manifold_info.cs_value:.3f}")
print(f"  Filter width: {result.manifold_info.filter_width:.2e} m")

print("\nTurbulent Kinetic Energy Spectrum:")
print(f"  Resolved: {result.topology_info.energy_resolved:.1f}%")
print(f"  Modeled: {result.topology_info.energy_modeled:.1f}%")
```

### 3.3 Automated Best Practice Check

```python
from math_anything import PropositionGenerator, TaskType

generator = PropositionGenerator()

theorem = generator.generate(
    engine="fluent",
    parameters={
        "simulation_type": "les",
        "mesh_refinement": "adaptive",
        "turbulence_model": "dynamic_smagorinsky",
    },
    task_type=TaskType.CONVERGENCE
)

print(theorem)
```

**Output**:
```
Theorem (LES Convergence):
  Consider the filtered Navier-Stokes equations on domain Ω
  with subgrid scale model τᵢⱖˢᴳˢ = -2(CsΔ)²|S̄|S̄ᵢⱖ.
  
  Given:
    - Grid resolution: Δ/η > 10 (η is Kolmogorov scale)
    - Dynamic Cs calculated via Germano identity
    - Time step satisfying CFL < 0.5
    
  The solution converges to filtered DNS as Δ → 0:
    lim_{Δ→0} ūᵢ → G * uᵢ
    
  where G is the filter kernel and * denotes convolution.
  
  Energy spectrum consistency:
    E(k) = Ck^{-5/3} for k in inertial range
```

---

## Chapter 4: Multi-Physics Coupling (Days 11-12)

### 4.1 Fluid-Structure Interaction

Michael couples CFD with structural analysis:

```python
from math_anything import CrossEngineSession

session = CrossEngineSession()

# CFD model
session.add_model("fluid_cfd", {
    "engine": "fluent",
    "physics": "compressible_flow",
    "turbulence": "k_omega_sst",
    "time": "unsteady",
})

# Structural model
session.add_model("structure_fea", {
    "engine": "ansys",
    "analysis": "modal",
    "elements": "solid186",
})

# Couple them
session.add_interface(
    "fluid_cfd", "structure_fea",
    coupling_type="two_way",
    scheme="implicit",
    relaxation=0.7,
)
```

### 4.2 Combustion Modeling

```python
# Extract combustion simulation
result = ma.extract_file("fluent", "combustor.cas")

print("Combustion Model:")
print(f"  Chemistry: {result.schema['combustion']['chemistry_model']}")
print(f"  Turbulence-Chemistry: {result.schema['combustion']['tci_model']}")

print("\nSpecies Transport:")
print("  ∂(ρYₖ)/∂t + ∇·(ρuYₖ) = ∇·(ρDₖ∇Yₖ) + ω̇ₖ")
print(f"  Number of species: {result.schema['species']['count']}")

print("\nReaction Mechanism:")
print(f"  Steps: {result.schema['reactions']['total_steps']}")
print(f"  Reversible: {result.schema['reactions']['reversible_count']}")
```

---

## Chapter 5: Complete Design Analysis (Days 13-14)

### 5.1 Automated Report Generation

```python
from math_anything import generate_report

analyses = {
    "steady_rans": rans_result.schema,
    "unsteady_les": les_result.schema,
    "conjugate_ht": cht_result.schema,
    "cooling_design": cooling_result.schema,
}

report = generate_report(analyses, format="pdf")

with open("turbine_blade_analysis.pdf", "wb") as f:
    f.write(report)
```

---

## CFD-Specific Tips

### Tip 1: Y+ Calculator

```python
def calculate_y_plus(reynolds, distance, fluid):
    """Estimate first cell height for desired y+"""
    nu = fluid["kinematic_viscosity"]
    u_tau = (0.02 * reynolds * nu / distance) ** 0.5
    
    y_plus_target = 1.0
    delta_y = y_plus_target * nu / u_tau
    
    return delta_y

# Example
height = calculate_y_plus(
    reynolds=1e6,
    distance=0.1,  # m
    fluid={"kinematic_viscosity": 1.5e-5}
)
print(f"First cell height for y+=1: {height:.2e} m")
```

### Tip 2: CFL Condition Check

```python
def check_cfl(result):
    """Verify CFL condition satisfaction"""
    cfl = result.schema["time"]["courant_max"]
    
    if result.schema["solver"]["density_based"]:
        limit = 1.0  # Explicit
    else:
        limit = 100.0  # Implicit
    
    if cfl < limit:
        print(f"✓ CFL = {cfl:.2f} < {limit} (stable)")
    else:
        print(f"✗ CFL = {cfl:.2f} > {limit} (reduce time step)")
```

---

## Summary: Michael's Achievements

After 14 days, Michael has:

1. ✅ Automated extraction of RANS and LES turbulence models
2. ✅ Validated conjugate heat transfer simulations
3. ✅ Generated mathematical documentation for certification
4. ✅ Established fluid-structure interaction workflows
5. ✅ Created automated best practice validation

**Impact**: Reduced CFD validation time by 60%, improved consistency across team.
