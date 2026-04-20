"""Generate harness skeletons for VASP, Ansys, and COMSOL using Harness Auto-Generator.

This demonstrates using the auto-generator to create harness skeletons,
then manually adding symbolic constraint logic.
"""

import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'math-anything' / 'core'))

from math_anything.codegen import HarnessAutoGenerator
from math_anything.schemas import SymbolicConstraint, ParameterRelationship


def create_vasp_source_mock(temp_dir: Path):
    """Create mock VASP source files for analysis."""
    src_dir = temp_dir / "vasp_src"
    src_dir.mkdir(exist_ok=True)
    
    # Create a mock INCAR-like parameter file
    incar_example = """
# VASP INCAR - Input parameters
# Mathematical constraints:
# - ENCUT > 0 (cutoff energy must be positive)
# - EDIFF > 0 (convergence threshold must be positive)
# - ISMEAR in [-5, -4, -3, -2, -1, 0, 1, 2] (smearing method valid)
# - SIGMA > 0 (smearing width must be positive)
# Constraint: ENCUT > max(ENMAX) for all POTCARs

ENCUT = 520.0        ! Cutoff energy in eV
EDIFF = 1E-6         ! Electronic convergence criterion
ISMEAR = 0           ! Gaussian smearing
SIGMA = 0.05         ! Smearing width in eV
ALGO = Normal        ! Electronic minimization algorithm
NELM = 60            ! Max electronic steps
NELMIN = 4           ! Min electronic steps
"""
    
    # Create mock source with DFT equations in comments
    dft_source = """
/*
 * VASP Density Functional Theory Implementation
 * 
 * Governing equation: Kohn-Sham equations
 * [-½∇² + V_eff(r)] φ_i(r) = ε_i φ_i(r)
 * 
 * where V_eff(r) = V_ext(r) + V_H(r) + V_xc(r)
 * 
 * Constraints:
 * - n(r) = Σ |φ_i(r)|² (electron density)
 * - ∫ n(r) dr = N_electrons (charge conservation)
 * - ε_i must be real (Hermitian Hamiltonian)
 * 
 * SCF iteration: ρ_in → H[ρ_in] → solve KS → ρ_out → mix → ρ_in'
 * Convergence: |ρ_out - ρ_in| < tolerance
 */

#include "dft.h"
#include "electronic.h"

class DFTSolver {
public:
    void solve_kohn_sham() {
        // Self-consistent field loop
        // Constraint: charge neutrality required
        // Constraint: occupation f(ε_i) ∈ [0, 2] for spin-unpolarized
        
        for (int scf_step = 0; scf_step < max_scf; ++scf_step) {
            // Build Hamiltonian from density
            // H = T + V_eff[ρ]
            
            // Diagonalize
            // H φ_i = ε_i φ_i
            
            // Check convergence
            if (density_residual < tolerance) break;
        }
    }
    
private:
    double tolerance;  // EDIFF - must be > 0
    double cutoff;     // ENCUT - must be > 0
    int max_scf;       // NELM
};
"""
    
    (src_dir / "INCAR.example").write_text(incar_example, encoding='utf-8')
    (src_dir / "dft_solver.cpp").write_text(dft_source, encoding='utf-8')
    
    return str(src_dir)


def create_ansys_source_mock(temp_dir: Path):
    """Create mock Ansys source files for analysis."""
    src_dir = temp_dir / "ansys_src"
    src_dir.mkdir(exist_ok=True)
    
    # Create APDL command documentation
    apdl_doc = """
! Ansys APDL Commands
! Mathematical model: Finite Element Analysis
! 
! Governing equation: [K]{u} = {F} (linear static)
!                    [M]{ü} + [C]{u̇} + [K]{u} = {F} (dynamic)
!
! Constraints:
! - Young's modulus E > 0
! - Poisson's ratio -1 < ν < 0.5
! - Density ρ > 0
! - Time step Δt < Δt_critical for explicit dynamics

/PREP7
! Material properties
! Constraint: MP,EX must be positive
MP,EX,1,210E9        ! Young's modulus [Pa]
MP,PRXY,1,0.3        ! Poisson's ratio
MP,DENS,1,7800       ! Density [kg/m³]

! Element type
ET,1,SOLID185        ! 3D solid element

! Solution
/SOLU
ANTYPE,STATIC        ! Static analysis
! For transient: ANTYPE,TRANS
! Constraint: time step must satisfy stability for explicit
"""
    
    ansys_source = """
/* Ansys Mechanical Solver
 * 
 * Element formulation:
 * - Shape functions: N_i(ξ, η, ζ)
 * - Jacobian: J = ∂(x,y,z)/∂(ξ,η,ζ)
 * - Constraint: det(J) > 0 (valid element mapping)
 * 
 * Stiffness matrix: K = ∫ Bᵀ D B dV
 * Mass matrix: M = ∫ ρ Nᵀ N dV
 * 
 * Eigenvalue problem: (K - ω²M)φ = 0
 * Constraint: ω² > 0 for stable system
 */

class Element {
public:
    void compute_stiffness() {
        // Gauss quadrature
        // Constraint: det(J) at Gauss points must be positive
        
        for (int gp = 0; gp < num_gauss; ++gp) {
            compute_jacobian();
            if (detJ <= 0) {
                error("Negative Jacobian - element distorted");
            }
            // K += Bᵀ * D * B * detJ * weight
        }
    }
    
private:
    double detJ;  // det(J) > 0 required
};
"""
    
    (src_dir / "apdl_commands.txt").write_text(apdl_doc, encoding='utf-8')
    (src_dir / "element.cpp").write_text(ansys_source, encoding='utf-8')
    
    return str(src_dir)


def create_comsol_source_mock(temp_dir: Path):
    """Create mock COMSOL source files for analysis."""
    src_dir = temp_dir / "comsol_src"
    src_dir.mkdir(exist_ok=True)
    
    # COMSOL model file structure
    comsol_model = """
% COMSOL Multiphysics Model
% Mathematical models supported:
% - Heat transfer: ρC_p ∂T/∂t = ∇·(k∇T) + Q
% - Structural: ρ ∂²u/∂t² = ∇·σ + F
% - Electromagnetic: ∇×(μ⁻¹∇×A) = J
% - Fluid: ρ(∂u/∂t + u·∇u) = -∇p + ∇·τ + F
%
% Constraints:
% - Material properties must be positive (ρ, k, C_p, E, μ...)
% - Mesh quality > threshold (Jacobian positive)
% - Time step < stability limit for transient
% - Reynolds number Re = ρUL/μ for turbulence modeling

Model = create('Model');
Model.param.set('rho', '1000[kg/m^3]');  % Density > 0
Model.param.set('k', '0.5[W/(m*K)]');    % Conductivity > 0
Model.param.set('Cp', '4200[J/(kg*K)]'); % Heat capacity > 0

% Physics interfaces
ht = Model.physics.create('ht', 'HeatTransfer');
solid = Model.physics.create('solid', 'SolidMechanics');

% Study
study = Model.study.create('std1');
study.create('time', 'Transient');  % Time-dependent
"""
    
    comsol_physics = """
/* COMSOL Physics Implementation
 * 
 * Weak form for heat transfer:
 * ∫ (ρ C_p ∂T/∂t v + k ∇T·∇v) dx = ∫ Q v dx + ∫ q v ds
 * 
 * where v is test function
 * 
 * Stability constraints for time stepping:
 * - Explicit: Δt < h²/(2α) where α = k/(ρ C_p)
 * - Implicit: unconditionally stable but accuracy limited
 * 
 * Constraint: Peclet number Pe = ρ C_p u h / (2k) < 2 for accuracy
 */

template<typename T>
class HeatTransfer {
public:
    void assemble() {
        // Check material properties
        if (density <= 0 || conductivity <= 0 || heat_capacity <= 0) {
            throw PhysicsError("Material properties must be positive");
        }
        
        // Assemble weak form
        // Constraint: mesh must be valid (positive Jacobians)
    }
    
    void solve_transient() {
        // Time stepping
        // Constraint: Δt must satisfy stability or use implicit
        
        double dt = timestep;
        double alpha = conductivity / (density * heat_capacity);
        double h = mesh_size;
        
        // Courant-Friedrichs-Lewy-like condition
        if (explicit_method && dt > h*h / (2*alpha)) {
            warning("Time step may violate stability");
        }
    }
    
private:
    double density;       // ρ > 0
    double conductivity;  // k > 0
    double heat_capacity; // C_p > 0
};
"""
    
    (src_dir / "model.m").write_text(comsol_model, encoding='utf-8')
    (src_dir / "heat_transfer.hpp").write_text(comsol_physics, encoding='utf-8')
    
    return str(src_dir)


def generate_and_enhance_harness(generator: HarnessAutoGenerator, 
                                  engine_name: str,
                                  source_dir: str,
                                  constraint_adder):
    """Generate harness and add symbolic constraints."""
    print(f"\n{'='*70}")
    print(f"Generating harness for {engine_name}")
    print('='*70)
    
    # Generate from source
    template = generator.generate_from_source(
        source_dir=source_dir,
        engine_name=engine_name.lower(),
        engine_version="1.0.0",
        file_patterns=["*.cpp", "*.hpp", "*.h", "*.txt", "*.m", "*.example"],
    )
    
    print(f"\n📊 Extraction Results:")
    print(f"   Commands: {len(template.extracted_commands)}")
    print(f"   Equations: {len(template.mathematical_mappings.get('equations', []))}")
    print(f"   Constraints: {len(template.constraint_expressions)}")
    
    # Save the skeleton
    output_dir = Path(__file__).parent.parent / 'math-anything' / f'{engine_name.lower()}-harness-auto'
    generator.save_harness(template, str(output_dir))
    print(f"\n💾 Skeleton saved to: {output_dir}")
    
    # Now enhance with symbolic constraints
    print(f"\n🔧 Enhancing with symbolic constraints...")
    enhanced_code = constraint_adder(template.generated_code)
    
    # Save enhanced version
    enhanced_file = output_dir / f"math_anything" / engine_name.lower() / "core" / "harness_enhanced.py"
    enhanced_file.parent.mkdir(parents=True, exist_ok=True)
    enhanced_file.write_text(enhanced_code, encoding='utf-8')
    print(f"💾 Enhanced harness saved to: {enhanced_file}")
    
    return template


def add_vasp_constraints(original_code: str) -> str:
    """Add VASP-specific symbolic constraints to generated harness."""
    
    # Find the location to insert constraints
    insert_marker = "# Add symbolic constraints (REVIEW CRITICAL)"
    
    vasp_constraints = '''
        # VASP-specific symbolic constraints
        # Electronic structure constraints
        vasp_constraints = [
            SymbolicConstraint(
                expression="ENCUT > 0",
                description="Plane-wave cutoff energy must be positive",
                variables=["ENCUT"],
                confidence=1.0,
                inferred_from="VASP INCAR parameter",
            ),
            SymbolicConstraint(
                expression="ENCUT > max(ENMAX)",
                description="Cutoff must exceed maximum ENMAX from POTCARs",
                variables=["ENCUT", "ENMAX"],
                confidence=1.0,
                inferred_from="VASP accuracy requirement",
            ),
            SymbolicConstraint(
                expression="EDIFF > 0",
                description="Electronic convergence criterion must be positive",
                variables=["EDIFF"],
                confidence=1.0,
                inferred_from="VASP INCAR",
            ),
            SymbolicConstraint(
                expression="ISMEAR in [-5, -4, -3, -2, -1, 0, 1, 2]",
                description="Smearing method must be valid integer",
                variables=["ISMEAR"],
                confidence=1.0,
                inferred_from="VASP valid smearing options",
            ),
            SymbolicConstraint(
                expression="SIGMA > 0",
                description="Smearing width must be positive",
                variables=["SIGMA"],
                confidence=1.0,
                inferred_from="VASP INCAR",
            ),
            SymbolicConstraint(
                expression="occupation(f_i) in [0, 2]",
                description="Band occupation must be between 0 and 2 (spin-unpolarized)",
                variables=["f_i"],
                confidence=1.0,
                inferred_from="Pauli exclusion principle",
            ),
            SymbolicConstraint(
                expression="integral(n(r)) = N_electrons",
                description="Charge conservation - electron density integrates to total electrons",
                variables=["n", "N_electrons"],
                confidence=1.0,
                inferred_from="DFT charge neutrality",
            ),
        ]
        
        for constraint in vasp_constraints:
            schema.add_symbolic_constraint(constraint)
        
        # VASP parameter relationships
        vasp_relationships = [
            ParameterRelationship(
                name="kohn_sham_equation",
                expression="[-½∇² + V_eff(r)] φ_i(r) = ε_i φ_i(r)",
                variables=["V_eff", "φ_i", "ε_i", "r"],
                relation_type="eigenvalue_equation",
                description="Kohn-Sham equations for non-interacting electrons",
                physical_meaning="Effective single-particle Schrödinger equation",
            ),
            ParameterRelationship(
                name="electron_density",
                expression="n(r) = Σ_i f_i |φ_i(r)|²",
                variables=["n", "f_i", "φ_i", "r"],
                relation_type="equality",
                description="Electron density from Kohn-Sham orbitals",
                physical_meaning="Density functional theory definition",
            ),
            ParameterRelationship(
                name="total_energy",
                expression="E = Σ_i f_i ε_i - ½∫∫ n(r)n(r')/|r-r'| dr dr' + E_xc[n] - ∫ V_xc(r)n(r) dr",
                variables=["E", "f_i", "ε_i", "n", "E_xc", "V_xc"],
                relation_type="equality",
                description="DFT total energy including double-counting correction",
                physical_meaning="Variational energy functional",
            ),
        ]
        
        for relationship in vasp_relationships:
            schema.add_parameter_relationship(relationship)
'''
    
    if insert_marker in original_code:
        code = original_code.replace(insert_marker, insert_marker + vasp_constraints)
    else:
        code = original_code + "\n" + vasp_constraints
    
    return code


def add_ansys_constraints(original_code: str) -> str:
    """Add Ansys-specific symbolic constraints to generated harness."""
    
    insert_marker = "# Add symbolic constraints (REVIEW CRITICAL)"
    
    ansys_constraints = '''
        # Ansys-specific symbolic constraints
        ansys_constraints = [
            SymbolicConstraint(
                expression="E > 0",
                description="Young's modulus must be positive",
                variables=["E"],
                confidence=1.0,
                inferred_from="Material stability",
            ),
            SymbolicConstraint(
                expression="-1 < nu < 0.5",
                description="Poisson's ratio must be in valid range for positive definite stiffness",
                variables=["nu"],
                confidence=1.0,
                inferred_from="Elasticity tensor positive definiteness",
            ),
            SymbolicConstraint(
                expression="density > 0",
                description="Material density must be positive",
                variables=["density"],
                confidence=1.0,
                inferred_from="Physical material property",
            ),
            SymbolicConstraint(
                expression="det(J) > 0",
                description="Element Jacobian determinant must be positive (no distortion)",
                variables=["J"],
                confidence=1.0,
                inferred_from="Valid finite element mapping",
            ),
            SymbolicConstraint(
                expression="dt < dt_critical",
                description="Time step must be less than critical for explicit dynamics",
                variables=["dt", "dt_critical"],
                confidence=0.9,
                inferred_from="CFL stability condition for explicit time integration",
            ),
            SymbolicConstraint(
                expression="aspect_ratio < threshold",
                description="Element aspect ratio should be bounded for accuracy",
                variables=["aspect_ratio"],
                confidence=0.8,
                inferred_from="Mesh quality requirement",
            ),
        ]
        
        for constraint in ansys_constraints:
            schema.add_symbolic_constraint(constraint)
        
        # Ansys parameter relationships
        ansys_relationships = [
            ParameterRelationship(
                name="stiffness_matrix",
                expression="K = ∫_Ω Bᵀ D B dV",
                variables=["K", "B", "D"],
                relation_type="integral_equation",
                description="Element stiffness matrix from constitutive and shape function matrices",
                physical_meaning="Virtual work principle discretization",
            ),
            ParameterRelationship(
                name="mass_matrix",
                expression="M = ∫_Ω ρ Nᵀ N dV",
                variables=["M", "ρ", "N"],
                relation_type="integral_equation",
                description="Consistent mass matrix",
                physical_meaning="D'Alembert principle",
            ),
            ParameterRelationship(
                name="equilibrium_equation",
                expression="[K]{u} = {F}",
                variables=["K", "u", "F"],
                relation_type="linear_system",
                description="Linear static equilibrium (algebraic system)",
                physical_meaning="Discretized equilibrium of forces",
            ),
            ParameterRelationship(
                name="dynamic_equation",
                expression="[M]{ü} + [C]{u̇} + [K]{u} = {F}",
                variables=["M", "C", "K", "u", "F"],
                relation_type="differential_equation",
                description="Dynamic equilibrium with damping",
                physical_meaning="Newton's second law for structural dynamics",
            ),
            ParameterRelationship(
                name="eigenvalue_problem",
                expression="([K] - ω²[M]){φ} = {0}",
                variables=["K", "M", "ω", "φ"],
                relation_type="generalized_eigenvalue",
                description="Free vibration eigenvalue problem",
                physical_meaning="Natural frequencies and mode shapes",
            ),
        ]
        
        for relationship in ansys_relationships:
            schema.add_parameter_relationship(relationship)
'''
    
    if insert_marker in original_code:
        code = original_code.replace(insert_marker, insert_marker + ansys_constraints)
    else:
        code = original_code + "\n" + ansys_constraints
    
    return code


def add_comsol_constraints(original_code: str) -> str:
    """Add COMSOL-specific symbolic constraints to generated harness."""
    
    insert_marker = "# Add symbolic constraints (REVIEW CRITICAL)"
    
    comsol_constraints = '''
        # COMSOL-specific symbolic constraints
        comsol_constraints = [
            SymbolicConstraint(
                expression="rho > 0",
                description="Density must be positive",
                variables=["rho"],
                confidence=1.0,
                inferred_from="Physical material property",
            ),
            SymbolicConstraint(
                expression="k > 0",
                description="Thermal conductivity must be positive",
                variables=["k"],
                confidence=1.0,
                inferred_from="Second law of thermodynamics",
            ),
            SymbolicConstraint(
                expression="Cp > 0",
                description="Heat capacity must be positive",
                variables=["Cp"],
                confidence=1.0,
                inferred_from="Thermodynamic stability",
            ),
            SymbolicConstraint(
                expression="mu > 0",
                description="Dynamic viscosity must be positive",
                variables=["mu"],
                confidence=1.0,
                inferred_from="Second law of thermodynamics (entropy production)",
            ),
            SymbolicConstraint(
                expression="dt < dx²/(2*alpha)",
                description="Explicit time step stability for heat conduction",
                variables=["dt", "dx", "alpha"],
                confidence=0.9,
                inferred_from="Courant-Friedrichs-Lewy condition",
            ),
            SymbolicConstraint(
                expression="Pe < 2",
                description="Peclet number should be less than 2 for accuracy",
                variables=["Pe"],
                confidence=0.8,
                inferred_from="Finite element accuracy for convection-diffusion",
            ),
            SymbolicConstraint(
                expression="Re > 4000 -> turbulence_model_required",
                description="High Reynolds number requires turbulence modeling",
                variables=["Re"],
                confidence=0.9,
                inferred_from="Flow regime physics",
            ),
        ]
        
        for constraint in comsol_constraints:
            schema.add_symbolic_constraint(constraint)
        
        # COMSOL parameter relationships
        comsol_relationships = [
            ParameterRelationship(
                name="heat_equation",
                expression="ρ C_p ∂T/∂t = ∇·(k∇T) + Q",
                variables=["ρ", "C_p", "T", "t", "k", "Q"],
                relation_type="partial_differential_equation",
                description="Transient heat conduction with source term",
                physical_meaning="Energy conservation (First Law of Thermodynamics)",
            ),
            ParameterRelationship(
                name="thermal_diffusivity",
                expression="α = k / (ρ C_p)",
                variables=["α", "k", "ρ", "C_p"],
                relation_type="equality",
                description="Thermal diffusivity",
                physical_meaning="Rate of heat propagation",
            ),
            ParameterRelationship(
                name="peclet_number",
                expression="Pe = ρ C_p u L / k = u L / α",
                variables=["Pe", "u", "L", "α"],
                relation_type="equality",
                description="Ratio of convective to diffusive heat transfer",
                physical_meaning="Dimensionless measure of convection strength",
            ),
            ParameterRelationship(
                name="navier_stokes",
                expression="ρ(∂u/∂t + u·∇u) = -∇p + ∇·(μ(∇u + (∇u)ᵀ)) + F",
                variables=["ρ", "u", "t", "p", "μ", "F"],
                relation_type="partial_differential_equation",
                description="Incompressible Navier-Stokes equations",
                physical_meaning="Momentum conservation in fluid flow",
            ),
            ParameterRelationship(
                name="reynolds_number",
                expression="Re = ρ u L / μ",
                variables=["Re", "ρ", "u", "L", "μ"],
                relation_type="equality",
                description="Ratio of inertial to viscous forces",
                physical_meaning="Dimensionless flow parameter determining regime",
            ),
            ParameterRelationship(
                name="continuity",
                expression="∇·u = 0",
                variables=["u"],
                relation_type="partial_differential_equation",
                description="Incompressibility constraint",
                physical_meaning="Mass conservation for incompressible flow",
            ),
            ParameterRelationship(
                name="maxwell_equations",
                expression="∇×E = -∂B/∂t, ∇×H = J + ∂D/∂t, ∇·D = ρ, ∇·B = 0",
                variables=["E", "B", "H", "D", "J", "ρ"],
                relation_type="partial_differential_equation_system",
                description="Maxwell's equations for electromagnetics",
                physical_meaning="Unified electromagnetic theory",
            ),
        ]
        
        for relationship in comsol_relationships:
            schema.add_parameter_relationship(relationship)
'''
    
    if insert_marker in original_code:
        code = original_code.replace(insert_marker, insert_marker + comsol_constraints)
    else:
        code = original_code + "\n" + comsol_constraints
    
    return code


def main():
    import tempfile
    
    print("=" * 70)
    print("Harness Auto-Generator: VASP, Ansys, COMSOL")
    print("=" * 70)
    print()
    print("This script:")
    print("1. Creates mock source files for each software")
    print("2. Uses Harness Auto-Generator to extract commands")
    print("3. Enhances with domain-specific symbolic constraints")
    print()
    
    generator = HarnessAutoGenerator()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        
        # Generate VASP harness
        vasp_src = create_vasp_source_mock(temp_path)
        generate_and_enhance_harness(
            generator, "vasp", vasp_src, add_vasp_constraints
        )
        
        # Generate Ansys harness
        ansys_src = create_ansys_source_mock(temp_path)
        generate_and_enhance_harness(
            generator, "ansys", ansys_src, add_ansys_constraints
        )
        
        # Generate COMSOL harness
        comsol_src = create_comsol_source_mock(temp_path)
        generate_and_enhance_harness(
            generator, "comsol", comsol_src, add_comsol_constraints
        )
    
    print()
    print("=" * 70)
    print("✅ All harnesses generated and enhanced!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review generated harnesses in math-anything/*-harness-auto/")
    print("2. Implement actual parsing logic in the TODO sections")
    print("3. Test with real input files")
    print("4. Register harnesses after validation")


if __name__ == "__main__":
    main()
