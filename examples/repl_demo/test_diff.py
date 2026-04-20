"""
Test script for Math Anything Diff functionality.

Tests semantic comparison between:
1. Two VASP calculations (same physics, different parameters)
2. VASP vs LAMMPS (different physics)
"""

import sys
import os
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'math-anything', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'math-anything', 'vasp-harness'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'math-anything', 'lammps-harness'))

from math_anything.repl import MathDiff
from math_anything.schemas import (
    MathSchema, MetaInfo, MathematicalModel, GoverningEquation,
    NumericalMethod, Discretization, Solver, SymbolicConstraint
)


def create_vasp_schema_high_accuracy() -> MathSchema:
    """Create VASP schema with high accuracy settings."""
    return MathSchema(
        schema_version="1.0.0",
        meta=MetaInfo(extracted_by="test", extractor_version="1.0.0", source_files={}),
        mathematical_model=MathematicalModel(
            governing_equations=[
                GoverningEquation(
                    id="kohn_sham",
                    type="eigenvalue_problem",
                    name="Kohn-Sham Equations",
                    mathematical_form="[-½∇² + V_eff]ψ = εψ",
                    variables=["wavefunction", "eigenvalue"],
                ),
            ],
            boundary_conditions=[],
            constitutive_relations=[],
        ),
        numerical_method=NumericalMethod(
            discretization=Discretization(
                space_discretization="plane_wave_basis",
                time_integrator="scf_iteration",
            ),
            solver=Solver(
                algorithm="scf_diagonalization",
                tolerance=1e-6,
            ),
        ),
        symbolic_constraints=[
            SymbolicConstraint(expression="ENCUT > 0", description="ENCUT=800 [SATISFIED]"),
            SymbolicConstraint(expression="EDIFF > 0", description="EDIFF=1e-06 [SATISFIED]"),
        ],
        raw_symbols={
            "incar": {"ENCUT": 800, "EDIFF": 1e-6, "ISMEAR": 0}
        },
    )


def create_vasp_schema_low_accuracy() -> MathSchema:
    """Create VASP schema with low accuracy settings."""
    return MathSchema(
        schema_version="1.0.0",
        meta=MetaInfo(extracted_by="test", extractor_version="1.0.0", source_files={}),
        mathematical_model=MathematicalModel(
            governing_equations=[
                GoverningEquation(
                    id="kohn_sham",
                    type="eigenvalue_problem",
                    name="Kohn-Sham Equations",
                    mathematical_form="[-½∇² + V_eff]ψ = εψ",
                    variables=["wavefunction", "eigenvalue"],
                ),
            ],
            boundary_conditions=[],
            constitutive_relations=[],
        ),
        numerical_method=NumericalMethod(
            discretization=Discretization(
                space_discretization="plane_wave_basis",
                time_integrator="scf_iteration",
            ),
            solver=Solver(
                algorithm="scf_diagonalization",
                tolerance=1e-4,
            ),
        ),
        symbolic_constraints=[
            SymbolicConstraint(expression="ENCUT > 0", description="ENCUT=400 [SATISFIED]"),
            SymbolicConstraint(expression="EDIFF > 0", description="EDIFF=1e-04 [SATISFIED]"),
        ],
        raw_symbols={
            "incar": {"ENCUT": 400, "EDIFF": 1e-4, "ISMEAR": 0}
        },
    )


def create_lammps_schema() -> MathSchema:
    """Create LAMMPS MD schema."""
    return MathSchema(
        schema_version="1.0.0",
        meta=MetaInfo(extracted_by="test", extractor_version="1.0.0", source_files={}),
        mathematical_model=MathematicalModel(
            governing_equations=[
                GoverningEquation(
                    id="newton",
                    type="newtonian_dynamics",
                    name="Newton's Equations of Motion",
                    mathematical_form="F = ma, dr/dt = v, dv/dt = F/m",
                    variables=["position", "velocity", "force"],
                ),
            ],
            boundary_conditions=[],
            constitutive_relations=[
                {"type": "lennard_jones", "form": "U(r) = 4ε[(σ/r)¹² - (σ/r)⁶]"}
            ],
        ),
        numerical_method=NumericalMethod(
            discretization=Discretization(
                space_discretization="particle_discrete",
                time_integrator="velocity_verlet",
            ),
            solver=Solver(
                algorithm="md_time_integration",
                tolerance=1e-10,
            ),
        ),
        symbolic_constraints=[
            SymbolicConstraint(expression="timestep > 0", description="timestep=0.001 [SATISFIED]"),
            SymbolicConstraint(expression="epsilon > 0", description="epsilon=1.0 [SATISFIED]"),
        ],
        raw_symbols={
            "input": {"timestep": 0.001, "epsilon": 1.0, "sigma": 3.4}
        },
    )


def test_same_physics_different_params():
    """Test comparing same physics, different parameters."""
    print("\n" + "=" * 70)
    print("  Test 1: Same Physics (DFT), Different Parameters")
    print("=" * 70)
    
    vasp_high = create_vasp_schema_high_accuracy()
    vasp_low = create_vasp_schema_low_accuracy()
    
    diff = MathDiff.compare(vasp_high, vasp_low)
    
    print(f"\nSimilarity Score: {diff.similarity_score:.1%}")
    print(f"Expected: High (>80%)")
    
    print(f"\n✓ Common Equations: {len(diff.common_equations)}")
    for eq in diff.common_equations:
        print(f"  - {eq}")
    
    print(f"\n📊 Constraint Differences:")
    for cd in diff.constraint_differences:
        print(f"  {cd['parameter']}: {cd['first']} vs {cd['second']}")
    
    print(f"\n💡 {diff.analysis_summary}")
    
    assert diff.similarity_score > 0.8, "Should have high similarity"
    assert len(diff.common_equations) > 0, "Should have common equations"
    print("\n✓ Test PASSED")


def test_different_physics():
    """Test comparing different physics (DFT vs MD)."""
    print("\n" + "=" * 70)
    print("  Test 2: Different Physics (DFT vs MD)")
    print("=" * 70)
    
    vasp = create_vasp_schema_high_accuracy()
    lammps = create_lammps_schema()
    
    diff = MathDiff.compare(vasp, lammps)
    
    print(f"\nSimilarity Score: {diff.similarity_score:.1%}")
    print(f"Expected: Low (<50%)")
    
    print(f"\n📌 Unique to VASP:")
    for item in diff.unique_to_first:
        print(f"  - {item}")
    
    print(f"\n📌 Unique to LAMMPS:")
    for item in diff.unique_to_second:
        print(f"  - {item}")
    
    print(f"\n💡 {diff.analysis_summary}")
    
    assert diff.similarity_score < 0.5, "Should have low similarity"
    print("\n✓ Test PASSED")


def test_cross_engine_mapping():
    """Test cross-engine parameter mapping."""
    print("\n" + "=" * 70)
    print("  Test 3: Cross-Engine Parameter Mapping")
    print("=" * 70)
    
    vasp = create_vasp_schema_high_accuracy()
    
    suggestions = MathDiff.cross_engine_compare(vasp, "quantum_espresso")
    
    print(f"\nTarget Engine: {suggestions['target_engine']}")
    print(f"\n🔄 Mappable Parameters:")
    for m in suggestions['mappable_parameters']:
        print(f"  {m['source']:15} -> {m['target']:15} ({m['meaning']})")
    
    print(f"\n✓ Test PASSED")


def main():
    """Run all tests."""
    print("=" * 70)
    print("  Math Anything Diff - Test Suite")
    print("=" * 70)
    
    try:
        test_same_physics_different_params()
        test_different_physics()
        test_cross_engine_mapping()
        
        print("\n" + "=" * 70)
        print("  ALL TESTS PASSED!")
        print("=" * 70)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
