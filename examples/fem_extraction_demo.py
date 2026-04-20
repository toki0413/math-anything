"""Simplified FEM extraction demo without external dependencies.

This demonstrates the symbolic constraint and parameter relationship
extraction for finite element analysis.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'math-anything' / 'core'))

from math_anything.schemas import (
    MathSchema, MetaInfo, MathematicalModel, NumericalMethod,
    GoverningEquation, SymbolicConstraint, ParameterRelationship,
    Discretization, Solver
)


@dataclass
class Material:
    """Elastic material properties."""
    name: str
    youngs_modulus: float
    poisson_ratio: float


def create_fem_schema(material: Material) -> MathSchema:
    """Create MathSchema for FEM linear elasticity analysis.
    
    This demonstrates the extraction of:
    - Governing equations (equilibrium, constitutive)
    - Parameter relationships (Lamé parameters)
    - Symbolic constraints (material stability)
    """
    
    schema = MathSchema(
        schema_version="1.0.0",
        meta=MetaInfo(
            extracted_by="math-anything-fem-demo",
            extractor_version="0.2.0",
            source_files={"input": ["beam_bending.inp"]},
        ),
        mathematical_model=MathematicalModel(),
        numerical_method=NumericalMethod(
            discretization=Discretization(
                space_discretization="FEM_CPS4",
            ),
            solver=Solver(
                algorithm="direct_sparse",
                convergence_criterion="residual_norm",
                tolerance=1e-6,
            ),
        ),
    )
    
    # Add governing equations
    schema.mathematical_model.governing_equations = [
        GoverningEquation(
            id="equilibrium",
            type="partial_differential_equation",
            name="Equilibrium Equation",
            mathematical_form="∇·σ + b = 0",
            variables=["stress", "body_force", "displacement"],
            description="Static equilibrium in the absence of inertia",
        ),
        GoverningEquation(
            id="constitutive",
            type="constitutive_relation",
            name="Hooke's Law (Isotropic)",
            mathematical_form="σ = λ tr(ε) I + 2με",
            variables=["stress", "strain", "lambda", "mu"],
            description="Linear elastic constitutive relation",
        ),
        GoverningEquation(
            id="weak_form",
            type="variational_principle",
            name="Principle of Virtual Work",
            mathematical_form="∫_Ω σ:δε dΩ = ∫_Ω b·δu dΩ + ∫_Γ t·δu dΓ",
            variables=["stress", "virtual_strain", "body_force", "traction"],
            description="Weak form used in FEM",
        ),
    ]
    
    # Add parameter relationships
    E = material.youngs_modulus
    nu = material.poisson_ratio
    
    schema.mathematical_model.parameter_relationships = [
        ParameterRelationship(
            name="lame_first_parameter",
            expression="lambda = E*nu / ((1+nu)*(1-2*nu))",
            variables=["lambda", "E", "nu"],
            relation_type="equality",
            description="First Lamé parameter",
            physical_meaning="Material stiffness in bulk deformation",
        ),
        ParameterRelationship(
            name="lame_second_parameter",
            expression="mu = E / (2*(1+nu))",
            variables=["mu", "E", "nu"],
            relation_type="equality",
            description="Second Lamé parameter (shear modulus)",
            physical_meaning="Material stiffness in shear",
        ),
        ParameterRelationship(
            name="bulk_modulus",
            expression="K = E / (3*(1-2*nu))",
            variables=["K", "E", "nu"],
            relation_type="equality",
            description="Bulk modulus",
            physical_meaning="Resistance to uniform compression",
        ),
        ParameterRelationship(
            name="effective_modulus_plane_stress",
            expression="E' = E / (1-nu^2)",
            variables=["E_prime", "E", "nu"],
            relation_type="equality",
            description="Effective modulus for plane stress",
            physical_meaning="Apparent stiffness in 2D",
        ),
    ]
    
    # Add symbolic constraints
    constraints = [
        SymbolicConstraint(
            expression="E > 0",
            description="Young's modulus must be positive",
            variables=["E"],
            confidence=1.0,
            inferred_from="material_stability",
        ),
        SymbolicConstraint(
            expression="-1 < nu < 0.5",
            description="Poisson's ratio bounds for positive definite tensor",
            variables=["nu"],
            confidence=1.0,
            inferred_from="stability_requirement",
        ),
        SymbolicConstraint(
            expression="mu > 0",
            description="Shear modulus must be positive",
            variables=["mu"],
            confidence=1.0,
            inferred_from="material_stability",
        ),
        SymbolicConstraint(
            expression="3*lambda + 2*mu > 0",
            description="Bulk modulus must be positive",
            variables=["lambda", "mu"],
            confidence=1.0,
            inferred_from="positive_definite_requirement",
        ),
        SymbolicConstraint(
            expression="det(K) != 0",
            description="Stiffness matrix must be non-singular",
            variables=["K"],
            confidence=0.9,
            inferred_from="fem_solution_existence",
        ),
    ]
    
    # Add validation results
    if E > 0:
        constraints.append(SymbolicConstraint(
            expression=f"E ({E}) > 0 ✓",
            description="Young's modulus positive",
            variables=["E"],
            confidence=1.0,
            inferred_from="validation",
        ))
    
    if -1 < nu < 0.5:
        constraints.append(SymbolicConstraint(
            expression=f"-1 < nu ({nu}) < 0.5 ✓",
            description="Poisson's ratio in valid range",
            variables=["nu"],
            confidence=1.0,
            inferred_from="validation",
        ))
    else:
        constraints.append(SymbolicConstraint(
            expression=f"nu ({nu}) outside valid range",
            description="WARNING: Invalid Poisson's ratio",
            variables=["nu"],
            confidence=1.0,
            inferred_from="validation",
        ))
    
    schema.symbolic_constraints = constraints
    
    # Add constitutive relations
    schema.mathematical_model.constitutive_relations = [{
        "type": "elastic",
        "name": "linear_isotropic",
        "form": "σ = C : ε",
        "parameters": {"E": E, "nu": nu, "G": E / (2 * (1 + nu))},
    }]
    
    return schema


def main():
    print("=" * 70)
    print("Math Anything - FEM Symbolic Extraction Demo")
    print("=" * 70)
    print()
    
    # Create material (Steel)
    material = Material(
        name="Steel",
        youngs_modulus=210000.0,  # MPa
        poisson_ratio=0.3,
    )
    
    print(f"📦 Material: {material.name}")
    print(f"   E = {material.youngs_modulus} MPa")
    print(f"   ν = {material.poisson_ratio}")
    print()
    
    # Generate schema
    print("🔍 Extracting mathematical structures...")
    print("-" * 70)
    schema = create_fem_schema(material)
    print()
    
    # Display results
    print("=" * 70)
    print("📊 Extraction Results")
    print("=" * 70)
    print()
    
    # Governing Equations
    equations = schema.mathematical_model.governing_equations
    print(f"🧮 Governing Equations: {len(equations)}")
    for eq in equations:
        print(f"   • {eq.name}")
        print(f"     Form: {eq.mathematical_form}")
    print()
    
    # Parameter Relationships
    relationships = schema.mathematical_model.parameter_relationships
    print(f"📐 Parameter Relationships: {len(relationships)}")
    for rel in relationships:
        print(f"   • {rel.name}")
        print(f"     Expression: {rel.expression}")
        if rel.physical_meaning:
            print(f"     Physical: {rel.physical_meaning}")
    print()
    
    # Calculate Lamé parameters
    E = material.youngs_modulus
    nu = material.poisson_ratio
    lambda_param = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_param = E / (2 * (1 + nu))
    K_param = E / (3 * (1 - 2 * nu))
    
    print("📊 Calculated Lamé Parameters:")
    print(f"   λ = {lambda_param:.2f} MPa")
    print(f"   μ = {mu_param:.2f} MPa")
    print(f"   K = {K_param:.2f} MPa")
    print()
    
    # Symbolic Constraints
    constraints = schema.symbolic_constraints
    print(f"✅ Symbolic Constraints: {len(constraints)}")
    for c in constraints:
        status = "✓" if "✓" in c.expression else "⚠" if "WARNING" in c.description else "•"
        print(f"   {status} {c.expression}")
    print()
    
    # Save schema
    output_file = Path(__file__).parent / "fem_simulation" / "beam_bending_schema.json"
    schema.save(str(output_file))
    print(f"💾 Schema saved to: {output_file}")
    
    print()
    print("=" * 70)
    print("✅ FEM Extraction Complete!")
    print("=" * 70)
    print()
    print("Key insights for LLM:")
    print("1. Governing PDE: ∇·σ + b = 0 (equilibrium)")
    print("2. Constitutive: σ = λ tr(ε) I + 2με")
    print("3. Material constraints: E > 0, -1 < ν < 0.5 ✓")
    print("4. Lamé parameters: λ, μ, K derived from E, ν")
    print("5. FEM solves K u = F (sparse linear system)")


if __name__ == "__main__":
    main()
