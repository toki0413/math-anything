"""Example: Cross-Engine Session - Multi-scale Model Coupling

This example demonstrates how to extract models from multiple engines
at different scales and identify coupling interfaces between them.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from math_anything.core import (
    CrossEngineSession,
    CouplingInterface,
    ModelScale,
    CouplingType,
)
from math_anything.schemas import MathSchema, GoverningEquation, MathematicalObject


def create_mock_md_schema():
    """Create a mock MD schema with stress tensor."""
    schema = MathSchema()
    
    # Add virial stress equation
    schema.mathematical_model.governing_equations.append(
        GoverningEquation(
            id="virial_stress",
            type="tensor_equation",
            name="Virial Stress Tensor",
            mathematical_form="σ_{αβ} = (1/V)[Σ m_i v_iα v_iβ + Σ r_iα F_iβ]",
            variables=["stress_tensor", "positions", "velocities", "forces", "volume"],
            description="Per-atom virial stress from MD simulation",
        )
    )
    
    # Add temperature equation
    schema.mathematical_model.governing_equations.append(
        GoverningEquation(
            id="temperature",
            type="scalar",
            name="Kinetic Temperature",
            mathematical_form="T = (2/3k_B) * (1/N) Σ (1/2 m_i v_i²)",
            variables=["temperature", "kinetic_energy", "velocity"],
        )
    )
    
    return schema


def create_mock_fem_schema():
    """Create a mock FEM schema with stress."""
    schema = MathSchema()
    
    # Add equilibrium equation
    schema.mathematical_model.governing_equations.append(
        GoverningEquation(
            id="equilibrium",
            type="pde",
            name="Equilibrium Equation",
            mathematical_form="∂σ_{ij}/∂x_j + f_i = 0",
            variables=["stress", "displacement", "body_force"],
            description="Static equilibrium in continuum mechanics",
        )
    )
    
    # Add constitutive relation
    schema.mathematical_model.governing_equations.append(
        GoverningEquation(
            id="constitutive",
            type="tensor_equation",
            name="Constitutive Relation",
            mathematical_form="σ_{ij} = C_{ijkl} ε_{kl}",
            variables=["stress", "strain", "stiffness_tensor"],
        )
    )
    
    # Add heat equation
    schema.mathematical_model.governing_equations.append(
        GoverningEquation(
            id="heat_equation",
            type="pde",
            name="Heat Conduction",
            mathematical_form="ρ c_p ∂T/∂t = ∇·(k ∇T) + Q",
            variables=["temperature", "time", "density", "conductivity", "heat_source"],
        )
    )
    
    return schema


def create_mock_quantum_schema():
    """Create a mock quantum/DFT schema."""
    schema = MathSchema()
    
    schema.mathematical_model.governing_equations.append(
        GoverningEquation(
            id="kohn_sham",
            type="eigenvalue_problem",
            name="Kohn-Sham Equations",
            mathematical_form="[-ℏ²∇²/2m + V_eff(r)] ψ_i(r) = ε_i ψ_i(r)",
            variables=["wavefunction", "eigenvalue", "effective_potential"],
            description="Density functional theory electronic structure",
        )
    )
    
    return schema


def example_1_basic_coupling():
    """Example 1: Basic MD-to-FEM coupling."""
    print("=" * 70)
    print("EXAMPLE 1: Basic MD-to-FEM Coupling")
    print("=" * 70)
    print()
    
    session = CrossEngineSession()
    
    # Add MD model
    md_schema = create_mock_md_schema()
    from math_anything.core.cross_engine import ScaleModel
    session.models["md_bulk"] = ScaleModel(
        model_id="md_bulk",
        scale=ModelScale.ATOMISTIC,
        engine="lammps",
        schema=md_schema,
        domain=" Representative Volume Element",
    )
    print("✓ Added MD model (atomistic scale)")
    
    # Add FEM model
    fem_schema = create_mock_fem_schema()
    session.models["fem_structure"] = ScaleModel(
        model_id="fem_structure",
        scale=ModelScale.CONTINUUM,
        engine="abaqus",
        schema=fem_schema,
        domain="structural component",
    )
    print("✓ Added FEM model (continuum scale)")
    print()
    
    # Auto-detect coupling
    print("Auto-detecting coupling interfaces...")
    interfaces = session.auto_detect_coupling()
    
    print(f"  Found {len(interfaces)} coupling interface(s):")
    for iface in interfaces:
        print(f"    - {iface.interface_id}")
        print(f"      Type: {iface.transfer_quantity}")
        print(f"      Mapping: {iface.mapping_type}")
        print(f"      Conservation check: {iface.conservation_check}")
    print()


def example_2_three_scale_hierarchy():
    """Example 2: Three-scale hierarchy (Quantum -> MD -> FEM)."""
    print("=" * 70)
    print("EXAMPLE 2: Three-Scale Hierarchy")
    print("=" * 70)
    print()
    
    session = CrossEngineSession()
    
    # Quantum scale (DFT)
    qm_schema = create_mock_quantum_schema()
    from math_anything.core.cross_engine import ScaleModel
    session.models["dft_bulk"] = ScaleModel(
        model_id="dft_bulk",
        scale=ModelScale.QUANTUM,
        engine="vasp",
        schema=qm_schema,
        domain="interatomic potential training",
    )
    print("✓ Added DFT model (quantum scale)")
    
    # Atomistic scale (MD)
    md_schema = create_mock_md_schema()
    session.models["md_defect"] = ScaleModel(
        model_id="md_defect",
        scale=ModelScale.ATOMISTIC,
        engine="lammps",
        schema=md_schema,
        domain="crack tip region",
    )
    print("✓ Added MD model (atomistic scale)")
    
    # Continuum scale (FEM)
    fem_schema = create_mock_fem_schema()
    session.models["fem_specimen"] = ScaleModel(
        model_id="fem_specimen",
        scale=ModelScale.CONTINUUM,
        engine="abaqus",
        schema=fem_schema,
        domain="fracture specimen",
    )
    print("✓ Added FEM model (continuum scale)")
    print()
    
    # Show scale hierarchy
    print("Scale Hierarchy (micro to macro):")
    hierarchy = session.get_scale_hierarchy()
    for i, (model_id, scale) in enumerate(hierarchy):
        arrow = "→" if i < len(hierarchy) - 1 else ""
        print(f"  {scale.value:12} ({model_id}) {arrow}")
    print()
    
    # Detect all coupling
    interfaces = session.auto_detect_coupling()
    print(f"Detected {len(interfaces)} coupling interfaces:")
    for iface in interfaces:
        print(f"  • {iface.from_scale} → {iface.to_scale}")
        print(f"    Transfer: {iface.transfer_quantity}")
    print()


def example_3_manual_coupling():
    """Example 3: Manually defining coupling interfaces."""
    print("=" * 70)
    print("EXAMPLE 3: Manual Coupling Definition")
    print("=" * 70)
    print()
    
    session = CrossEngineSession()
    
    # Add models
    from math_anything.core.cross_engine import ScaleModel
    session.models["md"] = ScaleModel(
        model_id="md",
        scale=ModelScale.ATOMISTIC,
        engine="lammps",
        schema=create_mock_md_schema(),
    )
    
    session.models["fem"] = ScaleModel(
        model_id="fem",
        scale=ModelScale.CONTINUUM,
        engine="abaqus",
        schema=create_mock_fem_schema(),
    )
    
    # Manually add a specialized coupling
    custom_iface = CouplingInterface(
        interface_id="multiscale_stress_hill",
        from_scale="md",
        to_scale="fem",
        from_objects=["equation:virial_stress"],
        to_objects=["equation:constitutive"],
        mapping_type="hill_average",
        transfer_quantity="stress_tensor",
        conservation_check=True,
    )
    
    session.add_manual_interface(custom_iface)
    print("✓ Added manual coupling interface:")
    print(f"  ID: {custom_iface.interface_id}")
    print(f"  Mapping: {custom_iface.mapping_type}")
    print(f"  From: {custom_iface.from_objects}")
    print(f"  To: {custom_iface.to_objects}")
    print()


def example_4_consistency_check():
    """Example 4: Checking coupled system consistency."""
    print("=" * 70)
    print("EXAMPLE 4: Consistency Checking")
    print("=" * 70)
    print()
    
    session = CrossEngineSession()
    
    # Create inconsistent system
    from math_anything.core.cross_engine import ScaleModel
    session.models["md"] = ScaleModel(
        model_id="md",
        scale=ModelScale.ATOMISTIC,
        engine="lammps",
        schema=MathSchema(),
    )
    
    # Add interface to non-existent model
    session.coupling_interfaces.append(CouplingInterface(
        interface_id="bad_coupling",
        from_scale="md",
        to_scale="nonexistent_fem",
    ))
    
    # Check consistency
    report = session.check_consistency()
    
    print("Consistency Report:")
    print(f"  Consistent: {report['consistent']}")
    print(f"  Errors: {len(report['errors'])}")
    print(f"  Warnings: {len(report['warnings'])}")
    
    if report['errors']:
        print("\n  Errors found:")
        for error in report['errors']:
            print(f"    ✗ {error}")
    print()
    
    # Now create consistent system
    print("Creating consistent system...")
    session2 = CrossEngineSession()
    session2.models["md"] = ScaleModel(
        model_id="md",
        scale=ModelScale.ATOMISTIC,
        engine="lammps",
        schema=MathSchema(),
    )
    session2.models["fem"] = ScaleModel(
        model_id="fem",
        scale=ModelScale.CONTINUUM,
        engine="abaqus",
        schema=MathSchema(),
    )
    session2.coupling_interfaces.append(CouplingInterface(
        interface_id="valid_coupling",
        from_scale="md",
        to_scale="fem",
    ))
    
    report2 = session2.check_consistency()
    print(f"  Consistent: {report2['consistent']}")
    print(f"  Errors: {len(report2['errors'])}")
    print()


def example_5_generate_coupled_schema():
    """Example 5: Generating the final coupled schema."""
    print("=" * 70)
    print("EXAMPLE 5: Generating Coupled Schema")
    print("=" * 70)
    print()
    
    session = CrossEngineSession()
    
    # Add models
    from math_anything.core.cross_engine import ScaleModel
    session.models["md"] = ScaleModel(
        model_id="md",
        scale=ModelScale.ATOMISTIC,
        engine="lammps",
        schema=create_mock_md_schema(),
        domain="grain boundary",
    )
    
    session.models["fem"] = ScaleModel(
        model_id="fem",
        scale=ModelScale.CONTINUUM,
        engine="abaqus",
        schema=create_mock_fem_schema(),
        domain="polycrystal",
    )
    
    # Detect coupling
    session.auto_detect_coupling()
    
    # Generate coupled schema
    coupled = session.generate_coupled_schema(CouplingType.HIERARCHICAL)
    
    print("Generated Coupled Schema:")
    print(f"  Version: {coupled.schema_version}")
    print(f"  Coupling Type: {coupled.coupling_type.name}")
    print(f"  Models: {len(coupled.models)}")
    print(f"  Interfaces: {len(coupled.coupling_interfaces)}")
    print()
    
    # Show JSON structure
    import json
    data = coupled.to_dict()
    print("JSON Structure:")
    print(json.dumps(data, indent=2, default=str)[:1500])
    print("\n...")
    print()


def example_6_summary():
    """Example 6: Session summary."""
    print("=" * 70)
    print("EXAMPLE 6: Session Summary")
    print("=" * 70)
    print()
    
    session = CrossEngineSession()
    
    from math_anything.core.cross_engine import ScaleModel
    session.models["dft"] = ScaleModel(
        model_id="dft",
        scale=ModelScale.QUANTUM,
        engine="vasp",
        schema=MathSchema(),
    )
    session.models["md1"] = ScaleModel(
        model_id="md1",
        scale=ModelScale.ATOMISTIC,
        engine="lammps",
        schema=MathSchema(),
    )
    session.models["md2"] = ScaleModel(
        model_id="md2",
        scale=ModelScale.ATOMISTIC,
        engine="lammps",
        schema=MathSchema(),
    )
    session.models["fem"] = ScaleModel(
        model_id="fem",
        scale=ModelScale.CONTINUUM,
        engine="abaqus",
        schema=MathSchema(),
    )
    
    summary = session.get_summary()
    
    print("Session Summary:")
    print(f"  Total Models: {summary['n_models']}")
    print(f"  Coupling Interfaces: {summary['n_interfaces']}")
    print(f"  Scales Present: {summary['scales_present']}")
    print(f"  Engines Used: {summary['engines_used']}")
    print(f"  Model IDs: {summary['model_ids']}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "CROSS-ENGINE SESSION - MULTI-SCALE COUPLING" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    example_1_basic_coupling()
    example_2_three_scale_hierarchy()
    example_3_manual_coupling()
    example_4_consistency_check()
    example_5_generate_coupled_schema()
    example_6_summary()
    
    print("=" * 70)
    print("Cross-Engine Session Demo Complete!")
    print("=" * 70)
    print("""
Key Takeaways:
1. CrossEngineSession manages models from multiple engines
2. Auto-detect coupling identifies transfer quantities (stress, temp, etc.)
3. Supports hierarchical, concurrent, sequential, and embedded coupling
4. CouplingInterface defines micro-to-macro mappings
5. Consistency checking validates the coupled system
6. Generates unified CoupledSchema for multi-scale workflows

Use Cases:
- Atomistic-to-continuum (AtC) coupling
- Quantum-informed MD (DFT + MD)
- Hierarchical multiscale modeling
- Concurrent multiscale simulation
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
