"""Test Mathematical Precision Extraction for All Harnesses.

Demonstrates how the enhanced Math Schema provides precise mathematical
structure representation for LLM understanding across all supported engines.
"""

from math_anything.schemas import (
    extract_abaqus_mathematical_precision,
    extract_ansys_mathematical_precision,
    extract_comsol_mathematical_precision,
    extract_gromacs_mathematical_precision,
    extract_lammps_mathematical_precision,
    extract_multiwfn_mathematical_precision,
    extract_vasp_mathematical_precision,
)


def print_result(name: str, result: dict):
    """Print formatted result."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print("=" * 70)

    ms = result.get("mathematical_structure", {})
    print(f"\n📐 Mathematical Structure:")
    print(f"  Problem Type: {ms.get('problem_type')}")
    print(f"  Canonical Form: {ms.get('canonical_form')}")

    deps = result.get("variable_dependencies", [])
    if deps:
        print(f"\n🔗 Variable Dependencies:")
        for dep in deps[:3]:
            circular = "→ Iteration needed" if dep.get("circular") else ""
            print(f"  • {dep.get('relation')} {circular}")

    approxs = result.get("approximations", [])
    if approxs:
        print(f"\n📝 Approximations:")
        for approx in approxs[:3]:
            print(f"  • {approx.get('name')}: {approx.get('consequence')[:50]}...")

    hierarchy = result.get("mathematical_decoding", {}).get(
        "mathematical_hierarchy", []
    )
    if hierarchy:
        print(f"\n🔍 Mathematical Hierarchy:")
        for level in hierarchy[:4]:
            print(f"  [{level.get('level')}] {level.get('description')}")


def test_all_harnesses():
    """Test all harness precision extractors."""

    print("=" * 70)
    print("Math Anything - Mathematical Precision Extraction for All Harnesses")
    print("=" * 70)

    # VASP (DFT)
    vasp_result = extract_vasp_mathematical_precision(
        {
            "ENCUT": 520,
            "EDIFF": 1e-6,
            "ISMEAR": 0,
            "GGA": "PE",
        }
    )
    print_result("VASP (DFT - Density Functional Theory)", vasp_result)

    # LAMMPS (MD)
    lammps_result = extract_lammps_mathematical_precision(
        {
            "dt": 0.001,
            "run": 100000,
            "ensemble": "NVT",
            "pair_style": "lj/cut",
            "n_atoms": 1000,
        }
    )
    print_result("LAMMPS (MD - Molecular Dynamics)", lammps_result)

    # Abaqus (FEM)
    abaqus_result = extract_abaqus_mathematical_precision(
        {
            "analysis_type": "static",
            "element_type": "C3D8R",
            "mesh_size": 0.01,
            "material_model": "elastic",
        }
    )
    print_result("Abaqus (FEM - Finite Element Method)", abaqus_result)

    # Ansys (FEM)
    ansys_result = extract_ansys_mathematical_precision(
        {
            "analysis_type": "modal",
            "element_type": "SOLID185",
            "EX": 2.1e11,
            "PRXY": 0.3,
        }
    )
    print_result("Ansys (FEM - Finite Element Method)", ansys_result)

    # COMSOL (Multiphysics)
    comsol_result = extract_comsol_mathematical_precision(
        {
            "physics_modules": ["solid_mechanics", "heat_transfer"],
            "study_type": "stationary",
            "coupling_type": "bidirectional",
            "element_order": 2,
        }
    )
    print_result("COMSOL (Multiphysics)", comsol_result)

    # GROMACS (Biomolecular MD)
    gromacs_result = extract_gromacs_mathematical_precision(
        {
            "integrator": "md",
            "dt": 0.002,
            "nsteps": 500000,
            "forcefield": "amber99sb-ildn",
            "constraints": "h-bonds",
        }
    )
    print_result("GROMACS (Biomolecular MD)", gromacs_result)

    # Multiwfn (Wavefunction Analysis)
    multiwfn_result = extract_multiwfn_mathematical_precision(
        {
            "analysis_type": "aim",
            "method": "B3LYP",
            "basis_type": "def2-TZVP",
            "grid_type": "becke",
        }
    )
    print_result("Multiwfn (Wavefunction Analysis)", multiwfn_result)

    print("\n" + "=" * 70)
    print("✓ All Harness Mathematical Precision Extraction Complete")
    print("=" * 70)

    print("""
Summary of Mathematical Structures:
───────────────────────────────────
| Engine    | Problem Type              | Key Characteristic          |
|-----------|---------------------------|----------------------------|
| VASP      | nonlinear_eigenvalue      | SCF iteration required      |
| LAMMPS    | initial_value_ode         | Time integration            |
| Abaqus    | boundary_value_problem    | FEM discretization          |
| Ansys     | eigenvalue/boundary_value | Modal/static analysis       |
| COMSOL    | coupled_pde_system        | Multiphysics coupling       |
| GROMACS   | initial_value_ode         | Biomolecular MD             |
| Multiwfn  | post_processing_analysis  | Wavefunction analysis       |

LLM Benefits:
1. Understand mathematical essence, not just parameter values
2. Compare different computational approaches mathematically
3. Decrypt black-box computational setups
4. Guide ML models with physics context
""")


if __name__ == "__main__":
    test_all_harnesses()
