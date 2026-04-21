"""Example: Math Anything Diff - Comparing simulation variants.

This example demonstrates how to use math-anything diff to track
mathematical structure changes between simulation versions.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lammps-harness"))

from math_anything import load_harness
from math_anything.lammps.core.harness import LammpsHarness
from math_anything.utils import MathDiffer

# Variant 1: NVE with small timestep
INPUT_V1 = """
units           metal
boundary        p p p
atom_style      atomic
lattice         fcc 3.52
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box
mass            1 58.69
pair_style      lj/cut 10.0
pair_coeff      1 1 0.54 2.5
velocity        all create 300.0 12345
fix             1 all nve
timestep        0.001
run             100
"""

# Variant 2: NVT with larger timestep (THERMOSTAT ADDED + TIMESTEP CHANGED)
INPUT_V2 = """
units           metal
boundary        p p p
atom_style      atomic
lattice         fcc 3.52
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box
mass            1 58.69
pair_style      lj/cut 10.0
pair_coeff      1 1 0.54 2.5
velocity        all create 300.0 12345
fix             1 all nvt temp 300 300 0.1
timestep        0.005
run             100
"""

# Variant 3: With fix deform (TENSOR BC ADDED)
INPUT_V3 = """
units           metal
boundary        p p p
atom_style      atomic
lattice         fcc 3.52
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box
mass            1 58.69
pair_style      lj/cut 10.0
pair_coeff      1 1 0.54 2.5
velocity        all create 300.0 12345
fix             1 all nve
fix             2 all deform 1 x erate 0.01
timestep        0.001
run             100
"""


def extract_schema(input_content, name):
    """Extract schema from LAMMPS input."""
    harness = load_harness("lammps")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False) as f:
        f.write(input_content)
        temp_path = f.name

    try:
        schema = harness.extract({"input": temp_path})
        return schema
    finally:
        os.unlink(temp_path)


def main():
    """Run diff examples."""
    print("=" * 70)
    print("Math Anything Diff - Example: Tracking Simulation Changes")
    print("=" * 70)
    print()

    # Extract schemas
    print("Extracting schemas...")
    schema_v1 = extract_schema(INPUT_V1, "v1_nve_small_dt")
    schema_v2 = extract_schema(INPUT_V2, "v2_nvt_large_dt")
    schema_v3 = extract_schema(INPUT_V3, "v3_nve_deform")
    print("✓ Extracted 3 variants")
    print()

    # Example 1: NVE -> NVT (integrator change)
    print("-" * 70)
    print("EXAMPLE 1: NVE → NVT (Integrator Change)")
    print("-" * 70)

    differ = MathDiffer()
    report1 = differ.compare(schema_v1, schema_v2, "NVE_small_dt", "NVT_large_dt")
    report1.print_summary()

    print("\nKey Observations:")
    print(
        "  • Time integrator changed: velocity_verlet → velocity_verlet_with_nose_hoover"
    )
    print("  • Timestep increased by 400% (0.001 → 0.005)")
    print("  • Energy conservation LOST (thermostat introduces thermal coupling)")
    print("  • Computational graph: added implicit_loop for thermostat convergence")
    print()

    # Example 2: NVE -> NVE + deform (tensor BC added)
    print("-" * 70)
    print("EXAMPLE 2: NVE → NVE + fix deform (Tensor BC Added)")
    print("-" * 70)

    differ = MathDiffer()
    report2 = differ.compare(schema_v1, schema_v3, "NVE", "NVE_deform")
    report2.print_summary()

    print("\nKey Observations:")
    print("  • Boundary condition added: fix_2 with rank-2 tensor")
    print("  • Tensor form: F_{ij} = ∂x_i/∂X_j (deformation gradient)")
    print("  • Computational node added: explicit_update deformation")
    print("  • Dual role: both BC and external drive")
    print()

    # Example 3: Critical changes only
    print("-" * 70)
    print("EXAMPLE 3: Critical Changes Filter")
    print("-" * 70)

    print(f"\nVariant 1 → 2 has {len(report1.critical_changes)} critical changes:")
    for change in report1.critical_changes:
        print(f"  • [{change.type.name}] {change.description}")

    print(f"\nVariant 1 → 3 has {len(report2.critical_changes)} critical changes:")
    for change in report2.critical_changes:
        print(f"  • [{change.type.name}] {change.description}")
    print()

    # Example 4: JSON export
    print("-" * 70)
    print("EXAMPLE 4: JSON Export for LLM Consumption")
    print("-" * 70)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_path = f.name
        f.write(report1.to_json(indent=2))

    print(f"\nSaved diff report to: {json_path}")
    print("\nJSON Structure (first 30 lines):")
    with open(json_path, "r") as f:
        lines = f.readlines()
        for line in lines[:30]:
            print(line.rstrip())
        if len(lines) > 30:
            print(f"... ({len(lines) - 30} more lines)")

    os.unlink(json_path)
    print()

    # Example 5: User workflow simulation
    print("-" * 70)
    print("EXAMPLE 5: User Debugging Workflow")
    print("-" * 70)

    print("""
User Story: "I changed my simulation parameters and now results are different.
             What mathematically changed?"

Step 1: Extract both versions
  $ math-anything extract -e lammps -i old.in -o old.json
  $ math-anything extract -e lammps -i new.in -o new.json

Step 2: Compare mathematically
  $ math-anything diff old.json new.json

Step 3: Check for critical changes
  $ math-anything diff old.json new.json --critical-only

Output Interpretation:
  ✓ INTEGRATOR_CHANGED (critical): Time integration scheme changed
  ✓ TIMESTEP_CHANGED (warning): 400% increase may affect accuracy
  ✓ CONSERVATION_LOST (critical): Energy no longer conserved

Decision: The timestep increase and thermostat addition are the likely
causes of result differences. Consider:
  1. Reducing timestep for accuracy testing
  2. Checking if thermostat parameters are appropriate
  3. Verifying if energy conservation is required for your physics
""")

    print("=" * 70)
    print("Diff functionality demonstration complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
