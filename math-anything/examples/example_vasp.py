"""Example: VASP DFT extraction.

This example demonstrates extracting mathematical structures from
VASP density functional theory calculations.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vasp-harness"))

from math_anything import load_harness
from math_anything.vasp.core.harness import VaspHarness

# Example VASP input files
EXAMPLE_INCAR = """
SYSTEM = Silicon bulk DFT calculation
ENCUT = 300
eV    # Plane wave cutoff
ISMEAR = 0
SIGMA = 0.1
EDIFF = 1E-6
IBRION = -1
NSW = 0
NELM = 100
NELMIN = 4
ISPIN = 1
ALGO = Normal
PREC = Normal
"""

EXAMPLE_POSCAR = """
Si_Diamond
1.0
    5.4300000000    0.0000000000    0.0000000000
    0.0000000000    5.4300000000    0.0000000000
    0.0000000000    0.0000000000    5.4300000000
   Si
    8
Direct
    0.0000000000    0.0000000000    0.0000000000
    0.0000000000    0.5000000000    0.5000000000
    0.5000000000    0.0000000000    0.5000000000
    0.5000000000    0.5000000000    0.0000000000
    0.2500000000    0.2500000000    0.2500000000
    0.2500000000    0.7500000000    0.7500000000
    0.7500000000    0.2500000000    0.7500000000
    0.7500000000    0.7500000000    0.2500000000
"""

EXAMPLE_KPOINTS = """
Automatic mesh
0
Gamma
  8  8  8
  0.  0.  0.
"""

EXAMPLE_OUTCAR = """
 vasp.5.4.4.18Apr17-6-g9f103f2a35 (build Apr 18 2017 09:54:44) complex
 
 POSCAR = Si_Diamond
  PAW_PBE Si 05Jan2001                   :
 energy of atom  1       EATOM= -101.1747
  kinetic energy error for atom=    0.0004 (will be added to EATOM!!)
 
 
 -------------------------------------------------------------------------
  iteration      Etot          Eband        Ehartree      dEtot  
 -------------------------------------------------------------------------
      1      -43.624567     -16.230958     -40.234902   0.144E+03
      2      -43.816293     -16.456721     -40.012344   0.192E+00
      3      -43.816510     -16.458012     -40.010987   0.217E-03
      4      -43.816512     -16.458123     -40.010876   0.235E-05
      5      -43.816512     -16.458145     -40.010854   0.145E-07
  reached required accuracy - stopping structural energy minimisation
 -------------------------------------------------------------------------
 
  E-fermi :  -0.000042
 
  free energy   TOTEN  =       -43.816512 eV
 
  energy without entropy =      -43.816512
  energy(sigma->0) =      -43.816512
"""


def main():
    """Run VASP extraction example."""
    print("=" * 70)
    print("Math Anything - VASP DFT Extraction Example")
    print("=" * 70)
    print()

    # Create temporary files
    temp_dir = tempfile.mkdtemp()

    incar_path = os.path.join(temp_dir, "INCAR")
    poscar_path = os.path.join(temp_dir, "POSCAR")
    kpoints_path = os.path.join(temp_dir, "KPOINTS")
    outcar_path = os.path.join(temp_dir, "OUTCAR")

    with open(incar_path, "w") as f:
        f.write(EXAMPLE_INCAR)
    with open(poscar_path, "w") as f:
        f.write(EXAMPLE_POSCAR)
    with open(kpoints_path, "w") as f:
        f.write(EXAMPLE_KPOINTS)
    with open(outcar_path, "w") as f:
        f.write(EXAMPLE_OUTCAR)

    print("Created example VASP input files:")
    print(f"  - INCAR: {incar_path}")
    print(f"  - POSCAR: {poscar_path}")
    print(f"  - KPOINTS: {kpoints_path}")
    print(f"  - OUTCAR: {outcar_path}")
    print()

    try:
        # Load VASP harness
        print("Loading VASP harness...")
        harness = load_harness("vasp")
        print(f"✓ Loaded harness: {harness.engine_name}")
        print(f"  Schema version: {harness.supported_schema_version}")
        print()

        # Extract mathematical structure
        print("Extracting mathematical structure from DFT calculation...")
        schema = harness.extract(
            {
                "incar": incar_path,
                "poscar": poscar_path,
                "kpoints": kpoints_path,
                "outcar": outcar_path,
            }
        )
        print("✓ Extraction complete")
        print()

        # Display key findings
        data = schema.to_dict()

        print("-" * 70)
        print("EXTRACTED DFT MATHEMATICAL MODEL")
        print("-" * 70)

        # Governing equations
        print("\n1. GOVERNING EQUATIONS (DFT Core)")
        for eq in data["mathematical_model"]["governing_equations"]:
            print(f"   [{eq['id']}] {eq['name']}")
            print(f"       Type: {eq['type']}")
            print(f"       Form: {eq['mathematical_form']}")
            if eq.get("description"):
                print(f"       Description: {eq['description']}")
            if eq.get("parameters"):
                print(f"       Parameters: {eq['parameters']}")

        # Boundary conditions
        print("\n2. BOUNDARY CONDITIONS")
        for bc in data["mathematical_model"]["boundary_conditions"]:
            print(f"   [{bc['id']}] Type: {bc['type']}")
            mo = bc.get("mathematical_object", {})
            if mo.get("tensor_form"):
                print(f"       Tensor Form: {mo['tensor_form']}")
            if mo.get("tensor_rank") is not None:
                print(f"       Tensor Rank: {mo['tensor_rank']}")

        # Numerical method
        print("\n3. NUMERICAL METHOD")
        nm = data["numerical_method"]
        disc = nm.get("discretization", {})
        print(f"   Space Discretization: {disc.get('space_discretization')}")
        print(f"   Time Integrator: {disc.get('time_integrator')}")
        solver = nm.get("solver", {})
        print(f"   Algorithm: {solver.get('algorithm')}")
        print(
            f"   Convergence: {solver.get('convergence_criterion')} < {solver.get('tolerance')}"
        )

        # Computational graph
        print("\n4. COMPUTATIONAL GRAPH (SCF Cycle)")
        cg = data["computational_graph"]
        print(f"   Version: {cg.get('version')}")
        print(f"   Nodes: {len(cg.get('nodes', []))}")
        print(f"   Edges: {len(cg.get('edges', []))}")
        print("\n   SCF Iteration Nodes:")
        for node in cg.get("nodes", []):
            print(f"     - {node['id']}: {node['type']}")
            semantics = node.get("math_semantics", {})
            updates = semantics.get("updates", {})
            print(f"       Mode: {updates.get('mode', 'N/A')}")

        # Conservation properties
        print("\n5. CONSERVATION PROPERTIES")
        for prop_name, prop_data in data.get("conservation_properties", {}).items():
            preserved = prop_data.get("preserved", False)
            status = "✓" if preserved else "✗"
            print(f"   {status} {prop_name}")
            if prop_data.get("mechanism"):
                print(f"       Mechanism: {prop_data['mechanism']}")

        # Save to JSON
        output_file = os.path.join(temp_dir, "dft_model.json")
        schema.save(output_file)
        print()
        print("-" * 70)
        print(f"✓ Saved DFT schema to: {output_file}")

        # Show JSON preview
        import json

        print("\nJSON Preview (first 50 lines):")
        print("-" * 70)
        with open(output_file, "r") as f:
            lines = f.readlines()
            for line in lines[:50]:
                print(line.rstrip())
            if len(lines) > 50:
                print(f"... ({len(lines) - 50} more lines)")

        print()
        print("=" * 70)
        print("VASP DFT Extraction Complete!")
        print("=" * 70)
        print("""
Key DFT Mathematical Structures Extracted:
• Kohn-Sham equations (nonlinear eigenvalue problem)
• Hohenberg-Kohn variational principle
• Plane wave basis representation
• SCF iteration cycle (implicit loop)
• Periodic boundary conditions (Bloch theorem)
• Exchange-correlation functional
• PAW pseudopotentials

Physics Captured:
- Quantum mechanical electronic structure
- Self-consistent field iteration
- Brillouin zone sampling
- Fermi-Dirac occupation
""")

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
