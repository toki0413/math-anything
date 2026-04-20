"""
Math Anything REPL Demo

Demonstrates the interactive REPL interface for extracting
and comparing mathematical structures.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'math-anything', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'math-anything', 'vasp-harness'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'math-anything', 'ansys-harness'))

from math_anything.repl import MathAnythingREPL, MathDiff
from math_anything.schemas import MathSchema


def demo_interactive_repl():
    """Demo interactive REPL mode."""
    print("=" * 70)
    print("  Math Anything REPL Demo")
    print("=" * 70)
    print("""
This demo shows the interactive REPL interface.

Available commands:
  load <engine> <files>  - Load input files
  extract               - Extract mathematical schema
  constraints           - Validate symbolic constraints
  compare <s1> <s2>     - Compare two schemas
  list                  - List loaded sessions
  save <file>           - Save session
  export <s> <file>     - Export schema to JSON
  exit                  - Quit REPL

Example workflow:
  math-anything> load vasp INCAR POSCAR KPOINTS
  ✓ Loaded as 'vasp_1'
  
  math-anything> extract
  📐 Governing Equations:
     [eigenvalue_problem] Kohn-Sham Equations
  
  math-anything> constraints
  ✓ ENCUT > 0
  ✓ EDIFF > 0
  ...
  Total: 11 passed, 0 failed

  math-anything> save my_session.json
  ✓ Session saved

To start the actual REPL, run:
  python -m math_anything.cli repl
""")


def demo_math_diff():
    """Demo math-anything diff functionality."""
    print("\n" + "=" * 70)
    print("  Math Anything Diff Demo")
    print("=" * 70)
    print("""
Math Anything Diff compares mathematical schemas at the semantic level.

Use cases:
1. Compare two VASP calculations (different ENCUT)
2. Compare VASP vs LAMMPS (different physics)
3. Compare converged vs non-converged calculations

Example:
  $ math-anything diff vasp_high_accuracy.json vasp_low_accuracy.json
  
  Similarity: 85.0%
  
  ✓ Common Equations:
    - Kohn-Sham Equations
  
  📌 Different Constraints:
    - ENCUT: 800 eV vs 400 eV
    - EDIFF: 1e-06 vs 1e-04
  
  💡 High similarity: Same mathematical model, different precision settings.

Cross-engine comparison:
  $ math-anything cross vasp_schema.json quantum_espresso
  
  🔄 Mappable Parameters:
    ENCUT          -> ecutwfc        (energy_cutoff)
    ISMEAR         -> smearing       (smearing_method)
    SIGMA          -> degauss        (smearing_width)
  
  ⚠️ Unmapped Parameters (manual attention needed):
    - LORBIT
    - LCHARG
""")


def demo_cli_commands():
    """Demo CLI commands."""
    print("\n" + "=" * 70)
    print("  Math Anything CLI Commands")
    print("=" * 70)
    print("""
One-shot commands (no REPL):

1. Extract schema:
   $ math-anything extract vasp INCAR POSCAR KPOINTS --output schema.json

2. Validate constraints:
   $ math-anything validate schema.json
   
   Symbolic Constraint Validation
   ==============================
   ✓ ENCUT > 0
   ✓ EDIFF > 0
   ✓ ISMEAR in [-5,...,2]
   Total: 11 passed, 0 failed

3. Compare schemas:
   $ math-anything diff calc1.json calc2.json

4. Cross-engine mapping:
   $ math-anything cross vasp_calc.json quantum_espresso

5. Interactive REPL:
   $ math-anything repl
   $ math-anything repl --session saved_session.json
""")


def main():
    """Run all demos."""
    demo_interactive_repl()
    demo_math_diff()
    demo_cli_commands()
    
    print("\n" + "=" * 70)
    print("  To try the actual REPL:")
    print("=" * 70)
    print("""
  cd math-anything/math-anything/core
  python -m math_anything.cli repl
  
  Or install and run:
  pip install -e .
  math-anything repl
""")


if __name__ == "__main__":
    main()
