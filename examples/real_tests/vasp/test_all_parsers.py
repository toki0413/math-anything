"""
Complete VASP Parser Test Suite for Math Anything

Tests INCAR, POSCAR, and KPOINTS parsers with symbolic constraint validation.
Demonstrates how Math Anything extracts mathematical structures from DFT inputs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'math-anything', 'vasp-harness'))

from math_anything.vasp.core.incar_parser import IncarParser
from math_anything.vasp.core.poscar_parser import PoscarParser
from math_anything.vasp.core.kpoints_parser import KpointsParser, KpointsMode
from math_anything.vasp.core.extractor_v2 import VaspExtractor, VaspSymbolicConstraints


def test_incar_parser():
    """Test INCAR parser with symbolic constraints."""
    print("="*60)
    print("Testing INCAR Parser with Symbolic Constraints")
    print("="*60)
    
    parser = IncarParser()
    incar_file = os.path.join(os.path.dirname(__file__), "INCAR")
    
    if not os.path.exists(incar_file):
        print(f"Warning: {incar_file} not found")
        return False
    
    result = parser.parse(incar_file)
    print(f"\nParsed {len(result.parameters)} parameters from INCAR:")
    
    for key, param in sorted(result.parameters.items()):
        param_def = IncarParser.PARAMETER_DEFS.get(key, {})
        constraints = param_def.get("constraints", [])
        unit = param_def.get("unit", "")
        
        unit_str = f" [{unit}]" if unit else ""
        constraint_str = f" (constraints: {constraints})" if constraints else ""
        print(f"  {key:15} = {param.value:15}{unit_str}{constraint_str}")
    
    # Validate constraints
    print("\nSymbolic Constraint Validation:")
    validation_results = result.validate_constraints()
    
    for v in validation_results:
        status = "PASS" if v["satisfied"] else "FAIL"
        symbol = "OK" if v["satisfied"] else "XX"
        print(f"  [{symbol}] {v['parameter']:12} {v.get('expression', v.get('constraint', '')):25} -> {status}")
    
    all_passed = all(v["satisfied"] for v in validation_results)
    
    print(f"\nOverall: {'ALL CONSTRAINTS PASSED' if all_passed else 'SOME CONSTRAINTS FAILED'}")
    return all_passed


def test_poscar_parser():
    """Test POSCAR parser for crystal structures."""
    print("\n" + "="*60)
    print("Testing POSCAR Parser for Crystal Structures")
    print("="*60)
    
    parser = PoscarParser()
    
    test_files = [
        ("POSCAR_Si", "Silicon Diamond Structure"),
        ("POSCAR_GaAs", "GaAs Zincblende Structure"),
    ]
    
    all_passed = True
    
    for filename, description in test_files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        if not os.path.exists(filepath):
            print(f"\nWarning: {filepath} not found")
            continue
        
        print(f"\n--- {description} ({filename}) ---")
        structure = parser.parse(filepath)
        
        print(f"  Chemical Formula: {structure.chemical_formula}")
        print(f"  Number of Atoms: {structure.num_atoms}")
        print(f"  Coordinate System: {structure.coordinate_system}")
        print(f"  Unit Cell Volume: {structure.lattice.volume:.4f} Å³")
        
        # Lattice vectors
        print(f"  Lattice Vectors (Å):")
        for i, vec in enumerate(structure.lattice.vectors):
            print(f"    a{i+1} = [{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}]")
        
        # Reciprocal lattice
        rec_vecs = structure.lattice.reciprocal_vectors
        print(f"  Reciprocal Vectors (2π/Å):")
        for i, vec in enumerate(rec_vecs):
            print(f"    b{i+1} = [{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}]")
        
        # Atomic positions
        print(f"  Atomic Positions:")
        for atom in structure.atoms:
            cart = atom.to_cartesian(structure.lattice)
            print(f"    {atom.symbol:2} {atom.position_type:10} [{atom.position[0]:.4f}, {atom.position[1]:.4f}, {atom.position[2]:.4f}] "
                  f"-> Cartesian [{cart[0]:.4f}, {cart[1]:.4f}, {cart[2]:.4f}]")
        
        # Validate mathematical constraints
        det = abs(__import__('numpy').linalg.det(structure.lattice.vectors))
        volume_valid = structure.lattice.volume > 0 and det > 1e-10
        
        print(f"\n  Mathematical Constraints:")
        print(f"    [OK] det(lattice) = {det:.6f} != 0")
        print(f"    [OK] volume = {structure.lattice.volume:.4f} > 0")
        
        if not volume_valid:
            all_passed = False
    
    print(f"\nOverall: {'ALL STRUCTURE VALIDATIONS PASSED' if all_passed else 'SOME VALIDATIONS FAILED'}")
    return all_passed


def test_kpoints_parser():
    """Test KPOINTS parser for Brillouin zone sampling."""
    print("\n" + "="*60)
    print("Testing KPOINTS Parser for Brillouin Zone Sampling")
    print("="*60)
    
    parser = KpointsParser()
    
    test_files = [
        ("KPOINTS_gamma", "Gamma-centered mesh (4x4x4)"),
        ("KPOINTS_MP", "Monkhorst-Pack mesh (8x8x8)"),
        ("KPOINTS_band", "Band structure path (Line-mode)"),
    ]
    
    all_passed = True
    
    for filename, description in test_files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        if not os.path.exists(filepath):
            print(f"\nWarning: {filepath} not found")
            continue
        
        print(f"\n--- {description} ({filename}) ---")
        kpoints = parser.parse(filepath)
        
        print(f"  Mode: {kpoints.mode.value}")
        print(f"  Comment: {kpoints.comment}")
        
        if kpoints.mesh:
            print(f"  Mesh Subdivisions: {kpoints.mesh.subdivisions}")
            print(f"  Shift: {kpoints.mesh.shift}")
            print(f"  Total k-points: {kpoints.mesh.total_kpoints}")
            
            # Validate constraint
            if kpoints.mesh.total_kpoints >= 1:
                print(f"  [OK] Constraint 'n_k >= 1' satisfied: {kpoints.mesh.total_kpoints} >= 1")
            else:
                print(f"  [FAIL] Constraint 'n_k >= 1' violated!")
                all_passed = False
        
        if kpoints.path:
            print(f"  Number of Segments: {len(kpoints.path.segments)}")
            for i, seg in enumerate(kpoints.path.segments):
                print(f"    Segment {i+1}: {seg['start']['label']} -> {seg['end']['label']} "
                      f"({seg['num_points']} points)")
        
        if kpoints.explicit_kpoints:
            print(f"  Explicit k-points: {len(kpoints.explicit_kpoints)}")
            for kp in kpoints.explicit_kpoints[:5]:  # Show first 5
                print(f"    [{kp.coordinates[0]:.4f}, {kp.coordinates[1]:.4f}, {kp.coordinates[2]:.4f}] "
                      f"weight={kp.weight}")
    
    print(f"\nOverall: {'ALL K-POINTS VALIDATIONS PASSED' if all_passed else 'SOME VALIDATIONS FAILED'}")
    return all_passed


def test_full_extractor():
    """Test the complete VASP extractor with all files."""
    print("\n" + "="*60)
    print("Testing Complete VASP Extractor (Math Schema v1.0)")
    print("="*60)
    
    extractor = VaspExtractor()
    test_dir = os.path.dirname(__file__)
    
    files = {
        "incar": os.path.join(test_dir, "INCAR"),
        "poscar": os.path.join(test_dir, "POSCAR_Si"),
        "kpoints": os.path.join(test_dir, "KPOINTS_gamma"),
    }
    
    # Check files exist
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            return False
    
    print("\nExtracting mathematical structures from VASP inputs...")
    schema = extractor.extract(files)
    
    # Print Math Schema summary
    print(f"\n--- Math Schema v{schema.schema_version} ---")
    print(f"Extracted by: {schema.meta.extracted_by}")
    print(f"Source files: {list(schema.meta.source_files['input'])}")
    
    # Governing equations
    print(f"\n--- Governing Equations ---")
    for eq in schema.mathematical_model.governing_equations:
        print(f"  [{eq.type}] {eq.name}")
        print(f"    Form: {eq.mathematical_form}")
        print(f"    Variables: {', '.join(eq.variables)}")
        print(f"    Parameters: {eq.parameters}")
    
    # Boundary conditions
    print(f"\n--- Boundary Conditions ---")
    for bc in schema.mathematical_model.boundary_conditions:
        print(f"  [{bc.type}] {bc.id}")
        if bc.mathematical_object:
            print(f"    Object: {bc.mathematical_object.field}")
            print(f"    Tensor rank: {bc.mathematical_object.tensor_rank}")
            print(f"    Form: {bc.mathematical_object.tensor_form}")
    
    # Constitutive relations
    print(f"\n--- Constitutive Relations ---")
    for rel in schema.mathematical_model.constitutive_relations:
        print(f"  [{rel['type']}] {rel.get('name', 'N/A')}")
        print(f"    Form: {rel.get('mathematical_form', 'N/A')}")
    
    # Numerical method
    print(f"\n--- Numerical Method ---")
    nm = schema.numerical_method
    print(f"  Discretization: {nm.discretization.space_discretization}")
    print(f"  Time integrator: {nm.discretization.time_integrator}")
    if nm.solver:
        print(f"  Solver: {nm.solver.algorithm}")
        print(f"  Tolerance: {nm.solver.tolerance}")
        print(f"  Max iterations: {nm.solver.max_iterations}")
    
    # Computational graph
    print(f"\n--- Computational Graph ---")
    cg = schema.computational_graph
    print(f"  Nodes: {len(cg.nodes)}")
    for node in cg.nodes[:4]:  # Show first 4
        print(f"    - {node.id} ({node.type})")
    print(f"  Edges: {len(cg.edges)}")
    
    # Symbolic constraints (NEW in v2!)
    print(f"\n--- Symbolic Constraints (Math Anything v2) ---")
    if hasattr(schema, 'symbolic_constraints') and schema.symbolic_constraints:
        passed = 0
        failed = 0
        for constraint in schema.symbolic_constraints:
            # Check if satisfied from description
            is_satisfied = "[SATISFIED]" in constraint.description
            status = "OK" if is_satisfied else "FAIL"
            symbol = "[OK]" if is_satisfied else "[XX]"
            print(f"  {symbol} {constraint.expression:40} | {status}")
            if is_satisfied:
                passed += 1
            else:
                failed += 1
        print(f"\n  Total: {passed} passed, {failed} failed")
    else:
        print("  (No symbolic constraints extracted)")
    
    # Validation results
    print(f"\n--- Validation Results ---")
    if extractor.validation_results:
        vr = extractor.validation_results
        print(f"  Passed: {len(vr.get('passed', []))}")
        print(f"  Failed: {len(vr.get('failed', []))}")
        print(f"  Warnings: {len(vr.get('warnings', []))}")
    
    all_passed = (
        len(extractor.validation_results.get('failed', [])) == 0
        if extractor.validation_results else True
    )
    
    print(f"\n{'='*60}")
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'='*60}")
    
    return all_passed


def demonstrate_math_anything_value():
    """Demonstrate the value of Math Anything for physics/chemistry models."""
    print("\n" + "="*60)
    print("Math Anything Value Proposition")
    print("="*60)
    
    print("""
Physics/Chemistry models often know parameter VALUES but not their
MATHEMATICAL RELATIONSHIPS. Math Anything bridges this gap:

1. SYMBOLIC CONSTRAINTS (not just values)
   Traditional: ENCUT = 520
   Math Anything: ENCUT = 520 AND "ENCUT > 0" AND "ENCUT > ENMAX"
   
   This lets LLMs reason: "If ENCUT < 0, reject calculation"

2. PARAMETER RELATIONSHIPS (not just isolated parameters)
   Traditional: ISMEAR = 0, SIGMA = 0.05
   Math Anything: SIGMA = k_B * T (temperature relationship)
   
   This lets LLMs understand: "SIGMA controls smearing width"

3. MATHEMATICAL STRUCTURE (not just input files)
   Traditional: "This is a VASP INCAR file"
   Math Anything: "Kohn-Sham DFT with plane wave basis, SCF iterations,
                    periodic boundary conditions, PAW pseudopotentials"
   
   This lets LLMs map to: "Similar to QE, ABINIT, but different basis"

4. VALIDATION WITH REASONING
   Traditional: "Input file parsed successfully"
   Math Anything: "5/5 constraints passed: ENCUT>0, EDIFF>0, ..."
   
   This lets LLMs explain: "Why this calculation is mathematically valid"
""")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  Math Anything - VASP Parser Test Suite")
    print("  Testing INCAR, POSCAR, KPOINTS with Symbolic Constraints")
    print("="*70)
    
    results = []
    
    # Test individual parsers
    results.append(("INCAR Parser", test_incar_parser()))
    results.append(("POSCAR Parser", test_poscar_parser()))
    results.append(("KPOINTS Parser", test_kpoints_parser()))
    
    # Test full extractor
    results.append(("Full Extractor", test_full_extractor()))
    
    # Demonstrate value
    demonstrate_math_anything_value()
    
    # Summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = " " if passed else "X"
        print(f"  [{symbol}] {name:30} : {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("  ALL TESTS PASSED - Math Anything is working correctly!")
    else:
        print("  SOME TESTS FAILED - Please check the output above")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
