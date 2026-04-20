"""
Ansys APDL Parser Test Suite for Math Anything

Tests enhanced APDL parser with symbolic constraint validation.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'math-anything', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'math-anything', 'ansys-harness'))

# Import directly from file to avoid package import issues
import importlib.util
spec = importlib.util.spec_from_file_location("apdl_parser", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'math-anything', 'ansys-harness', 
                 'math_anything', 'ansys', 'core', 'apdl_parser.py'))
apdl_module = importlib.util.module_from_spec(spec)
sys.modules['apdl_parser'] = apdl_module
spec.loader.exec_module(apdl_module)

EnhancedAPDLParser = apdl_module.EnhancedAPDLParser
parse_apdl = apdl_module.parse_apdl


def test_beam_bending():
    """Test beam bending analysis parsing."""
    print("=" * 60)
    print("Testing Beam Bending APDL Parser")
    print("=" * 60)
    
    filepath = os.path.join(os.path.dirname(__file__), "beam_bending.inp")
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return False
    
    parser = EnhancedAPDLParser()
    result = parser.parse_file(filepath)
    
    print(f"\n--- APDL File Summary ---")
    print(f"  Total commands: {len(result.commands)}")
    print(f"  Parameters: {len(result.parameters)}")
    print(f"  Materials: {len(result.materials)}")
    print(f"  Analysis type: {result.analysis_type.value}")
    
    print(f"\n--- Parameters ---")
    for name, value in sorted(result.parameters.items()):
        print(f"  {name:12} = {value}")
    
    print(f"\n--- Material Properties ---")
    for mat in result.materials:
        constraints = ", ".join(mat.constraints) if mat.constraints else "None"
        print(f"  {mat.name:10} (mat {mat.material_id}) = {mat.value:.2e} [{mat.unit}]")
        print(f"    Constraints: {constraints}")
    
    print(f"\n--- Symbolic Constraint Validation ---")
    passed = 0
    failed = 0
    for c in result.constraints:
        status = "OK" if c["satisfied"] else "FAIL"
        symbol = "[OK]" if c["satisfied"] else "[XX]"
        print(f"  {symbol} {c['property']:8} = {c['value']:12.2e} | {c['constraint']:25} -> {status}")
        if c["satisfied"]:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Summary: {passed} passed, {failed} failed")
    
    # Extract FEM mathematics
    print(f"\n--- FEM Mathematical Structures ---")
    fem_math = parser.extract_fem_mathematics()
    
    print(f"  Governing Equations:")
    for eq in fem_math["governing_equations"]:
        print(f"    [{eq['type']}] {eq['form']}")
        print(f"      Description: {eq['description']}")
    
    print(f"\n  Discretization:")
    disc = fem_math["discretization"]
    print(f"    Method: {disc['method']}")
    print(f"    Element size: {disc['element_size']}")
    print(f"    Formulation: {disc['formulation']}")
    
    print(f"\n  Material Models:")
    for model in fem_math["material_models"]:
        print(f"    Material {model['material_id']}: {model['type']}")
        if 'youngs_modulus' in model:
            print(f"      E = {model['youngs_modulus']:.2e}, nu = {model['poissons_ratio']}")
            print(f"      Lamé parameters: λ = {model['lame_first']:.2e}, μ = {model['shear_modulus']:.2e}")
            print(f"      σ = λtr(ε)I + 2με")
    
    print(f"\n  Boundary Conditions: {len(fem_math['boundary_conditions'])}")
    for bc in fem_math["boundary_conditions"][:5]:  # Show first 5
        print(f"    {bc['type']:15} on {bc['node']} (DOF: {bc['dof']}) = {bc['value']}")
    
    return failed == 0


def test_thermal_analysis():
    """Test thermal analysis parsing."""
    print("\n" + "=" * 60)
    print("Testing Thermal Analysis APDL Parser")
    print("=" * 60)
    
    filepath = os.path.join(os.path.dirname(__file__), "thermal_steady.inp")
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return False
    
    parser = EnhancedAPDLParser()
    result = parser.parse_file(filepath)
    
    print(f"\n--- APDL File Summary ---")
    print(f"  Total commands: {len(result.commands)}")
    print(f"  Analysis type: {result.analysis_type.value}")
    
    print(f"\n--- Thermal Material Properties ---")
    for mat in result.materials:
        print(f"  {mat.name:8} = {mat.value:.2e} [{mat.unit}]")
    
    print(f"\n--- Symbolic Constraint Validation ---")
    for c in result.constraints:
        status = "OK" if c["satisfied"] else "FAIL"
        symbol = "[OK]" if c["satisfied"] else "[XX]"
        print(f"  {symbol} {c['property']:8} = {c['value']:12.2e} | {c['constraint']}")
    
    # Extract FEM mathematics
    fem_math = parser.extract_fem_mathematics()
    
    print(f"\n--- Governing Equation ---")
    for eq in fem_math["governing_equations"]:
        print(f"  {eq['form']}")
        print(f"  Description: {eq['description']}")
    
    return True


def demonstrate_ansys_value():
    """Demonstrate Math Anything value for FEM."""
    print("\n" + "=" * 60)
    print("Math Anything Value for Ansys/FEM")
    print("=" * 60)
    
    print("""
Traditional APDL parsing extracts:
  - "MP, EX, 1, 2.1e11" -> EX = 2.1e11

Math Anything extracts:
  - EX = 2.1e11 AND "EX > 0" AND "EX < 1e15"
  - Stress-strain relation: σ = λtr(ε)I + 2με
  - Lamé parameters: λ = 1.21e11, μ = 8.08e10
  - Equation: ∇·σ + f = 0 (Static equilibrium)
  - Constraint: det(K) > 0 (Stiffness matrix positive definite)

This transforms geometric parameters into MATHEMATICAL SEMANTICS:
  - Element size → Discretization error bound
  - Material properties → Constitutive relations
  - Boundary conditions → Mathematical constraints
  - Solver settings → Convergence criteria
""")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  Math Anything - Ansys APDL Parser Test Suite")
    print("=" * 70)
    
    results = []
    
    results.append(("Beam Bending", test_beam_bending()))
    results.append(("Thermal Analysis", test_thermal_analysis()))
    
    demonstrate_ansys_value()
    
    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = " " if passed else "X"
        print(f"  [{symbol}] {name:30} : {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("  ALL TESTS PASSED - Ansys APDL parser working correctly!")
    else:
        print("  SOME TESTS FAILED - Please check the output above")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
