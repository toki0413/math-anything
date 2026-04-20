"""Multiwfn Harness Test Suite."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'math-anything', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'math-anything', 'multiwfn-harness'))

import importlib.util
spec = importlib.util.spec_from_file_location("enhanced_parser",
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'math-anything', 'multiwfn-harness',
                 'math_anything', 'multiwfn', 'core', 'enhanced_parser.py'))
module = importlib.util.module_from_spec(spec)
sys.modules['enhanced_parser'] = module
spec.loader.exec_module(module)

EnhancedMultiwfnParser = module.EnhancedMultiwfnParser
MultiwfnSymbolicConstraints = module.MultiwfnSymbolicConstraints
MultiwfnAnalysisType = module.MultiwfnAnalysisType


def test_input_parsing():
    print("=" * 60)
    print("Testing Multiwfn Input Parser")
    print("=" * 60)
    
    test_input = """! H2O wavefunction analysis
2
1
H2O.wfn
5
1
1
2
0
0"""
    
    parser = EnhancedMultiwfnParser()
    result = parser.parse(test_input)
    
    print(f"\n  Commands parsed: {len(result.commands)}")
    print(f"  Analysis types: {[a.value for a in result.analysis_types]}")
    
    for cmd in result.commands:
        print(f"    Function {cmd.function_number}: {cmd.sub_functions}")
    
    assert len(result.commands) > 0, "Should parse commands"
    print("\n  [OK] Input parsing test passed")


def test_schema_extraction():
    print("\n" + "=" * 60)
    print("Testing MathSchema Extraction")
    print("=" * 60)
    
    test_input = """! Full QTAIM analysis
2
1
molecule.wfn
5
1
1
2
0
0
12
1"""
    
    parser = EnhancedMultiwfnParser()
    result = parser.parse(test_input)
    schema = parser._build_schema(result)
    
    print(f"\n  Schema version: {schema.schema_version}")
    print(f"  Governing equations: {len(schema.mathematical_model.governing_equations)}")
    for eq in schema.mathematical_model.governing_equations:
        print(f"    [{eq.type}] {eq.name}")
        print(f"      Form: {eq.mathematical_form}")
    
    print(f"\n  Symbolic constraints: {len(schema.symbolic_constraints)}")
    for c in schema.symbolic_constraints:
        print(f"    {c.expression}")
    
    assert len(schema.mathematical_model.governing_equations) > 0, "Should have equations"
    assert len(schema.symbolic_constraints) > 0, "Should have constraints"
    print("\n  [OK] Schema extraction test passed")


def test_constraint_validation():
    print("\n" + "=" * 60)
    print("Testing Symbolic Constraint Validation")
    print("=" * 60)
    
    validations = MultiwfnSymbolicConstraints.validate_parameter(
        "occupation", 1.5, category="orbital"
    )
    print(f"\n  Validating occupation = 1.5:")
    for v in validations:
        status = "OK" if v["satisfied"] else "FAIL"
        print(f"    [{status}] {v['constraint']}")
    
    validations = MultiwfnSymbolicConstraints.validate_parameter(
        "homo_lumo_gap", 3.5, category="orbital"
    )
    print(f"\n  Validating HOMO-LUMO gap = 3.5 eV:")
    for v in validations:
        status = "OK" if v["satisfied"] else "FAIL"
        print(f"    [{status}] {v['constraint']}")
    
    print("\n  [OK] Constraint validation test passed")


def main():
    print("\n" + "=" * 70)
    print("  Math Anything - Multiwfn Harness Test Suite")
    print("=" * 70)
    
    try:
        test_input_parsing()
        test_schema_extraction()
        test_constraint_validation()
        
        print("\n" + "=" * 70)
        print("  ALL TESTS PASSED!")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
