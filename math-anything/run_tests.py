"""Run all Math Anything Phase 0 tests."""

import sys
import os
import subprocess


def run_test_file(test_file):
    """Run a test file and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print('='*70)
    
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=False,
    )
    
    return result.returncode == 0


def main():
    """Run all tests."""
    print("="*70)
    print("Math Anything Phase 0 - Test Suite")
    print("="*70)
    
    test_files = [
        "core/tests/test_schema.py",
        "core/tests/test_math_diff.py",
        "core/tests/test_extensions.py",
        "core/tests/test_cross_engine.py",
        "lammps-harness/tests/test_fix_deform.py",
    ]
    
    results = []
    
    for test_file in test_files:
        full_path = os.path.join(os.path.dirname(__file__), test_file)
        if os.path.exists(full_path):
            success = run_test_file(full_path)
            results.append((test_file, success))
        else:
            print(f"\nTest file not found: {test_file}")
            results.append((test_file, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_file, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_file}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print()
    print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())