"""Math Anything CLI Entry Point.

Provides command-line interface for:
- Interactive REPL mode
- One-shot extraction
- Math diff comparison
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from .repl import MathAnythingREPL, MathDiff
    from .schemas import MathSchema
except ImportError:
    from math_anything.repl import MathAnythingREPL, MathDiff
    from math_anything.schemas import MathSchema


def create_parser(prog_name: str = "math-anything") -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description="Math Anything - Mathematical Semantic Layer for Computational Materials Science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Interactive REPL mode
  {prog_name} repl

  # Extract from VASP files
  {prog_name} extract vasp INCAR POSCAR KPOINTS --output schema.json

  # Compare two schemas
  {prog_name} diff schema1.json schema2.json

  # Cross-engine comparison (VASP vs QE)
  {prog_name} cross vasp_INCAR.json quantum_espresso

For more information: https://github.com/toki/math-anything
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # REPL mode
    repl_parser = subparsers.add_parser(
        "repl",
        help="Start interactive REPL mode",
    )
    repl_parser.add_argument(
        "--session",
        type=str,
        help="Load existing session file",
    )

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract mathematical schema from input files",
    )
    extract_parser.add_argument(
        "engine",
        choices=["vasp", "lammps", "ansys", "abaqus", "comsol"],
        help="Computational engine type",
    )
    extract_parser.add_argument(
        "files",
        nargs="+",
        help="Input file(s)",
    )
    extract_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output JSON file",
    )
    extract_parser.add_argument(
        "--format",
        choices=["json", "yaml", "pretty"],
        default="pretty",
        help="Output format",
    )

    # Diff command
    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare two mathematical schemas (math-anything diff)",
    )
    diff_parser.add_argument(
        "file1",
        help="First schema file (JSON)",
    )
    diff_parser.add_argument(
        "file2",
        help="Second schema file (JSON)",
    )
    diff_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output diff result to file",
    )
    diff_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    # Cross-engine comparison
    cross_parser = subparsers.add_parser(
        "cross",
        help="Cross-engine parameter mapping",
    )
    cross_parser.add_argument(
        "schema_file",
        help="Source schema file",
    )
    cross_parser.add_argument(
        "target_engine",
        help="Target engine (e.g., quantum_espresso, gromacs)",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate symbolic constraints",
    )
    validate_parser.add_argument(
        "schema_file",
        help="Schema file to validate",
    )

    # Lean4 verification
    lean4_parser = subparsers.add_parser(
        "lean4",
        help="Lean4/Mathlib formal verification",
    )
    lean4_parser.add_argument(
        "schema_file",
        nargs="?",
        default=None,
        help="Schema file to translate and verify with Lean4",
    )
    lean4_parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if Lean4 is available",
    )

    # Cross-validation matrix
    crossval_parser = subparsers.add_parser(
        "crossval",
        help="Build cross-validation matrix from schema",
    )
    crossval_parser.add_argument(
        "schema_file",
        help="Schema file to build cross-validation matrix",
    )

    # Falsifiable predictions
    predict_parser = subparsers.add_parser(
        "predict",
        help="Build falsifiable prediction table from schema",
    )
    predict_parser.add_argument(
        "schema_file",
        help="Schema file to build prediction table",
    )

    # Version
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    return parser


def cmd_repl(args):
    """Start REPL mode."""
    repl = MathAnythingREPL()

    if args.session:
        try:
            from repl.core import REPLSession

            repl.session = REPLSession.load(args.session)
            print(f"Loaded session: {repl.session.name}")
        except Exception as e:
            print(f"Error loading session: {e}")

    try:
        repl.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")


def cmd_extract(args):
    """Extract schema from files."""
    print(f"Extracting from {args.engine.upper()} files...")

    # Import appropriate extractor
    try:
        if args.engine == "vasp":
            from vasp.core.extractor_v2 import VaspExtractor

            extractor = VaspExtractor()

            file_dict = {}
            for f in args.files:
                f_lower = f.lower()
                if "incar" in f_lower:
                    file_dict["incar"] = f
                elif "poscar" in f_lower:
                    file_dict["poscar"] = f
                elif "kpoints" in f_lower:
                    file_dict["kpoints"] = f

            schema = extractor.extract(file_dict)

        elif args.engine == "lammps":
            from lammps.core.extractor import LammpsExtractor

            extractor = LammpsExtractor()
            schema = extractor.extract({"input": args.files[0]})

        elif args.engine == "ansys":
            from ansys.core.apdl_parser import EnhancedAPDLParser

            parser = EnhancedAPDLParser()
            result = parser.parse_file(args.files[0])

            # Convert to schema (simplified)
            from schemas import MathematicalModel, MathSchema, MetaInfo

            schema = MathSchema(
                schema_version="1.0.0",
                meta=MetaInfo(
                    extracted_by="math-anything-cli",
                    source_files={"input": args.files},
                ),
                mathematical_model=MathematicalModel(),
            )

        else:
            print(f"Engine {args.engine} not yet implemented")
            return 1

        # Output
        if args.output:
            schema.save(args.output)
            print(f"Schema saved to {args.output}")
        else:
            if args.format == "json":
                import json

                print(json.dumps(schema.to_dict(), indent=2))
            elif args.format == "pretty":
                _print_pretty_schema(schema)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_diff(args):
    """Compare two schemas."""
    print(f"Comparing {args.file1} vs {args.file2}...")

    try:
        # Load schemas
        schema1 = MathSchema.load(args.file1)
        schema2 = MathSchema.load(args.file2)

        # Compare
        diff = MathDiff.compare(schema1, schema2)

        # Output
        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(diff.to_dict(), f, indent=2)
            print(f"Diff saved to {args.output}")
        else:
            if args.format == "json":
                import json

                print(json.dumps(diff.to_dict(), indent=2))
            else:
                _print_pretty_diff(diff, args.file1, args.file2)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_cross(args):
    """Cross-engine comparison."""
    print(f"Mapping {args.schema_file} to {args.target_engine}...")

    try:
        schema = MathSchema.load(args.schema_file)
        suggestions = MathDiff.cross_engine_compare(schema, args.target_engine)

        print("\n" + "=" * 60)
        print(f"Cross-Engine Mapping: {args.schema_file} -> {args.target_engine}")
        print("=" * 60)

        if suggestions["mappable_parameters"]:
            print("\n🔄 Mappable Parameters:")
            for m in suggestions["mappable_parameters"]:
                print(f"  {m['source']:15} -> {m['target']:15} ({m['meaning']})")
                print(f"    Value: {m['value']}")

        if suggestions["unmappable_parameters"]:
            print("\n⚠️ Unmapped Parameters (manual attention needed):")
            for p in suggestions["unmappable_parameters"]:
                print(f"  - {p}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_validate(args):
    """Validate schema constraints."""
    print(f"Validating {args.schema_file}...")

    try:
        schema = MathSchema.load(args.schema_file)

        print("\n" + "=" * 60)
        print("Symbolic Constraint Validation")
        print("=" * 60)

        if hasattr(schema, "symbolic_constraints") and schema.symbolic_constraints:
            passed = 0
            failed = 0

            for constraint in schema.symbolic_constraints:
                is_satisfied = "SATISFIED" in constraint.description
                symbol = "✓" if is_satisfied else "✗"

                print(f"  {symbol} {constraint.expression:40}", end="")
                if constraint.description:
                    desc = constraint.description.replace(" [SATISFIED]", "").replace(
                        " [VIOLATED]", ""
                    )
                    print(f" | {desc[:30]}")
                else:
                    print()

                if is_satisfied:
                    passed += 1
                else:
                    failed += 1

            print(f"\n  Total: {passed} passed, {failed} failed")
            return 0 if failed == 0 else 1
        else:
            print("  No symbolic constraints found in schema")
            return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_lean4(args):
    """Lean4/Mathlib formal verification."""
    if args.check:
        from .lean4_bridge import Lean4Bridge

        bridge = Lean4Bridge()
        if bridge.is_available():
            print("✓ Lean4 is available")
            return 0
        else:
            print("✗ Lean4 is not available on this system")
            print("  Install: https://leanprover.github.io/lean4/doc/quickstart.html")
            return 1

    if not args.schema_file:
        print("Error: schema_file is required when not using --check")
        return 1

    print(f"Translating {args.schema_file} to Lean4...")

    try:
        schema = MathSchema.load(args.schema_file)
        from .lean4_bridge import Lean4Bridge

        bridge = Lean4Bridge()
        lean_code = bridge.math_schema_to_lean(schema)

        print("\n" + "=" * 60)
        print("Lean4 Translation")
        print("=" * 60)
        print(lean_code)

        imports = bridge.suggest_mathlib_imports(schema)
        if imports:
            print("\n" + "=" * 60)
            print("Suggested Mathlib Imports")
            print("=" * 60)
            for imp in imports:
                print(f"  import {imp}")

        if bridge.is_available():
            print("\n" + "=" * 60)
            print("Lean4 Verification")
            print("=" * 60)
            result = bridge.verify_statement(lean_code)
            print(f"  Status: {result.status.value}")
            print(f"  Time: {result.proof_time_ms:.0f}ms")
            if result.message:
                print(f"  Message: {result.message[:200]}")
            bridge.close()
        else:
            print("\n  (Lean4 not available - skipping verification)")

        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_crossval(args):
    """Build cross-validation matrix from schema."""
    print(f"Building cross-validation matrix for {args.schema_file}...")

    try:
        schema = MathSchema.load(args.schema_file)
        from .validation_toolkit import create_cross_validation_from_schema

        matrix = create_cross_validation_from_schema(schema)
        print(matrix.report())
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_predict(args):
    """Build falsifiable prediction table from schema."""
    print(f"Building prediction table for {args.schema_file}...")

    try:
        schema = MathSchema.load(args.schema_file)
        from .validation_toolkit import create_prediction_table_from_schema

        table = create_prediction_table_from_schema(schema)
        print(table.report())
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def _print_pretty_schema(schema: MathSchema):
    """Print schema in human-readable format."""
    print("\n" + "=" * 60)
    print("Mathematical Schema")
    print("=" * 60)

    if schema.mathematical_model:
        if schema.mathematical_model.governing_equations:
            print("\n📐 Governing Equations:")
            for eq in schema.mathematical_model.governing_equations:
                print(f"  [{eq.type}] {eq.name}")
                print(f"    {eq.mathematical_form}")

        if schema.mathematical_model.boundary_conditions:
            print(
                f"\n📏 Boundary Conditions: {len(schema.mathematical_model.boundary_conditions)}"
            )

    if hasattr(schema, "symbolic_constraints") and schema.symbolic_constraints:
        print(f"\n✓ Symbolic Constraints: {len(schema.symbolic_constraints)}")


def _print_pretty_diff(diff, file1: str, file2: str):
    """Print diff in human-readable format."""
    print("\n" + "=" * 60)
    print(f"Comparison: {file1} vs {file2}")
    print("=" * 60)

    similarity = diff.similarity_score * 100
    print(f"\nSimilarity: {similarity:.1f}%")

    if diff.common_equations:
        print("\n✓ Common Equations:")
        for eq in diff.common_equations:
            print(f"  - {eq}")

    if diff.unique_to_first:
        print(f"\n📌 Unique to {file1}:")
        for item in diff.unique_to_first:
            print(f"  - {item}")

    if diff.unique_to_second:
        print(f"\n📌 Unique to {file2}:")
        for item in diff.unique_to_second:
            print(f"  - {item}")

    if diff.parameter_mapping:
        print("\n🔄 Parameter Mappings:")
        for m in diff.parameter_mapping[:10]:
            conf = m.get("confidence", "medium")
            print(
                f"  {m['first']} <-> {m['second']} ({m.get('meaning', 'unknown')}) [{conf}]"
            )

    print(f"\n💡 {diff.analysis_summary}")


def main():
    """Main entry point."""
    # Determine program name from how it was invoked
    prog_name = os.path.basename(sys.argv[0]) if sys.argv else "math-anything"
    parser = create_parser(prog_name=prog_name)
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to command handler
    commands = {
        "repl": cmd_repl,
        "extract": cmd_extract,
        "diff": cmd_diff,
        "cross": cmd_cross,
        "validate": cmd_validate,
        "lean4": cmd_lean4,
        "crossval": cmd_crossval,
        "predict": cmd_predict,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
