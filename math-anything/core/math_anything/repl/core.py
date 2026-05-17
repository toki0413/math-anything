"""Math Anything REPL Core Implementation.

Provides an interactive command-line interface for extracting and
manipulating mathematical structures from computational software.
"""

import cmd
import json
import os

try:
    import readline
except ImportError:
    readline = None
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from math_anything.schemas import MathSchema


@dataclass
class LoadedFile:
    """A loaded input file with metadata."""

    path: str
    engine: str  # 'vasp', 'lammps', 'ansys', etc.
    schema: Optional[MathSchema] = None
    loaded_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "engine": self.engine,
            "loaded_at": self.loaded_at.isoformat(),
            "has_schema": self.schema is not None,
        }


@dataclass
class REPLSession:
    """REPL session state."""

    name: str = "default"
    loaded_files: Dict[str, LoadedFile] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    current_engine: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def add_file(
        self, name: str, path: str, engine: str, schema: Optional[MathSchema] = None
    ):
        """Add a loaded file to session."""
        self.loaded_files[name] = LoadedFile(
            path=path,
            engine=engine,
            schema=schema,
        )
        self.current_engine = engine

    def get_file(self, name: str) -> Optional[LoadedFile]:
        """Get a loaded file by name."""
        return self.loaded_files.get(name)

    def list_files(self) -> List[Tuple[str, LoadedFile]]:
        """List all loaded files."""
        return list(self.loaded_files.items())

    def save(self, filepath: str):
        """Save session to file."""
        data = {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "current_engine": self.current_engine,
            "files": {k: v.to_dict() for k, v in self.loaded_files.items()},
            "history": self.history[-100:],  # Keep last 100 commands
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "REPLSession":
        """Load session from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        session = cls(name=data.get("name", "default"))
        session.current_engine = data.get("current_engine")
        session.history = data.get("history", [])
        # Note: schema objects are not restored, only metadata
        return session


class MathAnythingREPL(cmd.Cmd):
    """Unified REPL for Math Anything.

    Provides an interactive command-line interface with:
    - File loading and extraction
    - Mathematical structure inspection
    - Symbolic constraint validation
    - Cross-engine comparison
    - Session management

    Example:
        ```python
        from math_anything.repl import MathAnythingREPL

        repl = MathAnythingREPL()
        repl.cmdloop()  # Start interactive session
        ```
    """

    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }

    def __init__(self):
        super().__init__()
        self.session = REPLSession()
        self.prompt = self._color("cyan", "math-anything> ")
        self.intro = self._banner()
        self._setup_readline()

    def _color(self, color: str, text: str) -> str:
        """Apply color to text."""
        if os.name == "nt" and not os.environ.get("ANSICON"):
            # Windows without ANSI support
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def _banner(self) -> str:
        """Generate welcome banner."""
        banner = f"""
{self._color('cyan', '╔══════════════════════════════════════════════════════════╗')}
{self._color('cyan', '║')}  {self._color('bold', 'Math Anything')} - Mathematical Semantic Layer for Materials  {self._color('cyan', '║')}
{self._color('cyan', '║')}  Type 'help' for commands, 'exit' to quit                 {self._color('cyan', '║')}
{self._color('cyan', '╚══════════════════════════════════════════════════════════╝')}
"""
        return banner

    def _setup_readline(self):
        """Setup readline for command history."""
        if readline is None:
            return

        histfile = Path.home() / ".math_anything_history"
        try:
            readline.read_history_file(str(histfile))
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass

        import atexit

        atexit.register(readline.write_history_file, str(histfile))

    def _print_success(self, message: str):
        """Print success message."""
        print(f"{self._color('green', '✓')} {message}")

    def _print_error(self, message: str):
        """Print error message."""
        print(f"{self._color('red', '✗')} {message}")

    def _print_info(self, message: str):
        """Print info message."""
        print(f"{self._color('blue', 'ℹ')} {message}")

    def _print_warning(self, message: str):
        """Print warning message."""
        print(f"{self._color('yellow', '⚠')} {message}")

    # ==================== Commands ====================

    def do_load(self, arg: str):
        """
        Load input file(s) from a computational engine.

        Usage: load <engine> <file1> [file2] ...

        Examples:
            load vasp INCAR POSCAR KPOINTS
            load lammps simulation.in
            load ansys beam_bending.inp
        """
        args = arg.split()
        if len(args) < 2:
            self._print_error("Usage: load <engine> <file1> [file2] ...")
            return

        engine = args[0].lower()
        files = args[1:]

        # Validate engine
        supported_engines = ["vasp", "lammps", "ansys", "abaqus", "comsol"]
        if engine not in supported_engines:
            self._print_error(f"Unsupported engine: {engine}")
            self._print_info(f"Supported: {', '.join(supported_engines)}")
            return

        # Load files
        self._print_info(f"Loading {len(files)} file(s) from {engine.upper()}...")

        try:
            schema = self._extract_schema(engine, files)
            if schema:
                session_name = f"{engine}_{len(self.session.loaded_files) + 1}"
                self.session.add_file(session_name, " ".join(files), engine, schema)
                self._print_success(f"Loaded as '{session_name}'")
                self._print_summary(schema)
            else:
                self._print_error("Extraction failed")
        except Exception as e:
            self._print_error(f"Error: {str(e)}")

    def do_extract(self, arg: str):
        """
        Extract mathematical structures from loaded files.

        Usage: extract [session_name]

        Examples:
            extract          # Extract from current session
            extract vasp_1   # Extract from specific session
        """
        if not self.session.loaded_files:
            self._print_error("No files loaded. Use 'load' first.")
            return

        if arg:
            file_info = self.session.get_file(arg)
            if not file_info:
                self._print_error(f"Session '{arg}' not found")
                return
            schema = file_info.schema
        else:
            # Use most recent
            name, file_info = list(self.session.loaded_files.items())[-1]
            schema = file_info.schema
            self._print_info(f"Using session: {name}")

        if schema:
            self._print_detailed_schema(schema)
        else:
            self._print_error("No schema available")

    def do_constraints(self, arg: str):
        """
        Validate symbolic constraints.

        Usage: constraints [session_name]

        Examples:
            constraints
            constraints vasp_1
        """
        if not self.session.loaded_files:
            self._print_error("No files loaded. Use 'load' first.")
            return

        name, file_info = self._get_current_or_named(arg)
        if not file_info or not file_info.schema:
            self._print_error("No schema available")
            return

        schema = file_info.schema

        print(f"\n{self._color('bold', 'Symbolic Constraints Validation')}")
        print("=" * 60)

        if hasattr(schema, "symbolic_constraints") and schema.symbolic_constraints:
            passed = 0
            failed = 0

            for constraint in schema.symbolic_constraints:
                is_satisfied = "SATISFIED" in constraint.description
                symbol = "✓" if is_satisfied else "✗"
                color = "green" if is_satisfied else "red"

                print(
                    f"  {self._color(color, symbol)} {constraint.expression:40}", end=""
                )
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

            print(
                f"\n  Total: {self._color('green', str(passed))} passed, {self._color('red', str(failed))} failed"
            )
        else:
            self._print_warning("No symbolic constraints available")

    def do_compare(self, arg: str):
        """
        Compare mathematical structures between sessions.

        Usage: compare <session1> <session2>

        Examples:
            compare vasp_1 vasp_2
            compare lammps_1 ansys_1
        """
        args = arg.split()
        if len(args) != 2:
            self._print_error("Usage: compare <session1> <session2>")
            return

        name1, name2 = args

        file1 = self.session.get_file(name1)
        file2 = self.session.get_file(name2)

        if not file1 or not file1.schema:
            self._print_error(f"Session '{name1}' not found or has no schema")
            return

        if not file2 or not file2.schema:
            self._print_error(f"Session '{name2}' not found or has no schema")
            return

        self._print_info(f"Comparing {name1} vs {name2}...")

        from .diff import MathDiff

        diff = MathDiff.compare(file1.schema, file2.schema)
        self._print_diff_result(diff, name1, name2)

    def do_list(self, arg: str):
        """
        List all loaded sessions.

        Usage: list
        """
        if not self.session.loaded_files:
            self._print_info("No files loaded")
            return

        print(f"\n{self._color('bold', 'Loaded Sessions')}")
        print("=" * 60)

        for name, file_info in self.session.loaded_files.items():
            schema_status = "✓" if file_info.schema else "✗"
            print(f"  {name:20} | {file_info.engine:10} | Schema: {schema_status}")
            print(f"    Path: {file_info.path}")

    def do_save(self, arg: str):
        """
        Save current session to file.

        Usage: save <filepath>

        Examples:
            save my_session.json
        """
        if not arg:
            self._print_error("Usage: save <filepath>")
            return

        try:
            self.session.save(arg)
            self._print_success(f"Session saved to {arg}")
        except Exception as e:
            self._print_error(f"Failed to save: {str(e)}")

    def do_export(self, arg: str):
        """
        Export schema to JSON file.

        Usage: export <session_name> <filepath>

        Examples:
            export vasp_1 output.json
        """
        args = arg.split()
        if len(args) != 2:
            self._print_error("Usage: export <session_name> <filepath>")
            return

        name, filepath = args
        file_info = self.session.get_file(name)

        if not file_info or not file_info.schema:
            self._print_error(f"Session '{name}' not found or has no schema")
            return

        try:
            file_info.schema.save(filepath)
            self._print_success(f"Schema exported to {filepath}")
        except Exception as e:
            self._print_error(f"Failed to export: {str(e)}")

    def do_exit(self, arg: str):
        """
        Exit the REPL.

        Usage: exit
        """
        self._print_info("Goodbye!")
        return True

    def do_quit(self, arg: str):
        """Alias for exit."""
        return self.do_exit(arg)

    def do_EOF(self, arg: str):
        """Handle Ctrl+D."""
        print()
        return self.do_exit(arg)

    def emptyline(self):
        """Do nothing on empty line."""
        pass

    # ==================== Helper Methods ====================

    def _extract_schema(self, engine: str, files: List[str]) -> Optional[MathSchema]:
        """Extract schema using appropriate harness."""
        # Import dynamically to avoid circular imports
        try:
            if engine == "vasp":
                from vasp.core.extractor_v2 import VaspExtractor

                extractor = VaspExtractor()

                file_dict = {}
                for f in files:
                    f_lower = f.lower()
                    if "incar" in f_lower:
                        file_dict["incar"] = f
                    elif "poscar" in f_lower:
                        file_dict["poscar"] = f
                    elif "kpoints" in f_lower:
                        file_dict["kpoints"] = f

                return extractor.extract(file_dict)

            elif engine == "lammps":
                from lammps.core.extractor import LammpsExtractor

                extractor = LammpsExtractor()
                return extractor.extract({"input": files[0]})

            elif engine == "ansys":
                from ansys.core.apdl_parser import EnhancedAPDLParser

                parser = EnhancedAPDLParser()
                result = parser.parse_file(files[0])

                # Convert to schema format
                return self._create_schema_from_apdl(result)

            # Add more engines...
            else:
                self._print_warning(
                    f"Schema extraction for {engine} not yet implemented"
                )
                return None

        except Exception as e:
            self._print_error(f"Extraction error: {str(e)}")
            return None

    def _create_schema_from_apdl(self, result) -> MathSchema:
        """Convert APDL result to MathSchema."""
        # Simplified conversion - full implementation would be more comprehensive
        from schemas import (
            BoundaryCondition,
            GoverningEquation,
            MathematicalModel,
            MathSchema,
            MetaInfo,
        )

        fem_math = result.extract_fem_mathematics()

        equations = []
        for eq in fem_math.get("governing_equations", []):
            equations.append(
                GoverningEquation(
                    id=eq.get("type", "unknown"),
                    type=eq.get("type", "unknown"),
                    name=eq.get("description", "Unknown"),
                    mathematical_form=eq.get("form", ""),
                    variables=eq.get("variables", []),
                )
            )

        model = MathematicalModel(
            governing_equations=equations,
            boundary_conditions=[],
            constitutive_relations=fem_math.get("material_models", []),
        )

        return MathSchema(
            schema_version="1.0.0",
            meta=MetaInfo(
                extracted_by="math-anything-repl",
                source_files={
                    "input": [result.commands[0].raw if result.commands else ""]
                },
            ),
            mathematical_model=model,
        )

    def _get_current_or_named(self, name: str) -> Tuple[str, Optional[LoadedFile]]:
        """Get current or named file info."""
        if name:
            return name, self.session.get_file(name)
        elif self.session.loaded_files:
            name, info = list(self.session.loaded_files.items())[-1]
            return name, info
        return "", None

    def _print_summary(self, schema: MathSchema):
        """Print brief summary of schema."""
        print(f"\n{self._color('bold', 'Extraction Summary')}")
        print("-" * 40)

        if schema.mathematical_model:
            n_eqs = len(schema.mathematical_model.governing_equations)
            n_bcs = len(schema.mathematical_model.boundary_conditions)
            print(f"  Governing Equations: {n_eqs}")
            print(f"  Boundary Conditions: {n_bcs}")

        if hasattr(schema, "symbolic_constraints"):
            n_constraints = len(schema.symbolic_constraints)
            print(f"  Symbolic Constraints: {n_constraints}")

        print()

    def _print_detailed_schema(self, schema: MathSchema):
        """Print detailed schema information."""
        print(f"\n{self._color('bold', 'Mathematical Schema')}")
        print("=" * 60)

        # Governing equations
        if schema.mathematical_model and schema.mathematical_model.governing_equations:
            print(f"\n{self._color('cyan', '📐 Governing Equations')}")
            for eq in schema.mathematical_model.governing_equations:
                print(f"  [{eq.type}] {eq.name}")
                print(f"    Form: {eq.mathematical_form}")

        # Boundary conditions
        if schema.mathematical_model and schema.mathematical_model.boundary_conditions:
            print(f"\n{self._color('cyan', '📏 Boundary Conditions')}")
            for bc in schema.mathematical_model.boundary_conditions:
                print(f"  [{bc.type}] {bc.id}")
                if bc.mathematical_object:
                    print(f"    Object: {bc.mathematical_object.field}")

        # Numerical method
        if schema.numerical_method:
            print(f"\n{self._color('cyan', '🔢 Numerical Method')}")
            nm = schema.numerical_method
            print(f"  Space: {nm.discretization.space_discretization}")
            print(f"  Time: {nm.discretization.time_integrator}")

    def _print_diff_result(self, diff, name1: str, name2: str):
        """Print comparison result."""
        from .diff import DiffResult

        print(f"\n{self._color('bold', f'Comparison: {name1} vs {name2}')}")
        print("=" * 60)

        # Similarity score
        similarity = diff.similarity_score * 100
        color = "green" if similarity > 70 else "yellow" if similarity > 40 else "red"
        print(f"\n  Similarity: {self._color(color, f'{similarity:.1f}%')}")

        # Common aspects
        if diff.common_equations:
            print(f"\n{self._color('green', '✓ Common Governing Equations')}")
            for eq in diff.common_equations:
                print(f"  - {eq}")

        # Differences
        if diff.different_equations:
            print(f"\n{self._color('yellow', '⚠ Different Equations')}")
            for item in diff.different_equations:
                print(f"  - {item}")

        if diff.unique_to_first:
            print(f"\n{self._color('blue', f'📌 Unique to {name1}')}")
            for item in diff.unique_to_first:
                print(f"  - {item}")

        if diff.unique_to_second:
            print(f"\n{self._color('blue', f'📌 Unique to {name2}')}")
            for item in diff.unique_to_second:
                print(f"  - {item}")

        # Constraint comparison
        if diff.constraint_differences:
            print(f"\n{self._color('cyan', '📊 Constraint Differences')}")
            for cd in diff.constraint_differences:
                print(f"  {cd['parameter']}: {cd['first']} vs {cd['second']}")


def main():
    """Entry point for REPL."""
    repl = MathAnythingREPL()
    try:
        repl.cmdloop()
    except KeyboardInterrupt:
        print(f"\n{repl._color('yellow', 'Interrupted')}")
        repl.do_exit("")


if __name__ == "__main__":
    main()
