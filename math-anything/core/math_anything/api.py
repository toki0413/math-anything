"""Unified Math Anything API - Simple interface for end users.

This module provides a high-level API that makes Math Anything easy to use:
- Automatic file parsing
- Unified interface across all engines
- Built-in visualization
- Error handling with helpful messages

Example:
    >>> from math_anything import MathAnything
    >>> ma = MathAnything()
    >>> result = ma.extract("vasp", "path/to/INCAR")
    >>> print(result.mathematical_structure.canonical_form)
    H[n]ψ = εψ
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .schemas import (
    extract_abaqus_mathematical_precision,
    extract_ansys_mathematical_precision,
    extract_comsol_mathematical_precision,
    extract_gromacs_mathematical_precision,
    extract_lammps_mathematical_precision,
    extract_multiwfn_mathematical_precision,
    extract_vasp_mathematical_precision,
)
from .schemas.precision import EnhancedMathSchema

# Engine registry
ENGINE_EXTRACTORS = {
    "vasp": extract_vasp_mathematical_precision,
    "lammps": extract_lammps_mathematical_precision,
    "abaqus": extract_abaqus_mathematical_precision,
    "ansys": extract_ansys_mathematical_precision,
    "comsol": extract_comsol_mathematical_precision,
    "gromacs": extract_gromacs_mathematical_precision,
    "multiwfn": extract_multiwfn_mathematical_precision,
}

# File parsers for each engine
ENGINE_FILE_PARSERS = {
    "vasp": {
        "incar": "math_anything.vasp.core.incar_parser.parse_incar",
    },
}


class MathAnythingError(Exception):
    """Base exception for Math Anything errors."""

    pass


class UnsupportedEngineError(MathAnythingError):
    """Raised when an unsupported engine is specified."""

    pass


class FileNotFoundError(MathAnythingError):
    """Raised when input file is not found."""

    pass


class ParseError(MathAnythingError):
    """Raised when file parsing fails."""

    pass


@dataclass
class ExtractionResult:
    """Result of mathematical structure extraction.

    Attributes:
        engine: Name of the computational engine
        files: Dictionary of parsed input files
        schema: Enhanced mathematical schema
        success: Whether extraction was successful
        errors: List of error messages (if any)
        warnings: List of warning messages
    """

    engine: str
    files: Dict[str, Any]
    schema: Dict[str, Any]
    success: bool
    errors: List[str]
    warnings: List[str]

    def __getattr__(self, name: str) -> Any:
        """Allow accessing schema keys as attributes."""
        if name in self.schema:
            return self.schema[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def visualize(self, format: str = "mermaid") -> str:
        """Generate visualization of mathematical structure.

        Args:
            format: Output format ("mermaid", "graphviz", "text")

        Returns:
            Visualization string
        """
        from .visualization import Visualizer

        viz = Visualizer()
        return viz.render(self.schema, format=format)

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram."""
        return self.visualize(format="mermaid")

    def to_graphviz(self) -> str:
        """Generate Graphviz DOT format."""
        return self.visualize(format="graphviz")

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = []
        lines.append(f"Engine: {self.engine.upper()}")
        lines.append(f"Status: {'✓ Success' if self.success else '✗ Failed'}")

        if not self.success:
            lines.append(f"Errors: {', '.join(self.errors)}")
            return "\n".join(lines)

        # Mathematical structure
        math_struct = self.schema.get("mathematical_structure", {})
        if math_struct:
            lines.append(f"\nProblem Type: {math_struct.get('problem_type', 'N/A')}")
            lines.append(f"Canonical Form: {math_struct.get('canonical_form', 'N/A')}")

        # Approximations
        approxs = self.schema.get("approximations", [])
        if approxs:
            lines.append(f"\nApproximations Applied ({len(approxs)}):")
            for i, app in enumerate(approxs[:5], 1):
                lines.append(f"  {i}. {app.get('name', 'Unknown')}")
            if len(approxs) > 5:
                lines.append(f"  ... and {len(approxs) - 5} more")

        # Constraints
        constraints = self.schema.get("mathematical_decoding", {}).get(
            "constraints", []
        )
        if constraints:
            satisfied = sum(1 for c in constraints if c.get("satisfied"))
            lines.append(f"\nConstraints: {satisfied}/{len(constraints)} satisfied")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings[:3]:
                lines.append(f"  ! {w}")

        return "\n".join(lines)


class MathAnything:
    """Unified interface for Math Anything.

    This class provides a simple, high-level API for extracting mathematical
    structures from computational materials science engines.

    Example:
        >>> ma = MathAnything()
        >>>
        >>> # Extract from VASP
        >>> result = ma.extract("vasp", {"incar": "INCAR", "poscar": "POSCAR"})
        >>> print(result.schema["mathematical_structure"]["canonical_form"])

        >>> # Or use convenience method
        >>> result = ma.extract_file("vasp", "INCAR")
        >>> print(result.to_mermaid())  # Generate visualization
    """

    def __init__(self):
        """Initialize Math Anything."""
        self._engines = list(ENGINE_EXTRACTORS.keys())
        self._warnings: List[str] = []

    @property
    def supported_engines(self) -> List[str]:
        """List of supported computational engines."""
        return self._engines.copy()

    def extract(
        self, engine: str, params: Dict[str, Any], validate: bool = True
    ) -> ExtractionResult:
        """Extract mathematical structure from parameters.

        Args:
            engine: Engine name (vasp, lammps, abaqus, etc.)
            params: Dictionary of engine parameters
            validate: Whether to validate constraints

        Returns:
            ExtractionResult containing mathematical schema

        Raises:
            UnsupportedEngineError: If engine is not supported
            MathAnythingError: If extraction fails

        Example:
            >>> ma = MathAnything()
            >>> result = ma.extract("vasp", {
            ...     "ENCUT": 520,
            ...     "SIGMA": 0.05,
            ...     "EDIFF": 1e-6
            ... })
        """
        engine = engine.lower()

        # Check engine support
        if engine not in ENGINE_EXTRACTORS:
            available = ", ".join(self._engines)
            raise UnsupportedEngineError(
                f"Engine '{engine}' not supported. " f"Available: {available}"
            )

        # Extract
        self._warnings = []
        try:
            extractor = ENGINE_EXTRACTORS[engine]
            schema = extractor(params)

            return ExtractionResult(
                engine=engine,
                files={"params": params},
                schema=schema,
                success=True,
                errors=[],
                warnings=self._warnings,
            )

        except Exception as e:
            return ExtractionResult(
                engine=engine,
                files={"params": params},
                schema={},
                success=False,
                errors=[str(e)],
                warnings=self._warnings,
            )

    def extract_file(
        self, engine: str, filepath: Union[str, Path, Dict[str, str]], **kwargs
    ) -> ExtractionResult:
        """Extract from file(s).

        Automatically parses input files and extracts mathematical structure.

        Args:
            engine: Engine name
            filepath: Path to input file, or dict mapping file types to paths
            **kwargs: Additional options

        Returns:
            ExtractionResult

        Example:
            >>> ma = MathAnything()
            >>> result = ma.extract_file("vasp", "INCAR")
            >>> result = ma.extract_file("vasp", {
            ...     "incar": "INCAR",
            ...     "poscar": "POSCAR"
            ... })
        """
        engine = engine.lower()

        # Handle single file path
        if isinstance(filepath, (str, Path)):
            filepath = Path(filepath)

            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            # Auto-detect file type from extension or name
            file_type = self._detect_file_type(filepath, engine)
            files = {file_type: str(filepath)}
        else:
            # Dict of file paths
            files = {}
            for ftype, fpath in filepath.items():
                p = Path(fpath)
                if not p.exists():
                    raise FileNotFoundError(f"File not found: {fpath}")
                files[ftype] = str(fpath)

        # Parse files based on engine
        params = self._parse_files(engine, files)

        # Extract mathematical structure
        return self.extract(engine, params, **kwargs)

    def _detect_file_type(self, filepath: Path, engine: str) -> str:
        """Detect file type from path."""
        name = filepath.name.upper()

        if engine == "vasp":
            if "INCAR" in name:
                return "incar"
            elif "POSCAR" in name or "CONTCAR" in name:
                return "poscar"
            elif "KPOINTS" in name:
                return "kpoints"
            elif "POTCAR" in name:
                return "potcar"

        # Default to extension
        return filepath.suffix.lstrip(".") or "input"

    def _parse_files(self, engine: str, files: Dict[str, str]) -> Dict[str, Any]:
        """Parse input files to extract parameters."""
        params = {}

        if engine == "vasp":
            # Parse INCAR
            if "incar" in files:
                try:
                    from .vasp.core.incar_parser import parse_incar

                    incar_result = parse_incar(files["incar"])
                    params.update(
                        {
                            name: param.value
                            for name, param in incar_result.parameters.items()
                        }
                    )
                except Exception as e:
                    self._warnings.append(f"Failed to parse INCAR: {e}")

            # Try to load other files
            if "poscar" in files:
                params["_has_poscar"] = True
            if "kpoints" in files:
                params["_has_kpoints"] = True

        # For other engines, use file presence as parameters
        for ftype, fpath in files.items():
            if ftype not in params:
                params[f"_{ftype}_path"] = fpath

        return params

    def compare(
        self,
        result1: ExtractionResult,
        result2: ExtractionResult,
        critical_only: bool = False,
    ) -> Dict[str, Any]:
        """Compare two mathematical structures.

        Args:
            result1: First extraction result
            result2: Second extraction result
            critical_only: Only show critical changes

        Returns:
            Comparison report
        """
        from .utils.math_diff import MathDiffer

        differ = MathDiffer()
        return differ.compare(
            result1.schema, result2.schema, critical_only=critical_only
        )

    def visualize(
        self,
        result: ExtractionResult,
        format: str = "mermaid",
        output: Optional[str] = None,
    ) -> str:
        """Generate visualization.

        Args:
            result: Extraction result to visualize
            format: Output format ("mermaid", "graphviz", "html")
            output: Optional file path to save

        Returns:
            Visualization string
        """
        viz = result.visualize(format=format)

        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(viz)

        return viz

    def discover(
        self, X: Any, y: Any, variable_names: Optional[List[str]] = None, **kwargs
    ) -> str:
        """Discover mathematical equation from data using EML symbolic regression.

        Based on Andrzej Odrzywołek's paper "All elementary functions from a
        single binary operator" (arXiv:2603.21852).

        Args:
            X: Input data (numpy array or list), shape (n_samples, n_features)
            y: Target values (numpy array or list), shape (n_samples,)
            variable_names: Names of variables (default: x0, x1, ...)
            **kwargs: Additional options for SymbolicRegression

        Returns:
            Discovered equation in standard mathematical notation

        Example:
            >>> ma = MathAnything()
            >>>
            >>> # Discover damped oscillation: y = exp(-x) * cos(2*x)
            >>> x = np.linspace(0, 10, 100)
            >>> y = np.exp(-x) * np.cos(2*x)
            >>> equation = ma.discover(x.reshape(-1, 1), y, ['x'])
            >>> print(equation)
            "exp(-x) * cos(2*x)"

            >>> # From simulation output
            >>> data = load_simulation_output("vasp_output.dat")
            >>> equation = ma.discover(data[:, :-1], data[:, -1], ['t', 'x'])
        """
        import numpy as np

        from .eml_v2 import ImprovedSymbolicRegression

        # Convert inputs to numpy arrays
        X_arr = np.array(X)
        y_arr = np.array(y)

        # Run improved symbolic regression
        sr = ImprovedSymbolicRegression(**kwargs)
        best_tree = sr.fit(X_arr, y_arr, variable_names)

        # Convert to standard form
        if best_tree is None:
            return "No equation found"
        return best_tree.to_standard_form()

    def translate(
        self, result: ExtractionResult
    ) -> "MathematicalPropositions":
        """Translate extracted schema into LLM-solvable mathematical propositions.

        This is the core Translate functionality of Math Anything. It converts
        structured mathematical models into formal propositions and proof tasks
        that LLMs can reason about symbolically.

        Args:
            result: Extraction result from extract() or extract_file()

        Returns:
            MathematicalPropositions containing proof tasks, validation tasks,
            consistency checks, and error analysis

        Example:
            >>> ma = MathAnything()
            >>> result = ma.extract_file("lammps", "equil.lmp")
            >>> props = ma.translate(result)
            >>>
            >>> # Access proof tasks
            >>> for task in props.proof_tasks:
            ...     print(task.llm_prompt)
            >>>
            >>> # Access all tasks
            >>> all_tasks = props.all_tasks()
            >>> print(f"Generated {len(all_tasks)} mathematical tasks")
        """
        from .proposition import PropositionGenerator

        generator = PropositionGenerator()
        return generator.translate(result.schema)

    def translate_comparison(
        self, result1: ExtractionResult, result2: ExtractionResult
    ) -> "MathematicalPropositions":
        """Translate two schemas into cross-model comparison propositions.

        Generates mathematical tasks for proving relationships between
        different models (e.g., atomistic vs continuum).

        Args:
            result1: First extraction result
            result2: Second extraction result

        Returns:
            MathematicalPropositions with comparison tasks

        Example:
            >>> ma = MathAnything()
            >>> lmp = ma.extract_file("lammps", "equil.lmp")
            >>> abq = ma.extract_file("abaqus", "job.inp")
            >>>
            >>> props = ma.translate_comparison(lmp, abq)
            >>> for task in props.comparison_tasks:
            ...     print(task.statement)
        """
        from .proposition import PropositionGenerator

        generator = PropositionGenerator()
        return generator.translate_comparison(
            result1.schema, result2.schema
        )


# Convenience function for quick extraction
def extract(engine: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Quick extraction function.

    Example:
        >>> result = extract("vasp", {"ENCUT": 520, "SIGMA": 0.05})
        >>> print(result["mathematical_structure"]["canonical_form"])
    """
    ma = MathAnything()
    result = ma.extract(engine, params)
    return result.schema


def extract_file(engine: str, filepath: Union[str, Path]) -> Dict[str, Any]:
    """Quick file extraction function.

    Example:
        >>> result = extract_file("vasp", "INCAR")
        >>> print(result["mathematical_structure"]["canonical_form"])
    """
    ma = MathAnything()
    result = ma.extract_file(engine, filepath)
    return result.schema
