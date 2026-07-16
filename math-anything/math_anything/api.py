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

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from .exceptions import (
    ExtractionFileNotFoundError,
    MathAnythingError,
    UnsupportedEngineError,
)
from .schemas import (
    extract_abaqus_mathematical_precision,
    extract_ansys_mathematical_precision,
    extract_comsol_mathematical_precision,
    extract_gromacs_mathematical_precision,
    extract_lammps_mathematical_precision,
    extract_multiwfn_mathematical_precision,
    extract_vasp_mathematical_precision,
)
from .utils.llm_cache import SemanticCache

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
    files: Dict[str, object]
    schema: Dict[str, object]
    success: bool
    errors: List[str]
    warnings: List[str]

    def __getattr__(self, name: str) -> object:
        """Allow accessing schema keys as attributes."""
        if name in self.schema:
            return self.schema[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
            lines.append(f"\nProblem Type: {math_struct.get('problem_type', 'N/A')}")  # type: ignore[attr-defined]
            lines.append(f"Canonical Form: {math_struct.get('canonical_form', 'N/A')}")  # type: ignore[attr-defined]

        # Approximations
        approxs = self.schema.get("approximations", [])
        if approxs:
            lines.append(f"\nApproximations Applied ({len(approxs)}):")  # type: ignore[arg-type]
            for i, app in enumerate(approxs[:5], 1):  # type: ignore[index]
                lines.append(f"  {i}. {app.get('name', 'Unknown')}")
            if len(approxs) > 5:  # type: ignore[arg-type]
                lines.append(f"  ... and {len(approxs) - 5} more")  # type: ignore[arg-type]

        # Constraints
        constraints = self.schema.get("mathematical_decoding", {}).get("constraints", [])  # type: ignore[attr-defined]
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
        >>>
        >>> # Or use convenience method
        >>> result = ma.extract_file("vasp", "INCAR")
        >>> print(result.to_mermaid())  # Generate visualization
    """

    def __init__(self):
        """Initialize Math Anything."""
        self._engines = list(ENGINE_EXTRACTORS.keys())
        self._warnings: List[str] = []
        # internal metrics tracking
        self._extraction_count: int = 0
        self._verification_count: int = 0
        self._total_extraction_time_ms: float = 0.0
        self._start_time: float = time.monotonic()
        # semantic cache for LLM calls
        self._cache = SemanticCache()

    @property
    def supported_engines(self) -> List[str]:
        """List of supported computational engines."""
        return self._engines.copy()  # type: ignore[no-any-return]

    @staticmethod
    def _validate_and_coerce_params(params: object) -> Dict[str, object]:
        """Validate params type and coerce common patterns into extractor-expected format.

        The extractors expect a flat dict where:
        - Scalar params (ENCUT, EDIFF, etc.) stay as-is
        - kpoints should be a dict like {"grid": [4,4,4]}
        - lattice should be a dict like {"vectors": [[...],[...],[...]]}

        Users often pass bare lists for kpoints/lattice, which causes
        'list' object has no attribute 'get' crashes deep in the extractors.
        This method catches those cases and wraps them properly.
        """
        # Type check: params must be a dict
        if not isinstance(params, dict):
            type_name = type(params).__name__
            if isinstance(params, (list, tuple)):
                raise MathAnythingError(
                    f"Expected dict for params, got {type_name}. Use {{'encut': 520}} instead of {params!r}"
                )
            raise MathAnythingError(
                f"Expected dict for params, got {type_name}. Example: {{'ENCUT': 520, 'EDIFF': 1e-6}}"
            )

        coerced = {}
        for key, value in params.items():
            key_lower = key.lower()

            # kpoints: bare list -> {"grid": list}
            if key_lower == "kpoints" and isinstance(value, (list, tuple)):
                coerced[key] = {"grid": list(value)}
            # lattice: nested list -> {"vectors": list}
            elif key_lower == "lattice" and isinstance(value, (list, tuple)):
                coerced[key] = {"vectors": [list(row) if isinstance(row, (list, tuple)) else row for row in value]}
            # Any other non-dict, non-scalar value that isn't already handled
            elif isinstance(value, (list, tuple)) and key_lower not in ("kpoints", "lattice"):
                # Keep as-is for other keys — extractors may handle them differently
                coerced[key] = value  # type: ignore[assignment]
            else:
                coerced[key] = value

        return coerced  # type: ignore[return-value]

    def extract(self, engine: str, params: Dict[str, object], validate: bool = True) -> ExtractionResult:
        """Extract mathematical structure from parameters.

        Args:
            engine: Engine name (vasp, lammps, abaqus, etc.)
            params: Dictionary of engine parameters. Scalar params like
                ENCUT/EDIFF go directly; structured params like kpoints
                and lattice can be passed as bare lists and will be
                auto-wrapped into the format the extractor expects.
            validate: Whether to validate constraints

        Returns:
            ExtractionResult containing mathematical schema

        Raises:
            UnsupportedEngineError: If engine is not supported
            MathAnythingError: If params is not a dict or has invalid types

        Example:
            >>> ma = MathAnything()
            >>> result = ma.extract("vasp", {
            ...     "ENCUT": 520,
            ...     "SIGMA": 0.05,
            ...     "EDIFF": 1e-6,
            ...     "kpoints": [4, 4, 4],
            ...     "lattice": [[5.43,0,0],[0,5.43,0],[0,0,5.43]]
            ... })
        """
        engine = engine.lower()

        # Check engine support
        if engine not in ENGINE_EXTRACTORS:
            available = ", ".join(self._engines)
            raise UnsupportedEngineError(f"Engine '{engine}' not supported. Available: {available}")

        # Validate and coerce params
        params = self._validate_and_coerce_params(params)

        # Check semantic cache
        cache_key = f"{engine}:{sorted(params.items())}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

        # Extract
        self._warnings = []
        t0 = time.perf_counter()
        try:
            extractor = ENGINE_EXTRACTORS[engine]
            schema = extractor(params)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._extraction_count += 1
            self._total_extraction_time_ms += elapsed_ms

            result = ExtractionResult(
                engine=engine,
                files={"params": params},
                schema=schema,
                success=True,
                errors=[],
                warnings=self._warnings,
            )
            self._cache.set(cache_key, result)
            return result

        except (KeyError, ValueError, TypeError, ImportError) as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._extraction_count += 1
            self._total_extraction_time_ms += elapsed_ms

            return ExtractionResult(
                engine=engine,
                files={"params": params},
                schema={},
                success=False,
                errors=[str(e)],
                warnings=self._warnings,
            )

    def extract_file(self, engine: str, filepath: Union[str, Path, Dict[str, str]], **kwargs) -> ExtractionResult:
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

        # Validate filepath type
        if not isinstance(filepath, (str, Path, dict)):
            type_name = type(filepath).__name__
            if isinstance(filepath, (list, tuple)):
                raise MathAnythingError(
                    f"Expected str, Path, or dict for filepath, got {type_name}. "
                    f"Use a file path string or {{'incar': 'INCAR'}} instead of {filepath!r}"
                )
            raise MathAnythingError(
                f"Expected str, Path, or dict for filepath, got {type_name}. Example: ma.extract_file('vasp', 'INCAR')"
            )

        # Handle single file path
        if isinstance(filepath, (str, Path)):
            filepath = Path(filepath)

            if not filepath.exists():
                raise ExtractionFileNotFoundError(f"File not found: {filepath}")

            # Auto-detect file type from extension or name
            file_type = self._detect_file_type(filepath, engine)
            files = {file_type: str(filepath)}
        else:
            # Dict of file paths
            files = {}
            for ftype, fpath in filepath.items():
                p = Path(fpath)
                if not p.exists():
                    raise ExtractionFileNotFoundError(f"File not found: {fpath}")
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

    def _parse_files(self, engine: str, files: Dict[str, str]) -> Dict[str, object]:
        """Parse input files to extract parameters."""
        params = {}

        if engine == "vasp":
            # Parse INCAR
            if "incar" in files:
                try:
                    from vasp.core.incar_parser import parse_incar

                    incar_result = parse_incar(files["incar"])
                    params.update({name: param.value for name, param in incar_result.parameters.items()})
                except (KeyError, ValueError, TypeError, ImportError) as e:
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
    ) -> Dict[str, object]:
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
        report = differ.compare(result1.schema, result2.schema)
        if critical_only:
            return report.critical_changes  # type: ignore[return-value]
        return report  # type: ignore[return-value]

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

    def discover(self, X: object, y: object, variable_names: Optional[List[str]] = None, **kwargs) -> str:
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
        """
        import numpy as np

        from .eml_v2 import ImprovedSymbolicRegression

        # Convert inputs to numpy arrays
        X_arr = np.array(X)
        y_arr = np.array(y)

        # Run improved symbolic regression
        sr = ImprovedSymbolicRegression(**kwargs)
        try:
            best_tree = sr.fit(X_arr, y_arr, variable_names)
        except (AttributeError, ValueError, TypeError, ZeroDivisionError):
            return "No equation found (regression failed)"

        # Convert to standard form
        if best_tree is None:
            return "No equation found"
        try:
            return best_tree.to_standard_form()
        except (AttributeError, ValueError, TypeError):
            return "No equation found (expression invalid)"

    def health_check(self) -> dict:
        """Health check endpoint.

        Returns system status including:
        - status: "healthy" | "degraded" | "unhealthy"
        - version: package version
        - rust_acceleration: bool
        - engines_available: list of engine names
        - uptime_seconds: time since module import
        - python_version: sys.version
        """
        try:
            from .rust_bridge import EMLAccelerator

            rust_available = EMLAccelerator().using_rust
        except Exception:
            rust_available = False

        engines = self._engines
        if not engines:
            status = "unhealthy"
        elif not rust_available:
            status = "degraded"
        else:
            status = "healthy"

        try:
            from importlib.metadata import version as pkg_version

            # PyPI 发布名是 bourbaki；math-anything 是开发期 dist 名。
            # 按优先级尝试，避免 PackageNotFoundError 误判为版本缺失。
            for _dist in ("bourbaki", "math-anything", "math_anything"):
                try:
                    version = pkg_version(_dist)
                    break
                except Exception:
                    continue
            else:
                version = "3.0.0"
        except Exception:
            version = "3.0.0"

        return {
            "status": status,
            "version": version,
            "rust_acceleration": rust_available,
            "engines_available": engines.copy(),
            "uptime_seconds": round(time.monotonic() - self._start_time, 3),
            "python_version": sys.version,
        }

    def get_metrics(self) -> dict:
        """Prometheus-style metrics.

        Returns:
        - total_extractions: count
        - total_verifications: count
        - rust_acceleration_available: bool
        - engines_count: int
        - average_extraction_time_ms: float (tracked internally)
        """
        try:
            from .rust_bridge import EMLAccelerator

            rust_available = EMLAccelerator().using_rust
        except Exception:
            rust_available = False

        avg_time = self._total_extraction_time_ms / self._extraction_count if self._extraction_count > 0 else 0.0

        return {
            "total_extractions": self._extraction_count,
            "total_verifications": self._verification_count,
            "rust_acceleration_available": rust_available,
            "engines_count": len(self._engines),
            "average_extraction_time_ms": round(avg_time, 3),
            "cache_hit_rate": self._cache.hit_rate,
            "cache_size": self._cache.size,
            "cache_total_requests": self._cache.stats["total_requests"],
        }

    def get_prometheus_metrics(self) -> str:
        """Prometheus exposition format metrics.

        Returns:
            String in Prometheus text-based exposition format,
            suitable for scraping by a Prometheus server.
        """
        try:
            from .rust_bridge import EMLAccelerator

            rust_available = EMLAccelerator().using_rust
        except Exception:
            rust_available = False

        avg_time = self._total_extraction_time_ms / self._extraction_count if self._extraction_count > 0 else 0.0

        lines = [
            "# HELP math_anything_extractions_total Total number of extractions",
            "# TYPE math_anything_extractions_total counter",
            f"math_anything_extractions_total {self._extraction_count}",
            "",
            "# HELP math_anything_verifications_total Total number of verifications",
            "# TYPE math_anything_verifications_total counter",
            f"math_anything_verifications_total {self._verification_count}",
            "",
            "# HELP math_anything_rust_acceleration_available Whether Rust acceleration is available",
            "# TYPE math_anything_rust_acceleration_available gauge",
            f"math_anything_rust_acceleration_available {1 if rust_available else 0}",
            "",
            "# HELP math_anything_engines_available Number of available engines",
            "# TYPE math_anything_engines_available gauge",
            f"math_anything_engines_available {len(self._engines)}",
            "",
            "# HELP math_anything_avg_extraction_time_ms Average extraction time in milliseconds",
            "# TYPE math_anything_avg_extraction_time_ms gauge",
            f"math_anything_avg_extraction_time_ms {round(avg_time, 1)}",
        ]
        return "\n".join(lines)

    def verify(
        self, engine: str, params: Dict[str, object], schema: Optional[Dict[str, object]] = None
    ) -> Dict[str, object]:
        """Verify extracted mathematical structures.

        Args:
            engine: Engine name
            params: Engine parameters
            schema: Previously extracted schema to verify

        Returns:
            Verification result dict
        """
        self._verification_count += 1

        time.perf_counter()
        result = self.extract(engine, params)
        if not result.success:
            return {
                "valid": False,
                "violations": [{"name": "extraction_failed", "message": e} for e in result.errors],
                "warnings": result.warnings,
            }

        violations = []
        constraints = result.schema.get("mathematical_decoding", {}).get("constraints", [])  # type: ignore[attr-defined]
        for c in constraints:
            if not c.get("satisfied", True):
                violations.append(
                    {
                        "name": c.get("name", "unknown"),
                        "expression": c.get("expression", ""),
                        "severity": c.get("severity", "heuristic"),
                        "message": c.get("message", "Constraint not satisfied"),
                    }
                )

        is_valid = len(violations) == 0

        return {
            "valid": is_valid,
            "violations": violations,
            "warnings": result.warnings,
        }


# Convenience function for quick extraction
def extract(engine: str, params: Dict[str, object]) -> Dict[str, object]:
    """Quick extraction function.

    Example:
        >>> result = extract("vasp", {"ENCUT": 520, "SIGMA": 0.05})
        >>> print(result["mathematical_structure"]["canonical_form"])
    """
    ma = MathAnything()
    result = ma.extract(engine, params)
    return result.schema


def extract_file(engine: str, filepath: Union[str, Path]) -> Dict[str, object]:
    """Quick file extraction function.

    Example:
        >>> result = extract_file("vasp", "INCAR")
        >>> print(result["mathematical_structure"]["canonical_form"])
    """
    ma = MathAnything()
    result = ma.extract_file(engine, filepath)
    return result.schema
