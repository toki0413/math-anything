"""Math Anything - Mathematical structure extraction for computational materials.

Namespace package configuration for multi-source distribution.
"""

import sys
from pathlib import Path

# Auto-discover engine packages in engines/
_pkg_dir = Path(__file__).parent
_engines_dir = _pkg_dir.parent / "engines"
if _engines_dir.exists() and str(_engines_dir) not in sys.path:
    sys.path.insert(0, str(_engines_dir))

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

"""Math Anything - Mathematical structure extraction for computational materials.

Math Anything extracts universal mathematical structures (governing equations,
boundary conditions, numerical methods, computational graphs) from computational
software (VASP, LAMMPS, Abaqus, etc.) and outputs them as LLM-native structured data.

Quick Start:
    ```python
    from math_anything import MathAnything

    # Simple API
    ma = MathAnything()
    result = ma.extract("vasp", {"ENCUT": 520, "SIGMA": 0.05})
    print(result.schema["mathematical_structure"]["canonical_form"])

    # With file parsing
    result = ma.extract_file("vasp", "INCAR")
    print(result.to_mermaid())  # Visualize as diagram
    ```
"""

__version__ = "3.0.0"

# New simplified API
# ── Legacy API (deprecated) ──
# These exports are maintained for backward compatibility.
# Use MathAnything class instead. Will be removed in v4.0.
import warnings as _warnings

from .api import (
    ExtractionFileNotFoundError,
    ExtractionResult,
    MathAnything,
    MathAnythingError,
    UnsupportedEngineError,
    extract,
    extract_file,
)
from .core.extractor import ExtractorEngine
from .exceptions import ParseError


def _deprecated_class_warning(name):
    """Emit deprecation warning only when the deprecated class is actually used."""
    _warnings.warn(
        f"{name} is deprecated. Use MathAnything class instead. Will be removed in v4.0.",
        DeprecationWarning,
        stacklevel=3,
    )


class _DeprecatedProxy:
    """Lazy proxy that warns on first access of deprecated classes."""

    def __init__(self, module_path, class_name):
        self._module_path = module_path
        self._class_name = class_name
        self._wrapped = None

    def _resolve(self):
        if self._wrapped is None:
            _deprecated_class_warning(self._class_name)
            import importlib

            mod = importlib.import_module(self._module_path)
            self._wrapped = getattr(mod, self._class_name)
        return self._wrapped

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


HarnessRegistry = _DeprecatedProxy(".core.harness", "HarnessRegistry")
MathAnythingHarness = _DeprecatedProxy(".core.harness", "MathAnythingHarness")
ExtractionSession = _DeprecatedProxy(".core.session", "ExtractionSession")

# Schema classes (lightweight, keep eager import)
from .schemas import (
    BoundaryCondition,
    ComputationalGraph,
    GoverningEquation,
    MathematicalModel,
    MathSchema,
    NumericalMethod,
    SchemaValidator,
)

# ── Lazy imports for heavy modules ──
# EML Symbolic Regression, Multi-variable Discovery, Conjugacy, Constants,
# Bridge, Simplifier, and Visualization are heavy — defer until first use.

_lazy_imports = {
    # EML Symbolic Regression
    "NodeType": ".eml_v2",
    "Node": ".eml_v2",
    "ExprBuilder": ".eml_v2",
    "ImprovedSymbolicRegression": ".eml_v2",
    "SymbolicRegression": ".eml_v2",
    "discover_equation": ".eml_v2",
    "eml": ".eml_v2",
    # Multi-variable Discovery
    "MultiVariableDiscovery": ".multivar",
    "analyze_interactions": ".multivar",
    "discover_multivar": ".multivar",
    # EML Topological Conjugacy
    "EMLExpr": ".conjugacy",
    "EMLPrimitive": ".conjugacy",
    "EMLConjugacyEngine": ".conjugacy",
    "TopologicalConjugacy": ".conjugacy",
    "make_exp": ".conjugacy",
    "make_ln": ".conjugacy",
    "eml_expr": ".conjugacy",
    "const_expr": ".conjugacy",
    "CONJUGACIES": ".conjugacy",
    # e-π Constants Theory
    "EMLConstantEngine": ".constants",
    "ConstantDefinition": ".constants",
    "classify_constant": ".constants",
    "find_eml_form": ".constants",
    "list_known_constants": ".constants",
    "KNOWN_CONSTANTS": ".constants",
    # Structure Bridge
    "StructureBridge": ".bridge",
    # Expression Simplification
    "ExpressionSimplifier": ".simplifier",
    "simplify": ".simplifier",
    # Visualization
    "Visualizer": ".visualization",
    "to_mermaid": ".visualization",
    "to_graphviz": ".visualization",
    "save_html": ".visualization",
    # Topology loop engineering
    "Loop": ".topology",
    "LoopType": ".topology",
    "LoopEngine": ".topology",
    "LoopClassifier": ".topology",
}


def __getattr__(name: str):
    """Lazy import for heavy modules — defer until first use."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name], __package__)
        attr = getattr(module, name)
        globals()[name] = attr  # cache for subsequent access
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_harness(engine_name: str) -> MathAnythingHarness:
    """Load a harness by engine name.

    Args:
        engine_name: Name of the engine (e.g., 'lammps', 'vasp').

    Returns:
        Harness instance.

    Raises:
        ValueError: If engine not found.

    Example:
        ```python
        harness = ma.load_harness("lammps")
        schema = harness.extract({"input": "in.file"})
        ```
    """
    harness = HarnessRegistry.create(engine_name)
    if harness is None:
        available = HarnessRegistry.list_engines()
        raise ValueError(f"Engine '{engine_name}' not found. Available engines: {available}")
    return harness


def list_engines() -> list:
    """List all available engines.

    .. deprecated::
        Use ``MathAnything`` class instead. Will be removed in v4.0.

    Returns:
        List of engine names.
    """
    _warnings.warn(
        "list_engines is deprecated. Will be removed in v4.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return HarnessRegistry.list_engines()


__all__ = [
    # Version
    "__version__",
    # New API
    "MathAnything",
    "ExtractionResult",
    "extract",
    "extract_file",
    "MathAnythingError",
    "UnsupportedEngineError",
    "ExtractionFileNotFoundError",
    "ParseError",
    # Visualization
    "Visualizer",
    "to_mermaid",
    "to_graphviz",
    "save_html",
    # EML Symbolic Regression
    "NodeType",
    "Node",
    "ExprBuilder",
    "ImprovedSymbolicRegression",
    "SymbolicRegression",
    "discover_equation",
    "eml",
    # Expression Simplification
    "ExpressionSimplifier",
    "simplify",
    # Multi-variable Discovery
    "MultiVariableDiscovery",
    "discover_multivar",
    "analyze_interactions",
    # Legacy API
    "load_harness",
    "list_engines",
    "MathAnythingHarness",
    "HarnessRegistry",
    "ExtractorEngine",
    "ExtractionSession",
    "MathSchema",
    "SchemaValidator",
    "MathematicalModel",
    "NumericalMethod",
    "ComputationalGraph",
    "BoundaryCondition",
    "GoverningEquation",
    # EML Conjugacy
    "EMLExpr",
    "EMLPrimitive",
    "EMLConjugacyEngine",
    "TopologicalConjugacy",
    "make_exp",
    "make_ln",
    "eml_expr",
    "const_expr",
    "CONJUGACIES",
    # e-π Constants
    "EMLConstantEngine",
    "ConstantDefinition",
    "classify_constant",
    "find_eml_form",
    "list_known_constants",
    "KNOWN_CONSTANTS",
    # Structure Bridge
    "StructureBridge",
    # Topology loop engineering
    "Loop",
    "LoopType",
    "LoopEngine",
    "LoopClassifier",
]
