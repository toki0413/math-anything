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

__version__ = "1.0.0"

# New simplified API
from .api import (
    ExtractionResult,
    FileNotFoundError,
    MathAnything,
    MathAnythingError,
    ParseError,
    UnsupportedEngineError,
    extract,
    extract_file,
)
from .core.extractor import ExtractorEngine

# Legacy API (for backward compatibility)
from .core.harness import HarnessRegistry, MathAnythingHarness
from .core.session import ExtractionSession

# EML Symbolic Regression
from .eml_v2 import (
    ExprBuilder,
    ImprovedSymbolicRegression,
    Node,
    NodeType,
    SymbolicRegression,
    discover_equation,
    eml,
)

# Multi-variable Discovery
from .multivar import MultiVariableDiscovery, analyze_interactions, discover_multivar
from .schemas import (
    BoundaryCondition,
    ComputationalGraph,
    GoverningEquation,
    MathematicalModel,
    MathSchema,
    NumericalMethod,
    SchemaValidator,
)

# Expression Simplification
from .simplifier import ExpressionSimplifier, simplify

# Visualization
from .visualization import Visualizer, save_html, to_graphviz, to_mermaid


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
        raise ValueError(
            f"Engine '{engine_name}' not found. " f"Available engines: {available}"
        )
    return harness


def list_engines() -> list:
    """List all available engines.

    Returns:
        List of engine names.
    """
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
    "FileNotFoundError",
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
]
