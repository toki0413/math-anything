"""Math Anything - Mathematical structure extraction for computational materials.

Math Anything extracts universal mathematical structures (governing equations,
boundary conditions, numerical methods, computational graphs) from computational
software (VASP, LAMMPS, Abaqus, etc.) and outputs them as LLM-native structured data.

Example:
    ```python
    import math_anything as ma
    
    # Extract from LAMMPS
    harness = ma.load_harness("lammps")
    schema = harness.extract({"input": "in.file"})
    
    # Save as JSON
    schema.save("model.json")
    ```
"""

__version__ = "0.1.0"

from .core.harness import MathAnythingHarness, HarnessRegistry
from .core.extractor import ExtractorEngine
from .core.session import ExtractionSession
from .schemas import (
    MathSchema,
    SchemaValidator,
    MathematicalModel,
    NumericalMethod,
    ComputationalGraph,
    BoundaryCondition,
    GoverningEquation,
)


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
            f"Engine '{engine_name}' not found. "
            f"Available engines: {available}"
        )
    return harness


def list_engines() -> list:
    """List all available engines.
    
    Returns:
        List of engine names.
    """
    return HarnessRegistry.list_engines()


__all__ = [
    "__version__",
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