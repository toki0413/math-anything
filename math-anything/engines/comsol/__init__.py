"""COMSOL Multiphysics Harness for Math Anything.

Extracts mathematical structures from COMSOL multiphysics simulations.
Supports physics-based modeling and equation-based modeling.
"""

from .core.extractor import ComsolExtractor
from .core.parser import ComsolJavaParser

__version__ = "1.0.0"

__all__ = [
    "ComsolExtractor",
    "ComsolJavaParser",
]
