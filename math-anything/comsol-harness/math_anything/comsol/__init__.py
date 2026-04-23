"""COMSOL Multiphysics Harness for Math Anything.

Extracts mathematical structures from COMSOL multiphysics simulations.
Supports physics-based modeling and equation-based modeling.
"""

from .core.extractor import ComsolExtractor
from .core.extractor_enhanced import EnhancedComsolExtractor
from .core.harness import ComsolHarness
from .core.parser import JavaParser, ModelParser, MPHParser

__version__ = "1.0.0"

__all__ = [
    "ComsolHarness",
    "ComsolExtractor",
    "EnhancedComsolExtractor",
    "MPHParser",
    "ModelParser",
    "JavaParser",
]
