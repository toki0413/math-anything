"""GROMACS Harness for Math Anything.

Extracts mathematical structures from GROMACS biomolecular simulations.
Supports MD simulations of proteins, lipids, nucleic acids, and more.
"""

from .core.extractor import GromacsExtractor
from .core.extractor_enhanced import EnhancedGromacsExtractor
from .core.harness import GromacsHarness
from .core.parser import EDTParser, MDPParser, TOPParser, TPRParser

__version__ = "1.0.0"

__all__ = [
    "GromacsHarness",
    "GromacsExtractor",
    "EnhancedGromacsExtractor",
    "MDPParser",
    "TOPParser",
    "TPRParser",
    "EDTParser",
]
