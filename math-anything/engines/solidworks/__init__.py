"""SolidWorks Simulation Harness for Math Anything.

Extracts mathematical structures from SolidWorks Simulation FEA.
Supports structural, thermal, and modal analyses integrated with CAD.
"""

from .core.extractor import SolidWorksExtractor
from .core.harness import SolidWorksHarness
from .core.parser import CWRParser, MaterialParser, StudyParser

__version__ = "1.0.0"

__all__ = [
    "SolidWorksHarness",
    "SolidWorksExtractor",
    "CWRParser",
    "StudyParser",
    "MaterialParser",
]
