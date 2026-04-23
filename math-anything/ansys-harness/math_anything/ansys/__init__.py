"""Ansys Harness for Math Anything.

Extracts mathematical structures from Ansys FEA simulations.
Supports structural, thermal, and coupled-field analyses.
"""

from .core.extractor import AnsysExtractor
from .core.extractor_enhanced import EnhancedAnsysExtractor
from .core.harness import AnsysHarness
from .core.parser import APDLParser, CDBParser, RSTParser

__version__ = "1.0.0"

__all__ = [
    "AnsysHarness",
    "AnsysExtractor",
    "EnhancedAnsysExtractor",
    "APDLParser",
    "CDBParser",
    "RSTParser",
]
