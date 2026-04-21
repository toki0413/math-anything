"""Ansys Harness for Math Anything.

Extracts mathematical structures from Ansys FEA simulations.
Supports structural, thermal, and coupled-field analyses.
"""

from .core.extractor import AnsysExtractor
from .core.harness import AnsysHarness
from .core.parser import APDLParser, CDBParser, RSTParser

__version__ = "1.0.0"

__all__ = [
    "AnsysHarness",
    "AnsysExtractor",
    "APDLParser",
    "CDBParser",
    "RSTParser",
]
