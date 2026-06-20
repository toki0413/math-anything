"""Ansys Harness for Math Anything.

Extracts mathematical structures from Ansys FEA simulations.
Supports structural, thermal, modal, harmonic, transient, and buckling analyses.
"""

from .core.extractor import AnsysExtractor
from .core.parser import APDLParser, CDBParser, RSTParser

__version__ = "3.0.0"

__all__ = [
    "AnsysExtractor",
    "APDLParser",
    "CDBParser",
    "RSTParser",
]
