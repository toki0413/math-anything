"""GAMESS Harness for Math Anything.

Extracts mathematical structures from GAMESS quantum chemistry simulations.
"""

from .core.extractor import GAMESSExtractor

__all__ = ["GAMESSExtractor"]
