"""Math Anything Gaussian Harness.

Extracts mathematical structures from Gaussian quantum chemistry calculations,
including Hartree-Fock, DFT, MP2, CCSD(T), geometry optimization, frequency
analysis, and spectroscopic predictions.
"""

from .core.extractor import GaussianExtractor

__all__ = ["GaussianExtractor"]
