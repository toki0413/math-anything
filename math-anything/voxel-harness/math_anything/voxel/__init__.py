"""Voxel Harness for Math Anything.

Extracts mathematical structures from voxel-based simulations.
Supports various voxel-based methods including:
- Lattice Boltzmann Method (LBM)
- Finite Difference Time Domain (FDTD)
- Finite Volume Method on Cartesian grids
- Cellular Automata

Mathematical focus:
- Voxel grid as discretization domain
- Scale mapping between index and physical space
- Boundary condition numerical implementations
- Interpolation rules for continuous reconstruction
"""

from .core.harness import VoxelHarness
from .core.extractor import VoxelMathExtractor
from .core.lbm_extractor import LBMBoundaryExtractor

__version__ = "1.0.0"

__all__ = [
    "VoxelHarness",
    "VoxelMathExtractor",
    "LBMBoundaryExtractor",
]
