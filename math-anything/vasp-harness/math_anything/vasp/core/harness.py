"""VASP Harness - Math Anything implementation for VASP.

This module provides the VASP-specific implementation of the MathAnythingHarness
interface, extracting mathematical structures from VASP DFT calculations.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from math_anything.core.harness import HarnessRegistry, MathAnythingHarness
from math_anything.schemas import MathSchema

from .extractor import VaspExtractor


class VaspHarness(MathAnythingHarness):
    """VASP harness for Math Anything.

    Extracts mathematical structures from VASP density functional theory
    calculations, including:
    - Kohn-Sham equations (DFT governing equations)
    - Plane wave basis representation
    - Self-consistent field iterations
    - Periodic boundary conditions (Bloch theorem)
    - Electronic structure (eigenvalue problem)

    Example:
        ```python
        harness = VaspHarness()
        schema = harness.extract({
            "incar": "INCAR",
            "poscar": "POSCAR",
            "kpoints": "KPOINTS",
            "outcar": "OUTCAR"
        })

        # Save to JSON
        schema.save("dft_model.json")
        ```
    """

    @property
    def engine_name(self) -> str:
        """Engine identifier."""
        return "vasp"

    @property
    def supported_schema_version(self) -> str:
        """Supported Schema version."""
        return "1.0.0"

    def extract(self, files: dict, options: dict = None) -> MathSchema:
        """Extract mathematical structures from VASP files.

        Args:
            files: Dictionary with keys:
                   - 'incar': Path to INCAR file
                   - 'poscar': Path to POSCAR/CONTCAR file
                   - 'kpoints': Path to KPOINTS file
                   - 'outcar': Path to OUTCAR file
            options: Optional extraction parameters.

        Returns:
            MathSchema object with extracted mathematical structures.

        Raises:
            FileNotFoundError: If required files are missing.
            ValueError: If files cannot be parsed.
        """
        options = options or {}

        # Validate files
        self.validate_files(files)

        # Extract
        extractor = VaspExtractor()
        schema = extractor.extract(files, options)

        return schema

    def list_extractable_objects(self) -> list:
        """List types of mathematical objects this harness can extract.

        Returns:
            List of extractable object types.
        """
        return [
            "governing_equations",
            "boundary_conditions",
            "numerical_method",
            "computational_graph",
            "conservation_properties",
            "raw_symbols",
        ]

    def get_supported_extensions(self) -> list:
        """Get supported file extensions."""
        return [".INCAR", "POSCAR", "CONTCAR", "KPOINTS", "OUTCAR"]


# Register harness
HarnessRegistry.register(VaspHarness)
