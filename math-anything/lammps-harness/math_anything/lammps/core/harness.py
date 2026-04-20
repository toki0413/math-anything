"""LAMMPS Harness - Math Anything implementation for LAMMPS.

This module provides the LAMMPS-specific implementation of the MathAnythingHarness
interface, extracting mathematical structures from LAMMPS input files.
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from math_anything.core.harness import MathAnythingHarness, HarnessRegistry
from math_anything.schemas import MathSchema

from .extractor import LammpsExtractor


class LammpsHarness(MathAnythingHarness):
    """LAMMPS harness for Math Anything.
    
    Extracts mathematical structures from LAMMPS molecular dynamics simulations,
    including governing equations (Newton's laws), boundary conditions,
    numerical methods, and computational graphs.
    
    Example:
        ```python
        harness = LammpsHarness()
        schema = harness.extract({
            "input": "in.deform",
            "log": "log.lammps"
        })
        
        # Save to JSON
        schema.save("model.json")
        ```
    """
    
    @property
    def engine_name(self) -> str:
        """Engine identifier."""
        return "lammps"
    
    @property
    def supported_schema_version(self) -> str:
        """Supported Schema version."""
        return "1.0.0"
    
    def extract(self, files: dict, options: dict = None) -> MathSchema:
        """Extract mathematical structures from LAMMPS files.
        
        Args:
            files: Dictionary mapping file types to paths.
                   Required: 'input' - LAMMPS input script
                   Optional: 'log' - LAMMPS log file
            options: Optional extraction parameters.
                    - include_raw_symbols: Include raw symbol table (default: True)
                    - include_thermo_data: Include thermodynamic data from log (default: True)
        
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
        extractor = LammpsExtractor()
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
            "initial_conditions",
            "constitutive_relations",
            "numerical_method",
            "computational_graph",
            "conservation_properties",
            "raw_symbols",
        ]
    
    def get_supported_extensions(self) -> list:
        """Get supported file extensions.
        
        Returns:
            List of file extensions.
        """
        return [".in", ".lammps", ".input"]


# Register harness
HarnessRegistry.register(LammpsHarness)