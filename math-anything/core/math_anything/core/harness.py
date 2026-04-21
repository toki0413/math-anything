"""Harness interface for Math Anything.

Each computational engine (LAMMPS, VASP, Abaqus, etc.) implements this interface
to extract mathematical structures from their input/output files.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..schemas import MathSchema


class MathAnythingHarness(ABC):
    """Abstract base class for Math Anything harnesses.

    Each harness is an independent Python package that extracts mathematical
    structures from a specific computational engine.

    Example:
        ```python
        class LammpsHarness(MathAnythingHarness):
            @property
            def engine_name(self) -> str:
                return "lammps"

            def extract(self, files, options=None) -> Dict[str, Any]:
                # Parse LAMMPS input files
                # Return Math Schema compliant dictionary
                pass
        ```
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Engine identifier, e.g., 'lammps', 'vasp', 'abaqus'."""
        pass

    @property
    @abstractmethod
    def supported_schema_version(self) -> str:
        """Supported Schema version, e.g., '1.0.0'."""
        pass

    @abstractmethod
    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> MathSchema:
        """Extract mathematical structures from input files.

        Args:
            files: Dictionary mapping file types to paths.
                   Example: {"input": "in.file", "log": "log.lammps"}
            options: Optional extraction parameters.
                    Example: {"include_raw_symbols": True}

        Returns:
            MathSchema object compliant with schema version.

        Raises:
            FileNotFoundError: If required files are missing.
            ValueError: If files cannot be parsed.
        """
        pass

    @abstractmethod
    def list_extractable_objects(self) -> List[str]:
        """List the types of mathematical objects this harness can extract.

        Returns:
            List of object type strings, e.g., ["governing_equations",
            "boundary_conditions", "computational_graph"].
        """
        pass

    def validate_files(self, files: Dict[str, str]) -> bool:
        """Validate that required files exist.

        Args:
            files: Dictionary of file types to paths.

        Returns:
            True if all files exist.

        Raises:
            FileNotFoundError: If any file does not exist.
        """
        for file_type, path in files.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"{file_type} file not found: {path}")
        return True

    def get_file_content(self, path: str) -> str:
        """Read file content as string.

        Args:
            path: Path to file.

        Returns:
            File content as string.
        """
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions.

        Returns:
            List of file extensions, e.g., [".in", ".lammps"].
        """
        return []


class HarnessRegistry:
    """Registry for managing Math Anything harnesses.

    This registry allows dynamic discovery and loading of harnesses.
    """

    _harnesses: Dict[str, Type[MathAnythingHarness]] = {}

    @classmethod
    def register(cls, harness_class: Type[MathAnythingHarness]):
        """Register a harness class.

        Args:
            harness_class: Harness class to register.

        Example:
            ```python
            @HarnessRegistry.register
            class LammpsHarness(MathAnythingHarness):
                ...
            ```
        """
        instance = harness_class()
        cls._harnesses[instance.engine_name] = harness_class
        return harness_class

    @classmethod
    def get(cls, engine_name: str) -> Optional[Type[MathAnythingHarness]]:
        """Get a harness class by engine name.

        Args:
            engine_name: Name of the engine.

        Returns:
            Harness class or None if not found.
        """
        return cls._harnesses.get(engine_name)

    @classmethod
    def create(cls, engine_name: str) -> Optional[MathAnythingHarness]:
        """Create a harness instance by engine name.

        Args:
            engine_name: Name of the engine.

        Returns:
            Harness instance or None if not found.
        """
        harness_class = cls._harnesses.get(engine_name)
        if harness_class:
            return harness_class()
        return None

    @classmethod
    def list_engines(cls) -> List[str]:
        """List all registered engine names.

        Returns:
            List of engine names.
        """
        return list(cls._harnesses.keys())

    @classmethod
    def list_harnesses(cls) -> Dict[str, Type[MathAnythingHarness]]:
        """Get all registered harness classes.

        Returns:
            Dictionary mapping engine names to harness classes.
        """
        return cls._harnesses.copy()
