"""Core extraction engine for Math Anything.

This module provides the main extraction logic that coordinates harnesses
and produces Math Schema compliant output.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas import MathSchema, SchemaValidator
from .harness import HarnessRegistry, MathAnythingHarness


class ExtractorEngine:
    """Main extraction engine for Math Anything.

    This engine coordinates the extraction process:
    1. Loads appropriate harness for target engine
    2. Validates input files
    3. Calls harness.extract() to get mathematical structures
    4. Validates output against Math Schema
    5. Serializes to JSON/YAML

    Example:
        ```python
        engine = ExtractorEngine()
        result = engine.extract("lammps", {
            "input": "in.deform",
            "log": "log.lammps"
        })
        engine.save(result, "model.json")
        ```
    """

    def __init__(self):
        self.validator = SchemaValidator()
        self._current_harness: Optional[MathAnythingHarness] = None

    def extract(
        self,
        engine_name: str,
        files: Dict[str, str],
        options: Optional[Dict[str, Any]] = None,
    ) -> MathSchema:
        """Extract mathematical model from files.

        Args:
            engine_name: Name of the computational engine (e.g., 'lammps').
            files: Dictionary mapping file types to paths.
            options: Optional extraction parameters.

        Returns:
            MathSchema object containing extracted mathematical structures.

        Raises:
            ValueError: If engine not found or extraction fails.
            FileNotFoundError: If input files not found.
        """
        # Load harness
        harness = HarnessRegistry.create(engine_name)
        if harness is None:
            available = HarnessRegistry.list_engines()
            raise ValueError(
                f"Engine '{engine_name}' not found. " f"Available engines: {available}"
            )

        self._current_harness = harness

        # Validate files exist
        harness.validate_files(files)

        # Extract
        schema = harness.extract(files, options or {})

        # Validate output
        is_valid = self.validator.validate(schema.to_dict())
        if not is_valid:
            errors = "\n".join(self.validator.errors)
            raise ValueError(f"Schema validation failed:\n{errors}")

        return schema

    def extract_and_save(
        self,
        engine_name: str,
        files: Dict[str, str],
        output_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Extract and save to file.

        Args:
            engine_name: Name of the computational engine.
            files: Dictionary mapping file types to paths.
            output_path: Path to save JSON output.
            options: Optional extraction parameters.

        Returns:
            Path to saved file.
        """
        schema = self.extract(engine_name, files, options)
        schema.save(output_path)
        return output_path

    def validate_output(self, schema: MathSchema) -> bool:
        """Validate extracted schema.

        Args:
            schema: MathSchema to validate.

        Returns:
            True if valid, False otherwise.
        """
        return self.validator.validate(schema.to_dict())

    def get_validation_errors(self) -> List[str]:
        """Get validation errors from last validation.

        Returns:
            List of error messages.
        """
        return self.validator.errors.copy()

    def get_validation_warnings(self) -> List[str]:
        """Get validation warnings from last validation.

        Returns:
            List of warning messages.
        """
        return self.validator.warnings.copy()

    @property
    def current_harness(self) -> Optional[MathAnythingHarness]:
        """Get currently loaded harness."""
        return self._current_harness

    def list_available_engines(self) -> List[str]:
        """List all available engines.

        Returns:
            List of engine names.
        """
        return HarnessRegistry.list_engines()
