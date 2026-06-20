"""Parsers for GROMACS file formats."""

import re
import struct
from typing import Any, Dict, List, Optional


class MDPParser:
    """Parser for GROMACS MDP (Molecular Dynamics Parameters) files."""

    def __init__(self):
        self.parameters: Dict[str, Any] = {}

    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse MDP file.

        Args:
            filepath: Path to .mdp file

        Returns:
            Dictionary with MD parameters
        """
        self.parameters = {}

        with open(filepath, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith(";") or line.startswith("*"):
                continue

            # Parse parameter = value
            if "=" in line:
                parts = line.split("=", 1)
                key = parts[0].strip().lower()
                value = parts[1].strip()

                # Try to convert to number
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string

                self.parameters[key] = value

        return self.parameters

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse MDP file."""
        return self.parse(filepath)


class TOPParser:
    """Parser for GROMACS TOP topology files."""

    def __init__(self):
        self.data: Dict[str, Any] = {}

    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse TOP topology file.

        Args:
            filepath: Path to .top file

        Returns:
            Dictionary with topology data
        """
        self.data = {
            "title": "",
            "forcefield": "",
            "num_atoms": 0,
            "num_molecules": 0,
            "molecule_types": [],
            "atoms": [],
            "bonds": [],
            "angles": [],
            "dihedrals": [],
        }

        with open(filepath, "r") as f:
            lines = f.readlines()

        current_section = None

        for line in lines:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith(";") or line.startswith("*"):
                continue

            # Section headers
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip().lower()
                continue

            # Title
            if current_section == "defaults":
                if line.startswith("#include"):
                    self.data["forcefield"] = line.split()[-1].strip('"')

            # Molecule type
            if current_section == "moleculetype":
                parts = line.split()
                if parts:
                    self.data["molecule_types"].append(parts[0])

            # Atoms
            if current_section == "atoms":
                parts = line.split()
                if len(parts) >= 7:
                    self.data["atoms"].append(
                        {
                            "nr": int(parts[0]),
                            "type": parts[1],
                            "resnr": int(parts[2]),
                            "residue": parts[3],
                            "atom": parts[4],
                            "cgnr": int(parts[5]),
                            "charge": float(parts[6]),
                            "mass": float(parts[7]) if len(parts) > 7 else None,
                        }
                    )
                    self.data["num_atoms"] += 1

            # Bonds
            if current_section == "bonds":
                parts = line.split()
                if len(parts) >= 3:
                    self.data["bonds"].append(
                        {
                            "i": int(parts[0]),
                            "j": int(parts[1]),
                            "func": int(parts[2]),
                        }
                    )

            # Angles
            if current_section == "angles":
                parts = line.split()
                if len(parts) >= 4:
                    self.data["angles"].append(
                        {
                            "i": int(parts[0]),
                            "j": int(parts[1]),
                            "k": int(parts[2]),
                            "func": int(parts[3]),
                        }
                    )

            # Dihedrals
            if current_section == "dihedrals":
                parts = line.split()
                if len(parts) >= 5:
                    self.data["dihedrals"].append(
                        {
                            "i": int(parts[0]),
                            "j": int(parts[1]),
                            "k": int(parts[2]),
                            "l": int(parts[3]),
                            "func": int(parts[4]),
                        }
                    )

            # System
            if current_section == "system":
                self.data["title"] = line

            # Molecules count
            if current_section == "molecules":
                parts = line.split()
                if len(parts) >= 2:
                    self.data["num_molecules"] += int(parts[1])

        return self.data

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse TOP file."""
        return self.parse(filepath)


class TPRParser:
    """Basic parser for GROMACS TPR (run input) files.

    TPR files are binary and require detailed format knowledge.
    This parser extracts basic metadata.
    """

    def __init__(self):
        self.data: Dict[str, Any] = {}

    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse TPR file header.

        Args:
            filepath: Path to .tpr file

        Returns:
            Dictionary with TPR metadata
        """
        self.data = {
            "file_path": filepath,
            "version": "",
            "num_atoms": 0,
            "num_steps": 0,
            "timestep": 0.0,
        }

        try:
            with open(filepath, "rb") as f:
                # Read header
                header = f.read(256)

                # Try to detect version (simplified)
                if header[:4] == b"VERS":
                    self.data["version"] = "detected"

                # More detailed parsing would require knowing the exact format
                # which changes between GROMACS versions

        except Exception as e:
            self.data["error"] = str(e)

        return self.data

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse TPR file."""
        return self.parse(filepath)


class EDTParser:
    """Parser for GROMACS EDR (energy) files.

    EDR files contain energy and other data from simulations.
    """

    def __init__(self):
        self.data: Dict[str, Any] = {}

    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse EDR file metadata.

        Args:
            filepath: Path to .edr file

        Returns:
            Dictionary with energy file metadata
        """
        self.data = {
            "file_path": filepath,
            "num_frames": 0,
            "energy_terms": [],
            "time_range": (0.0, 0.0),
        }

        try:
            with open(filepath, "rb") as f:
                # Read header
                header = f.read(128)

                # Try to detect magic number
                if len(header) >= 4:
                    magic = struct.unpack("i", header[:4])[0]
                    self.data["magic"] = magic

        except Exception as e:
            self.data["error"] = str(e)

        return self.data

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse EDR file."""
        return self.parse(filepath)
