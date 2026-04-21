"""Parsers for Ansys file formats.

Supports:
- APDL command files (.inp, .mac)
- CDB database files
- RST results files (basic parsing)
"""

import re
import struct
from typing import Any, Dict, List, Optional


class APDLParser:
    """Parser for Ansys Parametric Design Language (APDL) scripts."""

    def __init__(self):
        self.commands: List[Dict[str, Any]] = []
        self.parameters: Dict[str, Any] = {}

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse APDL script content.

        Args:
            content: APDL script content

        Returns:
            Dictionary with parsed commands and parameters
        """
        self.commands = []
        self.parameters = {}

        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("!"):
                continue

            # Remove inline comments
            if "!" in line:
                line = line[: line.index("!")].strip()

            # Check for parameter assignment
            param_match = re.match(r"(\w+)\s*=\s*(.+)", line)
            if param_match:
                self.parameters[param_match.group(1)] = param_match.group(2).strip()
                continue

            # Parse APDL command
            parts = line.split(",")
            if parts:
                cmd = parts[0].strip().upper()
                args = [p.strip() for p in parts[1:]]

                self.commands.append(
                    {
                        "command": cmd,
                        "args": args,
                        "line": line_num,
                        "raw": line,
                    }
                )

        return {
            "commands": self.commands,
            "parameters": self.parameters,
            "num_commands": len(self.commands),
        }

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse APDL file."""
        with open(filepath, "r") as f:
            content = f.read()
        return self.parse(content)

    def extract_analysis_type(self) -> str:
        """Extract analysis type from parsed commands."""
        analysis_types = {
            "ANTYPE,0": "static_structural",
            "ANTYPE,2": "modal",
            "ANTYPE,3": "harmonic",
            "ANTYPE,4": "transient_structural",
            "ANTYPE,8": "buckling",
            "ANTYPE,STATIC": "static_structural",
            "ANTYPE,MODAL": "modal",
        }

        for cmd in self.commands:
            cmd_str = f"{cmd['command']},{','.join(cmd['args'])}"
            for key, atype in analysis_types.items():
                if cmd_str.startswith(key):
                    return atype

        return "unknown"

    def extract_material_properties(self) -> List[Dict[str, Any]]:
        """Extract material property definitions."""
        materials = []

        for cmd in self.commands:
            if cmd["command"] == "MP":
                if len(cmd["args"]) >= 3:
                    materials.append(
                        {
                            "property": cmd["args"][0],
                            "material_num": cmd["args"][1],
                            "value": cmd["args"][2],
                        }
                    )
            elif cmd["command"] == "MPDATA":
                if len(cmd["args"]) >= 3:
                    materials.append(
                        {
                            "property": cmd["args"][0],
                            "material_num": cmd["args"][1],
                            "values": cmd["args"][2:],
                        }
                    )

        return materials


class CDBParser:
    """Parser for Ansys CDB (Code Database) files.

    CDB files contain mesh, material, and boundary condition information.
    """

    def __init__(self):
        self.data: Dict[str, Any] = {}

    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse CDB file.

        Args:
            filepath: Path to .cdb file

        Returns:
            Dictionary with model data
        """
        self.data = {
            "title": "",
            "num_nodes": 0,
            "num_elements": 0,
            "num_element_types": 0,
            "num_materials": 0,
            "nodes": [],
            "elements": [],
            "element_types": {},
            "materials": [],
            "constraints": [],
            "loads": [],
            "coordinate_systems": [],
        }

        with open(filepath, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Title
            if line.startswith("/TITLE"):
                self.data["title"] = line[6:].strip()

            # Nodes
            elif line.startswith("NBLOCK"):
                i = self._parse_nblock(lines, i)

            # Elements
            elif line.startswith("EBLOCK"):
                i = self._parse_eblock(lines, i)

            # Element types
            elif line.startswith("ET"):
                parts = line.split(",")
                if len(parts) >= 3:
                    etype_num = parts[1].strip()
                    etype_name = parts[2].strip()
                    self.data["element_types"][etype_num] = etype_name

            # Material properties
            elif line.startswith("MP"):
                parts = line.split(",")
                if len(parts) >= 4:
                    self.data["materials"].append(
                        {
                            "property": parts[1].strip(),
                            "material_num": parts[2].strip(),
                            "value": parts[3].strip(),
                        }
                    )

            # Constraints (D command)
            elif line.startswith("D,"):
                parts = line[2:].split(",")
                if len(parts) >= 3:
                    self.data["constraints"].append(
                        {
                            "node": parts[0].strip(),
                            "dof": parts[1].strip(),
                            "value": parts[2].strip() if len(parts) > 2 else "0",
                        }
                    )

            # Loads (F command for forces)
            elif line.startswith("F,"):
                parts = line[2:].split(",")
                if len(parts) >= 3:
                    self.data["loads"].append(
                        {
                            "node": parts[0].strip(),
                            "dof": parts[1].strip(),
                            "value": parts[2].strip(),
                            "type": "force",
                        }
                    )

            # Pressure loads (SFE)
            elif line.startswith("SFE,"):
                parts = line[4:].split(",")
                if len(parts) >= 4:
                    self.data["loads"].append(
                        {
                            "element": parts[0].strip(),
                            "face": parts[1].strip(),
                            "type": "pressure",
                            "value": parts[-1].strip(),
                        }
                    )

            i += 1

        self.data["num_nodes"] = len(self.data["nodes"])
        self.data["num_elements"] = len(self.data["elements"])
        self.data["num_element_types"] = len(self.data["element_types"])
        self.data["num_materials"] = len(self.data["materials"])

        # Determine primary element type
        if self.data["element_types"]:
            self.data["element_type"] = list(self.data["element_types"].values())[0]
        else:
            self.data["element_type"] = "unknown"

        return self.data

    def _parse_nblock(self, lines: List[str], start_idx: int) -> int:
        """Parse NBLOCK (node block) section."""
        i = start_idx + 1

        # Skip format line
        if i < len(lines) and lines[i].strip().startswith("("):
            i += 1

        while i < len(lines):
            line = lines[i].strip()

            # End of block
            if line.startswith("N"):
                break

            # Parse node line
            parts = line.split()
            if len(parts) >= 4:
                self.data["nodes"].append(
                    {
                        "node_num": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "z": float(parts[3]),
                    }
                )

            i += 1

        return i

    def _parse_eblock(self, lines: List[str], start_idx: int) -> int:
        """Parse EBLOCK (element block) section."""
        i = start_idx + 1

        # Skip format line
        if i < len(lines) and lines[i].strip().startswith("("):
            i += 1

        while i < len(lines):
            line = lines[i].strip()

            # End of block
            if line.startswith("E"):
                break

            # Parse element line (simplified)
            parts = line.split()
            if len(parts) >= 10:
                self.data["elements"].append(
                    {
                        "element_num": int(parts[0]),
                        "material": int(parts[1]),
                        "element_type": int(parts[2]),
                        "real_constant": int(parts[3]),
                        "section": int(parts[4]),
                        "nodes": [int(x) for x in parts[10:]],
                    }
                )

            i += 1

        return i

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse CDB file."""
        return self.parse(filepath)


class RSTParser:
    """Basic parser for Ansys RST (Results) files.

    Note: RST files are binary and require detailed format knowledge.
    This parser provides basic header information extraction.
    """

    def __init__(self):
        self.data: Dict[str, Any] = {}

    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse RST file header.

        Args:
            filepath: Path to .rst file

        Returns:
            Dictionary with results metadata
        """
        self.data = {
            "file_path": filepath,
            "file_type": "rst",
            "num_loadsteps": 0,
            "num_substeps": [],
            "available_results": [],
            "units": "unknown",
        }

        try:
            with open(filepath, "rb") as f:
                # Read file header (simplified)
                header = f.read(512)

                # Try to extract basic info
                # Actual RST format is complex and proprietary
                self.data["file_size"] = len(header)

                # Check for valid RST signature
                if header[:4] == b"RST\x00":
                    self.data["valid_format"] = True
                else:
                    self.data["valid_format"] = False

        except Exception as e:
            self.data["error"] = str(e)

        return self.data

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse RST file."""
        return self.parse(filepath)

    def get_result_summary(self) -> Dict[str, Any]:
        """Get summary of available results."""
        return {
            "displacements": "Available",
            "stresses": "Available",
            "strains": "Available",
            "reactions": "Available",
            "energies": "Available",
        }
