"""LAMMPS input file and log file parsers.

Parses LAMMPS input scripts to extract commands, parameters, and simulation settings.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class LammpsCommandType(Enum):
    """Types of LAMMPS commands relevant for mathematical extraction."""

    ATOM_STYLE = "atom_style"
    BOUNDARY = "boundary"
    UNITS = "units"
    LATTICE = "lattice"
    REGION = "region"
    CREATE_BOX = "create_box"
    CREATE_ATOMS = "create_atoms"
    MASS = "mass"
    PAIR_STYLE = "pair_style"
    PAIR_COEFF = "pair_coeff"
    FIX = "fix"
    COMPUTE = "compute"
    TIMESTEP = "timestep"
    RUN = "run"
    MINIMIZE = "minimize"
    DUMP = "dump"
    THERMO = "thermo"
    VARIABLE = "variable"
    GROUP = "group"
    VELOCITY = "velocity"
    NEIGHBOR = "neighbor"
    NEIGH_MODIFY = "neigh_modify"
    READ_DATA = "read_data"
    RESTART = "restart"


@dataclass
class LammpsCommand:
    """A single LAMMPS command parsed from input file."""

    command: str
    args: List[str]
    line_number: int
    raw_line: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "args": self.args,
            "line_number": self.line_number,
            "raw_line": self.raw_line,
        }


@dataclass
class FixCommand:
    """Parsed fix command with semantics."""

    fix_id: str
    group_id: str
    fix_style: str
    args: List[str]
    raw: LammpsCommand

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fix_id": self.fix_id,
            "group_id": self.group_id,
            "fix_style": self.fix_style,
            "args": self.args,
        }


@dataclass
class PairStyle:
    """Pair style configuration."""

    style: str
    args: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style,
            "args": self.args,
        }


@dataclass
class ComputationalSettings:
    """Computational settings extracted from input."""

    units: str = "lj"
    atom_style: str = "atomic"
    boundary_style: List[str] = field(default_factory=lambda: ["p", "p", "p"])
    timestep: Optional[float] = None
    pair_style: Optional[PairStyle] = None
    fixes: List[FixCommand] = field(default_factory=list)
    computes: List[Dict[str, Any]] = field(default_factory=list)
    dumps: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    groups: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units": self.units,
            "atom_style": self.atom_style,
            "boundary_style": self.boundary_style,
            "timestep": self.timestep,
            "pair_style": self.pair_style.to_dict() if self.pair_style else None,
            "fixes": [f.to_dict() for f in self.fixes],
            "computes": self.computes,
            "dumps": self.dumps,
            "variables": self.variables,
            "groups": self.groups,
        }

    def get_integrator_fixes(self) -> List[FixCommand]:
        """Get integrator fixes (nve, nvt, npt, etc.).

        Returns:
            List of integrator fix commands.
        """
        integrator_styles = {"nve", "nvt", "npt", "nph", "langevin", "brownian"}
        return [f for f in self.fixes if f.fix_style in integrator_styles]

    def get_constraint_fixes(self) -> List[FixCommand]:
        """Get constraint fixes.

        Returns:
            List of constraint fix commands.
        """
        constraint_styles = {"spring", "wall", "deform", "indent", "move", "setforce"}
        return [f for f in self.fixes if f.fix_style in constraint_styles]

    def get_fixes_by_style(self, style: str) -> List[FixCommand]:
        """Get all fixes of a specific style.

        Args:
            style: Fix style name (e.g., 'nve', 'nvt', 'deform').

        Returns:
            List of matching FixCommand objects.
        """
        style_lower = style.lower()
        return [f for f in self.fixes if f.fix_style == style_lower]


class LammpsInputParser:
    """Parser for LAMMPS input files.

    Parses LAMMPS input scripts to extract commands, parameters, and settings.

    Example:
        ```python
        parser = LammpsInputParser()
        commands = parser.parse_file("in.lammps")
        settings = parser.extract_settings(commands)
        ```
    """

    # Commands that define the mathematical model
    PHYSICS_COMMANDS = {
        "pair_style",
        "pair_coeff",
        "bond_style",
        "bond_coeff",
        "angle_style",
        "angle_coeff",
        "dihedral_style",
        "dihedral_coeff",
        "improper_style",
        "improper_coeff",
        "kspace_style",
    }

    # Commands that define numerical methods
    NUMERICAL_COMMANDS = {
        "fix",
        "timestep",
        "run",
        "minimize",
        "neigh_modify",
    }

    # Commands that define boundary/initial conditions
    BC_COMMANDS = {
        "boundary",
        "region",
        "create_box",
        "fix",
        "velocity",
    }

    def __init__(self):
        self.commands: List[LammpsCommand] = []
        self.settings = ComputationalSettings()

    def parse_file(self, filepath: str) -> List[LammpsCommand]:
        """Parse a LAMMPS input file.

        Args:
            filepath: Path to LAMMPS input file.

        Returns:
            List of parsed commands.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse(content)

    def parse(self, content: str) -> List[LammpsCommand]:
        """Parse LAMMPS input content.

        Args:
            content: Input file content as string.

        Returns:
            List of parsed commands.
        """
        self.commands = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Skip comments and empty lines
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Remove inline comments
            if "#" in stripped:
                stripped = stripped[: stripped.index("#")].strip()

            # Skip continuation lines (handled with previous line)
            if stripped.endswith("&"):
                continue

            # Parse command
            parts = stripped.split()
            if parts:
                cmd = LammpsCommand(
                    command=parts[0].lower(),
                    args=parts[1:],
                    line_number=line_num,
                    raw_line=stripped,
                )
                self.commands.append(cmd)

        return self.commands

    def extract_settings(
        self, commands: Optional[List[LammpsCommand]] = None
    ) -> ComputationalSettings:
        """Extract computational settings from commands.

        Args:
            commands: List of commands. If None, uses previously parsed commands.

        Returns:
            ComputationalSettings object.
        """
        if commands is None:
            commands = self.commands

        self.settings = ComputationalSettings()

        for cmd in commands:
            self._process_command(cmd)

        return self.settings

    def _process_command(self, cmd: LammpsCommand):
        """Process a single command and update settings."""
        c = cmd.command
        args = cmd.args

        if c == "units" and args:
            self.settings.units = args[0]

        elif c == "atom_style" and args:
            self.settings.atom_style = args[0]

        elif c == "boundary" and len(args) >= 3:
            self.settings.boundary_style = args[:3]

        elif c == "timestep" and args:
            try:
                self.settings.timestep = float(args[0])
            except ValueError:
                pass

        elif c == "pair_style" and args:
            self.settings.pair_style = PairStyle(
                style=args[0],
                args=args[1:],
            )

        elif c == "fix" and len(args) >= 3:
            fix = FixCommand(
                fix_id=args[0],
                group_id=args[1],
                fix_style=args[2].lower(),
                args=args[3:],
                raw=cmd,
            )
            self.settings.fixes.append(fix)

        elif c == "compute" and len(args) >= 3:
            self.settings.computes.append(
                {
                    "compute_id": args[0],
                    "group_id": args[1],
                    "compute_style": args[2],
                    "args": args[3:],
                }
            )

        elif c == "variable" and len(args) >= 3:
            var_name = args[0]
            var_value = " ".join(args[2:])  # Skip the 'equal' or other style
            self.settings.variables[var_name] = var_value

        elif c == "group" and len(args) >= 2:
            group_id = args[0]
            group_def = " ".join(args[1:])
            self.settings.groups[group_id] = group_def

        elif c == "dump" and len(args) >= 3:
            self.settings.dumps.append(
                {
                    "dump_id": args[0],
                    "dump_freq": args[1],
                    "args": args[2:],
                }
            )


class LammpsLogParser:
    """Parser for LAMMPS log files.

    Extracts thermodynamic data and simulation information from LAMMPS logs.
    """

    def __init__(self):
        self.thermo_data: List[Dict[str, float]] = []
        self.run_info: Dict[str, Any] = {}

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse a LAMMPS log file.

        Args:
            filepath: Path to log file.

        Returns:
            Dictionary with thermo data and run info.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse(content)

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse LAMMPS log content.

        Args:
            content: Log file content.

        Returns:
            Dictionary with parsed data.
        """
        self.thermo_data = []
        self.run_info = {}

        lines = content.split("\n")
        in_thermo = False
        thermo_headers = []

        for line in lines:
            stripped = line.strip()

            # Detect thermo header
            if stripped.startswith("Step ") or stripped.startswith("Loop "):
                thermo_headers = stripped.split()
                in_thermo = True
                continue

            # Parse thermo data
            if in_thermo and stripped:
                # Check if it's data line
                parts = stripped.split()
                if len(parts) == len(thermo_headers):
                    try:
                        row = {}
                        for i, header in enumerate(thermo_headers):
                            row[header] = float(parts[i])
                        self.thermo_data.append(row)
                    except ValueError:
                        in_thermo = False
                else:
                    in_thermo = False

            # Extract run info
            if "Total # of neighbors" in stripped:
                match = re.search(r"(\d+)", stripped)
                if match:
                    self.run_info["total_neighbors"] = int(match.group(1))

            if "Ave neighs/atom" in stripped:
                match = re.search(r"([\d.]+)", stripped)
                if match:
                    self.run_info["ave_neighs_per_atom"] = float(match.group(1))

        return {
            "thermo_data": self.thermo_data,
            "run_info": self.run_info,
        }

    def get_final_values(self) -> Optional[Dict[str, float]]:
        """Get final thermodynamic values.

        Returns:
            Dictionary with final values, or None if no data.
        """
        if self.thermo_data:
            return self.thermo_data[-1]
        return None

    def get_energy_stats(self) -> Dict[str, float]:
        """Get energy statistics from thermo data.

        Returns:
            Dictionary with min, max, mean, final values for energy.
        """
        if not self.thermo_data:
            return {}

        stats = {}
        for key in ["TotEng", "PotEng", "KinEng"]:
            values = [row.get(key) for row in self.thermo_data if key in row]
            if values:
                stats[f"{key}_min"] = min(values)
                stats[f"{key}_max"] = max(values)
                stats[f"{key}_mean"] = sum(values) / len(values)
                stats[f"{key}_final"] = values[-1]

        return stats
