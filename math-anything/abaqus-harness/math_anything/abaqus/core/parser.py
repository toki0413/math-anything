"""Abaqus input file parser."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Material:
    """Material definition from Abaqus input."""
    name: str = ""
    elastic_modulus: Optional[float] = None
    poisson_ratio: Optional[float] = None
    density: Optional[float] = None
    material_type: str = "elastic"
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """Analysis step definition."""
    name: str = ""
    step_type: str = "static"
    nlgeom: bool = False
    time_period: float = 1.0
    increments: int = 100
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryCondition:
    """Boundary condition definition."""
    name: str = ""
    bc_type: str = "displacement"
    node_set: str = ""
    dof: List[int] = field(default_factory=list)
    magnitude: float = 0.0
    raw_data: Dict[str, Any] = field(default_factory=dict)


class AbaqusInputParser:
    """Parser for Abaqus .inp files."""

    def __init__(self):
        self.cards: Dict[str, list] = {}
        self.materials: List[Material] = []
        self.steps: List[Step] = []
        self.boundary_conditions: List[BoundaryCondition] = []

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse Abaqus input file."""
        with open(filepath, "r") as f:
            content = f.read()
        return self.parse(content)

    def parse(self, content: str) -> List[Dict[str, Any]]:
        """Parse Abaqus input content.
        
        Returns a list of command dictionaries with 'keyword' and 'data' keys.
        """
        commands = []
        current_cmd = None
        current_data = []

        for line in content.split("\n"):
            line = line.strip()

            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                if current_cmd is not None:
                    commands.append({"keyword": current_cmd, "data": current_data})
                current_cmd = line.split(",")[0].upper()
                current_data = []
            elif current_cmd:
                current_data.append(line)

        if current_cmd is not None:
            commands.append({"keyword": current_cmd, "data": current_data})

        self.cards = {cmd["keyword"]: cmd["data"] for cmd in commands}
        self._extract_structured_data()
        return commands

    def _extract_structured_data(self):
        """Extract structured data from parsed cards."""
        self._extract_materials()
        self._extract_steps()
        self._extract_boundary_conditions()

    def _extract_materials(self):
        """Extract material definitions."""
        for card_name, lines in self.cards.items():
            if card_name.startswith("MATERIAL"):
                mat = Material(name=card_name.replace("MATERIAL", "").strip(" ,"))
                for line in lines:
                    if "ELASTIC" in line.upper():
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try:
                                mat.elastic_modulus = float(parts[0])
                                mat.poisson_ratio = float(parts[1])
                            except ValueError:
                                pass
                self.materials.append(mat)

    def _extract_steps(self):
        """Extract step definitions."""
        for card_name, lines in self.cards.items():
            if card_name.startswith("STEP"):
                step = Step(name=card_name.replace("STEP", "").strip(" ,"))
                for line in lines:
                    if "NLGEOM" in line.upper():
                        step.nlgeom = True
                    if "STATIC" in line.upper():
                        step.step_type = "static"
                    if "DYNAMIC" in line.upper():
                        step.step_type = "dynamic"
                self.steps.append(step)

    def _extract_boundary_conditions(self):
        """Extract boundary condition definitions."""
        for card_name, lines in self.cards.items():
            if card_name == "BOUNDARY":
                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        bc = BoundaryCondition(
                            node_set=parts[0].strip(),
                            bc_type="displacement"
                        )
                        if len(parts) >= 3:
                            try:
                                bc.dof = [int(parts[1])]
                                bc.magnitude = float(parts[2]) if len(parts) > 2 else 0.0
                            except ValueError:
                                pass
                        self.boundary_conditions.append(bc)
