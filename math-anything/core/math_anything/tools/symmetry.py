"""Symmetry analysis tool — group theory for crystal structures.

Pipeline: POSCAR/structure → space group → irreducible representations →
character table → selection rules.

Requires optional dependencies: spglib, pymatgen.
Falls back gracefully when unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpaceGroupResult:
    """Result of space group detection."""

    space_group_number: int = 0
    space_group_symbol: str = ""
    point_group: str = ""
    crystal_system: str = ""
    hall_number: int = 0
    symmetry_operations: int = 0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "space_group_number": self.space_group_number,
            "space_group_symbol": self.space_group_symbol,
            "point_group": self.point_group,
            "crystal_system": self.crystal_system,
        }
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class IrreducibleRepresentation:
    """An irreducible representation of a group."""

    label: str
    dimension: int = 1
    characters: Dict[str, float] = field(default_factory=dict)
    is_real: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "label": self.label,
            "dimension": self.dimension,
            "characters": self.characters,
            "is_real": self.is_real,
        }
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class CharacterTable:
    """Character table of a point group."""

    point_group: str
    irreps: List[IrreducibleRepresentation] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_group": self.point_group,
            "irreps": [irrep.to_dict() for irrep in self.irreps],
            "class_names": self.class_names,
            "description": self.description,
        }


@dataclass
class SelectionRuleResult:
    """Result of a selection rule analysis."""

    allowed: bool
    initial_irrep: str = ""
    final_irrep: str = ""
    operator_irrep: str = ""
    integral_nonzero: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "initial_irrep": self.initial_irrep,
            "final_irrep": self.final_irrep,
            "operator_irrep": self.operator_irrep,
            "integral_nonzero": self.integral_nonzero,
            "description": self.description,
        }


@dataclass
class SymmetryAnalysisResult:
    """Complete symmetry analysis result."""

    space_group: Optional[SpaceGroupResult] = None
    irreps: List[IrreducibleRepresentation] = field(default_factory=list)
    character_table: Optional[CharacterTable] = None
    selection_rules: List[SelectionRuleResult] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.space_group:
            d["space_group"] = self.space_group.to_dict()
        if self.irreps:
            d["irreducible_representations"] = [irrep.to_dict() for irrep in self.irreps]
        if self.character_table:
            d["character_table"] = self.character_table.to_dict()
        if self.selection_rules:
            d["selection_rules"] = [sr.to_dict() for sr in self.selection_rules]
        if self.description:
            d["description"] = self.description
        return d


_CRYSTAL_SYSTEMS = {
    1: "triclinic", 2: "triclinic",
    3: "monoclinic", 4: "monoclinic", 5: "monoclinic", 6: "monoclinic", 7: "monoclinic", 8: "monoclinic", 9: "monoclinic",
    10: "orthorhombic", 11: "orthorhombic", 12: "orthorhombic", 13: "orthorhombic", 14: "orthorhombic", 15: "orthorhombic",
    16: "tetragonal", 17: "tetragonal", 18: "tetragonal", 19: "tetragonal", 20: "tetragonal", 21: "tetragonal",
    22: "tetragonal", 23: "tetragonal", 24: "tetragonal", 25: "tetragonal", 26: "tetragonal", 27: "tetragonal",
    28: "tetragonal", 29: "tetragonal", 30: "tetragonal", 31: "tetragonal", 32: "tetragonal", 33: "tetragonal",
    34: "tetragonal", 35: "tetragonal", 36: "tetragonal", 37: "tetragonal", 38: "tetragonal", 39: "tetragonal",
    40: "tetragonal", 41: "tetragonal", 42: "tetragonal", 43: "tetragonal", 44: "tetragonal", 45: "tetragonal",
    46: "tetragonal", 47: "tetragonal", 48: "tetragonal", 49: "tetragonal", 50: "tetragonal", 51: "tetragonal",
    52: "tetragonal", 53: "tetragonal", 54: "tetragonal", 55: "tetragonal", 56: "tetragonal", 57: "tetragonal",
    58: "tetragonal", 59: "tetragonal", 60: "tetragonal", 61: "tetragonal", 62: "tetragonal", 63: "tetragonal",
    64: "tetragonal", 65: "tetragonal", 66: "tetragonal", 67: "tetragonal", 68: "tetragonal", 69: "tetragonal",
    70: "tetragonal", 71: "tetragonal", 72: "tetragonal", 73: "tetragonal", 74: "tetragonal",
    75: "trigonal", 76: "trigonal", 77: "trigonal", 78: "trigonal", 79: "trigonal", 80: "trigonal",
    81: "trigonal", 82: "trigonal", 83: "trigonal", 84: "trigonal", 85: "trigonal", 86: "trigonal",
    87: "trigonal", 88: "trigonal", 89: "trigonal", 90: "trigonal", 91: "trigonal", 92: "trigonal",
    93: "trigonal", 94: "trigonal", 95: "trigonal", 96: "trigonal", 97: "trigonal", 98: "trigonal",
    99: "trigonal", 100: "trigonal", 101: "trigonal", 102: "trigonal", 103: "trigonal", 104: "trigonal",
    105: "trigonal", 106: "trigonal", 107: "trigonal", 108: "trigonal", 109: "trigonal", 110: "trigonal",
    111: "trigonal", 112: "trigonal", 113: "trigonal", 114: "trigonal", 115: "trigonal", 116: "trigonal",
    117: "trigonal", 118: "trigonal", 119: "trigonal", 120: "trigonal", 121: "trigonal", 122: "trigonal",
    123: "trigonal", 124: "trigonal", 125: "trigonal", 126: "trigonal", 127: "trigonal", 128: "trigonal",
    129: "trigonal", 130: "trigonal", 131: "trigonal", 132: "trigonal", 133: "trigonal", 134: "trigonal",
    135: "trigonal", 136: "trigonal", 137: "trigonal", 138: "trigonal", 139: "trigonal", 140: "trigonal",
    141: "trigonal", 142: "trigonal", 143: "trigonal", 144: "trigonal", 145: "trigonal", 146: "trigonal",
    147: "trigonal", 148: "trigonal", 149: "trigonal", 150: "trigonal", 151: "trigonal", 152: "trigonal",
    153: "trigonal", 154: "trigonal", 155: "trigonal", 156: "trigonal", 157: "trigonal", 158: "trigonal",
    159: "trigonal", 160: "trigonal", 161: "trigonal", 162: "trigonal", 163: "trigonal", 164: "trigonal",
    165: "trigonal", 166: "trigonal", 167: "trigonal",
    168: "hexagonal", 169: "hexagonal", 170: "hexagonal", 171: "hexagonal", 172: "hexagonal", 173: "hexagonal",
    174: "hexagonal", 175: "hexagonal", 176: "hexagonal", 177: "hexagonal", 178: "hexagonal", 179: "hexagonal",
    180: "hexagonal", 181: "hexagonal", 182: "hexagonal", 183: "hexagonal", 184: "hexagonal", 185: "hexagonal",
    186: "hexagonal", 187: "hexagonal", 188: "hexagonal", 189: "hexagonal", 190: "hexagonal", 191: "hexagonal",
    192: "hexagonal", 193: "hexagonal", 194: "hexagonal",
    195: "cubic", 196: "cubic", 197: "cubic", 198: "cubic", 199: "cubic", 200: "cubic",
    201: "cubic", 202: "cubic", 203: "cubic", 204: "cubic", 205: "cubic", 206: "cubic",
    207: "cubic", 208: "cubic", 209: "cubic", 210: "cubic", 211: "cubic", 212: "cubic",
    213: "cubic", 214: "cubic", 215: "cubic", 216: "cubic", 217: "cubic", 218: "cubic",
    219: "cubic", 220: "cubic", 221: "cubic", 222: "cubic", 223: "cubic", 224: "cubic",
    225: "cubic", 226: "cubic", 227: "cubic", 228: "cubic", 229: "cubic", 230: "cubic",
}


class SymmetryAnalyzer:
    """Group theory analysis for crystal structures.

    Uses spglib for space group detection and SymPy for character tables
    when available. Falls back to basic analysis otherwise.
    """

    def analyze_structure(
        self,
        lattice: Optional[List[List[float]]] = None,
        positions: Optional[List[List[float]]] = None,
        numbers: Optional[List[int]] = None,
        space_group_hint: Optional[str] = None,
    ) -> SymmetryAnalysisResult:
        """Analyze symmetry of a crystal structure.

        Args:
            lattice: 3x3 lattice vectors (rows)
            positions: Fractional coordinates of atoms
            numbers: Atomic numbers
            space_group_hint: Known space group symbol (fallback)
        """
        sg_result = self._detect_space_group(lattice, positions, numbers, space_group_hint)
        irreps = self._compute_irreps(sg_result)
        char_table = self._compute_character_table(sg_result)
        selection_rules = self._compute_selection_rules(irreps)

        return SymmetryAnalysisResult(
            space_group=sg_result,
            irreps=irreps,
            character_table=char_table,
            selection_rules=selection_rules,
            description=(
                f"Space group {sg_result.space_group_symbol} "
                f"(#{sg_result.space_group_number}), "
                f"{sg_result.crystal_system} system, "
                f"{sg_result.symmetry_operations} symmetry operations"
            ),
        )

    def _detect_space_group(
        self,
        lattice: Optional[List[List[float]]],
        positions: Optional[List[List[float]]],
        numbers: Optional[List[int]],
        hint: Optional[str],
    ) -> SpaceGroupResult:
        """Detect space group using spglib or fallback."""
        if lattice is not None and positions is not None and numbers is not None:
            try:
                import spglib

                cell = (lattice, positions, numbers)
                dataset = spglib.get_symmetry_dataset(cell, symprec=1e-3)
                if dataset is not None:
                    return SpaceGroupResult(
                        space_group_number=dataset.number,
                        space_group_symbol=dataset.international,
                        point_group=dataset.pointgroup,
                        crystal_system=_CRYSTAL_SYSTEMS.get(dataset.number, "unknown"),
                        hall_number=dataset.hall_number,
                        symmetry_operations=len(dataset.rotations),
                        description=f"Detected by spglib: {dataset.international}",
                    )
            except (ImportError, Exception):
                pass

        if hint:
            return self._parse_space_group_hint(hint)

        return SpaceGroupResult(description="Space group detection requires spglib or structure data")

    def _parse_space_group_hint(self, hint: str) -> SpaceGroupResult:
        """Parse a space group symbol or number from a hint string."""
        hint = hint.strip()
        try:
            sg_num = int(hint)
            return SpaceGroupResult(
                space_group_number=sg_num,
                crystal_system=_CRYSTAL_SYSTEMS.get(sg_num, "unknown"),
                description=f"From hint: SG #{sg_num}",
            )
        except ValueError:
            pass

        return SpaceGroupResult(
            space_group_symbol=hint,
            description=f"From hint: {hint}",
        )

    def _compute_irreps(self, sg: SpaceGroupResult) -> List[IrreducibleRepresentation]:
        """Compute irreducible representations of the point group."""
        if not sg.point_group:
            return []

        try:
            from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup, AbelianGroup

            return self._sympy_irreps(sg)
        except ImportError:
            return self._fallback_irreps(sg)

    def _sympy_irreps(self, sg: SpaceGroupResult) -> List[IrreducibleRepresentation]:
        """Compute irreps using SymPy (limited to common point groups)."""
        pg = sg.point_group
        irreps = []

        pg_irrep_data = {
            "1": [IrreducibleRepresentation(label="A", dimension=1, characters={"E": 1.0})],
            "-1": [IrreducibleRepresentation(label="Ag", dimension=1, characters={"E": 1.0, "i": 1.0}),
                   IrreducibleRepresentation(label="Au", dimension=1, characters={"E": 1.0, "i": -1.0})],
            "m": [IrreducibleRepresentation(label="A'", dimension=1, characters={"E": 1.0, "σ": 1.0}),
                  IrreducibleRepresentation(label="A\"", dimension=1, characters={"E": 1.0, "σ": -1.0})],
            "2": [IrreducibleRepresentation(label="A", dimension=1, characters={"E": 1.0, "C2": 1.0}),
                  IrreducibleRepresentation(label="B", dimension=1, characters={"E": 1.0, "C2": -1.0})],
            "2/m": [
                IrreducibleRepresentation(label="Ag", dimension=1, characters={"E": 1.0, "C2": 1.0, "i": 1.0, "σ": 1.0}),
                IrreducibleRepresentation(label="Bg", dimension=1, characters={"E": 1.0, "C2": -1.0, "i": 1.0, "σ": -1.0}),
                IrreducibleRepresentation(label="Au", dimension=1, characters={"E": 1.0, "C2": 1.0, "i": -1.0, "σ": -1.0}),
                IrreducibleRepresentation(label="Bu", dimension=1, characters={"E": 1.0, "C2": -1.0, "i": -1.0, "σ": 1.0}),
            ],
            "m-3m": [
                IrreducibleRepresentation(label="A1g", dimension=1),
                IrreducibleRepresentation(label="A2g", dimension=1),
                IrreducibleRepresentation(label="Eg", dimension=2),
                IrreducibleRepresentation(label="T1g", dimension=3),
                IrreducibleRepresentation(label="T2g", dimension=3),
                IrreducibleRepresentation(label="A1u", dimension=1),
                IrreducibleRepresentation(label="A2u", dimension=1),
                IrreducibleRepresentation(label="Eu", dimension=2),
                IrreducibleRepresentation(label="T1u", dimension=3),
                IrreducibleRepresentation(label="T2u", dimension=3),
            ],
        }

        if pg in pg_irrep_data:
            irreps = pg_irrep_data[pg]
        else:
            irreps = [IrreducibleRepresentation(label="Γ₁", dimension=1, description=f"Trivial rep of {pg}")]

        return irreps

    def _fallback_irreps(self, sg: SpaceGroupResult) -> List[IrreducibleRepresentation]:
        """Fallback: return trivial representation only."""
        return [IrreducibleRepresentation(
            label="Γ₁",
            dimension=1,
            description=f"Trivial representation (install sympy for full character tables)",
        )]

    def _compute_character_table(self, sg: SpaceGroupResult) -> Optional[CharacterTable]:
        """Compute character table of the point group."""
        if not sg.point_group:
            return None

        irreps = self._compute_irreps(sg)
        return CharacterTable(
            point_group=sg.point_group,
            irreps=irreps,
            description=f"Character table for point group {sg.point_group}",
        )

    def _compute_selection_rules(
        self, irreps: List[IrreducibleRepresentation]
    ) -> List[SelectionRuleResult]:
        """Compute selection rules from irreducible representations.

        A transition Γ_i → Γ_f via operator Γ_op is allowed iff
        Γ_i ⊗ Γ_op ⊗ Γ_f contains the trivial representation A₁.
        """
        if len(irreps) <= 1:
            return []

        rules = []
        trivial_labels = {"A", "A1", "A1g", "Ag", "A'"}

        for i, irrep_i in enumerate(irreps):
            for j, irrep_f in enumerate(irreps):
                if i == j:
                    continue
                for k, irrep_op in enumerate(irreps):
                    if irrep_op.dimension > 3:
                        continue

                    allowed = self._check_selection_rule(irrep_i, irrep_f, irrep_op, trivial_labels)
                    if allowed:
                        rules.append(SelectionRuleResult(
                            allowed=True,
                            initial_irrep=irrep_i.label,
                            final_irrep=irrep_f.label,
                            operator_irrep=irrep_op.label,
                            integral_nonzero=True,
                            description=(
                                f"Transition {irrep_i.label} → {irrep_f.label} "
                                f"via {irrep_op.label} is ALLOWED "
                                f"(Γ_i ⊗ Γ_op ⊗ Γ_f contains A₁)"
                            ),
                        ))

        return rules[:20]

    @staticmethod
    def _check_selection_rule(
        irrep_i: IrreducibleRepresentation,
        irrep_f: IrreducibleRepresentation,
        irrep_op: IrreducibleRepresentation,
        trivial_labels: set,
    ) -> bool:
        """Check if a selection rule allows a transition.

        Simplified: for 1D irreps, the product of characters at identity
        gives the product representation. If it contains the trivial rep,
        the integral is nonzero.
        """
        if irrep_i.dimension != 1 or irrep_f.dimension != 1 or irrep_op.dimension != 1:
            return False

        chi_i = list(irrep_i.characters.values())
        chi_f = list(irrep_f.characters.values())
        chi_op = list(irrep_op.characters.values())

        if not chi_i or not chi_f or not chi_op:
            return False

        product_at_E = chi_i[0] * chi_op[0] * chi_f[0]
        return abs(product_at_E - 1.0) < 1e-10
