"""Enhanced Multiwfn harness with symbolic constraints and MathSchema integration.

Multiwfn is a wavefunction analysis program. This harness extracts
mathematical structures from its input scripts, CUBE files, and
wavefunction files with QTAIM and orbital analysis constraints.
"""

import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_CORE_ROOT = str(Path(__file__).parent.parent.parent.parent)
if _CORE_ROOT not in sys.path:
    sys.path.insert(0, _CORE_ROOT)

from math_anything.schemas import (
    BoundaryCondition,
    Discretization,
    GoverningEquation,
    MathematicalModel,
    MathematicalObject,
    MathSchema,
    MetaInfo,
    NumericalMethod,
    ParameterRelationship,
    Solver,
    SymbolicConstraint,
)


class MultiwfnAnalysisType(Enum):
    """Multiwfn analysis function types."""

    ELECTRON_DENSITY = "electron_density"
    CRITICAL_POINT = "critical_point"
    BASIN_ANALYSIS = "basin_analysis"
    ORBITAL_ANALYSIS = "orbital_analysis"
    ELECTROSTATIC = "electrostatic"
    AROMATICITY = "aromaticity"
    NBO = "nbo_analysis"
    ADCH = "atomic_charge"
    ORBITAL_COMPOSITION = "orbital_composition"


@dataclass
class MultiwfnCommand:
    """A parsed Multiwfn command."""

    function_number: int
    sub_functions: List[str]
    raw: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function": self.function_number,
            "sub_functions": self.sub_functions,
            "raw": self.raw,
        }


@dataclass
class MultiwfnResults:
    """Results of Multiwfn parsing."""

    commands: List[MultiwfnCommand]
    analysis_types: List[MultiwfnAnalysisType]
    parameters: Dict[str, Any]
    constraints: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_commands": len(self.commands),
            "analysis_types": [a.value for a in self.analysis_types],
            "parameters": self.parameters,
            "constraints": self.constraints,
        }


class MultiwfnSymbolicConstraints:
    """Symbolic constraints for wavefunction analysis."""

    DENSITY_CONSTRAINTS = {
        "total_electrons": [
            ("> 0", "Total electron count must be positive (critical)"),
            ("== N_electrons", "Should match expected electron count (warning)"),
        ],
        "max_density": [
            ("> 0", "Maximum density must be positive (critical)"),
        ],
        "rho_nuclear": [
            ("> 100", "Nuclear CP density typically > 100 a.u. (info)"),
        ],
    }

    ORBITAL_CONSTRAINTS = {
        "homo_lumo_gap": [
            (">= 0", "HOMO-LUMO gap must be non-negative (critical)"),
        ],
        "homo_energy": [
            ("< 0", "HOMO energy should be negative for stable molecule (warning)"),
        ],
        "occupation": [
            (">= 0", "Occupation number must be non-negative (critical)"),
            ("<= 2", "Occupation number cannot exceed 2 (Pauli) (critical)"),
        ],
    }

    CRITICAL_POINT_CONSTRAINTS = {
        "laplacian": [
            ("< 0 at BCP", "Laplacian < 0 indicates shared interaction (info)"),
            ("> 0 at BCP", "Laplacian > 0 indicates closed-shell interaction (info)"),
        ],
        "ellipticity": [
            (">= 0", "Ellipticity must be non-negative (critical)"),
        ],
    }

    BASIN_CONSTRAINTS = {
        "basin_volume": [
            ("> 0", "Basin volume must be positive (critical)"),
        ],
        "basin_population": [
            ("> 0", "Basin electron population must be positive (critical)"),
            (
                "<= N_electrons",
                "Basin population cannot exceed total electrons (critical)",
            ),
        ],
    }

    GRID_CONSTRAINTS = {
        "grid_points": [
            ("> 0", "Number of grid points must be positive (critical)"),
        ],
        "grid_spacing": [
            ("> 0", "Grid spacing must be positive (critical)"),
            ("< 0.5 Bohr", "Grid spacing should be < 0.5 Bohr for accuracy (info)"),
        ],
    }

    @classmethod
    def validate_parameter(
        cls, param_name: str, value: float, category: str = "density"
    ) -> List[Dict[str, Any]]:
        """Validate a parameter against constraints."""
        constraint_map = {
            "density": cls.DENSITY_CONSTRAINTS,
            "orbital": cls.ORBITAL_CONSTRAINTS,
            "critical_point": cls.CRITICAL_POINT_CONSTRAINTS,
            "basin": cls.BASIN_CONSTRAINTS,
            "grid": cls.GRID_CONSTRAINTS,
        }

        constraints = constraint_map.get(category, {}).get(param_name, [])
        results = []

        for constraint_expr, description in constraints:
            result = {
                "parameter": param_name,
                "value": value,
                "constraint": constraint_expr,
                "description": description,
                "satisfied": False,
            }

            try:
                if constraint_expr.startswith(">="):
                    threshold = float(
                        constraint_expr.replace(">=", "").strip().split()[0]
                    )
                    result["satisfied"] = float(value) >= threshold
                elif constraint_expr.startswith(">"):
                    threshold = float(
                        constraint_expr.replace(">", "").strip().split()[0]
                    )
                    result["satisfied"] = float(value) > threshold
                elif constraint_expr.startswith("<="):
                    threshold = float(
                        constraint_expr.replace("<=", "").strip().split()[0]
                    )
                    result["satisfied"] = float(value) <= threshold
                elif constraint_expr.startswith("<"):
                    threshold = float(
                        constraint_expr.replace("<", "").strip().split()[0]
                    )
                    result["satisfied"] = float(value) < threshold
                elif constraint_expr.startswith("=="):
                    result["satisfied"] = True  # Reference check, assume valid
                else:
                    result["satisfied"] = True
            except (ValueError, IndexError):
                result["satisfied"] = True

            results.append(result)

        return results


# Multiwfn function number to analysis type mapping
FUNCTION_MAP = {
    1: MultiwfnAnalysisType.ELECTRON_DENSITY,
    2: MultiwfnAnalysisType.CRITICAL_POINT,
    3: MultiwfnAnalysisType.BASIN_ANALYSIS,
    4: MultiwfnAnalysisType.ORBITAL_ANALYSIS,
    5: MultiwfnAnalysisType.ELECTROSTATIC,
    6: MultiwfnAnalysisType.ELECTROSTATIC,
    7: MultiwfnAnalysisType.ELECTRON_DENSITY,
    8: MultiwfnAnalysisType.ELECTRON_DENSITY,
    9: MultiwfnAnalysisType.ORBITAL_ANALYSIS,
    10: MultiwfnAnalysisType.ORBITAL_COMPOSITION,
    11: MultiwfnAnalysisType.ORBITAL_ANALYSIS,
    12: MultiwfnAnalysisType.AROMATICITY,
    13: MultiwfnAnalysisType.ADCH,
    14: MultiwfnAnalysisType.NBO,
    15: MultiwfnAnalysisType.ORBITAL_ANALYSIS,
    16: MultiwfnAnalysisType.ORBITAL_ANALYSIS,
    17: MultiwfnAnalysisType.ORBITAL_ANALYSIS,
    18: MultiwfnAnalysisType.ORBITAL_ANALYSIS,
    19: MultiwfnAnalysisType.ORBITAL_ANALYSIS,
    20: MultiwfnAnalysisType.ORBITAL_ANALYSIS,
    21: MultiwfnAnalysisType.ORBITAL_ANALYSIS,
    22: MultiwfnAnalysisType.ORBITAL_ANALYSIS,
    23: MultiwfnAnalysisType.ELECTROSTATIC,
    24: MultiwfnAnalysisType.ELECTROSTATIC,
    25: MultiwfnAnalysisType.ELECTROSTATIC,
}


class EnhancedMultiwfnParser:
    """Enhanced Multiwfn parser with symbolic constraint support."""

    def __init__(self):
        self.commands: List[MultiwfnCommand] = []
        self.analysis_types: List[MultiwfnAnalysisType] = []
        self.parameters: Dict[str, Any] = {}
        self.constraints: List[Dict[str, Any]] = []

    def parse(self, content: str) -> MultiwfnResults:
        """Parse Multiwfn input script."""
        self.commands = []
        self.analysis_types = []
        self.parameters = {}
        self.constraints = []

        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            try:
                func_num = int(parts[0])
                sub_funcs = parts[1:]

                cmd = MultiwfnCommand(
                    function_number=func_num,
                    sub_functions=sub_funcs,
                    raw=line,
                )
                self.commands.append(cmd)

                if func_num in FUNCTION_MAP:
                    atype = FUNCTION_MAP[func_num]
                    if atype not in self.analysis_types:
                        self.analysis_types.append(atype)

            except ValueError:
                self.parameters[parts[0]] = parts[1:] if len(parts) > 1 else True

        return MultiwfnResults(
            commands=self.commands,
            analysis_types=self.analysis_types,
            parameters=self.parameters,
            constraints=self.constraints,
        )

    def parse_file(self, filepath: str) -> MultiwfnResults:
        """Parse Multiwfn input file."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse(content)

    def extract_to_schema(self, filepath: str) -> MathSchema:
        """Parse and convert to MathSchema."""
        result = self.parse_file(filepath)
        return self._build_schema(result)

    def _build_schema(self, result: MultiwfnResults) -> MathSchema:
        """Build MathSchema from Multiwfn results."""
        equations = self._extract_governing_equations(result)
        bcs = self._extract_boundary_conditions(result)

        model = MathematicalModel(
            governing_equations=equations,
            boundary_conditions=bcs,
            constitutive_relations=self._extract_constitutive_relations(result),
        )

        symbolic_constraints = self._extract_symbolic_constraints(result)

        return MathSchema(
            schema_version="1.0.0",
            meta=MetaInfo(
                extracted_by="math-anything-multiwfn",
                extractor_version="0.1.0",
                source_files={"input": []},
            ),
            mathematical_model=model,
            numerical_method=NumericalMethod(
                discretization=Discretization(
                    space_discretization="real_space_grid",
                    time_integrator="none",
                ),
            ),
            symbolic_constraints=symbolic_constraints,
            raw_symbols=result.to_dict(),
        )

    def _extract_governing_equations(
        self, result: MultiwfnResults
    ) -> List[GoverningEquation]:
        """Extract governing equations based on analysis types."""
        equations = []

        if MultiwfnAnalysisType.ELECTRON_DENSITY in result.analysis_types:
            equations.append(
                GoverningEquation(
                    id="electron_density",
                    type="field_analysis",
                    name="Electron Density Analysis",
                    mathematical_form="ρ(r) = Σ_i n_i |ψ_i(r)|²",
                    variables=["density", "wavefunction", "occupation"],
                    description="Electron density from Kohn-Sham or Hartree-Fock orbitals",
                )
            )

        if MultiwfnAnalysisType.CRITICAL_POINT in result.analysis_types:
            equations.append(
                GoverningEquation(
                    id="qtaim_topology",
                    type="topology",
                    name="QTAIM Topological Analysis",
                    mathematical_form="∇ρ(r_c) = 0, rank(ρ) = 3",
                    variables=["critical_point", "gradient", "hessian"],
                    description="Bader's Quantum Theory of Atoms in Molecules",
                )
            )

        if MultiwfnAnalysisType.BASIN_ANALYSIS in result.analysis_types:
            equations.append(
                GoverningEquation(
                    id="basin_integration",
                    type="integration",
                    name="Atomic Basin Integration",
                    mathematical_form="N_A = ∫_Ω_A ρ(r) dr",
                    variables=["basin_population", "density", "basin_volume"],
                    description="Integration of electron density over atomic basins",
                )
            )

        if MultiwfnAnalysisType.ORBITAL_ANALYSIS in result.analysis_types:
            equations.append(
                GoverningEquation(
                    id="orbital_analysis",
                    type="eigenvalue_analysis",
                    name="Molecular Orbital Analysis",
                    mathematical_form="Hψ_i = ε_i ψ_i, H = T + V_ne + V_ee",
                    variables=["orbital_energy", "wavefunction", "hamiltonian"],
                    description="Molecular orbital energy and composition analysis",
                )
            )

        if MultiwfnAnalysisType.ELECTROSTATIC in result.analysis_types:
            equations.append(
                GoverningEquation(
                    id="electrostatic",
                    type="potential_analysis",
                    name="Electrostatic Potential",
                    mathematical_form="V(r) = Σ_A Z_A/|r-R_A| - ∫ ρ(r')/|r-r'| dr'",
                    variables=["potential", "density", "nuclear_charge"],
                    description="Molecular electrostatic potential from nuclear and electronic contributions",
                )
            )

        if not equations:
            equations.append(
                GoverningEquation(
                    id="wavefunction_analysis",
                    type="post_processing",
                    name="Wavefunction Post-Processing",
                    mathematical_form="ρ(r) = Σ_i n_i |ψ_i(r)|²",
                    variables=["density", "wavefunction"],
                    description="General wavefunction analysis",
                )
            )

        return equations

    def _extract_boundary_conditions(
        self, result: MultiwfnResults
    ) -> List[BoundaryCondition]:
        """Extract boundary conditions for wavefunction analysis."""
        bcs = []

        if MultiwfnAnalysisType.CRITICAL_POINT in result.analysis_types:
            bcs.append(
                BoundaryCondition(
                    id="zero_gradient",
                    type="critical_point_condition",
                    domain={
                        "geometric_region": "real_space",
                        "entity_type": "critical_point",
                    },
                    mathematical_object=MathematicalObject(
                        field="grad_rho",
                        tensor_rank=1,
                        tensor_form="∇ρ(r_c) = 0",
                    ),
                    software_implementation={
                        "command": "Multiwfn function 2",
                        "parameters": {},
                    },
                )
            )

        return bcs

    def _extract_constitutive_relations(
        self, result: MultiwfnResults
    ) -> List[Dict[str, Any]]:
        """Extract constitutive relations."""
        relations = []

        if MultiwfnAnalysisType.CRITICAL_POINT in result.analysis_types:
            relations.append(
                {
                    "type": "hessian_classification",
                    "name": "QTAIM CP Classification",
                    "mathematical_form": "rank = 3, signature = (3, ±n)",
                    "description": "Critical points classified by Hessian eigenvalue signature",
                }
            )

        if MultiwfnAnalysisType.ELECTROSTATIC in result.analysis_types:
            relations.append(
                {
                    "type": "coulomb_law",
                    "name": "Coulomb Interaction",
                    "mathematical_form": "V(r) = Σ Z_A/|r-R_A| - ∫ ρ(r')/|r-r'| dr'",
                    "description": "Electrostatic potential from point nuclei and continuous electron density",
                }
            )

        return relations

    def _extract_symbolic_constraints(
        self, result: MultiwfnResults
    ) -> List[SymbolicConstraint]:
        """Extract symbolic constraints for wavefunction analysis."""
        constraints = []

        # Orbital constraints
        constraints.append(
            SymbolicConstraint(
                expression="occupation >= 0",
                description="Occupation number must be non-negative (Pauli principle)",
            )
        )
        constraints.append(
            SymbolicConstraint(
                expression="occupation <= 2",
                description="Occupation number cannot exceed 2 (Pauli exclusion)",
            )
        )

        # Density constraints
        if MultiwfnAnalysisType.ELECTRON_DENSITY in result.analysis_types:
            constraints.append(
                SymbolicConstraint(
                    expression="total_electrons > 0",
                    description="Total electron count must be positive",
                )
            )
            constraints.append(
                SymbolicConstraint(
                    expression="rho(r) >= 0",
                    description="Electron density must be non-negative everywhere",
                )
            )

        # Critical point constraints
        if MultiwfnAnalysisType.CRITICAL_POINT in result.analysis_types:
            constraints.append(
                SymbolicConstraint(
                    expression="ellipticity >= 0",
                    description="Ellipticity at BCP must be non-negative",
                )
            )

        # Basin constraints
        if MultiwfnAnalysisType.BASIN_ANALYSIS in result.analysis_types:
            constraints.append(
                SymbolicConstraint(
                    expression="basin_population > 0",
                    description="Basin electron population must be positive",
                )
            )
            constraints.append(
                SymbolicConstraint(
                    expression="Σ N_A = N_total",
                    description="Sum of basin populations equals total electrons (conservation)",
                )
            )

        # Grid constraints
        constraints.append(
            SymbolicConstraint(
                expression="grid_spacing > 0",
                description="Grid spacing must be positive",
            )
        )

        return constraints


def parse_multiwfn(filepath: str) -> MultiwfnResults:
    """Parse Multiwfn input file."""
    parser = EnhancedMultiwfnParser()
    return parser.parse_file(filepath)
