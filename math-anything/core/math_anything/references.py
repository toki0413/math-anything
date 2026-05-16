"""Constraint reference tracking — link constraints to literature sources.

Inspired by Perplexity's citation tracking, every mathematical constraint
can be traced to its physical basis, VASP manual section, or published paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConstraintReference:
    """A reference linking a constraint to its source documentation."""

    constraint: str = ""
    vasp_manual_section: str = ""
    paper_doi: Optional[str] = None
    paper_title: Optional[str] = None
    physical_basis: str = ""
    category: str = ""
    severity: str = "warning"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "constraint": self.constraint,
            "severity": self.severity,
        }
        if self.vasp_manual_section:
            d["vasp_manual_section"] = self.vasp_manual_section
        if self.paper_doi:
            d["paper_doi"] = self.paper_doi
        if self.paper_title:
            d["paper_title"] = self.paper_title
        if self.physical_basis:
            d["physical_basis"] = self.physical_basis
        if self.category:
            d["category"] = self.category
        return d


_VASP_REFERENCES: List[ConstraintReference] = [
    ConstraintReference(
        constraint="ENCUT > max(ENMAX)",
        vasp_manual_section="ENCUT",
        physical_basis="Plane-wave cutoff must exceed PAW dataset recommendation for convergence",
        category="convergence",
        severity="error",
    ),
    ConstraintReference(
        constraint="ENCUT >= 400 eV (production)",
        vasp_manual_section="ENCUT",
        physical_basis="Typical production-quality calculations require at least 400 eV",
        category="convergence",
        severity="warning",
    ),
    ConstraintReference(
        constraint="KPOINTS density sufficient for metallic/insulating systems",
        vasp_manual_section="KPOINTS",
        physical_basis="Brillouin zone sampling must resolve Fermi surface features",
        category="convergence",
        severity="warning",
    ),
    ConstraintReference(
        constraint="SIGMA < 0.2 for accurate DOS",
        vasp_manual_section="SIGMA",
        physical_basis="Smearing width must be small enough to resolve electronic features",
        category="accuracy",
        severity="warning",
    ),
    ConstraintReference(
        constraint="ISMEAR = 0 for semiconductors",
        vasp_manual_section="ISMEAR",
        physical_basis="Gaussian smearing preserves gap structure in insulators",
        category="methodology",
        severity="warning",
    ),
    ConstraintReference(
        constraint="ISMEAR = 1 or 0 for metals",
        vasp_manual_section="ISMEAR",
        physical_basis="Metals require smearing for Fermi surface integration stability",
        category="methodology",
        severity="warning",
    ),
    ConstraintReference(
        constraint="EDIFF < 1e-5 for geometry optimization",
        vasp_manual_section="EDIFF",
        physical_basis="Electronic convergence must be tight enough for accurate forces",
        category="convergence",
        severity="warning",
    ),
    ConstraintReference(
        constraint="EDIFFG negative for force convergence",
        vasp_manual_section="EDIFFG",
        physical_basis="Negative EDIFFG sets force convergence criterion in eV/Angstrom",
        category="convergence",
        severity="info",
    ),
    ConstraintReference(
        constraint="POTIM ~ 0.5-1.0 for ionic steps",
        vasp_manual_section="POTIM",
        physical_basis="Scaling factor for ionic steps must be small enough for stable optimization",
        category="stability",
        severity="info",
    ),
    ConstraintReference(
        constraint="NSW sufficient for convergence",
        vasp_manual_section="NSW",
        physical_basis="Maximum ionic steps must allow geometry to reach equilibrium",
        category="convergence",
        severity="info",
    ),
]

_LAMMPS_REFERENCES: List[ConstraintReference] = [
    ConstraintReference(
        constraint="timestep ~ 1 fs for atomistic MD",
        physical_basis="Timestep must resolve fastest vibrational mode (typically C-H stretch ~10 fs period)",
        category="stability",
        severity="error",
    ),
    ConstraintReference(
        constraint="NVT equilibration before NPT production",
        physical_basis="Volume must equilibrate before pressure coupling is activated",
        category="methodology",
        severity="warning",
    ),
    ConstraintReference(
        constraint="thermo output frequency reasonable",
        physical_basis="Output frequency affects correlation time estimation and disk usage",
        category="performance",
        severity="info",
    ),
    ConstraintReference(
        constraint="neighbor skin >= 2.0 Angstrom",
        physical_basis="Neighbor list skin distance must accommodate atom movement between rebuilds",
        category="stability",
        severity="warning",
    ),
]

_ABAQUS_REFERENCES: List[ConstraintReference] = [
    ConstraintReference(
        constraint="mesh convergence study required",
        physical_basis="FEM results must be mesh-independent for physical validity",
        category="convergence",
        severity="warning",
    ),
    ConstraintReference(
        constraint="element type appropriate for physics",
        physical_basis="Element formulation must match the deformation mode (bending, shear, etc.)",
        category="methodology",
        severity="warning",
    ),
    ConstraintReference(
        constraint="increment size small enough for nonlinear analysis",
        physical_basis="Load increments must be small enough for Newton-Raphson convergence",
        category="convergence",
        severity="warning",
    ),
]


class ReferenceTracker:
    """Look up constraint references by engine and category.

    Usage:
        tracker = ReferenceTracker()
        refs = tracker.lookup("vasp", "ENCUT")
        for ref in refs:
            print(ref.constraint, ref.vasp_manual_section)
    """

    _ENGINE_REFS: Dict[str, List[ConstraintReference]] = {
        "vasp": _VASP_REFERENCES,
        "lammps": _LAMMPS_REFERENCES,
        "abaqus": _ABAQUS_REFERENCES,
    }

    def lookup(
        self,
        engine: str,
        keyword: Optional[str] = None,
        category: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[ConstraintReference]:
        """Look up constraint references.

        Args:
            engine: Simulation engine name
            keyword: Filter by keyword in constraint or physical_basis
            category: Filter by category (convergence, methodology, stability, etc.)
            severity: Filter by severity (error, warning, info)
        """
        refs = self._ENGINE_REFS.get(engine.lower(), [])

        if keyword:
            kw = keyword.lower()
            refs = [
                r for r in refs
                if kw in r.constraint.lower()
                or kw in r.physical_basis.lower()
                or kw in r.vasp_manual_section.lower()
            ]

        if category:
            refs = [r for r in refs if r.category == category]

        if severity:
            refs = [r for r in refs if r.severity == severity]

        return refs

    def get_all(self, engine: str) -> List[ConstraintReference]:
        """Get all references for an engine."""
        return list(self._ENGINE_REFS.get(engine.lower(), []))

    def add_reference(self, engine: str, ref: ConstraintReference) -> None:
        """Add a custom reference for an engine."""
        engine_lower = engine.lower()
        if engine_lower not in self._ENGINE_REFS:
            self._ENGINE_REFS[engine_lower] = []
        self._ENGINE_REFS[engine_lower].append(ref)

    def engines(self) -> List[str]:
        """List engines with reference data."""
        return list(self._ENGINE_REFS.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            engine: [r.to_dict() for r in refs]
            for engine, refs in self._ENGINE_REFS.items()
        }
