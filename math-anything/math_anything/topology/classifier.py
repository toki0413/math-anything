"""Rule-based classification of morphism loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .loop import Loop, LoopType


@dataclass
class LoopClassifier:
    """Classify loops using structural heuristics and morphism metadata."""

    convergence_markers: set[str] = field(
        default_factory=lambda: {
            "self_consistent",
            "scf",
            "mixer",
            "charge_density",
            "density",
            "wavefunction",
            "potential",
            "poisson",
            "ks_solve",
        }
    )
    coupling_markers: set[str] = field(
        default_factory=lambda: {
            "concurrent",
            "coupling",
            "two_way",
            "atomistic",
            "continuum",
            "micro",
            "macro",
        }
    )
    migration_markers: set[str] = field(
        default_factory=lambda: {
            "vasp",
            "quantum_espresso",
            "qe",
            "cp2k",
            "lammps",
            "gromacs",
            "abaqus",
            "ansys",
            "comsol",
        }
    )

    def classify(
        self,
        loop: Loop,
        morphism_lookup: dict[str, Any] | None = None,
    ) -> LoopType:
        """Return the LoopType best matching the loop."""
        text = " ".join(loop.nodes + loop.edges).lower()

        if self._matches(text, self.convergence_markers, threshold=2):
            return LoopType.CONVERGENCE

        if self._matches(text, self.coupling_markers, threshold=2):
            return LoopType.COUPLING

        if self._matches(text, self.migration_markers, threshold=2):
            return LoopType.MIGRATION

        if morphism_lookup is not None:
            categories = set()
            for edge in loop.edges:
                morph = morphism_lookup.get(edge)
                if morph is not None:
                    categories.add(getattr(morph, "category", "").lower())
            if len(categories) >= 2:
                return LoopType.MULTISCALE

        return LoopType.UNKNOWN

    @staticmethod
    def _matches(text: str, markers: set[str], threshold: int) -> bool:
        count = sum(1 for marker in markers if marker in text)
        return count >= threshold
