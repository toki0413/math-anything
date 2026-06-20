"""Base domain class for Bourbaki domain instantiation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainAnalysis:
    """Result of analyzing a domain — what's preserved and what's lost."""

    domain_name: str
    conservation_field: dict[str, Any] = field(default_factory=dict)
    morphism_chain: list[dict[str, Any]] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    weakened: list[str] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    emerged: list[str] = field(default_factory=list)
    eigenvalues: list[float] = field(default_factory=list)
    cfl_condition: dict[str, float] = field(default_factory=dict)
    stability: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of the domain analysis."""
        lines = [
            f"Domain: {self.domain_name}",
            f"Morphism chain length: {len(self.morphism_chain)}",
            f"Preserved invariants: {len(self.preserved)}",
            f"Weakened invariants: {len(self.weakened)}",
            f"Lost invariants: {len(self.lost)}",
            f"Emerged invariants: {len(self.emerged)}",
        ]
        if self.preserved:
            lines.append(f"  Preserved: {', '.join(self.preserved[:5])}")
        if self.lost:
            lines.append(f"  Lost: {', '.join(self.lost[:5])}")
        if self.eigenvalues:
            lines.append(f"  Characteristic speeds: {[f'{e:.3f}' for e in self.eigenvalues[:5]]}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the domain analysis to a dictionary."""
        return {
            "domain_name": self.domain_name,
            "preserved": self.preserved,
            "weakened": self.weakened,
            "lost": self.lost,
            "emerged": self.emerged,
            "eigenvalues": self.eigenvalues,
            "cfl_condition": self.cfl_condition,
            "stability": self.stability,
            "morphism_chain": self.morphism_chain,
        }


class Domain:
    """Base class for Bourbaki domain instantiation.

    A Domain represents a physics discipline as a specific configuration
    of mathematical structures (conservation fields + morphism chains).

    The core proposition: DFT, MD, CFD, FEM are not different software tools —
    they are different morphism chains applied to the same conservation fields.
    """

    name: str = "base"
    description: str = ""
    equation_type: str = ""
    default_params: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = {**self.default_params, **(params or {})}

    def build_conservation_field(self) -> dict[str, Any]:
        """Build the conservation field for this domain."""
        raise NotImplementedError

    def build_morphism_chain(self) -> list[dict[str, Any]]:
        """Build the morphism chain for this domain."""
        raise NotImplementedError

    def analyze(self) -> DomainAnalysis:
        """Full domain analysis: conservation + morphisms + constraints."""
        conservation = self.build_conservation_field()
        morphism_chain = self.build_morphism_chain()

        # Propagate constraints along the morphism chain
        preserved = list(conservation.get("conservation_laws", []))
        weakened = []
        lost = []
        emerged = []

        for morphism in morphism_chain:
            kept = morphism.get("invariants_kept", [])
            lost_in_step = morphism.get("invariants_lost", [])
            introduced = morphism.get("invariants_introduced", [])

            # Update tracking
            new_preserved = []
            for inv in preserved:
                if inv in lost_in_step:
                    lost.append(inv)
                elif inv in kept:
                    new_preserved.append(inv)
                elif inv not in weakened:
                    weakened.append(inv)
            preserved = new_preserved

            # Track items weakened in earlier steps but lost now
            new_weakened = []
            for inv in weakened:
                if inv in lost_in_step:
                    lost.append(inv)
                else:
                    new_weakened.append(inv)
            weakened = new_weakened

            for inv in introduced:
                if inv not in emerged:
                    emerged.append(inv)

        return DomainAnalysis(
            domain_name=self.name,
            conservation_field=conservation,
            morphism_chain=morphism_chain,
            preserved=preserved,
            weakened=weakened,
            lost=lost,
            emerged=emerged,
            eigenvalues=conservation.get("eigenvalues", []),
            cfl_condition=conservation.get("cfl_condition", {}),
            stability=conservation.get("stability", {}),
        )

    def what_is_lost(self) -> list[str]:
        """What mathematical properties are lost in this domain?"""
        return self.analyze().lost

    def what_is_kept(self) -> list[str]:
        """What mathematical properties survive all morphisms?"""
        return self.analyze().preserved

    def compare_with(self, other: "Domain") -> dict[str, Any]:
        """Compare two domains — same conservation field, different morphism chains."""
        my_analysis = self.analyze()
        other_analysis = other.analyze()

        common_preserved = set(my_analysis.preserved) & set(other_analysis.preserved)
        only_in_self = set(my_analysis.preserved) - set(other_analysis.preserved)
        only_in_other = set(other_analysis.preserved) - set(my_analysis.preserved)

        return {
            "domain_a": self.name,
            "domain_b": other.name,
            "common_preserved": sorted(common_preserved),
            "only_in_a": sorted(only_in_self),
            "only_in_b": sorted(only_in_other),
            "a_lost": my_analysis.lost,
            "b_lost": other_analysis.lost,
        }
