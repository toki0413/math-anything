"""Loop datamodel for topology-aware morphism engineering."""

from __future__ import annotations

from dataclasses import dataclass

from math_anything._compat import StrEnum


class LoopType(StrEnum):
    """Taxonomy of loops in an approximation/morphism graph."""

    CONVERGENCE = "convergence"
    COUPLING = "coupling"
    MIGRATION = "migration"
    MULTISCALE = "multiscale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Loop:
    """A closed walk in a morphism graph.

    Attributes:
        nodes: Ordered tuple of structure/engine names. The first and last node
            are equal for a closed loop.
        edges: Ordered tuple of morphism names connecting consecutive nodes.
        is_directed: Whether the loop respects edge direction.
        canonical_form: Human-readable normalized representation.
    """

    nodes: tuple[str, ...]
    edges: tuple[str, ...]
    is_directed: bool
    canonical_form: str

    def __post_init__(self) -> None:
        if len(self.nodes) < 2:
            raise ValueError("A loop must contain at least two nodes.")
        if self.nodes[0] != self.nodes[-1]:
            raise ValueError("A loop must start and end at the same node.")
        if len(self.edges) != len(self.nodes) - 1:
            raise ValueError("Number of edges must equal number of node transitions.")

    def to_dict(self) -> dict:
        return {
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "is_directed": self.is_directed,
            "canonical_form": self.canonical_form,
        }
