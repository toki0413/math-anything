"""Cross-engine / cross-path homotopy checking over CategoryEngine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from math_anything.categories.engine import CategoryEngine


@dataclass
class HomotopyWitness:
    """Result of comparing two morphism paths for homotopy equivalence."""

    equivalent: bool
    path_a: tuple[str, ...]
    path_b: tuple[str, ...]
    source: str
    target: str
    shared_invariants: list[str]
    confidence: float


def cumulative_invariants_along_path(
    category_engine: CategoryEngine, path: list[str]
) -> dict[str, set[str]]:
    """Accumulate kept and lost invariants along an explicit morphism path."""
    kept: set[str] = set()
    lost: set[str] = set()

    for name in path:
        morphism = category_engine.morphisms.get(name)
        if morphism is None:
            raise KeyError(f"Morphism '{name}' not registered")
        if not kept:
            kept.update(getattr(morphism, "invariants_kept", []))
        else:
            for inv in list(kept):
                if inv in getattr(morphism, "invariants_lost", []):
                    kept.discard(inv)
            kept &= set(getattr(morphism, "invariants_kept", []))
        lost.update(getattr(morphism, "invariants_lost", []))

    return {"kept": kept, "lost": lost}


def are_paths_homotopic(
    category_engine: CategoryEngine,
    path_a: list[str],
    path_b: list[str],
) -> HomotopyWitness:
    """Check whether two explicit morphism paths are homotopic.

    Two paths are considered homotopic when they connect the same source and
    target structures and preserve the same final set of invariants.
    """
    if not path_a or not path_b:
        return HomotopyWitness(
            equivalent=False,
            path_a=tuple(path_a),
            path_b=tuple(path_b),
            source="",
            target="",
            shared_invariants=[],
            confidence=0.0,
        )

    links = {link.morphism.name: link for link in category_engine.morphism_links}
    if path_a[0] not in links or path_b[0] not in links:
        raise ValueError("Path contains morphisms that are not linked")

    source_a = links[path_a[0]].source_structure
    source_b = links[path_b[0]].source_structure
    target_a = links[path_a[-1]].target_structure
    target_b = links[path_b[-1]].target_structure

    if source_a != source_b or target_a != target_b:
        return HomotopyWitness(
            equivalent=False,
            path_a=tuple(path_a),
            path_b=tuple(path_b),
            source=source_a,
            target=target_a,
            shared_invariants=[],
            confidence=0.0,
        )

    inv_a = cumulative_invariants_along_path(category_engine, path_a)
    inv_b = cumulative_invariants_along_path(category_engine, path_b)

    shared = sorted(inv_a["kept"] & inv_b["kept"])
    equivalent = inv_a["kept"] == inv_b["kept"]

    confidence = 1.0 if equivalent else len(shared) / max(len(inv_a["kept"] | inv_b["kept"]), 1)

    return HomotopyWitness(
        equivalent=equivalent,
        path_a=tuple(path_a),
        path_b=tuple(path_b),
        source=source_a,
        target=target_a,
        shared_invariants=shared,
        confidence=round(confidence, 4),
    )
