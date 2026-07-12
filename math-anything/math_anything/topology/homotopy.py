"""Cross-engine / cross-path homotopy checking over CategoryEngine."""

from __future__ import annotations

from dataclasses import dataclass

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


def cumulative_invariants_along_path(category_engine: CategoryEngine, path: list[str]) -> dict[str, list[str]]:
    """Accumulate kept and lost invariants along an explicit morphism path.

    Returns a JSON-serializable dict mapping ``"kept"`` and ``"lost"`` to sorted
    lists of invariant names encountered along ``path``.
    """
    kept: set[str] = set()
    lost: set[str] = set()

    for name in path:
        morphism = category_engine.morphisms.get(name)
        if morphism is None:
            raise KeyError(f"Morphism '{name}' not registered")
        invariants_kept = getattr(morphism, "invariants_kept", [])
        lost_source = getattr(morphism, "get_invariants_lost", None)
        if callable(lost_source):
            invariants_lost = lost_source()
        else:
            invariants_lost = getattr(morphism, "invariants_lost", [])
        if not kept:
            kept.update(invariants_kept)
        else:
            for inv in list(kept):
                if inv in invariants_lost:
                    kept.discard(inv)
            kept &= set(invariants_kept)
        lost.update(invariants_lost)

    return {"kept": sorted(kept), "lost": sorted(lost)}


def are_paths_homotopic(
    category_engine: CategoryEngine,
    path_a: list[str],
    path_b: list[str],
) -> HomotopyWitness:
    """Check whether two explicit morphism paths are homotopic.

    Two paths are considered homotopic when they connect the same source and
    target structures and preserve the same final sets of kept and lost
    invariants.
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
    for name in path_a + path_b:
        if name not in links:
            raise ValueError(f"Morphism '{name}' is registered but not linked")

    for path_label, path in (("path_a", path_a), ("path_b", path_b)):
        for i in range(len(path) - 1):
            current_target = links[path[i]].target_structure
            next_source = links[path[i + 1]].source_structure
            if current_target != next_source:
                raise ValueError(
                    f"{path_label} is disconnected: morphism '{path[i]}' ends at "
                    f"'{current_target}' but morphism '{path[i + 1]}' starts at "
                    f"'{next_source}'"
                )

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

    kept_a = set(inv_a["kept"])
    kept_b = set(inv_b["kept"])
    lost_a = set(inv_a["lost"])
    lost_b = set(inv_b["lost"])
    shared = sorted(kept_a & kept_b)
    equivalent = kept_a == kept_b and lost_a == lost_b

    if equivalent:
        confidence = 1.0
    else:
        all_a = kept_a | lost_a
        all_b = kept_b | lost_b
        union = all_a | all_b
        confidence = len(all_a & all_b) / len(union) if union else 1.0

    return HomotopyWitness(
        equivalent=equivalent,
        path_a=tuple(path_a),
        path_b=tuple(path_b),
        source=source_a,
        target=target_a,
        shared_invariants=shared,
        confidence=round(confidence, 4),
    )
