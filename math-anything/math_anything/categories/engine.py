"""范畴论轻量推理引擎。

提供态射合成、交换图验证、结构性质保持性查询等操作。
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from math_anything.topology.loop_engine import LoopEngine


@dataclass
class MorphismLink:
    """一个注册到引擎的态射实例."""

    morphism: Any
    source_structure: str
    target_structure: str


@dataclass
class CategoryEngine:
    """轻量范畴论推理引擎."""

    morphisms: dict[str, Any] = field(default_factory=dict)
    morphism_links: list[MorphismLink] = field(default_factory=list)
    structures: dict[str, Any] = field(default_factory=dict)

    def register_morphism(self, morphism: Any) -> None:
        self.morphisms[morphism.name] = morphism

    def register_structure(self, name: str, structure: Any) -> None:
        self.structures[name] = structure

    def link(self, morphism_name: str, source: str, target: str) -> None:
        if morphism_name not in self.morphisms:
            raise KeyError(f"Morphism '{morphism_name}' not registered")
        self.morphism_links.append(
            MorphismLink(
                morphism=self.morphisms[morphism_name],
                source_structure=source,
                target_structure=target,
            )
        )

    def compose(self, f_name: str, g_name: str) -> Any:
        f = self.morphisms[f_name]
        g = self.morphisms[g_name]
        return f.compose(g)

    def invariant_under(self, invariant_name: str, morphism_name: str) -> bool:
        m = self.morphisms.get(morphism_name)
        if m is None:
            return False
        return invariant_name in m.invariants_kept

    def kernel_of(self, morphism_name: str) -> str:
        m = self.morphisms.get(morphism_name)
        if m is None:
            return ""
        return m.kernel_description  # type: ignore[no-any-return]

    def what_is_lost(self, morphism_name: str) -> list[str]:
        m = self.morphisms.get(morphism_name)
        if m is None:
            return []
        return m.invariants_lost  # type: ignore[no-any-return]

    def what_is_kept(self, morphism_name: str) -> list[str]:
        m = self.morphisms.get(morphism_name)
        if m is None:
            return []
        return m.invariants_kept  # type: ignore[no-any-return]

    def get_morphism_chain(self, from_structure: str, to_structure: str) -> list[dict]:
        graph: dict[str, list[tuple[str, str]]] = {}
        for link in self.morphism_links:
            graph.setdefault(link.source_structure, []).append((link.target_structure, link.morphism.name))
        from collections import deque

        queue = deque([(from_structure, [])])  # type: ignore[var-annotated]
        visited = {from_structure}
        while queue:
            current, path = queue.popleft()
            if current == to_structure:
                return [{"step": i + 1, "morphism": m} for i, (_, m) in enumerate(path)]
            for neighbor, morph_name in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(neighbor, morph_name)]))
        return []

    def cumulative_loss(self, from_structure: str, to_structure: str) -> dict[str, Any]:
        chain = self.get_morphism_chain(from_structure, to_structure)
        if not chain:
            return {"error": f"No path from {from_structure} to {to_structure}"}
        lost_all: set[str] = set()
        kept_all: set[str] = set()
        chain_names: list[str] = []
        for step in chain:
            m = self.morphisms.get(step["morphism"])
            if m:
                chain_names.append(step["morphism"])
                lost_all.update(m.invariants_lost)
                if not kept_all:
                    kept_all.update(m.invariants_kept)
                else:
                    for inv in list(kept_all):
                        if inv in m.invariants_lost:
                            kept_all.discard(inv)
                    kept_all = kept_all & set(m.invariants_kept)
        return {
            "chain": chain_names,
            "final_invariants_kept": sorted(kept_all),
            "total_invariants_lost": sorted(lost_all),
            "kernel_chain": " -> ".join(self.kernel_of(name) for name in chain_names),
        }

    def find_structures_by_family(self, family: str) -> list[str]:
        return [
            name
            for name, s in self.structures.items()
            if hasattr(s, "family") and str(getattr(s, "family", "")) == family
        ]

    def find_morphisms_between(self, src: str, dst: str) -> list[str]:
        return [name for name, m in self.morphisms.items() if src in m.source_type and dst in m.target_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "morphisms_count": len(self.morphisms),
            "structures_count": len(self.structures),
            "links_count": len(self.morphism_links),
            "morphisms": {name: m.to_dict() for name, m in self.morphisms.items()},
            "structures": list(self.structures.keys()),
        }

    @functools.cached_property
    def loop_engine(self) -> "LoopEngine":
        """Return a topology-aware LoopEngine over this category."""
        from math_anything.topology.loop_engine import LoopEngine

        return LoopEngine(self)
