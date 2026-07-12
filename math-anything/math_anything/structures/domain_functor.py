"""Functors and natural transformations over domain morphism chains."""

from __future__ import annotations

from typing import Any

from math_anything.categories.engine import CategoryEngine
from math_anything.domains import DOMAIN_REGISTRY
from math_anything.structures.functor import Functor, NaturalTransformation
from math_anything.topology.homotopy import cumulative_invariants_along_path


class DomainFunctor(Functor):
    """Functor that maps source domain structures/morphisms to target names."""

    def __init__(
        self,
        object_map: dict[str, str],
        morphism_map: dict[str, str],
    ):
        self.object_map = object_map
        self.morphism_map = morphism_map

    def map_object(self, obj: Any) -> Any:
        if obj not in self.object_map:
            raise KeyError(f"Object {obj} not mapped by functor")
        return self.object_map[obj]

    def map_morphism(self, morphism: Any) -> Any:
        name = getattr(morphism, "name", morphism)
        if name not in self.morphism_map:
            raise KeyError(f"Morphism {name} not mapped by functor")
        return self.morphism_map[name]


def build_domain_pair_engine(
    domain_a_name: str,
    params_a: dict[str, Any],
    domain_b_name: str,
    params_b: dict[str, Any],
    prefix_a: str = "a",
    prefix_b: str = "b",
) -> tuple[CategoryEngine, list[str], list[str]]:
    """Build a merged CategoryEngine containing two domain morphism chains."""
    if domain_a_name not in DOMAIN_REGISTRY or domain_b_name not in DOMAIN_REGISTRY:
        available = sorted(DOMAIN_REGISTRY.keys())
        raise KeyError(f"Unknown domain. Available: {available}")

    dom_a = DOMAIN_REGISTRY[domain_a_name](params_a)
    dom_b = DOMAIN_REGISTRY[domain_b_name](params_b)
    chain_a = dom_a.build_morphism_chain()
    chain_b = dom_b.build_morphism_chain()

    engine = CategoryEngine()
    path_a: list[str] = []
    path_b: list[str] = []

    prev_a = f"{prefix_a}_start"
    terminal_kept_a: list[str] = []
    for i, step in enumerate(chain_a):
        name = f"{prefix_a}_{step['name']}"
        kept = step.get("invariants_kept", [])
        terminal_kept_a = list(kept)
        engine.register_morphism(type("M", (), {
            "name": name,
            "invariants_kept": kept,
            "invariants_lost": step.get("invariants_lost", []),
        })())
        target = f"{prefix_a}_state_{i}"
        engine.link(name, prev_a, target)
        path_a.append(name)
        prev_a = target
    final_a = f"{prefix_a}_end"
    terminal_a = type("M", (), {
        "name": f"{prefix_a}_terminal",
        "invariants_kept": terminal_kept_a,
        "invariants_lost": [],
    })()
    engine.register_morphism(terminal_a)
    engine.link(f"{prefix_a}_terminal", prev_a, final_a)
    path_a.append(f"{prefix_a}_terminal")

    prev_b = f"{prefix_b}_start"
    terminal_kept_b: list[str] = []
    for i, step in enumerate(chain_b):
        name = f"{prefix_b}_{step['name']}"
        kept = step.get("invariants_kept", [])
        terminal_kept_b = list(kept)
        engine.register_morphism(type("M", (), {
            "name": name,
            "invariants_kept": kept,
            "invariants_lost": step.get("invariants_lost", []),
        })())
        target = f"{prefix_b}_state_{i}"
        engine.link(name, prev_b, target)
        path_b.append(name)
        prev_b = target
    final_b = f"{prefix_b}_end"
    terminal_b = type("M", (), {
        "name": f"{prefix_b}_terminal",
        "invariants_kept": terminal_kept_b,
        "invariants_lost": [],
    })()
    engine.register_morphism(terminal_b)
    engine.link(f"{prefix_b}_terminal", prev_b, final_b)
    path_b.append(f"{prefix_b}_terminal")

    return engine, path_a, path_b


def build_bridge_natural_transformation(
    engine: CategoryEngine,
    source_prefix: str,
    target_prefix: str,
    bridge_invariants_kept: list[str],
    bridge_invariants_lost: list[str],
) -> NaturalTransformation:
    """Register bridge morphisms from every source structure to its target counterpart.

    Assumes structures are named `{source_prefix}_start`, `{source_prefix}_state_i`,
    `{source_prefix}_end` and similarly for target.
    """
    components: dict[str, str] = {}
    source_structures = {link.source_structure for link in engine.morphism_links}
    source_structures.update(link.target_structure for link in engine.morphism_links)

    for src in sorted(source_structures):
        if not src.startswith(f"{source_prefix}_"):
            continue
        dst = src.replace(f"{source_prefix}_", f"{target_prefix}_", 1)
        if dst not in source_structures:
            continue
        name = f"bridge_{source_prefix}_to_{target_prefix}_{src}"
        engine.register_morphism(type("M", (), {
            "name": name,
            "invariants_kept": bridge_invariants_kept,
            "invariants_lost": bridge_invariants_lost,
        })())
        engine.link(name, src, dst)
        components[src] = name

    return NaturalTransformation(components)


def is_domain_natural_transformation(
    F: DomainFunctor,
    G: DomainFunctor,
    eta: NaturalTransformation,
    engine: CategoryEngine,
    test_morphisms: list[str],
) -> tuple[bool, str]:
    """Check whether eta: F => G is a natural transformation on the given engine.

    For each test morphism f: X -> Y, verifies that the two target paths
    G(f) ∘ eta_X and eta_Y ∘ F(f) preserve the same cumulative invariants.
    """
    links = {link.morphism.name: link for link in engine.morphism_links}

    for f_name in test_morphisms:
        if f_name not in links:
            return False, f"Morphism '{f_name}' not found in engine"

        source_obj = links[f_name].source_structure
        target_obj = links[f_name].target_structure

        f_f = F.map_morphism(f_name)
        f_g = G.map_morphism(f_name)
        eta_src = eta.components.get(F.map_object(source_obj))
        eta_dst = eta.components.get(F.map_object(target_obj))

        if eta_src is None or eta_dst is None:
            return False, f"Missing bridge for {source_obj} or {target_obj}"

        path1 = [f_f, eta_dst]
        path2 = [eta_src, f_g]

        try:
            inv1 = cumulative_invariants_along_path(engine, path1)
            inv2 = cumulative_invariants_along_path(engine, path2)
        except KeyError as e:
            return False, f"Path construction failed: {e}"

        if inv1 != inv2:
            return False, (
                f"Square fails for {f_name}: "
                f"kept1={inv1['kept']} lost1={inv1['lost']} vs "
                f"kept2={inv2['kept']} lost2={inv2['lost']}"
            )

    return True, ""
