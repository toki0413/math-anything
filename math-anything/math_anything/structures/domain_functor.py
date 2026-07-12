"""Functors and natural transformations over domain morphism chains."""

from __future__ import annotations

from typing import Any

from math_anything.categories.engine import CategoryEngine
from math_anything.domains import DOMAIN_REGISTRY
from math_anything.structures.functor import Functor, NaturalTransformation
from math_anything.topology.homotopy import cumulative_invariants_along_path

__all__ = [
    "DomainFunctor",
    "build_domain_pair_engine",
    "build_bridge_natural_transformation",
    "is_domain_natural_transformation",
]


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


def _make_morphism(
    name: str,
    invariants_kept: list[str],
    invariants_lost: list[str],
) -> Any:
    """Return a simple morphism object with the required invariant attributes."""
    return type(
        "M",
        (),
        {
            "name": name,
            "invariants_kept": invariants_kept,
            "invariants_lost": invariants_lost,
        },
    )()


def _build_prefixed_chain(
    engine: CategoryEngine,
    chain_steps: list[dict[str, Any]],
    prefix: str,
) -> tuple[list[str], str]:
    """Register one prefixed domain chain in ``engine``.

    Returns the list of morphism names and the ``{prefix}_end`` object name
    (target of the terminal morphism).
    """
    path: list[str] = []
    prev = f"{prefix}_start"
    for i, step in enumerate(chain_steps):
        name = f"{prefix}_{step['name']}"
        kept = step.get("invariants_kept", [])
        engine.register_morphism(
            _make_morphism(
                name,
                invariants_kept=kept,
                invariants_lost=step.get("invariants_lost", []),
            )
        )
        target = f"{prefix}_state_{i}"
        engine.link(name, prev, target)
        path.append(name)
        prev = target
    # Terminal morphisms inherit the last real step's kept invariants so that
    # identity functors stay natural across the whole chain.
    terminal_kept: list[str] = list(chain_steps[-1].get("invariants_kept", [])) if chain_steps else []
    final = f"{prefix}_end"
    terminal = _make_morphism(
        f"{prefix}_terminal",
        invariants_kept=terminal_kept,
        invariants_lost=[],
    )
    engine.register_morphism(terminal)
    engine.link(f"{prefix}_terminal", prev, final)
    path.append(f"{prefix}_terminal")
    return path, final


def build_domain_pair_engine(
    domain_a_name: str,
    params_a: dict[str, Any],
    domain_b_name: str,
    params_b: dict[str, Any],
    prefix_a: str = "a",
    prefix_b: str = "b",
) -> tuple[CategoryEngine, list[str], list[str]]:
    """Build a merged CategoryEngine containing two domain morphism chains.

    Each chain ends with a terminal morphism that closes the path at the
    ``{prefix}_end`` object.  Terminal morphisms inherit the ``invariants_kept``
    of the last real chain step so that identity functors remain natural across
    the whole path.
    """
    if domain_a_name not in DOMAIN_REGISTRY or domain_b_name not in DOMAIN_REGISTRY:
        available = sorted(DOMAIN_REGISTRY.keys())
        raise KeyError(f"Unknown domain. Available: {available}")

    if prefix_a == prefix_b:
        raise ValueError("prefix_a and prefix_b must differ")

    dom_a = DOMAIN_REGISTRY[domain_a_name](params_a)
    dom_b = DOMAIN_REGISTRY[domain_b_name](params_b)

    engine = CategoryEngine()
    path_a, _ = _build_prefixed_chain(engine, dom_a.build_morphism_chain(), prefix_a)
    path_b, _ = _build_prefixed_chain(engine, dom_b.build_morphism_chain(), prefix_b)

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
    if source_prefix == target_prefix:
        raise ValueError("source_prefix and target_prefix must differ")

    components: dict[str, str] = {}
    linked_structures = {link.source_structure for link in engine.morphism_links}
    linked_structures.update(link.target_structure for link in engine.morphism_links)

    prefix_len = len(source_prefix) + 1  # include trailing '_'
    for src in sorted(linked_structures):
        if not src.startswith(f"{source_prefix}_"):
            continue
        dst = f"{target_prefix}_{src[prefix_len:]}"
        if dst not in linked_structures:
            continue
        name = f"bridge_{source_prefix}_to_{target_prefix}_{src}"
        engine.register_morphism(
            _make_morphism(
                name,
                invariants_kept=bridge_invariants_kept,
                invariants_lost=bridge_invariants_lost,
            )
        )
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

        try:
            f_f = F.map_morphism(f_name)
            f_g = G.map_morphism(f_name)
        except KeyError as e:
            return False, f"Functor does not map morphism: {e}"

        eta_src = eta.components.get(source_obj)
        eta_dst = eta.components.get(target_obj)

        if eta_src is None or eta_dst is None:
            return False, f"Missing bridge for {source_obj} or {target_obj}"

        path1 = [f_f, eta_dst]
        path2 = [eta_src, f_g]

        for path in (path1, path2):
            for name in path:
                if name not in links:
                    return False, f"Morphism '{name}' is not registered or linked"

        try:
            for label, path in (("η_Y ∘ F(f)", path1), ("G(f) ∘ η_X", path2)):
                for i in range(len(path) - 1):
                    current_target = links[path[i]].target_structure
                    next_source = links[path[i + 1]].source_structure
                    if current_target != next_source:
                        return False, (
                            f"{label} is disconnected: morphism '{path[i]}' ends at "
                            f"'{current_target}' but morphism '{path[i + 1]}' starts at "
                            f"'{next_source}'"
                        )

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
