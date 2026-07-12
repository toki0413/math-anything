"""Cross-domain homotopy checking for Bourbaki domains."""

from __future__ import annotations

from typing import Any

from math_anything.categories.engine import CategoryEngine
from math_anything.domains import DOMAIN_REGISTRY
from math_anything.topology.homotopy import HomotopyWitness, are_paths_homotopic


def cross_domain_homotopy(
    domain_a_name: str,
    params_a: dict[str, Any],
    domain_b_name: str,
    params_b: dict[str, Any],
) -> HomotopyWitness:
    """Check whether two domain instantiation paths are homotopic.

    Builds a merged CategoryEngine from both domain morphism chains and compares
    the cumulative invariants along each path.
    """
    if domain_a_name not in DOMAIN_REGISTRY or domain_b_name not in DOMAIN_REGISTRY:
        available = sorted(DOMAIN_REGISTRY.keys())
        raise KeyError(f"Unknown domain. Available: {available}")

    dom_a = DOMAIN_REGISTRY[domain_a_name](params_a)
    dom_b = DOMAIN_REGISTRY[domain_b_name](params_b)

    chain_a = dom_a.build_morphism_chain()
    chain_b = dom_b.build_morphism_chain()

    ce = CategoryEngine()
    path_a: list[str] = []
    path_b: list[str] = []

    prev_a = "ManyBodySchrodinger" if domain_a_name == "dft" else f"{domain_a_name}_start"
    for i, step in enumerate(chain_a):
        name = f"a_{step['name']}"
        ce.register_morphism(
            type(
                "M",
                (),
                {
                    "name": name,
                    "invariants_kept": step.get("invariants_kept", []),
                    "invariants_lost": step.get("invariants_lost", []),
                },
            )()
        )
        target = f"a_state_{i}"
        ce.link(name, prev_a, target)
        path_a.append(name)
        prev_a = target

    prev_b = f"{domain_b_name}_start"
    for i, step in enumerate(chain_b):
        name = f"b_{step['name']}"
        ce.register_morphism(
            type(
                "M",
                (),
                {
                    "name": name,
                    "invariants_kept": step.get("invariants_kept", []),
                    "invariants_lost": step.get("invariants_lost", []),
                },
            )()
        )
        target = f"b_state_{i}"
        ce.link(name, prev_b, target)
        path_b.append(name)
        prev_b = target

    return are_paths_homotopic(ce, path_a, path_b)
