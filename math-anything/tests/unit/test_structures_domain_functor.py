import pytest

from math_anything.structures.domain_functor import (
    DomainFunctor,
    build_bridge_natural_transformation,
    build_domain_pair_engine,
    is_domain_natural_transformation,
)
from math_anything.structures.functor import NaturalTransformation


def _build_identity_object_map(engine: object, path: list[str], prefix: str) -> dict[str, str]:
    """Build an object map that sends every {prefix} structure to itself."""
    object_map: dict[str, str] = {}
    for link in engine.morphism_links:
        if link.source_structure.startswith(f"{prefix}_"):
            object_map[link.source_structure] = link.source_structure
            object_map[link.target_structure] = link.target_structure
    return object_map


def test_build_domain_pair_engine_creates_two_paths():
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
    )
    assert len(path_a) == len(path_b)
    assert all(name.startswith("a_") for name in path_a)
    assert all(name.startswith("b_") for name in path_b)
    assert engine.morphism_links[0].source_structure == "a_start"


def test_build_domain_pair_engine_unknown_domain_raises():
    with pytest.raises(KeyError):
        build_domain_pair_engine(
            "not_a_domain",
            {},
            "supervised_learning",
            {"input_dim": 2, "output_dim": 1},
        )


def test_identity_domain_functor_is_natural():
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
    )
    object_map = _build_identity_object_map(engine, path_a, "a")
    morphism_map = {name: name for name in path_a}
    F = DomainFunctor(object_map, morphism_map)
    G = DomainFunctor(object_map, morphism_map)

    eta = build_bridge_natural_transformation(
        engine,
        source_prefix="a",
        target_prefix="b",
        bridge_invariants_kept=["parameter_space"],
        bridge_invariants_lost=[],
    )

    valid, reason = is_domain_natural_transformation(
        F, G, eta, engine, test_morphisms=path_a
    )
    assert valid, reason


def test_mismatched_functor_is_not_natural():
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
    )
    object_map = _build_identity_object_map(engine, path_a, "a")

    F = DomainFunctor(object_map, {name: name for name in path_a})
    # G maps every a morphism to the first b morphism, which breaks the square.
    G = DomainFunctor(object_map, {name: path_b[0] for name in path_a})
    eta = build_bridge_natural_transformation(
        engine,
        source_prefix="a",
        target_prefix="b",
        bridge_invariants_kept=["parameter_space"],
        bridge_invariants_lost=[],
    )

    valid, reason = is_domain_natural_transformation(
        F, G, eta, engine, test_morphisms=path_a
    )
    assert not valid


def test_cross_domain_functor_is_natural():
    """A functor from supervised_learning to dft plus a bridge is natural."""
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "dft",
        {"n_atoms": 1, "n_electrons": 1, "k_points": [1, 1, 1]},
    )

    # Map each a structure to the corresponding b structure (both chains are
    # long enough that the first len(path_a) b objects cover the a chain).
    object_map: dict[str, str] = {}
    for link in engine.morphism_links:
        if link.source_structure.startswith("a_"):
            object_map[link.source_structure] = link.source_structure.replace("a_", "b_", 1)
            object_map[link.target_structure] = link.target_structure.replace("a_", "b_", 1)

    # Map each a morphism to the corresponding b morphism in order.
    morphism_map = {
        a_name: b_name for a_name, b_name in zip(path_a, path_b)
    }
    F = DomainFunctor(object_map, morphism_map)
    G = DomainFunctor(object_map, morphism_map)

    # Bridges from a structures to their b counterparts.  Preserve every
    # invariant kept by the target chain so the bridge does not change the
    # cumulative invariant count on either side of the naturality square.
    target_kept = {
        inv
        for name in path_b
        for inv in getattr(engine.morphisms[name], "invariants_kept", [])
    }
    eta = build_bridge_natural_transformation(
        engine,
        source_prefix="a",
        target_prefix="b",
        bridge_invariants_kept=sorted(target_kept),
        bridge_invariants_lost=[],
    )

    valid, reason = is_domain_natural_transformation(
        F, G, eta, engine, test_morphisms=path_a
    )
    assert valid, reason


def test_missing_bridge_component_is_not_natural():
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
    )
    object_map = _build_identity_object_map(engine, path_a, "a")
    morphism_map = {name: name for name in path_a}
    F = DomainFunctor(object_map, morphism_map)
    G = DomainFunctor(object_map, morphism_map)

    full_eta = build_bridge_natural_transformation(
        engine,
        source_prefix="a",
        target_prefix="b",
        bridge_invariants_kept=["parameter_space"],
        bridge_invariants_lost=[],
    )
    # Drop the start component so the first test morphism lacks a bridge.
    partial_components = dict(full_eta.components)
    del partial_components["a_start"]
    eta = NaturalTransformation(partial_components)

    valid, reason = is_domain_natural_transformation(
        F, G, eta, engine, test_morphisms=path_a
    )
    assert not valid
    assert "a_start" in reason
