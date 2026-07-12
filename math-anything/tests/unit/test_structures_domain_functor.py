import pytest

from math_anything.structures.domain_functor import (
    DomainFunctor,
    build_bridge_natural_transformation,
    build_domain_pair_engine,
    is_domain_natural_transformation,
)
from math_anything.structures.functor import NaturalTransformation


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


def test_identity_domain_functor_is_natural():
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
    )
    object_map = {"a_start": "a_start", "a_end": "a_end"}
    for link in engine.morphism_links:
        if link.source_structure.startswith("a_"):
            object_map[link.source_structure] = link.source_structure
            object_map[link.target_structure] = link.target_structure
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
    object_map = {"a_start": "a_start", "a_end": "a_end"}
    for link in engine.morphism_links:
        if link.source_structure.startswith("a_"):
            object_map[link.source_structure] = link.source_structure
            object_map[link.target_structure] = link.target_structure

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
