import pytest

from math_anything.structures.domain_functor import (
    DomainFunctor,
    build_bridge_natural_transformation,
    build_domain_pair_engine,
    is_domain_natural_transformation,
)
from math_anything.structures.functor import NaturalTransformation


def _build_identity_object_map(engine: object, prefix: str) -> dict[str, str]:
    """Build an object map that sends every {prefix} structure to itself."""
    object_map: dict[str, str] = {}
    for link in engine.morphism_links:
        if link.source_structure.startswith(f"{prefix}_"):
            object_map[link.source_structure] = link.source_structure
            object_map[link.target_structure] = link.target_structure
    return object_map


def _build_cross_object_map(
    engine: object, source_prefix: str, target_prefix: str
) -> dict[str, str]:
    """Map every {source_prefix} structure to its {target_prefix} counterpart."""
    object_map: dict[str, str] = {}
    for link in engine.morphism_links:
        for struct in (link.source_structure, link.target_structure):
            if struct.startswith(f"{source_prefix}_"):
                object_map[struct] = struct.replace(
                    f"{source_prefix}_", f"{target_prefix}_", 1
                )
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
    # F is the identity functor on the a chain.
    object_map_f = _build_identity_object_map(engine, "a")
    morphism_map_f = {name: name for name in path_a}
    F = DomainFunctor(object_map_f, morphism_map_f)

    # G maps the a chain to the parallel b chain.
    object_map_g = _build_cross_object_map(engine, "a", "b")
    morphism_map_g = {a_name: b_name for a_name, b_name in zip(path_a, path_b)}
    G = DomainFunctor(object_map_g, morphism_map_g)

    # η_X: F(X)=a_X -> G(X)=b_X for every a structure X.
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
    # F is the identity functor on the a chain.
    object_map_f = _build_identity_object_map(engine, "a")
    morphism_map_f = {name: name for name in path_a}
    F = DomainFunctor(object_map_f, morphism_map_f)

    # G maps objects a -> b but collapses every morphism to a single b morphism,
    # breaking the naturality square.
    object_map_g = _build_cross_object_map(engine, "a", "b")
    G = DomainFunctor(object_map_g, {name: path_b[0] for name in path_a})

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
    """A cross-instance functor with matching cumulative invariants is natural."""
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {
            "input_dim": 2,
            "output_dim": 1,
            "architecture": "cnn",
            "loss": "cross_entropy",
        },
    )

    # F is the identity functor on the a chain.
    object_map_f = _build_identity_object_map(engine, "a")
    morphism_map_f = {name: name for name in path_a}
    F = DomainFunctor(object_map_f, morphism_map_f)

    # G maps the a chain to the parallel b chain (same domain, different params).
    object_map_g = _build_cross_object_map(engine, "a", "b")
    morphism_map_g = {
        a_name: b_name for a_name, b_name in zip(path_a, path_b)
    }
    G = DomainFunctor(object_map_g, morphism_map_g)

    # Preserve every invariant kept by the target chain so the bridge does not
    # change the cumulative invariant count on either side of the square.
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


def test_bad_morphism_mapping_returns_false_not_keyerror():
    """A functor mapping to an unregistered/unlinked morphism must return (False, str)."""
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
    )
    object_map_f = _build_identity_object_map(engine, "a")
    morphism_map_f = {name: name for name in path_a}
    F = DomainFunctor(object_map_f, morphism_map_f)

    object_map_g = _build_cross_object_map(engine, "a", "b")
    morphism_map_g = {a_name: b_name for a_name, b_name in zip(path_a, path_b)}
    # Map one source morphism to a name that is not registered or linked.
    morphism_map_g[path_a[0]] = "not_a_registered_morphism"
    G = DomainFunctor(object_map_g, morphism_map_g)

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
    assert "not_a_registered_morphism" in reason


def test_terminal_morphism_inherits_last_step_invariants():
    """Terminal morphisms must inherit the last real chain step's invariants_kept."""
    engine, path_a, _ = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
    )
    terminal_name = path_a[-1]
    terminal_morphism = engine.morphisms[terminal_name]
    # Last real step for supervised_learning is optimizer_step.
    assert terminal_morphism.invariants_kept == ["parameter_space"]


def test_build_domain_pair_engine_rejects_equal_prefixes():
    with pytest.raises(ValueError, match="must differ"):
        build_domain_pair_engine(
            "supervised_learning",
            {"input_dim": 2, "output_dim": 1},
            "supervised_learning",
            {"input_dim": 2, "output_dim": 1},
            prefix_a="same",
            prefix_b="same",
        )


def test_missing_bridge_component_is_not_natural():
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
    )
    # F is the identity functor on the a chain; G maps a -> b.
    object_map_f = _build_identity_object_map(engine, "a")
    morphism_map_f = {name: name for name in path_a}
    F = DomainFunctor(object_map_f, morphism_map_f)

    object_map_g = _build_cross_object_map(engine, "a", "b")
    morphism_map_g = {a_name: b_name for a_name, b_name in zip(path_a, path_b)}
    G = DomainFunctor(object_map_g, morphism_map_g)

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
