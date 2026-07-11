from math_anything.categories.engine import CategoryEngine
from math_anything.topology.homotopy import (
    HomotopyWitness,
    are_paths_homotopic,
    cumulative_invariants_along_path,
)


def _make_morphism(name, source, target, kept=None, lost=None):
    kept = kept or []
    lost = lost or []
    return type(
        "Morphism",
        (),
        {
            "name": name,
            "source_type": source,
            "target_type": target,
            "invariants_kept": kept,
            "invariants_lost": lost,
        },
    )()


def test_homotopic_paths_share_invariants():
    ce = CategoryEngine()
    ce.register_morphism(_make_morphism("bo", "A", "B", kept=["energy"], lost=["nuclear_qm"]))
    ce.register_morphism(_make_morphism("ks", "B", "C", kept=["energy", "density"], lost=["correlation"]))
    ce.register_morphism(_make_morphism("pw", "C", "D", kept=["energy"], lost=["completeness"]))
    ce.register_morphism(_make_morphism("alt", "A", "X", kept=["energy"], lost=["nuclear_qm"]))
    ce.register_morphism(_make_morphism("alt2", "X", "D", kept=["energy"], lost=["correlation", "completeness"]))
    ce.link("bo", "A", "B")
    ce.link("ks", "B", "C")
    ce.link("pw", "C", "D")
    ce.link("alt", "A", "X")
    ce.link("alt2", "X", "D")

    witness = are_paths_homotopic(ce, ["bo", "ks", "pw"], ["alt", "alt2"])
    assert witness.equivalent is True
    assert "energy" in witness.shared_invariants


def test_non_homotopic_paths_differ():
    ce = CategoryEngine()
    ce.register_morphism(_make_morphism("m1", "A", "B", kept=["energy"], lost=[]))
    ce.register_morphism(_make_morphism("m2", "B", "C", kept=["energy"], lost=["momentum"]))
    ce.register_morphism(_make_morphism("m3", "A", "C", kept=["energy", "momentum"], lost=[]))
    ce.link("m1", "A", "B")
    ce.link("m2", "B", "C")
    ce.link("m3", "A", "C")

    witness = are_paths_homotopic(ce, ["m1", "m2"], ["m3"])
    assert witness.equivalent is False


def test_empty_path_returns_not_equivalent():
    ce = CategoryEngine()
    ce.register_morphism(_make_morphism("m1", "A", "B", kept=["energy"], lost=[]))
    ce.link("m1", "A", "B")

    witness = are_paths_homotopic(ce, [], ["m1"])
    assert witness.equivalent is False
    assert witness.path_a == ()
    assert witness.path_b == ("m1",)

    witness = are_paths_homotopic(ce, ["m1"], [])
    assert witness.equivalent is False
    assert witness.path_a == ("m1",)
    assert witness.path_b == ()

    witness = are_paths_homotopic(ce, [], [])
    assert witness.equivalent is False


def test_unlinked_morphism_raises_value_error():
    ce = CategoryEngine()
    ce.register_morphism(_make_morphism("m1", "A", "B", kept=["energy"], lost=[]))
    ce.register_morphism(_make_morphism("m2", "B", "C", kept=["energy"], lost=[]))
    ce.register_morphism(_make_morphism("m3", "B", "C", kept=["energy"], lost=[]))
    ce.link("m1", "A", "B")
    ce.link("m2", "B", "C")
    # m3 is registered but not linked

    try:
        are_paths_homotopic(ce, ["m1", "m2"], ["m1", "m3"])
        raise AssertionError("Expected ValueError for unlinked morphism")
    except ValueError as exc:
        assert "m3" in str(exc)


def test_source_target_mismatch_returns_not_equivalent():
    ce = CategoryEngine()
    ce.register_morphism(_make_morphism("m1", "A", "B", kept=["energy"], lost=[]))
    ce.register_morphism(_make_morphism("m2", "B", "C", kept=["energy"], lost=[]))
    ce.register_morphism(_make_morphism("m3", "A", "D", kept=["energy"], lost=[]))
    ce.link("m1", "A", "B")
    ce.link("m2", "B", "C")
    ce.link("m3", "A", "D")

    witness = are_paths_homotopic(ce, ["m1", "m2"], ["m3"])
    assert witness.equivalent is False


def test_cumulative_invariants_unregistered_morphism_raises_key_error():
    ce = CategoryEngine()
    try:
        cumulative_invariants_along_path(ce, ["missing"])
        raise AssertionError("Expected KeyError for unregistered morphism")
    except KeyError as exc:
        assert "missing" in str(exc)


def test_cumulative_invariants_returns_serializable_lists():
    ce = CategoryEngine()
    ce.register_morphism(_make_morphism("m1", "A", "B", kept=["energy"], lost=["momentum"]))
    result = cumulative_invariants_along_path(ce, ["m1"])
    assert result == {"kept": ["energy"], "lost": ["momentum"]}
    assert isinstance(result["kept"], list)
    assert isinstance(result["lost"], list)


def test_same_kept_different_lost_not_equivalent_and_confidence_below_one():
    ce = CategoryEngine()
    ce.register_morphism(_make_morphism("m1", "A", "B", kept=["energy"], lost=["spin"]))
    ce.register_morphism(_make_morphism("m2", "B", "C", kept=["energy"], lost=[]))
    ce.register_morphism(_make_morphism("alt", "A", "C", kept=["energy"], lost=["density"]))
    ce.link("m1", "A", "B")
    ce.link("m2", "B", "C")
    ce.link("alt", "A", "C")

    witness = are_paths_homotopic(ce, ["m1", "m2"], ["alt"])
    assert witness.equivalent is False
    assert witness.confidence < 1.0


def test_identical_kept_and_lost_sets_are_equivalent_with_confidence_one():
    ce = CategoryEngine()
    ce.register_morphism(_make_morphism("m1", "A", "B", kept=["energy"], lost=["spin"]))
    ce.register_morphism(_make_morphism("m2", "B", "C", kept=["energy"], lost=["density"]))
    ce.register_morphism(
        _make_morphism("alt", "A", "C", kept=["energy"], lost=["spin", "density"])
    )
    ce.link("m1", "A", "B")
    ce.link("m2", "B", "C")
    ce.link("alt", "A", "C")

    witness = are_paths_homotopic(ce, ["m1", "m2"], ["alt"])
    assert witness.equivalent is True
    assert witness.confidence == 1.0
