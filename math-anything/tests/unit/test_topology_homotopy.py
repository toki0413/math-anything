import pytest

from math_anything.categories.engine import CategoryEngine
from math_anything.topology.homotopy import HomotopyWitness, are_paths_homotopic


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
