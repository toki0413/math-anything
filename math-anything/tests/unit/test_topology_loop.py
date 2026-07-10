# tests/unit/test_topology_loop.py
"""Tests for Loop dataclass and LoopType enum."""

from math_anything.topology.loop import Loop, LoopType


def test_loop_creation():
    loop = Loop(
        nodes=("A", "B", "C", "A"),
        edges=("e1", "e2", "e3"),
        is_directed=True,
        canonical_form="A->B->C->A",
    )
    assert loop.nodes == ("A", "B", "C", "A")
    assert loop.edges == ("e1", "e2", "e3")
    assert loop.is_directed is True


def test_loop_type_enum():
    assert LoopType.CONVERGENCE.value == "convergence"
    assert LoopType.COUPLING.value == "coupling"
    assert LoopType.MIGRATION.value == "migration"
    assert LoopType.MULTISCALE.value == "multiscale"
    assert LoopType.UNKNOWN.value == "unknown"
