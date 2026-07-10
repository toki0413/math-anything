# tests/unit/test_topology_imports.py
"""Smoke tests for the topology package scaffold."""


def test_topology_package_imports():
    from math_anything import topology

    assert hasattr(topology, "Loop")
    assert hasattr(topology, "LoopEngine")
    assert hasattr(topology, "LoopClassifier")


def test_top_level_imports():
    from math_anything import Loop, LoopEngine, LoopClassifier, LoopType

    assert Loop is not None
    assert LoopType.CONVERGENCE is not None
