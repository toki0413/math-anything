import pytest

from math_anything.topology.curvature import (
    discrete_curvature,
    holonomy,
    riemannian_curvature_bridge,
)
from math_anything.topology.loop import Loop


def test_holonomy_of_flat_loop():
    loop = Loop(
        nodes=("A", "B", "A"),
        edges=("m1", "m2"),
        is_directed=True,
        canonical_form="A -> B -> A",
    )
    weights = {"m1": 0.0, "m2": 0.0}
    assert holonomy(loop, weights) == pytest.approx(1.0)


def test_discrete_curvature_of_lossy_loop():
    loop = Loop(
        nodes=("A", "B", "C", "A"),
        edges=("m1", "m2", "m3"),
        is_directed=True,
        canonical_form="A -> B -> C -> A",
    )
    weights = {"m1": 0.1, "m2": 0.1, "m3": 0.1}
    curvature = discrete_curvature(loop, weights)
    assert curvature > 0.0
    assert curvature < 1.0


def test_riemannian_curvature_bridge_flat_space():
    from math_anything.structures.geometry_riemannian import flat_metric

    metric = flat_metric(dim=2)
    curvature = riemannian_curvature_bridge(metric, {"x0": 1.0, "x1": 2.0}, reference=1.0)
    assert curvature == pytest.approx(0.0, abs=1e-5)
