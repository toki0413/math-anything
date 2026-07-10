import pytest

from math_anything.categories.engine import CategoryEngine
from math_anything.topology.loop import LoopType
from math_anything.topology.loop_engine import LoopEngine


@pytest.fixture
def triangle_engine():
    ce = CategoryEngine()
    ce.register_morphism(type("m1", (), {"name": "m1"})())
    ce.register_morphism(type("m2", (), {"name": "m2"})())
    ce.register_morphism(type("m3", (), {"name": "m3"})())
    ce.link("m1", "A", "B")
    ce.link("m2", "B", "C")
    ce.link("m3", "C", "A")
    return ce


def test_build_graph(triangle_engine):
    le = LoopEngine(triangle_engine)
    g = le.build_graph()
    assert set(g.nodes()) == {"A", "B", "C"}
    assert g.has_edge("A", "B")
    assert g.has_edge("B", "C")
    assert g.has_edge("C", "A")


def test_find_loops(triangle_engine):
    le = LoopEngine(triangle_engine)
    loops = le.find_loops()
    assert len(loops) == 1
    assert loops[0].nodes[0] == loops[0].nodes[-1]


def test_betti_numbers_triangle(triangle_engine):
    le = LoopEngine(triangle_engine)
    betti = le.betti_numbers()
    assert betti["beta0"] == 1
    assert betti["beta1"] == 1


def test_betti_numbers_disconnected():
    ce = CategoryEngine()
    ce.register_morphism(type("m1", (), {"name": "m1"})())
    ce.register_morphism(type("m2", (), {"name": "m2"})())
    ce.link("m1", "A", "B")
    ce.link("m2", "C", "D")
    le = LoopEngine(ce)
    betti = le.betti_numbers()
    assert betti["beta0"] == 2
    assert betti["beta1"] == 0
