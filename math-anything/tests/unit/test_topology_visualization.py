from math_anything.categories.engine import CategoryEngine
from math_anything.topology.loop_engine import LoopEngine
from math_anything.topology.visualization import to_mermaid, to_graphviz


def _triangle_engine():
    ce = CategoryEngine()
    for name in ("ab", "bc", "ca"):
        ce.register_morphism(type(name.capitalize(), (), {"name": name})())
    ce.link("ab", "A", "B")
    ce.link("bc", "B", "C")
    ce.link("ca", "C", "A")
    return ce


def test_to_mermaid_contains_nodes_and_edges():
    ce = _triangle_engine()
    le = LoopEngine(ce)
    loops = le.find_loops()
    text = to_mermaid(ce, loops, curvature_map={})
    assert "graph TD" in text or "graph LR" in text
    assert "A" in text and "B" in text and "C" in text


def test_to_graphviz_contains_digraph():
    ce = _triangle_engine()
    le = LoopEngine(ce)
    loops = le.find_loops()
    text = to_graphviz(ce, loops, curvature_map={})
    assert "digraph" in text
    assert "A" in text and "B" in text
