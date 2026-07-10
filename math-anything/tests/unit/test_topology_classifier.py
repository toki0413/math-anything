from math_anything.topology.classifier import LoopClassifier
from math_anything.topology.loop import Loop, LoopType


def test_classify_convergence_loop():
    loop = Loop(
        nodes=("V_eff", "density", "wavefunction", "V_eff"),
        edges=("poisson", "ks_solve", "mixer"),
        is_directed=True,
        canonical_form="V_eff->density->wavefunction->V_eff",
    )
    classifier = LoopClassifier()
    assert classifier.classify(loop) == LoopType.CONVERGENCE


def test_classify_coupling_loop():
    loop = Loop(
        nodes=("atomistic", "continuum", "atomistic"),
        edges=("concurrent_up", "concurrent_down"),
        is_directed=True,
        canonical_form="atomistic<->continuum",
    )
    classifier = LoopClassifier()
    assert classifier.classify(loop) == LoopType.COUPLING


def test_classify_unknown():
    loop = Loop(
        nodes=("A", "B", "A"),
        edges=("m1", "m2"),
        is_directed=True,
        canonical_form="A->B->A",
    )
    classifier = LoopClassifier()
    assert classifier.classify(loop) == LoopType.UNKNOWN
