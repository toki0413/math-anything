from math_anything.categories.engine import CategoryEngine
from math_anything.morphisms.approximations import (
    BornOppenheimerApproximation,
    KohnShamMapping,
    PlaneWaveTruncation,
)


def test_category_engine_loop_engine():
    ce = CategoryEngine()
    ce.register_morphism(BornOppenheimerApproximation())
    ce.register_morphism(KohnShamMapping())
    ce.register_morphism(PlaneWaveTruncation(encut=520))
    ce.link("born_oppenheimer", "FullManyBody", "ElectronicSchrodinger")
    ce.link("kohn_sham", "ElectronicSchrodinger", "KohnSham_Full")
    ce.link("plane_wave_truncation", "KohnSham_Full", "KohnSham_Truncated")

    le = ce.loop_engine
    assert le is not None
    assert ce.loop_engine is le  # cached property returns same instance
    loops = le.find_loops()
    assert len(loops) == 0  # DAG example
    betti = le.betti_numbers()
    assert betti["beta0"] == 1
    assert betti["beta1"] == 0
