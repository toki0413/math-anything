from math_anything.topology.cross_domain import cross_domain_homotopy


def test_cross_domain_homotopy_returns_witness():
    witness = cross_domain_homotopy(
        "dft",
        {"n_electrons": 2, "ecutwfc": 50.0},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1, "architecture": "mlp"},
    )
    assert witness.source == "ManyBodySchrodinger"
    assert isinstance(witness.source, str) and witness.source
    assert isinstance(witness.target, str) and witness.target
    assert 0.0 <= witness.confidence <= 1.0
    if witness.equivalent:
        assert len(witness.shared_invariants) > 0
    else:
        assert isinstance(witness.shared_invariants, list)
