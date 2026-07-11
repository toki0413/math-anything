from math_anything.topology.cross_domain import cross_domain_homotopy


def test_cross_domain_homotopy_returns_witness():
    witness = cross_domain_homotopy(
        "dft",
        {"n_electrons": 2, "ecutwfc": 50.0},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1, "architecture": "mlp"},
    )
    assert witness.source == "ManyBodySchrodinger"
    assert "energy_conservation" in witness.shared_invariants or not witness.equivalent
