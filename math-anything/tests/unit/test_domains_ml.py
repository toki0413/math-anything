from math_anything.domains import DOMAIN_REGISTRY


def test_supervised_learning_domain_registered():
    assert "supervised_learning" in DOMAIN_REGISTRY


def test_supervised_learning_analysis():
    domain = DOMAIN_REGISTRY["supervised_learning"](
        {
            "input_dim": 2,
            "output_dim": 1,
            "architecture": "mlp",
        }
    )
    analysis = domain.analyze()
    assert analysis.domain_name == "supervised_learning"
    assert "function_approximation" in analysis.conservation_field.get("equation_type", "")
    assert len(analysis.morphism_chain) > 0
