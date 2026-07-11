"""Machine-learning domain as a Bourbaki instantiation."""

from __future__ import annotations

from typing import Any

from math_anything.domains import register_domain
from math_anything.domains.base import Domain


@register_domain("supervised_learning")
class SupervisedLearningDomain(Domain):
    """Supervised learning as a morphism chain over function spaces."""

    name = "supervised_learning"
    description = "Supervised learning — function approximation from data"
    equation_type = "function_approximation"
    default_params = {
        "input_dim": 2,
        "output_dim": 1,
        "architecture": "mlp",
        "loss": "mse",
        "activation": "relu",
    }

    def build_conservation_field(self) -> dict[str, Any]:
        return {
            "equation_type": "function_approximation",
            "conservation_laws": [
                "expected_risk_minimization",
                "gradient_flow",
            ],
            "symmetries": ["data_permutation_invariance"],
            "eigenvalues": [],
        }

    def build_morphism_chain(self) -> list[dict[str, Any]]:
        chain = [
            {
                "name": "data_sampling",
                "type": "restriction",
                "description": "Replace true distribution with finite dataset",
                "invariants_kept": ["empirical_risk"],
                "invariants_lost": ["true_risk", "population_distribution"],
                "invariants_introduced": ["finite_sample_noise", "generalization_gap"],
            },
            {
                "name": "feature_map",
                "type": "embedding",
                "description": "Embed raw inputs into representation space",
                "invariants_kept": ["input_topology"],
                "invariants_lost": ["raw_feature_semantics"],
                "invariants_introduced": ["learned_representation"],
            },
            {
                "name": f"model_{self.params.get('architecture', 'mlp')}",
                "type": "surrogate",
                "description": "Parametric function family approximating the target",
                "invariants_kept": ["differentiability"],
                "invariants_lost": ["true_target_function"],
                "invariants_introduced": ["approximation_error", "optimization_landscape"],
            },
            {
                "name": f"loss_{self.params.get('loss', 'mse')}",
                "type": "projection",
                "description": "Project predictions and targets onto scalar objective",
                "invariants_kept": ["differentiability"],
                "invariants_lost": ["full_prediction_state"],
                "invariants_introduced": ["gradient_direction"],
            },
            {
                "name": "optimizer_step",
                "type": "transformation",
                "description": "Update parameters along gradient direction",
                "invariants_kept": ["parameter_space"],
                "invariants_lost": ["exact_minimum"],
                "invariants_introduced": ["learning_rate_dependence", "convergence_dynamics"],
            },
        ]
        return chain
