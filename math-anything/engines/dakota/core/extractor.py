"""Dakota mathematical structure extractor.

Extracts mathematical structures from Dakota uncertainty quantification simulations.
Focus: Bayesian inference, ensemble sampling, sensitivity analysis, optimization under uncertainty.
Mathematical structure: EnsembleProblem (BayesianInference).
"""

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from math_anything.core.harness import HarnessRegistry, MathAnythingHarness
from math_anything.schemas import (
    ComputationalEdge,
    ComputationalGraph,
    ComputationalNode,
    Discretization,
    GoverningEquation,
    MathematicalModel,
    MathSchema,
    MetaInfo,
    NumericalMethod,
    Solver,
    UpdateMode,
)


class DakotaExtractor(MathAnythingHarness):
    """Extracts mathematical structures from Dakota UQ simulations.

    Dakota (Design Analysis Kit for Optimization and Terascale Applications)
    provides a platform for uncertainty quantification, sensitivity analysis,
    parameter estimation, and optimization under uncertainty.

    Mathematical structure: EnsembleProblem (BayesianInference).
    File types: .in
    """

    SUPPORTED_EXTENSIONS = [".in"]

    @property
    def engine_name(self) -> str:
        return "dakota"

    @property
    def supported_schema_version(self) -> str:
        return "1.0.0"

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> MathSchema:
        options = options or {}
        method = options.get("method", "sampling")

        schema = MathSchema(
            meta=MetaInfo(
                extracted_by="math-anything-dakota",
                extractor_version="0.1.0",
            ),
            mathematical_model=self._build_mathematical_model(method),
            numerical_method=self._build_numerical_method(method),
            computational_graph=self._build_computational_graph(method),
        )
        return schema

    def _build_mathematical_model(self, method: str) -> MathematicalModel:
        model = MathematicalModel()

        model.governing_equations = [
            GoverningEquation(
                id="bayes_theorem",
                type="statistical_inference",
                name="Bayes' Theorem",
                mathematical_form="p(theta|D) = p(D|theta) * p(theta) / p(D)",
                variables=["posterior", "likelihood", "prior", "evidence"],
                parameters={"form": "probability_density"},
                description="Bayesian inference: posterior updates prior with likelihood of observed data",
            ),
            GoverningEquation(
                id="multivariate_normal_prior",
                type="probability_distribution",
                name="Multivariate Normal Prior Distribution",
                mathematical_form="p(theta) = (2*pi)^(-k/2) * |Sigma|^(-1/2) * exp(-0.5*(theta-mu)^T Sigma^(-1) (theta-mu))",
                variables=["parameters", "mean_vector", "covariance_matrix"],
                parameters={"dimension": "k", "conjugate": False},
                description="Multivariate normal prior over uncertain parameters",
            ),
        ]

        if method == "sampling":
            model.governing_equations.append(
                GoverningEquation(
                    id="monte_carlo_estimator",
                    type="statistical_estimator",
                    name="Monte Carlo Statistical Estimator",
                    mathematical_form="E[f(theta)] ≈ (1/N) * sum_{i=1}^{N} f(theta_i),  theta_i ~ p(theta)",
                    variables=["estimator", "samples", "quantity_of_interest"],
                    parameters={"convergence": "O(1/sqrt(N))", "bias": "unbiased"},
                    description="Monte Carlo estimator for statistical moments of QoI",
                )
            )

        if method == "polynomial_chaos":
            model.governing_equations.append(
                GoverningEquation(
                    id="pce_expansion",
                    type="spectral_expansion",
                    name="Polynomial Chaos Expansion (PCE)",
                    mathematical_form="f(xi) = sum_{alpha} c_alpha * Psi_alpha(xi)",
                    variables=["expansion_coefficients", "orthogonal_polynomials", "random_variables"],
                    parameters={"basis": "hermite_legendre", "truncation": "total_order"},
                    description="Spectral expansion of stochastic response in orthogonal polynomial basis",
                )
            )

        if method == "stochastic_collocation":
            model.governing_equations.append(
                GoverningEquation(
                    id="collocation_interpolant",
                    type="interpolation_approximation",
                    name="Stochastic Collocation Interpolant",
                    mathematical_form="f(xi) ≈ sum_{j} f(xi_j) * L_j(xi)",
                    variables=["interpolant", "collocation_points", "lagrange_basis"],
                    parameters={"grid": "sparse_gauss", "dimension": "adaptive"},
                    description="Interpolation-based surrogate on sparse grid collocation points",
                )
            )

        if method == "gaussian_process":
            model.governing_equations.append(
                GoverningEquation(
                    id="gp_surrogate",
                    type="surrogate_model",
                    name="Gaussian Process (Kriging) Surrogate",
                    mathematical_form="f(x) ~ GP(m(x), k(x, x')), k(x, x') = sigma^2 * exp(-0.5*(x-x')^T Theta^(-2) (x-x'))",
                    variables=["mean_function", "kernel_hyperparameters", "variance"],
                    parameters={"kernel": "squared_exponential", "noise": "nugget"},
                    description="Gaussian process surrogate with squared exponential kernel",
                )
            )

        if method in ("optimization", "optimization_under_uncertainty"):
            model.governing_equations.append(
                GoverningEquation(
                    id="optimization_problem",
                    type="optimization_problem",
                    name="Uncertainty-Aware Optimization",
                    mathematical_form="min_{theta} J(theta, xi),  s.t. c(theta, xi) ≤ 0,  xi ~ p(xi)",
                    variables=["objective", "constraints", "uncertain_parameters"],
                    parameters={"formulation": "stochastic" if method == "optimization_under_uncertainty" else "deterministic"},
                    description="Optimization with or without explicit uncertainty quantification",
                )
            )

        if method == "sensitivity":
            model.governing_equations.append(
                GoverningEquation(
                    id="sobol_indices",
                    type="sensitivity_analysis",
                    name="Sobol' Variance-Based Sensitivity Indices",
                    mathematical_form="S_i = Var[E[f|theta_i]] / Var[f],  S_Ti = 1 - Var[E[f|theta_{-i}]] / Var[f]",
                    variables=["first_order_index", "total_effect_index", "conditional_expectation"],
                    parameters={"decomposition": "ANOVA", "convergence": "asymptotic"},
                    description="Variance-based global sensitivity analysis via Sobol' decomposition",
                )
            )

        if method == "calibration":
            model.governing_equations.append(
                GoverningEquation(
                    id="likelihood_function",
                    type="statistical_inference",
                    name="Gaussian Likelihood Function",
                    mathematical_form="p(D|theta) = (2*pi*sigma^2)^(-N/2) * exp(-0.5*sum (D_i - M(theta))^2 / sigma^2)",
                    variables=["data", "model_prediction", "measurement_noise"],
                    parameters={"noise_model": "iid_gaussian"},
                    description="Gaussian likelihood for model-data misfit in parameter estimation",
                )
            )

        return model

    def _build_numerical_method(self, method: str) -> NumericalMethod:
        nm = NumericalMethod()
        nm.solver = Solver(
            algorithm=method,
            convergence_criterion="statistical_convergence",
        )
        return nm

    def _build_computational_graph(self, method: str) -> ComputationalGraph:
        graph = ComputationalGraph()

        graph.add_node(ComputationalNode(
            id="generate_samples",
            type="sample",
            math_semantics={
                "operator_type": method if method == "sampling" else "parameter_init",
                "updates": {"target": "theta_samples", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        if method in ("polynomial_chaos", "gaussian_process", "stochastic_collocation"):
            graph.add_node(ComputationalNode(
                id="build_surrogate",
                type="construct",
                math_semantics={
                    "operator_type": f"{method}_surrogate",
                    "updates": {"target": "surrogate_model", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))

        graph.add_node(ComputationalNode(
            id="evaluate_model",
            type="compute",
            math_semantics={
                "operator_type": "black_box_evaluation",
                "updates": {"target": "qoi", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="post_process",
            type="analyze",
            math_semantics={
                "operator_type": "statistical_postprocess",
                "updates": {"target": "moments_indices", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        if method == "calibration":
            graph.add_node(ComputationalNode(
                id="mcmc_sampler",
                type="sample",
                math_semantics={
                    "operator_type": "mcmc_sampler",
                    "updates": {"target": "posterior_samples", "mode": UpdateMode.IMPLICIT_LOOP.value},
                    "convergence": {"required": True, "criterion": "gelman_rubin"},
                },
            ))

        if method in ("optimization", "optimization_under_uncertainty"):
            graph.add_node(ComputationalNode(
                id="optimizer",
                type="optimize",
                math_semantics={
                    "operator_type": "gradient_based_optimizer",
                    "updates": {"target": "theta_optimal", "mode": UpdateMode.IMPLICIT_LOOP.value},
                    "convergence": {"required": True, "criterion": "objective_gradient_norm"},
                },
            ))

        graph.add_edge(ComputationalEdge(
            from_node="generate_samples", to_node="evaluate_model",
            data_type="parameter_set", dependency="evaluate",
        ))
        graph.add_edge(ComputationalEdge(
            from_node="evaluate_model", to_node="post_process",
            data_type="qoi_values", dependency="compute",
        ))

        graph.execution_topology = {
            "schedule": "ensemble_parallel",
            "implicit_loops": [
                {
                    "loop_id": method,
                    "nested_in": "evaluate_model",
                    "convergence_guarantee": "statistical_threshold",
                    "max_iterations_source": "input",
                }
            ] if method in ("calibration", "optimization", "optimization_under_uncertainty") else [],
        }
        return graph

    def list_extractable_objects(self) -> List[str]:
        return ["governing_equations", "numerical_method", "computational_graph", "uncertainty_quantification"]

    def get_supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTENSIONS


HarnessRegistry.register(DakotaExtractor)
