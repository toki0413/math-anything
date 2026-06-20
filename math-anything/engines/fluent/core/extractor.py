"""ANSYS Fluent mathematical structure extractor.

Extracts mathematical structures from ANSYS Fluent CFD simulations.
Focus: Conservation laws (mass, momentum, energy), pressure-based and density-based solvers.
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


class FluentExtractor(MathAnythingHarness):
    """Extracts mathematical structures from ANSYS Fluent CFD simulations.

    Fluent solves conservation laws using finite volume methods with cell-centered or node-based schemes.
    Mathematical structure: ConservationLawSystem.

    File types: .cas, .dat, .msh
    Methods: pressure-based, density-based solvers
    """

    SUPPORTED_EXTENSIONS = [".cas", ".dat", ".msh"]

    @property
    def engine_name(self) -> str:
        return "fluent"

    @property
    def supported_schema_version(self) -> str:
        return "1.0.0"

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> MathSchema:
        options = options or {}
        solver = options.get("solver_type", "pressure_based")

        schema = MathSchema(
            meta=MetaInfo(
                extracted_by="math-anything-fluent",
                extractor_version="0.1.0",
            ),
            mathematical_model=self._build_mathematical_model(solver),
            numerical_method=self._build_numerical_method(solver),
            computational_graph=self._build_computational_graph(solver),
        )
        return schema

    def _build_mathematical_model(self, solver: str) -> MathematicalModel:
        model = MathematicalModel()

        model.governing_equations = [
            GoverningEquation(
                id="mass_conservation",
                type="conservation_law",
                name="Mass Conservation",
                mathematical_form="d(rho)/dt + div(rho * U) = 0",
                variables=["density", "velocity"],
                parameters={"form": "continuity"},
                description="General mass conservation equation (compressible form)",
            ),
            GoverningEquation(
                id="momentum_conservation",
                type="conservation_law",
                name="Momentum Conservation (Navier-Stokes)",
                mathematical_form="d(rho*U)/dt + div(rho*UU) = -grad(p) + div(tau) + rho*g + F",
                variables=["velocity", "pressure", "stress_tensor", "body_force"],
                parameters={"form": "reynolds_averaged"},
                description="Reynolds-averaged momentum conservation in conservative form",
            ),
            GoverningEquation(
                id="energy_conservation",
                type="conservation_law",
                name="Energy Conservation",
                mathematical_form="d(rho*E)/dt + div(U*(rho*E + p)) = div(k_eff*grad(T) + tau_eff*U) + S_h",
                variables=["total_energy", "temperature", "velocity", "pressure"],
                parameters={"form": "total_energy"},
                description="Total energy conservation with viscous dissipation",
            ),
        ]

        if solver == "pressure_based":
            model.governing_equations.append(
                GoverningEquation(
                    id="pressure_correction",
                    type="elliptic_equation",
                    name="Pressure Correction Equation",
                    mathematical_form="div((1/a_p) * grad(p')) = div(U*)",
                    variables=["pressure_correction", "predicted_velocity", "momentum_coefficient"],
                    parameters={"algorithm": "SIMPLE_PISO"},
                    description="Pressure correction from discretized continuity-momentum coupling",
                )
            )
        elif solver == "density_based":
            model.governing_equations.append(
                GoverningEquation(
                    id="flux_splitting",
                    type="hyperbolic_system",
                    name="Riemann-Based Flux Splitting",
                    mathematical_form="F = (F_L + F_R)/2 - 0.5*|A|*(W_R - W_L)",
                    variables=["flux", "conservative_variables", "jacobian"],
                    parameters={"scheme": "Roe_AUSM"},
                    description="Upwind flux splitting based on characteristic decomposition",
                )
            )

        return model

    def _build_numerical_method(self, solver: str) -> NumericalMethod:
        method = NumericalMethod()
        method.discretization = Discretization(
            space_discretization="finite_volume_cell_centered",
            time_integrator="implicit" if solver == "pressure_based" else "runge_kutta_explicit",
        )
        method.solver = Solver(
            algorithm="SIMPLE" if solver == "pressure_based" else "coupled_explicit",
            convergence_criterion="residual_norm",
        )
        return method

    def _build_computational_graph(self, solver: str) -> ComputationalGraph:
        graph = ComputationalGraph()

        if solver == "pressure_based":
            graph.add_node(ComputationalNode(
                id="momentum_solve",
                type="solve",
                math_semantics={
                    "operator_type": "segregated_momentum",
                    "updates": {"target": "U_star", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))
            graph.add_node(ComputationalNode(
                id="pressure_solve",
                type="solve",
                math_semantics={
                    "operator_type": "pressure_correction_solve",
                    "updates": {"target": "p_prime", "mode": UpdateMode.IMPLICIT_LOOP.value},
                    "convergence": {"required": True, "criterion": "mass_imbalance"},
                },
            ))
            graph.add_node(ComputationalNode(
                id="flux_correct",
                type="update",
                math_semantics={
                    "operator_type": "velocity_pressure_correction",
                    "updates": {"target": "U", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))
        else:
            graph.add_node(ComputationalNode(
                id="coupled_solve",
                type="solve",
                math_semantics={
                    "operator_type": "coupled_conservation",
                    "updates": {"target": "W", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))

        graph.add_node(ComputationalNode(
            id="turbulence_solve",
            type="solve",
            math_semantics={
                "operator_type": "turbulence_transport",
                "updates": {"target": "k_epsilon", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="energy_solve",
            type="solve",
            math_semantics={
                "operator_type": "energy_transport",
                "updates": {"target": "T", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        return graph

    def list_extractable_objects(self) -> List[str]:
        return ["governing_equations", "numerical_method", "computational_graph"]

    def get_supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTENSIONS


HarnessRegistry.register(FluentExtractor)
