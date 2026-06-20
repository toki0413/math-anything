"""SU2 mathematical structure extractor.

Extracts mathematical structures from SU2 (Stanford University Unstructured) CFD simulations.
Focus: Conservation laws with adjoint-based sensitivity analysis and optimization.
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


class SU2Extractor(MathAnythingHarness):
    """Extracts mathematical structures from SU2 CFD simulations.

    SU2 is an open-source collection of C++ tools for multi-physics simulation
    and design optimization. It uses unstructured grids with finite volume and
    discontinuous Galerkin methods.

    Mathematical structure: ConservationLawSystem with adjoint.
    Special capability: adjoint-based optimization.

    File types: .cfg
    """

    SUPPORTED_EXTENSIONS = [".cfg"]

    @property
    def engine_name(self) -> str:
        return "su2"

    @property
    def supported_schema_version(self) -> str:
        return "1.0.0"

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> MathSchema:
        options = options or {}
        mode = options.get("mode", "direct")

        schema = MathSchema(
            meta=MetaInfo(
                extracted_by="math-anything-su2",
                extractor_version="0.1.0",
            ),
            mathematical_model=self._build_mathematical_model(mode),
            numerical_method=self._build_numerical_method(mode),
            computational_graph=self._build_computational_graph(mode),
        )
        return schema

    def _build_mathematical_model(self, mode: str) -> MathematicalModel:
        model = MathematicalModel()

        model.governing_equations = [
            GoverningEquation(
                id="conservation",
                type="conservation_law",
                name="Compressible Navier-Stokes Conservation",
                mathematical_form="dq/dt + div F_c(q) - div F_v(q, grad q) = S",
                variables=["conservative_variables", "convective_flux", "viscous_flux", "source"],
                parameters={"form": "integral_form", "domain": "unstructured"},
                description="Compressible Navier-Stokes equations in conservative integral form",
            ),
            GoverningEquation(
                id="turbulence_conservation",
                type="conservation_law",
                name="Turbulence Model Equations",
                mathematical_form="dq_t/dt + div F_c(q_t) - div F_v(q_t) = S_t",
                variables=["turbulent_variables", "turbulent_flux", "turbulent_source"],
                parameters={"model": "SA_SST"},
                description="Spalart-Allmaras or SST turbulence model transport",
            ),
        ]

        if mode == "adjoint":
            model.governing_equations.extend([
                GoverningEquation(
                    id="adjoint_continuity",
                    type="adjoint_equation",
                    name="Continuous Adjoint Continuity Equation",
                    mathematical_form="(dR/dw)^T * psi = (dJ/dw)^T",
                    variables=["adjoint_variable", "jacobian", "objective_sensitivity"],
                    parameters={"type": "continuous_adjoint"},
                    description="Continuous adjoint for flow residual sensitivity",
                ),
                GoverningEquation(
                    id="adjoint_momentum",
                    type="adjoint_equation",
                    name="Continuous Adjoint Momentum Equation",
                    mathematical_form="(dR_mom/dw)^T * psi = (dJ_mom/dw)^T",
                    variables=["adjoint_momentum", "flow_jacobian", "force_sensitivity"],
                    parameters={"type": "continuous_adjoint"},
                    description="Continuous adjoint momentum for force-based objectives",
                ),
                GoverningEquation(
                    id="surface_sensitivity",
                    type="sensitivity_equation",
                    name="Surface Sensitivity (Hadamard Form)",
                    mathematical_form="dJ/dS = nu * grad(U_adj) · n · n - grad(p_adj) · n",
                    variables=["surface_sensitivity", "adjoint_velocity", "adjoint_pressure", "surface_normal"],
                    parameters={"form": "Hadamard"},
                    description="Shape sensitivity from adjoint solution via Hadamard boundary formula",
                ),
            ])

        return model

    def _build_numerical_method(self, mode: str) -> NumericalMethod:
        method = NumericalMethod()
        method.discretization = Discretization(
            space_discretization="finite_volume_unstructured",
            time_integrator="implicit_euler",
        )
        method.solver = Solver(
            algorithm="multigrid_Krylov",
            convergence_criterion="residual_relative",
        )
        if mode == "adjoint":
            method.solver.algorithm = "multigrid_Krylov_adjoint"
        return method

    def _build_computational_graph(self, mode: str) -> ComputationalGraph:
        graph = ComputationalGraph()

        graph.add_node(ComputationalNode(
            id="flow_solve",
            type="solve",
            math_semantics={
                "operator_type": "multi_zone_flow",
                "updates": {"target": "w", "mode": UpdateMode.IMPLICIT_LOOP.value},
                "convergence": {"required": True, "criterion": "density_residual"},
            },
        ))
        graph.add_node(ComputationalNode(
            id="turbulence_solve",
            type="solve",
            math_semantics={
                "operator_type": "turbulence_model",
                "updates": {"target": "nu_t", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        if mode == "adjoint":
            graph.add_node(ComputationalNode(
                id="adjoint_solve",
                type="solve",
                math_semantics={
                    "operator_type": "continuous_adjoint",
                    "updates": {"target": "psi", "mode": UpdateMode.IMPLICIT_LOOP.value},
                    "convergence": {"required": True, "criterion": "adjoint_residual"},
                },
            ))
            graph.add_node(ComputationalNode(
                id="sensitivity_compute",
                type="compute",
                math_semantics={
                    "operator_type": "surface_sensitivity",
                    "updates": {"target": "dJ_dS", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))
            graph.add_node(ComputationalNode(
                id="gradient_verify",
                type="verify",
                math_semantics={
                    "operator_type": "gradient_verification",
                    "updates": {"target": "gradient_accuracy", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))

            graph.add_edge(ComputationalEdge(
                from_node="flow_solve", to_node="adjoint_solve",
                data_type="flow_solution", dependency="prerequisite",
            ))
            graph.add_edge(ComputationalEdge(
                from_node="adjoint_solve", to_node="sensitivity_compute",
                data_type="adjoint_solution", dependency="compute",
            ))
            graph.add_edge(ComputationalEdge(
                from_node="sensitivity_compute", to_node="gradient_verify",
                data_type="sensitivity_map", dependency="verify",
            ))

        graph.execution_topology = {
            "schedule": "sequential",
            "implicit_loops": [{
                "loop_id": "flow_convergence",
                "nested_in": "flow_solve",
                "convergence_guarantee": "monitored_residual",
                "max_iterations_source": "cfg",
            }],
        }
        return graph

    def list_extractable_objects(self) -> List[str]:
        return ["governing_equations", "numerical_method", "computational_graph", "adjoint"]

    def get_supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTENSIONS


HarnessRegistry.register(SU2Extractor)
