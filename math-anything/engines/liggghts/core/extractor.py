"""LIGGGHTS mathematical structure extractor.

Extracts mathematical structures from LIGGGHTS discrete element method (DEM) simulations.
Focus: Hamiltonian mechanics with contact dynamics, Hertz-Mindlin contact model.
Mathematical structure: HamiltonianSystem with contact mechanics.
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


class LIGGGHTSExtractor(MathAnythingHarness):
    """Extracts mathematical structures from LIGGGHTS DEM simulations.

    LIGGGHTS (LAMMPS Improved for General Granular and Granular Heat Transfer Simulations)
    simulates granular and particle flows using the Discrete Element Method (DEM).

    Mathematical structure: HamiltonianSystem with contact mechanics.
    Contact model: Hertz-Mindlin (normal force from Hertz theory, tangential from Mindlin-Deresiewicz).

    File types: .liggghts, .in
    """

    SUPPORTED_EXTENSIONS = [".liggghts", ".in"]

    @property
    def engine_name(self) -> str:
        return "liggghts"

    @property
    def supported_schema_version(self) -> str:
        return "1.0.0"

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> MathSchema:
        options = options or {}
        contact_model = options.get("contact_model", "hertz_mindlin")

        schema = MathSchema(
            meta=MetaInfo(
                extracted_by="math-anything-liggghts",
                extractor_version="0.1.0",
            ),
            mathematical_model=self._build_mathematical_model(contact_model),
            numerical_method=self._build_numerical_method(),
            computational_graph=self._build_computational_graph(),
        )
        return schema

    def _build_mathematical_model(self, contact_model: str) -> MathematicalModel:
        model = MathematicalModel()

        model.governing_equations = [
            GoverningEquation(
                id="newton_euler",
                type="hamiltonian_system",
                name="Newton-Euler Equations of Rigid Body Motion",
                mathematical_form="m*d²r/dt² = F_total; I*d(omega)/dt + omega x (I*omega) = M_total",
                variables=["position", "orientation", "force", "torque", "mass", "inertia_tensor"],
                parameters={"integrator": "velocity_verlet", "degrees_of_freedom": [3, 3]},
                description="Newton-Euler equations for translational and rotational rigid body dynamics",
            ),
            GoverningEquation(
                id="hertz_contact_normal",
                type="contact_mechanics",
                name="Hertz Normal Contact Force",
                mathematical_form="F_n = (4/3) * E_eff * sqrt(R_eff) * delta_n^(3/2)",
                variables=["normal_force", "effective_youngs_modulus", "effective_radius", "normal_overlap"],
                parameters={"constitutive": "hertzian_elastic"},
                description="Hertz elastic normal contact force from spherical particle overlap",
            ),
            GoverningEquation(
                id="mindlin_tangential",
                type="contact_mechanics",
                name="Mindlin-Deresiewicz Tangential Contact Force",
                mathematical_form="dF_t = 8 * G_eff * sqrt(R_eff * delta_n) * d_delta_t - mu * dF_n * (1 - (1 - F_t/(mu*F_n))^(1/3))",
                variables=["tangential_force", "effective_shear_modulus", "tangential_displacement", "friction_coefficient"],
                parameters={"constitutive": "mindlin_no_slip"},
                description="Mindlin-Deresiewicz tangential contact with partial slip and Coulomb friction limit",
            ),
            GoverningEquation(
                id="rolling_friction",
                type="contact_mechanics",
                name="Rolling Resistance Torque",
                mathematical_form="tau_r = -mu_r * R_eff * |F_n| * omega_rel / |omega_rel|",
                variables=["rolling_torque", "rolling_friction_coefficient", "relative_angular_velocity"],
                parameters={"type": "constant_directional"},
                description="Rolling resistance torque model for granular flow",
            ),
        ]

        if contact_model == "hertz_history":
            model.governing_equations[2] = GoverningEquation(
                id="mindlin_tangential_history",
                type="contact_mechanics",
                name="Mindlin-Deresiewicz Tangential with Loading History",
                mathematical_form="F_t = F_t_prev + k_t * d_delta_t, with |F_t| <= mu * |F_n|",
                variables=["tangential_force", "tangential_stiffness", "tangential_displacement_increment", "friction_coefficient"],
                parameters={"constitutive": "mindlin_history"},
                description="Incremental Mindlin model with full loading-unloading-reloading history",
            )

        model.governing_equations.append(
            GoverningEquation(
                id="cohesive_force",
                type="contact_mechanics",
                name="Johnson-Kendall-Roberts (JKR) Cohesive Force",
                mathematical_form="F_JKR = -4 * sqrt(pi * gamma * E_eff) * a^(3/2) + (4*E_eff*a^3)/(3*R_eff)",
                variables=["jkr_force", "contact_radius", "surface_energy", "effective_modulus"],
                parameters={"model": "JKR"},
                description="JKR adhesive contact model incorporating surface energy effects",
            )
        )

        return model

    def _build_numerical_method(self) -> NumericalMethod:
        method = NumericalMethod()
        method.discretization = Discretization(
            space_discretization="discrete_particle",
            time_integrator="explicit_velocity_verlet",
        )
        method.solver = Solver(
            algorithm="explicit_DEM",
            convergence_criterion="none",
        )
        return method

    def _build_computational_graph(self) -> ComputationalGraph:
        graph = ComputationalGraph()

        graph.add_node(ComputationalNode(
            id="neighbor_list",
            type="search",
            math_semantics={
                "operator_type": "verlet_pair_search",
                "updates": {"target": "neighbor_list", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="contact_detect",
            type="compute",
            math_semantics={
                "operator_type": "particle_overlap_detection",
                "updates": {"target": "overlap_delta", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="contact_force",
            type="compute",
            math_semantics={
                "operator_type": "hertz_mindlin_contact",
                "updates": {"target": "F_pair", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="integrate",
            type="integrate",
            math_semantics={
                "operator_type": "velocity_verlet_integrate",
                "updates": {"target": "position_velocity", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="walls",
            type="compute",
            math_semantics={
                "operator_type": "wall_contact",
                "updates": {"target": "F_wall", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        graph.add_edge(ComputationalEdge(
            from_node="neighbor_list", to_node="contact_detect",
            data_type="pairs", dependency="prerequisite",
        ))
        graph.add_edge(ComputationalEdge(
            from_node="contact_detect", to_node="contact_force",
            data_type="overlaps", dependency="compute",
        ))
        graph.add_edge(ComputationalEdge(
            from_node="contact_force", to_node="integrate",
            data_type="forces", dependency="integrate",
        ))
        graph.add_edge(ComputationalEdge(
            from_node="walls", to_node="integrate",
            data_type="wall_forces", dependency="integrate",
        ))

        graph.execution_topology = {
            "schedule": "explicit_sequential_per_timestep",
        }
        return graph

    def list_extractable_objects(self) -> List[str]:
        return ["governing_equations", "numerical_method", "computational_graph", "contact_mechanics"]

    def get_supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTENSIONS


HarnessRegistry.register(LIGGGHTSExtractor)
