"""NWChem mathematical structure extractor.

Extracts mathematical structures from NWChem computational chemistry simulations.
Focus: DFT, Coupled Cluster, Molecular Dynamics, QM/MM.
Mathematical structure: SpectralProblem + CoupledSystem.
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


class NWChemExtractor(MathAnythingHarness):
    """Extracts mathematical structures from NWChem simulations.

    NWChem provides a broad range of quantum chemistry and molecular dynamics methods
    including DFT, Coupled Cluster, Car-Parrinello MD, and QM/MM.

    Mathematical structure: SpectralProblem + CoupledSystem.
    File types: .nw
    """

    SUPPORTED_EXTENSIONS = [".nw"]

    @property
    def engine_name(self) -> str:
        return "nwchem"

    @property
    def supported_schema_version(self) -> str:
        return "1.0.0"

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> MathSchema:
        options = options or {}
        task = options.get("task", "dft")

        schema = MathSchema(
            meta=MetaInfo(
                extracted_by="math-anything-nwchem",
                extractor_version="0.1.0",
            ),
            mathematical_model=self._build_mathematical_model(task),
            numerical_method=self._build_numerical_method(task),
            computational_graph=self._build_computational_graph(task),
        )
        return schema

    def _build_mathematical_model(self, task: str) -> MathematicalModel:
        model = MathematicalModel()

        equations = [
            GoverningEquation(
                id="kohn_sham_dft",
                type="spectral_problem",
                name="Kohn-Sham Density Functional Theory",
                mathematical_form="(-nabla^2/2 + V_eff[rho]) psi_i = epsilon_i psi_i",
                variables=["wavefunction", "eigenvalue", "effective_potential", "density"],
                parameters={"form": "self_consistent_eigenvalue_problem"},
                description="Self-consistent Kohn-Sham equations with XC functional",
            ),
            GoverningEquation(
                id="electron_density_construction",
                type="integral_equation",
                name="Electron Density from KS Orbitals",
                mathematical_form="rho(r) = sum_i n_i |psi_i(r)|^2",
                variables=["density", "occupation", "wavefunction"],
                parameters={"form": "sum_over_states"},
                description="Electron density from occupied Kohn-Sham orbitals",
            ),
        ]

        if task in ("ccsd", "ccsdt", "cc"):
            equations.append(
                GoverningEquation(
                    id="coupled_cluster",
                    type="exponential_ansatz",
                    name="Coupled Cluster Wavefunction",
                    mathematical_form="|Psi_CC> = exp(T_1 + T_2 + ...) |HF>, E_CC = <HF| exp(-T) H exp(T) |HF>",
                    variables=["cluster_amplitudes", "excitation_operator", "reference_determinant"],
                    parameters={"truncation": task.upper()},
                    description="Coupled cluster exponential ansatz for electron correlation",
                )
            )

        if task in ("md", "cpmd", "car_parrinello"):
            equations.append(
                GoverningEquation(
                    id="newton_md",
                    type="second_order_ode",
                    name="Newton's Equations of Motion (Classical MD)",
                    mathematical_form="m_I * d²R_I/dt² = -dE/dR_I",
                    variables=["nuclear_position", "nuclear_force", "mass"],
                    parameters={"integrator": "velocity_verlet"},
                    description="Classical nuclear dynamics via Newton's equations",
                )
            )

            if task == "cpmd":
                equations.append(
                    GoverningEquation(
                        id="car_parrinello",
                        type="coupled_system",
                        name="Car-Parrinello Extended Lagrangian Dynamics",
                        mathematical_form="mu * d²psi_i/dt² = -delta_E/delta_psi_i + sum_j lambda_ij * psi_j",
                        variables=["wavefunction_velocity", "wavefunction_acceleration", "fictitious_mass", "lagrange_multiplier"],
                        parameters={"fictitious_mass": "mu", "constraint": "orthonormality"},
                        description="Coupled electron-nuclear dynamics via extended Lagrangian",
                    )
                )

        if task == "qmmm":
            equations.append(
                GoverningEquation(
                    id="qm_mm_coupling",
                    type="coupled_system",
                    name="QM/MM Coupling Hamiltonian",
                    mathematical_form="H_total = H_QM + H_MM + H_QM/MM",
                    variables=["qm_wavefunction", "mm_coordinates", "coupling_operator"],
                    parameters={"scheme": "electrostatic_embedding"},
                    description="QM/MM Hamiltonian with electrostatic and mechanical embedding",
                )
            )

        model.governing_equations = equations
        return model

    def _build_numerical_method(self, task: str) -> NumericalMethod:
        nm = NumericalMethod()
        nm.discretization = Discretization(
            space_discretization="gaussian_basis_set",
        )
        if task == "cpmd":
            nm.discretization.space_discretization = "plane_wave_basis"

        nm.solver = Solver(
            algorithm="self_consistent_field",
            convergence_criterion="density_residual",
            tolerance=1e-6,
        )
        if task == "cpmd":
            nm.discretization.time_integrator = "verlet"
        return nm

    def _build_computational_graph(self, task: str) -> ComputationalGraph:
        graph = ComputationalGraph()

        graph.add_node(ComputationalNode(
            id="scf_solve",
            type="iterate",
            math_semantics={
                "operator_type": "ks_scf_cycle",
                "updates": {"target": "rho_E", "mode": UpdateMode.IMPLICIT_LOOP.value},
                "convergence": {"required": True, "criterion": "density_residual"},
            },
        ))

        if task in ("ccsd", "ccsdt", "cc"):
            graph.add_node(ComputationalNode(
                id="cc_solve",
                type="solve",
                math_semantics={
                    "operator_type": "coupled_cluster_amplitudes",
                    "updates": {"target": "T_amplitudes", "mode": UpdateMode.IMPLICIT_LOOP.value},
                    "convergence": {"required": True, "criterion": "amplitude_residual"},
                },
            ))
            graph.add_node(ComputationalNode(
                id="cc_energy",
                type="compute",
                math_semantics={
                    "operator_type": "cc_energy_evaluation",
                    "updates": {"target": "E_CC", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))

        if task in ("md", "cpmd"):
            graph.add_node(ComputationalNode(
                id="force_compute",
                type="compute",
                math_semantics={
                    "operator_type": "energy_gradient",
                    "updates": {"target": "F_I", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))
            graph.add_node(ComputationalNode(
                id="velocity_verlet",
                type="integrate",
                math_semantics={
                    "operator_type": "verlet_integrator",
                    "updates": {"target": "R_P", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))

        if task == "qmmm":
            graph.add_node(ComputationalNode(
                id="qm_solve",
                type="solve",
                math_semantics={
                    "operator_type": "qm_scf",
                    "updates": {"target": "rho_QM", "mode": UpdateMode.IMPLICIT_LOOP.value},
                },
            ))
            graph.add_node(ComputationalNode(
                id="mm_force",
                type="compute",
                math_semantics={
                    "operator_type": "mm_force_field",
                    "updates": {"target": "F_MM", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))
            graph.add_node(ComputationalNode(
                id="coupling",
                type="compute",
                math_semantics={
                    "operator_type": "qm_mm_embedding",
                    "updates": {"target": "V_coupling", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))

        return graph

    def list_extractable_objects(self) -> List[str]:
        return ["governing_equations", "numerical_method", "computational_graph"]

    def get_supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTENSIONS


HarnessRegistry.register(NWChemExtractor)
