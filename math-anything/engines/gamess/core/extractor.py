"""GAMESS mathematical structure extractor.

Extracts mathematical structures from GAMESS quantum chemistry simulations.
Focus: Hartree-Fock and post-HF methods, MCSCF, MRCI.
Mathematical structure: SpectralProblem (HF/post-HF).
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


class GAMESSExtractor(MathAnythingHarness):
    """Extracts mathematical structures from GAMESS quantum chemistry simulations.

    GAMESS (General Atomic and Molecular Electronic Structure System) provides
    a wide range of quantum chemistry methods including HF, DFT, MCSCF, MRCI, and CC.

    Mathematical structure: SpectralProblem (eigenvalue problem in Hilbert space).
    Advanced: MCSCF (multi-configurational), MRCI (multi-reference CI).

    File types: .inp
    """

    SUPPORTED_EXTENSIONS = [".inp"]

    @property
    def engine_name(self) -> str:
        return "gamess"

    @property
    def supported_schema_version(self) -> str:
        return "1.0.0"

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> MathSchema:
        options = options or {}
        method = options.get("method", "rhf")

        schema = MathSchema(
            meta=MetaInfo(
                extracted_by="math-anything-gamess",
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
                id="hartree_fock",
                type="spectral_problem",
                name="Hartree-Fock (Roothaan-Hall) Equations",
                mathematical_form="F(C) * C = S * C * epsilon",
                variables=["fock_matrix", "molecular_orbital_coefficients", "overlap_matrix", "orbital_energies"],
                parameters={"form": "generalized_eigenvalue"},
                description="Self-consistent field equation in atomic orbital basis (Roothaan-Hall form)",
            ),
            GoverningEquation(
                id="fock_construction",
                type="integral_equation",
                name="Fock Matrix Construction",
                mathematical_form="F_mu_nu = h_mu_nu + sum_{lambda_sigma} P_lambda_sigma * [(mu_nu|lambda_sigma) - 0.5*(mu_lambda|nu_sigma)]",
                variables=["core_hamiltonian", "density_matrix", "two_electron_integrals"],
                parameters={"type": "closed_shell"},
                description="Fock matrix from one-electron core Hamiltonian and two-electron repulsion integrals",
            ),
        ]

        if method == "mcscf":
            model.governing_equations.append(
                GoverningEquation(
                    id="mcscf",
                    type="multi_configurational",
                    name="Multi-Configurational Self-Consistent Field",
                    mathematical_form="H_eff * C = E * C, simultaneously optimize C and orbital rotation parameters",
                    variables=["ci_coefficients", "orbital_rotation", "effective_hamiltonian"],
                    parameters={"active_space": "CAS", "optimization": "second_order_newton_raphson"},
                    description="Simultaneous optimization of CI coefficients and orbital parameters",
                )
            )

        if method == "mrci":
            model.governing_equations.append(
                GoverningEquation(
                    id="mrci",
                    type="configuration_interaction",
                    name="Multi-Reference Configuration Interaction",
                    mathematical_form="H_CI * C = E * C, with H_CI in the space of reference + excitations",
                    variables=["ci_vector", "ci_hamiltonian", "reference_space"],
                    parameters={"single_excitations": True, "double_excitations": True, "reference": "MCSCF"},
                    description="Configuration interaction expanded about multi-reference wavefunction",
                )
            )

        if method in ("mp2", "ccsd", "ccsdt"):
            model.governing_equations.append(
                GoverningEquation(
                    id="correlation_energy",
                    type="perturbation_equation",
                    name="Electron Correlation via Post-HF Method",
                    mathematical_form="E_corr = trace(P_c * H_c), T|HF> generates correlated wavefunction",
                    variables=["correlation_energy", "amplitudes", "excitation_operator"],
                    parameters={"method": method},
                    description="Post-HF correlation energy via perturbation theory or coupled cluster",
                )
            )

        return model

    def _build_numerical_method(self, method: str) -> NumericalMethod:
        nm = NumericalMethod()
        nm.discretization = Discretization(
            space_discretization="gaussian_basis_set",
        )
        nm.solver = Solver(
            algorithm="diis_extrapolation",
            convergence_criterion="density_matrix_residual",
            tolerance=1e-7,
        )
        if method == "mcscf":
            nm.solver.algorithm = "mcscf_newton_raphson"
        return nm

    def _build_computational_graph(self, method: str) -> ComputationalGraph:
        graph = ComputationalGraph()

        graph.add_node(ComputationalNode(
            id="guess",
            type="initialize",
            math_semantics={
                "operator_type": "extended_huckel",
                "updates": {"target": "P_0", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="int1e",
            type="compute",
            math_semantics={
                "operator_type": "one_electron_integrals",
                "updates": {"target": "h_mu_nu", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="int2e",
            type="compute",
            math_semantics={
                "operator_type": "two_electron_integrals",
                "updates": {"target": "g_mu_nu_lambda_sigma", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="scf_cycle",
            type="iterate",
            math_semantics={
                "operator_type": "fock_diagonalization_cycle",
                "updates": {"target": "P_F_C", "mode": UpdateMode.IMPLICIT_LOOP.value},
                "convergence": {"required": True, "criterion": "density_change"},
            },
        ))

        if method in ("mp2", "ccsd", "ccsdt", "mcscf", "mrci"):
            graph.add_node(ComputationalNode(
                id="post_scf",
                type="compute",
                math_semantics={
                    "operator_type": f"{method}_correlation",
                    "updates": {"target": "E_corr", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            ))

        graph.add_edge(ComputationalEdge(
            from_node="guess", to_node="scf_cycle",
            data_type="initial_density", dependency="initialize",
        ))
        graph.add_edge(ComputationalEdge(
            from_node="int1e", to_node="scf_cycle",
            data_type="core_hamiltonian", dependency="prerequisite",
        ))
        graph.add_edge(ComputationalEdge(
            from_node="int2e", to_node="scf_cycle",
            data_type="eri_tensor", dependency="prerequisite",
        ))

        graph.execution_topology = {
            "schedule": "sequential_integrals_then_scf",
            "implicit_loops": [{
                "loop_id": "scf_iteration",
                "nested_in": "scf_cycle",
                "convergence_guarantee": "diis_accelerated",
                "max_iterations_source": "input",
            }],
        }
        return graph

    def list_extractable_objects(self) -> List[str]:
        return ["governing_equations", "numerical_method", "computational_graph"]

    def get_supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTENSIONS


HarnessRegistry.register(GAMESSExtractor)
