"""CP2K mathematical structure extractor.

Extracts mathematical structures from CP2K DFT simulations.
Focus: Self-consistent field (Kohn-Sham DFT), Gaussian and Plane Wave (GPW/GAPW) methods.
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


class CP2KExtractor(MathAnythingHarness):
    """Extracts mathematical structures from CP2K DFT simulations.

    CP2K implements mixed Gaussian/Plane-wave density functional theory.
    Mathematical structure: SelfConsistentProblem (DFT).

    File types: .inp, .restart
    Methods: GPW (Gaussian Plane Wave), GAPW (Gaussian Augmented Plane Wave)
    """

    SUPPORTED_EXTENSIONS = [".inp", ".restart"]

    @property
    def engine_name(self) -> str:
        return "cp2k"

    @property
    def supported_schema_version(self) -> str:
        return "1.0.0"

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> MathSchema:
        options = options or {}
        method = options.get("method", "gpw")

        schema = MathSchema(
            meta=MetaInfo(
                extracted_by="math-anything-cp2k",
                extractor_version="0.1.0",
            ),
            mathematical_model=self._build_mathematical_model(method),
            numerical_method=self._build_numerical_method(method),
            computational_graph=self._build_computational_graph(),
        )
        return schema

    def _build_mathematical_model(self, method: str) -> MathematicalModel:
        model = MathematicalModel()

        equations = [
            GoverningEquation(
                id="kohn_sham",
                type="partial_differential_equation",
                name="Kohn-Sham Equations",
                mathematical_form="(-hbar^2/2m nabla^2 + V_eff[rho]) psi_i = epsilon_i psi_i",
                variables=["wavefunction", "eigenvalue", "effective_potential", "density"],
                parameters={"form": "self_consistent_eigenvalue_problem"},
                description="Self-consistent Kohn-Sham density functional theory",
            ),
            GoverningEquation(
                id="electron_density",
                type="integral_equation",
                name="Electron Density from KS Orbitals",
                mathematical_form="rho(r) = sum_i f_i |psi_i(r)|^2",
                variables=["density", "occupation", "wavefunction"],
                parameters={"form": "sum_over_occupied_states"},
                description="Electron density constructed from occupied Kohn-Sham orbitals",
            ),
        ]

        if method == "gpw":
            equations.append(
                GoverningEquation(
                    id="collocation",
                    type="transform_equation",
                    name="Gaussian to Plane-Wave Collocation",
                    mathematical_form="psi(G) = FFT[psi(r)], V_eff(G) = FFT[V_eff(r)]",
                    variables=["wavefunction_realspace", "wavefunction_reciprocal", "potential_realspace"],
                    parameters={"method": "gaussian_plane_wave", "transform": "FFT"},
                    description="Mixed Gaussian/Plane-wave representation via FFT collocation",
                )
            )
        elif method == "gapw":
            equations.append(
                GoverningEquation(
                    id="augmentation",
                    type="transform_equation",
                    name="Gaussian Augmented Plane Wave Projection",
                    mathematical_form="psi = psi_smooth + sum_a (psi_a - psi_a_smooth)",
                    variables=["wavefunction", "smooth_part", "atomic_correction"],
                    parameters={"method": "gapw"},
                    description="GAPW augmentation with atomic corrections near nuclei",
                )
            )

        equations.append(
            GoverningEquation(
                id="poisson",
                type="partial_differential_equation",
                name="Poisson Equation for Hartree Potential",
                mathematical_form="nabla^2 V_H = -4 pi rho",
                variables=["hartree_potential", "density"],
                parameters={"solver": "fft_poisson"},
                description="Electrostatic (Hartree) potential from charge density",
            )
        )

        model.governing_equations = equations
        return model

    def _build_numerical_method(self, method: str) -> NumericalMethod:
        nm = NumericalMethod()
        nm.discretization = Discretization(
            space_discretization="gaussian_plane_wave" if method == "gpw" else "gaussian_augmented_plane_wave",
        )
        nm.solver = Solver(
            algorithm="self_consistent_field",
            convergence_criterion="wavefunction_gradient",
            tolerance=1e-6,
        )
        return nm

    def _build_computational_graph(self) -> ComputationalGraph:
        graph = ComputationalGraph()

        graph.add_node(ComputationalNode(
            id="initial_guess",
            type="initialize",
            math_semantics={
                "operator_type": "superposition_guess",
                "updates": {"target": "rho", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="build_hamiltonian",
            type="construct",
            math_semantics={
                "operator_type": "ks_hamiltonian",
                "updates": {"target": "H", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="diagonalize",
            type="eigensolve",
            math_semantics={
                "operator_type": "ot_diagonalization",
                "updates": {"target": "psi", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))
        graph.add_node(ComputationalNode(
            id="mix_density",
            type="update",
            math_semantics={
                "operator_type": "broyden_mixing",
                "updates": {"target": "rho", "mode": UpdateMode.IMPLICIT_LOOP.value},
                "convergence": {"required": True, "criterion": "density_residual"},
            },
        ))
        graph.add_node(ComputationalNode(
            id="energy_forces",
            type="compute",
            math_semantics={
                "operator_type": "total_energy_forces",
                "updates": {"target": "E_total", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        graph.add_edge(ComputationalEdge(
            from_node="initial_guess", to_node="build_hamiltonian",
            data_type="density", dependency="construct",
        ))
        graph.add_edge(ComputationalEdge(
            from_node="build_hamiltonian", to_node="diagonalize",
            data_type="hamiltonian", dependency="solve",
        ))
        graph.add_edge(ComputationalEdge(
            from_node="diagonalize", to_node="mix_density",
            data_type="wavefunctions", dependency="compute",
        ))
        graph.add_edge(ComputationalEdge(
            from_node="mix_density", to_node="build_hamiltonian",
            data_type="density", dependency="iterate",
        ))
        graph.add_edge(ComputationalEdge(
            from_node="mix_density", to_node="energy_forces",
            data_type="converged_density", dependency="compute",
        ))

        graph.execution_topology = {
            "schedule": "self_consistent_loop",
            "implicit_loops": [{
                "loop_id": "scf_cycle",
                "nested_in": "mix_density",
                "convergence_guarantee": "user_controlled",
                "max_iterations_source": "max_scf",
            }],
        }
        return graph

    def list_extractable_objects(self) -> List[str]:
        return ["governing_equations", "numerical_method", "computational_graph"]

    def get_supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTENSIONS


HarnessRegistry.register(CP2KExtractor)
