"""VASP mathematical structure extractor.

Extracts mathematical structures from VASP calculations and maps them
to Math Schema v1.0 representation.
"""

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from math_anything.schemas import (BoundaryCondition, ComputationalEdge,
                                   ComputationalGraph, ComputationalNode,
                                   Discretization, GoverningEquation,
                                   MathematicalModel, MathematicalObject,
                                   MathSchema, MetaInfo, NumericalMethod,
                                   Solver, TensorComponent, UpdateMode)

from .parser import (CrystalStructure, ElectronicParameters, VaspInputParser,
                     VaspOutputParser)


class VaspExtractor:
    """Extracts mathematical structures from VASP calculations.

    Maps VASP parameters and results to Math Schema v1.0 representation,
    including:
    - Kohn-Sham equations (DFT governing equations)
    - Plane wave basis (numerical method)
    - Self-consistent field iterations (computational graph)
    - Electronic structure (eigenvalue problem)

    Example:
        ```python
        extractor = VaspExtractor()
        schema = extractor.extract({
            "incar": "INCAR",
            "poscar": "POSCAR",
            "kpoints": "KPOINTS",
            "outcar": "OUTCAR"
        })
        ```
    """

    def __init__(self):
        self.input_parser = VaspInputParser()
        self.output_parser = VaspOutputParser()
        self.elec_params: Optional[ElectronicParameters] = None
        self.structure: Optional[CrystalStructure] = None

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> MathSchema:
        """Extract mathematical structures from VASP files.

        Args:
            files: Dictionary with 'incar', 'poscar', 'kpoints', 'outcar'
            options: Optional extraction parameters

        Returns:
            MathSchema object
        """
        options = options or {}

        # Parse input files
        if "incar" in files:
            self.input_parser.parse_incar(files["incar"])
            self.elec_params = self.input_parser.get_electronic_parameters()

        if "poscar" in files:
            self.structure = self.input_parser.parse_poscar(files["poscar"])

        if "kpoints" in files:
            self.input_parser.parse_kpoints(files["kpoints"])

        # Parse output
        results = None
        if "outcar" in files:
            results = self.output_parser.parse_outcar(files["outcar"])

        # Build Math Schema
        schema = MathSchema(
            schema_version="1.0.0",
            meta=MetaInfo(
                extracted_by="math-anything-vasp",
                extractor_version="0.1.0",
                source_files={
                    "input": [f for f in files.values()],
                },
            ),
            mathematical_model=self._extract_mathematical_model(),
            numerical_method=self._extract_numerical_method(),
            computational_graph=self._extract_computational_graph(results),
            conservation_properties=self._extract_conservation_properties(),
            raw_symbols=self._extract_raw_symbols(),
        )

        return schema

    def _extract_mathematical_model(self) -> MathematicalModel:
        """Extract mathematical model (equations, BCs, ICs)."""
        model = MathematicalModel()

        # Kohn-Sham equations (core of DFT)
        model.governing_equations = self._extract_kohn_sham_equations()

        # Boundary conditions (periodic)
        model.boundary_conditions = self._extract_boundary_conditions()

        # Constitutive relations (XC functional, pseudopotentials)
        model.constitutive_relations = self._extract_constitutive_relations()

        return model

    def _extract_kohn_sham_equations(self) -> List[GoverningEquation]:
        """Extract Kohn-Sham equations."""
        equations = []

        # Main Kohn-Sham equation
        equations.append(
            GoverningEquation(
                id="kohn_sham",
                type="eigenvalue_problem",
                name="Kohn-Sham Equations",
                mathematical_form="[-ℏ²∇²/2m + V_eff[n](r)] ψ_i(r) = ε_i ψ_i(r)",
                variables=[
                    "wavefunction",
                    "eigenvalue",
                    "effective_potential",
                    "electron_density",
                ],
                parameters={
                    "form": "nonlinear_eigenvalue",
                    "self_consistent": True,
                    "spin_polarized": (
                        self.elec_params.ispin == 2 if self.elec_params else False
                    ),
                },
                description="Density Functional Theory governing equations",
            )
        )

        # Hohenberg-Kohn theorem (density formulation)
        equations.append(
            GoverningEquation(
                id="hohenberg_kohn",
                type="variational_principle",
                name="Hohenberg-Kohn Theorem",
                mathematical_form="E[n] = T_s[n] + ∫ V_ext(r)n(r)dr + E_H[n] + E_xc[n]",
                variables=[
                    "energy",
                    "density",
                    "kinetic_energy",
                    "external_potential",
                    "hartree_energy",
                    "exchange_correlation",
                ],
                parameters={"universality": True},
                description="Ground state density minimizes energy functional",
            )
        )

        # Charge density from wavefunctions
        equations.append(
            GoverningEquation(
                id="charge_density",
                type="density_construction",
                name="Charge Density Construction",
                mathematical_form="n(r) = Σ f_i |ψ_i(r)|²",
                variables=["density", "occupation", "wavefunction"],
                parameters={"occupation": "fermi_dirac"},
                description="Electron density from occupied states",
            )
        )

        return equations

    def _extract_boundary_conditions(self) -> List[BoundaryCondition]:
        """Extract boundary conditions (periodic)."""
        bcs = []

        if self.structure:
            # Periodic boundary conditions on crystal lattice
            lattice = self.structure.lattice_vectors

            # Create tensor for periodicity
            components = []
            for i in range(3):
                for j in range(3):
                    components.append(
                        TensorComponent(
                            index=[i + 1, j + 1],
                            value=str(lattice[i, j]),
                            unit="Angstrom",
                        )
                    )

            bcs.append(
                BoundaryCondition(
                    id="periodic_lattice",
                    type="periodic",
                    domain={
                        "geometric_region": "unit_cell",
                        "entity_type": "crystal_lattice",
                    },
                    mathematical_object=MathematicalObject(
                        field="lattice_vectors",
                        tensor_rank=2,
                        tensor_form="a_i = Σ_j a_{ij} e_j",
                        components=components,
                        symmetry="none",
                    ),
                    software_implementation={
                        "command": "POSCAR",
                        "parameters": {
                            "scale": self.structure.scale,
                            "coord_type": self.structure.coord_type,
                        },
                    },
                    dual_role={
                        "is_boundary_condition": True,
                        "is_external_drive": False,
                        "note": "Periodic boundary conditions enforced by plane wave basis",
                    },
                )
            )

            # Bloch theorem for k-points
            if self.input_parser.kpoints:
                kgrid = self.input_parser.kpoints.grid
                bcs.append(
                    BoundaryCondition(
                        id="bloch_theorem",
                        type="quasi_periodic",
                        domain={
                            "geometric_region": "reciprocal_space",
                            "entity_type": "kpoint_grid",
                        },
                        mathematical_object=MathematicalObject(
                            field="bloch_wavefunction",
                            tensor_rank=1,
                            tensor_form="ψ_{k+G}(r) = ψ_k(r) e^{iG·r}",
                            components=[
                                TensorComponent(
                                    index=[i + 1], value=str(k), unit="1/Angstrom"
                                )
                                for i, k in enumerate(kgrid)
                            ],
                        ),
                        software_implementation={
                            "command": "KPOINTS",
                            "parameters": {
                                "grid": kgrid,
                                "mode": self.input_parser.kpoints.mode,
                            },
                        },
                        equivalent_formulations=[
                            {
                                "type": "bloch_theorem",
                                "form": "ψ_{nk}(r+R) = e^{ik·R} ψ_{nk}(r)",
                            }
                        ],
                    )
                )

        return bcs

    def _extract_constitutive_relations(self) -> List[Dict[str, Any]]:
        """Extract constitutive relations (XC functional, pseudopotentials)."""
        relations = []

        if not self.elec_params:
            return relations

        # Exchange-correlation functional
        incar = self.input_parser.incar_params

        # Determine XC functional
        if incar.get("LHFCALC", False):
            xc_type = "hybrid"
            xc_name = "HSE06"
            form = "E_xc = E_xc^DFA + α(E_x^exact - E_x^DFA)"
        elif incar.get("GGA"):
            xc_type = "gga"
            xc_name = incar.get("GGA")
            form = f"E_xc^{xc_name} = ∫ f(n, ∇n) dr"
        elif incar.get("LDAU", False):
            xc_type = "dft_plus_u"
            xc_name = "LDA+U"
            form = "E = E_DFT + Σ U/2 (n - n²)"
        else:
            xc_type = "lda"
            xc_name = "LDA"
            form = "E_xc^LDA = ∫ ε_xc(n(r)) n(r) dr"

        relations.append(
            {
                "type": "exchange_correlation",
                "name": xc_name,
                "functional_type": xc_type,
                "mathematical_form": form,
                "parameters": {
                    "ismear": self.elec_params.ismear,
                    "sigma": self.elec_params.sigma,
                },
            }
        )

        # Pseudopotential / PAW
        relations.append(
            {
                "type": "pseudopotential",
                "name": "PAW",
                "mathematical_form": "V_eff = V_local + Σ |p_i⟩ D_ij ⟨p_j|",
                "description": "Projector Augmented Wave method",
            }
        )

        return relations

    def _extract_numerical_method(self) -> NumericalMethod:
        """Extract numerical method (plane wave basis, SCF)."""
        method = NumericalMethod()

        if self.elec_params:
            # Plane wave basis
            method.discretization.space_discretization = "plane_wave_basis"
            method.discretization.time_integrator = "scf_iteration"
            method.discretization.order = 1  # Linear mixing default

            # Convergence criteria
            method.solver = Solver(
                algorithm="scf_diagonalization",
                convergence_criterion="energy_difference",
                tolerance=self.elec_params.ediff,
                max_iterations=self.elec_params.nelm,
            )

            # Parallelization (from INCAR)
            method.parallelization = {
                "kpoint_parallel": True,
                "band_parallel": True,
                "plane_wave_parallel": True,
            }

        return method

    def _extract_computational_graph(
        self, results: Optional[Any]
    ) -> ComputationalGraph:
        """Extract computational graph with SCF loop structure."""
        graph = ComputationalGraph()

        # SCF iteration nodes
        graph.add_node(
            ComputationalNode(
                id="hamiltonian_construction",
                type="operator_application",
                math_semantics={
                    "operator_type": "hamiltonian_construction",
                    "updates": {
                        "target": "H",
                        "mode": UpdateMode.EXPLICIT_UPDATE.value,
                    },
                },
            )
        )

        graph.add_node(
            ComputationalNode(
                id="diagonalization",
                type="eigenvalue_solver",
                math_semantics={
                    "operator_type": "dense_eigensolver",
                    "updates": {
                        "target": "wavefunctions",
                        "mode": UpdateMode.EXPLICIT_UPDATE.value,
                    },
                },
            )
        )

        graph.add_node(
            ComputationalNode(
                id="density_update",
                type="density_construction",
                math_semantics={
                    "operator_type": "density_summing",
                    "updates": {
                        "target": "density",
                        "mode": UpdateMode.EXPLICIT_UPDATE.value,
                    },
                },
            )
        )

        # SCF convergence check (implicit loop)
        graph.add_node(
            ComputationalNode(
                id="scf_convergence",
                type="convergence_check",
                math_semantics={
                    "operator_type": "residual_evaluation",
                    "updates": {
                        "target": "energy_difference",
                        "mode": UpdateMode.IMPLICIT_LOOP.value,
                    },
                    "convergence": {
                        "required": True,
                        "criterion": (
                            f"energy_change < {self.elec_params.ediff}"
                            if self.elec_params
                            else "energy_change < threshold"
                        ),
                        "max_iterations": (
                            self.elec_params.nelm if self.elec_params else 60
                        ),
                    },
                },
            )
        )

        # Edges (SCF cycle)
        graph.add_edge(
            ComputationalEdge(
                from_node="hamiltonian_construction",
                to_node="diagonalization",
                data_type="hamiltonian_matrix",
                dependency="read_only",
            )
        )

        graph.add_edge(
            ComputationalEdge(
                from_node="diagonalization",
                to_node="density_update",
                data_type="eigenvectors",
                dependency="read_only",
            )
        )

        graph.add_edge(
            ComputationalEdge(
                from_node="density_update",
                to_node="scf_convergence",
                data_type="density",
                dependency="read_only",
            )
        )

        # SCF loop back edge
        graph.add_edge(
            ComputationalEdge(
                from_node="scf_convergence",
                to_node="hamiltonian_construction",
                data_type="convergence_status",
                dependency="feedback",
            )
        )

        # Execution topology
        graph.execution_topology = {
            "schedule": "scf_loop",
            "implicit_loops": [
                {
                    "loop_id": "scf_cycle",
                    "nested_in": "electronic_optimization",
                    "convergence_guarantee": "none_for_general_systems",
                    "max_iterations_source": "incar_nelm",
                }
            ],
        }

        return graph

    def _extract_conservation_properties(self) -> Dict[str, Any]:
        """Extract conservation properties."""
        props = {}

        # Charge conservation
        props["charge"] = {
            "preserved": True,
            "mechanism": "number_of_electrons_fixed",
        }

        # Spin conservation (if non-spin polarized)
        if self.elec_params and self.elec_params.ispin == 1:
            props["spin"] = {
                "preserved": True,
                "mechanism": "collinear_calculation",
            }

        # Energy should decrease (variational principle)
        props["energy_variational"] = {
            "preserved": True,
            "mechanism": "hohenberg_kohn_theorem",
            "note": "Ground state minimizes energy functional",
        }

        return props

    def _extract_raw_symbols(self) -> Dict[str, Any]:
        """Extract raw symbols from VASP input."""
        symbols = {
            "incar": self.input_parser.incar_params if self.input_parser else {},
            "structure": self.structure.to_dict() if self.structure else {},
            "kpoints": (
                self.input_parser.kpoints.to_dict() if self.input_parser.kpoints else {}
            ),
        }

        return symbols
