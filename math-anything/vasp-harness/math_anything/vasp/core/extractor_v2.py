"""Enhanced VASP mathematical structure extractor with full symbolic constraint support.

Extracts mathematical structures from VASP calculations and maps them
to Math Schema v1.0 representation with SymbolicConstraint support.
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
                                   ParameterRelationship, Solver,
                                   SymbolicConstraint, TensorComponent,
                                   UpdateMode)

# Import the new parsers with symbolic constraint support
from .incar_parser import IncarParser
from .kpoints_parser import KpointsData, KpointsMode, KpointsParser
from .poscar_parser import CrystalStructure, Lattice, PoscarParser


class VaspSymbolicConstraints:
    """Symbolic constraints for VASP parameters.

    Maps VASP input parameters to their mathematical constraints
    and relationships for LLM symbolic reasoning.
    """

    # INCAR parameter constraints
    INCAR_CONSTRAINTS = {
        # Energy cutoff constraints
        "ENCUT": [
            SymbolicConstraint(
                expression="ENCUT > 0",
                description="Energy cutoff must be positive (critical)",
            ),
            SymbolicConstraint(
                expression="ENCUT > max(ENMAX)",
                description="Energy cutoff should exceed maximum ENMAX of pseudopotentials (warning)",
            ),
        ],
        "EDIFF": [
            SymbolicConstraint(
                expression="EDIFF > 0",
                description="Electronic convergence threshold must be positive (critical)",
            ),
            SymbolicConstraint(
                expression="EDIFF < 1e-3",
                description="Standard precision: EDIFF < 1e-3 (info)",
            ),
        ],
        "ISMEAR": [
            SymbolicConstraint(
                expression="ISMEAR in [-5, -4, -3, -2, -1, 0, 1, 2]",
                description="ISMEAR must be one of the valid values (critical)",
            ),
        ],
        "SIGMA": [
            SymbolicConstraint(
                expression="SIGMA > 0",
                description="Gaussian smearing width must be positive (critical)",
            ),
            SymbolicConstraint(
                expression="SIGMA < 0.1",
                description="For accurate forces: SIGMA < 0.1 eV (info)",
            ),
        ],
        "NELM": [
            SymbolicConstraint(
                expression="NELM > 0",
                description="Maximum SCF steps must be positive (critical)",
            ),
            SymbolicConstraint(
                expression="NELM >= 60",
                description="Standard SCF iterations: NELM >= 60 (info)",
            ),
        ],
        "EDIFFG": [
            SymbolicConstraint(
                expression="EDIFFG != 0",
                description="EDIFFG must be non-zero for ionic relaxation (critical)",
            ),
            SymbolicConstraint(
                expression="abs(EDIFFG) < abs(EDIFF) * 10",
                description="Force convergence should be tighter than energy (warning)",
            ),
        ],
    }

    # k-point sampling constraints
    KPOINTS_CONSTRAINTS = {
        "mesh_density": [
            SymbolicConstraint(
                expression="n_k >= 1",
                description="At least one k-point required (critical)",
            ),
            SymbolicConstraint(
                expression="n_k^(1/3) * a >= 20 Angstrom",
                description="Minimum k-point density rule of thumb (info)",
            ),
        ],
        "grid_consistency": [
            SymbolicConstraint(
                expression="all(n_i > 0 for n_i in subdivisions)",
                description="All subdivisions must be positive (critical)",
            ),
        ],
    }

    # Crystal structure constraints
    STRUCTURE_CONSTRAINTS = {
        "lattice_vectors": [
            SymbolicConstraint(
                expression="det(lattice) != 0",
                description="Lattice vectors must be linearly independent (critical)",
            ),
            SymbolicConstraint(
                expression="volume > 0",
                description="Unit cell volume must be positive (critical)",
            ),
        ],
        "atomic_positions": [
            SymbolicConstraint(
                expression="0 <= x_i <= 1",
                description="Direct coordinates in [0, 1] (critical)",
            ),
        ],
    }

    @classmethod
    def get_constraints_for_parameter(cls, param_name: str) -> List[SymbolicConstraint]:
        """Get constraints for a specific parameter."""
        return cls.INCAR_CONSTRAINTS.get(param_name, [])

    @classmethod
    def validate_all_constraints(
        cls,
        incar_params: Dict[str, Any],
        structure: Optional[CrystalStructure] = None,
        kpoints: Optional[KpointsData] = None,
    ) -> Dict[str, Any]:
        """Validate all constraints against actual values."""
        results = {
            "passed": [],
            "failed": [],
            "warnings": [],
        }

        # Validate INCAR constraints
        for param, constraints in cls.INCAR_CONSTRAINTS.items():
            if param in incar_params:
                value = incar_params[param]
                for constraint in constraints:
                    validation = cls._validate_constraint(constraint, param, value)
                    if validation["passed"]:
                        results["passed"].append(validation)
                    elif "critical" in constraint.description.lower():
                        results["failed"].append(validation)
                    else:
                        results["warnings"].append(validation)

        # Validate structure constraints
        if structure:
            for constraint in cls.STRUCTURE_CONSTRAINTS["lattice_vectors"]:
                if "det" in constraint.expression:
                    det = np.linalg.det(structure.lattice.vectors)
                    if abs(det) > 1e-10:
                        results["passed"].append(
                            {
                                "constraint": constraint.expression,
                                "param": "lattice",
                                "value": det,
                                "passed": True,
                            }
                        )
                    else:
                        results["failed"].append(
                            {
                                "constraint": constraint.expression,
                                "param": "lattice",
                                "value": det,
                                "passed": False,
                            }
                        )
                elif constraint.expression == "volume > 0":
                    if structure.lattice.volume > 0:
                        results["passed"].append(
                            {
                                "constraint": "volume > 0",
                                "param": "volume",
                                "value": structure.lattice.volume,
                                "passed": True,
                            }
                        )
                    else:
                        results["failed"].append(
                            {
                                "constraint": "volume > 0",
                                "param": "volume",
                                "value": structure.lattice.volume,
                                "passed": False,
                            }
                        )

        # Validate k-points constraints
        if kpoints and kpoints.mesh:
            total_k = kpoints.mesh.total_kpoints
            if total_k >= 1:
                results["passed"].append(
                    {
                        "constraint": "n_k >= 1",
                        "param": "kpoints",
                        "value": total_k,
                        "passed": True,
                    }
                )

        return results

    @classmethod
    def _validate_constraint(
        cls, constraint: SymbolicConstraint, param_name: str, value: Any
    ) -> Dict[str, Any]:
        """Validate a single constraint."""
        result = {
            "constraint": constraint.expression,
            "param": param_name,
            "value": value,
            "passed": False,
        }

        try:
            expr = constraint.expression

            if expr.startswith(">="):
                import re

                match = re.search(r">=\s*([-\d.eE+]+)", expr)
                if match:
                    result["passed"] = float(value) >= float(match.group(1))
            elif expr.startswith(">"):
                import re

                match = re.search(r">\s*([-\d.eE+]+)", expr)
                if match:
                    result["passed"] = float(value) > float(match.group(1))
            elif expr.startswith("<="):
                import re

                match = re.search(r"<=\s*([-\d.eE+]+)", expr)
                if match:
                    result["passed"] = float(value) <= float(match.group(1))
            elif expr.startswith("<"):
                import re

                match = re.search(r"<\s*([-\d.eE+]+)", expr)
                if match:
                    result["passed"] = float(value) < float(match.group(1))
            elif "in [" in expr:
                import re

                match = re.search(r"\[(.*?)\]", expr)
                if match:
                    valid_str = match.group(1)
                    valid_values = [int(x.strip()) for x in valid_str.split(",")]
                    result["passed"] = int(value) in valid_values
            elif "!=" in expr:
                result["passed"] = float(value) != 0
            else:
                result["passed"] = True  # Unknown constraint, assume valid

        except Exception as e:
            result["error"] = str(e)
            result["passed"] = False

        return result


class VaspParameterRelationships:
    """Parameter relationships for DFT calculations."""

    RELATIONSHIPS = [
        ParameterRelationship(
            name="encut_prec_relation",
            expression="ENCUT = ENMAX * factor(PREC)",
            variables=["ENCUT", "PREC", "ENMAX"],
            relation_type="equality",
            description="Energy cutoff scales with precision setting",
        ),
        ParameterRelationship(
            name="sigma_temperature_relation",
            expression="SIGMA = k_B * T (for ISMEAR > 0)",
            variables=["ISMEAR", "SIGMA", "T"],
            relation_type="equality",
            description="Gaussian smearing width related to temperature",
        ),
        ParameterRelationship(
            name="ediff_ediffg_relation",
            expression="|EDIFFG| = 10 * |EDIFF| (typical)",
            variables=["EDIFF", "EDIFFG"],
            relation_type="equality",
            description="Force convergence tighter than energy",
        ),
        ParameterRelationship(
            name="kspacing_lattice_relation",
            expression="k_spacing ~ 1 / (n_k^(1/3) * a)",
            variables=["k_mesh", "lattice", "n_k", "a"],
            relation_type="approximate",
            description="k-point spacing inversely related to real-space lattice",
        ),
    ]

    @classmethod
    def get_relationships(cls) -> List[ParameterRelationship]:
        """Get all parameter relationships."""
        return cls.RELATIONSHIPS


class VaspExtractor:
    """Enhanced VASP extractor with symbolic constraint support.

    Extracts mathematical structures from VASP calculations with full
    symbolic constraint and parameter relationship tracking.

    Example:
        ```python
        extractor = VaspExtractor()
        schema = extractor.extract({
            "incar": "INCAR",
            "poscar": "POSCAR",
            "kpoints": "KPOINTS",
        })

        # Check constraint validation
        print(schema.validation_results)

        # Access symbolic constraints
        for constraint in schema.symbolic_constraints:
            print(f"{constraint.expression}: {constraint.satisfied}")
        ```
    """

    def __init__(self):
        self.incar_parser = IncarParser()
        self.poscar_parser = PoscarParser()
        self.kpoints_parser = KpointsParser()

        self.incar_params: Optional[Dict[str, Any]] = None
        self.structure: Optional[CrystalStructure] = None
        self.kpoints: Optional[KpointsData] = None
        self.validation_results: Optional[Dict[str, Any]] = None

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> MathSchema:
        """Extract mathematical structures from VASP files.

        Args:
            files: Dictionary with 'incar', 'poscar', 'kpaths'
            options: Optional extraction parameters

        Returns:
            MathSchema object with symbolic constraints
        """
        options = options or {}

        # Parse INCAR
        if "incar" in files:
            self.incar_params = self.incar_parser.parse(files["incar"])

        # Parse POSCAR
        if "poscar" in files:
            self.structure = self.poscar_parser.parse(files["poscar"])

        # Parse KPOINTS
        if "kpoints" in files:
            self.kpoints = self.kpoints_parser.parse(files["kpoints"])

        # Validate symbolic constraints
        # Convert IncarResult to dict of parameter values
        incar_dict = {}
        if self.incar_params:
            incar_dict = {
                name: param.value
                for name, param in self.incar_params.parameters.items()
            }

        self.validation_results = VaspSymbolicConstraints.validate_all_constraints(
            incar_dict, self.structure, self.kpoints
        )

        # Build Math Schema
        schema = MathSchema(
            schema_version="1.0.0",
            meta=MetaInfo(
                extracted_by="math-anything-vasp-v2",
                extractor_version="0.2.0",
                source_files={
                    "input": [f for f in files.values()],
                },
            ),
            mathematical_model=self._extract_mathematical_model(),
            numerical_method=self._extract_numerical_method(),
            computational_graph=self._extract_computational_graph(),
            conservation_properties=self._extract_conservation_properties(),
            raw_symbols=self._extract_raw_symbols(),
            # NEW: Symbolic constraints
            symbolic_constraints=self._extract_symbolic_constraints(),
        )

        return schema

    def _extract_mathematical_model(self) -> MathematicalModel:
        """Extract mathematical model with DFT equations."""
        model = MathematicalModel()

        # Kohn-Sham equations
        model.governing_equations = self._extract_kohn_sham_equations()

        # Periodic boundary conditions
        model.boundary_conditions = self._extract_boundary_conditions()

        # XC functional and pseudopotentials
        model.constitutive_relations = self._extract_constitutive_relations()

        return model

    def _extract_kohn_sham_equations(self) -> List[GoverningEquation]:
        """Extract Kohn-Sham DFT equations."""
        equations = []

        # Main Kohn-Sham equation
        ispin = self.incar_params.get_value("ISPIN", 1) if self.incar_params else 1

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
                    "spin_polarized": ispin == 2,
                },
                description="Density Functional Theory governing equations",
            )
        )

        # Charge density
        equations.append(
            GoverningEquation(
                id="charge_density",
                type="density_construction",
                name="Charge Density Construction",
                mathematical_form="n(r) = Σ f_i |ψ_i(r)|²",
                variables=["density", "occupation", "wavefunction"],
                parameters={
                    "occupation": (
                        "fermi_dirac"
                        if self.incar_params
                        and self.incar_params.get_value("ISMEAR", 0) >= 0
                        else "tetrahedron"
                    ),
                    "smearing_width": (
                        self.incar_params.get_value("SIGMA", 0.2)
                        if self.incar_params
                        else 0.2
                    ),
                },
                description="Electron density from occupied Kohn-Sham states",
            )
        )

        return equations

    def _extract_boundary_conditions(self) -> List[BoundaryCondition]:
        """Extract periodic boundary conditions."""
        bcs = []

        if self.structure:
            lattice = self.structure.lattice

            # Periodic BC on crystal lattice
            components = []
            for i in range(3):
                for j in range(3):
                    components.append(
                        TensorComponent(
                            index=[i + 1, j + 1],
                            value=str(lattice.vectors[i, j]),
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
                            "scale": lattice.scale,
                            "volume": lattice.volume,
                        },
                    },
                )
            )

            # Bloch theorem for k-points
            if self.kpoints and self.kpoints.mesh:
                bcs.append(
                    BoundaryCondition(
                        id="bloch_theorem",
                        type="quasi_periodic",
                        domain={
                            "geometric_region": "reciprocal_space",
                            "entity_type": "kpoint_mesh",
                        },
                        mathematical_object=MathematicalObject(
                            field="bloch_wavefunction",
                            tensor_rank=1,
                            tensor_form="ψ_{k+G}(r) = ψ_k(r) e^{iG·r}",
                            components=[
                                TensorComponent(
                                    index=[i + 1], value=str(n), unit="grid"
                                )
                                for i, n in enumerate(self.kpoints.mesh.subdivisions)
                            ],
                        ),
                        software_implementation={
                            "command": "KPOINTS",
                            "parameters": {
                                "grid": self.kpoints.mesh.subdivisions,
                                "mode": self.kpoints.mode.value,
                                "total_kpoints": self.kpoints.mesh.total_kpoints,
                            },
                        },
                    )
                )

        return bcs

    def _extract_constitutive_relations(self) -> List[Dict[str, Any]]:
        """Extract constitutive relations (XC functional, PAW)."""
        relations = []

        if not self.incar_params:
            return relations

        # Determine XC functional
        if self.incar_params.get_value("LHFCALC", False):
            xc_type = "hybrid"
            xc_name = "HSE06"
            form = "E_xc = E_xc^DFA + α(E_x^exact - E_x^DFA)"
        elif self.incar_params.get_value("GGA"):
            xc_type = "gga"
            xc_name = self.incar_params.get_value("GGA")
            form = f"E_xc^{xc_name} = ∫ f(n, ∇n) dr"
        elif self.incar_params.get_value("LDAU", False):
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
                    "ismear": self.incar_params.get_value("ISMEAR", 0),
                    "sigma": self.incar_params.get_value("SIGMA", 0.2),
                },
            }
        )

        # PAW
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
        """Extract numerical method (plane wave basis)."""
        method = NumericalMethod()

        if self.incar_params:
            method.discretization.space_discretization = "plane_wave_basis"
            method.discretization.time_integrator = "scf_iteration"
            method.discretization.order = 1

            method.solver = Solver(
                algorithm="scf_diagonalization",
                convergence_criterion="energy_difference",
                tolerance=self.incar_params.get_value("EDIFF", 1e-4),
                max_iterations=self.incar_params.get_value("NELM", 60),
            )

            method.parallelization = {
                "kpoint_parallel": True,
                "band_parallel": True,
                "plane_wave_parallel": True,
            }

        return method

    def _extract_computational_graph(self) -> ComputationalGraph:
        """Extract SCF computational graph."""
        graph = ComputationalGraph()

        # SCF nodes
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
                            f"energy_change < {self.incar_params.get('EDIFF', 1e-4)}"
                            if self.incar_params
                            else "energy_change < threshold"
                        ),
                        "max_iterations": (
                            self.incar_params.get_value("NELM", 60)
                            if self.incar_params
                            else 60
                        ),
                    },
                },
            )
        )

        # Edges
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

        # SCF loop
        graph.add_edge(
            ComputationalEdge(
                from_node="scf_convergence",
                to_node="hamiltonian_construction",
                data_type="convergence_status",
                dependency="feedback",
            )
        )

        return graph

    def _extract_conservation_properties(self) -> Dict[str, Any]:
        """Extract conservation properties."""
        props = {}

        props["charge"] = {
            "preserved": True,
            "mechanism": "number_of_electrons_fixed",
        }

        if self.incar_params and self.incar_params.get_value("ISPIN", 1) == 1:
            props["spin"] = {
                "preserved": True,
                "mechanism": "collinear_calculation",
            }

        props["energy_variational"] = {
            "preserved": True,
            "mechanism": "hohenberg_kohn_theorem",
            "note": "Ground state minimizes energy functional",
        }

        return props

    def _extract_raw_symbols(self) -> Dict[str, Any]:
        """Extract raw symbols from all input files."""
        symbols = {
            "incar": self.incar_params or {},
            "structure": self.structure.to_dict() if self.structure else {},
            "kpoints": self.kpoints.to_dict() if self.kpoints else {},
            "validation": self.validation_results or {},
        }
        return symbols

    def _extract_symbolic_constraints(self) -> List[SymbolicConstraint]:
        """Extract all symbolic constraints with validation status."""
        constraints = []

        # Add INCAR parameter constraints
        if self.incar_params:
            for param_name, param in self.incar_params.parameters.items():
                param_constraints = (
                    VaspSymbolicConstraints.get_constraints_for_parameter(param_name)
                )
                for constraint in param_constraints:
                    # Validate and update status
                    validation = VaspSymbolicConstraints._validate_constraint(
                        constraint, param_name, param.value
                    )
                    # Create a new constraint with validation result
                    constraint_dict = {
                        "expression": constraint.expression,
                        "description": constraint.description,
                    }
                    if validation["passed"]:
                        constraint_dict["description"] += " [SATISFIED]"
                    else:
                        constraint_dict["description"] += " [VIOLATED]"
                    constraints.append(SymbolicConstraint(**constraint_dict))

        # Add structure constraints
        if self.structure:
            for constraint in VaspSymbolicConstraints.STRUCTURE_CONSTRAINTS[
                "lattice_vectors"
            ]:
                if "det" in constraint.expression:
                    det = np.linalg.det(self.structure.lattice.vectors)
                    satisfied = abs(det) > 1e-10
                    constraints.append(
                        SymbolicConstraint(
                            expression=constraint.expression,
                            description=constraint.description
                            + (" [SATISFIED]" if satisfied else " [VIOLATED]"),
                        )
                    )

        # Add k-points constraints
        if self.kpoints and self.kpoints.mesh:
            total_k = self.kpoints.mesh.total_kpoints
            satisfied = total_k >= 1
            constraints.append(
                SymbolicConstraint(
                    expression="n_k >= 1",
                    description=(
                        f"At least one k-point required [SATISFIED: {total_k} >= 1]"
                        if satisfied
                        else "At least one k-point required [VIOLATED]"
                    ),
                )
            )

        return constraints
