"""Abaqus mathematical structure extractor for FEM simulations.

Extracts mathematical structures from Abaqus input files, including:
- Governing equations (linear elasticity, heat conduction, etc.)
- Constitutive relations (Hooke's law, material models)
- Boundary conditions (Dirichlet, Neumann, mixed)
- Numerical methods (Newton-Raphson, linear solver)
- Symbolic constraints (material stability conditions)
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

from .parser import AbaqusInputParser
from .parser import BoundaryCondition as AbaqusBC
from .parser import Material, Step


@dataclass
class FEMSettings:
    """FEM-specific settings extracted from Abaqus input."""

    material: Optional[Material] = None
    steps: List[Step] = None
    boundary_conditions: List[AbaqusBC] = None
    elements: List[str] = None
    nodes: int = 0
    analysis_type: str = "static"  # static, dynamic, heat, etc.
    nlgeom: bool = False

    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.boundary_conditions is None:
            self.boundary_conditions = []
        if self.elements is None:
            self.elements = []


class AbaqusExtractor:
    """Extracts mathematical structures from Abaqus FEM simulations.

    Maps Abaqus input files to Math Schema v1.0 representation, including:
    - Governing PDEs (elasticity, heat conduction, etc.)
    - Constitutive relations (Hooke's law, nonlinear materials)
    - Boundary conditions with tensor expressions
    - Symbolic constraints (material stability, convergence criteria)

    Example:
        ```python
        extractor = AbaqusExtractor()
        schema = extractor.extract({"input": "beam.inp"})
        ```
    """

    def __init__(self):
        self.parser = AbaqusInputParser()
        self.settings: Optional[FEMSettings] = None

    def extract(
        self, files: Dict[str, str], options: Dict[str, Any] = None
    ) -> MathSchema:
        """Extract mathematical structures from Abaqus files.

        Args:
            files: Dictionary with 'input' (required).
            options: Optional extraction parameters.

        Returns:
            MathSchema object with FEM mathematical structures.
        """
        options = options or {}

        input_path = files.get("input")
        if not input_path:
            raise ValueError("Input file required")

        # Parse input file
        commands = self.parser.parse_file(input_path)
        self.settings = self._extract_settings(commands)

        # Build Math Schema
        schema = MathSchema(
            schema_version="1.0.0",
            meta=MetaInfo(
                extracted_by="math-anything-abaqus",
                extractor_version="0.2.0",
                source_files={"input": [input_path]},
            ),
            mathematical_model=self._extract_mathematical_model(),
            numerical_method=self._extract_numerical_method(),
            computational_graph=self._extract_computational_graph(),
            conservation_properties=self._extract_conservation_properties(),
            raw_symbols=self._extract_raw_symbols(),
        )

        # Add symbolic constraints (material stability, convergence criteria)
        schema.symbolic_constraints = self._extract_symbolic_constraints()

        return schema

    def _extract_settings(self, commands: List[Dict[str, Any]]) -> FEMSettings:
        """Extract FEM-specific settings from parsed commands."""
        settings = FEMSettings()

        for cmd in commands:
            keyword = cmd.get("keyword", "").upper()
            data = cmd.get("data", [])

            if keyword == "*MATERIAL":
                # Parse material properties
                for line in data:
                    if line.startswith("*ELASTIC"):
                        # Next line should have E, nu
                        idx = data.index(line)
                        if idx + 1 < len(data):
                            values = data[idx + 1].split(",")
                            if len(values) >= 2:
                                settings.material = Material(
                                    name="elastic",
                                    youngs_modulus=float(values[0]),
                                    poisson_ratio=float(values[1]),
                                )

            elif keyword == "*STEP":
                # Parse step definition
                step = Step()
                for line in data:
                    if "nlgeom=YES" in line.upper():
                        settings.nlgeom = True
                        step.nlgeom = True
                settings.steps.append(step)

            elif keyword == "*BOUNDARY":
                # Parse boundary conditions
                for line in data:
                    if not line.startswith("*"):
                        parts = line.split(",")
                        if len(parts) >= 3:
                            settings.boundary_conditions.append(
                                AbaqusBC(
                                    node_set=parts[0].strip(),
                                    dof_start=int(parts[1]),
                                    dof_end=int(parts[2]),
                                    value=float(parts[3]) if len(parts) > 3 else 0.0,
                                )
                            )

            elif keyword == "*ELEMENT":
                # Track element types
                for line in data:
                    if "type=" in line.upper():
                        elem_type = line.upper().split("TYPE=")[1].split(",")[0].strip()
                        settings.elements.append(elem_type)

        return settings

    def _extract_mathematical_model(self) -> MathematicalModel:
        """Extract mathematical model for FEM."""
        model = MathematicalModel()

        # Governing equations based on analysis type
        model.governing_equations = self._extract_governing_equations()

        # Boundary conditions
        model.boundary_conditions = self._extract_boundary_conditions()

        # Constitutive relations (material models)
        model.constitutive_relations = self._extract_constitutive_relations()

        # Parameter relationships (material property relationships)
        model.parameter_relationships = self._extract_parameter_relationships()

        return model

    def _extract_governing_equations(self) -> List[GoverningEquation]:
        """Extract governing equations for FEM.

        For linear elasticity:
        - Equilibrium: ∇·σ + b = 0
        - Constitutive: σ = C:ε
        - Strain-displacement: ε = ½(∇u + (∇u)ᵀ)
        """
        equations = []

        # Equilibrium equation
        equations.append(
            GoverningEquation(
                id="equilibrium",
                type="partial_differential_equation",
                name="Equilibrium Equation",
                mathematical_form="∇·σ + b = 0",
                variables=["stress", "body_force", "displacement"],
                parameters={"form": "vector_pde"},
                description="Static equilibrium in the absence of inertia",
            )
        )

        # Strain-displacement relation
        equations.append(
            GoverningEquation(
                id="strain_displacement",
                type="kinematic_relation",
                name="Strain-Displacement Relation",
                mathematical_form="ε = ½(∇u + (∇u)ᵀ)",
                variables=["strain", "displacement"],
                parameters={"linearity": "linear"},
                description="Infinitesimal strain tensor",
            )
        )

        # Constitutive (Hooke's law for isotropic material)
        if self.settings and self.settings.material:
            equations.append(
                GoverningEquation(
                    id="constitutive",
                    type="constitutive_relation",
                    name="Hooke's Law (Isotropic)",
                    mathematical_form="σ = λ tr(ε) I + 2με",
                    variables=["stress", "strain", "lambda", "mu"],
                    parameters={
                        "isotropic": True,
                        "linear": True,
                    },
                    description="Linear elastic constitutive relation for isotropic materials",
                )
            )

        # Weak form / Variational principle
        equations.append(
            GoverningEquation(
                id="weak_form",
                type="variational_principle",
                name="Principle of Virtual Work",
                mathematical_form="∫_Ω σ:δε dΩ = ∫_Ω b·δu dΩ + ∫_Γ t·δu dΓ",
                variables=[
                    "stress",
                    "virtual_strain",
                    "body_force",
                    "traction",
                    "virtual_displacement",
                ],
                parameters={"form": "integral_equation"},
                description="Weak form of equilibrium equation used in FEM",
            )
        )

        return equations

    def _extract_constitutive_relations(self) -> List[Dict[str, Any]]:
        """Extract constitutive relations (material models)."""
        relations = []

        if self.settings and self.settings.material:
            mat = self.settings.material
            relations.append(
                {
                    "type": "elastic",
                    "name": "linear_isotropic",
                    "form": "σ = C : ε",
                    "parameters": {
                        "E": mat.youngs_modulus,
                        "nu": mat.poisson_ratio,
                        "G": mat.youngs_modulus / (2 * (1 + mat.poisson_ratio)),
                    },
                    "stiffness_tensor": "C_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)",
                }
            )

        return relations

    def _extract_parameter_relationships(self) -> List[ParameterRelationship]:
        """Extract mathematical relationships between material parameters.

        These are the symbolic equations that define elastic constants.
        """
        relationships = []

        if self.settings and self.settings.material:
            mat = self.settings.material
            E = mat.youngs_modulus
            nu = mat.poisson_ratio

            # Lamé parameters from E and ν
            relationships.append(
                ParameterRelationship(
                    name="lame_first_parameter",
                    expression="lambda = E*nu / ((1+nu)*(1-2*nu))",
                    variables=["lambda", "E", "nu"],
                    relation_type="equality",
                    description="First Lamé parameter from Young's modulus and Poisson's ratio",
                    physical_meaning="Material stiffness in bulk deformation",
                )
            )

            relationships.append(
                ParameterRelationship(
                    name="lame_second_parameter",
                    expression="mu = E / (2*(1+nu))",
                    variables=["mu", "E", "nu"],
                    relation_type="equality",
                    description="Second Lamé parameter (shear modulus) from E and ν",
                    physical_meaning="Material stiffness in shear deformation",
                )
            )

            relationships.append(
                ParameterRelationship(
                    name="bulk_modulus",
                    expression="K = E / (3*(1-2*nu))",
                    variables=["K", "E", "nu"],
                    relation_type="equality",
                    description="Bulk modulus from Young's modulus and Poisson's ratio",
                    physical_meaning="Resistance to uniform compression",
                )
            )

            # Plane stress vs plane strain relations
            relationships.append(
                ParameterRelationship(
                    name="effective_modulus_plane_stress",
                    expression="E' = E / (1-nu^2)",
                    variables=["E_prime", "E", "nu"],
                    relation_type="equality",
                    description="Effective modulus for plane stress conditions",
                    physical_meaning="Apparent stiffness in 2D plane stress analysis",
                )
            )

        return relationships

    def _extract_symbolic_constraints(self) -> List[SymbolicConstraint]:
        """Extract symbolic mathematical constraints for FEM.

        These ensure material stability and solution validity.
        """
        constraints = []

        # Material stability constraints
        if self.settings and self.settings.material:
            mat = self.settings.material

            # Young's modulus must be positive
            constraints.append(
                SymbolicConstraint(
                    expression="E > 0",
                    description="Young's modulus must be positive for stable material",
                    variables=["E"],
                    confidence=1.0,
                    inferred_from="material_stability",
                )
            )

            # Poisson's ratio bounds for positive definite stiffness
            constraints.append(
                SymbolicConstraint(
                    expression="-1 < nu < 0.5",
                    description="Poisson's ratio bounds for positive definite elasticity tensor",
                    variables=["nu"],
                    confidence=1.0,
                    inferred_from="stability_requirement",
                )
            )

            # Check actual values
            if mat.youngs_modulus > 0:
                constraints.append(
                    SymbolicConstraint(
                        expression=f"E ({mat.youngs_modulus}) > 0 ✓",
                        description="Young's modulus positive - constraint satisfied",
                        variables=["E"],
                        confidence=1.0,
                        inferred_from="validation",
                    )
                )

            if -1 < mat.poisson_ratio < 0.5:
                constraints.append(
                    SymbolicConstraint(
                        expression=f"-1 < nu ({mat.poisson_ratio}) < 0.5 ✓",
                        description="Poisson's ratio in valid range",
                        variables=["nu"],
                        confidence=1.0,
                        inferred_from="validation",
                    )
                )
            else:
                constraints.append(
                    SymbolicConstraint(
                        expression=f"nu ({mat.poisson_ratio}) outside valid range (-1, 0.5)",
                        description="WARNING: Poisson's ratio may violate stability",
                        variables=["nu"],
                        confidence=1.0,
                        inferred_from="validation",
                    )
                )

            # Positive definite constraints on Lamé parameters
            constraints.append(
                SymbolicConstraint(
                    expression="mu > 0",
                    description="Shear modulus must be positive",
                    variables=["mu"],
                    confidence=1.0,
                    inferred_from="material_stability",
                )
            )

            constraints.append(
                SymbolicConstraint(
                    expression="3*lambda + 2*mu > 0",
                    description="Bulk modulus must be positive (3λ + 2μ > 0)",
                    variables=["lambda", "mu"],
                    confidence=1.0,
                    inferred_from="positive_definite_requirement",
                )
            )

        # FEM solution convergence constraints
        constraints.append(
            SymbolicConstraint(
                expression="det(K) != 0",
                description="Stiffness matrix must be non-singular (adequate constraints)",
                variables=["K"],
                confidence=0.9,
                inferred_from="fem_solution_existence",
            )
        )

        # Element quality constraints (for mesh convergence)
        constraints.append(
            SymbolicConstraint(
                expression="aspect_ratio < threshold",
                description="Element aspect ratio should be bounded for accuracy",
                variables=["aspect_ratio"],
                confidence=0.8,
                inferred_from="mesh_quality",
            )
        )

        return constraints

    def _extract_boundary_conditions(self) -> List[BoundaryCondition]:
        """Extract boundary conditions with mathematical expressions."""
        bcs = []

        for bc in self.settings.boundary_conditions if self.settings else []:
            # Determine BC type
            if bc.value == 0.0:
                bc_type = "dirichlet"
                bc_form = f"u_{bc.dof_start} = 0"
            else:
                bc_type = "neumann"
                bc_form = f"t_{bc.dof_start} = {bc.value}"

            bcs.append(
                BoundaryCondition(
                    id=f"bc_{bc.node_set}_{bc.dof_start}",
                    type=bc_type,
                    domain={
                        "geometric_region": bc.node_set,
                        "entity_type": "node_set",
                    },
                    mathematical_object=MathematicalObject(
                        field="displacement" if bc_type == "dirichlet" else "traction",
                        tensor_rank=1 if bc.dof_start == bc.dof_end else 2,
                        tensor_form=bc_form,
                    ),
                    software_implementation={
                        "command": "*Boundary" if bc_type == "dirichlet" else "*Cload",
                        "node_set": bc.node_set,
                        "dof": f"{bc.dof_start}-{bc.dof_end}",
                        "value": bc.value,
                    },
                )
            )

        return bcs

    def _extract_numerical_method(self) -> NumericalMethod:
        """Extract numerical method for FEM."""
        method = NumericalMethod()

        # Spatial discretization
        if self.settings and self.settings.elements:
            method.discretization.space_discretization = (
                f"FEM_{self.settings.elements[0]}"
            )

        # Solver for linear system K u = F
        if self.settings and not self.settings.nlgeom:
            method.solver.algorithm = "direct_sparse"  # Default for linear static
            method.solver.convergence_criterion = "residual_norm"
            method.solver.tolerance = 1e-6
        else:
            method.solver.algorithm = "newton_raphson"
            method.solver.convergence_criterion = "force_residual"
            method.solver.max_iterations = 16

        return method

    def _extract_computational_graph(self) -> ComputationalGraph:
        """Extract computational graph for FEM."""
        graph = ComputationalGraph()

        # Assembly node
        graph.add_node(
            ComputationalNode(
                id="assembly",
                type="matrix_assembly",
                math_semantics={
                    "operator_type": "stiffness_assembly",
                    "updates": {
                        "target": "K",
                        "mode": UpdateMode.EXPLICIT_UPDATE.value,
                    },
                },
            )
        )

        # Solver node
        graph.add_node(
            ComputationalNode(
                id="solver",
                type="linear_solver",
                math_semantics={
                    "operator_type": "sparse_direct_solver",
                    "updates": {
                        "target": "displacement",
                        "mode": UpdateMode.EXPLICIT_UPDATE.value,
                    },
                },
            )
        )

        # Post-processing
        graph.add_node(
            ComputationalNode(
                id="post_process",
                type="stress_recovery",
                math_semantics={
                    "operator_type": "strain_stress_computation",
                    "updates": {
                        "target": "stress",
                        "mode": UpdateMode.EXPLICIT_UPDATE.value,
                    },
                },
            )
        )

        # Edges
        graph.add_edge(
            ComputationalEdge(
                from_node="assembly",
                to_node="solver",
                data_type="sparse_matrix",
                dependency="solve",
            )
        )

        graph.add_edge(
            ComputationalEdge(
                from_node="solver",
                to_node="post_process",
                data_type="displacement_vector",
                dependency="compute",
            )
        )

        return graph

    def _extract_conservation_properties(self) -> Dict[str, Any]:
        """Extract conservation properties for FEM."""
        props = {}

        # Linear static analysis preserves nothing special
        # But the weak form ensures equilibrium is satisfied
        props["equilibrium"] = {"preserved": True, "mechanism": "weak_formulation"}

        # Check if conservative loading (dead loads are conservative)
        props["conservative_system"] = {
            "preserved": True,
            "mechanism": "potential_energy",
        }

        return props

    def _extract_raw_symbols(self) -> Dict[str, Any]:
        """Extract raw symbols from input."""
        symbols = {
            "elements": self.settings.elements if self.settings else [],
            "material": (
                self.settings.material.__dict__
                if self.settings and self.settings.material
                else {}
            ),
            "steps": [
                s.__dict__ for s in (self.settings.steps if self.settings else [])
            ],
            "boundary_conditions": (
                len(self.settings.boundary_conditions) if self.settings else 0
            ),
            "nlgeom": self.settings.nlgeom if self.settings else False,
        }
        return symbols


@dataclass
class Material:
    """Elastic material properties."""

    name: str
    youngs_modulus: float
    poisson_ratio: float


@dataclass
class Step:
    """Analysis step."""

    nlgeom: bool = False
    analysis_type: str = "static"


@dataclass
class BoundaryCondition:
    """Boundary condition definition."""

    node_set: str
    dof_start: int
    dof_end: int
    value: float = 0.0
