"""Abaqus mathematical structure extractor for FEM simulations.

Extracts mathematical structures from Abaqus input files, including:
- Governing equations (linear elasticity, heat conduction, etc.)
- Constitutive relations (Hooke's law, material models)
- Boundary conditions (Dirichlet, Neumann, mixed)
- Numerical methods (Newton-Raphson, linear solver)
- Symbolic constraints (material stability conditions)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from math_anything.schemas import (
    BoundaryCondition,
    ComputationalEdge,
    ComputationalGraph,
    ComputationalNode,
    GoverningEquation,
    MathematicalModel,
    MathematicalObject,
    MathSchema,
    NumericalMethod,
    ParameterRelationship,
    SymbolicConstraint,
    UpdateMode,
)

from engines.base import BaseEngineExtractor

from .parser import AbaqusInputParser


@dataclass
class Material:
    """Material properties."""

    name: str = ""
    density: Optional[float] = None
    youngs_modulus: Optional[float] = None
    poisson_ratio: Optional[float] = None
    model_type: str = "unknown"  # elastic, plastic, cdpm, etc.
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """Analysis step."""

    name: str = ""
    nlgeom: bool = False
    analysis_type: str = "static"  # static, dynamic, heat, coupled
    max_increments: int = 100
    initial_inc: Optional[float] = None
    total_time: Optional[float] = None
    min_inc: Optional[float] = None
    max_inc: Optional[float] = None


@dataclass
class AbaqusBC:
    """Boundary condition definition."""

    node_set: str = ""
    dof_start: int = 1
    dof_end: int = 1
    value: float = 0.0
    bc_type: str = "displacement"  # displacement, velocity, acceleration, temperature


@dataclass
class FEMSettings:
    """FEM-specific settings extracted from Abaqus input."""

    materials: List[Material] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    boundary_conditions: List[AbaqusBC] = field(default_factory=list)
    elements: List[str] = field(default_factory=list)
    nodes: int = 0
    analysis_type: str = "static"
    nlgeom: bool = False
    cards: Dict[str, List[str]] = field(default_factory=dict)


class AbaqusExtractor(BaseEngineExtractor):
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

    @property
    def engine_name(self) -> str:
        return "abaqus"

    @property
    def extractor_version(self) -> str:
        return "0.2.0"

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
        cards = self.parser.parse_file(input_path)
        self.settings = self._extract_settings(cards)

        # Build Math Schema via base class
        source_files = {"input": [input_path]}
        return self.build_schema(source_files)

    # ------------------------------------------------------------------
    # 输入解析
    # ------------------------------------------------------------------

    def _extract_settings(self, cards: Dict[str, List[str]]) -> FEMSettings:
        """Extract FEM-specific settings from parsed cards."""
        settings = FEMSettings()
        settings.cards = cards

        # Parse materials
        settings.materials = self._parse_materials(cards)

        # Parse steps
        settings.steps = self._parse_steps(cards)

        # Parse boundary conditions
        settings.boundary_conditions = self._parse_bcs(cards)

        # Parse element types
        settings.elements = self._parse_elements(cards)

        # Count nodes
        settings.nodes = len(cards.get("NODE", []))

        # Determine if any step uses nlgeom
        settings.nlgeom = any(s.nlgeom for s in settings.steps)
        if settings.steps:
            settings.analysis_type = settings.steps[0].analysis_type

        return settings

    def _parse_materials(self, cards: Dict[str, List[str]]) -> List[Material]:
        """Parse *Material cards and associated property cards."""
        materials = []
        lines = self._flatten_cards(cards)

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.upper().startswith("*MATERIAL"):
                # Extract material name
                name = ""
                parts = line.split(",")
                for p in parts[1:]:
                    if "name=" in p.lower():
                        name = p.split("=")[1].strip()
                        break

                mat = Material(name=name)
                i += 1

                # Read property cards until next * card
                while i < len(lines) and not lines[i].strip().startswith("*"):
                    i += 1

                # Now read property cards
                while i < len(lines):
                    prop_line = lines[i].strip()
                    if prop_line.startswith("*") and not prop_line.upper().startswith("*MATERIAL"):
                        prop_name = prop_line.split(",")[0][1:].upper()

                        if prop_name == "DENSITY":
                            i += 1
                            if i < len(lines) and not lines[i].strip().startswith("*"):
                                try:
                                    mat.density = float(lines[i].strip().split(",")[0])
                                except ValueError:
                                    pass
                            i += 1

                        elif prop_name == "ELASTIC":
                            i += 1
                            if i < len(lines) and not lines[i].strip().startswith("*"):
                                vals = lines[i].strip().split(",")
                                try:
                                    if len(vals) >= 2:
                                        mat.youngs_modulus = float(vals[0])
                                        mat.poisson_ratio = float(vals[1])
                                        mat.model_type = "elastic"
                                except ValueError:
                                    pass
                            i += 1

                        elif "PLASTICITY" in prop_name or "PLASTIC" in prop_name:
                            mat.model_type = "plastic"
                            i += 1
                            # Skip data lines
                            while i < len(lines) and not lines[i].strip().startswith("*"):
                                i += 1

                        elif "DAMPED" in prop_name or "DAMAGE" in prop_name:
                            if mat.model_type == "elastic":
                                mat.model_type = "cdp"  # concrete damaged plasticity
                            i += 1
                            while i < len(lines) and not lines[i].strip().startswith("*"):
                                i += 1

                        elif prop_name in ("EXPANSION", "CONDUCTIVITY", "SPECIFIC HEAT"):
                            i += 1
                            while i < len(lines) and not lines[i].strip().startswith("*"):
                                i += 1

                        else:
                            # Unknown property - skip data lines
                            i += 1
                            while i < len(lines) and not lines[i].strip().startswith("*"):
                                i += 1
                    elif prop_line.upper().startswith("*MATERIAL"):
                        break
                    else:
                        i += 1

                materials.append(mat)
            else:
                i += 1

        return materials

    def _parse_steps(self, cards: Dict[str, List[str]]) -> List[Step]:
        """Parse *Step cards and associated control cards."""
        steps = []
        lines = self._flatten_cards(cards)

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.upper().startswith("*STEP"):
                # Extract step parameters
                name = ""
                nlgeom = False
                max_inc = 100

                parts = line.split(",")
                for p in parts[1:]:
                    pl = p.lower().strip()
                    if "name=" in pl:
                        name = p.split("=")[1].strip()
                    elif "nlgeom=" in pl:
                        nlgeom = "yes" in pl
                    elif "inc=" in pl:
                        try:
                            max_inc = int(p.split("=")[1].strip())
                        except ValueError:
                            pass

                step = Step(name=name, nlgeom=nlgeom, max_increments=max_inc)
                i += 1

                # Read step contents until *End Step
                while i < len(lines):
                    step_line = lines[i].strip()
                    if step_line.upper().startswith("*END STEP"):
                        i += 1
                        break

                    if step_line.upper().startswith("*STATIC"):
                        step.analysis_type = "static"
                        i += 1
                        if i < len(lines) and not lines[i].strip().startswith("*"):
                            vals = lines[i].strip().split(",")
                            try:
                                if len(vals) >= 1:
                                    step.initial_inc = float(vals[0])
                                if len(vals) >= 2:
                                    step.total_time = float(vals[1])
                                if len(vals) >= 3:
                                    step.min_inc = float(vals[2])
                                if len(vals) >= 4:
                                    step.max_inc = float(vals[3])
                            except ValueError:
                                pass
                            i += 1
                        continue

                    if step_line.upper().startswith("*DYNAMIC"):
                        step.analysis_type = "dynamic"
                        i += 1
                        continue

                    if step_line.upper().startswith("*HEAT"):
                        step.analysis_type = "heat"
                        i += 1
                        continue

                    if step_line.upper().startswith("*COUPLED"):
                        step.analysis_type = "coupled"
                        i += 1
                        continue

                    i += 1

                steps.append(step)
            else:
                i += 1

        return steps

    def _parse_bcs(self, cards: Dict[str, List[str]]) -> List[AbaqusBC]:
        """Parse *Boundary cards."""
        bcs = []
        lines = self._flatten_cards(cards)

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.upper().startswith("*BOUNDARY"):
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("*"):
                    data = lines[i].strip()
                    if data:
                        parts = [p.strip() for p in data.split(",")]
                        if len(parts) >= 3:
                            try:
                                bc = AbaqusBC(
                                    node_set=parts[0],
                                    dof_start=int(parts[1]),
                                    dof_end=int(parts[2]),
                                    value=float(parts[3]) if len(parts) > 3 and parts[3] else 0.0,
                                )
                                bcs.append(bc)
                            except (ValueError, IndexError):
                                pass
                    i += 1
            else:
                i += 1

        return bcs

    def _parse_elements(self, cards: Dict[str, List[str]]) -> List[str]:
        """Parse *Element cards to extract element types."""
        elements = []
        headers = self.parser.headers if hasattr(self.parser, "headers") else {}
        for hdr in headers.get("ELEMENT", []):
            lup = hdr.upper().strip()
            if "TYPE=" in lup:
                try:
                    elem_type = lup.split("TYPE=")[1].split(",")[0].strip()
                    if elem_type and elem_type not in elements:
                        elements.append(elem_type)
                except IndexError:
                    pass
        return elements

    def _flatten_cards(self, cards: Dict[str, List[str]]) -> List[str]:
        """Reconstruct ordered line list from parsed cards, preserving headers."""
        ordered_keys = [
            "HEADING", "PREPRINT", "PART", "END PART",
            "ASSEMBLY", "INSTANCE", "END INSTANCE", "NODE", "ELEMENT",
            "NSET", "ELSET", "SURFACE", "CONSTRAINT", "COUPLING",
            "END ASSEMBLY", "MATERIAL", "DENSITY", "ELASTIC", "PLASTIC",
            "EXPANSION", "CONDUCTIVITY", "AMPLITUDE", "STEP", "STATIC",
            "DYNAMIC", "HEAT TRANSFER", "BOUNDARY", "CLOAD", "DLOAD",
            "OUTPUT", "NODE OUTPUT", "ELEMENT OUTPUT", "END STEP",
        ]
        lines = []
        seen = set()
        headers = self.parser.headers if hasattr(self.parser, "headers") else {}
        for key in ordered_keys:
            if key in cards:
                # Use original header line if available, otherwise synthetic
                hdrs = headers.get(key, [])
                if hdrs:
                    lines.append(hdrs[0])
                else:
                    lines.append(f"*{key}")
                lines.extend(cards[key])
                seen.add(key)
        # Append any remaining keys not in ordered list
        for key, data in cards.items():
            if key not in seen:
                hdrs = headers.get(key, [])
                if hdrs:
                    lines.append(hdrs[0])
                else:
                    lines.append(f"*{key}")
                lines.extend(data)
        return lines

    # ------------------------------------------------------------------
    # 数学模型管线（覆写以添加 parameter_relationships）
    # ------------------------------------------------------------------

    def _extract_mathematical_model(self) -> MathematicalModel:
        """Extract mathematical model for FEM."""
        model = MathematicalModel()
        model.governing_equations = self._extract_governing_equations()
        model.boundary_conditions = self._extract_boundary_conditions()
        model.constitutive_relations = self._extract_constitutive_relations()
        model.parameter_relationships = self._extract_parameter_relationships()
        return model

    # ------------------------------------------------------------------
    # 控制方程
    # ------------------------------------------------------------------

    def _extract_governing_equations(self) -> List[GoverningEquation]:
        """Extract governing equations for FEM."""
        equations = []
        analysis = self.settings.analysis_type if self.settings else "static"

        if analysis in ("static", "dynamic"):
            # Equilibrium equation
            equations.append(
                GoverningEquation(
                    id="equilibrium",
                    type="partial_differential_equation",
                    name="Equilibrium Equation",
                    mathematical_form="div(sigma) + b = rho * a",
                    variables=["stress", "body_force", "displacement", "density", "acceleration"],
                    parameters={"form": "vector_pde"},
                    description="Cauchy momentum balance" if analysis == "dynamic" else "Static equilibrium in the absence of inertia",
                )
            )

            # Strain-displacement relation
            if self.settings and self.settings.nlgeom:
                equations.append(
                    GoverningEquation(
                        id="strain_displacement",
                        type="kinematic_relation",
                        name="Green-Lagrange Strain",
                        mathematical_form="E = 1/2 (grad(u) + grad(u)^T + grad(u)^T * grad(u))",
                        variables=["strain", "displacement"],
                        parameters={"linearity": "nonlinear"},
                        description="Finite strain tensor for geometrically nonlinear analysis",
                    )
                )
            else:
                equations.append(
                    GoverningEquation(
                        id="strain_displacement",
                        type="kinematic_relation",
                        name="Strain-Displacement Relation",
                        mathematical_form="epsilon = 1/2 (grad(u) + grad(u)^T)",
                        variables=["strain", "displacement"],
                        parameters={"linearity": "linear"},
                        description="Infinitesimal strain tensor",
                    )
                )

            # Constitutive
            mat = self.settings.materials[0] if (self.settings and self.settings.materials) else None
            if mat and mat.model_type == "elastic":
                equations.append(
                    GoverningEquation(
                        id="constitutive",
                        type="constitutive_relation",
                        name="Hooke's Law (Isotropic)",
                        mathematical_form="sigma = lambda tr(epsilon) I + 2 mu epsilon",
                        variables=["stress", "strain", "lambda", "mu"],
                        parameters={"isotropic": True, "linear": True},
                        description="Linear elastic constitutive relation for isotropic materials",
                    )
                )
            elif mat and mat.model_type in ("plastic", "cdp"):
                equations.append(
                    GoverningEquation(
                        id="constitutive",
                        type="constitutive_relation",
                        name="Elasto-Plasticity",
                        mathematical_form="sigma = C : (epsilon - epsilon_p)",
                        variables=["stress", "strain", "plastic_strain"],
                        parameters={"isotropic": False, "linear": False},
                        description="Rate-independent elasto-plastic constitutive relation with yield surface",
                    )
                )

            # Weak form
            equations.append(
                GoverningEquation(
                    id="weak_form",
                    type="variational_principle",
                    name="Principle of Virtual Work",
                    mathematical_form="int_Omega sigma : delta_epsilon dOmega = int_Omega b . delta_u dOmega + int_Gamma t . delta_u dGamma",
                    variables=["stress", "virtual_strain", "body_force", "traction", "virtual_displacement"],
                    parameters={"form": "integral_equation"},
                    description="Weak form of equilibrium equation used in FEM",
                )
            )

        elif analysis == "heat":
            equations.append(
                GoverningEquation(
                    id="heat_conduction",
                    type="partial_differential_equation",
                    name="Heat Conduction Equation",
                    mathematical_form="div(k grad(T)) + q = rho c dT/dt",
                    variables=["temperature", "thermal_conductivity", "heat_source", "density", "specific_heat"],
                    parameters={"form": "scalar_pde"},
                    description="Fourier heat conduction with transient term",
                )
            )

        elif analysis == "coupled":
            equations.append(
                GoverningEquation(
                    id="thermomechanical",
                    type="partial_differential_equation",
                    name="Coupled Thermomechanics",
                    mathematical_form="div(sigma) + b = rho a; div(k grad(T)) + r = rho c dT/dt + alpha T0 tr(depsilon/dt)",
                    variables=["stress", "temperature", "displacement"],
                    parameters={"form": "coupled_pde"},
                    description="Coupled thermal-mechanical analysis with thermal expansion feedback",
                )
            )

        return equations

    # ------------------------------------------------------------------
    # 本构关系
    # ------------------------------------------------------------------

    def _extract_constitutive_relations(self) -> List[Dict[str, Any]]:
        """Extract constitutive relations (material models)."""
        relations = []

        for mat in (self.settings.materials if self.settings else []):
            if mat.model_type == "elastic" and mat.youngs_modulus is not None:
                relations.append({
                    "type": "elastic",
                    "name": "linear_isotropic",
                    "form": "sigma = C : epsilon",
                    "parameters": {
                        "E": mat.youngs_modulus,
                        "nu": mat.poisson_ratio,
                        "G": mat.youngs_modulus / (2 * (1 + (mat.poisson_ratio or 0))),
                    },
                    "stiffness_tensor": "C_ijkl = lambda delta_ij delta_kl + mu (delta_ik delta_jl + delta_il delta_jk)",
                })
            elif mat.model_type in ("plastic", "cdp"):
                relations.append({
                    "type": mat.model_type,
                    "name": "elasto_plastic",
                    "form": "sigma = C : (epsilon - epsilon_p) with yield condition f(sigma, q) <= 0",
                    "parameters": {"density": mat.density},
                })
            elif mat.density is not None:
                relations.append({
                    "type": "general",
                    "name": mat.name or "material",
                    "form": "unknown",
                    "parameters": {"density": mat.density},
                })

        return relations

    # ------------------------------------------------------------------
    # 参数关系
    # ------------------------------------------------------------------

    def _extract_parameter_relationships(self) -> List[ParameterRelationship]:
        """Extract mathematical relationships between material parameters."""
        relationships = []

        for mat in (self.settings.materials if self.settings else []):
            if mat.youngs_modulus is None or mat.poisson_ratio is None:
                continue

            relationships.append(
                ParameterRelationship(
                    name="lame_first_parameter",
                    expression="lambda = E*nu / ((1+nu)*(1-2*nu))",
                    variables=["lambda", "E", "nu"],
                    relation_type="equality",
                    description="First Lame parameter from Young's modulus and Poisson's ratio",
                    physical_meaning="Material stiffness in bulk deformation",
                )
            )

            relationships.append(
                ParameterRelationship(
                    name="lame_second_parameter",
                    expression="mu = E / (2*(1+nu))",
                    variables=["mu", "E", "nu"],
                    relation_type="equality",
                    description="Second Lame parameter (shear modulus) from E and nu",
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

        return relationships

    # ------------------------------------------------------------------
    # 边界条件
    # ------------------------------------------------------------------

    def _extract_boundary_conditions(self) -> List[BoundaryCondition]:
        """Extract boundary conditions with mathematical expressions."""
        bcs = []

        for bc in (self.settings.boundary_conditions if self.settings else []):
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

    # ------------------------------------------------------------------
    # 符号约束
    # ------------------------------------------------------------------

    def _enrich_schema(self, schema: MathSchema) -> None:
        """添加 symbolic_constraints。"""
        schema.symbolic_constraints = self._extract_symbolic_constraints()

    def _extract_symbolic_constraints(self) -> List[SymbolicConstraint]:
        """Extract symbolic mathematical constraints for FEM."""
        constraints = []

        for mat in (self.settings.materials if self.settings else []):
            if mat.youngs_modulus is not None:
                constraints.append(
                    SymbolicConstraint(
                        expression="E > 0",
                        description="Young's modulus must be positive for stable material",
                        variables=["E"],
                        confidence=1.0,
                        inferred_from="material_stability",
                    )
                )
                if mat.youngs_modulus > 0:
                    constraints.append(
                        SymbolicConstraint(
                            expression=f"E ({mat.youngs_modulus}) > 0 ok",
                            description="Young's modulus positive - constraint satisfied",
                            variables=["E"],
                            confidence=1.0,
                            inferred_from="validation",
                        )
                    )

            if mat.poisson_ratio is not None:
                constraints.append(
                    SymbolicConstraint(
                        expression="-1 < nu < 0.5",
                        description="Poisson's ratio bounds for positive definite elasticity tensor",
                        variables=["nu"],
                        confidence=1.0,
                        inferred_from="stability_requirement",
                    )
                )
                if -1 < mat.poisson_ratio < 0.5:
                    constraints.append(
                        SymbolicConstraint(
                            expression=f"-1 < nu ({mat.poisson_ratio}) < 0.5 ok",
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

        # FEM solution existence constraints
        constraints.append(
            SymbolicConstraint(
                expression="det(K) != 0",
                description="Stiffness matrix must be non-singular (adequate constraints)",
                variables=["K"],
                confidence=0.9,
                inferred_from="fem_solution_existence",
            )
        )

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

    # ------------------------------------------------------------------
    # 数值方法
    # ------------------------------------------------------------------

    def _extract_numerical_method(self) -> NumericalMethod:
        """Extract numerical method for FEM."""
        method = NumericalMethod()

        # Spatial discretization
        if self.settings and self.settings.elements:
            method.discretization.space_discretization = f"FEM_{self.settings.elements[0]}"

        # Solver
        if self.settings and self.settings.nlgeom:
            method.solver.algorithm = "newton_raphson"
            method.solver.convergence_criterion = "force_residual"
            method.solver.max_iterations = 16
        else:
            method.solver.algorithm = "direct_sparse"
            method.solver.convergence_criterion = "residual_norm"
            method.solver.tolerance = 1e-6

        # Time integration if dynamic
        step = self.settings.steps[0] if (self.settings and self.settings.steps) else None
        if step and step.analysis_type == "dynamic":
            method.solver.algorithm = "newmark_beta"
            method.solver.convergence_criterion = "force_residual"

        return method

    # ------------------------------------------------------------------
    # 计算图
    # ------------------------------------------------------------------

    def _extract_computational_graph(self) -> ComputationalGraph:
        """Extract computational graph for FEM."""
        graph = ComputationalGraph()

        graph.add_node(
            ComputationalNode(
                id="assembly",
                type="matrix_assembly",
                math_semantics={
                    "operator_type": "stiffness_assembly",
                    "updates": {"target": "K", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            )
        )

        solver_type = "linear_solver"
        solver_op = "sparse_direct_solver"
        if self.settings and self.settings.nlgeom:
            solver_type = "nonlinear_solver"
            solver_op = "newton_raphson"

        graph.add_node(
            ComputationalNode(
                id="solver",
                type=solver_type,
                math_semantics={
                    "operator_type": solver_op,
                    "updates": {"target": "displacement", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            )
        )

        graph.add_node(
            ComputationalNode(
                id="post_process",
                type="stress_recovery",
                math_semantics={
                    "operator_type": "strain_stress_computation",
                    "updates": {"target": "stress", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                },
            )
        )

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

    # ------------------------------------------------------------------
    # 守恒性质
    # ------------------------------------------------------------------

    def _extract_conservation_properties(self) -> Dict[str, Any]:
        """Extract conservation properties for FEM."""
        props = {}
        props["equilibrium"] = {"preserved": True, "mechanism": "weak_formulation"}
        props["conservative_system"] = {"preserved": True, "mechanism": "potential_energy"}
        return props

    # ------------------------------------------------------------------
    # 原始符号
    # ------------------------------------------------------------------

    def _extract_raw_symbols(self) -> Dict[str, Any]:
        """Extract raw symbols from input."""
        symbols = {
            "elements": self.settings.elements if self.settings else [],
            "materials": [
                {
                    "name": m.name,
                    "density": m.density,
                    "youngs_modulus": m.youngs_modulus,
                    "poisson_ratio": m.poisson_ratio,
                    "model_type": m.model_type,
                }
                for m in (self.settings.materials if self.settings else [])
            ],
            "steps": [
                {
                    "name": s.name,
                    "analysis_type": s.analysis_type,
                    "nlgeom": s.nlgeom,
                    "max_increments": s.max_increments,
                    "initial_inc": s.initial_inc,
                    "total_time": s.total_time,
                }
                for s in (self.settings.steps if self.settings else [])
            ],
            "boundary_conditions": [
                {
                    "node_set": b.node_set,
                    "dof": f"{b.dof_start}-{b.dof_end}",
                    "value": b.value,
                }
                for b in (self.settings.boundary_conditions if self.settings else [])
            ],
            "nlgeom": self.settings.nlgeom if self.settings else False,
            "nodes": self.settings.nodes if self.settings else 0,
            "cards": list(self.settings.cards.keys()) if self.settings else [],
        }
        return symbols
