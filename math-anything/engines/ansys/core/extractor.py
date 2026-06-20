"""Ansys mathematical structure extractor for FEM simulations.

Extracts mathematical structures from Ansys input files and results, including:
- Governing equations for all analysis types (static, modal, harmonic, transient, thermal, buckling)
- Constitutive relations with full Lame parameter relationships
- Boundary conditions with tensor expressions (Dirichlet/Neumann)
- Symbolic constraints (material stability, mesh quality, solution existence)
- Computational graph with proper nodes and edges
- Conservation properties
- Post-processing utilities (stress fields, strain energy, modal results)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

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

from .apdl_parser import EnhancedAPDLParser


@dataclass
class Material:
    """Material properties."""

    name: str = ""
    material_id: int = 1
    density: Optional[float] = None
    youngs_modulus: Optional[float] = None
    poisson_ratio: Optional[float] = None
    shear_modulus: Optional[float] = None
    bulk_modulus: Optional[float] = None
    lame_first: Optional[float] = None
    thermal_expansion: Optional[float] = None
    thermal_conductivity: Optional[float] = None
    specific_heat: Optional[float] = None
    model_type: str = "unknown"  # elastic, plastic, thermal, unknown
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnsysBC:
    """Boundary condition definition."""

    node_set: str = ""
    dof: str = "ALL"
    value: float = 0.0
    bc_type: str = "displacement"  # displacement, force, pressure, temperature


@dataclass
class Step:
    """Analysis step / load step."""

    name: str = ""
    analysis_type: str = "static"
    total_time: Optional[float] = None
    max_increments: Optional[int] = None
    initial_inc: Optional[float] = None
    min_inc: Optional[float] = None
    max_inc: Optional[float] = None
    nlgeom: bool = False


@dataclass
class FEMSettings:
    """FEM-specific settings extracted from Ansys input."""

    materials: List[Material] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    boundary_conditions: List[AnsysBC] = field(default_factory=list)
    elements: List[str] = field(default_factory=list)
    nodes: int = 0
    analysis_type: str = "static"
    nlgeom: bool = False
    element_size: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "file"  # file or dict


class AnsysExtractor(BaseEngineExtractor):
    """Extracts mathematical structures from Ansys FEM simulations.

    Merges APDL input parsing with post-processing capabilities, producing
    MathSchema at the same depth as the Abaqus extractor.

    Supports:
    - APDL input file parsing via EnhancedAPDLParser
    - Dict parameter extraction (API-style: {'E': 210e9, 'nu': 0.3, ...})
    - All analysis types: static, modal, harmonic, transient, thermal, buckling
    - Full Lame parameter relationships and constitutive relations
    - Post-processing utilities (stress, strain energy, modal results, etc.)

    Example:
        extractor = AnsysExtractor()
        schema = extractor.extract({"input": "beam.inp"})
        # or from dict:
        schema = extractor.extract({"params": {"E": 210e9, "nu": 0.3, "analysis_type": "modal"}})
    """

    def __init__(self):
        self.parser = EnhancedAPDLParser()
        self.settings: Optional[FEMSettings] = None

    @property
    def engine_name(self) -> str:
        return "ansys"

    @property
    def extractor_version(self) -> str:
        return "0.3.0"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def extract(
        self, files: Dict[str, str], options: Dict[str, Any] = None
    ) -> MathSchema:
        """Extract mathematical structures from Ansys files or dict parameters.

        Args:
            files: Dictionary with either 'input' (APDL file path) or
                   'params' (dict of parameters like {'E': 210e9, 'nu': 0.3}).
            options: Optional extraction parameters.

        Returns:
            MathSchema object with FEM mathematical structures.
        """
        options = options or {}

        input_path = files.get("input")
        params = files.get("params")

        if input_path:
            self.settings = self._extract_from_file(input_path)
        elif params:
            self.settings = self._extract_from_dict(params)
        else:
            raise ValueError("Either 'input' file path or 'params' dict required")

        # Build Math Schema via base class
        source_files = self._get_source_files(files)
        return self.build_schema(source_files)

    # ------------------------------------------------------------------
    # Input extraction strategies
    # ------------------------------------------------------------------

    def _extract_from_file(self, input_path: str) -> FEMSettings:
        """Parse APDL input file and build FEMSettings."""
        result = self.parser.parse_file(input_path)
        settings = FEMSettings(source="file")

        # Analysis type
        settings.analysis_type = result.analysis_type.value

        # Materials
        settings.materials = self._build_materials(result)

        # Boundary conditions
        settings.boundary_conditions = self._build_bcs(result)

        # Steps
        settings.steps = self._build_steps(result)

        # Elements
        elements = []
        for cmd in result.commands:
            if cmd.command == "ET" and len(cmd.args) >= 2:
                etype = cmd.args[1]
                if etype and etype not in elements:
                    elements.append(etype)
        settings.elements = elements

        # Element size
        for cmd in result.commands:
            if cmd.command == "ESIZE" and cmd.args:
                try:
                    settings.element_size = float(cmd.args[0])
                except ValueError:
                    pass

        # Nonlinear geometry
        settings.nlgeom = any(
            c.command == "NLGEOM" and c.args and c.args[0].upper() == "ON"
            for c in result.commands
        )

        # Raw data
        settings.parameters = result.parameters
        settings.commands = [c.to_dict() for c in result.commands]
        settings.constraints = result.constraints

        return settings

    def _extract_from_dict(self, params: Dict[str, Any]) -> FEMSettings:
        """Build FEMSettings from a parameter dictionary (API-style)."""
        settings = FEMSettings(source="dict")

        # Analysis type
        atype = params.get("analysis_type", "static_structural")
        settings.analysis_type = atype

        # Material
        mat = Material(name="material_1", material_id=1)
        if "E" in params or "EX" in params:
            mat.youngs_modulus = params.get("E", params.get("EX"))
            mat.model_type = "elastic"
        if "nu" in params or "PRXY" in params:
            mat.poisson_ratio = params.get("nu", params.get("PRXY"))
        if "density" in params or "DENS" in params:
            mat.density = params.get("density", params.get("DENS"))
        if "alpha" in params or "ALPX" in params:
            mat.thermal_expansion = params.get("alpha", params.get("ALPX"))
        if "k" in params or "KXX" in params:
            mat.thermal_conductivity = params.get("k", params.get("KXX"))
        if "c" in params or "C" in params:
            mat.specific_heat = params.get("c", params.get("C"))

        # Compute derived properties
        if mat.youngs_modulus and mat.poisson_ratio is not None:
            nu = mat.poisson_ratio
            E = mat.youngs_modulus
            mat.shear_modulus = E / (2 * (1 + nu))
            if abs(1 - 2 * nu) > 1e-10:
                mat.lame_first = E * nu / ((1 + nu) * (1 - 2 * nu))
                mat.bulk_modulus = E / (3 * (1 - 2 * nu))

        if mat.youngs_modulus or mat.thermal_conductivity:
            settings.materials = [mat]

        # Boundary conditions from dict
        bcs = params.get("boundary_conditions", [])
        for bc_data in bcs:
            settings.boundary_conditions.append(AnsysBC(
                node_set=bc_data.get("node_set", ""),
                dof=bc_data.get("dof", "ALL"),
                value=bc_data.get("value", 0.0),
                bc_type=bc_data.get("bc_type", "displacement"),
            ))

        # Step info
        step = Step(name="step_1", analysis_type=atype)
        if "time" in params:
            step.total_time = params["time"]
        if "nsubst" in params:
            step.max_increments = params["nsubst"]
        if "deltim" in params:
            step.initial_inc = params["deltim"]
        if "nlgeom" in params:
            step.nlgeom = params["nlgeom"]
            settings.nlgeom = params["nlgeom"]
        settings.steps = [step]

        # Element types
        if "element_type" in params:
            settings.elements = [params["element_type"]]

        # Store raw params
        settings.parameters = params

        return settings

    # ------------------------------------------------------------------
    # APDL result -> internal data structures
    # ------------------------------------------------------------------

    def _build_materials(self, result: Any) -> List[Material]:
        """Convert APDLResults materials to Material list."""
        mat_props: Dict[int, Dict[str, Any]] = {}
        for m in result.materials:
            mid = m.material_id
            if mid not in mat_props:
                mat_props[mid] = {"id": mid, "properties": {}}
            mat_props[mid]["properties"][m.name] = m.value

        materials = []
        for mid, props in mat_props.items():
            p = props["properties"]
            mat = Material(
                name=f"material_{mid}",
                material_id=mid,
                youngs_modulus=p.get("EX"),
                poisson_ratio=p.get("PRXY"),
                density=p.get("DENS"),
                thermal_expansion=p.get("ALPX"),
                thermal_conductivity=p.get("KXX"),
                specific_heat=p.get("C"),
            )

            # Determine model type
            if "EX" in p:
                mat.model_type = "elastic"
            elif "KXX" in p:
                mat.model_type = "thermal"

            # Compute derived properties
            if mat.youngs_modulus and mat.poisson_ratio is not None:
                E = mat.youngs_modulus
                nu = mat.poisson_ratio
                mat.shear_modulus = E / (2 * (1 + nu))
                if abs(1 - 2 * nu) > 1e-10:
                    mat.lame_first = E * nu / ((1 + nu) * (1 - 2 * nu))
                    mat.bulk_modulus = E / (3 * (1 - 2 * nu))

            materials.append(mat)

        return materials

    def _build_bcs(self, result: Any) -> List[AnsysBC]:
        """Extract boundary conditions from APDL commands."""
        bcs = []
        for cmd in result.commands:
            if cmd.command == "D" and len(cmd.args) >= 2:
                bcs.append(AnsysBC(
                    node_set=cmd.args[0],
                    dof=cmd.args[1],
                    value=float(cmd.args[2]) if len(cmd.args) > 2 else 0.0,
                    bc_type="displacement",
                ))
            elif cmd.command == "F" and len(cmd.args) >= 2:
                bcs.append(AnsysBC(
                    node_set=cmd.args[0],
                    dof=cmd.args[1],
                    value=float(cmd.args[2]) if len(cmd.args) > 2 else 0.0,
                    bc_type="force",
                ))
        return bcs

    def _build_steps(self, result: Any) -> List[Step]:
        """Extract step info from APDL commands."""
        step = Step(name="step_1", analysis_type=result.analysis_type.value)

        for cmd in result.commands:
            if cmd.command == "TIME" and cmd.args:
                try:
                    step.total_time = float(cmd.args[0])
                except ValueError:
                    pass
            elif cmd.command == "NSUBST" and cmd.args:
                try:
                    step.max_increments = int(cmd.args[0])
                except ValueError:
                    pass
            elif cmd.command == "DELTIM" and cmd.args:
                try:
                    step.initial_inc = float(cmd.args[0])
                except ValueError:
                    pass
            elif cmd.command == "NLGEOM" and cmd.args:
                step.nlgeom = cmd.args[0].upper() == "ON"

        return [step] if (step.total_time or step.max_increments) else [step]

    def _get_source_files(self, files: Dict[str, str]) -> Dict[str, List[str]]:
        """Build source_files dict for MetaInfo."""
        src = {}
        if "input" in files:
            src["input"] = [files["input"]]
        if "params" in files:
            src["params"] = ["dict"]
        return src

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
        """Extract governing equations for all Ansys analysis types."""
        equations = []
        analysis = self.settings.analysis_type if self.settings else "static_structural"

        if analysis in ("static_structural", "static"):
            equations.extend(self._static_equations())
        elif analysis == "modal":
            equations.extend(self._modal_equations())
        elif analysis == "harmonic":
            equations.extend(self._harmonic_equations())
        elif analysis in ("transient_structural", "transient"):
            equations.extend(self._transient_equations())
        elif analysis == "thermal":
            equations.extend(self._thermal_equations())
        elif analysis == "buckling":
            equations.extend(self._buckling_equations())
        else:
            # fallback: static
            equations.extend(self._static_equations())

        return equations

    def _static_equations(self) -> List[GoverningEquation]:
        """Static structural governing equations."""
        equations = []

        # Equilibrium
        equations.append(
            GoverningEquation(
                id="equilibrium",
                type="partial_differential_equation",
                name="Equilibrium Equation",
                mathematical_form="div(sigma) + b = 0",
                variables=["stress", "body_force", "displacement"],
                parameters={"form": "vector_pde"},
                description="Static equilibrium in the absence of inertia",
            )
        )

        # Strain-displacement
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
        mat = self._first_material()
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

        return equations

    def _modal_equations(self) -> List[GoverningEquation]:
        """Modal analysis governing equations: eigenvalue problem K*phi = omega^2*M*phi."""
        equations = []

        # Eigenvalue problem
        equations.append(
            GoverningEquation(
                id="modal_eigenvalue",
                type="generalized_eigenvalue_problem",
                name="Modal Eigenvalue Problem",
                mathematical_form="K*phi = omega^2 * M * phi",
                variables=[
                    "K: global stiffness matrix (symmetric positive semi-definite)",
                    "M: global mass matrix (symmetric positive semi-definite)",
                    "phi: eigenvector (mode shape vector)",
                    "omega: natural circular frequency (rad/s)",
                ],
                parameters={
                    "form": "generalized_eigenvalue",
                    "eigenvalue": "omega^2",
                    "eigenvector": "phi",
                    "solver_method": "Lanczos / Block Lanczos / Subspace",
                },
                description="Generalized eigenvalue problem for free vibration: K*phi = omega^2*M*phi, where omega = 2*pi*f and f is natural frequency in Hz",
            )
        )

        # Strain-displacement (kinematic)
        equations.append(
            GoverningEquation(
                id="strain_displacement",
                type="kinematic_relation",
                name="Strain-Displacement Relation",
                mathematical_form="epsilon = 1/2 (grad(u) + grad(u)^T)",
                variables=["strain", "displacement"],
                parameters={"linearity": "linear"},
                description="Infinitesimal strain tensor (modal assumes linear kinematics)",
            )
        )

        # Constitutive
        mat = self._first_material()
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

        # Orthogonality condition
        equations.append(
            GoverningEquation(
                id="modal_orthogonality",
                type="orthogonality_condition",
                name="Mode Shape Orthogonality",
                mathematical_form="phi_i^T * M * phi_j = delta_ij;  phi_i^T * K * phi_j = omega_i^2 * delta_ij",
                variables=["phi_i", "phi_j", "M", "K", "omega_i"],
                parameters={"condition": "mass_normalized"},
                description="Eigenvectors are M-orthogonal and K-orthogonal; mass normalization implies phi_i^T * M * phi_i = 1",
            )
        )

        return equations

    def _harmonic_equations(self) -> List[GoverningEquation]:
        """Harmonic (frequency response) governing equations: (-omega^2*M + i*omega*C + K)*u = F(omega)."""
        equations = []

        # Forced harmonic response
        equations.append(
            GoverningEquation(
                id="harmonic_response",
                type="complex_linear_system",
                name="Harmonic Forced Response",
                mathematical_form="(-omega^2*M + i*omega*C + K)*u = F(omega)",
                variables=[
                    "M: global mass matrix",
                    "C: global damping matrix (Rayleigh: C = alpha*M + beta*K)",
                    "K: global stiffness matrix",
                    "u: complex displacement amplitude vector",
                    "F(omega): harmonic force amplitude vector",
                    "omega: excitation circular frequency (rad/s)",
                ],
                parameters={
                    "form": "complex_linear_system",
                    "damping_model": "Rayleigh damping C = alpha*M + beta*K",
                    "solution_method": "full / modal superposition / mode-superposition",
                },
                description="Steady-state harmonic response equation in frequency domain; u(t) = Re{u * exp(i*omega*t))}",
            )
        )

        # Strain-displacement
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
        mat = self._first_material()
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

        # Rayleigh damping relation
        equations.append(
            GoverningEquation(
                id="rayleigh_damping",
                type="damping_model",
                name="Rayleigh Damping",
                mathematical_form="C = alpha*M + beta*K",
                variables=[
                    "C: damping matrix",
                    "M: mass matrix",
                    "K: stiffness matrix",
                    "alpha: mass-proportional damping coefficient",
                    "beta: stiffness-proportional damping coefficient",
                ],
                parameters={"model": "rayleigh"},
                description="Rayleigh damping model; alpha and beta determined from damping ratios at two frequencies",
            )
        )

        return equations

    def _transient_equations(self) -> List[GoverningEquation]:
        """Transient dynamic governing equations: M*u_ddot + C*u_dot + K*u = F(t) with Newmark-beta."""
        equations = []

        # Dynamic equilibrium
        equations.append(
            GoverningEquation(
                id="transient_dynamics",
                type="partial_differential_equation",
                name="Transient Dynamic Equation",
                mathematical_form="M*u_ddot + C*u_dot + K*u = F(t)",
                variables=[
                    "M: global mass matrix",
                    "C: global damping matrix",
                    "K: global stiffness matrix",
                    "u: displacement vector (function of time t)",
                    "u_dot: velocity vector (du/dt)",
                    "u_ddot: acceleration vector (d^2u/dt^2)",
                    "F(t): time-dependent external force vector",
                ],
                parameters={
                    "form": "second_order_ode",
                    "time_integration": "Newmark-beta",
                },
                description="Second-order ordinary differential equation for structural dynamics; semi-discrete form after FEM spatial discretization",
            )
        )

        # Newmark-beta integration
        equations.append(
            GoverningEquation(
                id="newmark_beta",
                type="time_integration_scheme",
                name="Newmark-beta Integration",
                mathematical_form="u_{n+1} = u_n + dt*u_dot_n + dt^2*((0.5-beta)*u_ddot_n + beta*u_ddot_{n+1}); u_dot_{n+1} = u_dot_n + dt*((1-gamma)*u_ddot_n + gamma*u_ddot_{n+1})",
                variables=[
                    "beta: Newmark parameter (0.25 for average acceleration, unconditionally stable)",
                    "gamma: Newmark parameter (0.5 for average acceleration)",
                    "dt: time step size",
                ],
                parameters={
                    "method": "implicit",
                    "stability": "unconditionally stable when beta >= 0.25 and gamma >= 0.5",
                    "average_acceleration": "beta=0.25, gamma=0.5",
                    "linear_acceleration": "beta=1/6, gamma=0.5",
                },
                description="Newmark-beta time integration scheme; average acceleration method (beta=0.25, gamma=0.5) is unconditionally stable and second-order accurate",
            )
        )

        # Strain-displacement
        if self.settings and self.settings.nlgeom:
            equations.append(
                GoverningEquation(
                    id="strain_displacement",
                    type="kinematic_relation",
                    name="Green-Lagrange Strain",
                    mathematical_form="E = 1/2 (grad(u) + grad(u)^T + grad(u)^T * grad(u))",
                    variables=["strain", "displacement"],
                    parameters={"linearity": "nonlinear"},
                    description="Finite strain tensor for geometrically nonlinear transient analysis",
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
        mat = self._first_material()
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

        return equations

    def _thermal_equations(self) -> List[GoverningEquation]:
        """Thermal analysis governing equations."""
        equations = []

        # Heat conduction
        equations.append(
            GoverningEquation(
                id="heat_conduction",
                type="partial_differential_equation",
                name="Heat Conduction Equation",
                mathematical_form="rho*c*dT/dt = div(k*grad(T)) + Q",
                variables=[
                    "T: temperature field",
                    "k: thermal conductivity tensor",
                    "rho: density",
                    "c: specific heat capacity",
                    "Q: internal heat generation rate per unit volume",
                ],
                parameters={"form": "scalar_pde"},
                description="Fourier heat conduction equation with transient term; steady-state when dT/dt = 0",
            )
        )

        # Fourier's law
        equations.append(
            GoverningEquation(
                id="fourier_law",
                type="constitutive_relation",
                name="Fourier's Law of Heat Conduction",
                mathematical_form="q = -k * grad(T)",
                variables=["q: heat flux vector", "k: thermal conductivity", "T: temperature"],
                parameters={"isotropic": True},
                description="Constitutive relation for heat conduction; q is heat flux, k is thermal conductivity",
            )
        )

        # Weak form
        equations.append(
            GoverningEquation(
                id="thermal_weak_form",
                type="variational_principle",
                name="Thermal Weak Form",
                mathematical_form="int_Omega rho*c*dT/dt*delta_T dOmega + int_Omega k*grad(T)*grad(delta_T) dOmega = int_Omega Q*delta_T dOmega + int_Gamma q_n*delta_T dGamma",
                variables=["T", "delta_T", "k", "rho", "c", "Q", "q_n"],
                parameters={"form": "integral_equation"},
                description="Weak form of heat conduction equation for FEM discretization",
            )
        )

        return equations

    def _buckling_equations(self) -> List[GoverningEquation]:
        """Linear buckling governing equations: (K + lambda*K_sigma)*phi = 0."""
        equations = []

        # Pre-buckling static equilibrium
        equations.append(
            GoverningEquation(
                id="prebuckle_equilibrium",
                type="partial_differential_equation",
                name="Pre-buckling Static Equilibrium",
                mathematical_form="K*u = F",
                variables=["K: stiffness matrix", "u: displacement vector", "F: applied load vector"],
                parameters={"form": "linear_system"},
                description="Static equilibrium under reference load to compute stress state for geometric stiffness",
            )
        )

        # Buckling eigenvalue problem
        equations.append(
            GoverningEquation(
                id="buckling_eigenvalue",
                type="generalized_eigenvalue_problem",
                name="Linear Buckling Eigenvalue Problem",
                mathematical_form="(K + lambda*K_sigma)*phi = 0",
                variables=[
                    "K: elastic stiffness matrix (symmetric positive definite)",
                    "K_sigma: stress stiffness matrix (geometric stiffness)",
                    "lambda: buckling load multiplier (eigenvalue)",
                    "phi: buckling mode shape (eigenvector)",
                ],
                parameters={
                    "form": "generalized_eigenvalue",
                    "eigenvalue": "lambda (load multiplier)",
                    "eigenvector": "phi (buckling mode)",
                    "critical_load": "F_cr = lambda * F_ref",
                    "solver_method": "Lanczos / Subspace iteration",
                },
                description="Linear buckling eigenvalue problem; critical load F_cr = lambda * F_ref where F_ref is the reference applied load",
            )
        )

        # Constitutive
        mat = self._first_material()
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

        return equations

    # ------------------------------------------------------------------
    # 本构关系
    # ------------------------------------------------------------------

    def _extract_constitutive_relations(self) -> List[Dict[str, Any]]:
        """Extract constitutive relations (material models)."""
        relations = []

        for mat in (self.settings.materials if self.settings else []):
            if mat.model_type == "elastic" and mat.youngs_modulus is not None:
                rel = {
                    "type": "elastic",
                    "name": "linear_isotropic",
                    "form": "sigma = C : epsilon",
                    "parameters": {
                        "E": mat.youngs_modulus,
                        "nu": mat.poisson_ratio,
                    },
                    "stiffness_tensor": "C_ijkl = lambda delta_ij delta_kl + mu (delta_ik delta_jl + delta_il delta_jk)",
                }
                if mat.shear_modulus is not None:
                    rel["parameters"]["G"] = mat.shear_modulus
                if mat.bulk_modulus is not None:
                    rel["parameters"]["K"] = mat.bulk_modulus
                if mat.lame_first is not None:
                    rel["parameters"]["lambda"] = mat.lame_first
                relations.append(rel)

            elif mat.model_type == "thermal" and mat.thermal_conductivity is not None:
                relations.append({
                    "type": "thermal",
                    "name": "fourier_conduction",
                    "form": "q = -k * grad(T)",
                    "parameters": {
                        "k": mat.thermal_conductivity,
                        "rho": mat.density,
                        "c": mat.specific_heat,
                    },
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

            relationships.append(
                ParameterRelationship(
                    name="youngs_from_lame",
                    expression="E = mu*(3*lambda + 2*mu) / (lambda + mu)",
                    variables=["E", "lambda", "mu"],
                    relation_type="equality",
                    description="Young's modulus expressed in terms of Lame parameters",
                    physical_meaning="Inverse relationship: E from Lame parameters",
                )
            )

            relationships.append(
                ParameterRelationship(
                    name="poisson_from_lame",
                    expression="nu = lambda / (2*(lambda + mu))",
                    variables=["nu", "lambda", "mu"],
                    relation_type="equality",
                    description="Poisson's ratio expressed in terms of Lame parameters",
                    physical_meaning="Inverse relationship: nu from Lame parameters",
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
            if bc.bc_type == "displacement":
                bc_type = "dirichlet"
                if bc.value == 0.0:
                    bc_form = f"u_{bc.dof} = 0"
                else:
                    bc_form = f"u_{bc.dof} = {bc.value}"
                field = "displacement"
            elif bc.bc_type == "force":
                bc_type = "neumann"
                bc_form = f"t_{bc.dof} = {bc.value}"
                field = "traction"
            elif bc.bc_type == "pressure":
                bc_type = "neumann"
                bc_form = f"p = {bc.value}"
                field = "pressure"
            elif bc.bc_type == "temperature":
                bc_type = "dirichlet"
                bc_form = f"T = {bc.value}"
                field = "temperature"
            else:
                bc_type = "unknown"
                bc_form = f"value = {bc.value}"
                field = "unknown"

            bcs.append(
                BoundaryCondition(
                    id=f"bc_{bc.node_set}_{bc.dof}",
                    type=bc_type,
                    domain={
                        "geometric_region": bc.node_set,
                        "entity_type": "node",
                    },
                    mathematical_object=MathematicalObject(
                        field=field,
                        tensor_rank=1,
                        tensor_form=bc_form,
                    ),
                    software_implementation={
                        "command": "D" if bc.bc_type == "displacement" else "F",
                        "node": bc.node_set,
                        "dof": bc.dof,
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

            if mat.density is not None:
                constraints.append(
                    SymbolicConstraint(
                        expression="rho > 0",
                        description="Density must be positive for physical consistency",
                        variables=["rho"],
                        confidence=1.0,
                        inferred_from="material_stability",
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

        # Analysis-specific constraints
        analysis = self.settings.analysis_type if self.settings else "static"
        if analysis == "modal":
            constraints.append(
                SymbolicConstraint(
                    expression="det(K) > 0 or det(K) = 0 with rigid body modes",
                    description="Stiffness matrix must be positive semi-definite; zero eigenvalues correspond to rigid body modes",
                    variables=["K"],
                    confidence=0.9,
                    inferred_from="modal_eigenvalue_existence",
                )
            )
            constraints.append(
                SymbolicConstraint(
                    expression="M positive definite",
                    description="Mass matrix must be positive definite for well-posed eigenvalue problem",
                    variables=["M"],
                    confidence=0.9,
                    inferred_from="modal_eigenvalue_theory",
                )
            )
        elif analysis in ("transient_structural", "transient"):
            constraints.append(
                SymbolicConstraint(
                    expression="dt < dt_critical (CFL-like condition for explicit)",
                    description="Time step must satisfy stability condition for explicit time integration",
                    variables=["dt", "dt_critical"],
                    confidence=0.85,
                    inferred_from="time_integration_stability",
                )
            )
        elif analysis == "buckling":
            constraints.append(
                SymbolicConstraint(
                    expression="lambda > 0 for physical buckling",
                    description="Buckling load multiplier must be positive for physical buckling under compressive load",
                    variables=["lambda"],
                    confidence=0.9,
                    inferred_from="buckling_theory",
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

        analysis = self.settings.analysis_type if self.settings else "static"
        nlgeom = self.settings.nlgeom if self.settings else False

        if analysis == "modal":
            method.solver.algorithm = "lanczos_eigenvalue"
            method.solver.convergence_criterion = "eigenvalue_tolerance"
            method.solver.tolerance = 1e-6
        elif analysis == "harmonic":
            method.solver.algorithm = "harmonic_frequency_sweep"
            method.solver.convergence_criterion = "residual_norm"
            method.solver.tolerance = 1e-6
        elif analysis in ("transient_structural", "transient"):
            method.solver.algorithm = "newmark_beta"
            method.solver.convergence_criterion = "force_residual"
            method.solver.max_iterations = 16
        elif analysis == "buckling":
            method.solver.algorithm = "lanczos_eigenvalue"
            method.solver.convergence_criterion = "eigenvalue_tolerance"
            method.solver.tolerance = 1e-6
        elif analysis == "thermal":
            if nlgeom:
                method.solver.algorithm = "newton_raphson"
                method.solver.convergence_criterion = "heat_flux_residual"
            else:
                method.solver.algorithm = "direct_sparse"
                method.solver.convergence_criterion = "residual_norm"
                method.solver.tolerance = 1e-6
        else:
            # Static
            if nlgeom:
                method.solver.algorithm = "newton_raphson"
                method.solver.convergence_criterion = "force_residual"
                method.solver.max_iterations = 16
            else:
                method.solver.algorithm = "direct_sparse"
                method.solver.convergence_criterion = "residual_norm"
                method.solver.tolerance = 1e-6

        return method

    # ------------------------------------------------------------------
    # 计算图
    # ------------------------------------------------------------------

    def _extract_computational_graph(self) -> ComputationalGraph:
        """Extract computational graph for FEM."""
        graph = ComputationalGraph()

        analysis = self.settings.analysis_type if self.settings else "static"
        nlgeom = self.settings.nlgeom if self.settings else False

        # Node 1: Assembly
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

        # Node 2: Solver (varies by analysis type)
        if analysis == "modal":
            solver_type = "eigenvalue_solver"
            solver_op = "lanczos_eigenvalue"
        elif analysis == "harmonic":
            solver_type = "frequency_response_solver"
            solver_op = "harmonic_frequency_sweep"
        elif analysis in ("transient_structural", "transient"):
            solver_type = "time_integration_solver"
            solver_op = "newmark_beta"
        elif analysis == "buckling":
            solver_type = "eigenvalue_solver"
            solver_op = "lanczos_eigenvalue"
        elif nlgeom:
            solver_type = "nonlinear_solver"
            solver_op = "newton_raphson"
        else:
            solver_type = "linear_solver"
            solver_op = "sparse_direct_solver"

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

        # Node 3: Post-processing
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

        # For transient/modal/harmonic: add extra edges
        if analysis in ("transient_structural", "transient"):
            graph.add_node(
                ComputationalNode(
                    id="mass_assembly",
                    type="matrix_assembly",
                    math_semantics={
                        "operator_type": "mass_assembly",
                        "updates": {"target": "M", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                    },
                )
            )
            graph.add_edge(
                ComputationalEdge(
                    from_node="mass_assembly",
                    to_node="solver",
                    data_type="mass_matrix",
                    dependency="solve",
                )
            )

        if analysis == "harmonic":
            graph.add_node(
                ComputationalNode(
                    id="damping_assembly",
                    type="matrix_assembly",
                    math_semantics={
                        "operator_type": "damping_assembly",
                        "updates": {"target": "C", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                    },
                )
            )
            graph.add_edge(
                ComputationalEdge(
                    from_node="damping_assembly",
                    to_node="solver",
                    data_type="damping_matrix",
                    dependency="solve",
                )
            )

        if analysis == "buckling":
            graph.add_node(
                ComputationalNode(
                    id="stress_stiffness",
                    type="matrix_assembly",
                    math_semantics={
                        "operator_type": "geometric_stiffness_assembly",
                        "updates": {"target": "K_sigma", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                    },
                )
            )
            graph.add_edge(
                ComputationalEdge(
                    from_node="stress_stiffness",
                    to_node="solver",
                    data_type="geometric_stiffness_matrix",
                    dependency="solve",
                )
            )

        return graph

    # ------------------------------------------------------------------
    # 守恒性质
    # ------------------------------------------------------------------

    def _extract_conservation_properties(self) -> Dict[str, Any]:
        """Extract conservation properties for FEM."""
        analysis = self.settings.analysis_type if self.settings else "static"
        props = {}

        if analysis in ("static_structural", "static"):
            props["equilibrium"] = {"preserved": True, "mechanism": "weak_formulation"}
            props["conservative_system"] = {"preserved": True, "mechanism": "potential_energy"}
        elif analysis == "modal":
            props["equilibrium"] = {"preserved": True, "mechanism": "weak_formulation"}
            props["energy_conservation"] = {"preserved": True, "mechanism": "modal_orthogonality"}
            props["mass_conservation"] = {"preserved": True, "mechanism": "consistent_mass_matrix"}
        elif analysis == "harmonic":
            props["energy_balance"] = {"preserved": True, "mechanism": "frequency_domain_energy"}
            props["reciprocity"] = {"preserved": True, "mechanism": "maxwell_betti_reciprocal"}
        elif analysis in ("transient_structural", "transient"):
            props["momentum_conservation"] = {"preserved": True, "mechanism": "newmark_beta_integration"}
            props["energy_conservation"] = {"preserved": True, "mechanism": "symplectic_condition_gamma=0.5"}
        elif analysis == "thermal":
            props["energy_conservation"] = {"preserved": True, "mechanism": "first_law_thermodynamics"}
        elif analysis == "buckling":
            props["equilibrium"] = {"preserved": True, "mechanism": "weak_formulation"}
            props["potential_energy_stationarity"] = {"preserved": True, "mechanism": "euler_critical_load"}

        return props

    # ------------------------------------------------------------------
    # 原始符号
    # ------------------------------------------------------------------

    def _extract_raw_symbols(self) -> Dict[str, Any]:
        """Extract raw symbols from input."""
        symbols = {
            "analysis_type": self.settings.analysis_type if self.settings else "unknown",
            "elements": self.settings.elements if self.settings else [],
            "materials": [
                {
                    "name": m.name,
                    "material_id": m.material_id,
                    "density": m.density,
                    "youngs_modulus": m.youngs_modulus,
                    "poisson_ratio": m.poisson_ratio,
                    "shear_modulus": m.shear_modulus,
                    "bulk_modulus": m.bulk_modulus,
                    "lame_first": m.lame_first,
                    "thermal_expansion": m.thermal_expansion,
                    "thermal_conductivity": m.thermal_conductivity,
                    "specific_heat": m.specific_heat,
                    "model_type": m.model_type,
                }
                for m in (self.settings.materials if self.settings else [])
            ],
            "steps": [
                {
                    "name": s.name,
                    "analysis_type": s.analysis_type,
                    "total_time": s.total_time,
                    "max_increments": s.max_increments,
                    "initial_inc": s.initial_inc,
                    "nlgeom": s.nlgeom,
                }
                for s in (self.settings.steps if self.settings else [])
            ],
            "boundary_conditions": [
                {
                    "node_set": b.node_set,
                    "dof": b.dof,
                    "value": b.value,
                    "bc_type": b.bc_type,
                }
                for b in (self.settings.boundary_conditions if self.settings else [])
            ],
            "nlgeom": self.settings.nlgeom if self.settings else False,
            "element_size": self.settings.element_size if self.settings else None,
            "parameters": self.settings.parameters if self.settings else {},
            "source": self.settings.source if self.settings else "unknown",
        }
        return symbols

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _first_material(self) -> Optional[Material]:
        """Return the first material from settings, if any."""
        if self.settings and self.settings.materials:
            return self.settings.materials[0]
        return None

    # ------------------------------------------------------------------
    # Post-processing utility methods
    # ------------------------------------------------------------------

    def extract_stress_field(
        self,
        stress_data: np.ndarray,
        element_centroids: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract stress field statistics and invariants.

        Args:
            stress_data: Array of stress tensors (n_elements, 6)
                        [Sxx, Syy, Szz, Sxy, Syz, Sxz]
            element_centroids: Optional element centroid coordinates

        Returns:
            Dictionary with stress statistics
        """
        if stress_data.shape[0] == 0:
            return {}

        sxx, syy, szz = stress_data[:, 0], stress_data[:, 1], stress_data[:, 2]
        sxy, syz, sxz = stress_data[:, 3], stress_data[:, 4], stress_data[:, 5]

        von_mises = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3 * (sxy**2 + syz**2 + sxz**2)
        )

        principal_stresses = self._compute_principal_stresses(stress_data)

        I1 = sxx + syy + szz
        I2 = sxx * syy + syy * szz + szz * sxx - sxy**2 - syz**2 - sxz**2
        I3 = (
            sxx * syy * szz
            + 2 * sxy * syz * sxz
            - sxx * syz**2
            - syy * sxz**2
            - szz * sxy**2
        )

        return {
            "von_mises_max": float(np.max(von_mises)),
            "von_mises_min": float(np.min(von_mises)),
            "von_mises_mean": float(np.mean(von_mises)),
            "von_mises_std": float(np.std(von_mises)),
            "principal_max": float(np.max(principal_stresses[:, 0])),
            "principal_min": float(np.min(principal_stresses[:, 2])),
            "invariant_I1_mean": float(np.mean(I1)),
            "invariant_I2_mean": float(np.mean(I2)),
            "invariant_I3_mean": float(np.mean(I3)),
        }

    def _compute_principal_stresses(self, stress_data: np.ndarray) -> np.ndarray:
        """Compute principal stresses for each element."""
        n = stress_data.shape[0]
        principal = np.zeros((n, 3))

        for i in range(n):
            sigma = np.array(
                [
                    [stress_data[i, 0], stress_data[i, 3], stress_data[i, 5]],
                    [stress_data[i, 3], stress_data[i, 1], stress_data[i, 4]],
                    [stress_data[i, 5], stress_data[i, 4], stress_data[i, 2]],
                ]
            )
            eigenvalues = np.linalg.eigvalsh(sigma)
            principal[i] = np.sort(eigenvalues)[::-1]

        return principal

    def extract_strain_energy(
        self,
        stress_data: np.ndarray,
        strain_data: np.ndarray,
        volumes: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract strain energy from stress and strain fields."""
        energy_density = 0.5 * np.sum(stress_data * strain_data, axis=1)
        strain_energy = np.sum(energy_density * volumes)

        return {
            "total_strain_energy": float(strain_energy),
            "max_energy_density": float(np.max(energy_density)),
            "mean_energy_density": float(np.mean(energy_density)),
        }

    def extract_modal_results(
        self,
        frequencies: np.ndarray,
        mode_shapes: np.ndarray,
        mass_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract modal analysis results."""
        n_modes = len(frequencies)

        results = {
            "num_modes": n_modes,
            "frequencies_hz": frequencies.tolist(),
            "frequencies_rad": (2 * np.pi * frequencies).tolist(),
            "periods": (1.0 / frequencies).tolist(),
        }

        if mass_matrix is not None:
            modal_masses = []
            for i in range(n_modes):
                phi = mode_shapes[i]
                m_modal = np.dot(phi, np.dot(mass_matrix, phi))
                modal_masses.append(float(m_modal))

            results["modal_masses"] = modal_masses
            results["effective_modal_mass"] = sum(modal_masses)

        return results

    def extract_contact_info(
        self,
        contact_status: np.ndarray,
        contact_pressure: np.ndarray,
        contact_friction: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract contact mechanics information."""
        n_contact = len(contact_status)

        open_count = np.sum(contact_status == 0)
        sliding_count = np.sum(contact_status == 1)
        sticking_count = np.sum(contact_status == 2)

        active = contact_status > 0

        return {
            "num_contact_elements": n_contact,
            "open_elements": int(open_count),
            "sliding_elements": int(sliding_count),
            "sticking_elements": int(sticking_count),
            "contact_area_fraction": (
                float(np.sum(active) / n_contact) if n_contact > 0 else 0
            ),
            "max_contact_pressure": (
                float(np.max(contact_pressure[active])) if np.any(active) else 0.0
            ),
            "mean_contact_pressure": (
                float(np.mean(contact_pressure[active])) if np.any(active) else 0.0
            ),
            "max_friction_stress": (
                float(np.max(contact_friction[active])) if np.any(active) else 0.0
            ),
        }

    def compute_reaction_forces(
        self,
        constraint_nodes: np.ndarray,
        constraint_dofs: np.ndarray,
        internal_forces: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute reaction forces at constraints."""
        reactions = internal_forces[constraint_dofs]

        return {
            "total_reaction_magnitude": float(np.linalg.norm(reactions)),
            "max_reaction": float(np.max(np.abs(reactions))),
            "sum_reactions": float(np.sum(reactions)),
            "reaction_components": reactions.tolist(),
        }

    def extract_mesh_quality(
        self,
        element_types: np.ndarray,
        node_coords: np.ndarray,
        element_connectivity: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract mesh quality metrics."""
        n_elements = len(element_types)

        aspect_ratios = []
        for i in range(n_elements):
            nodes = element_connectivity[i]
            coords = node_coords[nodes - 1]  # ANSYS uses 1-based indexing

            edges = []
            for j in range(len(coords)):
                for k in range(j + 1, len(coords)):
                    edge_len = np.linalg.norm(coords[j] - coords[k])
                    edges.append(edge_len)

            if edges:
                aspect_ratios.append(max(edges) / min(edges))

        aspect_ratios = np.array(aspect_ratios)

        return {
            "num_elements": n_elements,
            "num_nodes": len(node_coords),
            "mean_aspect_ratio": (
                float(np.mean(aspect_ratios)) if len(aspect_ratios) > 0 else 0
            ),
            "max_aspect_ratio": (
                float(np.max(aspect_ratios)) if len(aspect_ratios) > 0 else 0
            ),
            "bad_elements": (
                int(np.sum(aspect_ratios > 10)) if len(aspect_ratios) > 0 else 0
            ),
        }
