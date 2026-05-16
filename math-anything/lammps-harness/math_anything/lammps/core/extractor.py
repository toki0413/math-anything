"""LAMMPS mathematical structure extractor.

Extracts mathematical structures from parsed LAMMPS commands and maps them
to Math Schema v1.0 representation.
"""

import os
import sys
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from math_anything.schemas import (
    BoundaryCondition,
    ComputationalEdge,
    ComputationalGraph,
    ComputationalNode,
    Discretization,
    GoverningEquation,
    MathematicalModel,
    MathematicalObject,
    MathSchema,
    MetaInfo,
    NumericalMethod,
    ParameterRelationship,
    Solver,
    SymbolicConstraint,
    TensorComponent,
    UpdateMode,
)

from .parser import (
    ComputationalSettings,
    FixCommand,
    LammpsInputParser,
    LammpsLogParser,
)


class LammpsExtractor:
    """Extracts mathematical structures from LAMMPS simulations.

    Maps LAMMPS commands to Math Schema v1.0 representation, including:
    - Governing equations (Newton's laws, force fields)
    - Boundary conditions (periodic, fixed, deform)
    - Numerical methods (integrators, solvers)
    - Computational graph with explicit/implicit loop distinction

    Example:
        ```python
        extractor = LammpsExtractor()
        schema = extractor.extract({
            "input": "in.deform",
            "log": "log.lammps"
        })
        ```
    """

    def __init__(self):
        self.input_parser = LammpsInputParser()
        self.log_parser = LammpsLogParser()
        self.settings: Optional[ComputationalSettings] = None

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> MathSchema:
        """Extract mathematical structures from LAMMPS files.

        Args:
            files: Dictionary with 'input' (required) and optionally 'log'.
            options: Optional extraction parameters.

        Returns:
            MathSchema object.
        """
        options = options or {}

        # Parse input file
        input_path = files.get("input")
        if not input_path:
            raise ValueError("Input file required")

        commands = self.input_parser.parse_file(input_path)
        self.settings = self.input_parser.extract_settings(commands)

        # Parse log file if provided
        log_data = None
        self.dynamics_result = None
        if "log" in files:
            log_data = self.log_parser.parse_file(files["log"])
            self.dynamics_result = self._analyze_dynamics(log_data)

        # Build Math Schema
        schema = MathSchema(
            schema_version="1.0.0",
            meta=MetaInfo(
                extracted_by="math-anything-lammps",
                extractor_version="0.1.0",
                source_files={
                    "input": [input_path],
                    "log": [files["log"]] if "log" in files else [],
                },
            ),
            mathematical_model=self._extract_mathematical_model(),
            numerical_method=self._extract_numerical_method(),
            computational_graph=self._extract_computational_graph(),
            conservation_properties=self._extract_conservation_properties(),
            raw_symbols=self._extract_raw_symbols(),
        )

        # Add symbolic constraints (physical/mathematical constraints)
        schema.symbolic_constraints = self._extract_symbolic_constraints()

        return schema

    def _extract_mathematical_model(self) -> MathematicalModel:
        """Extract mathematical model (equations, BCs, ICs)."""
        model = MathematicalModel()

        # Governing equations
        model.governing_equations = self._extract_governing_equations()

        # Boundary conditions
        model.boundary_conditions = self._extract_boundary_conditions()

        # Initial conditions
        model.initial_conditions = self._extract_initial_conditions()

        # Constitutive relations (force fields)
        model.constitutive_relations = self._extract_constitutive_relations()

        # Parameter relationships (mathematical relationships between parameters)
        model.parameter_relationships = self._extract_parameter_relationships()

        return model

    def _extract_governing_equations(self) -> List[GoverningEquation]:
        """Extract governing equations."""
        equations = []

        # Newton's second law (always present in MD)
        equations.append(
            GoverningEquation(
                id="newton_second_law",
                type="second_order_ode",
                name="Newton's Second Law",
                mathematical_form="m * d²r/dt² = F",
                variables=["position", "velocity", "force", "mass"],
                parameters={"form": "vector"},
                description="Classical Newtonian equations of motion",
            )
        )

        # Hamiltonian formulation (if NVE/NVT)
        integrators = self.settings.get_integrator_fixes()
        if any(f.fix_style in ("nve", "nvt") for f in integrators):
            equations.append(
                GoverningEquation(
                    id="hamiltonian_dynamics",
                    type="hamiltonian_system",
                    name="Hamiltonian Dynamics",
                    mathematical_form="H = T(p) + V(q), dq/dt = ∂H/∂p, dp/dt = -∂H/∂q",
                    variables=["position", "momentum", "hamiltonian"],
                    parameters={"separable": True},
                    description="Hamiltonian formulation of dynamics",
                )
            )

        return equations

    def _extract_boundary_conditions(self) -> List[BoundaryCondition]:
        """Extract boundary conditions with tensor-complete expression."""
        bcs = []

        # Domain boundary conditions
        bc_map = {
            "p": "periodic",
            "f": "fixed",
            "s": "shrink-wrapped",
            "m": "shrink-wrapped-min",
        }
        for i, dim in enumerate(["x", "y", "z"]):
            if i < len(self.settings.boundary_style):
                bc_type = bc_map.get(
                    self.settings.boundary_style[i], self.settings.boundary_style[i]
                )
                bcs.append(
                    BoundaryCondition(
                        id=f"domain_bc_{dim}",
                        type=bc_type,
                        domain={
                            "geometric_region": f"{dim}-boundary",
                            "entity_type": "domain",
                        },
                        mathematical_object=MathematicalObject(
                            field=f"{dim}-boundary",
                            tensor_rank=0,
                            tensor_form=bc_type,
                        ),
                        software_implementation={
                            "command": "boundary",
                            "parameters": {dim: self.settings.boundary_style[i]},
                        },
                    )
                )

        # Constraint fixes as boundary conditions
        for fix in self.settings.get_constraint_fixes():
            bc = self._convert_fix_to_bc(fix)
            if bc:
                bcs.append(bc)

        return bcs

    def _convert_fix_to_bc(self, fix: FixCommand) -> Optional[BoundaryCondition]:
        """Convert a fix command to boundary condition."""

        # Fix deform - tensor boundary condition for deformation
        if fix.fix_style == "deform":
            return self._convert_fix_deform(fix)

        # Fix wall - wall boundary condition
        if fix.fix_style == "wall":
            return BoundaryCondition(
                id=f"fix_{fix.fix_id}",
                type="wall",
                domain={"geometric_region": fix.group_id, "entity_type": "group"},
                mathematical_object=MathematicalObject(
                    field="position",
                    tensor_rank=0,
                    tensor_form="wall_constraint",
                ),
                software_implementation={
                    "command": "fix",
                    "style": "wall",
                    "args": fix.args,
                },
            )

        # Fix spring - harmonic constraint
        if fix.fix_style == "spring":
            return BoundaryCondition(
                id=f"fix_{fix.fix_id}",
                type="harmonic_constraint",
                domain={"geometric_region": fix.group_id, "entity_type": "group"},
                mathematical_object=MathematicalObject(
                    field="position",
                    tensor_rank=0,
                    tensor_form="F = -k(r - r0)",
                ),
                software_implementation={
                    "command": "fix",
                    "style": "spring",
                    "args": fix.args,
                },
            )

        return None

    def _convert_fix_deform(self, fix: FixCommand) -> Optional[BoundaryCondition]:
        """Convert fix deform to tensor boundary condition.

        This is the critical case for Phase 0 - expressing fix deform
        as a 2nd-order tensor (deformation gradient F_ij).
        """
        if len(fix.args) < 2:
            return None

        dim = fix.args[0].lower()  # x, y, or z
        style = fix.args[1].lower()  # erate, trate, etc.

        # Build deformation gradient tensor F_ij
        # For uniaxial deformation along x:
        # F = [[1 + ε_dot*t, 0, 0], [0, 1, 0], [0, 0, 1]]

        components = []

        if style == "erate":  # Engineering strain rate
            rate = fix.args[2] if len(fix.args) > 2 else "0.0"

            # Diagonal components
            if dim == "x":
                components = [
                    TensorComponent(
                        index=[1, 1], value=f"1 + {rate} * t", unit="dimensionless"
                    ),
                    TensorComponent(index=[2, 2], value="1", unit="dimensionless"),
                    TensorComponent(index=[3, 3], value="1", unit="dimensionless"),
                ]
            elif dim == "y":
                components = [
                    TensorComponent(index=[1, 1], value="1", unit="dimensionless"),
                    TensorComponent(
                        index=[2, 2], value=f"1 + {rate} * t", unit="dimensionless"
                    ),
                    TensorComponent(index=[3, 3], value="1", unit="dimensionless"),
                ]
            elif dim == "z":
                components = [
                    TensorComponent(index=[1, 1], value="1", unit="dimensionless"),
                    TensorComponent(index=[2, 2], value="1", unit="dimensionless"),
                    TensorComponent(
                        index=[3, 3], value=f"1 + {rate} * t", unit="dimensionless"
                    ),
                ]

            # For incompressible materials: trace condition
            trace_condition = f"det(F) = 1 + {rate}*t"
        else:
            components = [
                TensorComponent(index=[i, i], value="1", unit="dimensionless")
                for i in range(1, 4)
            ]
            trace_condition = "det(F) = 1"

        return BoundaryCondition(
            id=f"fix_{fix.fix_id}",
            type="dirichlet",
            domain={
                "geometric_region": "all",
                "entity_type": "global",
            },
            mathematical_object=MathematicalObject(
                field="displacement_gradient",
                tensor_rank=2,
                tensor_form="F_{ij} = ∂x_i/∂X_j",
                components=components,
                symmetry="symmetric",
                trace_condition=trace_condition,
            ),
            dual_role={
                "is_boundary_condition": True,
                "is_external_drive": True,
                "note": "fix_deform applies homogeneous deformation to all atoms, acting as both Dirichlet BC on deformation gradient and external driving force",
            },
            software_implementation={
                "command": "fix deform",
                "style": style,
                "parameters": {
                    "dimension": dim,
                    "rate": fix.args[2] if len(fix.args) > 2 else None,
                },
            },
            equivalent_formulations=[
                {
                    "type": "strain_rate_tensor",
                    "form": "ε̇_{ij} = (L_{ij} + L_{ji})/2",
                    "velocity_gradient": "L = Ḟ·F⁻¹",
                }
            ],
        )

    def _extract_initial_conditions(self) -> List[Dict[str, Any]]:
        """Extract initial conditions."""
        ics = []

        # Check for velocity initialization
        # This would need more detailed parsing

        return ics

    def _extract_constitutive_relations(self) -> List[Dict[str, Any]]:
        """Extract constitutive relations (force fields)."""
        relations = []

        if self.settings.pair_style:
            relations.append(
                {
                    "type": "pair_potential",
                    "name": self.settings.pair_style.style,
                    "form": self._get_pair_potential_form(
                        self.settings.pair_style.style
                    ),
                    "parameters": self.settings.pair_style.args,
                }
            )

        return relations

    def _extract_parameter_relationships(self) -> List[ParameterRelationship]:
        """Extract mathematical relationships between parameters.

        These are the symbolic equations that define how parameters relate,
        enabling LLM to perform symbolic reasoning.
        """
        relationships = []

        # LJ potential relationship
        if self.settings.pair_style and self.settings.pair_style.style in (
            "lj/cut",
            "lj/cut/coul/long",
        ):
            relationships.append(
                ParameterRelationship(
                    name="lennard_jones_potential",
                    expression="U(r) = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)",
                    variables=["U", "r", "epsilon", "sigma"],
                    relation_type="equality",
                    description="Lennard-Jones 12-6 pair potential energy",
                    physical_meaning="Van der Waals interaction with repulsive and attractive terms",
                )
            )

            # Add force expression (derivative of potential)
            relationships.append(
                ParameterRelationship(
                    name="lennard_jones_force",
                    expression="F(r) = 24*epsilon*(2*(sigma/r)**12 - (sigma/r)**6)/r",
                    variables=["F", "r", "epsilon", "sigma"],
                    relation_type="equality",
                    description="Lennard-Jones force (negative gradient of potential)",
                    physical_meaning="Force derived from LJ potential",
                )
            )

            # Minimum energy distance
            relationships.append(
                ParameterRelationship(
                    name="lj_minimum_distance",
                    expression="r_min = 2**(1/6) * sigma",
                    variables=["r_min", "sigma"],
                    relation_type="equality",
                    description="Distance at which LJ potential is minimum",
                    physical_meaning="Equilibrium separation between atoms",
                )
            )

        # NVT thermostat relationship
        integrators = self.settings.get_integrator_fixes()
        for integrator in integrators:
            if integrator.fix_style == "nvt" and len(integrator.args) >= 3:
                # Nose-Hoover thermostat equation
                relationships.append(
                    ParameterRelationship(
                        name="nose_hoover_thermostat",
                        expression="dT/dt = (T_target - T) / tau_T",
                        variables=["T", "T_target", "tau_T", "t"],
                        relation_type="differential_equation",
                        description="Nose-Hoover thermostat temperature evolution",
                        physical_meaning="Temperature relaxation towards target with time constant tau_T",
                    )
                )

        return relationships

    def _get_pair_potential_form(self, style: str) -> str:
        """Get mathematical form of pair potential."""
        forms = {
            "lj/cut": "U(r) = 4ε[(σ/r)¹² - (σ/r)⁶]",
            "lj/cut/coul/long": "U(r) = 4ε[(σ/r)¹² - (σ/r)⁶] + qᵢqⱼ/(4πε₀r)",
            "eam": "E = Σᵢ Fᵢ(ρᵢ) + ½ Σᵢⱼ φᵢⱼ(rᵢⱼ)",
            "reax/c": "E = E_bond + E_over + E_under + E_val + E_pen + E_conj + E_hb + E_vdW + E_Coulomb",
            "sw": "E = Σᵢⱼ φ₂(rᵢⱼ) + Σᵢⱼₖ φ₃(rᵢⱼ, rᵢₖ, θᵢⱼₖ)",
            "tersoff": "E = ½ Σᵢⱼ f_c(rᵢⱼ)[f_R(rᵢⱼ) + bᵢⱼ f_A(rᵢⱼ)]",
            "buck": "U(r) = A exp(-r/ρ) - C/r⁶",
            "morse": "U(r) = Dₑ[1 - exp(-a(r-rₑ))]²",
        }
        return forms.get(style, f"Pair style: {style}")

    def _extract_symbolic_constraints(self) -> List[SymbolicConstraint]:
        """Extract symbolic mathematical constraints from simulation settings.

        These are the physical/mathematical constraints that must be satisfied
        for the simulation to be valid.
        """
        constraints = []

        # LJ potential constraints
        if self.settings.pair_style and self.settings.pair_style.style in (
            "lj/cut",
            "lj/cut/coul/long",
        ):
            # Get pair coefficients if available
            pair_coeffs = getattr(self.settings, "pair_coeffs", [])

            if pair_coeffs:
                for i, pc in enumerate(pair_coeffs):
                    epsilon = pc.get("epsilon", 0)
                    sigma = pc.get("sigma", 0)
                    r_cut = pc.get("r_cut", 0)

                    # Constraint: epsilon > 0 (energy well depth must be positive)
                    constraints.append(
                        SymbolicConstraint(
                            expression=f"epsilon_{i+1} > 0",
                            description="LJ energy well depth must be positive",
                            variables=[f"epsilon_{i+1}"],
                            confidence=1.0,
                            inferred_from="pair_coeff",
                        )
                    )

                    # Constraint: sigma > 0 (particle size must be positive)
                    constraints.append(
                        SymbolicConstraint(
                            expression=f"sigma_{i+1} > 0",
                            description="LJ length scale must be positive",
                            variables=[f"sigma_{i+1}"],
                            confidence=1.0,
                            inferred_from="pair_coeff",
                        )
                    )

                    # Constraint: r_cut > 2^(1/6) * sigma for attractive tail
                    if sigma > 0:
                        r_min = 2 ** (1 / 6) * sigma
                        constraints.append(
                            SymbolicConstraint(
                                expression=f"r_cut_{i+1} > 2**(1/6) * sigma_{i+1}",
                                description="Cutoff must include attractive region of LJ potential",
                                variables=[f"r_cut_{i+1}", f"sigma_{i+1}"],
                                confidence=0.8,
                                inferred_from="pair_style lj/cut",
                            )
                        )
            else:
                # Generic constraints when specific coeffs not available
                constraints.append(
                    SymbolicConstraint(
                        expression="epsilon > 0",
                        description="LJ energy well depth must be positive",
                        variables=["epsilon"],
                        confidence=1.0,
                        inferred_from="pair_style lj/cut",
                    )
                )
                constraints.append(
                    SymbolicConstraint(
                        expression="sigma > 0",
                        description="LJ length scale must be positive",
                        variables=["sigma"],
                        confidence=1.0,
                        inferred_from="pair_style lj/cut",
                    )
                )

        # NVT thermostat constraints
        integrators = self.settings.get_integrator_fixes()
        timestep = self.settings.timestep

        for integrator in integrators:
            if integrator.fix_style == "nvt":
                # Extract tau_T from fix nvt args: fix ID group-ID nvt temp Tstart Tstop Tdamp
                if len(integrator.args) >= 3:
                    try:
                        tau_t = float(integrator.args[2])  # Tdamp parameter

                        # Constraint: tau_T > 0
                        constraints.append(
                            SymbolicConstraint(
                                expression="tau_T > 0",
                                description="Thermostat damping time must be positive",
                                variables=["tau_T"],
                                confidence=1.0,
                                inferred_from="fix nvt",
                            )
                        )

                        # Constraint: dt < tau_T / 10 (stability condition)
                        if timestep and timestep > 0:
                            constraints.append(
                                SymbolicConstraint(
                                    expression="dt < tau_T / 10",
                                    description="Time step must be much smaller than thermostat damping for stability",
                                    variables=["dt", "tau_T"],
                                    confidence=0.9,
                                    inferred_from="NVT numerical stability analysis",
                                )
                            )

                            # Check if constraint is satisfied
                            if timestep < tau_t / 10:
                                constraints.append(
                                    SymbolicConstraint(
                                        expression=f"dt ({timestep}) < tau_T/10 ({tau_t/10:.4f}) ✓",
                                        description="Stability constraint satisfied",
                                        variables=["dt", "tau_T"],
                                        confidence=1.0,
                                        inferred_from="validation",
                                    )
                                )
                    except (ValueError, IndexError):
                        pass

                # Constraint: T > 0 (temperature must be positive)
                if len(integrator.args) >= 1:
                    try:
                        temp = float(integrator.args[0])
                        if temp > 0:
                            constraints.append(
                                SymbolicConstraint(
                                    expression="T > 0",
                                    description="Temperature must be positive (absolute temperature)",
                                    variables=["T"],
                                    confidence=1.0,
                                    inferred_from="fix nvt",
                                )
                            )
                    except (ValueError, IndexError):
                        pass

        # Time step constraint
        if timestep:
            constraints.append(
                SymbolicConstraint(
                    expression="dt > 0",
                    description="Time step must be positive",
                    variables=["dt"],
                    confidence=1.0,
                    inferred_from="timestep",
                )
            )

        return constraints

    def _extract_numerical_method(self) -> NumericalMethod:
        """Extract numerical method configuration."""
        method = NumericalMethod()

        # Discretization
        integrators = self.settings.get_integrator_fixes()
        if integrators:
            integrator = integrators[0]  # Take first integrator

            if integrator.fix_style == "nve":
                method.discretization.time_integrator = "velocity_verlet"
                method.discretization.order = 2
            elif integrator.fix_style == "nvt":
                method.discretization.time_integrator = (
                    "velocity_verlet_with_nose_hoover"
                )
                method.discretization.order = 2
            elif integrator.fix_style == "npt":
                method.discretization.time_integrator = (
                    "velocity_verlet_with_nose_hoover_and_barostat"
                )
                method.discretization.order = 2
            elif integrator.fix_style == "langevin":
                method.discretization.time_integrator = "langevin"
                method.discretization.order = 1

        if self.settings.timestep:
            method.discretization.time_step = self.settings.timestep

        method.discretization.space_discretization = "none"  # MD is particle-based

        # Solver (for minimization)
        # This would be extracted from "minimize" command

        return method

    def _extract_computational_graph(self) -> ComputationalGraph:
        """Extract computational graph with explicit/implicit loop distinction."""
        graph = ComputationalGraph()

        # Add nodes for each fix
        for fix in self.settings.fixes:
            node = self._fix_to_node(fix)
            if node:
                graph.add_node(node)

        # Add nodes for computes
        for compute in self.settings.computes:
            node = ComputationalNode(
                id=f"compute_{compute['compute_id']}",
                type="compute",
                math_semantics={
                    "operator_type": "observable_evaluation",
                    "updates": {
                        "target": compute["compute_style"],
                        "mode": UpdateMode.EXPLICIT_UPDATE.value,
                    },
                    "convergence": {"required": False},
                },
            )
            graph.add_node(node)

        # Add edges
        graph.add_edge(
            ComputationalEdge(
                from_node="pair_evaluation",
                to_node="integrator",
                data_type="force",
                dependency="read_only",
            )
        )

        # Execution topology
        graph.execution_topology = {
            "schedule": "sequential_per_timestep",
            "implicit_loops": [
                {
                    "loop_id": f"thermostat_{fix.fix_id}",
                    "nested_in": f"fix_{fix.fix_id}",
                    "convergence_guarantee": "none_explicitly",
                    "max_iterations_source": "fix_parameter",
                }
                for fix in self.settings.fixes
                if fix.fix_style in ("nvt", "npt")
            ],
        }

        return graph

    def _fix_to_node(self, fix: FixCommand) -> Optional[ComputationalNode]:
        """Convert fix to computational node."""

        # Time integrators
        if fix.fix_style in ("nve", "nvt", "npt", "langevin"):
            mode = (
                UpdateMode.IMPLICIT_LOOP
                if fix.fix_style in ("nvt", "npt")
                else UpdateMode.EXPLICIT_UPDATE
            )
            return ComputationalNode(
                id=f"fix_{fix.fix_id}",
                type="integrator",
                math_semantics={
                    "operator_type": (
                        "symplectic_integrator"
                        if fix.fix_style == "nve"
                        else "thermostat_integrator"
                    ),
                    "updates": {
                        "target": "phase_space",
                        "variables": ["position", "velocity"],
                        "mode": mode.value,
                    },
                    "convergence": (
                        {
                            "required": fix.fix_style in ("nvt", "npt"),
                            "criterion": (
                                "temperature_target_residual"
                                if fix.fix_style in ("nvt", "npt")
                                else None
                            ),
                        }
                        if fix.fix_style in ("nvt", "npt")
                        else {"required": False}
                    ),
                },
            )

        # Deformation
        if fix.fix_style == "deform":
            return ComputationalNode(
                id=f"fix_{fix.fix_id}",
                type="deformation",
                math_semantics={
                    "operator_type": "homogeneous_deformation",
                    "updates": {
                        "target": "box",
                        "variables": ["box_x", "box_y", "box_z"],
                        "mode": UpdateMode.EXPLICIT_UPDATE.value,
                    },
                    "convergence": {"required": False},
                },
            )

        # Constraints
        if fix.fix_style in ("spring", "wall"):
            return ComputationalNode(
                id=f"fix_{fix.fix_id}",
                type="constraint",
                math_semantics={
                    "operator_type": "constraint_projection",
                    "updates": {
                        "target": "force",
                        "variables": ["f_x", "f_y", "f_z"],
                        "mode": UpdateMode.EXPLICIT_UPDATE.value,
                    },
                    "convergence": {"required": False},
                },
            )

        return None

    def _extract_conservation_properties(self) -> Dict[str, Any]:
        """Extract conservation properties."""
        props = {}

        # Check integrator type
        integrators = self.settings.get_integrator_fixes()

        for integrator in integrators:
            if integrator.fix_style == "nve":
                props["energy"] = {
                    "preserved": True,
                    "mechanism": "symplectic_geometry",
                    "order": 2,
                }
                props["momentum"] = {"preserved": True}
                props["angular_momentum"] = {"preserved": True}

            elif integrator.fix_style in ("nvt", "npt"):
                props["energy"] = {
                    "preserved": False,
                    "mechanism": f"thermostat_{integrator.fix_style}",
                    "fluctuation": "controlled",
                }

        return props

    def _extract_raw_symbols(self) -> Dict[str, Any]:
        """Extract raw symbols from the input."""
        symbols = {
            "fixes": {
                f.fix_id: {"style": f.fix_style, "group": f.group_id}
                for f in self.settings.fixes
            },
            "computes": {c["compute_id"]: c for c in self.settings.computes},
            "variables": self.settings.variables,
            "groups": self.settings.groups,
            "units": self.settings.units,
        }
        if self.dynamics_result:
            symbols["dynamics_analysis"] = self.dynamics_result
        return symbols

    def _analyze_dynamics(self, log_data: Any) -> Optional[Dict[str, Any]]:
        """Run dynamics analysis on log file thermo data."""
        try:
            import numpy as np
            from math_anything.tools.dynamics import DynamicsAnalyzer

            if (
                not log_data
                or not hasattr(log_data, "thermo_data")
                or not log_data.thermo_data
            ):
                return None

            thermo = log_data.thermo_data
            energy_key = None
            for key in ["TotEng", "PotEng", "Temp", "Press"]:
                if key in thermo[0]:
                    energy_key = key
                    break

            if energy_key is None:
                return None

            ts = np.array([row[energy_key] for row in thermo])
            if len(ts) < 50:
                return None

            analyzer = DynamicsAnalyzer()
            result = analyzer.analyze(ts)
            return result.to_dict()
        except Exception:
            return None
