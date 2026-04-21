"""COMSOL Multiphysics Harness implementation.

Extracts mathematical structures from COMSOL multiphysics simulations.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add core to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent.parent / "core"))

from math_anything.core.harness import Harness
from math_anything.schemas.math_schema import (BoundaryCondition,
                                               ComputationalGraph,
                                               DiscretizationScheme,
                                               GoverningEquation,
                                               MathematicalObject, MathSchema,
                                               NumericalMethod,
                                               TensorComponent)
from math_anything.schemas.registry import HarnessRegistry

from .extractor import ComsolExtractor
from .parser import JavaParser, ModelParser, MPHParser


class ComsolHarness(Harness):
    """Harness for COMSOL Multiphysics simulations.

    COMSOL is a multiphysics simulation platform that supports
    coupling between various physics phenomena.

    Mathematical focus:
    - PDE-based modeling
    - Weak form formulations
    - Multiphysics coupling
    - Moving mesh (ALE)
    - Time-dependent and frequency-domain analyses

    Supported physics:
    - Solid mechanics
    - Heat transfer
    - Fluid flow
    - Electromagnetics
    - Chemical transport
    - Acoustics

    Example:
        ```python
        harness = ComsolHarness()
        schema = harness.extract_math(
            mph_file="model.mph",
            physics=["solid_mechanics", "heat_transfer"],
        )
        ```
    """

    ENGINE_NAME = "comsol"
    ENGINE_VERSION = "6.1"
    SUPPORTED_EXTENSIONS = [".mph", ".java", ".m", ".xml"]

    # Physics interfaces supported
    PHYSICS_INTERFACES = {
        "solid_mechanics": "Structural Mechanics Module",
        "heat_transfer": "Heat Transfer Module",
        "fluid_flow": "CFD Module",
        "electromagnetics": "AC/DC or RF Module",
        "chemical_transport": "Chemical Reaction Engineering",
        "acoustics": "Acoustics Module",
        "structural_mechanics": "Structural Mechanics",
        "electric_currents": "Electric Currents",
        "magnetic_fields": "Magnetic Fields",
        "transport_of_diluted_species": "Transport of Diluted Species",
    }

    # Study types
    STUDY_TYPES = {
        "stationary": "Steady-state analysis",
        "time_dependent": "Transient analysis",
        "eigenfrequency": "Modal analysis",
        "frequency_domain": "Harmonic analysis",
        "eigenvalue": "Eigenvalue analysis",
        "parametric_sweep": "Parametric sweep",
        "optimization": "Optimization study",
    }

    def __init__(self):
        self.extractor = ComsolExtractor()
        self.mph_parser = MPHParser()
        self.model_parser = ModelParser()
        self.java_parser = JavaParser()
        self._current_files: Dict[str, str] = {}

    def extract_math(
        self,
        mph_file: Optional[str] = None,
        java_file: Optional[str] = None,
        physics: Optional[List[str]] = None,
        study_type: str = "stationary",
        **kwargs,
    ) -> MathSchema:
        """Extract mathematical schema from COMSOL model.

        Args:
            mph_file: COMSOL model file (.mph)
            java_file: COMSOL Java API file
            physics: List of physics interfaces to extract
            study_type: Type of study/analysis
            **kwargs: Additional parameters

        Returns:
            MathSchema with extracted mathematical structures
        """
        self._current_files = {
            "mph": mph_file,
            "java": java_file,
        }

        physics = physics or ["solid_mechanics"]

        # Parse input files
        model_data = {}

        if mph_file and os.path.exists(mph_file):
            model_data = self.mph_parser.parse(mph_file)
        elif java_file and os.path.exists(java_file):
            model_data = self.java_parser.parse(java_file)

        # Build schema
        schema = MathSchema(
            schema_version="1.0.0",
            engine=self.ENGINE_NAME,
            engine_version=self.ENGINE_VERSION,
        )

        # Add governing equations for each physics
        for phys in physics:
            self._add_physics_equations(schema, phys, model_data)

        # Add multiphysics coupling if multiple physics
        if len(physics) > 1:
            self._add_multiphysics_coupling(schema, physics, model_data)

        # Add boundary conditions
        self._add_boundary_conditions(schema, model_data)

        # Add mathematical objects
        self._add_mathematical_objects(schema, physics, model_data)

        # Add numerical method
        self._add_numerical_method(schema, model_data)

        # Add computational graph
        self._add_computational_graph(schema, physics, study_type)

        return schema

    def _add_physics_equations(
        self,
        schema: MathSchema,
        physics: str,
        model_data: Dict[str, Any],
    ):
        """Add governing equations for a specific physics."""

        if physics in ["solid_mechanics", "structural_mechanics"]:
            # Linear elasticity in weak form
            weak_form = GoverningEquation(
                id=f"{physics}_weak_form",
                type="weak_form_pde",
                name="Linear Elasticity Weak Form",
                mathematical_form="∫_Ω σ:δε dΩ = ∫_Ω b·δu dΩ + ∫_Γ t·δu dΓ",
                description="Principle of virtual work",
                variables=[
                    {
                        "name": "σ",
                        "description": "Cauchy stress",
                        "type": "tensor",
                        "rank": 2,
                    },
                    {"name": "ε", "description": "Strain", "type": "tensor", "rank": 2},
                    {"name": "u", "description": "Displacement", "type": "vector"},
                    {"name": "b", "description": "Body force", "type": "vector"},
                    {"name": "t", "description": "Traction", "type": "vector"},
                ],
            )
            schema.add_governing_equation(weak_form)

        elif physics == "heat_transfer":
            # Heat equation
            heat_eq = GoverningEquation(
                id="heat_equation",
                type="pde",
                name="Heat Transfer Equation",
                mathematical_form="ρc_p ∂T/∂t + ∇·(-k∇T) = Q",
                description="Transient heat conduction with source",
                variables=[
                    {"name": "T", "description": "Temperature", "type": "scalar"},
                    {
                        "name": "k",
                        "description": "Thermal conductivity",
                        "type": "tensor",
                        "rank": 2,
                    },
                    {"name": "Q", "description": "Heat source", "type": "scalar"},
                ],
            )
            schema.add_governing_equation(heat_eq)

        elif physics == "fluid_flow":
            # Navier-Stokes equations
            continuity = GoverningEquation(
                id="continuity",
                type="pde",
                name="Continuity Equation",
                mathematical_form="∇·u = 0",
                description="Incompressibility constraint",
                variables=[
                    {"name": "u", "description": "Velocity vector", "type": "vector"},
                ],
            )
            schema.add_governing_equation(continuity)

            momentum = GoverningEquation(
                id="navier_stokes",
                type="pde",
                name="Navier-Stokes Momentum",
                mathematical_form="ρ(∂u/∂t + u·∇u) = -∇p + ∇·τ + f",
                description="Momentum conservation",
                variables=[
                    {"name": "u", "description": "Velocity", "type": "vector"},
                    {"name": "p", "description": "Pressure", "type": "scalar"},
                    {
                        "name": "τ",
                        "description": "Viscous stress",
                        "type": "tensor",
                        "rank": 2,
                    },
                    {"name": "f", "description": "Body force", "type": "vector"},
                ],
            )
            schema.add_governing_equation(momentum)

        elif physics == "electromagnetics":
            # Maxwell's equations (quasi-static)
            maxwell = GoverningEquation(
                id="maxwell_quasistatic",
                type="pde",
                name="Maxwell's Equations (Quasi-static)",
                mathematical_form="∇×(μ⁻¹∇×A) = J - σ∂A/∂t",
                description="Magnetic vector potential formulation",
                variables=[
                    {
                        "name": "A",
                        "description": "Magnetic vector potential",
                        "type": "vector",
                    },
                    {"name": "J", "description": "Current density", "type": "vector"},
                    {
                        "name": "μ",
                        "description": "Permeability",
                        "type": "tensor",
                        "rank": 2,
                    },
                    {
                        "name": "σ",
                        "description": "Conductivity",
                        "type": "tensor",
                        "rank": 2,
                    },
                ],
            )
            schema.add_governing_equation(maxwell)

        elif physics == "acoustics":
            # Helmholtz equation
            helmholtz = GoverningEquation(
                id="helmholtz",
                type="pde",
                name="Helmholtz Equation",
                mathematical_form="∇²p + k²p = 0",
                description="Time-harmonic acoustic pressure",
                variables=[
                    {"name": "p", "description": "Acoustic pressure", "type": "scalar"},
                    {"name": "k", "description": "Wavenumber", "type": "scalar"},
                ],
            )
            schema.add_governing_equation(helmholtz)

    def _add_multiphysics_coupling(
        self,
        schema: MathSchema,
        physics: List[str],
        model_data: Dict[str, Any],
    ):
        """Add multiphysics coupling equations."""

        # Thermal-stress coupling
        if "solid_mechanics" in physics and "heat_transfer" in physics:
            thermal_stress = GoverningEquation(
                id="thermal_stress_coupling",
                type="coupling",
                name="Thermal Stress Coupling",
                mathematical_form="σ_th = -Eα(T-T_ref)/(1-2ν) I",
                description="Thermal strain contribution to stress",
                variables=[
                    {
                        "name": "σ_th",
                        "description": "Thermal stress",
                        "type": "tensor",
                        "rank": 2,
                    },
                    {"name": "T", "description": "Temperature", "type": "scalar"},
                    {
                        "name": "T_ref",
                        "description": "Reference temperature",
                        "type": "scalar",
                    },
                    {
                        "name": "α",
                        "description": "Thermal expansion coefficient",
                        "type": "scalar",
                    },
                ],
            )
            schema.add_governing_equation(thermal_stress)

        # Fluid-structure interaction
        if "fluid_flow" in physics and "solid_mechanics" in physics:
            fsi = GoverningEquation(
                id="fsi_coupling",
                type="coupling",
                name="Fluid-Structure Interaction",
                mathematical_form="σ_s·n = -p n + τ·n on Γ_interface",
                description="Traction continuity at fluid-solid interface",
                variables=[
                    {
                        "name": "σ_s",
                        "description": "Solid stress",
                        "type": "tensor",
                        "rank": 2,
                    },
                    {"name": "p", "description": "Fluid pressure", "type": "scalar"},
                    {
                        "name": "τ",
                        "description": "Viscous stress",
                        "type": "tensor",
                        "rank": 2,
                    },
                    {"name": "n", "description": "Interface normal", "type": "vector"},
                ],
            )
            schema.add_governing_equation(fsi)

        # Joule heating
        if "electromagnetics" in physics and "heat_transfer" in physics:
            joule = GoverningEquation(
                id="joule_heating",
                type="coupling",
                name="Joule Heating",
                mathematical_form="Q = J·E = σ|E|²",
                description="Electromagnetic heat generation",
                variables=[
                    {"name": "Q", "description": "Heat generation", "type": "scalar"},
                    {"name": "J", "description": "Current density", "type": "vector"},
                    {"name": "E", "description": "Electric field", "type": "vector"},
                    {
                        "name": "σ",
                        "description": "Electric conductivity",
                        "type": "scalar",
                    },
                ],
            )
            schema.add_governing_equation(joule)

    def _add_boundary_conditions(self, schema: MathSchema, model_data: Dict[str, Any]):
        """Add boundary conditions from model."""

        bc_data = model_data.get("boundary_conditions", [])

        for bc in bc_data:
            boundary_condition = BoundaryCondition(
                id=f"bc_{bc.get('id', 'unknown')}",
                type=bc.get("type", "dirichlet"),
                region=bc.get("boundary", ""),
                mathematical_form=bc.get("equation", ""),
                variables=[
                    {
                        "name": bc.get("variable", "u"),
                        "type": bc.get("var_type", "scalar"),
                    },
                ],
                physical_meaning=bc.get("description", "Boundary condition"),
            )
            schema.add_boundary_condition(boundary_condition)

    def _add_mathematical_objects(
        self,
        schema: MathSchema,
        physics: List[str],
        model_data: Dict[str, Any],
    ):
        """Add mathematical objects for the physics."""

        # Degrees of freedom
        num_dofs = model_data.get("num_dofs", 0)

        if num_dofs > 0:
            solution_vector = MathematicalObject(
                id="solution_vector",
                name="Solution Vector",
                type="vector",
                symbol="U",
                shape=(num_dofs,),
                description="Degrees of freedom vector",
            )
            schema.add_mathematical_object(solution_vector)

        # Physics-specific objects
        if "solid_mechanics" in physics:
            stress = MathematicalObject(
                id="stress_tensor",
                name="Cauchy Stress Tensor",
                type="tensor_field",
                symbol="σ",
                tensor_rank=2,
                components=[
                    TensorComponent(name="σ_xx", symbol="σ_xx", index=[0, 0]),
                    TensorComponent(name="σ_yy", symbol="σ_yy", index=[1, 1]),
                    TensorComponent(name="σ_zz", symbol="σ_zz", index=[2, 2]),
                    TensorComponent(name="σ_xy", symbol="σ_xy", index=[0, 1]),
                    TensorComponent(name="σ_yz", symbol="σ_yz", index=[1, 2]),
                    TensorComponent(name="σ_xz", symbol="σ_xz", index=[0, 2]),
                ],
            )
            schema.add_mathematical_object(stress)

        if "heat_transfer" in physics:
            temp_field = MathematicalObject(
                id="temperature_field",
                name="Temperature Field",
                type="scalar_field",
                symbol="T",
                tensor_rank=0,
            )
            schema.add_mathematical_object(temp_field)

            heat_flux = MathematicalObject(
                id="heat_flux",
                name="Heat Flux Vector",
                type="vector_field",
                symbol="q",
                tensor_rank=1,
            )
            schema.add_mathematical_object(heat_flux)

        if "fluid_flow" in physics:
            velocity = MathematicalObject(
                id="velocity_field",
                name="Velocity Field",
                type="vector_field",
                symbol="u",
                tensor_rank=1,
            )
            schema.add_mathematical_object(velocity)

            pressure = MathematicalObject(
                id="pressure_field",
                name="Pressure Field",
                type="scalar_field",
                symbol="p",
                tensor_rank=0,
            )
            schema.add_mathematical_object(pressure)

    def _add_numerical_method(self, schema: MathSchema, model_data: Dict[str, Any]):
        """Add FEM numerical method."""

        element_order = model_data.get("element_order", "quadratic")
        solver_type = model_data.get("solver_type", "direct")

        method = NumericalMethod(
            id="fem_comsol",
            name="Finite Element Method (COMSOL)",
            description=f"Galerkin FEM with {element_order} elements",
            parameters={
                "element_order": element_order,
                "formulation": "galerkin",
                "shape_functions": "lagrange",
                "quadrature": "gaussian",
                "solver": solver_type,
            },
        )

        discretization = DiscretizationScheme(
            spatial_order=2 if element_order == "quadratic" else 1,
            temporal_order=0,  # Depends on study
            mesh_type="unstructured",
        )
        method.discretization = discretization

        schema.add_numerical_method(method)

    def _add_computational_graph(
        self,
        schema: MathSchema,
        physics: List[str],
        study_type: str,
    ):
        """Add computational graph."""

        graph = ComputationalGraph(
            id="comsol_analysis",
            name="COMSOL Multiphysics Workflow",
            description=f"{', '.join(physics)} with {study_type} study",
        )

        # Add nodes
        nodes = [
            ("geometry", "Geometry Definition"),
            ("materials", "Material Properties"),
            ("physics", "Physics Setup"),
            ("mesh", "Mesh Generation"),
            ("study", "Study Setup"),
            ("solve", "Solve"),
            ("postprocess", "Postprocessing"),
        ]

        for node_id, node_name in nodes:
            graph.add_node(node_id, node_name)

        # Add edges
        edges = [
            ("geometry", "materials"),
            ("materials", "physics"),
            ("physics", "mesh"),
            ("mesh", "study"),
            ("study", "solve"),
            ("solve", "postprocess"),
        ]

        for from_node, to_node in edges:
            graph.add_edge(from_node, to_node)

        schema.computational_graphs.append(graph)

    def get_capabilities(self) -> Dict[str, Any]:
        """Return harness capabilities."""
        return {
            "engine_name": self.ENGINE_NAME,
            "engine_version": self.ENGINE_VERSION,
            "supported_formats": self.SUPPORTED_EXTENSIONS,
            "physics_interfaces": list(self.PHYSICS_INTERFACES.keys()),
            "study_types": list(self.STUDY_TYPES.keys()),
            "features": [
                "multiphysics_coupling",
                "equation_based_modeling",
                "weak_formulation",
                "moving_mesh",
                "parameterized_geometry",
            ],
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        if not any(input_data.get(f) for f in ["mph_file", "java_file"]):
            return False

        for key in ["mph_file", "java_file"]:
            filepath = input_data.get(key)
            if filepath and not os.path.exists(filepath):
                return False

        return True


# Register harness
HarnessRegistry.register(ComsolHarness)
