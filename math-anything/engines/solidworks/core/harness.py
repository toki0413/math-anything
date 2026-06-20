"""SolidWorks Simulation Harness implementation.

Extracts mathematical structures from SolidWorks Simulation FEA.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add core to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent.parent / "core"))

from math_anything.core.harness import Harness
from math_anything.schemas.math_schema import (
    BoundaryCondition,
    ComputationalGraph,
    DiscretizationScheme,
    GoverningEquation,
    MathematicalObject,
    MathSchema,
    NumericalMethod,
    TensorComponent,
)
from math_anything.schemas.registry import HarnessRegistry

from .extractor import SolidWorksExtractor
from .parser import CWRParser, MaterialParser, StudyParser


class SolidWorksHarness(Harness):
    """Harness for SolidWorks Simulation.

    SolidWorks Simulation is an integrated FEA tool within SolidWorks CAD.
    It provides structural, thermal, and modal analysis capabilities.

    Mathematical focus:
    - Linear elasticity
    - Small deformation theory
    - Isotropic material models
    - Static, modal, and thermal analyses
    - Shell and solid elements

    Supported studies:
    - Static
    - Frequency (Modal)
    - Buckling
    - Thermal
    - Optimization
    - Nonlinear (Premium)

    Example:
        ```python
        harness = SolidWorksHarness()
        schema = harness.extract_math(
            cwr_file="results.cwr",
            study_name="Static Analysis",
        )
        ```
    """

    ENGINE_NAME = "solidworks_simulation"
    ENGINE_VERSION = "2024"
    SUPPORTED_EXTENSIONS = [".cwr", ".cwd", ".sldprt", ".sldasm"]

    # Study types
    STUDY_TYPES = {
        "static": "Static Analysis",
        "frequency": "Frequency (Modal)",
        "buckling": "Buckling",
        "thermal": "Thermal",
        "nonlinear": "Nonlinear",
        "drop_test": "Drop Test",
        "fatigue": "Fatigue",
        "optimization": "Optimization",
        "pressure_vessel": "Pressure Vessel Design",
    }

    # Element types
    ELEMENT_TYPES = {
        "draft_quality": {"order": "linear", "nodes": 4},
        "high_quality": {"order": "quadratic", "nodes": 10},
        "shell": {"order": "quadratic", "nodes": 6},
        "beam": {"order": "quadratic", "nodes": 3},
    }

    def __init__(self):
        self.extractor = SolidWorksExtractor()
        self.cwr_parser = CWRParser()
        self.study_parser = StudyParser()
        self.material_parser = MaterialParser()
        self._current_files: Dict[str, str] = {}

    def extract_math(
        self,
        cwr_file: Optional[str] = None,
        sldprt_file: Optional[str] = None,
        study_name: str = "Static Analysis",
        study_type: str = "static",
        **kwargs,
    ) -> MathSchema:
        """Extract mathematical schema from SolidWorks Simulation.

        Args:
            cwr_file: Results file (.cwr)
            sldprt_file: Part file (.sldprt)
            study_name: Name of the study
            study_type: Type of analysis
            **kwargs: Additional parameters

        Returns:
            MathSchema with extracted mathematical structures
        """
        self._current_files = {
            "cwr": cwr_file,
            "sldprt": sldprt_file,
        }

        # Parse input files
        results_data = {}
        study_data = {}

        if cwr_file and os.path.exists(cwr_file):
            results_data = self.cwr_parser.parse(cwr_file)

        # Build schema
        schema = MathSchema(
            schema_version="1.0.0",
            engine=self.ENGINE_NAME,
            engine_version=self.ENGINE_VERSION,
        )

        # Add governing equations
        self._add_governing_equations(schema, study_type, results_data)

        # Add boundary conditions
        self._add_boundary_conditions(schema, results_data)

        # Add mathematical objects
        self._add_mathematical_objects(schema, results_data)

        # Add numerical method
        self._add_numerical_method(schema, results_data)

        # Add computational graph
        self._add_computational_graph(schema, study_type)

        # Add CAD integration info
        self._add_cad_info(schema, sldprt_file)

        return schema

    def _add_governing_equations(
        self,
        schema: MathSchema,
        study_type: str,
        results_data: Dict[str, Any],
    ):
        """Add governing equations based on study type."""

        if study_type == "static":
            # Equilibrium equation
            equilibrium = GoverningEquation(
                id="equilibrium",
                type="pde",
                name="Equilibrium Equation",
                mathematical_form="∇·σ + b = 0",
                description="Static equilibrium (no inertia)",
                variables=[
                    {
                        "name": "σ",
                        "description": "Cauchy stress tensor",
                        "type": "tensor",
                        "rank": 2,
                    },
                    {"name": "b", "description": "Body force", "type": "vector"},
                ],
            )
            schema.add_governing_equation(equilibrium)

            # Constitutive (Hooke's law)
            hooke = GoverningEquation(
                id="hooke_law",
                type="constitutive",
                name="Hooke's Law",
                mathematical_form="σ = λ(tr ε)I + 2με",
                description="Linear elastic isotropic material",
                variables=[
                    {"name": "σ", "description": "Stress", "type": "tensor", "rank": 2},
                    {"name": "ε", "description": "Strain", "type": "tensor", "rank": 2},
                    {
                        "name": "λ",
                        "description": "Lame's first parameter",
                        "type": "scalar",
                    },
                    {"name": "μ", "description": "Shear modulus", "type": "scalar"},
                ],
            )
            schema.add_governing_equation(hooke)

        elif study_type == "frequency":
            # Modal equation
            modal = GoverningEquation(
                id="modal_equation",
                type="eigenvalue_problem",
                name="Modal Analysis",
                mathematical_form="(K - ω²M)φ = 0",
                description="Free vibration eigenvalue problem",
                variables=[
                    {"name": "K", "description": "Stiffness matrix", "type": "matrix"},
                    {"name": "M", "description": "Mass matrix", "type": "matrix"},
                    {"name": "ω", "description": "Natural frequency", "type": "scalar"},
                    {"name": "φ", "description": "Mode shape", "type": "vector"},
                ],
            )
            schema.add_governing_equation(modal)

        elif study_type == "thermal":
            # Fourier heat conduction
            fourier = GoverningEquation(
                id="fourier_law",
                type="pde",
                name="Fourier Heat Conduction",
                mathematical_form="∇·(k∇T) + Q = 0",
                description="Steady-state heat conduction",
                variables=[
                    {"name": "T", "description": "Temperature", "type": "scalar"},
                    {
                        "name": "k",
                        "description": "Thermal conductivity",
                        "type": "scalar",
                    },
                    {"name": "Q", "description": "Heat generation", "type": "scalar"},
                ],
            )
            schema.add_governing_equation(fourier)

        elif study_type == "buckling":
            # Linear buckling
            buckling = GoverningEquation(
                id="buckling_equation",
                type="eigenvalue_problem",
                name="Linear Buckling",
                mathematical_form="(K + λK_σ)φ = 0",
                description="Linear buckling eigenvalue problem",
                variables=[
                    {"name": "K", "description": "Elastic stiffness", "type": "matrix"},
                    {
                        "name": "K_σ",
                        "description": "Geometric stiffness",
                        "type": "matrix",
                    },
                    {"name": "λ", "description": "Load factor", "type": "scalar"},
                    {"name": "φ", "description": "Buckling mode", "type": "vector"},
                ],
            )
            schema.add_governing_equation(buckling)

    def _add_boundary_conditions(
        self, schema: MathSchema, results_data: Dict[str, Any]
    ):
        """Add boundary conditions."""

        # Fixed constraints
        fixed = results_data.get("fixed_constraints", [])
        for i, constraint in enumerate(fixed):
            bc = BoundaryCondition(
                id=f"fixed_{i}",
                type="dirichlet",
                region=constraint.get("faces", ""),
                mathematical_form="u = 0",
                variables=[
                    {"name": "u", "description": "Displacement", "type": "vector"}
                ],
                physical_meaning="Fixed geometric constraint",
            )
            schema.add_boundary_condition(bc)

        # Forces
        forces = results_data.get("forces", [])
        for i, force in enumerate(forces):
            bc = BoundaryCondition(
                id=f"force_{i}",
                type="neumann",
                region=force.get("faces", ""),
                mathematical_form=f"t = {force.get('magnitude', 0)} N",
                variables=[{"name": "t", "description": "Traction", "type": "vector"}],
                physical_meaning=force.get("type", "Applied force"),
            )
            schema.add_boundary_condition(bc)

        # Pressure
        pressures = results_data.get("pressures", [])
        for i, pressure in enumerate(pressures):
            bc = BoundaryCondition(
                id=f"pressure_{i}",
                type="neumann",
                region=pressure.get("faces", ""),
                mathematical_form=f"σ·n = -{pressure.get('value', 0)} n",
                variables=[{"name": "σ", "description": "Stress", "type": "tensor"}],
                physical_meaning="Pressure load",
            )
            schema.add_boundary_condition(bc)

    def _add_mathematical_objects(
        self, schema: MathSchema, results_data: Dict[str, Any]
    ):
        """Add mathematical objects."""

        num_nodes = results_data.get("num_nodes", 0)
        num_elements = results_data.get("num_elements", 0)

        # Stress tensor
        stress = MathematicalObject(
            id="stress_tensor",
            name="Cauchy Stress",
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

        # Strain tensor
        strain = MathematicalObject(
            id="strain_tensor",
            name="Engineering Strain",
            type="tensor_field",
            symbol="ε",
            tensor_rank=2,
            components=[
                TensorComponent(name="ε_xx", symbol="ε_xx", index=[0, 0]),
                TensorComponent(name="ε_yy", symbol="ε_yy", index=[1, 1]),
                TensorComponent(name="ε_zz", symbol="ε_zz", index=[2, 2]),
                TensorComponent(name="γ_xy", symbol="γ_xy", index=[0, 1]),
                TensorComponent(name="γ_yz", symbol="γ_yz", index=[1, 2]),
                TensorComponent(name="γ_xz", symbol="γ_xz", index=[0, 2]),
            ],
        )
        schema.add_mathematical_object(strain)

        # Displacement
        if num_nodes > 0:
            displacement = MathematicalObject(
                id="displacement",
                name="Displacement Vector",
                type="vector_field",
                symbol="u",
                tensor_rank=1,
            )
            schema.add_mathematical_object(displacement)

        # Von Mises stress (derived scalar)
        von_mises = MathematicalObject(
            id="von_mises",
            name="Von Mises Stress",
            type="scalar_field",
            symbol="σ_vm",
            tensor_rank=0,
        )
        schema.add_mathematical_object(von_mises)

    def _add_numerical_method(self, schema: MathSchema, results_data: Dict[str, Any]):
        """Add FEM numerical method."""

        element_quality = results_data.get("element_quality", "high")
        element_info = self.ELEMENT_TYPES.get(
            element_quality, {"order": "quadratic", "nodes": 10}
        )

        method = NumericalMethod(
            id="fem_sw",
            name="Finite Element Method (SolidWorks)",
            description=f"FEM with {element_quality} tetrahedral elements",
            parameters={
                "element_type": "tetrahedral",
                "element_quality": element_quality,
                "element_order": element_info["order"],
                "nodes_per_element": element_info["nodes"],
                "formulation": "displacement",
                "solver": "sparse_direct",
            },
        )

        discretization = DiscretizationScheme(
            spatial_order=2 if element_info["order"] == "quadratic" else 1,
            temporal_order=0,
            mesh_type="tetrahedral",
        )
        method.discretization = discretization

        schema.add_numerical_method(method)

    def _add_computational_graph(self, schema: MathSchema, study_type: str):
        """Add computational graph."""

        graph = ComputationalGraph(
            id="sw_simulation",
            name="SolidWorks Simulation Workflow",
            description=self.STUDY_TYPES.get(study_type, study_type),
        )

        nodes = [
            ("cad_model", "CAD Model"),
            ("material", "Assign Material"),
            ("fixture", "Apply Fixtures"),
            ("load", "Apply Loads"),
            ("mesh", "Create Mesh"),
            ("solve", "Run Study"),
            ("results", "View Results"),
        ]

        for node_id, node_name in nodes:
            graph.add_node(node_id, node_name)

        edges = [
            ("cad_model", "material"),
            ("material", "fixture"),
            ("fixture", "load"),
            ("load", "mesh"),
            ("mesh", "solve"),
            ("solve", "results"),
        ]

        for from_node, to_node in edges:
            graph.add_edge(from_node, to_node)

        schema.computational_graphs.append(graph)

    def _add_cad_info(self, schema: MathSchema, sldprt_file: Optional[str]):
        """Add CAD integration information."""

        if sldprt_file:
            schema.metadata["cad_integration"] = {
                "software": "SolidWorks",
                "file": sldprt_file,
                "associativity": True,
                "bi_directional": True,
            }

    def get_capabilities(self) -> Dict[str, Any]:
        """Return harness capabilities."""
        return {
            "engine_name": self.ENGINE_NAME,
            "engine_version": self.ENGINE_VERSION,
            "supported_formats": self.SUPPORTED_EXTENSIONS,
            "study_types": list(self.STUDY_TYPES.keys()),
            "element_types": list(self.ELEMENT_TYPES.keys()),
            "features": [
                "cad_integration",
                "associative_geometry",
                "design_study",
                "optimization",
                "shell_elements",
                "beam_elements",
                "solid_elements",
            ],
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        if not any(input_data.get(f) for f in ["cwr_file", "sldprt_file"]):
            return False

        for key in ["cwr_file", "sldprt_file"]:
            filepath = input_data.get(key)
            if filepath and not os.path.exists(filepath):
                return False

        return True


# Register harness
HarnessRegistry.register(SolidWorksHarness)
