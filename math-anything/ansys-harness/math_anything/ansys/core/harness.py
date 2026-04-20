"""Ansys Harness implementation.

Extracts mathematical structures from Ansys FEA analysis.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add core to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent.parent / 'core'))

from math_anything.core.harness import MathAnythingHarness
from math_anything.schemas import (
    MathSchema, GoverningEquation, BoundaryCondition,
    ComputationalGraph, MathematicalObject, TensorComponent,
    NumericalMethod, Discretization,
)
from math_anything.schemas.registry import HarnessRegistry

from .extractor import AnsysExtractor
from .parser import APDLParser, CDBParser, RSTParser


class AnsysHarness(MathAnythingHarness):
    """Harness for Ansys FEA simulations.
    
    Ansys is a comprehensive FEA software for structural, thermal,
    fluid, and electromagnetic analyses.
    
    Mathematical focus:
    - Linear/nonlinear elasticity
    - Heat conduction and convection
    - Modal and harmonic analysis
    - Transient dynamics
    - Contact mechanics
    
    Supported analyses:
    - Static structural
    - Modal
    - Transient structural
    - Thermal
    - Coupled-field
    
    Example:
        ```python
        harness = AnsysHarness()
        schema = harness.extract_math(
            db_file="model.cdb",
            rst_file="results.rst",
            analysis_type="structural_static",
        )
        ```
    """
    
    ENGINE_NAME = "ansys"
    ENGINE_VERSION = "2023 R2"
    SUPPORTED_EXTENSIONS = ['.cdb', '.db', '.rst', '.rth', '.rmg', '.apdl', '.inp']
    
    # Analysis types supported
    ANALYSIS_TYPES = {
        'static_structural': 'Static structural analysis',
        'modal': 'Modal analysis',
        'harmonic': 'Harmonic analysis',
        'transient_structural': 'Transient structural analysis',
        'thermal_steady': 'Steady-state thermal',
        'thermal_transient': 'Transient thermal',
        'buckling': 'Linear buckling',
        'spectrum': 'Spectrum analysis',
    }
    
    # Element types mapping
    ELEMENT_TYPES = {
        'solid185': {'type': 'solid', 'order': 'linear', 'nodes': 8},
        'solid186': {'type': 'solid', 'order': 'quadratic', 'nodes': 20},
        'solid187': {'type': 'solid', 'order': 'quadratic', 'nodes': 10},
        'beam188': {'type': 'beam', 'order': 'linear', 'nodes': 2},
        'beam189': {'type': 'beam', 'order': 'quadratic', 'nodes': 3},
        'shell181': {'type': 'shell', 'order': 'linear', 'nodes': 4},
        'shell281': {'type': 'shell', 'order': 'quadratic', 'nodes': 8},
        'link180': {'type': 'link', 'order': 'linear', 'nodes': 2},
    }
    
    def __init__(self):
        self.extractor = AnsysExtractor()
        self.apdl_parser = APDLParser()
        self.cdb_parser = CDBParser()
        self.rst_parser = RSTParser()
        self._current_files: Dict[str, str] = {}
        
    def extract_math(
        self,
        db_file: Optional[str] = None,
        rst_file: Optional[str] = None,
        apdl_file: Optional[str] = None,
        analysis_type: str = 'static_structural',
        **kwargs
    ) -> MathSchema:
        """Extract mathematical schema from Ansys analysis.
        
        Args:
            db_file: Database file (.cdb, .db)
            rst_file: Results file (.rst, .rth)
            apdl_file: APDL script file
            analysis_type: Type of analysis
            **kwargs: Additional parameters
            
        Returns:
            MathSchema with extracted mathematical structures
        """
        self._current_files = {
            'db': db_file,
            'rst': rst_file,
            'apdl': apdl_file,
        }
        
        # Parse input files
        model_data = {}
        result_data = {}
        apdl_data = {}
        
        if db_file and os.path.exists(db_file):
            model_data = self.cdb_parser.parse(db_file)
        
        if rst_file and os.path.exists(rst_file):
            result_data = self.rst_parser.parse(rst_file)
        
        if apdl_file and os.path.exists(apdl_file):
            apdl_data = self.apdl_parser.parse(apdl_file)
        
        # Build schema
        schema = MathSchema(
            schema_version="1.0.0",
            engine=self.ENGINE_NAME,
            engine_version=self.ENGINE_VERSION,
        )
        
        # Add governing equations based on analysis type
        self._add_governing_equations(schema, analysis_type, model_data)
        
        # Add boundary conditions
        self._add_boundary_conditions(schema, model_data, apdl_data)
        
        # Add mathematical objects (tensors)
        self._add_mathematical_objects(schema, model_data, result_data)
        
        # Add numerical method (FEM)
        self._add_numerical_method(schema, model_data)
        
        # Add computational graph
        self._add_computational_graph(schema, analysis_type)
        
        return schema
    
    def _add_governing_equations(
        self,
        schema: MathSchema,
        analysis_type: str,
        model_data: Dict[str, Any],
    ):
        """Add governing equations based on analysis type."""
        
        if analysis_type in ['static_structural', 'transient_structural']:
            # Linear momentum balance
            momentum_eq = GoverningEquation(
                id="linear_momentum",
                type="pde",
                name="Linear Momentum Balance",
                mathematical_form="∇·σ + ρb = ρü",
                description="Cauchy momentum equation",
                variables=[
                    {"name": "σ", "description": "Cauchy stress tensor", "type": "tensor", "rank": 2},
                    {"name": "ρ", "description": "Density", "type": "scalar"},
                    {"name": "b", "description": "Body force", "type": "vector"},
                    {"name": "u", "description": "Displacement", "type": "vector"},
                ],
            )
            schema.add_governing_equation(momentum_eq)
            
            # Constitutive relation
            constitutive_eq = GoverningEquation(
                id="constitutive_elastic",
                type="constitutive",
                name="Linear Elastic Constitutive Relation",
                mathematical_form="σ = C : ε",
                description="Hooke's law in tensor form",
                variables=[
                    {"name": "C", "description": "Elasticity tensor", "type": "tensor", "rank": 4},
                    {"name": "ε", "description": "Strain tensor", "type": "tensor", "rank": 2},
                ],
            )
            schema.add_governing_equation(constitutive_eq)
            
            # Strain-displacement
            strain_eq = GoverningEquation(
                id="strain_displacement",
                type="kinematic",
                name="Strain-Displacement Relation",
                mathematical_form="ε = ½(∇u + ∇uᵀ)",
                description="Infinitesimal strain tensor",
                variables=[
                    {"name": "ε", "description": "Strain tensor", "type": "tensor", "rank": 2},
                    {"name": "u", "description": "Displacement", "type": "vector"},
                ],
            )
            schema.add_governing_equation(strain_eq)
            
        elif analysis_type in ['thermal_steady', 'thermal_transient']:
            # Heat equation
            heat_eq = GoverningEquation(
                id="heat_equation",
                type="pde",
                name="Heat Conduction Equation",
                mathematical_form="ρcₚ ∂T/∂t = ∇·(k∇T) + Q",
                description="Transient heat conduction",
                variables=[
                    {"name": "T", "description": "Temperature", "type": "scalar"},
                    {"name": "k", "description": "Thermal conductivity", "type": "tensor", "rank": 2},
                    {"name": "Q", "description": "Heat generation", "type": "scalar"},
                ],
            )
            schema.add_governing_equation(heat_eq)
            
        elif analysis_type == 'modal':
            # Eigenvalue problem
            eigen_eq = GoverningEquation(
                id="modal_equation",
                type="eigenvalue_problem",
                name="Modal Analysis Eigenvalue Problem",
                mathematical_form="(K - ω²M)φ = 0",
                description="Free vibration eigenvalue problem",
                variables=[
                    {"name": "K", "description": "Stiffness matrix", "type": "matrix"},
                    {"name": "M", "description": "Mass matrix", "type": "matrix"},
                    {"name": "ω", "description": "Natural frequency", "type": "scalar"},
                    {"name": "φ", "description": "Mode shape", "type": "vector"},
                ],
            )
            schema.add_governing_equation(eigen_eq)
    
    def _add_boundary_conditions(
        self,
        schema: MathSchema,
        model_data: Dict[str, Any],
        apdl_data: Dict[str, Any],
    ):
        """Add boundary conditions from model and APDL."""
        
        # Extract BCs from model data
        constraints = model_data.get('constraints', [])
        loads = model_data.get('loads', [])
        
        for constraint in constraints:
            bc = BoundaryCondition(
                id=f"bc_{constraint.get('id', 'unknown')}",
                type=constraint.get('type', 'dirichlet'),
                region=constraint.get('region', ''),
                mathematical_form=constraint.get('equation', 'u = 0'),
                variables=[
                    {"name": "u", "description": "Displacement", "type": "vector"},
                ],
                physical_meaning=constraint.get('description', 'Constraint'),
            )
            schema.add_boundary_condition(bc)
        
        for load in loads:
            bc = BoundaryCondition(
                id=f"load_{load.get('id', 'unknown')}",
                type='neumann',
                region=load.get('region', ''),
                mathematical_form=load.get('equation', 'σ·n = t'),
                variables=[
                    {"name": "t", "description": "Traction", "type": "vector"},
                    {"name": "σ", "description": "Stress", "type": "tensor"},
                ],
                physical_meaning=load.get('description', 'Applied load'),
            )
            schema.add_boundary_condition(bc)
    
    def _add_mathematical_objects(
        self,
        schema: MathSchema,
        model_data: Dict[str, Any],
        result_data: Dict[str, Any],
    ):
        """Add mathematical objects (tensors, matrices)."""
        
        # Get element info
        element_type = model_data.get('element_type', 'solid185')
        element_info = self.ELEMENT_TYPES.get(element_type, {'nodes': 8, 'order': 'linear'})
        
        num_nodes = model_data.get('num_nodes', 0)
        num_elements = model_data.get('num_elements', 0)
        num_dofs = num_nodes * 3  # Assuming 3 DOF per node
        
        # Stiffness matrix
        if num_dofs > 0:
            K = MathematicalObject(
                id="stiffness_matrix",
                name="Global Stiffness Matrix",
                type="sparse_matrix",
                symbol="K",
                shape=(num_dofs, num_dofs),
                tensor_rank=2,
                properties={"symmetric": True, "positive_definite": True},
            )
            schema.add_mathematical_object(K)
            
            # Mass matrix (for dynamic analyses)
            M = MathematicalObject(
                id="mass_matrix",
                name="Global Mass Matrix",
                type="sparse_matrix",
                symbol="M",
                shape=(num_dofs, num_dofs),
                tensor_rank=2,
                properties={"symmetric": True, "positive_definite": True},
            )
            schema.add_mathematical_object(M)
        
        # Stress tensor field
        stress_field = MathematicalObject(
            id="stress_field",
            name="Cauchy Stress Tensor Field",
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
        schema.add_mathematical_object(stress_field)
        
        # Strain tensor field
        strain_field = MathematicalObject(
            id="strain_field",
            name="Infinitesimal Strain Tensor Field",
            type="tensor_field",
            symbol="ε",
            tensor_rank=2,
            components=[
                TensorComponent(name="ε_xx", symbol="ε_xx", index=[0, 0]),
                TensorComponent(name="ε_yy", symbol="ε_yy", index=[1, 1]),
                TensorComponent(name="ε_zz", symbol="ε_zz", index=[2, 2]),
                TensorComponent(name="ε_xy", symbol="ε_xy", index=[0, 1]),
                TensorComponent(name="ε_yz", symbol="ε_yz", index=[1, 2]),
                TensorComponent(name="ε_xz", symbol="ε_xz", index=[0, 2]),
            ],
        )
        schema.add_mathematical_object(strain_field)
        
        # Elasticity tensor (for anisotropic materials)
        elasticity_tensor = MathematicalObject(
            id="elasticity_tensor",
            name="Elasticity Tensor",
            type="tensor",
            symbol="C",
            tensor_rank=4,
            properties={"major_symmetry": True, "minor_symmetry": True},
        )
        schema.add_mathematical_object(elasticity_tensor)
    
    def _add_numerical_method(self, schema: MathSchema, model_data: Dict[str, Any]):
        """Add FEM numerical method."""
        
        element_type = model_data.get('element_type', 'solid185')
        element_info = self.ELEMENT_TYPES.get(element_type, {})
        
        method = NumericalMethod(
            id="fem",
            name="Finite Element Method",
            description=f"FEM with {element_type} elements",
            parameters={
                "element_type": element_type,
                "element_order": element_info.get('order', 'linear'),
                "num_nodes_per_element": element_info.get('nodes', 8),
                "formulation": "displacement",
                "integration": "gaussian_quadrature",
            },
        )
        
        # Discretization
        discretization = DiscretizationScheme(
            spatial_order=2 if element_info.get('order') == 'quadratic' else 1,
            temporal_order=0,
            mesh_type="unstructured",
        )
        method.discretization = discretization
        
        schema.add_numerical_method(method)
    
    def _add_computational_graph(self, schema: MathSchema, analysis_type: str):
        """Add computational graph for FEA workflow."""
        
        graph = ComputationalGraph(
            id="ansys_analysis",
            name="Ansys FEA Workflow",
            description=f"Workflow for {self.ANALYSIS_TYPES.get(analysis_type, analysis_type)}",
        )
        
        # Common FEA steps
        nodes = [
            ("preprocess", "Preprocessing"),
            ("mesh", "Mesh Generation"),
            ("assemble", "Matrix Assembly"),
            ("apply_bc", "Apply Boundary Conditions"),
            ("solve", "Solve System"),
            ("postprocess", "Postprocessing"),
        ]
        
        for node_id, node_name in nodes:
            graph.add_node(node_id, node_name)
        
        # Add edges
        edges = [
            ("preprocess", "mesh"),
            ("mesh", "assemble"),
            ("assemble", "apply_bc"),
            ("apply_bc", "solve"),
            ("solve", "postprocess"),
        ]
        
        for from_node, to_node in edges:
            graph.add_edge(from_node, to_node)
        
        schema.computational_graphs.append(graph)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return harness capabilities."""
        return {
            'engine_name': self.ENGINE_NAME,
            'engine_version': self.ENGINE_VERSION,
            'supported_formats': self.SUPPORTED_EXTENSIONS,
            'analysis_types': list(self.ANALYSIS_TYPES.keys()),
            'element_types': list(self.ELEMENT_TYPES.keys()),
            'features': [
                'structural_analysis',
                'thermal_analysis',
                'modal_analysis',
                'transient_analysis',
                'nonlinear_materials',
                'contact',
            ],
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        # At least one input file required
        if not any(input_data.get(f) for f in ['db_file', 'rst_file', 'apdl_file']):
            return False
        
        # Check file existence
        for key in ['db_file', 'rst_file', 'apdl_file']:
            filepath = input_data.get(key)
            if filepath and not os.path.exists(filepath):
                return False
        
        return True


# Register harness
HarnessRegistry.register(AnsysHarness)
