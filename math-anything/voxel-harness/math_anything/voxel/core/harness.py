"""Voxel Harness implementation.

Extracts mathematical structures from voxel-based simulations.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add core to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent.parent / "core"))

import numpy as np
from math_anything.core.harness import Harness
from math_anything.schemas.math_schema import (BoundaryCondition,
                                               ComputationalGraph,
                                               DiscretizationScheme,
                                               GoverningEquation,
                                               MathematicalObject, MathSchema,
                                               NumericalMethod,
                                               TensorComponent)
from math_anything.schemas.registry import HarnessRegistry


class VoxelHarness(Harness):
    """Harness for voxel-based simulations.

    Extracts mathematical structures from voxel grids used in:
    - Lattice Boltzmann Method (LBM)
    - Finite Difference Time Domain (FDTD)
    - Finite Volume Method on Cartesian grids
    - Cellular Automata simulations

    Mathematical focus:
    - Voxel grid as discretization domain
    - Scale mapping between index and physical space
    - Boundary condition numerical implementations
    - Interpolation rules for continuous reconstruction

    Example:
        ```python
        harness = VoxelHarness()

        # Load voxel data
        voxel_data = np.load('simulation.npy')
        flag_field = np.load('flags.npy')

        schema = harness.extract_math(
            voxel_file='simulation.npy',
            flag_file='flags.npy',
            physical_origin=(0.0, 0.0, 0.0),
            voxel_size=0.001,
            simulation_type='lattice_boltzmann',
        )
        ```
    """

    ENGINE_NAME = "voxel"
    ENGINE_VERSION = "1.0.0"
    SUPPORTED_EXTENSIONS = [".npy", ".raw", ".vdb", ".voxel", ".bin"]

    # Supported simulation types
    SIMULATION_TYPES = {
        "lattice_boltzmann": "Lattice Boltzmann Method",
        "fdtd": "Finite-Difference Time-Domain",
        "fvm_cartesian": "Finite Volume on Cartesian Grid",
        "cellular_automata": "Cellular Automata",
        "phase_field": "Phase Field Method",
        "generic": "Generic Voxel Simulation",
    }

    def __init__(self):
        self._current_files: Dict[str, str] = {}

    def extract_math(
        self,
        voxel_file: str,
        flag_file: Optional[str] = None,
        physical_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        voxel_size: Union[float, Tuple[float, float, float]] = 1.0,
        simulation_type: str = "generic",
        **kwargs,
    ) -> MathSchema:
        """Extract mathematical schema from voxel data.

        Args:
            voxel_file: Path to voxel data file (.npy, .raw)
            flag_file: Optional boundary flag file
            physical_origin: Physical coordinates of voxel (0,0,0)
            voxel_size: Physical size of voxels
            simulation_type: Type of voxel simulation
            **kwargs: Additional parameters

        Returns:
            MathSchema with extracted mathematical structures
        """
        self._current_files = {
            "voxel": voxel_file,
            "flag": flag_file,
        }

        # Load voxel data
        voxel_data = self._load_voxel_data(voxel_file)
        flag_data = self._load_voxel_data(flag_file) if flag_file else None

        # Build schema
        schema = MathSchema(
            schema_version="1.0.0",
            engine=self.ENGINE_NAME,
            engine_version=self.ENGINE_VERSION,
        )

        # Add governing equations based on simulation type
        self._add_governing_equations(schema, simulation_type, voxel_data)

        # Add scale mapping as boundary condition (domain definition)
        self._add_scale_mapping_bc(
            schema, physical_origin, voxel_size, voxel_data.shape
        )

        # Add numerical method (voxel discretization)
        self._add_numerical_method(schema, simulation_type, voxel_data)

        # Add voxel-specific boundary conditions
        if flag_data is not None:
            self._add_voxel_boundary_conditions(schema, flag_data, simulation_type)

        # Add mathematical objects (fields)
        self._add_mathematical_objects(schema, voxel_data, simulation_type)

        # Add computational graph
        self._add_computational_graph(schema, simulation_type)

        return schema

    def _load_voxel_data(self, filepath: str) -> np.ndarray:
        """Load voxel data from file."""
        ext = Path(filepath).suffix.lower()

        if ext == ".npy":
            return np.load(filepath)
        elif ext in [".raw", ".bin"]:
            # Assume raw binary, need dimensions from kwargs
            return np.fromfile(filepath, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported voxel format: {ext}")

    def _add_governing_equations(
        self,
        schema: MathSchema,
        simulation_type: str,
        voxel_data: np.ndarray,
    ):
        """Add governing equations based on simulation type."""

        if simulation_type == "lattice_boltzmann":
            # Discrete velocity Boltzmann equation
            boltzmann = GoverningEquation(
                id="discrete_boltzmann",
                type="kinetic_equation",
                name="Discrete Velocity Boltzmann Equation",
                mathematical_form="f_i(x + c_iΔt, t + Δt) - f_i(x, t) = Ω_i(f)",
                description="Lattice Boltzmann evolution equation",
                variables=[
                    {
                        "name": "f_i",
                        "description": "Population in direction i",
                        "type": "scalar_field",
                    },
                    {
                        "name": "c_i",
                        "description": "Discrete velocity",
                        "type": "vector",
                    },
                    {
                        "name": "Ω_i",
                        "description": "Collision operator",
                        "type": "functional",
                    },
                ],
            )
            schema.add_governing_equation(boltzmann)

            # BGK collision
            bgk = GoverningEquation(
                id="bgk_collision",
                type="collision_operator",
                name="BGK Collision",
                mathematical_form="Ω_i = -(1/τ)(f_i - f_i^eq)",
                description="Bhatnagar-Gross-Krook relaxation",
                variables=[
                    {"name": "τ", "description": "Relaxation time", "type": "scalar"},
                    {
                        "name": "f_i^eq",
                        "description": "Equilibrium distribution",
                        "type": "scalar_field",
                    },
                ],
            )
            schema.add_governing_equation(bgk)

            # Equilibrium distribution
            equilibrium = GoverningEquation(
                id="equilibrium_distribution",
                type="equilibrium_function",
                name="Lattice Equilibrium Distribution",
                mathematical_form="f_i^eq = w_i ρ [1 + (c_i·u)/c_s² + (c_i·u)²/(2c_s⁴) - u²/(2c_s²)]",
                description="Second-order expansion of Maxwell-Boltzmann",
                variables=[
                    {"name": "w_i", "description": "Lattice weight", "type": "scalar"},
                    {"name": "ρ", "description": "Density", "type": "scalar_field"},
                    {"name": "u", "description": "Velocity", "type": "vector_field"},
                    {"name": "c_s", "description": "Speed of sound", "type": "scalar"},
                ],
            )
            schema.add_governing_equation(equilibrium)

        elif simulation_type == "fdtd":
            # Maxwell's equations
            maxwell = GoverningEquation(
                id="maxwell_equations",
                type="pde",
                name="Maxwell's Equations (Yee Grid)",
                mathematical_form="∂E/∂t = (1/ε)∇×H - (σ/ε)E, ∂H/∂t = -(1/μ)∇×E",
                description="Electromagnetic field evolution",
                variables=[
                    {
                        "name": "E",
                        "description": "Electric field",
                        "type": "vector_field",
                    },
                    {
                        "name": "H",
                        "description": "Magnetic field",
                        "type": "vector_field",
                    },
                    {
                        "name": "ε",
                        "description": "Permittivity",
                        "type": "tensor_field",
                    },
                    {
                        "name": "μ",
                        "description": "Permeability",
                        "type": "tensor_field",
                    },
                ],
            )
            schema.add_governing_equation(maxwell)

        elif simulation_type == "fvm_cartesian":
            conservation = GoverningEquation(
                id="conservation_law",
                type="pde",
                name="Conservation Law",
                mathematical_form="∂U/∂t + ∇·F(U) = S",
                description="Integral conservation form",
                variables=[
                    {
                        "name": "U",
                        "description": "Conserved quantity",
                        "type": "vector_field",
                    },
                    {"name": "F", "description": "Flux tensor", "type": "tensor_field"},
                    {"name": "S", "description": "Source term", "type": "vector_field"},
                ],
            )
            schema.add_governing_equation(conservation)

        # Add generic discretized field equation
        discretized_field = GoverningEquation(
            id="discretized_field",
            type="discrete_field_equation",
            name="Voxel Field Representation",
            mathematical_form="f(r) ≈ Σ_{i,j,k} f_{i,j,k} χ_{i,j,k}(r)",
            description="Field represented on voxel grid with characteristic functions",
            variables=[
                {
                    "name": "f_{i,j,k}",
                    "description": "Field value at voxel (i,j,k)",
                    "type": "scalar",
                },
                {
                    "name": "χ_{i,j,k}",
                    "description": "Voxel characteristic function",
                    "type": "function",
                },
            ],
        )
        schema.add_governing_equation(discretized_field)

    def _add_scale_mapping_bc(
        self,
        schema: MathSchema,
        origin: Tuple[float, float, float],
        voxel_size: Union[float, Tuple[float, float, float]],
        shape: Tuple[int, ...],
    ):
        """Add scale mapping as domain definition boundary condition."""

        if isinstance(voxel_size, (int, float)):
            voxel_size = (voxel_size, voxel_size, voxel_size)

        physical_dims = (
            shape[0] * voxel_size[0],
            shape[1] * voxel_size[1],
            shape[2] * voxel_size[2],
        )

        scale_bc = BoundaryCondition(
            id="scale_mapping",
            type="domain_definition",
            region=f"voxel_grid_{shape[:3]}",
            mathematical_form=f"r_phys = ({origin[0]}, {origin[1]}, {origin[2]}) + (i,j,k) * ({voxel_size[0]}, {voxel_size[1]}, {voxel_size[2]})",
            variables=[
                {
                    "name": "r_phys",
                    "description": "Physical coordinate",
                    "type": "vector",
                },
                {
                    "name": "(i,j,k)",
                    "description": "Voxel index",
                    "type": "integer_vector",
                },
            ],
            physical_meaning=f"Affine mapping from index space (0..{shape[0]-1}, 0..{shape[1]-1}, 0..{shape[2]-1}) to physical domain {physical_dims}",
        )
        schema.add_boundary_condition(scale_bc)

    def _add_voxel_boundary_conditions(
        self,
        schema: MathSchema,
        flag_data: np.ndarray,
        simulation_type: str,
    ):
        """Add boundary conditions from voxel flags."""

        # Detect domain boundaries
        nx, ny, nz = flag_data.shape

        face_boundaries = [
            ("xmin", flag_data[0, :, :]),
            ("xmax", flag_data[nx - 1, :, :]),
            ("ymin", flag_data[:, 0, :]),
            ("ymax", flag_data[:, ny - 1, :]),
            ("zmin", flag_data[:, :, 0]),
            ("zmax", flag_data[:, :, nz - 1]),
        ]

        for location, face_data in face_boundaries:
            # Detect boundary type from flags
            unique_flags = np.unique(face_data)

            for flag in unique_flags:
                if flag == 0:  # Fluid/internal
                    continue

                bc = self._create_boundary_condition(location, flag, simulation_type)
                if bc:
                    schema.add_boundary_condition(bc)

    def _create_boundary_condition(
        self,
        location: str,
        flag: int,
        simulation_type: str,
    ) -> Optional[BoundaryCondition]:
        """Create boundary condition from flag value."""

        flag_types = {
            1: ("wall", "No-slip wall"),
            2: ("inlet", "Velocity inlet"),
            3: ("outlet", "Pressure outlet"),
            4: ("slip", "Slip wall"),
            5: ("symmetry", "Symmetry plane"),
            6: ("periodic", "Periodic boundary"),
            7: ("moving_wall", "Moving wall"),
        }

        bc_type, description = flag_types.get(flag, ("unknown", "Unknown boundary"))

        if simulation_type == "lattice_boltzmann":
            implementations = {
                "wall": ("bounce_back", "f_{opp} = f_i (mid-link bounce-back)"),
                "inlet": ("zou_he_velocity", "Moment-based velocity BC"),
                "outlet": ("zou_he_pressure", "Moment-based pressure BC"),
                "slip": ("specular", "Specular reflection"),
                "moving_wall": (
                    "equilibrium_enhanced",
                    "f_{opp} = f_i - 6w_iρ(c_i·u_w)/c_s²",
                ),
            }
            scheme, math_form = implementations.get(bc_type, ("unknown", "Unknown"))
        else:
            scheme = "standard"
            math_form = f"{bc_type} boundary"

        return BoundaryCondition(
            id=f"bc_{location}_{bc_type}",
            type=bc_type,
            region=location,
            mathematical_form=math_form,
            variables=[{"name": "f_i", "description": "Population", "type": "scalar"}],
            physical_meaning=f"{description} with {scheme} numerical implementation",
        )

    def _add_numerical_method(
        self,
        schema: MathSchema,
        simulation_type: str,
        voxel_data: np.ndarray,
    ):
        """Add voxel discretization numerical method."""

        shape = voxel_data.shape

        method = NumericalMethod(
            id="voxel_discretization",
            name="Voxel-Based Discretization",
            description=f"{self.SIMULATION_TYPES.get(simulation_type, simulation_type)} on Cartesian voxel grid",
            parameters={
                "grid_dimensions": shape[:3],
                "total_voxels": shape[0] * shape[1] * shape[2],
                "field_components": shape[3] if len(shape) > 3 else 1,
                "discretization_type": "cell_centered",
                "stencil": (
                    "nearest_neighbor"
                    if simulation_type == "cellular_automata"
                    else "compact"
                ),
            },
        )

        discretization = DiscretizationScheme(
            spatial_order=1,  # First-order in space for voxel
            temporal_order=(
                1
                if simulation_type in ["lattice_boltzmann", "cellular_automata"]
                else 2
            ),
            mesh_type="cartesian_voxel",
        )
        method.discretization = discretization

        schema.add_numerical_method(method)

    def _add_mathematical_objects(
        self,
        schema: MathSchema,
        voxel_data: np.ndarray,
        simulation_type: str,
    ):
        """Add mathematical objects representing voxel fields."""

        shape = voxel_data.shape
        num_components = shape[3] if len(shape) > 3 else 1

        if num_components == 1:
            # Scalar field
            scalar_field = MathematicalObject(
                id="voxel_scalar_field",
                name="Voxel Scalar Field",
                type="discrete_scalar_field",
                symbol="f_{i,j,k}",
                tensor_rank=0,
                shape=shape[:3],
                description="Scalar quantity defined on voxel centers",
            )
            schema.add_mathematical_object(scalar_field)

        elif num_components == 3:
            # Vector field
            vector_field = MathematicalObject(
                id="voxel_vector_field",
                name="Voxel Vector Field",
                type="discrete_vector_field",
                symbol="u_{i,j,k}",
                tensor_rank=1,
                shape=(*shape[:3], 3),
                description="Vector quantity defined on voxel centers or staggered",
            )
            schema.add_mathematical_object(vector_field)

        elif num_components in [6, 9]:
            # Tensor field (symmetric or full)
            tensor_field = MathematicalObject(
                id="voxel_tensor_field",
                name="Voxel Tensor Field",
                type="discrete_tensor_field",
                symbol="σ_{i,j,k}",
                tensor_rank=2,
                shape=(*shape[:3], num_components),
                description="Tensor quantity in voxel grid",
            )
            schema.add_mathematical_object(tensor_field)

        # Add voxel geometry object
        voxel_geometry = MathematicalObject(
            id="voxel_geometry",
            name="Voxel Grid Geometry",
            type="discretization_domain",
            symbol="Ω_h",
            tensor_rank=0,
            description="Cartesian voxel grid domain with characteristic function representation",
        )
        schema.add_mathematical_object(voxel_geometry)

        # Add interpolation operator
        interpolation_op = MathematicalObject(
            id="voxel_interpolation",
            name="Voxel-to-Continuous Interpolation",
            type="interpolation_operator",
            symbol="I_h",
            tensor_rank=0,
            description="Operator mapping discrete voxel values to continuous field",
        )
        schema.add_mathematical_object(interpolation_op)

    def _add_computational_graph(self, schema: MathSchema, simulation_type: str):
        """Add computational graph for voxel simulation."""

        graph = ComputationalGraph(
            id="voxel_simulation",
            name="Voxel-Based Simulation Workflow",
            description=self.SIMULATION_TYPES.get(simulation_type, simulation_type),
        )

        if simulation_type == "lattice_boltzmann":
            nodes = [
                ("initialize", "Initialize Populations f_i"),
                ("stream", "Streaming (Advection)"),
                ("bc_pre", "Apply Boundary Conditions"),
                ("collide", "Collision (Relaxation)"),
                ("macro", "Compute Macroscopic ρ, u"),
                ("bc_post", "Apply Post-Collision BCs"),
            ]
            edges = [
                ("initialize", "stream"),
                ("stream", "bc_pre"),
                ("bc_pre", "collide"),
                ("collide", "macro"),
                ("macro", "bc_post"),
                ("bc_post", "stream"),
            ]
        else:
            nodes = [
                ("setup", "Setup Grid"),
                ("initialize", "Initialize Fields"),
                ("evolve", "Time Evolution"),
                ("bc", "Apply BCs"),
                ("output", "Extract Results"),
            ]
            edges = [
                ("setup", "initialize"),
                ("initialize", "evolve"),
                ("evolve", "bc"),
                ("bc", "evolve"),
                ("evolve", "output"),
            ]

        for node_id, node_name in nodes:
            graph.add_node(node_id, node_name)

        for from_node, to_node in edges:
            graph.add_edge(from_node, to_node)

        schema.computational_graphs.append(graph)

    def get_capabilities(self) -> Dict[str, Any]:
        """Return harness capabilities."""
        return {
            "engine_name": self.ENGINE_NAME,
            "engine_version": self.ENGINE_VERSION,
            "supported_formats": self.SUPPORTED_EXTENSIONS,
            "simulation_types": list(self.SIMULATION_TYPES.keys()),
            "features": [
                "scale_mapping",
                "boundary_condition_extraction",
                "voxel_to_continuous_interpolation",
                "lbm_specific_analysis",
            ],
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        if "voxel_file" not in input_data:
            return False

        voxel_file = input_data["voxel_file"]
        if not os.path.exists(voxel_file):
            return False

        # Check format
        ext = Path(voxel_file).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return False

        return True


# Register harness
HarnessRegistry.register(VoxelHarness)
