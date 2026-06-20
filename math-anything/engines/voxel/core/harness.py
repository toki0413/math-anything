"""Voxel Harness — mathematical structure extraction from voxel-based simulations.

Extracts structures from:
  - Lattice Boltzmann Method (LBM)
  - Finite Difference Time Domain (FDTD)
  - Finite Volume Method on Cartesian grids
"""

from math_anything.schemas.math_schema import (
    BoundaryCondition,
    ComputationalEdge,
    ComputationalGraph,
    ComputationalNode,
    ConservationProperty,
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
from math_anything.core.harness import MathAnythingHarness, HarnessRegistry


class VoxelHarness(MathAnythingHarness):
    """Harness for voxel-based simulations."""

    ENGINE_NAME = "voxel"
    SUPPORTED_EXTENSIONS = [".npy", ".raw", ".vdb", ".bin"]

    SIMULATION_TYPES = {
        "lattice_boltzmann": "Lattice Boltzmann Method",
        "fdtd": "Finite-Difference Time-Domain",
        "fvm_cartesian": "Finite Volume on Cartesian Grid",
        "generic": "Generic Voxel Simulation",
    }

    @property
    def engine_name(self) -> str:
        return "voxel"

    @property
    def supported_schema_version(self) -> str:
        return "1.0.0"

    def extract(self, files, options=None):
        schema = MathSchema(
            meta=MetaInfo(
                extracted_by="math-anything-voxel",
                extractor_version="1.0.0",
            )
        )

        sim_type = (options or {}).get("simulation_type", "generic")

        # Build mathematical model
        model = MathematicalModel()
        equations = self._build_governing_equations(sim_type)
        for eq in equations:
            model.governing_equations.append(eq)
        schema.mathematical_model = model

        # Numerical method
        schema.numerical_method = NumericalMethod(
            discretization=Discretization(
                space_discretization="voxel_cartesian_grid",
            ),
            solver=Solver(algorithm="explicit_timestep"),
        )

        # Computational graph
        graph = ComputationalGraph(version="1.0")
        nodes = self._build_computational_nodes(sim_type)
        for n in nodes:
            graph.add_node(n)
        schema.computational_graph = graph

        return schema

    def _build_governing_equations(self, sim_type: str) -> list[GoverningEquation]:
        equations = []

        if sim_type == "lattice_boltzmann":
            equations.append(GoverningEquation(
                id="discrete_boltzmann",
                type="kinetic_equation",
                name="Discrete Velocity Boltzmann Equation",
                mathematical_form="f_i(x + c_i*dt, t + dt) - f_i(x, t) = Omega_i(f)",
                variables=["f_i", "c_i", "Omega_i"],
            ))
        elif sim_type == "fdtd":
            equations.append(GoverningEquation(
                id="maxwell",
                type="pde",
                name="Maxwells Equations on Yee Grid",
                mathematical_form="dE/dt = (1/eps) curl H, dH/dt = -(1/mu) curl E",
                variables=["E", "H"],
            ))
        else:
            equations.append(GoverningEquation(
                id="conservation_law",
                type="pde",
                name="Conservation Law on Voxel Grid",
                mathematical_form="dU/dt + div F(U) = S",
                variables=["U", "F", "S"],
            ))

        equations.append(GoverningEquation(
            id="voxel_field_representation",
            type="discrete_field",
            name="Voxel Field Representation",
            mathematical_form="f(r) = sum_{i,j,k} f_{ijk} chi_{ijk}(r)",
            variables=["f_ijk", "chi_ijk"],
        ))

        return equations

    def _build_computational_nodes(self, sim_type: str) -> list[ComputationalNode]:
        nodes = []
        nodes.append(ComputationalNode(
            id="initialize",
            type="setup",
            math_semantics={"description": "Initialize grid and fields"},
        ))
        nodes.append(ComputationalNode(
            id="evolve",
            type="time_evolution",
            math_semantics={
                "description": "Time stepping evolution",
                "updates": {"mode": "explicit_update"},
            },
        ))
        nodes.append(ComputationalNode(
            id="output",
            type="output",
            math_semantics={"description": "Extract results"},
        ))
        return nodes

    def list_extractable_objects(self):
        return ["governing_equations", "numerical_method", "computational_graph"]

    def get_supported_extensions(self):
        return self.SUPPORTED_EXTENSIONS


HarnessRegistry.register(VoxelHarness)
