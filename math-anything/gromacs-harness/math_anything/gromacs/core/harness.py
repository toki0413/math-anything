"""GROMACS Harness implementation.

Extracts mathematical structures from GROMACS biomolecular simulations.
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
)
from math_anything.schemas.registry import HarnessRegistry

from .extractor import GromacsExtractor
from .parser import EDTParser, MDPParser, TOPParser, TPRParser


class GromacsHarness(Harness):
    """Harness for GROMACS biomolecular simulations.

    GROMACS is a high-performance molecular dynamics package
    designed for simulations of proteins, lipids, and nucleic acids.

    Mathematical focus:
    - Classical Newtonian dynamics
    - Statistical mechanics ensembles
    - Force field energy functions
    - Periodic boundary conditions
    - Constraint algorithms (LINCS/SHAKE)

    Supported simulations:
    - Energy minimization
    - NVT/NPT equilibration
    - Production MD
    - Free energy calculations
    - Enhanced sampling (umbrella, metadynamics)

    Example:
        ```python
        harness = GromacsHarness()
        schema = harness.extract_math(
            mdp_file="md.mdp",
            top_file="topol.top",
            tpr_file="topol.tpr",
        )
        ```
    """

    ENGINE_NAME = "gromacs"
    ENGINE_VERSION = "2023.3"
    SUPPORTED_EXTENSIONS = [".mdp", ".top", ".tpr", ".gro", ".xtc", ".trr", ".edr"]

    # Simulation types
    SIMULATION_TYPES = {
        "em": "Energy Minimization",
        "md": "Molecular Dynamics",
        "nvt": "NVT Ensemble",
        "npt": "NPT Ensemble",
        "sd": "Steepest Descent",
        "cg": "Conjugate Gradient",
        "l-bfgs": "L-BFGS",
    }

    # Thermostats
    THERMOSTATS = {
        "v-rescale": "Velocity rescaling",
        "berendsen": "Berendsen weak coupling",
        "nose-hoover": "Nose-Hoover chain",
        "andersen": "Andersen stochastic",
    }

    # Barostats
    BAROSTATS = {
        "berendsen": "Berendsen pressure coupling",
        "parrinello-rahman": "Parrinello-Rahman",
        "c-rescale": "C-rescale",
        "mttk": "Martyna-Tuckerman-Tobias-Klein",
    }

    def __init__(self):
        self.extractor = GromacsExtractor()
        self.mdp_parser = MDPParser()
        self.top_parser = TOPParser()
        self.tpr_parser = TPRParser()
        self.edr_parser = EDTParser()
        self._current_files: Dict[str, str] = {}

    def extract_math(
        self,
        mdp_file: Optional[str] = None,
        top_file: Optional[str] = None,
        tpr_file: Optional[str] = None,
        edr_file: Optional[str] = None,
        **kwargs,
    ) -> MathSchema:
        """Extract mathematical schema from GROMACS simulation.

        Args:
            mdp_file: MD parameter file
            top_file: Topology file
            tpr_file: Binary run input file
            edr_file: Energy file
            **kwargs: Additional parameters

        Returns:
            MathSchema with extracted mathematical structures
        """
        self._current_files = {
            "mdp": mdp_file,
            "top": top_file,
            "tpr": tpr_file,
            "edr": edr_file,
        }

        # Parse input files
        mdp_data = {}
        top_data = {}
        tpr_data = {}
        edr_data = {}

        if mdp_file and os.path.exists(mdp_file):
            mdp_data = self.mdp_parser.parse(mdp_file)

        if top_file and os.path.exists(top_file):
            top_data = self.top_parser.parse(top_file)

        if tpr_file and os.path.exists(tpr_file):
            tpr_data = self.tpr_parser.parse(tpr_file)

        if edr_file and os.path.exists(edr_file):
            edr_data = self.edr_parser.parse(edr_file)

        # Build schema
        schema = MathSchema(
            schema_version="1.0.0",
            engine=self.ENGINE_NAME,
            engine_version=self.ENGINE_VERSION,
        )

        # Add governing equations
        self._add_governing_equations(schema, mdp_data, top_data)

        # Add boundary conditions (periodic)
        self._add_boundary_conditions(schema, mdp_data)

        # Add mathematical objects
        self._add_mathematical_objects(schema, top_data, tpr_data)

        # Add numerical method (integrator)
        self._add_numerical_method(schema, mdp_data)

        # Add computational graph
        self._add_computational_graph(schema, mdp_data)

        return schema

    def _add_governing_equations(
        self,
        schema: MathSchema,
        mdp_data: Dict[str, Any],
        top_data: Dict[str, Any],
    ):
        """Add governing equations."""

        # Newton's equations of motion
        newton = GoverningEquation(
            id="newton_equations",
            type="ode_system",
            name="Newton's Equations of Motion",
            mathematical_form="mᵢ d²rᵢ/dt² = Fᵢ = -∇ᵢV(rᴺ)",
            description="Classical molecular dynamics",
            variables=[
                {"name": "rᵢ", "description": "Position of atom i", "type": "vector"},
                {"name": "mᵢ", "description": "Mass of atom i", "type": "scalar"},
                {"name": "Fᵢ", "description": "Force on atom i", "type": "vector"},
                {"name": "V", "description": "Potential energy", "type": "scalar"},
            ],
        )
        schema.add_governing_equation(newton)

        # Force field energy function
        force_field = GoverningEquation(
            id="force_field",
            type="energy_function",
            name="Molecular Mechanics Force Field",
            mathematical_form="V = V_bond + V_angle + V_dihedral + V_nonbond",
            description="Additive force field energy decomposition",
            variables=[
                {"name": "V_bond", "description": "Bond stretching", "type": "scalar"},
                {"name": "V_angle", "description": "Angle bending", "type": "scalar"},
                {"name": "V_dihedral", "description": "Torsional", "type": "scalar"},
                {"name": "V_nonbond", "description": "Non-bonded", "type": "scalar"},
            ],
        )
        schema.add_governing_equation(force_field)

        # Bond potential
        bond = GoverningEquation(
            id="bond_potential",
            type="potential_function",
            name="Harmonic Bond Potential",
            mathematical_form="V_bond = ½k_b(r - r₀)²",
            description="Harmonic bond stretching",
            variables=[
                {"name": "k_b", "description": "Bond force constant", "type": "scalar"},
                {"name": "r", "description": "Bond length", "type": "scalar"},
                {
                    "name": "r₀",
                    "description": "Equilibrium bond length",
                    "type": "scalar",
                },
            ],
        )
        schema.add_governing_equation(bond)

        # Angle potential
        angle = GoverningEquation(
            id="angle_potential",
            type="potential_function",
            name="Harmonic Angle Potential",
            mathematical_form="V_angle = ½k_θ(θ - θ₀)²",
            description="Harmonic angle bending",
            variables=[
                {
                    "name": "k_θ",
                    "description": "Angle force constant",
                    "type": "scalar",
                },
                {"name": "θ", "description": "Bond angle", "type": "scalar"},
                {"name": "θ₀", "description": "Equilibrium angle", "type": "scalar"},
            ],
        )
        schema.add_governing_equation(angle)

        # Lennard-Jones potential
        lj = GoverningEquation(
            id="lj_potential",
            type="potential_function",
            name="Lennard-Jones Potential",
            mathematical_form="V_LJ = 4ε[(σ/r)¹² - (σ/r)⁶]",
            description="Van der Waals interactions",
            variables=[
                {"name": "ε", "description": "Well depth", "type": "scalar"},
                {"name": "σ", "description": "Collision diameter", "type": "scalar"},
                {"name": "r", "description": "Interatomic distance", "type": "scalar"},
            ],
        )
        schema.add_governing_equation(lj)

        # Coulomb potential
        coulomb = GoverningEquation(
            id="coulomb_potential",
            type="potential_function",
            name="Coulomb Potential",
            mathematical_form="V_coul = (1/4πε₀) Σ qᵢqⱼ/rᵢⱼ",
            description="Electrostatic interactions",
            variables=[
                {"name": "qᵢ", "description": "Charge on atom i", "type": "scalar"},
                {"name": "ε₀", "description": "Vacuum permittivity", "type": "scalar"},
            ],
        )
        schema.add_governing_equation(coulomb)

        # Thermostat coupling (if NVT)
        integrator = mdp_data.get("integrator", "md")
        if integrator in ["md", "sd"]:
            tcoupl = mdp_data.get("tcoupl", "no")
            if tcoupl != "no":
                thermostat_eq = GoverningEquation(
                    id="thermostat_coupling",
                    type="coupling_algorithm",
                    name=f"{self.THERMOSTATS.get(tcoupl, tcoupl)} Thermostat",
                    mathematical_form="dT/dt = (T₀ - T)/τ_T",
                    description="Temperature coupling",
                    variables=[
                        {
                            "name": "T",
                            "description": "Instantaneous temperature",
                            "type": "scalar",
                        },
                        {
                            "name": "T₀",
                            "description": "Target temperature",
                            "type": "scalar",
                        },
                        {
                            "name": "τ_T",
                            "description": "Coupling time constant",
                            "type": "scalar",
                        },
                    ],
                )
                schema.add_governing_equation(thermostat_eq)

        # Barostat coupling (if NPT)
        pcoupl = mdp_data.get("pcoupl", "no")
        if pcoupl != "no":
            barostat_eq = GoverningEquation(
                id="barostat_coupling",
                type="coupling_algorithm",
                name=f"{self.BAROSTATS.get(pcoupl, pcoupl)} Barostat",
                mathematical_form="dP/dt = (P₀ - P)/τ_P",
                description="Pressure coupling",
                variables=[
                    {
                        "name": "P",
                        "description": "Instantaneous pressure",
                        "type": "scalar",
                    },
                    {"name": "P₀", "description": "Target pressure", "type": "scalar"},
                    {
                        "name": "τ_P",
                        "description": "Coupling time constant",
                        "type": "scalar",
                    },
                ],
            )
            schema.add_governing_equation(barostat_eq)

    def _add_boundary_conditions(self, schema: MathSchema, mdp_data: Dict[str, Any]):
        """Add boundary conditions (periodic)."""

        pbc = mdp_data.get("pbc", "xyz")

        if pbc != "no":
            pbc_bc = BoundaryCondition(
                id="periodic_boundary",
                type="periodic",
                region=f"pbc={pbc}",
                mathematical_form="rᵢ(t) = rᵢ(t) + nL, n ∈ ℤ³",
                description=f"Periodic boundary conditions in {pbc} directions",
                variables=[
                    {"name": "L", "description": "Box vector", "type": "vector"},
                ],
                physical_meaning="Periodic boundary conditions",
            )
            schema.add_boundary_condition(pbc_bc)

        # Constraints (if any)
        constraints = mdp_data.get("constraints", "none")
        if constraints != "none":
            constraint_bc = BoundaryCondition(
                id="holonomic_constraints",
                type="constraint",
                region="bonds/angles",
                mathematical_form="|rᵢ - rⱼ| = d₀",
                description=f"Holonomic constraints using {constraints}",
                variables=[
                    {
                        "name": "d₀",
                        "description": "Constraint distance",
                        "type": "scalar",
                    },
                ],
                physical_meaning=f"{constraints.upper()} constraints",
            )
            schema.add_boundary_condition(constraint_bc)

    def _add_mathematical_objects(
        self,
        schema: MathSchema,
        top_data: Dict[str, Any],
        tpr_data: Dict[str, Any],
    ):
        """Add mathematical objects."""

        # Number of atoms
        num_atoms = top_data.get("num_atoms", tpr_data.get("num_atoms", 0))

        if num_atoms > 0:
            positions = MathematicalObject(
                id="positions",
                name="Atomic Positions",
                type="vector_array",
                symbol="r",
                shape=(num_atoms, 3),
                description="3D coordinates of all atoms",
            )
            schema.add_mathematical_object(positions)

            velocities = MathematicalObject(
                id="velocities",
                name="Atomic Velocities",
                type="vector_array",
                symbol="v",
                shape=(num_atoms, 3),
                description="Velocities of all atoms",
            )
            schema.add_mathematical_object(velocities)

            forces = MathematicalObject(
                id="forces",
                name="Atomic Forces",
                type="vector_array",
                symbol="F",
                shape=(num_atoms, 3),
                description="Forces on all atoms",
            )
            schema.add_mathematical_object(forces)

        # Box vectors
        box = MathematicalObject(
            id="simulation_box",
            name="Simulation Box Vectors",
            type="matrix",
            symbol="L",
            shape=(3, 3),
            description="Triclinic box vectors",
        )
        schema.add_mathematical_object(box)

        # Energy terms
        energy = MathematicalObject(
            id="total_energy",
            name="Total Energy",
            type="scalar",
            symbol="E",
            description="Hamiltonian of the system",
        )
        schema.add_mathematical_object(energy)

    def _add_numerical_method(self, schema: MathSchema, mdp_data: Dict[str, Any]):
        """Add MD integrator numerical method."""

        integrator = mdp_data.get("integrator", "md")
        dt = mdp_data.get("dt", 0.002)  # ps

        method = NumericalMethod(
            id=f"integrator_{integrator}",
            name=f"GROMACS {integrator.upper()} Integrator",
            description=self.SIMULATION_TYPES.get(integrator, integrator),
            parameters={
                "integrator": integrator,
                "timestep": dt,
                "unit": "ps",
            },
        )

        # Discretization (temporal only for MD)
        discretization = DiscretizationScheme(
            spatial_order=0,  # No spatial discretization in MD
            temporal_order=2,  # Leapfrog is 2nd order
            mesh_type="particle",
        )
        method.discretization = discretization

        schema.add_numerical_method(method)

    def _add_computational_graph(self, schema: MathSchema, mdp_data: Dict[str, Any]):
        """Add computational graph for MD workflow."""

        graph = ComputationalGraph(
            id="gromacs_workflow",
            name="GROMACS MD Workflow",
            description="Molecular dynamics simulation pipeline",
        )

        nodes = [
            ("setup", "System Setup"),
            ("minimization", "Energy Minimization"),
            ("equilibration", "Equilibration"),
            ("production", "Production Run"),
            ("analysis", "Trajectory Analysis"),
        ]

        for node_id, node_name in nodes:
            graph.add_node(node_id, node_name)

        edges = [
            ("setup", "minimization"),
            ("minimization", "equilibration"),
            ("equilibration", "production"),
            ("production", "analysis"),
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
            "integrators": list(self.SIMULATION_TYPES.keys()),
            "thermostats": list(self.THERMOSTATS.keys()),
            "barostats": list(self.BAROSTATS.keys()),
            "features": [
                "parallel_md",
                "gpu_acceleration",
                "free_energy_calculations",
                "enhanced_sampling",
                "replica_exchange",
            ],
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        # At least one input file required
        if not any(input_data.get(f) for f in ["mdp_file", "top_file", "tpr_file"]):
            return False

        for key in ["mdp_file", "top_file", "tpr_file", "edr_file"]:
            filepath = input_data.get(key)
            if filepath and not os.path.exists(filepath):
                return False

        return True


# Register harness
HarnessRegistry.register(GromacsHarness)
