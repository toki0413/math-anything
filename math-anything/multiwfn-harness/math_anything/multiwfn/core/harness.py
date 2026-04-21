"""Multiwfn Harness implementation.

Extracts mathematical structures from Multiwfn wavefunction analysis.
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

from .extractor import MultiwfnExtractor
from .parser import CubeFileParser, MultiwfnInputParser, WfnFileParser


class MultiwfnHarness(Harness):
    """Harness for Multiwfn wavefunction analysis.

    Multiwfn is a powerful wavefunction analysis program for quantum chemistry.
    It analyzes electron density, electrostatic potential, molecular orbitals,
    and various topology properties.

    Mathematical focus:
    - Quantum mechanical operators (Hamiltonian, density)
    - Real-space density functional theory quantities
    - Topological analysis (QTAIM, ELF, etc.)
    - Real-space orbital analysis

    Example:
        ```python
        harness = MultiwfnHarness()
        schema = harness.extract_math(
            input_file="input.wfn",
            analysis_type="density",
            grid_spacing=0.1,
        )
        ```
    """

    ENGINE_NAME = "multiwfn"
    ENGINE_VERSION = "3.8"
    SUPPORTED_EXTENSIONS = [".wfn", ".wfx", ".fch", ".fchk", ".cube", ".cub"]

    # Analysis types supported by Multiwfn
    ANALYSIS_TYPES = {
        "topology": "QTAIM topological analysis",
        "elf": "Electron localization function",
        "density": "Electron density analysis",
        "esp": "Electrostatic potential",
        "mo": "Molecular orbital analysis",
        "lol": "Localized orbital locator",
        "rdg": "Reduced density gradient",
        "igm": "Independent gradient model",
    }

    def __init__(self):
        self.extractor = MultiwfnExtractor()
        self.input_parser = MultiwfnInputParser()
        self.wfn_parser = WfnFileParser()
        self.cube_parser = CubeFileParser()
        self._current_files: Dict[str, str] = {}

    def extract_math(
        self,
        input_file: str,
        analysis_type: str = "density",
        grid_spacing: float = 0.1,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> MathSchema:
        """Extract mathematical schema from Multiwfn analysis.

        Args:
            input_file: Path to input file (.wfn, .wfx, .fchk, .cube)
            analysis_type: Type of analysis to perform
            grid_spacing: Grid spacing for real-space analysis (Bohr)
            output_dir: Directory for output files
            **kwargs: Additional parameters

        Returns:
            MathSchema with extracted mathematical structures
        """
        self._current_files = {
            "input": input_file,
            "output_dir": output_dir or ".",
        }

        # Parse input file
        file_ext = Path(input_file).suffix.lower()
        if file_ext in [".wfn", ".wfx"]:
            wfn_data = self.wfn_parser.parse(input_file)
        elif file_ext in [".fch", ".fchk"]:
            wfn_data = self._parse_fchk(input_file)
        elif file_ext in [".cube", ".cub"]:
            wfn_data = self.cube_parser.parse(input_file)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Build schema
        schema = MathSchema(
            schema_version="1.0.0",
            engine=self.ENGINE_NAME,
            engine_version=self.ENGINE_VERSION,
        )

        # Add governing equations
        self._add_quantum_equations(schema, wfn_data, analysis_type)

        # Add mathematical objects (density, orbitals, etc.)
        self._add_quantum_objects(schema, wfn_data, analysis_type)

        # Add numerical method for grid integration
        self._add_numerical_method(schema, grid_spacing)

        # Add computational graph
        self._add_computational_graph(schema, analysis_type)

        return schema

    def _add_quantum_equations(
        self,
        schema: MathSchema,
        wfn_data: Dict[str, Any],
        analysis_type: str,
    ):
        """Add quantum mechanical governing equations."""

        # Electron density equation
        density_eq = GoverningEquation(
            id="electron_density",
            type="real_space_functional",
            name="Electron Density",
            mathematical_form="ρ(r) = Σᵢ nᵢ |φᵢ(r)|²",
            description="Electron density from occupied molecular orbitals",
            variables=[
                {
                    "name": "ρ",
                    "description": "Electron density",
                    "type": "scalar_field",
                },
                {
                    "name": "φᵢ",
                    "description": "Molecular orbital i",
                    "type": "wavefunction",
                },
                {"name": "nᵢ", "description": "Occupation number", "type": "scalar"},
            ],
        )
        schema.add_governing_equation(density_eq)

        # Kohn-Sham effective potential (if DFT)
        if wfn_data.get("method", "").upper() in ["DFT", "B3LYP", "PBE"]:
            ks_eq = GoverningEquation(
                id="kohn_sham_potential",
                type="effective_potential",
                name="Kohn-Sham Effective Potential",
                mathematical_form="V_eff[ρ](r) = V_ext(r) + V_H[ρ](r) + V_xc[ρ](r)",
                description="Effective potential in Kohn-Sham DFT",
                variables=[
                    {
                        "name": "V_eff",
                        "description": "Effective potential",
                        "type": "scalar_field",
                    },
                    {
                        "name": "V_ext",
                        "description": "External potential",
                        "type": "scalar_field",
                    },
                    {
                        "name": "V_H",
                        "description": "Hartree potential",
                        "type": "scalar_field",
                    },
                    {
                        "name": "V_xc",
                        "description": "Exchange-correlation potential",
                        "type": "scalar_field",
                    },
                ],
            )
            schema.add_governing_equation(ks_eq)

        # Analysis-specific equations
        if analysis_type == "elf":
            elf_eq = GoverningEquation(
                id="elf",
                type="localization_function",
                name="Electron Localization Function",
                mathematical_form="ELF(r) = [1 + (D_σ/D_σ^0)²]⁻¹",
                description="Becke and Edgecombe ELF",
                variables=[
                    {
                        "name": "ELF",
                        "description": "Electron localization function",
                        "type": "scalar_field",
                    },
                    {
                        "name": "D_σ",
                        "description": "Curvature of Fermi hole",
                        "type": "scalar_field",
                    },
                    {
                        "name": "D_σ^0",
                        "description": "Thomas-Fermi reference",
                        "type": "scalar_field",
                    },
                ],
            )
            schema.add_governing_equation(elf_eq)

        elif analysis_type == "esp":
            esp_eq = GoverningEquation(
                id="electrostatic_potential",
                type="potential_field",
                name="Electrostatic Potential",
                mathematical_form="Φ(r) = Σ_A Z_A/|r-R_A| - ∫ ρ(r')/|r-r'| dr'",
                description="Molecular electrostatic potential",
                variables=[
                    {
                        "name": "Φ",
                        "description": "Electrostatic potential",
                        "type": "scalar_field",
                    },
                    {"name": "Z_A", "description": "Nuclear charge", "type": "scalar"},
                    {
                        "name": "R_A",
                        "description": "Nuclear position",
                        "type": "vector",
                    },
                ],
            )
            schema.add_governing_equation(esp_eq)

    def _add_quantum_objects(
        self,
        schema: MathSchema,
        wfn_data: Dict[str, Any],
        analysis_type: str,
    ):
        """Add quantum mechanical mathematical objects."""

        # Number of basis functions
        n_basis = wfn_data.get("num_basis_functions", 0)
        n_mos = wfn_data.get("num_molecular_orbitals", 0)

        # Density matrix
        density_matrix = MathematicalObject(
            id="density_matrix",
            name="One-electron Density Matrix",
            type="matrix",
            symbol="P",
            shape=(n_basis, n_basis),
            tensor_rank=2,
            description="Density matrix in atomic orbital basis",
        )
        schema.add_mathematical_object(density_matrix)

        # Molecular orbitals
        mo_coefficients = MathematicalObject(
            id="mo_coefficients",
            name="Molecular Orbital Coefficients",
            type="matrix",
            symbol="C",
            shape=(n_basis, n_mos),
            tensor_rank=2,
            description="MO coefficients in AO basis",
        )
        schema.add_mathematical_object(mo_coefficients)

        # Grid-based scalar fields
        if analysis_type in ["density", "elf", "esp", "lol", "rdg"]:
            scalar_field = MathematicalObject(
                id=f"{analysis_type}_field",
                name=f"{self.ANALYSIS_TYPES.get(analysis_type, analysis_type)} Field",
                type="scalar_field",
                symbol="f(r)",
                tensor_rank=0,
                description=f"3D grid of {analysis_type} values",
            )
            schema.add_mathematical_object(scalar_field)

        # Critical points for topology analysis
        if analysis_type == "topology":
            critical_points = MathematicalObject(
                id="critical_points",
                name="Critical Points",
                type="point_set",
                symbol="{r_c}",
                tensor_rank=0,
                description="(3,-3), (3,-1), (3,+1), (3,+3) critical points",
            )
            schema.add_mathematical_object(critical_points)

            # Gradient vector field
            gradient_field = MathematicalObject(
                id="density_gradient",
                name="Density Gradient Field",
                type="vector_field",
                symbol="∇ρ(r)",
                tensor_rank=1,
                description="Gradient of electron density",
            )
            schema.add_mathematical_object(gradient_field)

            # Hessian field
            hessian_field = MathematicalObject(
                id="density_hessian",
                name="Density Hessian Field",
                type="tensor_field",
                symbol="∇²ρ(r)",
                tensor_rank=2,
                description="Hessian matrix of electron density",
            )
            schema.add_mathematical_object(hessian_field)

    def _add_numerical_method(self, schema: MathSchema, grid_spacing: float):
        """Add numerical method for grid integration."""

        method = NumericalMethod(
            id="grid_integration",
            name="Real-space Grid Integration",
            description="Numerical integration on 3D Cartesian grid",
            parameters={
                "grid_spacing": grid_spacing,
                "grid_type": "even_spacing",
                "integration_scheme": "trapezoidal",
            },
        )

        # Discretization
        discretization = DiscretizationScheme(
            spatial_order=1,
            temporal_order=0,
        )
        method.discretization = discretization

        schema.add_numerical_method(method)

    def _add_computational_graph(self, schema: MathSchema, analysis_type: str):
        """Add computational graph for the analysis workflow."""

        graph = ComputationalGraph(
            id="multiwfn_analysis",
            name="Multiwfn Analysis Pipeline",
            description=f"Workflow for {self.ANALYSIS_TYPES.get(analysis_type, analysis_type)}",
        )

        # Add nodes
        graph.add_node("read_wfn", "Read Wavefunction")
        graph.add_node("build_grid", "Build 3D Grid")
        graph.add_node("compute_ao", "Compute AO Values")
        graph.add_node("compute_mo", "Compute MO Values")
        graph.add_node("compute_density", "Compute Density")

        if analysis_type == "topology":
            graph.add_node("find_critical_points", "Find Critical Points")
            graph.add_node("classify_cp", "Classify Critical Points")
            graph.add_node("compute_gradient_paths", "Compute Gradient Paths")

        # Add edges
        graph.add_edge("read_wfn", "build_grid")
        graph.add_edge("build_grid", "compute_ao")
        graph.add_edge("compute_ao", "compute_mo")
        graph.add_edge("compute_mo", "compute_density")

        if analysis_type == "topology":
            graph.add_edge("compute_density", "find_critical_points")
            graph.add_edge("find_critical_points", "classify_cp")
            graph.add_edge("classify_cp", "compute_gradient_paths")

        schema.computational_graphs.append(graph)

    def _parse_fchk(self, filepath: str) -> Dict[str, Any]:
        """Parse Gaussian formatted checkpoint file."""
        data = {
            "num_atoms": 0,
            "num_basis_functions": 0,
            "num_molecular_orbitals": 0,
            "method": "unknown",
            "basis_set": "unknown",
        }

        with open(filepath, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "Number of atoms" in line:
                data["num_atoms"] = int(line.split()[-1])
            elif "Number of basis functions" in line:
                data["num_basis_functions"] = int(line.split()[-1])
            elif "Number of independent functions" in line:
                data["num_molecular_orbitals"] = int(line.split()[-1])
            elif "Method" in line:
                data["method"] = line.split()[-1]
            elif "Basis set" in line:
                data["basis_set"] = line.split()[-1]

        return data

    def get_capabilities(self) -> Dict[str, Any]:
        """Return harness capabilities."""
        return {
            "engine_name": self.ENGINE_NAME,
            "engine_version": self.ENGINE_VERSION,
            "supported_formats": self.SUPPORTED_EXTENSIONS,
            "analysis_types": list(self.ANALYSIS_TYPES.keys()),
            "supported_analyses": list(self.ANALYSIS_TYPES.values()),
            "features": [
                "wavefunction_analysis",
                "electron_density",
                "electrostatic_potential",
                "topology_analysis",
                "molecular_orbitals",
                "grid_integration",
            ],
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        required = ["input_file"]
        for key in required:
            if key not in input_data:
                return False

        # Check file exists
        if not os.path.exists(input_data["input_file"]):
            return False

        # Check file format
        ext = Path(input_data["input_file"]).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return False

        return True


# Register harness
HarnessRegistry.register(MultiwfnHarness)
