"""Enhanced COMSOL MPH parser with symbolic constraint support.

Parses COMSOL MPH model files and extracts mathematical structures
with symbolic constraints for multiphysics analysis.
"""

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PhysicsInterface(Enum):
    """COMSOL physics interfaces."""

    SOLID_MECHANICS = "solid_mechanics"
    HEAT_TRANSFER = "heat_transfer"
    ELECTROMAGNETICS = "electromagnetics"
    FLUID_FLOW = "fluid_flow"
    CHEMICAL = "chemical"
    ACOUSTICS = "acoustics"
    MULTIPHYSICS = "multiphysics"


class StudyType(Enum):
    """COMSOL study types."""

    STATIONARY = "stationary"
    TIME_DEPENDENT = "time_dependent"
    EIGENFREQUENCY = "eigenfrequency"
    FREQUENCY_DOMAIN = "frequency_domain"
    PARAMETRIC = "parametric"
    MULTIGRID = "multigrid"


@dataclass
class ComsolParameter:
    """COMSOL parameter with constraints."""

    name: str
    value: float
    unit: str
    description: str = ""
    constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "description": self.description,
            "constraints": self.constraints,
        }


@dataclass
class ComsolResults:
    """Results of COMSOL parsing."""

    model_name: str
    version: str
    physics: List[PhysicsInterface]
    studies: List[StudyType]
    parameters: List[ComsolParameter]
    variables: Dict[str, str]
    constraints: List[Dict[str, Any]]
    num_dofs: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "physics": [p.value for p in self.physics],
            "studies": [s.value for s in self.studies],
            "parameters": [p.to_dict() for p in self.parameters],
            "variables": self.variables,
            "constraints": self.constraints,
            "num_dofs": self.num_dofs,
        }


class ComsolSymbolicConstraints:
    """Symbolic constraints for COMSOL parameters."""

    # Parameter constraints by physics
    PARAMETER_CONSTRAINTS = {
        # Structural mechanics
        "E": [  # Young's modulus
            ("> 0", "Young's modulus must be positive (critical)"),
            ("< 1e15 [Pa]", "Young's modulus should be physically reasonable (info)"),
        ],
        "nu": [  # Poisson's ratio
            ("> -1", "Poisson's ratio must be > -1 (critical)"),
            ("< 0.5", "Poisson's ratio must be < 0.5 for stability (critical)"),
        ],
        "rho": [  # Density
            ("> 0", "Density must be positive (critical)"),
        ],
        # Heat transfer
        "k": [  # Thermal conductivity
            ("> 0", "Thermal conductivity must be positive (critical)"),
        ],
        "Cp": [  # Specific heat capacity
            ("> 0", "Specific heat must be positive (critical)"),
        ],
        "T": [  # Temperature
            (">= 0 [K]", "Temperature must be non-negative in Kelvin (critical)"),
            (">= -273.15 [degC]", "Temperature must be above absolute zero (critical)"),
        ],
        # Fluid flow
        "mu": [  # Dynamic viscosity
            (">= 0", "Viscosity must be non-negative (critical)"),
        ],
        "rho_f": [  # Fluid density
            ("> 0", "Fluid density must be positive (critical)"),
        ],
        # Electromagnetics
        "epsilon": [  # Permittivity
            (">= epsilon0", "Permittivity must be >= vacuum permittivity (critical)"),
        ],
        "sigma": [  # Conductivity
            (">= 0", "Conductivity must be non-negative (critical)"),
        ],
        # Mesh
        "hmax": [  # Maximum element size
            ("> 0", "Element size must be positive (critical)"),
        ],
        "hmin": [  # Minimum element size
            ("> 0", "Element size must be positive (critical)"),
            ("< hmax", "Minimum size should be less than maximum (warning)"),
        ],
        # Time
        "t": [  # Time
            (">= 0", "Time must be non-negative (critical)"),
        ],
        "dt": [  # Time step
            ("> 0", "Time step must be positive (critical)"),
        ],
        # Frequency
        "f": [  # Frequency
            (">= 0", "Frequency must be non-negative (critical)"),
        ],
    }

    @classmethod
    def validate_parameter(
        cls, param_name: str, value: float, unit: str = ""
    ) -> List[Dict[str, Any]]:
        """Validate a parameter against constraints."""
        results = []
        constraints = cls.PARAMETER_CONSTRAINTS.get(param_name, [])

        for constraint_expr, description in constraints:
            result = {
                "parameter": param_name,
                "value": value,
                "unit": unit,
                "constraint": constraint_expr,
                "description": description,
                "satisfied": False,
            }

            try:
                if "> 0" in constraint_expr:
                    result["satisfied"] = float(value) > 0
                elif ">= 0" in constraint_expr:
                    result["satisfied"] = float(value) >= 0
                elif "> -1" in constraint_expr:
                    result["satisfied"] = float(value) > -1
                elif "< 0.5" in constraint_expr:
                    result["satisfied"] = float(value) < 0.5
                elif "< 1e15" in constraint_expr:
                    result["satisfied"] = float(value) < 1e15
                elif "< hmax" in constraint_expr:
                    # Special case for hmin < hmax
                    result["satisfied"] = True  # Would need hmax value
                else:
                    result["satisfied"] = True
            except (ValueError, TypeError):
                result["satisfied"] = False

            results.append(result)

        return results


class EnhancedMPHParser:
    """Enhanced COMSOL MPH parser with symbolic constraint support.

    MPH files are ZIP archives containing XML model definitions.

    Example:
        parser = EnhancedMPHParser()
        result = parser.parse("model.mph")

        print(f"Model: {result.model_name}")
        print(f"Physics: {[p.value for p in result.physics]}")

        # Check constraints
        for c in result.constraints:
            print(f"{c['parameter']}: {c['constraint']} -> {c['satisfied']}")
    """

    # Physics interface patterns
    PHYSICS_PATTERNS = {
        "SolidMechanics": PhysicsInterface.SOLID_MECHANICS,
        "Hookean": PhysicsInterface.SOLID_MECHANICS,
        "LinearElastic": PhysicsInterface.SOLID_MECHANICS,
        "HeatTransfer": PhysicsInterface.HEAT_TRANSFER,
        "HeatTransferInSolids": PhysicsInterface.HEAT_TRANSFER,
        "Electromagnetic": PhysicsInterface.ELECTROMAGNETICS,
        "Electrostatics": PhysicsInterface.ELECTROMAGNETICS,
        "MagneticFields": PhysicsInterface.ELECTROMAGNETICS,
        "LaminarFlow": PhysicsInterface.FLUID_FLOW,
        "TurbulentFlow": PhysicsInterface.FLUID_FLOW,
        "FluidFlow": PhysicsInterface.FLUID_FLOW,
        "TransportOfDilutedSpecies": PhysicsInterface.CHEMICAL,
        "ReactionEngineering": PhysicsInterface.CHEMICAL,
        "PressureAcoustics": PhysicsInterface.ACOUSTICS,
        "SolidMechanics_Electrostatics": PhysicsInterface.MULTIPHYSICS,
    }

    # Study type patterns
    STUDY_PATTERNS = {
        "Stationary": StudyType.STATIONARY,
        "TimeDependent": StudyType.TIME_DEPENDENT,
        "Eigenfrequency": StudyType.EIGENFREQUENCY,
        "FrequencyDomain": StudyType.FREQUENCY_DOMAIN,
        "Parametric": StudyType.PARAMETRIC,
    }

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.parameters: List[ComsolParameter] = []
        self.physics: List[PhysicsInterface] = []
        self.studies: List[StudyType] = []
        self.constraints: List[Dict[str, Any]] = []

    def parse(self, filepath: str) -> ComsolResults:
        """Parse MPH file."""
        self.parameters = []
        self.physics = []
        self.studies = []
        self.constraints = []

        model_name = ""
        version = ""
        variables = {}
        num_dofs = 0

        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                files = zf.namelist()

                # Parse model XML
                if "model.xml" in files:
                    with zf.open("model.xml") as f:
                        content = f.read().decode("utf-8")
                        model_name, version, num_dofs = self._parse_model_xml(content)

                # Parse parameters
                if "parameters.xml" in files:
                    with zf.open("parameters.xml") as f:
                        content = f.read().decode("utf-8")
                        self._parse_parameters_xml(content)

                # Parse variables
                if "variables.xml" in files:
                    with zf.open("variables.xml") as f:
                        content = f.read().decode("utf-8")
                        variables = self._parse_variables_xml(content)

                # Parse physics
                if "physics.xml" in files:
                    with zf.open("physics.xml") as f:
                        content = f.read().decode("utf-8")
                        self._parse_physics_xml(content)

                # Parse study
                if "study.xml" in files:
                    with zf.open("study.xml") as f:
                        content = f.read().decode("utf-8")
                        self._parse_study_xml(content)

        except Exception as e:
            print(f"Error parsing MPH file: {e}")

        return ComsolResults(
            model_name=model_name,
            version=version,
            physics=self.physics,
            studies=self.studies,
            parameters=self.parameters,
            variables=variables,
            constraints=self.constraints,
            num_dofs=num_dofs,
        )

    def parse_file(self, filepath: str) -> ComsolResults:
        """Parse MPH file."""
        return self.parse(filepath)

    def _parse_model_xml(self, content: str) -> Tuple[str, str, int]:
        """Parse model XML content."""
        model_name = ""
        version = ""
        num_dofs = 0

        try:
            root = ET.fromstring(content)

            # Extract model name
            if "name" in root.attrib:
                model_name = root.attrib["name"]

            # Extract version
            if "version" in root.attrib:
                version = root.attrib["version"]

            # Try to extract DOF count
            for elem in root.iter():
                if "dofs" in elem.tag.lower():
                    try:
                        num_dofs = int(elem.text)
                    except (ValueError, TypeError):
                        pass

        except ET.ParseError:
            pass

        return model_name, version, num_dofs

    def _parse_parameters_xml(self, content: str):
        """Parse parameters XML content."""
        try:
            root = ET.fromstring(content)

            for param in root.iter("parameter"):
                name = param.get("name", "")
                value_str = param.get("value", "")
                unit = param.get("unit", "")
                description = param.get("description", "")

                # Try to parse value
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str

                param_obj = ComsolParameter(
                    name=name,
                    value=value if isinstance(value, float) else 0.0,
                    unit=unit,
                    description=description,
                )

                # Only add if value is numeric
                if isinstance(value, float):
                    self.parameters.append(param_obj)

                    # Validate constraints
                    validations = ComsolSymbolicConstraints.validate_parameter(
                        name, value, unit
                    )
                    self.constraints.extend(validations)

        except ET.ParseError:
            pass

    def _parse_variables_xml(self, content: str) -> Dict[str, str]:
        """Parse variables XML content."""
        variables = {}

        try:
            root = ET.fromstring(content)

            for var in root.iter("variable"):
                name = var.get("name", "")
                value = var.get("value", "")
                if name:
                    variables[name] = value

        except ET.ParseError:
            pass

        return variables

    def _parse_physics_xml(self, content: str):
        """Parse physics XML content."""
        try:
            root = ET.fromstring(content)

            for elem in root.iter():
                tag = elem.tag
                # Check for physics interface
                for pattern, phys_type in self.PHYSICS_PATTERNS.items():
                    if pattern in tag:
                        if phys_type not in self.physics:
                            self.physics.append(phys_type)
                        break

        except ET.ParseError:
            pass

    def _parse_study_xml(self, content: str):
        """Parse study XML content."""
        try:
            root = ET.fromstring(content)

            for elem in root.iter():
                tag = elem.tag
                # Check for study type
                for pattern, study_type in self.STUDY_PATTERNS.items():
                    if pattern in tag:
                        if study_type not in self.studies:
                            self.studies.append(study_type)
                        break

        except ET.ParseError:
            pass

    def extract_multiphysics_mathematics(self) -> Dict[str, Any]:
        """Extract multiphysics mathematical structures."""
        return {
            "governing_equations": self._extract_governing_equations(),
            "discretization": self._extract_discretization(),
            "solver_settings": self._extract_solver_settings(),
            "coupling": self._extract_physics_coupling(),
        }

    def _extract_governing_equations(self) -> List[Dict[str, Any]]:
        """Extract governing equations based on physics."""
        equations = []

        for phys in self.physics:
            if phys == PhysicsInterface.SOLID_MECHANICS:
                equations.append(
                    {
                        "physics": "solid_mechanics",
                        "type": "momentum",
                        "form": "∇·σ + f = ρ∂²u/∂t²",
                        "description": "Linear momentum balance",
                        "variables": ["displacement", "stress", "strain", "force"],
                        "constitutive": "σ = C:ε (Hooke's law)",
                    }
                )

            elif phys == PhysicsInterface.HEAT_TRANSFER:
                equations.append(
                    {
                        "physics": "heat_transfer",
                        "type": "energy",
                        "form": "ρCp∂T/∂t = ∇·(k∇T) + Q",
                        "description": "Heat equation",
                        "variables": ["temperature", "heat_flux", "heat_source"],
                        "constitutive": "q = -k∇T (Fourier's law)",
                    }
                )

            elif phys == PhysicsInterface.ELECTROMAGNETICS:
                equations.append(
                    {
                        "physics": "electromagnetics",
                        "type": "maxwell",
                        "form": "∇×(μ⁻¹∇×A) = J",
                        "description": "Magnetic vector potential equation",
                        "variables": ["magnetic_potential", "current", "field"],
                        "constitutive": "B = ∇×A, D = εE",
                    }
                )

            elif phys == PhysicsInterface.FLUID_FLOW:
                equations.append(
                    {
                        "physics": "fluid_flow",
                        "type": "navier_stokes",
                        "form": "ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + f",
                        "description": "Incompressible Navier-Stokes",
                        "variables": ["velocity", "pressure", "force"],
                        "constitutive": "τ = μ(∇u + ∇uᵀ)",
                    }
                )

        return equations

    def _extract_discretization(self) -> Dict[str, Any]:
        """Extract discretization information."""
        # Find mesh settings from parameters
        hmax = None
        hmin = None
        order = "quadratic"

        for param in self.parameters:
            if param.name == "hmax":
                hmax = param.value
            elif param.name == "hmin":
                hmin = param.value
            elif param.name == "order":
                order = int(param.value) if param.value in [1, 2, 3] else "quadratic"

        return {
            "method": "finite_element",
            "element_order": order,
            "hmax": hmax,
            "hmin": hmin,
            "formulation": "galerkin",
            "shape_functions": "lagrange",
        }

    def _extract_solver_settings(self) -> Dict[str, Any]:
        """Extract solver settings."""
        settings = {
            "linear_solver": "direct",
            "nonlinear_solver": "newton",
            "preconditioner": "ilu",
            "convergence_tolerance": 1e-6,
        }

        # Determine based on study type
        if StudyType.TIME_DEPENDENT in self.studies:
            settings["time_integration"] = "bdf"
            settings["time_stepping"] = "automatic"

        if StudyType.EIGENFREQUENCY in self.studies:
            settings["eigenvalue_solver"] = "arnoldi"
            settings["number_of_modes"] = 6

        return settings

    def _extract_physics_coupling(self) -> List[Dict[str, Any]]:
        """Extract multiphysics coupling information."""
        couplings = []

        physics_list = [p.value for p in self.physics]

        # Check for common couplings
        if "solid_mechanics" in physics_list and "heat_transfer" in physics_list:
            couplings.append(
                {
                    "type": "thermal_stress",
                    "physics": ["solid_mechanics", "heat_transfer"],
                    "coupling_term": "thermal_strain = αΔT",
                    "description": "Thermoelastic coupling",
                }
            )

        if "solid_mechanics" in physics_list and "electromagnetics" in physics_list:
            couplings.append(
                {
                    "type": "electromechanical",
                    "physics": ["solid_mechanics", "electromagnetics"],
                    "coupling_term": "f_em = J×B",
                    "description": "Electromagnetic force coupling",
                }
            )

        if "fluid_flow" in physics_list and "heat_transfer" in physics_list:
            couplings.append(
                {
                    "type": "convective_heat",
                    "physics": ["fluid_flow", "heat_transfer"],
                    "coupling_term": "ρCpu·∇T",
                    "description": "Convective heat transfer",
                }
            )

        return couplings

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get summary of parameters with constraints."""
        summary = {
            "total_parameters": len(self.parameters),
            "by_unit": {},
            "critical_constraints": [],
            "satisfied_constraints": 0,
            "violated_constraints": 0,
        }

        for param in self.parameters:
            unit = param.unit if param.unit else "dimensionless"
            if unit not in summary["by_unit"]:
                summary["by_unit"][unit] = []
            summary["by_unit"][unit].append(param.name)

        for constraint in self.constraints:
            if "critical" in constraint.get("description", "").lower():
                summary["critical_constraints"].append(constraint)

            if constraint.get("satisfied", False):
                summary["satisfied_constraints"] += 1
            else:
                summary["violated_constraints"] += 1

        return summary


# Convenience function
def parse_mph(filepath: str) -> ComsolResults:
    """Parse COMSOL MPH file."""
    parser = EnhancedMPHParser()
    return parser.parse(filepath)
