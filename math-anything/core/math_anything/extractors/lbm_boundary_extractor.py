"""Lattice Boltzmann Method (LBM) Boundary Condition Extractor.

Specialized extractor for LBM boundary condition numerical implementations.

LBM-specific features:
- Bounce-back boundary conditions (standard, interpolated, curved)
- Zou-He velocity/pressure boundaries
- Periodic and symmetry boundaries
- Moving wall implementations
- Immersed boundary methods

Mathematical focus:
- Link-wise vs node-wise boundary treatment
- Chapman-Enskog analysis of boundary schemes
- Moment-based vs population-based conditions
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LBMBoundaryType(Enum):
    """LBM-specific boundary types."""

    BOUNCE_BACK = "bounce_back"
    INTERPOLATED_BOUNCE_BACK = "interpolated_bounce_back"
    ZOU_HE_VELOCITY = "zou_he_velocity"
    ZOU_HE_PRESSURE = "zou_he_pressure"
    PERIODIC = "periodic"
    MOVING_WALL = "moving_wall"
    SLIP = "slip"
    OUTFLOW = "outflow"
    INLET = "inlet"
    CONVECTIVE_OUTFLOW = "convective_outflow"


class LBMCollisionModel(Enum):
    """LBM collision models."""

    BGK = "bgk"  # Single relaxation time
    TRT = "trt"  # Two relaxation times
    MRT = "mrt"  # Multiple relaxation times
    REGULARIZED = "regularized"
    CUMULANT = "cumulant"


@dataclass
class LBMBoundary:
    """LBM boundary condition specification."""

    boundary_type: LBMBoundaryType
    location: str
    lattice_directions: List[int]  # Which lattice velocities are affected
    unknown_populations: List[int]  # Populations to be determined
    known_macroscopic: Dict[str, float]  # Known macroscopic quantities
    mathematical_form: str
    implementation_details: Dict[str, Any]
    accuracy_order: int
    stability_constraints: List[str]


class LBMBoundaryExtractor:
    """Extract LBM boundary condition mathematical structures.

    Identifies and characterizes boundary condition implementations
    in Lattice Boltzmann simulations.

    Example:
        ```python
        extractor = LBMBoundaryExtractor()

        # Analyze boundary configuration
        boundaries = extractor.extract_boundaries(
            flag_field=flag_array,
            lattice_model='D3Q19',
            collision_model='BGK',
        )

        # Get mathematical description
        for bc in boundaries:
            print(bc.mathematical_form)
            print(bc.implementation_details)
        ```
    """

    # Lattice directions for common stencils
    D3Q19_DIRECTIONS = {
        0: (0, 0, 0),  # Rest
        1: (1, 0, 0),
        2: (-1, 0, 0),  # x
        3: (0, 1, 0),
        4: (0, -1, 0),  # y
        5: (0, 0, 1),
        6: (0, 0, -1),  # z
        7: (1, 1, 0),
        8: (-1, -1, 0),  # xy
        9: (1, -1, 0),
        10: (-1, 1, 0),
        11: (1, 0, 1),
        12: (-1, 0, -1),  # xz
        13: (1, 0, -1),
        14: (-1, 0, 1),
        15: (0, 1, 1),
        16: (0, -1, -1),  # yz
        17: (0, 1, -1),
        18: (0, -1, 1),
    }

    D3Q19_OPPOSITES = {
        0: 0,
        1: 2,
        2: 1,
        3: 4,
        4: 3,
        5: 6,
        6: 5,
        7: 8,
        8: 7,
        9: 10,
        10: 9,
        11: 12,
        12: 11,
        13: 14,
        14: 13,
        15: 16,
        16: 15,
        17: 18,
        18: 17,
    }

    def __init__(self):
        self.boundaries: List[LBMBoundary] = []
        self.lattice_model = "D3Q19"
        self.collision_model = LBMCollisionModel.BGK

    def extract_boundaries(
        self,
        flag_field: np.ndarray,
        velocity_field: Optional[np.ndarray] = None,
        density_field: Optional[np.ndarray] = None,
        lattice_model: str = "D3Q19",
        collision_model: str = "BGK",
    ) -> List[LBMBoundary]:
        """Extract all boundary conditions from simulation.

        Args:
            flag_field: Boundary flag array (nx, ny, nz)
            velocity_field: Optional velocity for BC identification
            density_field: Optional density for BC identification
            lattice_model: Lattice model (D3Q19, D3Q27, etc.)
            collision_model: Collision model (BGK, TRT, MRT)

        Returns:
            List of LBM boundary specifications
        """
        self.lattice_model = lattice_model
        self.collision_model = LBMCollisionModel(collision_model.lower())
        self.boundaries = []

        # Detect boundary types from flags
        boundary_types = self._detect_boundary_types(flag_field)

        for boundary_type, locations in boundary_types.items():
            for location in locations:
                bc = self._characterize_boundary(
                    boundary_type, location, flag_field, velocity_field, density_field
                )
                self.boundaries.append(bc)

        return self.boundaries

    def _detect_boundary_types(self, flag_field: np.ndarray) -> Dict[str, List[str]]:
        """Detect boundary types from flag field."""
        nx, ny, nz = flag_field.shape

        boundaries = {
            "wall": [],
            "inlet": [],
            "outlet": [],
            "slip": [],
            "periodic": [],
            "moving_wall": [],
        }

        # Check domain faces
        # X-min face
        if np.any(flag_field[0, :, :] == 1):
            boundaries["wall"].append("xmin")
        if np.any(flag_field[0, :, :] == 2):
            boundaries["inlet"].append("xmin")

        # X-max face
        if np.any(flag_field[nx - 1, :, :] == 1):
            boundaries["wall"].append("xmax")
        if np.any(flag_field[nx - 1, :, :] == 3):
            boundaries["outlet"].append("xmax")

        # Y faces
        if np.any(flag_field[:, 0, :] == 1):
            boundaries["wall"].append("ymin")
        if np.any(flag_field[:, ny - 1, :] == 1):
            boundaries["wall"].append("ymax")

        # Z faces
        if np.any(flag_field[:, :, 0] == 1):
            boundaries["wall"].append("zmin")
        if np.any(flag_field[:, :, nz - 1] == 1):
            boundaries["wall"].append("zmax")

        return boundaries

    def _characterize_boundary(
        self,
        boundary_type: str,
        location: str,
        flag_field: np.ndarray,
        velocity_field: Optional[np.ndarray],
        density_field: Optional[np.ndarray],
    ) -> LBMBoundary:
        """Characterize a specific boundary."""

        if boundary_type == "wall":
            return self._create_bounce_back_boundary(location)
        elif boundary_type == "inlet":
            return self._create_zou_he_velocity_boundary(location, velocity_field)
        elif boundary_type == "outlet":
            return self._create_zou_he_pressure_boundary(location, density_field)
        elif boundary_type == "slip":
            return self._create_slip_boundary(location)
        elif boundary_type == "moving_wall":
            return self._create_moving_wall_boundary(location, velocity_field)
        else:
            return self._create_generic_boundary(boundary_type, location)

    def _create_bounce_back_boundary(self, location: str) -> LBMBoundary:
        """Create bounce-back boundary specification."""

        # Determine affected lattice directions based on boundary location
        if location == "xmin":
            incoming = [2, 8, 10, 12, 14]  # Negative x directions
        elif location == "xmax":
            incoming = [1, 7, 9, 11, 13]  # Positive x directions
        elif location == "ymin":
            incoming = [4, 8, 10, 16, 18]
        elif location == "ymax":
            incoming = [3, 7, 9, 15, 17]
        elif location == "zmin":
            incoming = [6, 12, 14, 16, 18]
        elif location == "zmax":
            incoming = [5, 11, 13, 15, 17]
        else:
            incoming = []

        # Opposite directions are the unknown populations
        unknown = [self.D3Q19_OPPOSITES[i] for i in incoming]

        return LBMBoundary(
            boundary_type=LBMBoundaryType.BOUNCE_BACK,
            location=location,
            lattice_directions=incoming,
            unknown_populations=unknown,
            known_macroscopic={"u_wall": 0.0, "v_wall": 0.0, "w_wall": 0.0},
            mathematical_form="f_{\bar{i}} = f_i^{pre}",
            implementation_details={
                "algorithm": "link_wise_bounce_back",
                "description": "Post-collision populations are reversed at wall",
                "equation": "f_{opposite}(x_w, t+Δt) = f_i^*(x_w, t)",
                "physical_interpretation": "Mid-link bounce-back, 1st order accuracy",
                "staircase_effect": "Wall approximated as staircase",
                "wall_location": "midway_between_nodes",
                "timing": "post_collision",
            },
            accuracy_order=1,
            stability_constraints=[
                "τ > 0.5 for stability",
                "No constraint on lattice velocity magnitude",
            ],
        )

    def _create_zou_he_velocity_boundary(
        self,
        location: str,
        velocity_field: Optional[np.ndarray],
    ) -> LBMBoundary:
        """Create Zou-He velocity boundary specification."""

        if location == "xmin":
            incoming = [2, 8, 10, 12, 14]
            normal = "x"
        elif location == "xmax":
            incoming = [1, 7, 9, 11, 13]
            normal = "x"
        else:
            incoming = []
            normal = "unknown"

        unknown = [self.D3Q19_OPPOSITES[i] for i in incoming]

        # Get target velocity if available
        u_target = 0.0
        if velocity_field is not None:
            idx = {"xmin": 0, "xmax": -1}.get(location, 0)
            u_target = float(np.mean(velocity_field[idx, :, :, 0]))

        return LBMBoundary(
            boundary_type=LBMBoundaryType.ZOU_HE_VELOCITY,
            location=location,
            lattice_directions=incoming,
            unknown_populations=unknown,
            known_macroscopic={
                "u_specified": u_target,
                "v_specified": 0.0,
                "w_specified": 0.0,
            },
            mathematical_form="Solve for unknown f_i using mass conservation and velocity constraints",
            implementation_details={
                "algorithm": "zou_he",
                "description": "Moment-based boundary condition for velocity",
                "equations": [
                    "Mass: Σ f_i = ρ (known)",
                    "Momentum: Σ c_i f_i = ρ u (specified)",
                    "Bounce-back for non-equilibrium parts",
                ],
                "physical_interpretation": "Dirichlet velocity BC",
                "accuracy": "2nd order for velocity",
                "normal_direction": normal,
            },
            accuracy_order=2,
            stability_constraints=[
                "Velocity should be < 0.1 in lattice units",
                "Density should remain positive",
            ],
        )

    def _create_zou_he_pressure_boundary(
        self,
        location: str,
        density_field: Optional[np.ndarray],
    ) -> LBMBoundary:
        """Create Zou-He pressure boundary specification."""

        if location == "xmin":
            incoming = [2, 8, 10, 12, 14]
        elif location == "xmax":
            incoming = [1, 7, 9, 11, 13]
        else:
            incoming = []

        unknown = [self.D3Q19_OPPOSITES[i] for i in incoming]

        # Get target density if available
        rho_target = 1.0
        if density_field is not None:
            idx = {"xmin": 0, "xmax": -1}.get(location, 0)
            rho_target = float(np.mean(density_field[idx, :, :]))

        return LBMBoundary(
            boundary_type=LBMBoundaryType.ZOU_HE_PRESSURE,
            location=location,
            lattice_directions=incoming,
            unknown_populations=unknown,
            known_macroscopic={"rho_specified": rho_target},
            mathematical_form="ρ = Σ f_i = ρ_specified, velocity from extrapolation",
            implementation_details={
                "algorithm": "zou_he_pressure",
                "description": "Moment-based boundary for pressure (density)",
                "equations": [
                    "Density: Σ f_i = ρ_specified",
                    "Tangential velocity from interior (extrapolation)",
                    "Normal velocity from mass conservation",
                ],
                "physical_interpretation": "Dirichlet pressure BC, Neumann velocity",
                "accuracy": "1st order for velocity, exact for pressure",
            },
            accuracy_order=1,
            stability_constraints=[
                "Density must be positive",
                "Velocity at boundary should remain subsonic",
            ],
        )

    def _create_moving_wall_boundary(
        self,
        location: str,
        velocity_field: Optional[np.ndarray],
    ) -> LBMBoundary:
        """Create moving wall boundary specification."""

        u_wall = 0.1  # Default
        if velocity_field is not None:
            idx = {"xmin": 0, "xmax": -1}.get(location, 0)
            u_wall = float(np.mean(velocity_field[idx, :, :, 0]))

        return LBMBoundary(
            boundary_type=LBMBoundaryType.MOVING_WALL,
            location=location,
            lattice_directions=[],
            unknown_populations=[],
            known_macroscopic={"u_wall": u_wall, "v_wall": 0.0, "w_wall": 0.0},
            mathematical_form="f_{\bar{i}} = f_i - 6w_i ρ (c_i · u_wall)/c_s²",
            implementation_details={
                "algorithm": "equilibrium_enhanced_bounce_back",
                "description": "Bounce-back with momentum exchange for moving walls",
                "equation": "f_{opp} = f_i - 6 w_i ρ (c_i · u_w)/c_s²",
                "physical_interpretation": "Wall moving with velocity u_w",
                "accuracy": "1st order",
                "special_notes": "Can add shear to fluid",
            },
            accuracy_order=1,
            stability_constraints=[
                "Wall velocity should be < 0.1",
                "Density variations should be small",
            ],
        )

    def _create_slip_boundary(self, location: str) -> LBMBoundary:
        """Create slip boundary specification."""
        return LBMBoundary(
            boundary_type=LBMBoundaryType.SLIP,
            location=location,
            lattice_directions=[],
            unknown_populations=[],
            known_macroscopic={"u_n": 0.0},  # Normal velocity zero
            mathematical_form="Specular reflection of populations",
            implementation_details={
                "algorithm": "specular_reflection",
                "description": "Mirror reflection at boundary",
                "equation": "f_{reflected} = f_{incident} with c_⊥ mirrored",
                "physical_interpretation": "Zero shear stress, free slip",
            },
            accuracy_order=1,
            stability_constraints=[],
        )

    def _create_generic_boundary(
        self, boundary_type: str, location: str
    ) -> LBMBoundary:
        """Create generic boundary specification."""
        return LBMBoundary(
            boundary_type=LBMBoundaryType.BOUNCE_BACK,
            location=location,
            lattice_directions=[],
            unknown_populations=[],
            known_macroscopic={},
            mathematical_form="Generic boundary condition",
            implementation_details={"type": boundary_type},
            accuracy_order=1,
            stability_constraints=[],
        )

    def generate_mathematical_description(self, boundary: LBMBoundary) -> str:
        """Generate mathematical description of boundary."""
        desc = f"""## {boundary.boundary_type.value.replace('_', ' ').title()} Boundary

**Location**: {boundary.location}

**Mathematical Form**: {boundary.mathematical_form}

**Known Macroscopic Quantities**:
"""
        for name, value in boundary.known_macroscopic.items():
            desc += f"- {name} = {value}\n"

        desc += f"""
**Implementation Details**:
- Algorithm: {boundary.implementation_details.get('algorithm', 'N/A')}
- Accuracy: {boundary.accuracy_order}st/nd order

**Stability Constraints**:
"""
        for constraint in boundary.stability_constraints:
            desc += f"- {constraint}\n"

        return desc

    def get_chapman_enskog_analysis(self) -> Dict[str, Any]:
        """Get Chapman-Enskog analysis for boundary conditions."""
        return {
            "bounce_back": {
                "description": "Recovers no-slip condition with O(Δx²) error",
                "macroscopic_equivalent": "u = 0 at wall",
                "error_analysis": "Location error ~ Δx²/τ",
            },
            "zou_he": {
                "description": "Exact for specified moments",
                "macroscopic_equivalent": "Dirichlet for velocity/pressure",
                "error_analysis": "O(Δx²) for velocity, exact for pressure",
            },
            "moving_wall": {
                "description": "Recovers u = u_wall with O(Δx) error",
                "macroscopic_equivalent": "Dirichlet velocity at moving wall",
                "error_analysis": "1st order due to staircase approximation",
            },
        }
