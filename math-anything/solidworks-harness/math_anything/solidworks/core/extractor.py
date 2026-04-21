"""SolidWorks Simulation extractor for mathematical structure extraction."""

from typing import Any, Dict, List, Optional

import numpy as np


class SolidWorksExtractor:
    """Extractor for SolidWorks Simulation mathematical structures.

    Extracts:
    - Stress and strain results
    - Displacement fields
    - Modal results
    - Safety factors
    - Reaction forces
    """

    def __init__(self):
        self._current_data: Dict[str, Any] = {}

    def extract_stress_results(
        self,
        stress_data: np.ndarray,
        yield_strength: float,
    ) -> Dict[str, Any]:
        """Extract stress analysis results.

        Args:
            stress_data: Von Mises stress values at nodes
            yield_strength: Material yield strength

        Returns:
            Dictionary with stress statistics
        """
        max_stress = float(np.max(stress_data))
        min_stress = float(np.min(stress_data))

        # Safety factor
        safety_factor = yield_strength / max_stress if max_stress > 0 else float("inf")

        # Percentage of yield
        percent_yield = (max_stress / yield_strength) * 100 if yield_strength > 0 else 0

        return {
            "von_mises_max": max_stress,
            "von_mises_min": min_stress,
            "von_mises_mean": float(np.mean(stress_data)),
            "safety_factor_min": safety_factor,
            "percent_of_yield": percent_yield,
            "nodes_above_yield": int(np.sum(stress_data > yield_strength)),
        }

    def extract_displacement_results(
        self,
        displacement_data: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract displacement results.

        Args:
            displacement_data: Displacement vectors (n_nodes, 3)

        Returns:
            Dictionary with displacement statistics
        """
        # Magnitude
        magnitude = np.linalg.norm(displacement_data, axis=1)

        # Components
        ux, uy, uz = (
            displacement_data[:, 0],
            displacement_data[:, 1],
            displacement_data[:, 2],
        )

        return {
            "displacement_max": float(np.max(magnitude)),
            "displacement_min": float(np.min(magnitude)),
            "displacement_mean": float(np.mean(magnitude)),
            "ux_max": float(np.max(ux)),
            "ux_min": float(np.min(ux)),
            "uy_max": float(np.max(uy)),
            "uy_min": float(np.min(uy)),
            "uz_max": float(np.max(uz)),
            "uz_min": float(np.min(uz)),
        }

    def extract_modal_results(
        self,
        frequencies: np.ndarray,
        mode_shapes: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract modal analysis results.

        Args:
            frequencies: Natural frequencies (Hz)
            mode_shapes: Mode shape displacements

        Returns:
            Dictionary with modal properties
        """
        n_modes = len(frequencies)

        return {
            "num_modes": n_modes,
            "frequencies_hz": frequencies.tolist(),
            "frequencies_rad_s": (2 * np.pi * frequencies).tolist(),
            "periods_s": (1.0 / frequencies).tolist(),
            "frequency_range": (float(np.min(frequencies)), float(np.max(frequencies))),
        }

    def extract_strain_energy(
        self,
        strain_energy_density: np.ndarray,
        element_volumes: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract strain energy.

        Args:
            strain_energy_density: Strain energy per unit volume
            element_volumes: Element volumes

        Returns:
            Dictionary with energy quantities
        """
        total_energy = np.sum(strain_energy_density * element_volumes)

        return {
            "total_strain_energy": float(total_energy),
            "max_energy_density": float(np.max(strain_energy_density)),
            "mean_energy_density": float(np.mean(strain_energy_density)),
        }

    def extract_thermal_results(
        self,
        temperature_data: np.ndarray,
        flux_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract thermal analysis results.

        Args:
            temperature_data: Temperature values
            flux_data: Optional heat flux vectors

        Returns:
            Dictionary with thermal statistics
        """
        result = {
            "temperature_max": float(np.max(temperature_data)),
            "temperature_min": float(np.min(temperature_data)),
            "temperature_mean": float(np.mean(temperature_data)),
            "temperature_range": float(
                np.max(temperature_data) - np.min(temperature_data)
            ),
        }

        if flux_data is not None:
            flux_magnitude = np.linalg.norm(flux_data, axis=1)
            result.update(
                {
                    "heat_flux_max": float(np.max(flux_magnitude)),
                    "heat_flux_mean": float(np.mean(flux_magnitude)),
                }
            )

        return result

    def compute_reaction_forces(
        self,
        constraint_nodes: np.ndarray,
        internal_forces: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute reaction forces at constraints.

        Args:
            constraint_nodes: Node IDs at constraints
            internal_forces: Internal force vector

        Returns:
            Dictionary with reaction force components
        """
        reactions = internal_forces[constraint_nodes]

        total_reaction = np.sum(reactions, axis=0)

        return {
            "total_reaction_x": float(total_reaction[0]),
            "total_reaction_y": float(total_reaction[1]),
            "total_reaction_z": float(total_reaction[2]),
            "total_reaction_magnitude": float(np.linalg.norm(total_reaction)),
            "max_nodal_reaction": float(np.max(np.linalg.norm(reactions, axis=1))),
        }

    def extract_mesh_info(
        self,
        num_nodes: int,
        num_elements: int,
        element_quality_metrics: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract mesh information.

        Args:
            num_nodes: Number of mesh nodes
            num_elements: Number of elements
            element_quality_metrics: Optional quality metrics

        Returns:
            Dictionary with mesh statistics
        """
        result = {
            "num_nodes": num_nodes,
            "num_elements": num_elements,
            "average_nodes_per_element": (
                num_nodes / num_elements if num_elements > 0 else 0
            ),
        }

        if element_quality_metrics is not None:
            result.update(
                {
                    "min_element_quality": float(np.min(element_quality_metrics)),
                    "mean_element_quality": float(np.mean(element_quality_metrics)),
                    "elements_below_threshold": int(
                        np.sum(element_quality_metrics < 0.1)
                    ),
                }
            )

        return result
