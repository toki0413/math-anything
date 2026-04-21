"""COMSOL extractor for mathematical structure extraction.

Extracts mathematical objects from COMSOL simulation results.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ComsolExtractor:
    """Extractor for COMSOL multiphysics mathematical structures.

    Extracts:
    - Field variables (scalars, vectors, tensors)
    - Derived quantities (fluxes, gradients)
    - Global parameters and variables
    - Mesh-based quantities
    """

    def __init__(self):
        self._current_data: Dict[str, Any] = {}

    def extract_field_variable(
        self,
        field_data: np.ndarray,
        field_name: str,
        field_type: str = "scalar",
        coordinates: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract field variable statistics.

        Args:
            field_data: Field data array
            field_name: Name of the field
            field_type: Type ('scalar', 'vector', 'tensor')
            coordinates: Optional coordinates

        Returns:
            Dictionary with field statistics
        """
        result = {
            "name": field_name,
            "type": field_type,
            "num_points": field_data.shape[0] if len(field_data.shape) > 0 else 1,
        }

        if field_type == "scalar":
            result.update(
                {
                    "min": float(np.min(field_data)),
                    "max": float(np.max(field_data)),
                    "mean": float(np.mean(field_data)),
                    "std": float(np.std(field_data)),
                    "integral": float(np.sum(field_data)),
                }
            )

        elif field_type == "vector":
            # Compute magnitude
            magnitude = np.linalg.norm(field_data, axis=1)
            result.update(
                {
                    "magnitude_min": float(np.min(magnitude)),
                    "magnitude_max": float(np.max(magnitude)),
                    "magnitude_mean": float(np.mean(magnitude)),
                    "component_means": np.mean(field_data, axis=0).tolist(),
                }
            )

        elif field_type == "tensor":
            # Compute tensor invariants (for symmetric 3x3)
            if field_data.shape[1] == 6:  # Voigt notation [xx, yy, zz, xy, yz, xz]
                invariants = self._compute_tensor_invariants(field_data)
                result.update(invariants)

        return result

    def _compute_tensor_invariants(self, tensor_data: np.ndarray) -> Dict[str, Any]:
        """Compute invariants for tensor field."""
        n = tensor_data.shape[0]

        I1 = np.zeros(n)
        I2 = np.zeros(n)
        I3 = np.zeros(n)

        for i in range(n):
            # Build full tensor
            t = tensor_data[i]
            T = np.array(
                [
                    [t[0], t[3], t[5]],
                    [t[3], t[1], t[4]],
                    [t[5], t[4], t[2]],
                ]
            )

            # Invariants
            I1[i] = np.trace(T)
            I2[i] = 0.5 * (np.trace(T) ** 2 - np.trace(T @ T))
            I3[i] = np.linalg.det(T)

        return {
            "I1_mean": float(np.mean(I1)),
            "I2_mean": float(np.mean(I2)),
            "I3_mean": float(np.mean(I3)),
            "I1_range": (float(np.min(I1)), float(np.max(I1))),
        }

    def extract_line_integral(
        self,
        field_values: np.ndarray,
        line_coordinates: np.ndarray,
    ) -> float:
        """Compute line integral of field along curve.

        Args:
            field_values: Field values along line
            line_coordinates: Coordinates of line points

        Returns:
            Line integral value
        """
        # Compute arc length
        deltas = np.diff(line_coordinates, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)

        # Trapezoidal integration
        avg_values = 0.5 * (field_values[:-1] + field_values[1:])
        integral = np.sum(avg_values * segment_lengths)

        return float(integral)

    def extract_surface_integral(
        self,
        field_values: np.ndarray,
        face_areas: np.ndarray,
    ) -> float:
        """Compute surface integral of field.

        Args:
            field_values: Field values at surface points
            face_areas: Area of each face

        Returns:
            Surface integral value
        """
        return float(np.sum(field_values * face_areas))

    def extract_global_variables(
        self,
        variable_dict: Dict[str, float],
    ) -> Dict[str, Any]:
        """Extract global scalar variables.

        Args:
            variable_dict: Dictionary of variable names and values

        Returns:
            Dictionary with organized variables
        """
        organized = {
            "parameters": {},
            "results": {},
            "derived": {},
        }

        for name, value in variable_dict.items():
            # Categorize based on naming
            if any(x in name.lower() for x in ["max", "min", "mean", "int"]):
                organized["results"][name] = value
            elif any(x in name.lower() for x in ["E_", "nu_", "rho_", "k_"]):
                organized["parameters"][name] = value
            else:
                organized["derived"][name] = value

        return organized

    def extract_probe_data(
        self,
        probe_values: np.ndarray,
        probe_coordinates: np.ndarray,
        time_values: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract data from probe points.

        Args:
            probe_values: Values at probe points
            probe_coordinates: Probe coordinates
            time_values: Optional time values for transient

        Returns:
            Dictionary with probe statistics
        """
        result = {
            "num_probes": probe_values.shape[0],
            "coordinates": probe_coordinates.tolist(),
        }

        if time_values is not None:
            # Transient data
            result.update(
                {
                    "time_min": float(np.min(time_values)),
                    "time_max": float(np.max(time_values)),
                    "num_timesteps": len(time_values),
                }
            )

            # Statistics over time for each probe
            result["probe_max_over_time"] = np.max(probe_values, axis=1).tolist()
            result["probe_min_over_time"] = np.min(probe_values, axis=1).tolist()
            result["probe_mean_over_time"] = np.mean(probe_values, axis=1).tolist()
        else:
            # Stationary data
            result["probe_values"] = probe_values.tolist()

        return result

    def compute_sensitivity(
        self,
        output_values: np.ndarray,
        parameter_values: np.ndarray,
    ) -> Dict[str, float]:
        """Compute sensitivity of output to parameter.

        Args:
            output_values: Output values
            parameter_values: Parameter values

        Returns:
            Dictionary with sensitivity metrics
        """
        # Compute derivative using finite differences
        d_output = np.diff(output_values)
        d_param = np.diff(parameter_values)

        sensitivities = d_output / d_param

        return {
            "sensitivity_mean": float(np.mean(sensitivities)),
            "sensitivity_max": float(np.max(np.abs(sensitivities))),
            "sensitivity_range": (
                float(np.min(sensitivities)),
                float(np.max(sensitivities)),
            ),
        }
