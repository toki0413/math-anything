"""GROMACS extractor for mathematical structure extraction.

Extracts thermodynamic and kinetic properties from MD simulations.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GromacsExtractor:
    """Extractor for GROMACS MD simulation data.

    Extracts:
    - Thermodynamic properties
    - Structural properties
    - Transport coefficients
    - Free energy data
    """

    def __init__(self):
        self._current_data: Dict[str, Any] = {}

    def extract_thermodynamics(
        self,
        temperature: np.ndarray,
        pressure: np.ndarray,
        volume: np.ndarray,
        density: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract thermodynamic properties.

        Args:
            temperature: Temperature time series (K)
            pressure: Pressure time series (bar)
            volume: Volume time series (nm³)
            density: Optional density time series (kg/m³)

        Returns:
            Dictionary with thermodynamic statistics
        """
        results = {
            "temperature_mean": float(np.mean(temperature)),
            "temperature_std": float(np.std(temperature)),
            "temperature_min": float(np.min(temperature)),
            "temperature_max": float(np.max(temperature)),
            "pressure_mean": float(np.mean(pressure)),
            "pressure_std": float(np.std(pressure)),
            "pressure_min": float(np.min(pressure)),
            "pressure_max": float(np.max(pressure)),
            "volume_mean": float(np.mean(volume)),
            "volume_std": float(np.std(volume)),
        }

        if density is not None:
            results.update(
                {
                    "density_mean": float(np.mean(density)),
                    "density_std": float(np.std(density)),
                }
            )

        # Compressibility estimate (isothermal)
        if len(pressure) > 1:
            volume_mean = np.mean(volume)
            pressure_var = np.var(pressure)
            if pressure_var > 0:
                kappa_T = (
                    volume_mean * pressure_var / (np.mean(temperature) * 1.380649e-23)
                )
                results["compressibility_estimate"] = float(kappa_T)

        return results

    def extract_energies(
        self,
        total_energy: np.ndarray,
        potential_energy: np.ndarray,
        kinetic_energy: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract energy statistics.

        Args:
            total_energy: Total energy
            potential_energy: Potential energy
            kinetic_energy: Kinetic energy

        Returns:
            Dictionary with energy statistics
        """
        # Drift analysis
        n_frames = len(total_energy)
        if n_frames > 1:
            time = np.arange(n_frames)
            drift_coeff = np.polyfit(time, total_energy, 1)[0]
            drift_per_ns = drift_coeff * n_frames  # Assuming uniform sampling
        else:
            drift_per_ns = 0.0

        return {
            "total_energy_mean": float(np.mean(total_energy)),
            "total_energy_std": float(np.std(total_energy)),
            "total_energy_drift_per_ns": float(drift_per_ns),
            "potential_energy_mean": float(np.mean(potential_energy)),
            "potential_energy_std": float(np.std(potential_energy)),
            "kinetic_energy_mean": float(np.mean(kinetic_energy)),
            "kinetic_energy_std": float(np.std(kinetic_energy)),
            "energy_fluctuation": (
                float(np.std(total_energy) / np.mean(total_energy))
                if np.mean(total_energy) != 0
                else 0
            ),
        }

    def compute_rmsd(
        self,
        positions: np.ndarray,
        reference: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute RMSD from reference structure.

        Args:
            positions: Trajectory positions (n_frames, n_atoms, 3)
            reference: Reference positions (n_atoms, 3)

        Returns:
            Dictionary with RMSD statistics
        """
        n_frames = positions.shape[0]
        rmsd = np.zeros(n_frames)

        for i in range(n_frames):
            diff = positions[i] - reference
            rmsd[i] = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        return {
            "rmsd_mean": float(np.mean(rmsd)),
            "rmsd_std": float(np.std(rmsd)),
            "rmsd_max": float(np.max(rmsd)),
            "rmsd_final": float(rmsd[-1]),
            "rmsd_time_series": rmsd.tolist(),
        }

    def compute_radius_of_gyration(
        self,
        positions: np.ndarray,
        masses: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute radius of gyration.

        Args:
            positions: Atomic positions (n_frames, n_atoms, 3)
            masses: Optional atomic masses

        Returns:
            Dictionary with Rg statistics
        """
        n_frames = positions.shape[0]
        rg = np.zeros(n_frames)

        if masses is None:
            masses = np.ones(positions.shape[1])

        masses = masses / np.sum(masses)  # Normalize

        for i in range(n_frames):
            # Center of mass
            com = np.sum(positions[i] * masses[:, None], axis=0)
            # Rg
            rg[i] = np.sqrt(np.sum(masses * np.sum((positions[i] - com) ** 2, axis=1)))

        return {
            "rg_mean": float(np.mean(rg)),
            "rg_std": float(np.std(rg)),
            "rg_min": float(np.min(rg)),
            "rg_max": float(np.max(rg)),
            "rg_time_series": rg.tolist(),
        }

    def compute_diffusion_coefficient(
        self,
        msd: np.ndarray,
        time: np.ndarray,
        dimensionality: int = 3,
    ) -> Dict[str, Any]:
        """Compute diffusion coefficient from MSD.

        Args:
            msd: Mean squared displacement (nm²)
            time: Time array (ps)
            dimensionality: Spatial dimensions (1, 2, or 3)

        Returns:
            Dictionary with diffusion coefficient
        """
        # Linear fit to MSD
        # MSD = 2*d*D*t for d dimensions
        slope, intercept = np.polyfit(time, msd, 1)

        D = slope / (2 * dimensionality)  # nm²/ps
        D_m2_s = D * 1e-8  # Convert to m²/s

        return {
            "diffusion_coefficient_nm2_ps": float(D),
            "diffusion_coefficient_m2_s": float(D_m2_s),
            "msd_slope": float(slope),
            "msd_intercept": float(intercept),
        }

    def extract_hydrogen_bonds(
        self,
        hbond_counts: np.ndarray,
        time: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract hydrogen bond statistics.

        Args:
            hbond_counts: Number of hydrogen bonds per frame
            time: Time array

        Returns:
            Dictionary with H-bond statistics
        """
        return {
            "hbond_mean": float(np.mean(hbond_counts)),
            "hbond_std": float(np.std(hbond_counts)),
            "hbond_min": float(np.min(hbond_counts)),
            "hbond_max": float(np.max(hbond_counts)),
            "hbond_lifetime_estimate": float(np.mean(hbond_counts > 0)),
        }

    def compute_free_energy_profile(
        self,
        reaction_coordinate: np.ndarray,
        temperature: float,
        bins: int = 50,
    ) -> Dict[str, Any]:
        """Compute free energy profile along reaction coordinate.

        Args:
            reaction_coordinate: Collective variable values
            temperature: Temperature in K
            bins: Number of histogram bins

        Returns:
            Dictionary with free energy profile
        """
        # Histogram
        hist, bin_edges = np.histogram(reaction_coordinate, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Free energy: G = -kT ln(P)
        k_B = 1.380649e-23  # J/K
        N_A = 6.02214076e23  # Avogadro
        R = k_B * N_A  # J/(mol K)

        # Avoid log(0)
        hist_safe = np.where(hist > 0, hist, np.min(hist[hist > 0]) / 100)

        G = -R * temperature * np.log(hist_safe)
        G = G - np.min(G)  # Set minimum to zero

        return {
            "bin_centers": bin_centers.tolist(),
            "free_energy_kJ_mol": G.tolist(),
            "free_energy_min": float(np.min(G)),
            "free_energy_max": float(np.max(G)),
            "barrier_height": float(np.max(G) - np.min(G)),
        }
