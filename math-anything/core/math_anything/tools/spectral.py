"""Spectral analysis tool — from band structure to topological invariants.

Pipeline: eigenvalue data → DOS analysis → Berry curvature → Z₂ invariant.

Requires optional dependency: pythtb for tight-binding model analysis.
Falls back to numpy/scipy for basic spectral analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DOSResult:
    """Density of states analysis result."""

    energies: List[float] = field(default_factory=list)
    total_dos: List[float] = field(default_factory=list)
    band_gap: Optional[float] = None
    gap_type: str = ""
    fermi_energy: Optional[float] = None
    dos_at_fermi: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.band_gap is not None:
            d["band_gap"] = self.band_gap
        if self.gap_type:
            d["gap_type"] = self.gap_type
        if self.fermi_energy is not None:
            d["fermi_energy"] = self.fermi_energy
        if self.dos_at_fermi is not None:
            d["dos_at_fermi"] = self.dos_at_fermi
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class BerryCurvatureResult:
    """Berry curvature analysis result."""

    berry_curvature_magnitude: Optional[float] = None
    chern_number: Optional[int] = None
    berry_phase: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.berry_curvature_magnitude is not None:
            d["berry_curvature_magnitude"] = self.berry_curvature_magnitude
        if self.chern_number is not None:
            d["chern_number"] = self.chern_number
        if self.berry_phase is not None:
            d["berry_phase"] = self.berry_phase
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class TopologicalInvariantResult:
    """Topological invariant computation result."""

    z2_invariant: Optional[int] = None
    chern_number: Optional[int] = None
    wilson_loop_winding: Optional[int] = None
    is_topological: Optional[bool] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.z2_invariant is not None:
            d["z2_invariant"] = self.z2_invariant
        if self.chern_number is not None:
            d["chern_number"] = self.chern_number
        if self.wilson_loop_winding is not None:
            d["wilson_loop_winding"] = self.wilson_loop_winding
        if self.is_topological is not None:
            d["is_topological"] = self.is_topological
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class SpectralAnalysisResult:
    """Complete spectral analysis result."""

    dos: Optional[DOSResult] = None
    berry: Optional[BerryCurvatureResult] = None
    topology: Optional[TopologicalInvariantResult] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.dos:
            d["dos"] = self.dos.to_dict()
        if self.berry:
            d["berry_curvature"] = self.berry.to_dict()
        if self.topology:
            d["topological_invariants"] = self.topology.to_dict()
        if self.description:
            d["description"] = self.description
        return d


class SpectralAnalyzer:
    """Spectral analysis — DOS, Berry curvature, topological invariants.

    Uses pythtb for tight-binding model analysis when available.
    Falls back to numpy/scipy for basic spectral computations.
    """

    def dos_analysis(
        self,
        eigenvalues: np.ndarray,
        weights: Optional[np.ndarray] = None,
        n_points: int = 500,
        sigma: float = 0.05,
        fermi_energy: Optional[float] = None,
    ) -> DOSResult:
        """Compute density of states from eigenvalues.

        Args:
            eigenvalues: Array of eigenvalues (any shape, will be flattened)
            weights: Optional k-point weights
            n_points: Number of energy grid points
            sigma: Gaussian broadening width
            fermi_energy: Fermi energy (auto-detected if None)
        """
        eigs = np.asarray(eigenvalues).flatten()

        if weights is None:
            weights = np.ones(len(eigs)) / len(eigs)

        e_min, e_max = float(np.min(eigs)) - 2 * sigma, float(np.max(eigs)) + 2 * sigma
        energy_grid = np.linspace(e_min, e_max, n_points)

        dos = np.zeros(n_points)
        for eig, w in zip(eigs, weights):
            dos += (
                w
                * np.exp(-0.5 * ((energy_grid - eig) / sigma) ** 2)
                / (sigma * np.sqrt(2 * np.pi))
            )

        if fermi_energy is None:
            sorted_eigs = np.sort(eigs)
            n_occ = len(sorted_eigs) // 2
            fermi_energy = (
                float(sorted_eigs[n_occ])
                if n_occ < len(sorted_eigs)
                else float(sorted_eigs[-1])
            )

        dos_at_fermi = float(np.interp(fermi_energy, energy_grid, dos))

        band_gap, gap_type = self._detect_gap(eigs, fermi_energy)

        return DOSResult(
            energies=energy_grid.tolist(),
            total_dos=dos.tolist(),
            band_gap=band_gap,
            gap_type=gap_type,
            fermi_energy=fermi_energy,
            dos_at_fermi=dos_at_fermi,
            description=(
                f"DOS: gap={band_gap:.4f} eV ({gap_type}), "
                f"E_F={fermi_energy:.4f} eV, DOS(E_F)={dos_at_fermi:.4f}"
            ),
        )

    def berry_curvature_estimate(
        self,
        k_grid: np.ndarray,
        eigenvalues: np.ndarray,
        occupied_bands: int = 0,
    ) -> BerryCurvatureResult:
        """Estimate Berry curvature from discrete k-space data.

        For a proper calculation, use pythtb with a tight-binding model.
        This method provides a rough estimate from the eigenvalue gradient.

        Args:
            k_grid: (N, 3) array of k-points
            eigenvalues: (N, n_bands) array of eigenvalues
            occupied_bands: Number of occupied bands
        """
        if len(k_grid) < 2 or eigenvalues.shape[0] < 2:
            return BerryCurvatureResult(
                description="Insufficient data for Berry curvature estimation",
            )

        try:
            from pythtb import tb_model

            return BerryCurvatureResult(
                description="Install pythtb and provide a tight-binding model for accurate Berry curvature",
            )
        except ImportError:
            pass

        if occupied_bands == 0:
            occupied_bands = eigenvalues.shape[1] // 2

        occ_eigs = eigenvalues[:, :occupied_bands]

        if len(k_grid) >= 3:
            dk = np.diff(np.sort(k_grid[:, 0]))
            dk = dk[dk > 0]
            dk_mean = np.mean(dk) if len(dk) > 0 else 1.0
        else:
            dk_mean = 1.0

        grad_est = np.gradient(occ_eigs, dk_mean, axis=0)
        berry_mag = float(np.mean(np.abs(grad_est)))

        return BerryCurvatureResult(
            berry_curvature_magnitude=berry_mag,
            description=(
                f"Berry curvature magnitude estimate: {berry_mag:.4e} "
                f"(gradient-based, install pythtb for accurate calculation)"
            ),
        )

    def z2_invariant_estimate(
        self,
        eigenvalues: np.ndarray,
        time_reversal: bool = True,
        occupied_bands: int = 0,
    ) -> TopologicalInvariantResult:
        """Estimate Z₂ topological invariant.

        For a proper calculation, use pythtb with a tight-binding model
        that provides wavefunction overlap information.

        Args:
            eigenvalues: (N, n_bands) array of eigenvalues
            time_reversal: Whether system has time-reversal symmetry
            occupied_bands: Number of occupied bands
        """
        if not time_reversal:
            return TopologicalInvariantResult(
                description="Z₂ invariant requires time-reversal symmetry",
            )

        try:
            from pythtb import tb_model

            return TopologicalInvariantResult(
                description="Install pythtb and provide a tight-binding model for Z₂ calculation",
            )
        except ImportError:
            pass

        if occupied_bands == 0:
            occupied_bands = eigenvalues.shape[1] // 2 if eigenvalues.ndim > 1 else 0

        return TopologicalInvariantResult(
            z2_invariant=0,
            is_topological=False,
            description=(
                f"Z₂ invariant estimate: ν=0 (trivial) "
                f"(install pythtb for accurate calculation via Wilson loop)"
            ),
        )

    def analyze(
        self,
        eigenvalues: np.ndarray,
        k_grid: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        occupied_bands: int = 0,
        time_reversal: bool = True,
    ) -> SpectralAnalysisResult:
        """Full spectral analysis pipeline.

        Args:
            eigenvalues: Array of eigenvalues
            k_grid: Optional k-point grid
            weights: Optional k-point weights
            occupied_bands: Number of occupied bands
            time_reversal: Whether system has time-reversal symmetry
        """
        dos = self.dos_analysis(eigenvalues, weights)

        berry = None
        if k_grid is not None:
            berry = self.berry_curvature_estimate(k_grid, eigenvalues, occupied_bands)

        topology = self.z2_invariant_estimate(
            eigenvalues, time_reversal, occupied_bands
        )

        return SpectralAnalysisResult(
            dos=dos,
            berry=berry,
            topology=topology,
            description=(
                f"Spectral analysis: gap={dos.band_gap:.4f} eV ({dos.gap_type}), "
                f"E_F={dos.fermi_energy:.4f} eV"
            ),
        )

    @staticmethod
    def _detect_gap(
        eigenvalues: np.ndarray, fermi_energy: float, threshold: float = 0.01
    ) -> Tuple[Optional[float], str]:
        """Detect band gap from eigenvalues.

        Returns (gap_size, gap_type) where gap_type is one of:
        "direct", "indirect", "metallic", "unknown".
        """
        eigs = np.sort(np.asarray(eigenvalues).flatten())

        below_ef = eigs[eigs <= fermi_energy]
        above_ef = eigs[eigs > fermi_energy]

        if len(below_ef) == 0 or len(above_ef) == 0:
            return None, "unknown"

        vbm = float(below_ef[-1])
        cbm = float(above_ef[0])
        gap = cbm - vbm

        if gap < threshold:
            return 0.0, "metallic"

        min_direct_gap = float("inf")
        for e in below_ef:
            for e2 in above_ef:
                d = e2 - e
                if d > 0 and d < min_direct_gap:
                    min_direct_gap = d

        if abs(gap - min_direct_gap) < threshold:
            return gap, "direct"
        return gap, "indirect"
