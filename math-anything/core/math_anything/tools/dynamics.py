"""Dynamical systems analysis tool — chaos detection and modal decomposition.

Pipeline: time series → Lyapunov exponents → dynamic modes → chaos detection.

Requires optional dependencies: nolds, pydmd.
Falls back to numpy-based basic analysis when unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LyapunovResult:
    """Lyapunov exponent analysis result."""

    max_lyapunov: Optional[float] = None
    lyapunov_spectrum: List[float] = field(default_factory=list)
    is_chaotic: Optional[bool] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.max_lyapunov is not None:
            d["max_lyapunov"] = self.max_lyapunov
        if self.lyapunov_spectrum:
            d["lyapunov_spectrum"] = self.lyapunov_spectrum
        if self.is_chaotic is not None:
            d["is_chaotic"] = self.is_chaotic
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class FractalResult:
    """Fractal dimension analysis result."""

    correlation_dimension: Optional[float] = None
    hurst_exponent: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.correlation_dimension is not None:
            d["correlation_dimension"] = self.correlation_dimension
        if self.hurst_exponent is not None:
            d["hurst_exponent"] = self.hurst_exponent
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class DMDResult:
    """Dynamic Mode Decomposition result."""

    modes: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None
    frequencies: Optional[np.ndarray] = None
    growth_rates: Optional[np.ndarray] = None
    amplitudes: Optional[np.ndarray] = None
    n_modes: int = 0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"n_modes": self.n_modes}
        if self.eigenvalues is not None:
            d["eigenvalues"] = [
                {"real": float(e.real), "imag": float(e.imag)} for e in self.eigenvalues
            ]
        if self.frequencies is not None:
            d["frequencies"] = self.frequencies.tolist()
        if self.growth_rates is not None:
            d["growth_rates"] = self.growth_rates.tolist()
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class ChaosResult:
    """Comprehensive chaos detection result."""

    is_chaotic: bool = False
    max_lyapunov: Optional[float] = None
    correlation_dimension: Optional[float] = None
    hurst_exponent: Optional[float] = None
    entropy_rate: Optional[float] = None
    classification: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "is_chaotic": self.is_chaotic,
            "classification": self.classification,
        }
        if self.max_lyapunov is not None:
            d["max_lyapunov"] = self.max_lyapunov
        if self.correlation_dimension is not None:
            d["correlation_dimension"] = self.correlation_dimension
        if self.hurst_exponent is not None:
            d["hurst_exponent"] = self.hurst_exponent
        if self.entropy_rate is not None:
            d["entropy_rate"] = self.entropy_rate
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class DynamicsAnalysisResult:
    """Complete dynamics analysis result."""

    lyapunov: Optional[LyapunovResult] = None
    fractal: Optional[FractalResult] = None
    dmd: Optional[DMDResult] = None
    chaos: Optional[ChaosResult] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.lyapunov:
            d["lyapunov"] = self.lyapunov.to_dict()
        if self.fractal:
            d["fractal"] = self.fractal.to_dict()
        if self.dmd:
            d["dmd"] = self.dmd.to_dict()
        if self.chaos:
            d["chaos"] = self.chaos.to_dict()
        if self.description:
            d["description"] = self.description
        return d


class DynamicsAnalyzer:
    """Dynamical systems analysis — chaos detection and modal decomposition.

    Uses nolds for Lyapunov exponents and fractal dimensions,
    pydmd for Dynamic Mode Decomposition.
    Falls back to numpy-based analysis when unavailable.
    """

    def lyapunov_exponents(
        self, time_series: np.ndarray, min_tsep: int = 10
    ) -> LyapunovResult:
        """Compute Lyapunov exponents from time series.

        Args:
            time_series: 1D array of scalar observations
            min_tsep: Minimum temporal separation for vectors
        """
        ts = np.asarray(time_series).flatten()
        if len(ts) < 50:
            return LyapunovResult(
                description="Time series too short for Lyapunov estimation (need >50 points)",
            )

        try:
            import nolds

            lyap_max = nolds.lyap_r(ts, min_tsep=min_tsep, emb_dim=2)
            is_chaotic = lyap_max > 0

            return LyapunovResult(
                max_lyapunov=float(lyap_max),
                is_chaotic=is_chaotic,
                description=(
                    f"λ_max = {lyap_max:.4f} → {'CHAOTIC' if is_chaotic else 'REGULAR'} dynamics"
                ),
            )
        except ImportError:
            return self._fallback_lyapunov(ts)

    def correlation_dimension(self, time_series: np.ndarray) -> FractalResult:
        """Compute correlation dimension (fractal dimension of attractor).

        Args:
            time_series: 1D array of scalar observations
        """
        ts = np.asarray(time_series).flatten()
        if len(ts) < 100:
            return FractalResult(
                description="Time series too short for correlation dimension (need >100 points)",
            )

        try:
            import nolds

            corr_dim = nolds.corr_dim(ts, emb_dim=5)
            return FractalResult(
                correlation_dimension=float(corr_dim),
                description=f"D₂ = {corr_dim:.4f} (correlation dimension)",
            )
        except ImportError:
            return self._fallback_correlation_dimension(ts)

    def hurst_exponent(self, time_series: np.ndarray) -> FractalResult:
        """Compute Hurst exponent (long-range dependence).

        H < 0.5: mean-reverting (anti-persistent)
        H = 0.5: random walk (Brownian motion)
        H > 0.5: trending (persistent)

        Args:
            time_series: 1D array of scalar observations
        """
        ts = np.asarray(time_series).flatten()
        if len(ts) < 100:
            return FractalResult(
                description="Time series too short for Hurst exponent (need >100 points)",
            )

        try:
            import nolds

            hurst = nolds.hurst_rs(ts)
            if hurst < 0.5:
                behavior = "mean-reverting (anti-persistent)"
            elif hurst > 0.5:
                behavior = "trending (persistent)"
            else:
                behavior = "random walk"

            return FractalResult(
                hurst_exponent=float(hurst),
                description=f"H = {hurst:.4f} → {behavior}",
            )
        except ImportError:
            return self._fallback_hurst(ts)

    def dmd_analysis(self, time_series_data: np.ndarray, n_modes: int = 5) -> DMDResult:
        """Dynamic Mode Decomposition — extract coherent spatiotemporal modes.

        Args:
            time_series_data: (n_features, n_timesteps) array
            n_modes: Number of DMD modes to extract
        """
        data = np.asarray(time_series_data)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 2 * n_modes:
            return DMDResult(
                description="Insufficient time steps for DMD analysis",
            )

        try:
            from pydmd import DMD

            dmd = DMD(svd_rank=n_modes)
            dmd.fit(data)

            eigs = dmd.eigs
            modes = dmd.modes
            amplitudes = dmd.amplitudes

            freqs = np.angle(eigs) / (2 * np.pi)
            growth = np.log(np.abs(eigs))

            return DMDResult(
                modes=modes,
                eigenvalues=eigs,
                frequencies=freqs,
                growth_rates=growth,
                amplitudes=amplitudes,
                n_modes=len(eigs),
                description=(
                    f"DMD: {len(eigs)} modes extracted, "
                    f"dominant frequency = {float(freqs[np.argmax(np.abs(amplitudes))]):.4f}"
                ),
            )
        except ImportError:
            return self._fallback_dmd(data, n_modes)

    def detect_chaos(self, time_series: np.ndarray) -> ChaosResult:
        """Comprehensive chaos detection from time series.

        Combines Lyapunov exponents, correlation dimension,
        and Hurst exponent for a robust classification.
        """
        ts = np.asarray(time_series).flatten()
        if len(ts) < 50:
            return ChaosResult(
                classification="insufficient_data",
                description="Need at least 50 data points for chaos detection",
            )

        lyap = self.lyapunov_exponents(ts)
        fractal = self.correlation_dimension(ts)
        hurst = self.hurst_exponent(ts)

        is_chaotic = False
        if lyap.is_chaotic is not None:
            is_chaotic = lyap.is_chaotic

        if is_chaotic:
            classification = "chaotic"
        elif lyap.max_lyapunov is not None and lyap.max_lyapunov < -0.1:
            classification = "periodic"
        elif hurst.hurst_exponent is not None and hurst.hurst_exponent > 0.7:
            classification = "persistent"
        else:
            classification = "quasi-periodic"

        entropy_rate = lyap.max_lyapunov if lyap.max_lyapunov is not None else None

        return ChaosResult(
            is_chaotic=is_chaotic,
            max_lyapunov=lyap.max_lyapunov,
            correlation_dimension=fractal.correlation_dimension,
            hurst_exponent=hurst.hurst_exponent,
            entropy_rate=entropy_rate,
            classification=classification,
            description=(
                f"Dynamics classification: {classification}. "
                f"λ_max={lyap.max_lyapunov}, "
                f"D₂={fractal.correlation_dimension}, "
                f"H={hurst.hurst_exponent}"
            ),
        )

    def analyze(self, time_series: np.ndarray) -> DynamicsAnalysisResult:
        """Full dynamics analysis pipeline.

        Args:
            time_series: 1D or 2D array of time series data
        """
        ts = np.asarray(time_series)
        is_2d = ts.ndim > 1 and ts.shape[0] > 1

        lyap = self.lyapunov_exponents(ts.flatten() if not is_2d else ts[0])
        fractal = self.correlation_dimension(ts.flatten() if not is_2d else ts[0])
        chaos = self.detect_chaos(ts.flatten() if not is_2d else ts[0])

        dmd = None
        if is_2d:
            dmd = self.dmd_analysis(ts)

        return DynamicsAnalysisResult(
            lyapunov=lyap,
            fractal=fractal,
            dmd=dmd,
            chaos=chaos,
            description=(
                f"Dynamics: {chaos.classification}, " f"λ_max={lyap.max_lyapunov}"
            ),
        )

    def _fallback_lyapunov(self, ts: np.ndarray) -> LyapunovResult:
        """Rough Lyapunov estimate from divergence of nearby trajectories."""
        n = len(ts)
        if n < 20:
            return LyapunovResult(description="Too few points")

        diffs = np.diff(ts)
        variance = np.var(diffs)
        if variance < 1e-15:
            return LyapunovResult(
                max_lyapunov=0.0, is_chaotic=False, description="Constant signal: λ=0"
            )

        mean_abs_diff = np.mean(np.abs(diffs))
        lyap_est = np.log(mean_abs_diff + 1e-15) / n

        return LyapunovResult(
            max_lyapunov=float(lyap_est),
            is_chaotic=lyap_est > 0,
            description=f"λ_max ≈ {lyap_est:.4f} (rough estimate, install nolds for accuracy)",
        )

    def _fallback_correlation_dimension(self, ts: np.ndarray) -> FractalResult:
        """Basic correlation dimension estimate."""
        n = len(ts)
        if n < 20:
            return FractalResult(description="Too few points")

        diffs = np.abs(np.diff(ts))
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        if mean_diff < 1e-15:
            return FractalResult(
                correlation_dimension=0.0, description="Constant signal"
            )

        d2_est = 1.0 + std_diff / (mean_diff + 1e-15)
        d2_est = min(d2_est, 3.0)

        return FractalResult(
            correlation_dimension=float(d2_est),
            description=f"D₂ ≈ {d2_est:.4f} (rough estimate, install nolds for accuracy)",
        )

    def _fallback_hurst(self, ts: np.ndarray) -> FractalResult:
        """Basic Hurst exponent estimate via R/S analysis."""
        n = len(ts)
        if n < 20:
            return FractalResult(description="Too few points")

        mean_ts = np.mean(ts)
        deviations = ts - mean_ts
        cumulative = np.cumsum(deviations)
        R = float(np.max(cumulative) - np.min(cumulative))
        S = float(np.std(ts))

        if S < 1e-15:
            return FractalResult(hurst_exponent=0.5, description="Constant signal")

        rs = R / S
        H_est = np.log(rs) / np.log(n) if n > 1 else 0.5
        H_est = max(0.0, min(1.0, H_est))

        return FractalResult(
            hurst_exponent=float(H_est),
            description=f"H ≈ {H_est:.4f} (rough estimate, install nolds for accuracy)",
        )

    @staticmethod
    def _fallback_dmd(data: np.ndarray, n_modes: int) -> DMDResult:
        """Basic DMD via SVD without pydmd."""
        X = data[:, :-1]
        Y = data[:, 1:]

        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        r = min(n_modes, len(s))
        U_r = U[:, :r]
        s_r = s[:r]
        Vt_r = Vt[:r, :]

        A_tilde = U_r.T @ Y @ Vt_r.T @ np.diag(1.0 / s_r)
        eigs, W = np.linalg.eig(A_tilde)

        freqs = np.angle(eigs) / (2 * np.pi)
        growth = np.log(np.abs(eigs) + 1e-15)

        return DMDResult(
            eigenvalues=eigs,
            frequencies=freqs,
            growth_rates=growth,
            n_modes=r,
            description=f"DMD: {r} modes via SVD (install pydmd for full features)",
        )
