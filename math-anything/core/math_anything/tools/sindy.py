"""Sparse Identification of Nonlinear Dynamics (SINDy) — equation discovery.

Pipeline: time series → SINDy → sparse ODE system → symbolic equations.

Complements DynamicsAnalyzer (chaos detection) and PSRN (symbolic regression):
  DynamicsAnalyzer detects chaos → SINDy discovers governing equations → PSRN validates.

Requires optional dependency: pysindy>=1.7.0
Falls back to numpy-based sparse regression when unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DiscoveredEquation:
    """A single discovered ODE equation."""

    variable_name: str = ""
    coefficients: List[float] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    equation_string: str = ""
    score: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "variable_name": self.variable_name,
        }
        if self.equation_string:
            d["equation"] = self.equation_string
        if self.coefficients:
            d["coefficients"] = self.coefficients
        if self.feature_names:
            d["features"] = self.feature_names
        if self.score is not None:
            d["score"] = self.score
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class SINDyResult:
    """Complete SINDy analysis result."""

    equations: List[DiscoveredEquation] = field(default_factory=list)
    n_features: int = 0
    library_type: str = ""
    optimizer_type: str = ""
    model_score: Optional[float] = None
    complexity: int = 0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "equations": [eq.to_dict() for eq in self.equations],
            "n_features": self.n_features,
            "library_type": self.library_type,
            "optimizer_type": self.optimizer_type,
        }
        if self.model_score is not None:
            d["model_score"] = self.model_score
        if self.complexity > 0:
            d["complexity"] = self.complexity
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class PDEDiscoveredResult:
    """PDE discovery result via weak-form SINDy."""

    equations: List[DiscoveredEquation] = field(default_factory=list)
    spatial_dims: int = 1
    library_type: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "equations": [eq.to_dict() for eq in self.equations],
            "spatial_dims": self.spatial_dims,
            "library_type": self.library_type,
        }
        if self.description:
            d["description"] = self.description
        return d


class SINDyDiscoverer:
    """Sparse Identification of Nonlinear Dynamics.

    Uses PySINDy for equation discovery from time series data.
    Falls back to numpy-based STLSQ when pysindy is unavailable.

    Typical workflow:
        1. DynamicsAnalyzer detects chaotic/persistent dynamics
        2. SINDyDiscoverer discovers the governing equations
        3. PSRN validates the discovered equations against data
    """

    def discover_ode(
        self,
        time_series: np.ndarray,
        dt: float = 1.0,
        poly_order: int = 3,
        threshold: float = 0.1,
        variable_names: Optional[List[str]] = None,
    ) -> SINDyResult:
        """Discover ODE system from time series data.

        Args:
            time_series: (n_timesteps, n_vars) or (n_timesteps,) array
            dt: Time step between samples
            poly_order: Maximum polynomial degree in library
            threshold: Sparsity threshold (higher = sparser)
            variable_names: Names for each variable
        """
        data = np.asarray(time_series)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_vars = data.shape[1]
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(n_vars)]

        if len(data) < 10:
            return SINDyResult(
                library_type=f"polynomial_order_{poly_order}",
                description="Insufficient data for SINDy (need >= 10 time steps)",
            )

        if dt <= 0:
            return SINDyResult(
                library_type=f"polynomial_order_{poly_order}",
                description="dt must be positive for SINDy",
            )

        if threshold < 0:
            return SINDyResult(
                library_type=f"polynomial_order_{poly_order}",
                description="threshold must be non-negative for SINDy",
            )

        try:
            return self._pysindy_ode(data, dt, poly_order, threshold, variable_names)
        except ImportError:
            return self._fallback_ode(data, dt, poly_order, threshold, variable_names)

    def discover_pde(
        self,
        spatiotemporal_data: np.ndarray,
        dt: float = 1.0,
        dx: float = 1.0,
        poly_order: int = 2,
        threshold: float = 0.1,
    ) -> PDEDiscoveredResult:
        """Discover PDE from spatiotemporal data.

        Uses finite differences for spatial derivatives, then applies
        sparse regression on the library of candidate terms.

        Args:
            spatiotemporal_data: (n_time, n_space) array of field values
            dt: Time step
            dx: Spatial step
            poly_order: Maximum polynomial degree
            threshold: Sparsity threshold
        """
        data = np.asarray(spatiotemporal_data)
        if data.ndim != 2:
            return PDEDiscoveredResult(
                description="PDE discovery requires 2D array (n_time x n_space)",
            )

        n_time, n_space = data.shape
        if n_time < 10 or n_space < 5:
            return PDEDiscoveredResult(
                description="Insufficient data for PDE discovery",
            )

        return self._fallback_pde(data, dt, dx, poly_order, threshold)

    def _pysindy_ode(
        self,
        data: np.ndarray,
        dt: float,
        poly_order: int,
        threshold: float,
        variable_names: List[str],
    ) -> SINDyResult:
        """Use PySINDy for ODE discovery."""
        import pysindy as ps

        library = ps.PolynomialLibrary(degree=poly_order, include_interaction=True)
        optimizer = ps.STLSQ(threshold=threshold)

        model = ps.SINDy(
            feature_library=library,
            optimizer=optimizer,
            feature_names=variable_names,
        )

        model.fit(data, t=dt)

        equations = []
        for i, name in enumerate(variable_names):
            coeff = model.coefficients()[i]
            feat_names = model.get_feature_names()

            nonzero_mask = np.abs(coeff) > 1e-10
            active_coeff = coeff[nonzero_mask].tolist()
            active_features = [
                feat_names[j] for j in range(len(feat_names)) if nonzero_mask[j]
            ]

            eq_str = self._format_equation(name, active_coeff, active_features)
            equations.append(
                DiscoveredEquation(
                    variable_name=name,
                    coefficients=active_coeff,
                    feature_names=active_features,
                    equation_string=eq_str,
                    description=f"d{name}/dt = {eq_str}",
                )
            )

        score = None
        try:
            score = float(model.score(data, t=dt))
        except Exception:
            pass

        complexity = int(np.sum(np.abs(model.coefficients()) > 1e-10))

        return SINDyResult(
            equations=equations,
            n_features=len(model.get_feature_names()),
            library_type=f"polynomial_order_{poly_order}",
            optimizer_type=f"STLSQ(threshold={threshold})",
            model_score=score,
            complexity=complexity,
            description=(
                f"SINDy: {len(equations)} equations, {complexity} active terms, "
                f"score={score:.4f}"
                if score is not None
                else f"SINDy: {len(equations)} equations, {complexity} active terms"
            ),
        )

    def _fallback_ode(
        self,
        data: np.ndarray,
        dt: float,
        poly_order: int,
        threshold: float,
        variable_names: List[str],
    ) -> SINDyResult:
        """Fallback: numpy-based STLSQ for ODE discovery."""
        n_time, n_vars = data.shape

        derivatives = np.gradient(data, dt, axis=0)

        features, feat_names = self._build_library(data, variable_names, poly_order)

        coefficients = self._stlsq(features, derivatives, threshold)

        equations = []
        for i, name in enumerate(variable_names):
            coeff = coefficients[i]
            nonzero_mask = np.abs(coeff) > 1e-10
            active_coeff = coeff[nonzero_mask].tolist()
            active_features = [
                feat_names[j] for j in range(len(feat_names)) if nonzero_mask[j]
            ]

            eq_str = self._format_equation(name, active_coeff, active_features)
            equations.append(
                DiscoveredEquation(
                    variable_name=name,
                    coefficients=active_coeff,
                    feature_names=active_features,
                    equation_string=eq_str,
                    description=f"d{name}/dt = {eq_str}",
                )
            )

        complexity = int(np.sum(np.abs(coefficients) > 1e-10))

        residual = derivatives - features @ coefficients.T
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((derivatives - np.mean(derivatives, axis=0)) ** 2)
        score = float(1.0 - ss_res / (ss_tot + 1e-15))

        return SINDyResult(
            equations=equations,
            n_features=len(feat_names),
            library_type=f"polynomial_order_{poly_order}",
            optimizer_type=f"STLSQ_fallback(threshold={threshold})",
            model_score=score,
            complexity=complexity,
            description=(
                f"SINDy (fallback): {len(equations)} equations, {complexity} active terms, "
                f"score={score:.4f} (install pysindy for better results)"
            ),
        )

    def _fallback_pde(
        self,
        data: np.ndarray,
        dt: float,
        dx: float,
        poly_order: int,
        threshold: float,
    ) -> PDEDiscoveredResult:
        """Fallback PDE discovery via finite differences + sparse regression."""
        n_time, n_space = data.shape

        u_t = np.gradient(data, dt, axis=0)

        u_x = np.gradient(data, dx, axis=1)
        u_xx = np.gradient(u_x, dx, axis=1)

        u_flat = data.flatten()
        u_t_flat = u_t.flatten()
        u_x_flat = u_x.flatten()
        u_xx_flat = u_xx.flatten()

        theta = np.column_stack(
            [
                np.ones_like(u_flat),
                u_flat,
                u_flat**2,
                u_flat**3,
                u_x_flat,
                u_xx_flat,
                u_flat * u_x_flat,
                u_flat * u_xx_flat,
            ]
        )

        feat_names = ["1", "u", "u²", "u³", "u_x", "u_xx", "u·u_x", "u·u_xx"]

        coeff = self._stlsq_single(theta, u_t_flat, threshold)

        nonzero_mask = np.abs(coeff) > 1e-10
        active_coeff = coeff[nonzero_mask].tolist()
        active_features = [
            feat_names[j] for j in range(len(feat_names)) if nonzero_mask[j]
        ]

        eq_str = self._format_equation("u_t", active_coeff, active_features)

        return PDEDiscoveredResult(
            equations=[
                DiscoveredEquation(
                    variable_name="u_t",
                    coefficients=active_coeff,
                    feature_names=active_features,
                    equation_string=eq_str,
                    description=f"∂u/∂t = {eq_str}",
                )
            ],
            spatial_dims=1,
            library_type=f"polynomial_order_{poly_order}+derivatives",
            description=f"PDE discovery (1D spatial, fallback, install pysindy for PDELibrary)",
        )

    @staticmethod
    def _build_library(
        data: np.ndarray,
        var_names: List[str],
        poly_order: int,
    ) -> Tuple[np.ndarray, List[str]]:
        """Build polynomial feature library Θ(X).

        Includes constant, linear, and polynomial terms up to poly_order.
        """
        n_time, n_vars = data.shape
        features = [np.ones((n_time, 1))]
        names = ["1"]

        for i, name in enumerate(var_names):
            features.append(data[:, i : i + 1])
            names.append(name)

        if poly_order >= 2:
            for i in range(n_vars):
                for j in range(i, n_vars):
                    features.append((data[:, i] * data[:, j]).reshape(-1, 1))
                    if i == j:
                        names.append(f"{var_names[i]}²")
                    else:
                        names.append(f"{var_names[i]}·{var_names[j]}")

        if poly_order >= 3:
            for i in range(n_vars):
                features.append((data[:, i] ** 3).reshape(-1, 1))
                names.append(f"{var_names[i]}³")
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        features.append((data[:, i] ** 2 * data[:, j]).reshape(-1, 1))
                        names.append(f"{var_names[i]}²·{var_names[j]}")

        return np.hstack(features), names

    @staticmethod
    def _stlsq(
        features: np.ndarray,
        targets: np.ndarray,
        threshold: float,
        max_iter: int = 20,
    ) -> np.ndarray:
        """Sequential Thresholded Least Squares (STLSQ).

        Iteratively fits least squares and removes small coefficients.
        """
        n_vars = targets.shape[1]
        n_features = features.shape[1]
        coefficients = np.zeros((n_vars, n_features))

        for i in range(n_vars):
            coefficients[i] = SINDyDiscoverer._stlsq_single(
                features, targets[:, i], threshold, max_iter
            )

        return coefficients

    @staticmethod
    def _stlsq_single(
        features: np.ndarray,
        target: np.ndarray,
        threshold: float,
        max_iter: int = 20,
    ) -> np.ndarray:
        """STLSQ for a single variable."""
        n_features = features.shape[1]
        active = np.ones(n_features, dtype=bool)
        coeff = np.zeros(n_features)

        for _ in range(max_iter):
            if not np.any(active):
                coeff = np.zeros(n_features)
                break

            try:
                c, _, _, _ = np.linalg.lstsq(features[:, active], target, rcond=None)
            except np.linalg.LinAlgError:
                break

            coeff_active = np.zeros(n_features)
            coeff_active[active] = c

            newly_active = np.abs(coeff_active) > threshold
            if np.array_equal(newly_active, active):
                coeff = coeff_active
                break

            if not np.any(newly_active):
                coeff = np.zeros(n_features)
                break

            active = newly_active
            coeff = coeff_active

        return coeff

    @staticmethod
    def _format_equation(
        lhs: str,
        coefficients: List[float],
        feature_names: List[str],
    ) -> str:
        """Format discovered equation as a readable string."""
        if not coefficients:
            return "0"

        terms = []
        for c, f in zip(coefficients, feature_names):
            if abs(c) < 1e-10:
                continue

            c_str = f"{c:.4f}"
            if f == "1":
                terms.append(c_str)
            else:
                terms.append(f"{c_str}·{f}")

        if not terms:
            return "0"

        result = terms[0]
        for t in terms[1:]:
            if t.startswith("-"):
                result += f" - {t[1:]}"
            else:
                result += f" + {t}"

        return result
