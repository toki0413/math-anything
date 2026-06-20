"""数值分析 — 收敛性与误差估计结构。

误差估计、收敛阶、先验/后验估计、超收敛。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis_stability import NumericalAnalysis
from .base import StructureFamily, StructureMetadata
from .properties import StructuralInvariant


@dataclass
class ErrorEstimate(NumericalAnalysis):
    """Error estimation for numerical approximations.

    ∥u - u_h∥: the fundamental quantity to be controlled.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Error Estimate",
            canonical_form="∥u - u_h∥ ≤ C(u) h^p  or  ∥u - u_h∥ ≤ C η_h",
            description="Bound on the difference between exact and discrete solutions",
        )
    )
    norm_type: str = "L2"
    error_bound: float = 0.0

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="error_converges_to_zero",
                expression="∥u - u_h∥ → 0 as h → 0 (for consistent stable schemes)",
                theorem="Lax Equivalence Theorem",
                affected_quantities=["discretization_error", "mesh_size"],
            ),
        ]

    def compute_error_norm(self, u_exact: np.ndarray, u_h: np.ndarray) -> float:
        """计算误差范数 ||u - u_h||."""
        diff = u_exact - u_h
        if self.norm_type == "L2":
            return float(np.linalg.norm(diff))
        elif self.norm_type == "Linf":
            return float(np.max(np.abs(diff)))
        elif self.norm_type == "H1":
            # 简化：仅返回 L2 部分，完整 H1 需要梯度信息
            return float(np.linalg.norm(diff))
        return float(np.linalg.norm(diff))


@dataclass
class APrioriEstimate(ErrorEstimate):
    """A priori error estimate: bound in terms of exact solution regularity.

    ∥u - u_h∥ ≤ C(u) h^p

    Predicts convergence rate before computation.
    Requires knowledge of exact solution's Sobolev regularity.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="A Priori Error Estimate",
            canonical_form="∥u - u_h∥ ≤ C(u) h^p",
            description="Error bound using exact solution regularity, known before computation",
        )
    )
    polynomial_degree: int = 1
    regularity_index: int = 2
    mesh_dependent_constant: float = 0.0

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="ceas_lemma",
                    expression=f"∥u - u_h∥ ≤ C ∥u∥_{{H^{self.regularity_index}}} h^{self.polynomial_degree}",
                    theorem="Céa's Lemma (finite elements)",
                    affected_quantities=["error", "mesh_size", "regularity"],
                ),
                StructuralInvariant(
                    name="bramble_hilbert",
                    expression="∥u - I_h u∥ ≤ C h^{min(p+1, k)} abs:uabs:_{W^{k,p}}",
                    theorem="Bramble-Hilbert Lemma (interpolation error bound)",
                    affected_quantities=["interpolation_error", "polynomial_degree", "regularity"],
                ),
            ]
        )
        return invariants

    def fem_h1_estimate(self, h: float, p: int = None, u_regularity: int = None) -> float:
        """FEM H1 误差: C * h^min(p, u_regularity) * |u|_{H^{min(p,u_regularity)+1}}."""
        p = p if p is not None else self.polynomial_degree
        u_reg = u_regularity if u_regularity is not None else self.regularity_index
        return h ** min(p, u_reg)

    def fem_l2_estimate(self, h: float, p: int = None, u_regularity: int = None) -> float:
        """FEM L2 误差 (Aubin-Nitsche): C * h^{min(p, u_regularity) + 1}."""
        p = p if p is not None else self.polynomial_degree
        u_reg = u_regularity if u_regularity is not None else self.regularity_index
        return h ** (min(p, u_reg) + 1)


@dataclass
class APosterioriEstimate(ErrorEstimate):
    """A posteriori error estimate: computable bound from numerical solution.

    ∥u - u_h∥ ≤ C η_h(u_h)

    Uses the computed discrete solution to estimate error.
    Enables adaptive mesh refinement (AMR).
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="A Posteriori Error Estimate",
            canonical_form="∥u - u_h∥ ≤ C η_h(u_h)",
            description="Computable error bound from discrete solution, enables adaptivity",
        )
    )
    error_indicator_type: str = ""
    reliability_constant: float = 0.0
    efficiency_constant: float = 0.0
    elementwise: bool = True

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="reliability",
                    expression=f"∥u - u_h∥ ≤ {self.reliability_constant} η_h (upper bound)",
                    theorem="Reliability of error estimator (global upper bound)",
                    affected_quantities=["error", "error_indicator"],
                ),
                StructuralInvariant(
                    name="efficiency",
                    expression=f"η_h ≤ {self.efficiency_constant} ∥u - u_h∥ (lower bound up to data oscillation)",
                    theorem="Efficiency of error estimator (local lower bound)",
                    affected_quantities=["error_indicator", "error"],
                ),
            ]
        )
        return invariants

    def residual_estimate(self, residual_norm: float, condition_number: float) -> float:
        """误差 ≤ condition_number * ||residual||."""
        return condition_number * residual_norm

    def element_error_indicators(
        self, residual_per_element: np.ndarray, mesh_size_per_element: np.ndarray
    ) -> np.ndarray:
        """计算单元误差指示子 η_K = h_K * ||R_K||."""
        return mesh_size_per_element * residual_per_element


@dataclass
class SuperconvergenceEstimate(ErrorEstimate):
    """Superconvergence: higher-order accuracy at special points.

    At certain points (Gauss points, mesh nodes for some schemes),
    the error converges at one order higher: O(h^{p+1}) instead of O(h^p).
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Superconvergence Estimate",
            canonical_form="∥u - u_h∥_special_points = O(h^{p+1})",
            description="Higher-order convergence at special points (Gauss, nodes)",
        )
    )
    superconvergent_points: str = ""
    base_order: int = 1
    superconvergent_order: int = 2

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.append(
            StructuralInvariant(
                name="superconvergence_property",
                expression=f"∥u - u_h∥_SP = O(h^{self.superconvergent_order}) at superconvergent points",
                theorem="Superconvergence theory (Zienkiewicz-Zhu, Babuška)",
                affected_quantities=["error_at_special_points", "mesh_size"],
            ),
        )
        return invariants

    def compute_gain(self) -> float:
        """计算超收敛增益."""
        return float(self.superconvergent_order - self.base_order)

    def error_at_superconvergent_points(self, h: float, constant: float = 1.0) -> float:
        """估计超收敛点处误差 C * h^{p+δ}."""
        return constant * h**self.superconvergent_order

    def error_at_general_points(self, h: float, constant: float = 1.0) -> float:
        """估计一般点处误差 C * h^p."""
        return constant * h**self.base_order


@dataclass
class ConvergenceOrder(NumericalAnalysis):
    """Convergence order: rate at which discrete solution approaches exact.

    ∥e_h∥ ≍ h^p (polynomial) or exp(-αN) (exponential).
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Convergence Order",
            canonical_form="∥e_h∥ ~ h^p  or  exp(-αN)",
            description="Asymptotic rate of error reduction under mesh/DOF refinement",
        )
    )
    order: float = 0.0
    convergence_type: str = ""

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="limit_zero",
                expression="∥e_h∥ → 0 as h → 0 (for stable consistent schemes)",
                theorem="Lax Equivalence Theorem (convergence consequence)",
                affected_quantities=["error", "mesh_size"],
            ),
        ]

    def from_errors(self, errors: list[float], mesh_sizes: list[float]) -> float:
        """从 log-log 回归估计收敛阶 p."""
        if len(errors) < 2:
            return 0.0
        log_e = np.log(np.array(errors))
        log_h = np.log(np.array(mesh_sizes))
        p, _ = np.polyfit(log_h, log_e, 1)
        return float(p)

    def from_solutions(self, u_h: np.ndarray, u_2h: np.ndarray, u_4h: np.ndarray = None) -> float:
        """用两个/三个网格层级的解估计收敛阶."""
        e_h = np.linalg.norm(u_h - u_2h)
        if u_4h is not None:
            e_2h = np.linalg.norm(u_2h - u_4h)
            if e_2h > 0 and e_h > 0:
                return float(np.log2(e_2h / e_h))
        return 0.0


@dataclass
class AlgebraicConvergence(ConvergenceOrder):
    """Algebraic (polynomial) convergence: ∥e_h∥ ≤ C h^p.

    Typical of finite difference, finite volume, and low-order finite element methods.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Algebraic Convergence",
            canonical_form="∥e_h∥ ≤ C h^p",
            description="Polynomial error reduction with mesh refinement",
        )
    )
    polynomial_order: int = 1

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.append(
            StructuralInvariant(
                name="algebraic_rate",
                expression=f"∥e_h∥ = O(h^{self.polynomial_order})",
                theorem="A priori error estimate (Céa, Bramble-Hilbert)",
                affected_quantities=["error", "mesh_size", "convergence_rate"],
            ),
        )
        return invariants

    def error_bound(self, h: float, constant: float = 1.0) -> float:
        """计算代数收敛误差上界 C * h^p."""
        return constant * h**self.polynomial_order

    def estimate_order_from_ratios(self, error_ratios: list[float], refinement_factor: float = 2.0) -> float:
        """从误差比估计收敛阶: p = log(error_ratio) / log(refinement_factor)."""
        if not error_ratios:
            return 0.0
        avg_ratio = sum(error_ratios) / len(error_ratios)
        if avg_ratio <= 0 or refinement_factor <= 1:
            return 0.0
        return float(np.log(avg_ratio) / np.log(refinement_factor))


@dataclass
class ExponentialConvergence(ConvergenceOrder):
    """Exponential (spectral) convergence: ∥e_N∥ ≤ C exp(-α N^β).

    Typical of spectral methods (Chebyshev, Fourier) for smooth solutions.
    Error decays exponentially with number of degrees of freedom.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Exponential Convergence",
            canonical_form="∥e_N∥ ≤ C exp(-α N)",
            description="Spectral accuracy: exponential error reduction with DOFs",
        )
    )
    exponential_rate: float = 0.0
    dof_exponent: float = 1.0
    method_type: str = ""

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.append(
            StructuralInvariant(
                name="exponential_rate",
                expression=f"∥e_N∥ ≤ C exp(-{self.exponential_rate} N^{self.dof_exponent})",
                theorem="Spectral convergence (approximation theory for smooth functions)",
                affected_quantities=["error", "degrees_of_freedom", "convergence_rate"],
            ),
        )
        return invariants

    def error_bound(self, n_dof: int, constant: float = 1.0) -> float:
        """计算指数收敛误差上界 C * exp(-α * N^β)."""
        return constant * np.exp(-self.exponential_rate * n_dof**self.dof_exponent)

    def estimate_rate_from_errors(self, errors: list[float], dof_counts: list[int]) -> float:
        """从误差和自由度数估计指数收敛率 α."""
        if len(errors) < 2 or len(dof_counts) < 2:
            return 0.0
        log_e = np.log(np.array(errors))
        n_beta = np.array(dof_counts) ** self.dof_exponent
        rate, _ = np.polyfit(n_beta, -log_e, 1)
        return float(rate)


@dataclass
class SublinearConvergence(ConvergenceOrder):
    """Sublinear convergence for non-smooth problems.

    Error decays slower than any polynomial (e.g., 1/abs:log habs:, iterated logarithms).
    Occurs for problems with singularities, rough coefficients, or irregular boundaries.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Sublinear Convergence",
            canonical_form="∥e_h∥ = O(1 / abs:log habs:^β)",
            description="Slower than polynomial convergence for non-smooth problems",
        )
    )
    log_exponent: float = 1.0
    non_smoothness_source: str = ""

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.append(
            StructuralInvariant(
                name="sublinear_rate",
                expression=f"∥e_h∥ = O(abs:log habs:^{{-{self.log_exponent}}})",
                theorem="Convergence for non-smooth problems (Grisvard, Dauge)",
                affected_quantities=["error", "mesh_size"],
            ),
        )
        return invariants

    def error_bound(self, h: float, constant: float = 1.0) -> float:
        """计算次线性收敛误差上界 C / |log(h)|^β."""
        if h <= 0 or h >= 1:
            return float("inf")
        return constant / abs(np.log(h)) ** self.log_exponent

    def estimate_log_exponent(self, errors: list[float], mesh_sizes: list[float]) -> float:
        """从数值实验估计对数指数 β."""
        if len(errors) < 2:
            return 0.0
        log_inv_h = np.array([abs(np.log(h)) for h in mesh_sizes])
        log_inv_error = np.array([1.0 / e for e in errors if e > 0])
        if len(log_inv_error) < 2:
            return 0.0
        # log(1/error) ~ β * log(|log(h)|)
        log_log_h = np.log(log_inv_h)
        log_inv_e = np.log(log_inv_error[: len(log_log_h)])
        if len(log_log_h) < 2:
            return 0.0
        beta, _ = np.polyfit(log_log_h, log_inv_e, 1)
        return float(beta)
