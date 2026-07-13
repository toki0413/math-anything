"""Convergence-related morphisms for numerical methods.

Each morphism describes a transformation along the continuum→discrete→
convergence chain. Extends the base ContinuumToDiscrete morphism with
analysis-specific attributes: consistency, stability, convergence, CFL,
and superconvergence.

The Lax equivalence theorem provides the central organizing principle:
  consistency + stability ⇔ convergence (for linear well-posed problems)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ContinuumToDiscrete, Morphism, MorphismCategory

# ──────────────────────────────────────────────────────────────────
# Mesh refinement
# ──────────────────────────────────────────────────────────────────


@dataclass
class MeshRefinementMorphism(ContinuumToDiscrete):
    """Mesh refinement: h → h/2, preserving convergence order.

    Systematic mesh refinement is a morphism from coarse-grid solution
    to fine-grid solution. The convergence order of the discretization
    is preserved under uniform refinement.

    For a method of order p:
      ||u_h - u_exact|| ≤ C h^p
      ||u_{h/2} - u_exact|| ≤ C (h/2)^p
      → error ratio = 2^{-p}
    """

    name: str = "mesh_refinement"
    source_type: str = "DiscreteSolution_h"
    target_type: str = "DiscreteSolution_h/2"
    category: str = MorphismCategory.DISCRETIZATION

    refinement_factor: float = 2.0
    convergence_order: float = 2.0

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "convergence_order",
            "asymptotic_error_estimate",
            "consistency_of_scheme",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "coarse_grid_details",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "fine_grid_resolution",
            "increased_computational_cost (~ (1/h)^d)",
        ]
    )
    kernel_description: str = "Sub-h scales: frequencies between Nyquist(h) and Nyquist(h/2)"

    @property
    def mathematical_form(self) -> str:
        return (
            f"h → h/{self.refinement_factor}; "
            f"error ratio ≈ {self.refinement_factor}^(-{self.convergence_order})"
            f" = {self.refinement_factor ** (-self.convergence_order):.4f}"
        )

    def compute_convergence_order(self, errors: list[float], mesh_sizes: list[float]) -> float:
        """从 log-log 回归计算收敛阶."""
        if len(errors) < 2:
            return 0.0
        log_e = np.log(np.array(errors))
        log_h = np.log(np.array(mesh_sizes))
        order, _ = np.polyfit(log_h, log_e, 1)
        return float(order)


# ──────────────────────────────────────────────────────────────────
# Consistency
# ──────────────────────────────────────────────────────────────────


@dataclass
class ConsistencyMorphism(Morphism):
    """Consistency: truncation error T(h) → 0 as h → 0.

    A discrete scheme L_h u_h = f_h is consistent with the continuous
    PDE L u = f if for sufficiently smooth u:
      ||L_h u - L u|| → 0  as h → 0

    Equivalently: the truncation error T(h,u) = L_h(R_h u) - R_h(L u)
    tends to zero, where R_h is the restriction operator.

    Order of consistency p:
      ||T(h)|| ≤ C h^p  for smooth u
    """

    name: str = "consistency"
    source_type: str = "ContinuousPDE"
    target_type: str = "DiscretizedPDE"
    category: str = MorphismCategory.DISCRETIZATION

    consistency_order: float = 2.0
    norm_type: str = "L2"  # "L2", "Linf", "H1"
    is_consistent: bool = True

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "pde_structure_at_leading_order",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "exact_pde_satisfaction",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "truncation_error ~ O(h^p)",
        ]
    )
    kernel_description: str = "Higher-order derivatives in the Taylor expansion beyond the stencil approximation"

    @property
    def mathematical_form(self) -> str:
        return f"T(h, u) = L_h u - (L u)_h; ||T(h)||_{self.norm_type} = O(h^{self.consistency_order})"

    def truncation_error_bound(self, h: float, constant: float = 1.0) -> float:
        """计算截断误差上界 C * h^p."""
        return constant * h**self.consistency_order  # type: ignore[no-any-return]


# ──────────────────────────────────────────────────────────────────
# Stability
# ──────────────────────────────────────────────────────────────────


@dataclass
class StabilityMorphism(Morphism):
    """Stability: uniform bound ||u_h|| ≤ C ||f_h|| independent of h.

    A discrete scheme is stable if the inverse operator L_h^{-1} is
    uniformly bounded:
      ||u_h|| ≤ C_s ||f_h||   for all h < h_0

    For time-dependent problems (Lax-Richtmyer):
      ||G(Δt, k)^n|| ≤ C   for all n Δt ≤ T, all k
    where G is the amplification matrix.

    Von Neumann stability: |G(ξ)| ≤ 1 + O(Δt) for all Fourier modes ξ.
    """

    name: str = "stability"
    source_type: str = "DiscreteOperator"
    target_type: str = "StableDiscreteSolution"
    category: str = MorphismCategory.DISCRETIZATION

    stability_analysis: str = "von_neumann"  # "von_neumann", "energy_method", "matrix"
    amplification_factor_bound: float = 1.0
    is_stable: bool = True
    condition_number_bound: float | None = None

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "boundedness_of_discrete_solution",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "unconditional_solution",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "uniform_bound_on_inverse_operator",
            "stability_constraint_on_parameters",
        ]
    )
    kernel_description: str = "Unstable modes: eigencomponents with |λ| > 1 in the amplification matrix"

    @property
    def mathematical_form(self) -> str:
        if self.condition_number_bound is not None:
            return f"||L_h^(-1)|| ≤ C_s (independent of h); κ(L_h) ≤ {self.condition_number_bound}"
        return f"||u_h|| ≤ C_s ||f_h||; |G(ξ)| ≤ {self.amplification_factor_bound} (Von Neumann analysis)"

    def check_von_neumann(self, amplification_factors: list[complex]) -> bool:
        """检查 Von Neumann 稳定性条件 |G(ξ)| ≤ 1 + ε."""
        for g in amplification_factors:
            if abs(g) > self.amplification_factor_bound + 1e-10:
                return False
        return True

    def stability_ratio(self, solution_norm: float, rhs_norm: float) -> float:
        """计算稳定性比 ||u_h|| / ||f_h||."""
        if rhs_norm == 0:
            return float("inf")
        return solution_norm / rhs_norm


# ──────────────────────────────────────────────────────────────────
# Convergence (Lax theorem)
# ──────────────────────────────────────────────────────────────────


@dataclass
class ConvergenceMorphism(Morphism):
    """Convergence: u_h → u_exact as h → 0.

    Combines consistency and stability via the Lax equivalence theorem:
    For a linear well-posed initial value problem:
      consistency + stability ⇔ convergence

    Error estimate:
      ||u_h - u_exact|| ≤ ||L_h^{-1}|| · ||T(h)||
                        ≤ C_s · C h^p = C_conv h^p

    Where the convergence order equals the consistency order when
    the scheme is stable.
    """

    name: str = "convergence"
    source_type: str = "DiscreteSolution"
    target_type: str = "ExactSolution (limit)"
    category: str = MorphismCategory.DISCRETIZATION

    convergence_order: float = 2.0
    consistency_morphism: str = "consistency"
    stability_morphism: str = "stability"
    norm_type: str = "L2"
    error_constant: float | None = None

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "convergence_order_equals_consistency_order (Lax theorem)",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "exact_solution_at_finite_h",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "discretization_error ~ O(h^p)",
            "asymptotic_error_estimate",
        ]
    )
    kernel_description: str = "Finite-h truncation error; kernel vanishes as h → 0"

    @property
    def mathematical_form(self) -> str:
        const_str = f"C = {self.error_constant}, " if self.error_constant is not None else ""
        return f"||u_h - u||_{self.norm_type} ≤ {const_str}C h^{self.convergence_order} (Lax equivalence)"

    def richardson_extrapolation(self, f_h: float, f_2h: float, order: int = 2) -> float:
        """Richardson 外推：组合两个网格层级提高精度."""
        return f_h + (f_h - f_2h) / (2**order - 1)  # type: ignore[no-any-return]

    def a_posteriori_error_estimate(self, residual: np.ndarray, operator: np.ndarray | None = None) -> float:
        """从残差 r = Au - f 估计误差."""
        return float(np.linalg.norm(residual))


# ──────────────────────────────────────────────────────────────────
# CFL condition
# ──────────────────────────────────────────────────────────────────


@dataclass
class CFLConditionMorphism(Morphism):
    """CFL condition: time step constraint for explicit schemes.

    The Courant-Friedrichs-Lewy condition is a necessary condition for
    convergence of explicit time-marching schemes for hyperbolic PDEs:

      C = a Δt / Δx ≤ C_max

    where a is the characteristic wave speed, Δt is the time step,
    Δx is the spatial grid spacing, and C_max is scheme-dependent
    (typically 1 for upwind, 1/√d for central differences).

    Physical interpretation: the numerical domain of dependence must
    contain the physical domain of dependence.
    """

    name: str = "cfl_condition"
    source_type: str = "UnstableTimeStep"
    target_type: str = "StableTimeStep"
    category: str = MorphismCategory.RESTRICTION

    cfl_number: float = 0.5
    max_cfl: float = 1.0
    wave_speed: float | None = None
    is_satisfied: bool = True
    scheme_type: str = "explicit"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "wave_propagation_speed",
            "spatial_accuracy",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "large_timestep_efficiency",
            "unconditional_stability",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "timestep_upper_bound Δt ≤ C_max Δx / a",
        ]
    )
    kernel_description: str = (
        "Δt > C_max Δx / a: numerical domain of dependence smaller than physical domain of dependence"
    )

    @property
    def mathematical_form(self) -> str:
        wave_str = f"a={self.wave_speed}, " if self.wave_speed is not None else ""
        return f"C = {wave_str}a Δt/Δx = {self.cfl_number} ≤ {self.max_cfl}; Δt_max = {self.max_cfl} Δx/a"

    def compute_cfl(self, velocity: float, dx: float, dt: float) -> float:
        """计算 CFL 数: |v|*dt/dx."""
        if dx == 0:
            return float("inf")
        return abs(velocity) * dt / dx

    def max_stable_timestep(self, velocity: float, dx: float, cfl_target: float = 0.5) -> float:
        """计算给定 CFL 目标的最大稳定时间步."""
        if abs(velocity) == 0:
            return float("inf")
        return cfl_target * dx / abs(velocity)


# ──────────────────────────────────────────────────────────────────
# Superconvergence
# ──────────────────────────────────────────────────────────────────


@dataclass
class SuperconvergenceMorphism(Morphism):
    """Superconvergence: higher-order accuracy at special points.

    In certain numerical methods, the error at special points
    converges at a higher rate than the global error:

      ||u_h(x*) - u(x*)|| = O(h^{p+δ}) with δ > 0

    where δ is the superconvergence gain.

    Examples:
      - FEM: O(h^{2p}) at Gauss-Legendre points (for degree p elements)
      - FD: O(h^4) at midpoints when scheme is O(h^2) globally
      - DG: O(h^{2p+1}) at Radau points for outflow flux
      - Spline collocation: O(h^{2k}) at knots for degree k

    Mechanism: cancellation of leading-order error terms at
    special evaluation points due to symmetry / orthogonality.
    """

    name: str = "superconvergence"
    source_type: str = "DiscreteSolution_global_error"
    target_type: str = "DiscreteSolution_superconvergent_points"
    category: str = MorphismCategory.PROJECTION

    global_order: float = 2.0
    superconvergent_order: float = 4.0
    point_type: str = "gauss"  # "gauss", "lobatto", "radau", "midpoint", "knot"
    element_degree: int = 1
    method: str = "fem"  # "fem", "dg", "fd", "spline_collocation"
    gain: float = 2.0

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "global_error_estimate_away_from_special_points",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "uniform_error_convergence_rate",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "pointwise_error_O(h^{p+δ}) at special points",
            "postprocessing_recovery_possible",
        ]
    )
    kernel_description: str = (
        "Non-superconvergent points: leading-order error terms "
        "do not cancel. Kernel = evaluation points excluding special points."
    )

    @property
    def mathematical_form(self) -> str:
        points_desc = {
            "gauss": "Gauss-Legendre quadrature points",
            "lobatto": "Gauss-Lobatto points",
            "radau": "Radau points (upwind end for DG outflow)",
            "midpoint": "Element midpoints / cell centers",
            "knot": "Spline knots",
        }
        point_str = points_desc.get(self.point_type, self.point_type)
        return (
            f"||u_h(x*) - u(x*)|| = O(h^{self.superconvergent_order}) "
            f"(global: O(h^{self.global_order}))\n"
            f"Superconvergent at: {point_str}\n"
            f"Gain: δ = {self.gain}, method: {self.method}, p = {self.element_degree}"
        )

    def compute_gain(self) -> float:
        """计算超收敛增益 δ = superconvergent_order - global_order."""
        return self.superconvergent_order - self.global_order

    def error_at_superconvergent_point(self, h: float, constant: float = 1.0) -> float:
        """估计超收敛点处的误差 C * h^{p+δ}."""
        return constant * h**self.superconvergent_order  # type: ignore[no-any-return]

    def error_at_general_point(self, h: float, constant: float = 1.0) -> float:
        """估计一般点处的误差 C * h^p."""
        return constant * h**self.global_order  # type: ignore[no-any-return]
