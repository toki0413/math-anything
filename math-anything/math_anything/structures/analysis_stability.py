"""数值分析 — 稳定性结构。

数值分析元结构：离散化、相容性、稳定性、CFL 条件。
核心定理：Lax 等价定理 — 相容 + 稳定 ⇒ 收敛。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata
from .properties import StructuralInvariant


@dataclass
class NumericalAnalysis(AbstractMathematicalStructure):
    """Base class for all numerical analysis structures.

    Numerical analysis studies how to approximate solutions of
    continuous problems using finite discrete representations.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Numerical Analysis",
            canonical_form="L_h(u_h) = f_h  →  L(u) = f  as h → 0",
            description="Discretization, consistency, stability, and convergence of mathematical problems",
        )
    )
    mesh_size: float = 0.0
    spatial_dim: int = 1
    temporal: bool = False

    @property
    def function_space(self) -> str:
        return "V_h ⊂ V (discrete subspace → continuous function space)"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return []


@dataclass
class ConsistencyCondition(NumericalAnalysis):
    """Consistency: the discrete operator approximates the continuous one.

    ∥L_h(u) - L(u)∥ → 0 as h → 0

    The truncation error vanishes as the mesh is refined.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Consistency Condition",
            canonical_form="∥L_h(u) - L(u)∥ → 0 as h → 0",
            description="Truncation error vanishes with mesh refinement",
        )
    )
    truncation_order: int = 1
    smoothness_required: str = ""

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="truncation_error_vanishes",
                expression=f"∥L_h(u) - L(u)∥ ≤ C h^{self.truncation_order} (polynomial consistency)",
                theorem="Taylor expansion → finite difference/volume/element truncation error",
                affected_quantities=["truncation_error", "mesh_size"],
            ),
            StructuralInvariant(
                name="operator_consistency_order",
                expression=f"τ_h = L_h(u) - L(u) = O(h^{self.truncation_order})",
                theorem="Local truncation error analysis",
                affected_quantities=["local_truncation_error"],
            ),
        ]

    def truncation_error(self, h: float, constant: float = 1.0) -> float:
        """计算截断误差上界 C * h^p."""
        return constant * h**self.truncation_order


@dataclass
class StabilityCondition(NumericalAnalysis):
    """Stability: discrete solution remains bounded under mesh refinement.

    Stability prevents exponential growth of errors.
    Different notions apply to different problem types.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Stability Condition",
            canonical_form="∥(L_h)^{-1}∥ ≤ C  or  ∥u_h^{n}∥ ≤ K∥u_h^{0}∥",
            description="Discrete problem remains well-posed under mesh refinement",
        )
    )
    stability_type: str = ""
    norm_type: str = "L2"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="uniform_boundedness_principle",
                expression="∃C > 0: ∥(L_h)^{-1}∥ ≤ C for all h > 0",
                theorem="Lax-Richtmyer Equivalence Theorem (stability half)",
                affected_quantities=["discrete_solution", "condition_number"],
            ),
        ]

    def stability_bound(self, solution_norms: list[float], rhs_norms: list[float]) -> float:
        """计算稳定性常数 C_s = max(||u_h|| / ||f_h||)."""
        if not solution_norms or not rhs_norms:
            return 0.0
        ratios = [s / r if r > 0 else float("inf") for s, r in zip(solution_norms, rhs_norms)]
        return float(max(ratios))


@dataclass
class VonNeumannStability(StabilityCondition):
    """Von Neumann (Fourier) stability analysis.

    For linear constant-coefficient PDEs each Fourier mode evolves independently.
    ∥G(ξ)∥ ≤ 1 for all ξ ensures no mode grows exponentially.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Von Neumann Stability",
            canonical_form="∥G(ξ)∥ ≤ 1 for all ξ ∈ [0, 2π]",
            description="Fourier amplification factor must be ≤ 1 for all wavenumbers",
        )
    )
    amplification_factor_expr: str = ""
    courant_number: float = 0.0
    scheme_type: str = ""

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="amplification_bounded",
                    expression="∥G(ξ)∥ ≤ 1 for all ξ ∈ Brillouin zone",
                    theorem="Von Neumann Stability Criterion",
                    affected_quantities=["amplification_factor", "fourier_mode"],
                ),
                StructuralInvariant(
                    name="cfl_from_von_neumann",
                    expression="Δt ≤ C_min Δx / abs:λ_maxabs: (derived from ∥G∥ ≤ 1)",
                    theorem="CFL condition from Fourier stability",
                    condition="self.courant_number > 0",
                    affected_quantities=["timestep", "grid_spacing", "wave_speed"],
                ),
            ]
        )
        return invariants

    def amplification_factor(self, scheme: str, cfl: float, theta: float) -> complex:
        """计算给定格式的放大因子 g(θ).

        Args:
            scheme: 'ftcs', 'lax_friedrichs', 'lax_wendroff', 'upwind', 'crank_nicolson'
            cfl: Courant 数 σ = v*dt/dx
            theta: 相位角 θ = k*dx
        """
        if scheme == "ftcs":
            return 1.0 - 1j * cfl * np.sin(theta)  # type: ignore[no-any-return]
        elif scheme == "lax_friedrichs":
            return np.cos(theta) - 1j * cfl * np.sin(theta)  # type: ignore[no-any-return]
        elif scheme == "lax_wendroff":
            return 1 - cfl**2 * (1 - np.cos(theta)) - 1j * cfl * np.sin(theta)  # type: ignore[no-any-return]
        elif scheme == "upwind":
            return 1 - cfl * (1 - np.exp(-1j * theta))  # type: ignore[no-any-return]
        elif scheme == "crank_nicolson":
            return (1 - 0.5j * cfl * np.sin(theta)) / (1 + 0.5j * cfl * np.sin(theta))  # type: ignore[no-any-return]
        return 1.0

    def is_stable(self, scheme: str, cfl: float, n_angles: int = 100) -> bool:
        """检查给定 CFL 数下格式是否稳定."""
        for k in range(n_angles):
            theta = 2 * np.pi * k / n_angles
            g = self.amplification_factor(scheme, cfl, theta)
            if abs(g) > 1.0 + 1e-10:
                return False
        return True

    def critical_cfl(self, scheme: str, tolerance: float = 1e-6) -> float:
        """通过二分法查找临界 CFL 数（稳定性极限）."""
        lo, hi = 0.0, 5.0
        while hi - lo > tolerance:
            mid = (lo + hi) / 2
            if self.is_stable(scheme, mid):
                lo = mid
            else:
                hi = mid
        return lo


@dataclass
class EnergyStability(StabilityCondition):
    """Energy (norm) stability for dissipative and conservative systems.

    E(u^{n+1}) ≤ E(u^n) + C Δt

    Monitored energy does not grow unboundedly.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Energy Stability",
            canonical_form="E(u^{n+1}) ≤ E(u^n) + C Δt",
            description="Discrete energy remains bounded (up to order Δt truncation)",
        )
    )
    energy_functional: str = ""
    growth_constant: float = 0.0
    conservative: bool = False

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.append(
            StructuralInvariant(
                name="discrete_energy_dissipation",
                expression=f"E^{{n+1}} - E^n ≤ {self.growth_constant} Δt (bounded growth)",
                theorem="Discrete energy method (summation by parts)",
                affected_quantities=["discrete_energy", "timestep"],
            ),
        )
        if self.conservative:
            invariants.append(
                StructuralInvariant(
                    name="exact_energy_conservation",
                    expression="E^{n+1} = E^n (symplectic/energy-preserving scheme)",
                    theorem="Discrete Noether (energy-conserving integrators)",
                    affected_quantities=["discrete_energy"],
                ),
            )
        return invariants

    def energy_change(self, E_current: float, E_previous: float, dt: float) -> float:
        """计算能量变化 ΔE = E^{n+1} - E^n."""
        return E_current - E_previous

    def is_energy_stable(self, E_current: float, E_previous: float, dt: float) -> bool:
        """检查能量稳定性: E^{n+1} ≤ E^n + C*Δt."""
        return E_current <= E_previous + self.growth_constant * dt


@dataclass
class LaxRichtmyerStability(StabilityCondition):
    """Lax-Richtmyer stability: uniform boundedness of discrete solution operator.

    ∥(L_h)^{-1}∥ ≤ C uniform in h

    This is the theoretical definition of well-posedness for difference schemes.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Lax-Richtmyer Stability",
            canonical_form="∥(L_h)^{-1}∥ ≤ C for all h > 0",
            description="Uniform boundedness of the inverse discrete operator",
        )
    )
    uniform_bound: float = 0.0
    operator_family: str = "linear"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.append(
            StructuralInvariant(
                name="lax_richtmyer_uniform_boundedness",
                expression=f"∥(L_h)^{{-1}}∥ ≤ {self.uniform_bound} for all mesh sizes",
                theorem="Lax-Richtmyer Stability Theorem",
                affected_quantities=["inverse_operator_norm", "mesh_size"],
            ),
        )
        return invariants

    def check_uniform_bound(self, operator_norms: list[float], mesh_sizes: list[float]) -> bool:
        """检查逆算子范数是否关于 h 一致有界."""
        if not operator_norms:
            return True
        return max(operator_norms) <= self.uniform_bound if self.uniform_bound > 0 else True

    def compute_condition_number(self, operator: np.ndarray) -> float:
        """计算离散算子的条件数."""
        return float(np.linalg.cond(operator))


@dataclass
class LaxEquivalenceTheorem(NumericalAnalysis):
    """The Lax Equivalence Theorem: consistency + stability ⇒ convergence.

    This is THE central theorem of numerical analysis.
    For a well-posed linear initial value problem with a consistent
    approximation, stability is necessary and sufficient for convergence.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Lax Equivalence Theorem",
            canonical_form="consistent + stable => convergent",
            description="For linear well-posed problems: consistency + stability ⇔ convergence",
        )
    )
    problem_well_posed: bool = True
    linear: bool = True
    convergence_rate: float = 0.0

    @property
    def function_space(self) -> str:
        return "V_h → V (discrete to continuous, convergence in ∥·∥)"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="consistency_plus_stability_implies_convergence",
                expression="τ_h → 0 + ∥(L_h)^{-1}∥ ≤ C ⇒ ∥u - u_h∥ ≤ C τ_h → 0",
                theorem="Lax Equivalence Theorem (P. D. Lax, 1956)",
                condition="self.linear and self.problem_well_posed",
                affected_quantities=["convergence", "truncation_error", "stability_bound"],
                proof_sketch="u - u_h = (L_h)^{-1}(L_h u - f_h) = (L_h)^{-1} τ_h → 0",
            ),
            StructuralInvariant(
                name="stability_necessary_for_convergence",
                expression="convergence ⇒ stability (for linear well-posed problems)",
                theorem="Lax Equivalence Theorem (converse direction)",
                condition="self.linear and self.problem_well_posed",
                affected_quantities=["stability", "convergence"],
            ),
            StructuralInvariant(
                name="convergence_rate_equals_consistency_order",
                expression=f"∥u - u_h∥ = O(h^{self.convergence_rate}) when stable",
                theorem="Consistency order + stability ⇒ convergence order",
                condition="self.convergence_rate > 0",
                affected_quantities=["convergence_rate", "consistency_order", "mesh_size"],
            ),
        ]

    def check_convergence(self, is_consistent: bool, is_stable: bool) -> dict:
        """应用 Lax 等价定理判断收敛性."""
        return {
            "consistent": is_consistent,
            "stable": is_stable,
            "convergent": is_consistent and is_stable,
            "theorem": "Lax Equivalence: consistency + stability ⟺ convergence (for linear well-posed problems)",
            "applicable": self.linear and self.problem_well_posed,
        }


@dataclass
class CFL_Condition(NumericalAnalysis):
    """CFL (Courant-Friedrichs-Lewy) condition for explicit time stepping.

    The numerical domain of dependence must contain the physical domain of dependence.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="CFL Condition",
            canonical_form="Δt ≤ C h^α / abs:λabs: (domain of dependence constraint)",
            description="Timestep restriction for explicit schemes to maintain stability",
        )
    )
    max_timestep: float = 0.0
    min_grid_spacing: float = 0.0
    cfl_number: float = 0.0

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="domain_of_dependence",
                expression=f"Δt ≤ {self.cfl_number} h / abs:λabs: (numerical domain ⊇ physical domain)",
                theorem="CFL Necessary Condition (Courant-Friedrichs-Lewy, 1928)",
                affected_quantities=["timestep", "grid_spacing", "characteristic_speed"],
            ),
        ]

    def hyperbolic_cfl(self, max_wave_speed: float, dx: float, dim: int = 1) -> float:
        """双曲型 CFL 条件: dt <= dx / (dim * max_wave_speed)."""
        if max_wave_speed <= 0:
            return float("inf")
        return dx / (dim * max_wave_speed)

    def parabolic_cfl(self, diffusivity: float, dx: float, dim: int = 1) -> float:
        """抛物型 CFL 条件: dt <= dx^2 / (2 * dim * diffusivity)."""
        if diffusivity <= 0:
            return float("inf")
        return dx**2 / (2 * dim * diffusivity)

    def lax_equivalence_check(self, is_consistent: bool, is_stable: bool) -> dict:
        """应用 Lax 等价定理: consistency + stability ⟹ convergence."""
        return {
            "consistent": is_consistent,
            "stable": is_stable,
            "convergent": is_consistent and is_stable,
            "theorem": "Lax Equivalence: consistency + stability ⟺ convergence (for linear well-posed problems)",
        }


@dataclass
class HyperbolicCFL(CFL_Condition):
    """CFL for hyperbolic PDEs: Δt ≤ C_min Δx / abs:λ_maxabs:.

    Characteristic speeds determine the timestep restriction.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Hyperbolic CFL Condition",
            canonical_form="Δt ≤ Δx / abs:λ_maxabs:",
            description="Timestep limited by fastest characteristic speed for hyperbolic systems",
        )
    )
    max_characteristic_speed: float = 0.0

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.append(
            StructuralInvariant(
                name="cfl_hyperbolic",
                expression=f"Δt ≤ Δx / {self.max_characteristic_speed} (information propagation limited by characteristics)",  # noqa: E501
                theorem="CFL for hyperbolic conservation laws",
                affected_quantities=["timestep", "grid_spacing", "characteristic_speed"],
            ),
        )
        return invariants

    def compute_max_timestep(self, dx: float, dim: int = 1) -> float:
        """计算双曲型最大稳定时间步."""
        return self.hyperbolic_cfl(self.max_characteristic_speed, dx, dim)

    def compute_cfl_number(self, dx: float, dt: float) -> float:
        """计算 CFL 数."""
        if dx <= 0:
            return float("inf")
        return self.max_characteristic_speed * dt / dx


@dataclass
class ParabolicCFL(CFL_Condition):
    """CFL for parabolic PDEs: Δt ≤ C Δx² / (2D).

    Diffusion imposes a stricter timestep restriction (Δt ∝ Δx²).
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Parabolic CFL Condition",
            canonical_form="Δt ≤ Δx² / (2D)",
            description="Timestep limited by diffusion for parabolic systems",
        )
    )
    diffusion_coefficient: float = 0.0

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.append(
            StructuralInvariant(
                name="cfl_parabolic",
                expression=f"Δt ≤ Δx² / (2 * {self.diffusion_coefficient}) (diffusion timescale dominates)",
                theorem="CFL for parabolic (diffusion) equations",
                affected_quantities=["timestep", "grid_spacing", "diffusivity"],
            ),
        )
        return invariants

    def compute_max_timestep(self, dx: float, dim: int = 1) -> float:
        """计算抛物型最大稳定时间步."""
        return self.parabolic_cfl(self.diffusion_coefficient, dx, dim)

    def compute_diffusion_number(self, dx: float, dt: float) -> float:
        """计算扩散数 D*dt/dx^2."""
        if dx <= 0:
            return float("inf")
        return self.diffusion_coefficient * dt / dx**2


@dataclass
class GeneralCFL(CFL_Condition):
    """General CFL for mixed hyperbolic-parabolic systems.

    Combines advective and diffusive timescale restrictions.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="General CFL Condition",
            canonical_form="Δt ≤ min(C_adv h / abs:λabs:, C_diff h² / D)",
            description="Combined CFL for mixed advection-diffusion systems",
        )
    )
    advective_cfl: float = 0.0
    diffusive_cfl: float = 0.0
    mixed: bool = True

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.append(
            StructuralInvariant(
                name="cfl_mixed",
                expression=f"Δt ≤ min({self.advective_cfl} h/abs:λabs:, {self.diffusive_cfl} h²/D)",
                theorem="CFL for mixed hyperbolic-parabolic systems",
                affected_quantities=["timestep", "grid_spacing", "characteristic_speed", "diffusivity"],
            ),
        )
        return invariants

    def compute_max_timestep(self, dx: float, wave_speed: float, diffusivity: float, dim: int = 1) -> float:
        """计算混合型最大稳定时间步: min(双曲, 抛物)."""
        dt_hyperbolic = self.hyperbolic_cfl(wave_speed, dx, dim) if wave_speed > 0 else float("inf")
        dt_parabolic = self.parabolic_cfl(diffusivity, dx, dim) if diffusivity > 0 else float("inf")
        return min(dt_hyperbolic, dt_parabolic)

    def compute_cfl_and_diffusion_numbers(self, dx: float, dt: float, wave_speed: float, diffusivity: float) -> dict:
        """同时计算 CFL 数和扩散数."""
        cfl_num = wave_speed * dt / dx if dx > 0 else float("inf")
        diff_num = diffusivity * dt / dx**2 if dx > 0 else float("inf")
        return {
            "cfl_number": cfl_num,
            "diffusion_number": diff_num,
            "both_stable": cfl_num <= 1.0 and diff_num <= 0.5,
        }
