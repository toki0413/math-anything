"""演化问题结构家族。

EvolutionProblem = ∂u/∂t = F(u)

涵盖四大子类：
  - HamiltonianSystem: 辛流形上的 Hamilton 流（MD）
  - ConservationLawSystem: 守恒律 ∂U/∂t + ∇·F(U) = 0（NS, Maxwell）
  - DissipativeSystem: 梯度流/耗散系统（相场, 反应-扩散）
  - StochasticSystem: 随机微分方程（Langevin）

结构主义视角：
  所谓的"MD"、"CFD"、"相场"不过是这四种演化结构的具体实例。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata
from .properties import (
    StructuralInvariant,
    get_invariants,
)


@dataclass
class EvolutionProblem(AbstractMathematicalStructure):
    """演化问题基类：∂u/∂t = F(u).

    所有随时间演化的连续/离散系统的母类。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Evolution Problem",
            canonical_form="∂u/∂t = F(u)",
            description="Time evolution of a state vector under a flow map",
        )
    )
    phase_space_dim: int = 0
    time_dependent: bool = True
    autonomous: bool = True  # F 不显含 t

    @property
    def function_space(self) -> str:
        return f"ℝ^{self.phase_space_dim}" if self.phase_space_dim > 0 else "Γ (phase space)"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return []


# ──────────────────────────────────────────────────────────────────
# 子类一：HamiltonianSystem — 辛结构
# ──────────────────────────────────────────────────────────────────


@dataclass
class HamiltonianSystem(EvolutionProblem):
    """Hamilton 系统：辛流形上的 Hamilton 方程.

    dq/dt = ∂H/∂p,  dp/dt = -∂H/∂q

    实例：
      - 经典分子动力学（LAMMPS, GROMACS 的 NVE 系综）
      - 天体力学
      - 几何光学（程函方程）

    核心不变量：辛结构保持、相体积守恒（Liouville）、能量守恒（自动）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Hamiltonian System",
            canonical_form="dq/dt = ∂H/∂p,  dp/dt = -∂H/∂q",
            description="Symplectic flow on phase space from a Hamiltonian function",
        )
    )
    symplectic: bool = True
    reversible: bool = True
    integrable: bool = False  # 完全可积（存在 N 个独立守恒量）

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants: list[StructuralInvariant] = []
        props = {"hamiltonian": self.symplectic, "time_dependent": not self.autonomous}

        for inv in get_invariants("hamiltonian"):
            if inv.is_active(props):  # type: ignore[arg-type]
                invariants.append(inv)

        if self.symplectic:
            invariants.append(
                StructuralInvariant(
                    name="symplectic_form",
                    expression="ω = Σ dp_i ∧ dq_i is invariant under flow",
                    theorem="Symplectic geometry (Darboux theorem)",
                    affected_quantities=["phase_space", "integration"],
                )
            )

        if self.integrable:
            invariants.append(
                StructuralInvariant(
                    name="liouville_integrability",
                    expression="N independent conserved quantities in involution",
                    theorem="Liouville-Arnold Theorem",
                    affected_quantities=["action_angle_variables"],
                )
            )

        return invariants


# ──────────────────────────────────────────────────────────────────
# 子类二：ConservationLawSystem — 守恒律（CFD, 电磁学）
# ──────────────────────────────────────────────────────────────────


class FluxType:
    """守恒律系统中通量的分类常量."""

    CONVECTIVE = "convective"
    DIFFUSIVE = "diffusive"
    TOTAL = "total"
    ROTATIONAL = "rotational"  # Maxwell: ∇× 型通量


@dataclass
class ConservedQuantity:
    """一个守恒变量."""

    name: str
    symbol: str
    dimensions: dict[str, float] = field(default_factory=dict)
    description: str = ""


@dataclass
class ConservationLawSystem(EvolutionProblem):
    """守恒律系统：∂U/∂t + ∇·F(U) = S(U).

    这是 CFD（Navier-Stokes）和电磁学（Maxwell）的共同数学骨架。

    守恒律系统的普适特征：
      - 双曲性 → 特征速度、不连续性（激波）
      - 熵条件 → 物理解的唯一性
      - Rankine-Hugoniot 条件 → 不连续面跳跃关系
      - 不同 regime 之间的态射（不可压→可压，层流→湍流）

    实例：
      - 不可压 Navier-Stokes：∂u/∂t + u·∇u = -∇p/ρ + ν∇²u
      - 可压 Navier-Stokes：∂U/∂t + ∇·F_c(U) = ∇·F_d(U)
      - Maxwell 方程：∂B/∂t = -∇×E,  ∂D/∂t = ∇×H - J
      - 浅水方程、Euler 方程、Burgers 方程
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Conservation Law System",
            canonical_form="∂U/∂t + ∇·F(U) = S(U)",
            description="Hyperbolic/parabolic system expressing conservation of physical quantities",
        )
    )
    conserved_variables: list[ConservedQuantity] = field(default_factory=list)
    hyperbolic: bool = True
    flux_type: str = FluxType.CONVECTIVE
    has_diffusion: bool = False
    has_source: bool = False
    spatial_dim: int = 3

    @property
    def function_space(self) -> str:
        return f"[L²(ℝ^{self.spatial_dim})]^{len(self.conserved_variables)}"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants: list[StructuralInvariant] = []
        props: dict[str, Any] = {
            "hyperbolic": self.hyperbolic,
            "has_diffusion": self.has_diffusion,
            "has_source": self.has_source,
            "external_force": self.has_source,
        }

        for inv in get_invariants("conservation_law"):
            if inv.is_active(props):
                invariants.append(inv)

        if self.hyperbolic:
            invariants.append(
                StructuralInvariant(
                    name="finite_signal_speed",
                    expression="|λ_max| = max eigenvalue of ∂F/∂U",
                    theorem="Hyperbolicity of conservation laws",
                    affected_quantities=["characteristic_speed", "CFL_condition"],
                )
            )

        if self.has_diffusion:
            invariants.append(
                StructuralInvariant(
                    name="energy_dissipation",
                    expression="dE/dt = -∫ ν|∇u|² dV ≤ 0 (viscous dissipation)",
                    theorem="Second Law of Thermodynamics (entropy production)",
                    affected_quantities=["kinetic_energy", "entropy"],
                )
            )

        return invariants

    @property
    def dimensional_rank(self) -> int:
        """守恒律涉及的基础维度：M, L, T (至少 3 个).

        热传导附加：Θ（共 4 个）。
        """
        base = 3  # M, L, T
        # 检查是否有热力学量（温度维度）
        for q in self.conserved_variables:
            if "temperature" in q.dimensions:
                return 4
        return base


# ──────────────────────────────────────────────────────────────────
# Navier-Stokes 专门化
# ──────────────────────────────────────────────────────────────────


class NSRegime:
    """Navier-Stokes 的流动区域."""

    INCOMPRESSIBLE = "incompressible"
    COMPRESSIBLE_SUBSONIC = "compressible_subsonic"
    COMPRESSIBLE_TRANSONIC = "compressible_transonic"
    COMPRESSIBLE_SUPERSONIC = "compressible_supersonic"
    MULTIPHASE = "multiphase"


class NSTurbulenceModel:
    """湍流建模类型（态射选择）."""

    NONE = "none"  # DNS / 层流
    RANS_KEPSILON = "rans_k_epsilon"
    RANS_KOMEGA_SST = "rans_k_omega_sst"
    RANS_REYNOLDS_STRESS = "rans_reynolds_stress"
    LES_SMAGORINSKY = "les_smagorinsky"
    LES_DYNAMIC = "les_dynamic"
    DES = "des"  # 混合 RANS-LES
    WALL_FUNCTION = "wall_function_only"


@dataclass
class NavierStokesProblem(ConservationLawSystem):
    """Navier-Stokes 方程作为守恒律的实例.

    不可压形式：
      ∇·u = 0
      ∂u/∂t + u·∇u = -∇p/ρ + ν∇²u + f

    可压形式：
      ∂ρ/∂t + ∇·(ρu) = 0
      ∂(ρu)/∂t + ∇·(ρu⊗u + pI) = ∇·τ + f
      ∂(ρE)/∂t + ∇·((ρE+p)u) = ∇·(τ·u - q)

    结构主义含义：
      NS 是 ConservationLawSystem 的最重要实例。
      所有湍流模型、壁面函数、不可压近似
      都是结构态射，不是简单的"近似列表"。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Navier-Stokes Equations",
            canonical_form="∂u/∂t + u·∇u = -∇p/ρ + ν∇²u + f",
            description="Conservation of mass, momentum, and energy for a Newtonian fluid",
        )
    )
    regime: str = NSRegime.INCOMPRESSIBLE
    turbulence_model: str = NSTurbulenceModel.NONE
    include_energy: bool = False
    include_gravity: bool = False
    include_surface_tension: bool = False
    reynolds_number: float | None = None
    mach_number: float | None = None

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants

        # 不可压特有
        if self.regime == NSRegime.INCOMPRESSIBLE:
            invariants.append(
                StructuralInvariant(
                    name="divergence_free",
                    expression="∇·u = 0",
                    theorem="Mass conservation for constant density",
                    affected_quantities=["velocity_field"],
                )
            )
            invariants.append(
                StructuralInvariant(
                    name="pressure_poisson",
                    expression="∇²p = -ρ ∇·(u·∇u)  (from taking divergence of momentum)",
                    theorem="Incompressibility constraint → elliptic equation for p",
                    affected_quantities=["pressure"],
                )
            )

        # 可压特有
        if self.regime != NSRegime.INCOMPRESSIBLE:
            invariants.append(
                StructuralInvariant(
                    name="equation_of_state",
                    expression="p = p(ρ, T)  closes the system",
                    theorem="Thermodynamic closure",
                    affected_quantities=["pressure", "density", "temperature"],
                )
            )

        # 湍流模型的不变量（RANS）
        if "rans" in self.turbulence_model:
            invariants.append(
                StructuralInvariant(
                    name="reynolds_stress_closure",
                    expression="⟨u_i' u_j'⟩ ≈ -ν_t (∂ū_i/∂x_j + ∂ū_j/∂x_i) + (2/3)k δ_ij",
                    theorem="Boussinesq hypothesis",
                    condition="rans type turbulence model",
                    affected_quantities=["mean_velocity", "turbulent_viscosity"],
                )
            )

        return invariants

    @property
    def buckingham_pi_count(self) -> int:
        """根据包含的物理计算 Buckingham π 群数量.

        基础 NS（质量+动量）：5 个有量纲量 - 3 个独立维度 = 2 个 π 群
        每加一个物理（能量/重力/表面张力）+1 个 π 群
        """
        base_count = 2  # Re, St (or Eu)
        if self.include_energy:
            base_count += 2  # Pr, Ec (or Nu)
        if self.include_gravity:
            base_count += 1  # Fr
        if self.include_surface_tension:
            base_count += 1  # We or Ca
        if "compressible" in self.regime:
            base_count += 1  # Ma
        return base_count

    @property
    def named_pi_groups(self) -> dict[str, str]:
        """NS 方程的所有命名无量纲数."""
        groups: dict[str, str] = {
            "Re": "ρUL/μ = 惯性力/黏性力",
            "St": "fL/U = 非定常惯性力/对流惯性力",
        }
        if "compressible" in self.regime:
            groups["Ma"] = "U/c = 流速/声速"
        if self.include_energy:
            groups["Pr"] = "ν/α = 动量扩散/热扩散"
            groups["Ec"] = "U²/(c_p ΔT) = 动能/焓差"
        if self.include_gravity:
            groups["Fr"] = "U/√(gL) = 惯性力/重力"
        if self.include_surface_tension:
            groups["We"] = "ρU²L/σ = 惯性力/表面张力"
            groups["Ca"] = "μU/σ = 黏性力/表面张力"
        return groups


# ──────────────────────────────────────────────────────────────────
# 子类三：DissipativeSystem — 耗散/梯度流
# ──────────────────────────────────────────────────────────────────


@dataclass
class DissipativeSystem(EvolutionProblem):
    """耗散系统：能量单调递减的演化.

    ∂φ/∂t = -L δF/δφ  或  ∂φ/∂t = ∇·(M∇(δF/δφ))

    实例：
      - 相场模型（Cahn-Hilliard, Allen-Cahn）
      - 反应-扩散方程
      - 热传导方程
      - Ginzburg-Landau 动力学

    核心特征：存在 Lyapunov 泛函 F 使得 dF/dt ≤ 0。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Dissipative System",
            canonical_form="∂φ/∂t = -L δF/δφ",
            description="Gradient flow minimizing a free energy functional",
        )
    )
    order_parameter: str = ""  # φ: 序参量
    free_energy_functional: str = ""  # F[φ] = ∫ f(φ, ∇φ) dV
    conserved: bool = False  # Cahn-Hilliard (conserved) vs Allen-Cahn (not)
    mobility: str = "constant"  # L 或 M(φ) 的形式

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = [
            StructuralInvariant(
                name="lyapunov_functional",
                expression="dF/dt = -L ∫ |δF/δφ|² dV ≤ 0",
                theorem="Gradient flow property",
                affected_quantities=["free_energy"],
            ),
            StructuralInvariant(
                name="energy_monotonic",
                expression="F(t₂) ≤ F(t₁) for t₂ > t₁",
                theorem="H-theorem for dissipative systems",
                affected_quantities=["free_energy"],
            ),
        ]

        if self.conserved:
            invariants.append(
                StructuralInvariant(
                    name="mass_conservation",
                    expression="d/dt ∫ φ dV = 0",
                    theorem="Cahn-Hilliard: conservation of order parameter",
                    affected_quantities=["order_parameter", "integral"],
                )
            )

        return invariants


# ──────────────────────────────────────────────────────────────────
# 子类四：StochasticSystem — 随机系统
# ──────────────────────────────────────────────────────────────────


@dataclass
class StochasticSystem(EvolutionProblem):
    """随机微分方程：确定性 + 随机项.

    dx = F(x) dt + σ(x) dW

    实例：
      - Langevin 动力学（GROMACS 的恒温器）
      - Brownian 动力学
      - 随机 Landau-Lifshitz-Gilbert（微磁学）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EVOLUTION,
            name="Stochastic System",
            canonical_form="dx = F(x) dt + σ dW",
            description="Stochastic differential equation with drift and diffusion",
        )
    )
    drift_term: str = ""
    noise_type: str = "additive"  # "additive", "multiplicative"
    noise_strength: float = 0.0
    fluctuation_dissipation: bool = False  # 是否满足 FDT

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = [
            StructuralInvariant(
                name="ito_isometry",
                expression="E[∫ σ dW]² = ∫ σ² dt",
                theorem="Ito isometry",
                affected_quantities=["noise", "variance"],
            ),
        ]
        if self.fluctuation_dissipation:
            invariants.append(
                StructuralInvariant(
                    name="fluc_diss_theorem",
                    expression="σ² ∝ k_B T γ",
                    theorem="Fluctuation-Dissipation Theorem",
                    affected_quantities=["temperature", "damping", "noise"],
                )
            )
        return invariants


class SymplecticIntegrator:
    """Symplectic integrator for Hamiltonian systems.

    Preserves the symplectic 2-form ω = dp ∧ dq exactly,
    ensuring long-term energy conservation.
    """

    def __init__(self, hamiltonian: Callable, dim: int = 1):
        """
        Args:
            hamiltonian: Function H(q, p) returning scalar energy
            dim: Number of degrees of freedom
        """
        self.hamiltonian = hamiltonian
        self.dim = dim

    def velocity_verlet_step(
        self, q: np.ndarray, p: np.ndarray, dt: float, mass: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """One step of Velocity Verlet (symplectic, 2nd order).

        Args:
            q: Position array (dim,)
            p: Momentum array (dim,)
            dt: Time step
            mass: Particle mass

        Returns:
            (q_new, p_new) after one step
        """
        eps = 1e-8
        force = np.zeros(self.dim)
        for i in range(self.dim):
            q_plus = q.copy()
            q_plus[i] += eps
            q_minus = q.copy()
            q_minus[i] -= eps
            force[i] = -(self.hamiltonian(q_plus, p) - self.hamiltonian(q_minus, p)) / (2 * eps)

        p_half = p + 0.5 * dt * force
        q_new = q + dt * p_half / mass

        force_new = np.zeros(self.dim)
        for i in range(self.dim):
            q_plus = q_new.copy()
            q_plus[i] += eps
            q_minus = q_new.copy()
            q_minus[i] -= eps
            force_new[i] = -(self.hamiltonian(q_plus, p_half) - self.hamiltonian(q_minus, p_half)) / (2 * eps)

        p_new = p_half + 0.5 * dt * force_new
        return q_new, p_new

    def integrate(self, q0: np.ndarray, p0: np.ndarray, dt: float, n_steps: int, mass: float = 1.0) -> dict:
        """Integrate Hamiltonian system for n_steps.

        Returns:
            Dict with trajectory arrays and energy history
        """
        q_traj = [q0.copy()]
        p_traj = [p0.copy()]
        energy = [self.hamiltonian(q0, p0)]

        q, p = q0.copy(), p0.copy()
        for _ in range(n_steps):
            q, p = self.velocity_verlet_step(q, p, dt, mass)
            q_traj.append(q.copy())
            p_traj.append(p.copy())
            energy.append(self.hamiltonian(q, p))

        return {
            "q": np.array(q_traj),
            "p": np.array(p_traj),
            "energy": np.array(energy),
            "energy_drift": float(abs(energy[-1] - energy[0]) / abs(energy[0])) if abs(energy[0]) > 1e-15 else 0.0,
            "n_steps": n_steps,
            "dt": dt,
        }

    def verify_symplecticity(self, q0: np.ndarray, p0: np.ndarray, dt: float, mass: float = 1.0) -> dict:
        """Verify symplecticity by checking det(J) = 1 for the flow map.

        Uses finite differences to compute the Jacobian of the flow map.
        """
        eps = 1e-5
        dim = len(q0)
        phase_dim = 2 * dim

        q1, p1 = self.velocity_verlet_step(q0, p0, dt, mass)

        J = np.zeros((phase_dim, phase_dim))
        for i in range(phase_dim):
            q_pert = q0.copy()
            p_pert = p0.copy()
            if i < dim:
                q_pert[i] += eps
            else:
                p_pert[i - dim] += eps

            q1_pert, p1_pert = self.velocity_verlet_step(q_pert, p_pert, dt, mass)

            J[:dim, i] = (q1_pert - q1) / eps
            J[dim:, i] = (p1_pert - p1) / eps

        Omega = np.zeros((phase_dim, phase_dim))
        Omega[:dim, dim:] = np.eye(dim)
        Omega[dim:, :dim] = -np.eye(dim)

        symplectic_error = np.linalg.norm(J.T @ Omega @ J - Omega)
        det_J = np.linalg.det(J)

        return {
            "is_symplectic": symplectic_error < 1e-4,
            "symplectic_error": float(symplectic_error),
            "det_J": float(det_J),
            "det_J_minus_1": float(abs(det_J - 1.0)),
        }


class ConservationLawSolver:
    """Numerical solver for conservation law systems.

    Solves ∂U/∂t + ∂F(U)/∂x = 0 using finite volume methods.
    """

    def __init__(self, flux_function: Callable, n_vars: int = 1):
        """
        Args:
            flux_function: F(U) returning flux vector
            n_vars: Number of conserved variables
        """
        self.flux_function = flux_function
        self.n_vars = n_vars

    def flux_jacobian(self, U: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute flux Jacobian dF/dU via finite differences."""
        n = len(U)
        F0 = self.flux_function(U)
        m = len(F0)
        J = np.zeros((m, n))
        for i in range(n):
            U_pert = U.copy()
            U_pert[i] += eps
            J[:, i] = (self.flux_function(U_pert) - F0) / eps
        return J

    def characteristic_speeds(self, U: np.ndarray) -> np.ndarray:
        """Compute characteristic speeds (eigenvalues of flux Jacobian)."""
        J = self.flux_jacobian(U)
        return np.linalg.eigvals(J).real

    def max_wave_speed(self, U: np.ndarray) -> float:
        """Compute maximum wave speed for CFL condition."""
        return float(np.max(np.abs(self.characteristic_speeds(U))))

    def lax_friedrichs_step(self, U: np.ndarray, dx: float, dt: float) -> np.ndarray:
        """One step of Lax-Friedrichs scheme for 1D conservation law.

        U_i^{n+1} = 0.5*(U_{i-1} + U_{i+1}) - dt/(2*dx)*(F_{i+1} - F_{i-1})
        """
        n = len(U)
        U_new = np.zeros_like(U)
        for i in range(1, n - 1):
            F_left = self.flux_function(np.array([U[i - 1]]))[0]
            F_right = self.flux_function(np.array([U[i + 1]]))[0]
            U_new[i] = 0.5 * (U[i - 1] + U[i + 1]) - dt / (2 * dx) * (F_right - F_left)
        U_new[0] = U[0]
        U_new[-1] = U[-1]
        return U_new

    def cfl_condition(self, U: np.ndarray, dx: float, cfl_number: float = 0.5) -> float:
        """Compute maximum stable timestep."""
        max_speed = self.max_wave_speed(U)
        if max_speed < 1e-15:
            return float("inf")
        return cfl_number * dx / max_speed
