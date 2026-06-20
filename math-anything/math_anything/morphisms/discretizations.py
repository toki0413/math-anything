"""离散化态射：连续 → 离散的具体实现."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import ContinuumToDiscrete

# 共享的离散化不变量
_BASE_DISC_KEPT = [
    "conservation_laws_at_discrete_level",
    "linearity (if original is linear)",
]
_BASE_DISC_LOST = [
    "exact_solution",
    "infinite_dimensional_completeness",
    "pointwise_satisfaction_of_pde",
]
_BASE_DISC_INTRODUCED = [
    "discretization_error",
    "stability_constraint",
    "mesh_dependency",
]


@dataclass
class FEMDiscretization(ContinuumToDiscrete):
    """有限元离散化 — 组装刚度矩阵和质量矩阵."""

    name: str = "fem_discretization"
    method: str = "fem"
    element_type: str = "linear"
    element_shape: str = "tetrahedron"
    n_elements: int = 1000
    polynomial_order: int = 1
    domain_length: float = 1.0

    invariants_kept: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_KEPT
            + [
                "weak_form_satisfaction",
                "dirichlet_bc_exact_at_nodes",
            ]
        )
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_LOST
            + [
                "inter_element_stress_continuity",
            ]
        )
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_INTRODUCED
            + [
                "element_shape_function_error",
                "integration_quadrature_error",
            ]
        )
    )

    @property
    def mathematical_form(self) -> str:
        return f"u(x) = sum N_i(x) u_i, element: {self.element_shape}"

    def assemble_stiffness_matrix(self) -> np.ndarray:
        """组装一维刚度矩阵 -u''=f 线性元."""
        n = self.n_elements + 1
        h = self.domain_length / self.n_elements
        K = np.zeros((n, n))
        for i in range(self.n_elements):
            K[i, i] += 1.0 / h
            K[i, i + 1] += -1.0 / h
            K[i + 1, i] += -1.0 / h
            K[i + 1, i + 1] += 1.0 / h
        return K

    def assemble_mass_matrix(self) -> np.ndarray:
        """组装一维质量矩阵 线性元."""
        n = self.n_elements + 1
        h = self.domain_length / self.n_elements
        M = np.zeros((n, n))
        for i in range(self.n_elements):
            M[i, i] += h / 3
            M[i, i + 1] += h / 6
            M[i + 1, i] += h / 6
            M[i + 1, i + 1] += h / 3
        return M

    def galerkin_orthogonality_error(self, u_exact: Callable, f: Callable) -> float:
        """计算能量范数误差 ||u - u_h||_E."""
        K = self.assemble_stiffness_matrix()
        M = self.assemble_mass_matrix()
        n = self.n_elements + 1
        nodes = np.linspace(0, self.domain_length, n)
        f_vec = M @ np.array([f(x) for x in nodes])
        # 施加边界条件 u(0) = 0
        K_red = K[1:, 1:]
        f_red = f_vec[1:]
        u_h = np.zeros(n)
        u_h[1:] = np.linalg.solve(K_red, f_red)
        u_exact_vec = np.array([u_exact(x) for x in nodes])
        error = np.sqrt(np.abs((u_exact_vec - u_h) @ K @ (u_exact_vec - u_h)))
        return float(error)

    def convergence_rate(self, errors: list[float], mesh_sizes: list[float]) -> float:
        """从 log-log 回归估计收敛阶."""
        log_e = np.log(np.array(errors))
        log_h = np.log(np.array(mesh_sizes))
        rate, _ = np.polyfit(log_h, log_e, 1)
        return float(rate)


@dataclass
class FVMDiscretization(ContinuumToDiscrete):
    """有限体积法离散化 — 计算守恒律的通量矩阵."""

    name: str = "fvm_discretization"
    method: str = "fvm"
    scheme: str = "central"
    n_cells: int = 100000
    face_interpolation: str = "linear"
    domain_length: float = 1.0
    flux_function: Callable = field(default=None)

    invariants_kept: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_KEPT
            + [
                "discrete_conservation",
                "flux_balance_at_cell_level",
            ]
        )
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_LOST
            + [
                "pointwise_solution",
            ]
        )
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_INTRODUCED
            + [
                "numerical_diffusion",
                "flux_limiter_dependency",
            ]
        )
    )

    @property
    def mathematical_form(self) -> str:
        return f"int F*n dS = sum F_f*n_f A_f, scheme: {self.scheme}"

    def __post_init__(self):
        super().__post_init__()
        if self.flux_function is None:
            self.flux_function = lambda u: u  # 默认线性平流

    @property
    def dx(self) -> float:
        return self.domain_length / self.n_cells

    def compute_flux_matrix(self, u: np.ndarray) -> np.ndarray:
        """计算给定状态 u 的数值通量矩阵 (Lax-Friedrichs)."""
        n = len(u)
        F = np.zeros(n)
        dx = self.dx
        for i in range(n):
            if i < n - 1:
                f_left = self.flux_function(u[i])
                f_right = self.flux_function(u[i + 1])
                alpha = max(abs(self.flux_function(u[i])), abs(self.flux_function(u[i + 1])))
                F[i] += 0.5 * (f_left + f_right) - 0.5 * alpha * (u[i + 1] - u[i])
            if i > 0:
                f_left = self.flux_function(u[i - 1])
                f_right = self.flux_function(u[i])
                alpha = max(abs(self.flux_function(u[i - 1])), abs(self.flux_function(u[i])))
                F[i] -= 0.5 * (f_left + f_right) - 0.5 * alpha * (u[i] - u[i - 1])
        return F / dx


@dataclass
class FDTDDiscretization(ContinuumToDiscrete):
    """FDTD 离散化（Yee 网格）— Maxwell 方程时域有限差分."""

    name: str = "fdtd_discretization"
    method: str = "fdtd"
    grid_spacing: str = "uniform"
    cfl_number: float = 0.99
    n_cells: int = 50
    domain_length: float = 1.0
    c: float = 1.0

    invariants_kept: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_KEPT
            + [
                "divergence_free_magnetic_field",
                "leapfrog_structure",
            ]
        )
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_LOST
            + [
                "continuous_rotational_invariance",
            ]
        )
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_INTRODUCED
            + [
                "numerical_dispersion",
                "staircase_approximation",
            ]
        )
    )

    @property
    def mathematical_form(self) -> str:
        return "E^{n+1} = E^n + (dt/eps) curl H  on staggered Yee grid"

    @property
    def dx(self) -> float:
        return self.domain_length / self.n_cells

    @property
    def dt(self) -> float:
        return self.dx / (2 * self.c)  # CFL 条件

    def compute_cfl_number(self) -> float:
        """计算 Courant 数 c*dt/dx."""
        return self.c * self.dt / self.dx

    def yee_step(self, E: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """一维 Yee 算法单步推进."""
        dx = self.dx
        dt = self.dt
        # 更新 B: B^{n+1/2} = B^{n-1/2} - dt/dx * (E^n_{i+1} - E^n_i)
        B_new = B.copy()
        B_new[:-1] -= (dt / dx) * (E[1:] - E[:-1])
        # 更新 E: E^{n+1} = E^n + dt/dx * (B^{n+1/2}_{i} - B^{n+1/2}_{i-1})
        E_new = E.copy()
        E_new[1:] += (dt / dx) * (B_new[1:] - B_new[:-1])
        return E_new, B_new


@dataclass
class SpectralDiscretization(ContinuumToDiscrete):
    """谱方法离散化 — 计算微分矩阵."""

    name: str = "spectral_discretization"
    method: str = "spectral"
    basis: str = "fourier"
    n_modes: int = 64
    convergence: str = "exponential_for_smooth_solutions"
    domain_length: float = 2 * np.pi

    invariants_kept: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_KEPT
            + [
                "exponential_convergence",
                "spectral_accuracy",
            ]
        )
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_LOST
            + [
                "locality",
            ]
        )
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_INTRODUCED
            + [
                "aliasing_error",
            ]
        )
    )

    @property
    def mathematical_form(self) -> str:
        return f"u(x) = sum c_n phi_n(x), basis: {self.basis}"

    def chebyshev_diff_matrix(self, n: int) -> np.ndarray:
        """计算 (n+1)x(n+1) Chebyshev 微分矩阵."""
        if n == 0:
            return np.array([[0.0]])
        x = np.cos(np.pi * np.arange(n + 1) / n)
        c = np.ones(n + 1)
        c[0] = 2.0
        c[n] = 2.0
        c *= (-1.0) ** np.arange(n + 1)
        X = np.tile(x, (n + 1, 1))
        dX = X - X.T
        D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
        D -= np.diag(D.sum(axis=1))
        return D

    def fourier_diff_matrix(self, n: int) -> np.ndarray:
        """计算 Fourier 谱微分矩阵."""
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = 0.5 * (-1) ** (i - j) / np.tan(np.pi * (i - j) / n)
        return D


@dataclass
class ParticleDiscretization(ContinuumToDiscrete):
    """粒子法离散化 — 计算核插值算子."""

    name: str = "particle_discretization"
    method: str = "particle"
    n_particles: int = 10000
    kernel_function: str = "wendland"

    invariants_kept: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_KEPT
            + [
                "lagrangian_perspective",
                "free_surface_natural_treatment",
            ]
        )
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_LOST
            + [
                "kernel_consistency_near_boundaries",
            ]
        )
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: (
            _BASE_DISC_INTRODUCED
            + [
                "kernel_truncation_error",
                "tensile_instability",
            ]
        )
    )

    @property
    def mathematical_form(self) -> str:
        return f"A(r) = sum (m_j/rho_j) A_j W(r-r_j, h), kernel: {self.kernel_function}"

    def kernel(self, r: float, h: float = 0.1) -> float:
        """SPH 核函数."""
        if self.kernel_function == "gaussian" or self.kernel_function == "wendland":
            # 对 wendland 也用高斯作为默认计算核
            return float(np.exp(-0.5 * (r / h) ** 2) / (h * np.sqrt(2 * np.pi)))
        elif self.kernel_function == "cubic_spline":
            q = abs(r) / h
            if q <= 1:
                return (1 - 1.5 * q**2 + 0.75 * q**3) / h
            elif q <= 2:
                return 0.25 * (2 - q) ** 3 / h
            return 0.0
        return 0.0

    def density_estimate(self, positions: np.ndarray, masses: np.ndarray, h: float = 0.1) -> np.ndarray:
        """用 SPH 核函数估计每个粒子位置的密度."""
        n = len(positions)
        rho = np.zeros(n)
        for i in range(n):
            for j in range(n):
                r = abs(positions[i] - positions[j])
                rho[i] += masses[j] * self.kernel(r, h)
        return rho
