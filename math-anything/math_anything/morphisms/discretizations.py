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
        """组装一维刚度矩阵 -u''=f 线性元（三对角，向量化装配）."""
        n = self.n_elements + 1
        h = self.domain_length / self.n_elements
        # 单元局部刚度矩阵 [[1, -1], [-1, 1]] / h，组装到三对角
        main = np.full(n, 1.0 / h)
        # 内部节点被两个单元贡献，端点只被一个单元贡献
        main[0] = 1.0 / h
        main[-1] = 1.0 / h
        if n > 2:
            main[1:-1] = 2.0 / h
        off = np.full(n - 1, -1.0 / h)
        K = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)
        return K

    def assemble_mass_matrix(self) -> np.ndarray:
        """组装一维质量矩阵 线性元（三对角，向量化装配）."""
        n = self.n_elements + 1
        h = self.domain_length / self.n_elements
        # 单元局部质量矩阵 [[2, 1], [1, 2]] * h/6，组装到三对角
        main = np.full(n, 2.0 * h / 6.0)
        main[0] = h / 3.0
        main[-1] = h / 3.0
        if n > 2:
            main[1:-1] = 2.0 * h / 3.0
        off = np.full(n - 1, h / 6.0)
        M = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)
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
    flux_function: Callable = field(default=None)  # type: ignore[assignment]

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
        return "E^{n+1} = E^n - (dt/dx) * (B_{i} - B_{i-1})  on staggered Yee grid"

    @property
    def dx(self) -> float:
        return self.domain_length / self.n_cells

    @property
    def dt(self) -> float:
        # 一维 Yee CFL: dt <= dx/c；用 cfl_number 真正生效（之前是硬编码 dx/(2c)）
        return self.cfl_number * self.dx / self.c

    def compute_cfl_number(self) -> float:
        """计算 Courant 数 c*dt/dx."""
        return self.c * self.dt / self.dx

    def yee_step(self, E: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """一维 Yee 算法单步推进 (c=1 自然单位).

        Maxwell 方程（1D, c=1）:
            ∂B/∂t = -∂E/∂x   (Faraday)
            ∂E/∂t = -∂B/∂x   (Ampère, 无源)

        蛙跳格式下两个更新方程都用减号；之前 E 更新用 += 会导致
        电磁波反向传播或数值不稳定。
        """
        dx = self.dx
        dt = self.dt
        # 更新 B: B^{n+1/2} = B^{n-1/2} - dt/dx * (E^n_{i+1} - E^n_i)
        B_new = B.copy()
        B_new[:-1] -= (dt / dx) * (E[1:] - E[:-1])
        # 更新 E: E^{n+1} = E^n - dt/dx * (B^{n+1/2}_{i} - B^{n+1/2}_{i-1})
        E_new = E.copy()
        E_new[1:] -= (dt / dx) * (B_new[1:] - B_new[:-1])
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
        return D  # type: ignore[no-any-return]

    def fourier_diff_matrix(self, n: int) -> np.ndarray:
        """计算 Fourier 谱微分矩阵.

        Note: 偶数 n 的 Nyquist 模式需特殊处理，此处仅给出标准公式。
        建议传入奇数 n。
        """
        if n <= 0:
            return np.zeros((0, 0))
        idx = np.arange(n)
        # 利用 broadcasting 一次性计算所有 (i, j) 对的差
        diff = idx[:, None] - idx[None, :]  # shape (n, n)
        mask = diff != 0
        D = np.zeros((n, n))
        # 仅在非对角线计算；(-1)**diff 用 np.power 向量化
        D[mask] = 0.5 * np.power(-1.0, diff[mask]) / np.tan(np.pi * diff[mask] / n)
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
        """SPH 核函数（标量）."""
        return float(self._kernel_vec(np.asarray(r, dtype=float), h))

    def _kernel_vec(self, r: np.ndarray, h: float = 0.1) -> np.ndarray:
        """SPH 核函数向量化实现.

        注意：``kernel_function == "wendland"`` 时此处实际返回 Wendland C2
        紧支集核（替代之前误用的高斯核）。Wendland C2 在 |q| >= 2 时为 0，
        满足紧支集性质，再生性条件也与高斯不同。
        """
        q = np.abs(r) / h
        if self.kernel_function == "gaussian":
            return np.exp(-0.5 * (r / h) ** 2) / (h * np.sqrt(2 * np.pi))
        elif self.kernel_function == "wendland":
            # Wendland C2 (2D 归一化)：W(q) = 7/(4π h²) · (1 - q/2)⁴ · (1 + 2q), q < 2
            coef = 7.0 / (4.0 * np.pi * h * h)
            q_clipped = np.clip(q, 0.0, 2.0)
            return coef * (1.0 - q_clipped / 2.0) ** 4 * (1.0 + 2.0 * q_clipped) * (q < 2)
        elif self.kernel_function == "cubic_spline":
            coef = 1.0 / h
            w = np.zeros_like(r, dtype=float)
            m1 = q <= 1
            m2 = (q > 1) & (q <= 2)
            w[m1] = (1 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3) * coef
            w[m2] = 0.25 * (2 - q[m2]) ** 3 * coef
            return w
        return np.zeros_like(r, dtype=float)

    def density_estimate(self, positions: np.ndarray, masses: np.ndarray, h: float = 0.1) -> np.ndarray:
        """用 SPH 核函数估计每个粒子位置的密度.

        实现说明：当粒子数较大（默认 n_particles=10000）时，O(N²) 双层
        Python 循环不可用。这里用向量化 + broadcasting 一次性计算，
        复杂度仍为 O(N²) 但常数远小于 Python 循环。对超大规模建议
        调用方改用 scipy.spatial.cKDTree 仅对邻居求和。
        """
        positions = np.asarray(positions, dtype=float)
        masses = np.asarray(masses, dtype=float)
        n = len(positions)
        if n == 0:
            return np.zeros(0)
        # diff[i, j] = positions[i] - positions[j]
        diff = positions[:, None] - positions[None, :]
        r = np.abs(diff)
        w = self._kernel_vec(r, h)  # shape (n, n)
        # rho[i] = sum_j m_j * W(|r_i - r_j|, h)
        return w @ masses
