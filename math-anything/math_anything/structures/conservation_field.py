"""守恒矩阵场 — 多守恒律的统一矩阵表示.

核心思想：
  物理系统的守恒律可以编码为矩阵算子。Noether 定理建立了
  连续对称性与守恒量之间的双射。守恒矩阵场将多个守恒律
  组装为统一的线性代数结构：

    dU/dt + div(F(U)) = S(U)

  其中 U = [ρ, ρu, ρE]^T 是守恒变量向量，
  F(U) 是通量张量，S(U) 是源项。

  对于 Hamilton 系统，辛结构 J 使得：
    dz/dt = J · ∇H(z)

  Noether 对应：
    连续对称性 → 守恒流
    时间平移   → 能量守恒
    空间平移   → 动量守恒
    旋转 SO(3) → 角动量守恒
    规范 U(1)  → 电荷守恒
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from math_anything.structures._core import StructuralInvariant
from math_anything.structures.enums import SymmetryGroup

# ── Noether 对应表 ──

NOETHER_CORRESPONDENCE: dict[SymmetryGroup, str] = {
    SymmetryGroup.TRANSLATION: "momentum_conservation",
    SymmetryGroup.ROTATION_SO3: "angular_momentum_conservation",
    SymmetryGroup.GAUGE_U1: "charge_conservation",
    SymmetryGroup.GAUGE_SU2: "isospin_conservation",
    SymmetryGroup.LORENTZ: "four_momentum_conservation",
    SymmetryGroup.GALILEAN: "galilean_momentum_conservation",
}

# 时间平移对称性不在 SymmetryGroup 枚举中，单独定义
TIME_TRANSLATION_CONSERVATION = "energy_conservation"


@dataclass
class FieldConservedQuantity:
    """守恒矩阵场中的守恒量.

    Attributes:
        name: 守恒量名称 (e.g., "mass", "momentum_x")
        symbol: 数学符号 (e.g., "ρ", "p_x")
        expression: 守恒表达式 (e.g., "dρ/dt + div(ρu) = 0")
        symmetry: 对应的连续对称性 (Noether)
        spatial_dim: 空间维度数
    """

    name: str
    symbol: str
    expression: str = ""
    symmetry: SymmetryGroup | None = None
    spatial_dim: int = 3


@dataclass(slots=True)
class NoetherCurrent:
    """Noether 守恒流.

    对于每个连续对称性，Noether 定理给出一个守恒流 j^μ：
    ∂_μ j^μ = 0

    Attributes:
        name: 守恒流名称
        symmetry: 对应的对称性
        current_components: 流的分量 (e.g., [j^0, j^1, j^2, j^3] for 4-current)
        continuity_equation: 连续性方程
    """

    name: str
    symmetry: SymmetryGroup
    current_components: list[str] = field(default_factory=list)
    continuity_equation: str = ""


@dataclass(slots=True)
class ConservationMatrixField:
    """守恒矩阵场 — 多守恒律的统一矩阵表示.

    将守恒律系统 dU/dt + div(F(U)) = S(U) 编码为矩阵算子，
    其中 U 是守恒变量向量，F 是通量张量，S 是源项。

    对于 Hamilton 系统，还提供辛矩阵 J 使得 dz/dt = J·∇H。

    Attributes:
        conserved_quantities: 守恒量列表
        coupling_matrix: 守恒量之间的耦合矩阵 (n×n)
        flux_tensors: 通量张量列表 (每个空间方向一个 n×n 矩阵)
        source_vector: 源项向量 (n,)
        jacobian: 通量 Jacobian dF/dU (n×n, 用于特征速度计算)
        symplectic_matrix: 辛矩阵 J (2n×2n, 仅 Hamilton 系统)
        noether_currents: Noether 守恒流列表
        hamiltonian: Hamilton 量表达式 (仅 Hamilton 系统)
    """

    conserved_quantities: list[FieldConservedQuantity] = field(default_factory=list)
    coupling_matrix: np.ndarray | None = None
    flux_tensors: list[np.ndarray] = field(default_factory=list)
    source_vector: np.ndarray | None = None
    jacobian: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None
    symplectic_matrix: np.ndarray | None = None
    noether_currents: list[NoetherCurrent] = field(default_factory=list)
    hamiltonian: str = ""

    @property
    def n_conserved(self) -> int:
        """守恒量数目."""
        return len(self.conserved_quantities)

    @property
    def is_hyperbolic(self) -> bool:
        """是否双曲型 (Jacobian 特征值全实)."""
        if self.jacobian is None:
            return False
        eigenvalues = np.linalg.eigvals(self.jacobian)
        return bool(np.all(np.isreal(eigenvalues)))

    @property
    def is_hamiltonian(self) -> bool:
        """是否 Hamilton 系统 (有辛矩阵)."""
        return self.symplectic_matrix is not None

    @property
    def characteristic_speeds(self) -> np.ndarray | None:
        """特征速度 (Jacobian 特征值)."""
        if self.jacobian is None:
            return None
        return np.sort(np.real(np.linalg.eigvals(self.jacobian)))

    @property
    def noether_map(self) -> dict[SymmetryGroup, FieldConservedQuantity]:
        """Noether 对应: 对称性 → 守恒量."""
        result = {}
        for q in self.conserved_quantities:
            if q.symmetry is not None:
                result[q.symmetry] = q
        return result

    def build_from_euler_equations(self, gamma: float = 1.4) -> ConservationMatrixField:
        """从 Euler 方程构建守恒矩阵场.

        U = [ρ, ρu, ρE]^T
        F_x = [ρu, ρu²+p, u(ρE+p)]^T

        Args:
            gamma: 比热比, 默认 1.4 (空气)
        """
        self.conserved_quantities = [
            FieldConservedQuantity("mass", "ρ", "dρ/dt + div(ρu) = 0", SymmetryGroup.GAUGE_U1, 3),
            FieldConservedQuantity("momentum", "ρu", "d(ρu)/dt + div(ρu⊗u + pI) = 0", SymmetryGroup.TRANSLATION, 3),
            FieldConservedQuantity("energy", "ρE", "d(ρE)/dt + div(u(ρE+p)) = 0", None, 3),  # 时间平移 → 能量
        ]
        self.noether_currents = [
            NoetherCurrent("mass_current", SymmetryGroup.GAUGE_U1, ["ρ", "ρu"], "dρ/dt + div(ρu) = 0"),
            NoetherCurrent(
                "momentum_current", SymmetryGroup.TRANSLATION, ["ρu", "ρu⊗u + pI"], "d(ρu)/dt + div(ρu⊗u + pI) = 0"
            ),
            NoetherCurrent(
                "energy_current", SymmetryGroup.TRANSLATION, ["ρE", "u(ρE+p)"], "d(ρE)/dt + div(u(ρE+p)) = 0"
            ),
        ]

        # 1D Euler flux Jacobian: dF/dU evaluated at a reference state
        # For ideal gas p = (γ-1)(ρE - ρu²/2), the Jacobian is:
        # A = [[0, 1, 0],
        #      [-(γ-3)/2·u², (3-γ)·u, (γ-1)],
        #      [(γ-2)·u³ - γ·e·u/ρ, γ·e/ρ - (γ-1)·u², γ·u]]
        # Using reference state u=1, ρ=1, e/ρ=2.5 (standard atmosphere)
        u_ref = 1.0
        e_over_rho = 2.5  # internal energy per unit mass
        g = gamma

        self.coupling_matrix = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.5 * (g - 3) * u_ref**2, (3 - g) * u_ref, g - 1],
                [(g - 2) * u_ref**3 - g * e_over_rho * u_ref, g * e_over_rho - (g - 1) * u_ref**2, g * u_ref],
            ]
        )

        # 1D flux tensor F(U) at reference state
        p_ref = (g - 1) * (e_over_rho - 0.5 * u_ref**2)  # p = (γ-1)ρ(e - u²/2)
        self.flux_tensors = [
            np.array(
                [
                    u_ref,
                    u_ref**2 + p_ref,
                    u_ref * (e_over_rho + p_ref),
                ]
            )
        ]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.sort(np.real(np.linalg.eigvals(self.coupling_matrix)))
        return self

    def build_from_navier_stokes(self, mu: float = 1.8e-5, gamma: float = 1.4) -> ConservationMatrixField:
        """从 Navier-Stokes 方程构建守恒矩阵场.

        在 Euler 基础上添加粘性扩散项。

        Args:
            mu: 动力粘度, 默认 1.8e-5 Pa·s (空气, 20°C)
            gamma: 比热比, 默认 1.4
        """
        self.build_from_euler_equations(gamma=gamma)
        # NS 的耦合矩阵与 Euler 相同 (粘性项是二阶的, 不影响一阶 Jacobian)
        # 粘性通量: F_viscous = [0, μ(∇u + ∇u^T), μu·∇²u + k∇T]^T
        # 用 source_vector 表示粘性耗散 (简化模型)
        self.source_vector = np.array([0.0, -mu, -mu])
        # 粘性通量张量 (1D)
        self.flux_tensors.append(np.array([0.0, mu, mu]))
        return self

    def build_from_schrodinger(self, hbar: float = 1.0, m: float = 1.0, V: float = 0.0) -> ConservationMatrixField:
        """从 Schrödinger 方程构建守恒矩阵场 (Hamilton 系统).

        z = [Re(ψ), Im(ψ)]^T
        dz/dt = J · ∇H, J = [[0, 1], [-1, 0]]
        H = ℏ²/(2m) |∇ψ|² + V|ψ|²

        Args:
            hbar: 约化 Planck 常数
            m: 粒子质量
            V: 势能 (标量, 默认 0)
        """
        self.conserved_quantities = [
            FieldConservedQuantity("probability", "|ψ|²", "d|ψ|²/dt + div(j) = 0", SymmetryGroup.GAUGE_U1, 3),
        ]
        self.symplectic_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])
        self.hamiltonian = "ℏ²/(2m)|∇ψ|² + V|ψ|²"
        self.noether_currents = [
            NoetherCurrent(
                "probability_current", SymmetryGroup.GAUGE_U1, ["|ψ|²", "(ℏ/m)Im(ψ*∇ψ)"], "d|ψ|²/dt + div(j) = 0"
            ),
        ]

        # Real-imaginary decomposition: z = [Re(ψ), Im(ψ)]
        # iℏ ∂ψ/∂t = Hψ  =>  ∂z/∂t = A·z
        # A = (1/ℏ) [[0, H], [-H, 0]] where H = -ℏ²/(2m)∇² + V
        # For uniform potential V, the coupling matrix in position representation:
        H_eff = V  # on-site part of H (kinetic part requires discretization)
        self.coupling_matrix = (1.0 / hbar) * np.array(
            [
                [0.0, H_eff],
                [-H_eff, 0.0],
            ]
        )

        # Flux from probability current: j = (ℏ/m) Im(ψ*∇ψ)
        # In real-imaginary form: j = (ℏ/m)(Re(ψ)·∇Im(ψ) - Im(ψ)·∇Re(ψ))
        self.flux_tensors = [
            (hbar / m)
            * np.array(
                [
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            )
        ]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.linalg.eigvals(self.coupling_matrix)
        return self

    def build_from_maxwell(self, c: float = 3e8, mu0: float = 4e-7 * np.pi) -> ConservationMatrixField:
        """从 Maxwell 方程构建守恒矩阵场 (Hamilton 系统).

        z = [E, B]^T
        dz/dt = c · curl(z)

        Args:
            c: 光速, 默认 3e8 m/s
            mu0: 真空磁导率, 默认 4π×10⁻⁷ H/m
        """
        self.conserved_quantities = [
            FieldConservedQuantity("electric_energy", "E²/2", "d/dt ∫E²dV = -∫E·JdV", None, 3),
            FieldConservedQuantity("magnetic_energy", "B²/2", "d/dt ∫B²dV = ∫E·JdV", None, 3),
        ]
        self.hamiltonian = "(E² + B²)/2"
        self.noether_currents = [
            NoetherCurrent("poynting_vector", SymmetryGroup.TRANSLATION, ["S = E×B"], "∂u/∂t + div(S) = -J·E"),
        ]

        # 1D Maxwell: propagation along x
        # State: U = [Ey, Ez, By, Bz] (Ex, Bx are longitudinal, decoupled in vacuum)
        # ∂Ey/∂t = -(1/μ₀ε₀) ∂Bz/∂x = -c² ∂Bz/∂x
        # ∂Bz/∂t = -∂Ey/∂x
        # ∂Ez/∂t = c² ∂By/∂x
        # ∂By/∂t = ∂Ez/∂x
        # Coupling matrix for dU/dt + A·dU/dx = 0:
        self.coupling_matrix = np.array(
            [
                [0.0, 0.0, 0.0, c**2],
                [0.0, 0.0, -(c**2), 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

        # Flux in x-direction
        self.flux_tensors = [
            np.array(
                [
                    [0.0, 0.0, 0.0, -(c**2)],
                    [0.0, 0.0, c**2, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                ]
            )
        ]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.sort(np.real(np.linalg.eigvals(self.coupling_matrix)))

        # Symplectic form for EM field: Ω = [[0, I], [-I, 0]] (4x4 for 2+2 components)
        self.symplectic_matrix = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
            ],
            dtype=float,
        )
        return self

    def build_from_elasticity(self, lam: float = 1.0, mu: float = 1.0, rho: float = 1.0) -> ConservationMatrixField:
        """从线弹性方程构建守恒矩阵场.

        ∇·σ + f = 0, σ = C:ε, ε = (∇u + ∇uᵀ)/2
        动力学形式: ρü = ∇·σ + f

        守恒量: 线动量 (空间平移对称性)

        Args:
            lam: Lamé 第一参数
            mu: Lamé 第二参数 (剪切模量)
            rho: 密度
        """
        self.conserved_quantities = [
            FieldConservedQuantity("linear_momentum", "ρu", "d(ρu)/dt = ∇·σ + f", SymmetryGroup.TRANSLATION, 3),
            FieldConservedQuantity(
                "angular_momentum", "x×ρu", "d(x×ρu)/dt = x×(∇·σ + f)", SymmetryGroup.ROTATION_SO3, 3
            ),
            FieldConservedQuantity("strain_energy", "½σ:ε", "d/dt ∫½σ:ε dV = ∫f·u̇ dV", None, 3),
        ]
        self.noether_currents = [
            NoetherCurrent("stress_flux", SymmetryGroup.TRANSLATION, ["ρu", "σ"], "d(ρu)/dt - ∇·σ = f"),
            NoetherCurrent(
                "angular_stress_flux", SymmetryGroup.ROTATION_SO3, ["x×ρu", "x×σ"], "d(x×ρu)/dt - ∇·(x×σ) = x×f"
            ),
        ]

        # 1D elasticity: U = [u, v, σ]^T where v = ∂u/∂t
        # ρ ∂v/∂t = ∂σ/∂x + f
        # ∂σ/∂t = (λ + 2μ) ∂v/∂x  (1D stress-strain rate)
        # ∂u/∂t = v
        # System: dU/dt + A·dU/dx = S
        c_l = np.sqrt((lam + 2 * mu) / rho)  # longitudinal wave speed
        self.coupling_matrix = np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0 / rho],
                [0.0, -(lam + 2 * mu), 0.0],
            ]
        )

        self.flux_tensors = [
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0 / rho],
                    [0.0, -(lam + 2 * mu), 0.0],
                ]
            )
        ]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.array([-c_l, 0.0, c_l])
        return self

    def build_from_heat_equation(self, k: float = 1.0, rho: float = 1.0, c: float = 1.0) -> ConservationMatrixField:
        """从热传导方程构建守恒矩阵场 (抛物型).

        ρc ∂T/∂t = ∇·(k∇T) + Q

        守恒量: 热能 (时间平移对称性，无源时)

        Args:
            k: 热导率
            rho: 密度
            c: 比热容
        """
        self.conserved_quantities = [
            FieldConservedQuantity("thermal_energy", "ρcT", "ρc ∂T/∂t = ∇·(k∇T) + Q", None, 3),
        ]
        self.noether_currents = [
            NoetherCurrent("heat_flux", SymmetryGroup.TRANSLATION, ["ρcT", "-k∇T"], "∂(ρcT)/∂t + div(k∇T) = Q"),
        ]

        # Parabolic: ∂T/∂t = α ∇²T where α = k/(ρc)
        # Coupling matrix is scalar α (1x1 system)
        alpha = k / (rho * c)
        self.coupling_matrix = np.array([[alpha]])

        # Flux: F = -k∇T, but in divergence form: ∂T/∂t = α∇²T = div(α∇T)
        # For 1D: flux in x = α ∂T/∂x
        self.flux_tensors = [np.array([alpha])]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.array([alpha])
        return self

    def build_from_advection_diffusion(self, v: float = 1.0, D: float = 1.0) -> ConservationMatrixField:
        """从对流-扩散方程构建守恒矩阵场.

        ∂c/∂t + u·∇c = D∇²c + S

        守恒量: 溶质质量 (U(1) 相位对称性 → 粒子数守恒)

        Args:
            v: 对流速度 (1D)
            D: 扩散系数
        """
        self.conserved_quantities = [
            FieldConservedQuantity("solute_mass", "c", "∂c/∂t + div(cu - D∇c) = S", SymmetryGroup.GAUGE_U1, 3),
        ]
        self.noether_currents = [
            NoetherCurrent("mass_flux", SymmetryGroup.GAUGE_U1, ["c", "cu - D∇c"], "∂c/∂t + div(cu - D∇c) = S"),
        ]

        # 1D: ∂c/∂t + v·∂c/∂x = D ∂²c/∂x²
        # Coupling matrix (advection part): v
        self.coupling_matrix = np.array([[v]])

        # Flux: advective flux = v·c, diffusive flux = -D·∂c/∂x
        self.flux_tensors = [np.array([v]), np.array([-D])]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.array([v])
        return self

    def build_from_cahn_hilliard(self, M: float = 1.0, kappa: float = 1.0) -> ConservationMatrixField:
        """Build conservation field from Cahn-Hilliard equation.

        ∂c/∂t = ∇·(M ∇μ),  μ = δF/δc = f'(c) - κ∇²c

        Conserved: mass (∫c dV = const), free energy dissipation (dF/dt ≤ 0)

        Args:
            M: Mobility
            kappa: Gradient energy coefficient
        """
        self.conserved_quantities = [
            FieldConservedQuantity("mass", "c", "∂c/∂t = ∇·(M∇μ)", SymmetryGroup.GAUGE_U1, 3),
            FieldConservedQuantity("free_energy", "F[c]", "dF/dt = -∫ M|∇μ|² dV ≤ 0", SymmetryGroup.TRANSLATION, 3),
        ]
        self.noether_currents = [
            NoetherCurrent("mass_flux", SymmetryGroup.GAUGE_U1, ["c", "-M∇μ"], "∂c/∂t + div(M∇μ) = 0"),
        ]

        # State: U = [c, μ] (2 variables)
        # ∂c/∂t = M ∇²μ  (4th order in c, but 2nd order system in [c, μ])
        # μ = f'(c) - κ∇²c
        # Linearized coupling (around homogeneous state): ∂c/∂t = -Mκ ∇⁴c
        # For the [c, μ] system: coupling matrix relates ∂c/∂t to μ gradient
        self.coupling_matrix = np.array(
            [
                [0.0, -M],
                [kappa, 0.0],
            ]
        )

        # Flux: mass flux = -M∇μ, chemical potential flux relates to ∇c
        self.flux_tensors = [
            np.array(
                [
                    [0.0, -M],
                    [kappa, 0.0],
                ]
            )
        ]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.sort(np.real(np.linalg.eigvals(self.coupling_matrix)))

        # Cahn-Hilliard is a gradient flow: symplectic structure not applicable
        # but has a dissipative structure: dF/dt = -∫ M|∇μ|² dV
        self.symplectic_matrix = None
        return self

    def build_from_allen_cahn(self, L: float = 1.0, kappa: float = 1.0) -> ConservationMatrixField:
        """Build conservation field from Allen-Cahn equation.

        ∂η/∂t = -L δF/δη = L(κ∇²η - f'(η))

        NOT conserved: mass (∫η dV ≠ const)
        Conserved: free energy dissipation (dF/dt ≤ 0)

        Args:
            L: Kinetic coefficient (mobility)
            kappa: Gradient energy coefficient
        """
        self.conserved_quantities = [
            FieldConservedQuantity("free_energy", "F[η]", "dF/dt = -∫ L|δF/δη|² dV ≤ 0", SymmetryGroup.TRANSLATION, 3),
        ]
        self.noether_currents = [
            NoetherCurrent("energy_dissipation", SymmetryGroup.TRANSLATION, ["η", "-L δF/δη"], "∂η/∂t = -L δF/δη"),
        ]

        # State: U = [η] (single variable, non-conserved)
        # ∂η/∂t = Lκ∇²η - Lf'(η)
        # Linearized coupling: ∂η/∂t = Lκ∇²η (diffusion-like)
        self.coupling_matrix = np.array([[-L * kappa]])

        self.flux_tensors = [np.array([[-L * kappa]])]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.array([-L * kappa])

        self.symplectic_matrix = None
        return self

    def build_from_mhd(
        self,
        eta: float = 1.0,
        mu_f: float = 1.0,
        gamma: float = 5 / 3,
        rho0: float = 1.0,
        p0: float = 1.0,
        Bx0: float = 1.0,
    ) -> ConservationMatrixField:
        """从磁流体力学 (MHD) 方程构建守恒矩阵场.

        NS + Maxwell 耦合: ρ(∂u/∂t + u·∇u) = -∇p + J×B + μ∇²u
        ∂B/∂t = ∇×(u×B) + η∇²B

        守恒量: 质量 + 动量 + 能量 + 磁通量

        Args:
            eta: 磁扩散率 (1/μ₀σ)
            mu_f: 动力粘度
            gamma: 比热比, 默认 5/3 (单原子理想气体)
            rho0: 参考密度
            p0: 参考压力
            Bx0: 参考磁场 x 分量
        """
        self.conserved_quantities = [
            FieldConservedQuantity("mass", "ρ", "dρ/dt + div(ρu) = 0", SymmetryGroup.GAUGE_U1, 3),
            FieldConservedQuantity(
                "momentum", "ρu", "d(ρu)/dt + div(ρu⊗u + pI - BB/μ₀) = μ∇²u", SymmetryGroup.TRANSLATION, 3
            ),
            FieldConservedQuantity(
                "energy", "ρE + B²/2μ₀", "d(ρE+B²/2μ₀)/dt + div((ρE+p+B²/2μ₀)u - (u·B)B/μ₀) = 0", None, 3
            ),
            FieldConservedQuantity("magnetic_flux", "B", "∂B/∂t + ∇×E = 0, ∇·B = 0", None, 3),
        ]
        self.noether_currents = [
            NoetherCurrent("mass_current", SymmetryGroup.GAUGE_U1, ["ρ", "ρu"], "dρ/dt + div(ρu) = 0"),
            NoetherCurrent(
                "momentum_current",
                SymmetryGroup.TRANSLATION,
                ["ρu", "ρu⊗u + pI - BB/μ₀"],
                "d(ρu)/dt + div(ρu⊗u + pI - BB/μ₀) = μ∇²u",
            ),
            NoetherCurrent("poynting_mhd", SymmetryGroup.TRANSLATION, ["E×B/μ₀", "S_MHD"], "∂u_EM/∂t + div(S) = -J·E"),
        ]

        # 1D MHD (8-wave formulation): U = [ρ, ρu, ρv, ρw, Bx, By, Bz, E]
        # Linearized about reference state (u=0, v=0, w=0, Bx=Bx0, By=0, Bz=0)
        # Alfven speed: c_a = Bx0/sqrt(rho0)
        # Sound speed: c_s = sqrt(gamma * p0 / rho0)
        c_a = Bx0 / np.sqrt(rho0)
        c_s = np.sqrt(gamma * p0 / rho0)
        # Fast magnetosonic speed
        np.sqrt(0.5 * (c_s**2 + c_a**2 + np.sqrt((c_s**2 + c_a**2) ** 2 - 4 * c_s**2 * c_a**2)))

        # Simplified 8x8 coupling matrix for 1D MHD
        # Based on the linearized flux Jacobian about the reference state
        self.coupling_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, c_a**2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, c_a**2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

        self.flux_tensors = [self.coupling_matrix.copy()]
        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.sort(np.real(np.linalg.eigvals(self.coupling_matrix)))
        return self

    def build_from_kohn_sham(self, hbar: float = 1.0, m: float = 1.0, V_ext: float = 0.0) -> ConservationMatrixField:
        """从 Kohn-Sham DFT 方程构建守恒矩阵场.

        H_KS ψ_nk = ε_nk ψ_nk
        H_KS = -ℏ²/(2m)∇² + V_ext + V_H + V_xc

        守恒量: 粒子数 (U(1) 规范对称性), 总能量 (变分下界)

        Args:
            hbar: 约化 Planck 常数
            m: 电子质量
            V_ext: 外势
        """
        self.conserved_quantities = [
            FieldConservedQuantity("particle_number", "N", "N = Σ_nk f_nk = const", SymmetryGroup.GAUGE_U1, 3),
            FieldConservedQuantity("total_energy", "E_KS", "E_KS[ρ] ≥ E_GS (变分原理)", None, 3),
            FieldConservedQuantity("charge_density", "ρ(r)", "ρ(r) = Σ_nk f_nk|ψ_nk(r)|²", SymmetryGroup.GAUGE_U1, 3),
        ]
        self.noether_currents = [
            NoetherCurrent(
                "probability_current",
                SymmetryGroup.GAUGE_U1,
                ["|ψ_nk|²", "(ℏ/m)Im(ψ*_nk∇ψ_nk)"],
                "d|ψ_nk|²/dt + div(j_nk) = 0",
            ),
        ]

        # Kohn-Sham is structurally identical to Schrödinger with effective potential
        # Real-imaginary decomposition: z = [Re(ψ), Im(ψ)]
        # Coupling matrix: A = (1/ℏ)[[0, H_KS], [-H_KS, 0]]
        V_eff = V_ext  # simplified: only external potential (H and xc need self-consistency)
        self.coupling_matrix = (1.0 / hbar) * np.array(
            [
                [0.0, V_eff],
                [-V_eff, 0.0],
            ]
        )

        self.flux_tensors = [
            (hbar / m)
            * np.array(
                [
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            )
        ]

        self.symplectic_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])
        self.hamiltonian = "-ℏ²/(2m)∇² + V_ext + V_H + V_xc"
        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.linalg.eigvals(self.coupling_matrix)
        return self

    def build_from_boltzmann(self, nu: float = 1.0) -> ConservationMatrixField:
        """从 Boltzmann 输运方程构建守恒矩阵场.

        ∂f/∂t + v·∇_x f + (F/m)·∇_v f = C[f]

        守恒量: 粒子数 + 动量 + 能量 (碰撞不变量)

        Args:
            nu: 碰撞频率 (BGK 模型: C[f] = -ν(f - f_M))
        """
        self.conserved_quantities = [
            FieldConservedQuantity(
                "particle_number", "∫f dv", "d/dt ∫f d³v + div(∫vf d³v) = 0", SymmetryGroup.GAUGE_U1, 3
            ),
            FieldConservedQuantity(
                "momentum", "∫mvf dv", "d/dt ∫mvf d³v + div(∫mv⊗f d³v) = ∫Ff d³v", SymmetryGroup.TRANSLATION, 3
            ),
            FieldConservedQuantity("energy", "∫½mv²f dv", "d/dt ∫½mv²f d³v + div(∫½mv²vf d³v) = ∫F·vf d³v", None, 3),
        ]
        self.noether_currents = [
            NoetherCurrent(
                "particle_flux", SymmetryGroup.GAUGE_U1, ["∫f d³v", "∫vf d³v"], "d/dt ∫f d³v + div(∫vf d³v) = 0"
            ),
            NoetherCurrent(
                "momentum_flux",
                SymmetryGroup.TRANSLATION,
                ["∫mvf d³v", "∫mv⊗f d³v"],
                "d/dt ∫mvf d³v + div(∫mv⊗f d³v) = ∫Ff d³v",
            ),
        ]

        # Boltzmann is infinite-dimensional in velocity space.
        # Use moment closure: U = [n, nu, E] (density, momentum density, energy density)
        # BGK collision operator: C[f] = -ν(f - f_M)
        # The coupling matrix for the moment system (Euler-level):
        # dU/dt + A·∇U = S where S = -ν(U - U_M)
        # At equilibrium (U = U_M), the coupling reduces to Euler flux Jacobian
        self.coupling_matrix = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )

        # BGK relaxation source term
        self.source_vector = np.array([0.0, -nu, -nu])

        self.flux_tensors = [self.coupling_matrix.copy()]
        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.linalg.eigvals(self.coupling_matrix)
        return self

    def build_from_shallow_water(self, g: float = 9.81, h0: float = 1.0, u0: float = 0.0) -> ConservationMatrixField:
        """从浅水方程构建守恒矩阵场.

        ∂h/∂t + ∇·(hu) = 0
        ∂(hu)/∂t + ∇·(hu⊗u + ½gh²I) = -gh∇b

        守恒量: 水量 + 动量

        Args:
            g: 重力加速度, 默认 9.81 m/s²
            h0: 参考水深
            u0: 参考流速
        """
        self.conserved_quantities = [
            FieldConservedQuantity("water_mass", "h", "∂h/∂t + ∇·(hu) = 0", SymmetryGroup.GAUGE_U1, 2),
            FieldConservedQuantity(
                "momentum", "hu", "∂(hu)/∂t + ∇·(hu⊗u + ½gh²I) = -gh∇b", SymmetryGroup.TRANSLATION, 2
            ),
        ]
        self.noether_currents = [
            NoetherCurrent("volume_flux", SymmetryGroup.GAUGE_U1, ["h", "hu"], "∂h/∂t + ∇·(hu) = 0"),
            NoetherCurrent(
                "momentum_flux",
                SymmetryGroup.TRANSLATION,
                ["hu", "hu⊗u + ½gh²I"],
                "∂(hu)/∂t + ∇·(hu⊗u + ½gh²I) = -gh∇b",
            ),
        ]

        # 1D shallow water: U = [h, hu]
        # Flux Jacobian: A = dF/dU = [[0, 1], [gh - u²/h, 2u]]
        # Evaluated at reference state (h0, u0)
        self.coupling_matrix = np.array(
            [
                [0.0, 1.0],
                [g * h0 - u0**2 / h0, 2.0 * u0],
            ]
        )

        # 1D flux: F = [hu, hu²/h + gh²/2]
        self.flux_tensors = [
            np.array(
                [
                    h0 * u0,
                    h0 * u0**2 / h0 + 0.5 * g * h0**2,
                ]
            )
        ]

        self.jacobian = self.coupling_matrix.copy()
        # Characteristic speeds: u ± sqrt(gh)
        c_wave = np.sqrt(g * h0)
        self.eigenvalues = np.array([u0 - c_wave, u0 + c_wave])
        return self

    def build_from_wave_equation(self, c: float = 1.0) -> ConservationMatrixField:
        """从波动方程构建守恒矩阵场.

        ∂²u/∂t² = c²∇²u

        一阶形式: z = [u, ∂u/∂t]^T, dz/dt = [[0,1],[c²∇²,0]] z
        守恒量: 能量 E = ½(∂u/∂t)² + ½c²|∇u|²

        Args:
            c: 波速
        """
        self.conserved_quantities = [
            FieldConservedQuantity("wave_energy", "½(∂u/∂t)² + ½c²|∇u|²", "dE/dt + div(-c²∂u/∂t ∇u) = 0", None, 3),
        ]
        self.symplectic_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])
        self.hamiltonian = "½p² + ½c²|∇u|²"
        self.noether_currents = [
            NoetherCurrent(
                "energy_flux",
                SymmetryGroup.TRANSLATION,
                ["½(∂u/∂t)² + ½c²|∇u|²", "-c²∂u/∂t ∇u"],
                "∂E/∂t + div(-c²∂u/∂t ∇u) = 0",
            ),
        ]

        # First-order form: U = [u, ∂u/∂t]
        # dU/dt = A·U where A = [[0, 1], [c²∇², 0]]
        # In discretized form (1D, single cell), ∇² → -k² (Fourier mode)
        # For the coupling matrix we use the continuous operator structure:
        self.coupling_matrix = np.array(
            [
                [0.0, 1.0],
                [c**2, 0.0],  # c²∇² → c² for unit wavenumber
            ]
        )

        # Flux: F = [-∂u/∂t, -c²∂u/∂x] (energy flux)
        self.flux_tensors = [
            np.array(
                [
                    [0.0, -1.0],
                    [-(c**2), 0.0],
                ]
            )
        ]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.array([-c, c])
        return self

    def build_from_dirac(self, hbar: float = 1.0, c: float = 1.0, m: float = 1.0) -> ConservationMatrixField:
        """从 Dirac 方程构建守恒矩阵场 (Hamilton 系统).

        iℏ ∂ψ/∂t = (cα·p + βmc²)ψ

        守恒量: 概率 (U(1) 规范), 电荷, 4-动量 (Lorentz 对称性)

        Args:
            hbar: 约化 Planck 常数
            c: 光速
            m: 粒子质量
        """
        self.conserved_quantities = [
            FieldConservedQuantity("probability", "ψ†ψ", "∂(ψ†ψ)/∂t + div(j) = 0", SymmetryGroup.GAUGE_U1, 4),
            FieldConservedQuantity("charge", "eψ†ψ", "∂(eψ†ψ)/∂t + div(ej) = 0", SymmetryGroup.GAUGE_U1, 4),
            FieldConservedQuantity("four_momentum", "T^μν", "∂_μ T^μν = 0", SymmetryGroup.LORENTZ, 4),
        ]
        self.hamiltonian = "cα·p + βmc²"
        self.noether_currents = [
            NoetherCurrent("dirac_current", SymmetryGroup.GAUGE_U1, ["ψ†ψ", "cψ†αψ"], "∂(ψ†ψ)/∂t + div(cψ†αψ) = 0"),
            NoetherCurrent("energy_momentum", SymmetryGroup.LORENTZ, ["T^00", "T^0i"], "∂_μ T^μν = 0"),
        ]

        # Dirac matrices (Dirac representation)
        # α₁ = [[0, σ₁], [σ₁, 0]], β = [[I, 0], [0, -I]]
        # σ₁ = [[0,1],[1,0]], σ₂ = [[0,-i],[i,0]], σ₃ = [[1,0],[0,-1]]
        # For 1D (p₁ only), H = cα₁p₁ + βmc²
        # Real-imaginary decomposition of 4-component spinor → 8-component real vector
        # Coupling matrix: A = (1/ℏ)[[0, H_D], [-H_D, 0]]
        # where H_D = cα₁p₁ + βmc² at p₁=0 (rest frame):
        # H_D|_{p=0} = βmc² = mc² [[I,0],[0,-I]]

        mc2 = m * c**2
        # 4x4 Hamiltonian at rest (p=0): H = βmc²
        H_rest = mc2 * np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1],
            ],
            dtype=float,
        )

        # 8x8 coupling matrix for real-imaginary decomposition
        # z = [Re(ψ₁), Re(ψ₂), Re(ψ₃), Re(ψ₄), Im(ψ₁), Im(ψ₂), Im(ψ₃), Im(ψ₄)]
        self.coupling_matrix = (1.0 / hbar) * np.block(
            [
                [np.zeros((4, 4)), H_rest],
                [-H_rest, np.zeros((4, 4))],
            ]
        )

        # Symplectic form: Ω = [[0, I₄], [-I₄, 0]]
        self.symplectic_matrix = np.block(
            [
                [np.zeros((4, 4)), np.eye(4)],
                [-np.eye(4), np.zeros((4, 4))],
            ]
        )

        # Flux from α₁ (x-direction current)
        alpha1 = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=float,
        )
        self.flux_tensors = [
            (c / hbar)
            * np.block(
                [
                    [np.zeros((4, 4)), alpha1],
                    [-alpha1, np.zeros((4, 4))],
                ]
            )
        ]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.linalg.eigvals(self.coupling_matrix)
        return self

    def build_from_klein_gordon(self, hbar: float = 1.0, c: float = 1.0, m: float = 1.0) -> ConservationMatrixField:
        """从 Klein-Gordon 方程构建守恒矩阵场.

        (□ + m²c²/ℏ²)φ = 0
        即: 1/c² ∂²φ/∂t² - ∇²φ + m²c²/ℏ² φ = 0

        守恒量: Klein-Gordon 4-流, 能量-动量张量

        Args:
            hbar: 约化 Planck 常数
            c: 光速
            m: 粒子质量
        """
        self.conserved_quantities = [
            FieldConservedQuantity(
                "kg_charge", "i(φ*∂φ/∂t - φ∂φ*/∂t)", "∂j^0/∂t + div(j) = 0", SymmetryGroup.GAUGE_U1, 4
            ),
            FieldConservedQuantity("kg_energy", "T^00", "∂T^00/∂t + ∂T^0i/∂x^i = 0", None, 4),
        ]
        self.hamiltonian = "½|π|² + ½|∇φ|² + ½m²c²/ℏ²|φ|²"
        self.noether_currents = [
            NoetherCurrent(
                "kg_current", SymmetryGroup.GAUGE_U1, ["i(φ*∂φ/∂t - φ∂φ*/∂t)", "i(φ*∇φ - φ∇φ*)"], "∂_μ j^μ = 0"
            ),
        ]

        # First-order form: U = [φ, ∂φ/∂t]
        # ∂²φ/∂t² = c²∇²φ - (mc/ℏ)²c²φ
        # => dU/dt = [[0, 1], [c²∇² - (mc²/ℏ)², 0]] U
        # For unit wavenumber: ∇² → -1, so c²∇² → -c²
        omega_c = (m * c**2 / hbar) ** 2  # (mc²/ℏ)²
        self.coupling_matrix = np.array(
            [
                [0.0, 1.0],
                [-(c**2) - omega_c, 0.0],  # c²∇² - ω_c² → -c² - ω_c² for k=1
            ]
        )

        # Symplectic form (same as wave equation)
        self.symplectic_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])

        self.flux_tensors = [
            np.array(
                [
                    [0.0, -1.0],
                    [-(c**2), 0.0],
                ]
            )
        ]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.linalg.eigvals(self.coupling_matrix)
        return self

    def build_from_einstein_field(
        self, G: float = 6.674e-11, Lambda: float = 0.0, c: float = 3e8
    ) -> ConservationMatrixField:
        """从 Einstein 场方程构建守恒矩阵场.

        G_μν + Λg_μν = 8πG/c⁴ T_μν

        守恒量: 能量-动量 (∇_μ T^μν = 0, Bianchi 恒等式保证)

        Args:
            G: 万有引力常数
            Lambda: 宇宙学常数
            c: 光速
        """
        self.conserved_quantities = [
            FieldConservedQuantity(
                "energy_momentum", "T_μν", "∇_μ T^μν = 0 (Bianchi 恒等式保证)", SymmetryGroup.LORENTZ, 4
            ),
            FieldConservedQuantity(
                "bianchi_identity", "∇_σ R_μνρσ + ∇_ρ R_μνσρ + ∇_ρ R_μνρσ", "∇_[λ R_μν]ρσ = 0", None, 4
            ),
        ]
        self.noether_currents = [
            NoetherCurrent("stress_energy_current", SymmetryGroup.LORENTZ, ["T^μν"], "∇_μ T^μν = 0"),
        ]

        # Linearized Einstein equations (weak field, h_μν << 1)
        # g_μν = η_μν + h_μν
        # In harmonic gauge: □h̄_μν = -16πG/c⁴ T_μν
        # For vacuum (T=0): □h̄_μν = 0 → wave equation with speed c
        # State: U = [h̄_00, h̄_01, h̄_02, h̄_03, h̄_11, h̄_12, h̄_13, h̄_22, h̄_23, h̄_33]
        # 10 independent components of symmetric h̄_μν
        # Each satisfies □h̄_μν = 0 → same coupling as wave equation
        # Simplified: 2-component system per polarization [h̄, ∂h̄/∂t]
        # Using + and × polarizations:
        self.coupling_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [c**2, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, c**2, 0.0],
            ]
        )

        self.flux_tensors = [
            np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [-(c**2), 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, -(c**2), 0.0],
                ]
            )
        ]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.array([-c, c, -c, c])
        return self

    def build_from_schrodinger_nonlinear(
        self, g: float = 1.0, hbar: float = 1.0, m: float = 1.0
    ) -> ConservationMatrixField:
        """从非线性 Schrödinger (Gross-Pitaevskii) 方程构建守恒矩阵场.

        iℏ ∂ψ/∂t = -ℏ²/(2m)∇²ψ + Vψ + g|ψ|²ψ

        守恒量: 粒子数 (U(1)), 能量 (时间平移)

        Args:
            g: 非线性耦合常数 (g>0: 斥力, g<0: 引力)
            hbar: 约化 Planck 常数
            m: 粒子质量
        """
        self.conserved_quantities = [
            FieldConservedQuantity("particle_number", "N = ∫|ψ|²dV", "dN/dt = 0", SymmetryGroup.GAUGE_U1, 3),
            FieldConservedQuantity("energy", "E = ∫[ℏ²/(2m)|∇ψ|² + V|ψ|² + g/2|ψ|⁴]dV", "dE/dt = 0", None, 3),
        ]
        self.symplectic_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])
        self.hamiltonian = "ℏ²/(2m)|∇ψ|² + V|ψ|² + g/2|ψ|⁴"
        self.noether_currents = [
            NoetherCurrent(
                "nls_probability_current",
                SymmetryGroup.GAUGE_U1,
                ["|ψ|²", "(ℏ/m)Im(ψ*∇ψ)"],
                "∂|ψ|²/∂t + div((ℏ/m)Im(ψ*∇ψ)) = 0",
            ),
        ]

        # Same structure as linear Schrödinger but with nonlinear potential g|ψ|²
        # Real-imaginary: z = [Re(ψ), Im(ψ)]
        # Coupling: A = (1/ℏ)[[0, H_eff], [-H_eff, 0]]
        # H_eff = -ℏ²/(2m)∇² + V + g|ψ|² (mean-field at reference |ψ₀|²)
        psi0_sq = 1.0  # reference |ψ|²
        H_eff = g * psi0_sq  # nonlinear contribution at reference
        self.coupling_matrix = (1.0 / hbar) * np.array(
            [
                [0.0, H_eff],
                [-H_eff, 0.0],
            ]
        )

        self.flux_tensors = [
            (hbar / m)
            * np.array(
                [
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            )
        ]

        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.linalg.eigvals(self.coupling_matrix)
        return self

    def build_from_vlasov(self, q: float = 1.0, m_s: float = 1.0) -> ConservationMatrixField:
        """从 Vlasov 方程构建守恒矩阵场 (无碰撞等离子体).

        ∂f_s/∂t + v·∇_x f_s + (q_s/m_s)(E + v×B)·∇_v f_s = 0

        守恒量: 各物种粒子数 + 总动量 + 总能量 (电磁场 + 粒子)

        Args:
            q: 物种电荷
            m_s: 物种质量
        """
        self.conserved_quantities = [
            FieldConservedQuantity(
                "species_number", "∫f_s d³v d³x", "d/dt ∫f_s d³v d³x = 0", SymmetryGroup.GAUGE_U1, 3
            ),
            FieldConservedQuantity(
                "total_momentum",
                "Σ_s m_s∫vf_s d³v d³x + ε₀∫E×B d³x",
                "dP_total/dt = 0 (无外力时)",
                SymmetryGroup.TRANSLATION,
                3,
            ),
            FieldConservedQuantity(
                "total_energy", "Σ_s ∫½m_sv²f_s d³v d³x + ∫(ε₀E²/2 + B²/2μ₀)d³x", "dE_total/dt = 0", None, 3
            ),
        ]
        self.noether_currents = [
            NoetherCurrent(
                "vlasov_particle_current",
                SymmetryGroup.GAUGE_U1,
                ["∫f_s d³v", "∫vf_s d³v"],
                "d/dt ∫f_s d³v + div(∫vf_s d³v) = 0",
            ),
        ]

        # Vlasov is 6D+time (phase space). Use moment reduction:
        # U = [n, nu, E_kinetic] (density, momentum density, kinetic energy density)
        # The Vlasov equation conserves all moments in the collisionless limit
        # Coupling matrix from the characteristic equations:
        # dx/dt = v, dv/dt = (q/m)(E + v×B)
        # For moment system (similar to Euler but with EM coupling):
        qm = q / m_s
        self.coupling_matrix = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, qm],
                [0.0, 0.0, 0.0],
            ]
        )

        self.flux_tensors = [self.coupling_matrix.copy()]
        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.linalg.eigvals(self.coupling_matrix)
        return self

    def build_from_hartree_fock(self, hbar: float = 1.0, m: float = 1.0) -> ConservationMatrixField:
        """从 Hartree-Fock 方程构建守恒矩阵场.

        Fψ_i = ε_i ψ_i (Fock 算子本征值问题)
        F = h + Σ_j (J_j - K_j)

        守恒量: 粒子数 (U(1)), 总能量 (变分下界), Brillouin 定理

        Args:
            hbar: 约化 Planck 常数
            m: 电子质量
        """
        self.conserved_quantities = [
            FieldConservedQuantity(
                "particle_number", "N = Σ_i n_i", "N = const (Slater 行列式粒子数固定)", SymmetryGroup.GAUGE_U1, 3
            ),
            FieldConservedQuantity("total_energy", "E_HF", "E_HF ≥ E_exact (变分原理)", None, 3),
            FieldConservedQuantity("brillouin_theorem", "<Φ_0|H|Φ_i^a>", "<Φ_0|H|Φ_i^a> = 0 (收敛时)", None, 3),
        ]
        self.noether_currents = [
            NoetherCurrent(
                "hf_density_current", SymmetryGroup.GAUGE_U1, ["ρ(r)", "j(r)"], "粒子数守恒 (Slater 行列式归一化)"
            ),
        ]

        # HF is a stationary eigenvalue problem, not a time evolution.
        # For TDHF (time-dependent HF), structure is same as Schrödinger
        # with Fock operator as effective Hamiltonian.
        # Coupling matrix for real-imaginary decomposition:
        # A = (1/ℏ)[[0, F], [-F, 0]] where F is the Fock operator
        # At self-consistency, F is diagonal in the MO basis
        # Use a 2x2 representation for a single orbital:
        F_eff = 1.0  # placeholder Fock matrix element (Hartree)
        self.coupling_matrix = (1.0 / hbar) * np.array(
            [
                [0.0, F_eff],
                [-F_eff, 0.0],
            ]
        )

        self.flux_tensors = [
            (hbar / m)
            * np.array(
                [
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            )
        ]

        self.symplectic_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])
        self.hamiltonian = "h + Σ_j (J_j - K_j)"
        self.jacobian = self.coupling_matrix.copy()
        self.eigenvalues = np.linalg.eigvals(self.coupling_matrix)
        return self

    def verify_conservation(self, U: np.ndarray, dt: float = 1e-6) -> dict[str, bool]:
        """验证守恒律是否满足.

        Args:
            U: 守恒变量向量 (n_conserved,)
            dt: 时间步长

        Returns:
            每个守恒量是否满足守恒律
        """
        results = {}
        if self.coupling_matrix is not None and U is not None:
            dUdt = self.coupling_matrix @ U
            for i, q in enumerate(self.conserved_quantities):
                # 检查 dU_i/dt 是否接近零（无源项时）
                if self.source_vector is not None:
                    residual = abs(dUdt[i] - self.source_vector[i])
                else:
                    residual = abs(dUdt[i])
                results[q.name] = residual < 1e-8
        return results

    def structural_invariants(self) -> list[StructuralInvariant]:
        """生成守恒矩阵场的结构不变量."""
        invariants = []
        for q in self.conserved_quantities:
            invariants.append(
                StructuralInvariant(
                    name=f"{q.name}_conservation",
                    expression=q.expression,
                    theorem="Noether's theorem" if q.symmetry else "conservation law",
                    condition=f"symmetry: {q.symmetry.value if q.symmetry else 'time translation'}",
                    severity="conservation",
                    affected_quantities=[q.symbol],
                )
            )
        if self.is_hamiltonian:
            invariants.append(
                StructuralInvariant(
                    name="symplectic_structure",
                    expression="dω/dt = 0 (Liouville's theorem)",
                    theorem="Liouville's theorem",
                    condition="Hamiltonian flow",
                    severity="theorem",
                    affected_quantities=["ω"],
                )
            )
        return invariants

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "n_conserved": self.n_conserved,
            "conserved_quantities": [
                {
                    "name": q.name,
                    "symbol": q.symbol,
                    "expression": q.expression,
                    "symmetry": q.symmetry.value if q.symmetry else None,
                }
                for q in self.conserved_quantities
            ],
            "is_hyperbolic": self.is_hyperbolic,
            "is_hamiltonian": self.is_hamiltonian,
            "characteristic_speeds": self.characteristic_speeds.tolist()
            if self.characteristic_speeds is not None
            else None,
            "noether_map": {k.value: v.name for k, v in self.noether_map.items()},
            "hamiltonian": self.hamiltonian,
            "noether_currents": [
                {"name": c.name, "symmetry": c.symmetry.value, "continuity_equation": c.continuity_equation}
                for c in self.noether_currents
            ],
        }
