"""黎曼几何结构。

度量、联络、曲率、Lie 导数等 Riemann 几何核心概念。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata
from .geometry_manifold import TensorField
from .properties import StructuralInvariant

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetricFunction — callable metric for real Christoffel computation
# ---------------------------------------------------------------------------


class MetricFunction:
    """度量张量作为坐标的函数，支持在任意点求值和数值微分。

    与只存储单点矩阵的 Metric 不同，MetricFunction 将每个分量存为
    Callable[[dict[str, float]], float]，从而可以在不同坐标点求值，
    进而用中心差分计算偏导数和 Christoffel 符号。
    """

    def __init__(
        self,
        components: list[list[Callable[[dict[str, float]], float]]],
        coordinate_names: list[str],
    ):
        """
        Args:
            components: g_{ij}(x) — 二维列表，每个元素是接受坐标字典的 callable
            coordinate_names: 坐标名称列表，如 ['r', 'theta', 'phi']
        """
        self.components = components
        self.coordinate_names = coordinate_names
        self.dim = len(coordinate_names)

    def at(self, coords: dict[str, float]) -> np.ndarray:
        """在给定坐标处求值度量张量，返回 dim×dim numpy 数组。"""
        g = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                g[i, j] = self.components[i][j](coords)
        return g

    def inverse_at(self, coords: dict[str, float]) -> np.ndarray:
        """在给定坐标处求值逆度量 g^{ij}。"""
        return np.linalg.inv(self.at(coords))

    def christoffel_at(self, coords: dict[str, float], epsilon: float = 1e-6) -> np.ndarray:
        """用中心差分计算 Christoffel 符号 Γ^k_{ij}。

        公式：Γ^k_{ij} = ½ g^{kl} (∂_j g_{li} + ∂_i g_{lj} - ∂_l g_{ij})

        Returns:
            形状 (dim, dim, dim) 的 numpy 数组，索引 [k, i, j]
        """
        n = self.dim
        g = self.at(coords)
        g_inv = np.linalg.inv(g)

        # 中心差分求 ∂g_{ij}/∂x^k，存入 dg[k, i, j]
        dg = np.zeros((n, n, n))
        for k in range(n):
            cname = self.coordinate_names[k]
            cp = dict(coords)
            cm = dict(coords)
            cp[cname] = coords.get(cname, 0.0) + epsilon
            cm[cname] = coords.get(cname, 0.0) - epsilon
            g_plus = self.at(cp)
            g_minus = self.at(cm)
            dg[k] = (g_plus - g_minus) / (2.0 * epsilon)

        # Γ^k_{ij} = ½ g^{kl} (∂_j g_{li} + ∂_i g_{lj} - ∂_l g_{ij})
        # dg shape: (n, n, n), 索引 [d, l, i] = ∂_d g_{l, i}
        # term[l, i, j] = dg[j, l, i] + dg[i, l, j] - dg[l, i, j]
        #   dg[j, l, i] → axes (d=j, l, i) → permute 到 (l, i, j): transpose(1, 2, 0)
        #   dg[i, l, j] → axes (d=i, l, j) → permute 到 (l, i, j): transpose(1, 0, 2)
        #   dg[l, i, j] → 已经是 (l, i, j)，不变
        term = dg.transpose(1, 2, 0) + dg.transpose(1, 0, 2) - dg  # shape (l, i, j)
        gamma = 0.5 * np.einsum("kl,lij->kij", g_inv, term)
        return gamma

    def riemann_at(self, coords: dict[str, float], epsilon: float = 1e-5) -> np.ndarray:
        """用数值微分计算 Riemann 曲率张量 R^a_{bcd}。

        R^a_{bcd} = ∂_c Γ^a_{bd} - ∂_d Γ^a_{bc}
                    + Γ^a_{ce} Γ^e_{bd} - Γ^a_{de} Γ^e_{bc}

        Returns:
            形状 (dim, dim, dim, dim) 的 numpy 数组，索引 [a, b, c, d]
        """
        n = self.dim
        gamma = self.christoffel_at(coords, epsilon=epsilon)

        # 中心差分求 ∂_c Γ^a_{bd}
        dgamma = np.zeros((n, n, n, n))  # dgamma[c, a, b, d]
        for c in range(n):
            cname = self.coordinate_names[c]
            cp = dict(coords)
            cm = dict(coords)
            cp[cname] = coords.get(cname, 0.0) + epsilon
            cm[cname] = coords.get(cname, 0.0) - epsilon
            gp = self.christoffel_at(cp, epsilon=epsilon)
            gm = self.christoffel_at(cm, epsilon=epsilon)
            dgamma[c] = (gp - gm) / (2.0 * epsilon)

        # R^a_{bcd} = ∂_c Γ^a_{bd} - ∂_d Γ^a_{bc}
        #           + Γ^a_{ce} Γ^e_{bd} - Γ^a_{de} Γ^e_{bc}
        # dgamma 索引 [c, a, b, d] = ∂_c Γ^a_{bd}（注意第 0 轴是 c）
        # R 索引 [a, b, c, d]；需要把 dgamma 的 c 轴从位置 0 挪到位置 2：transpose(1, 2, 0, 3)
        # 第二项 dgamma[d, a, b, c] 在 R[a, b, c, d] 下需要 d 在位置 0 挪到 2、c 在位置 3 挪到 0... 不对
        # 实际上 dgamma[d, a, b, c] 中：c 轴对应 R 的 d 位置（3），d 轴对应 R 的 c 位置（2），
        # a 轴对应 R 的 a 位置（0），b 轴对应 R 的 b 位置（1）。
        # 原 dgamma 是 (c, a, b, d)。我们想取 dgamma[d, a, b, c] 然后映射到 R[a, b, c, d]。
        # 等价于先把 dgamma 的轴 0 与轴 3 交换：(d, a, b, c) = dgamma.transpose(3, 1, 2, 0)
        # 然后把 (d, a, b, c) 看作 R 的 (a, b, c, d) — 也就是把 axis 0(d)→position 3,
        # axis 1(a)→0, axis 2(b)→1, axis 3(c)→2，即 transpose(1, 2, 3, 0)
        # 但这两步等价于直接对 dgamma 做 transpose(1, 2, 3, 0)：
        #   dgamma.transpose(1, 2, 3, 0)[a, b, c, d] = dgamma[d, a, b, c] ✓
        # 第一项 dgamma[c, a, b, d] → R[a, b, c, d]：transpose(1, 2, 0, 3)
        #   dgamma.transpose(1, 2, 0, 3)[a, b, c, d] = dgamma[c, a, b, d] ✓
        R = dgamma.transpose(1, 2, 0, 3) - dgamma.transpose(1, 2, 3, 0)
        # Γ^a_{ce} Γ^e_{bd} → R[a, b, c, d]
        R = R + np.einsum("ace,ebd->abcd", gamma, gamma)
        # Γ^a_{de} Γ^e_{bc} → R[a, b, c, d]
        R = R - np.einsum("ade,ebc->abcd", gamma, gamma)
        return R

    def ricci_at(self, coords: dict[str, float], epsilon: float = 1e-5) -> np.ndarray:
        """计算 Ricci 张量 R_{ij} = R^k_{ikj}。"""
        R = self.riemann_at(coords, epsilon=epsilon)
        # R[a, b, c, d] = R^a_{bcd}; Ricci_{ij} = R^k_{ikj} = R[k, i, k, j]
        ricci = np.einsum("kikj->ij", R)
        return ricci

    def scalar_curvature_at(self, coords: dict[str, float], epsilon: float = 1e-5) -> float:
        """计算标量曲率 R = g^{ij} R_{ij}。"""
        ricci = self.ricci_at(coords, epsilon=epsilon)
        g_inv = self.inverse_at(coords)
        return float(np.einsum("ij,ij", g_inv, ricci))


# ---------------------------------------------------------------------------
# Predefined metric functions
# ---------------------------------------------------------------------------


def schwarzschild_metric(M: float = 1.0) -> MetricFunction:
    """Schwarzschild 度规（Schwarzschild 坐标系）。

    ds² = -(1-2M/r) dt² + (1-2M/r)^{-1} dr² + r² dθ² + r² sin²θ dφ²
    """

    def g00(c):
        r = c.get("r", 2.0)
        return -(1.0 - 2.0 * M / r)

    def g11(c):
        r = c.get("r", 2.0)
        return 1.0 / (1.0 - 2.0 * M / r)

    def g22(c):
        r = c.get("r", 2.0)
        return r**2

    def g33(c):
        r = c.get("r", 2.0)
        theta = c.get("theta", 1.0)
        return r**2 * np.sin(theta) ** 2

    def zero(c):
        return 0.0

    return MetricFunction(
        [[g00, zero, zero, zero], [zero, g11, zero, zero], [zero, zero, g22, zero], [zero, zero, zero, g33]],
        ["t", "r", "theta", "phi"],
    )


def flat_metric(dim: int = 3) -> MetricFunction:
    """欧氏平坦度规。"""

    def _make_ij(i, j, d):
        if i == j:
            return lambda c, _i=i: 1.0
        return lambda c: 0.0

    return MetricFunction(
        [[_make_ij(i, j, dim) for j in range(dim)] for i in range(dim)],
        [f"x{i}" for i in range(dim)],
    )


def spherical_metric() -> MetricFunction:
    """三维球坐标度规。

    ds² = dt² + dr² + r² dθ² + r² sin²θ dφ²
    """

    def g00(c):
        return 1.0

    def g11(c):
        return 1.0

    def g22(c):
        r = c.get("r", 1.0)
        return r**2

    def g33(c):
        r = c.get("r", 1.0)
        theta = c.get("theta", 1.0)
        return r**2 * np.sin(theta) ** 2

    def zero(c):
        return 0.0

    return MetricFunction(
        [[g00, zero, zero, zero], [zero, g11, zero, zero], [zero, zero, g22, zero], [zero, zero, zero, g33]],
        ["t", "r", "theta", "phi"],
    )


# ---------------------------------------------------------------------------
# Numerical Lie derivative
# ---------------------------------------------------------------------------


def lie_derivative_vector_field(
    X: Callable[[dict[str, float]], np.ndarray],
    Y: Callable[[dict[str, float]], np.ndarray],
    coords: dict[str, float],
    epsilon: float = 1e-6,
) -> np.ndarray:
    """数值计算向量场的 Lie 导数 (L_X Y)^i = X^j ∂Y^i/∂x^j - Y^j ∂X^i/∂x^j。

    Args:
        X: 向量场 X^i(x)，接受坐标字典，返回 numpy 数组
        Y: 向量场 Y^i(x)，接受坐标字典，返回 numpy 数组
        coords: 求值点的坐标
        epsilon: 中心差分步长

    Returns:
        (L_X Y)^i 作为 numpy 数组
    """
    coord_names = list(coords.keys())
    dim = len(coord_names)
    X_vals = X(coords)
    Y_vals = Y(coords)
    result = np.zeros(dim)

    for i in range(dim):
        for j in range(dim):
            cp = dict(coords)
            cm = dict(coords)
            cp[coord_names[j]] += epsilon
            cm[coord_names[j]] -= epsilon
            dYi_dxj = (Y(cp)[i] - Y(cm)[i]) / (2.0 * epsilon)
            dXi_dxj = (X(cp)[i] - X(cm)[i]) / (2.0 * epsilon)
            result[i] += X_vals[j] * dYi_dxj - Y_vals[j] * dXi_dxj

    return result


def lie_derivative_scalar(
    X: Callable[[dict[str, float]], np.ndarray],
    f: Callable[[dict[str, float]], float],
    coords: dict[str, float],
    epsilon: float = 1e-6,
) -> float:
    """数值计算标量函数的 Lie 导数 L_X f = X^i ∂_i f。

    Args:
        X: 向量场 X^i(x)
        f: 标量函数 f(x)
        coords: 求值点
        epsilon: 差分步长

    Returns:
        L_X f 在 coords 处的值
    """
    coord_names = list(coords.keys())
    X_vals = X(coords)
    result = 0.0
    for j in range(len(coord_names)):
        cp = dict(coords)
        cm = dict(coords)
        cp[coord_names[j]] += epsilon
        cm[coord_names[j]] -= epsilon
        df_dxj = (f(cp) - f(cm)) / (2.0 * epsilon)
        result += X_vals[j] * df_dxj
    return result


def lie_derivative_metric(
    X: Callable[[dict[str, float]], np.ndarray],
    metric_func: MetricFunction,
    coords: dict[str, float],
    epsilon: float = 1e-6,
) -> np.ndarray:
    """数值计算度量的 Lie 导数 (L_X g)_{ij}。

    用中心差分计算流拉回： (L_X g)_{ij} ≈ [φ*_ε g - g] / ε

    Args:
        X: 向量场
        metric_func: 可调用度量
        coords: 求值点
        epsilon: 差分步长

    Returns:
        (L_X g)_{ij} 作为 dim×dim numpy 数组
    """
    n = metric_func.dim
    coord_names = metric_func.coordinate_names
    X_vals = X(coords)

    # 用有限差分: (L_X g)_{ij} = X^k ∂_k g_{ij} + g_{kj} ∂_i X^k + g_{ik} ∂_j X^k
    g = metric_func.at(coords)
    result = np.zeros((n, n))

    # ∂_k g_{ij}
    dg = np.zeros((n, n, n))  # dg[k, i, j]
    for k in range(n):
        cp = dict(coords)
        cm = dict(coords)
        cp[coord_names[k]] = coords.get(coord_names[k], 0.0) + epsilon
        cm[coord_names[k]] = coords.get(coord_names[k], 0.0) - epsilon
        dg[k] = (metric_func.at(cp) - metric_func.at(cm)) / (2.0 * epsilon)

    # ∂_i X^k
    dX = np.zeros((n, n))  # dX[i, k] = ∂_i X^k
    for i in range(n):
        cp = dict(coords)
        cm = dict(coords)
        cp[coord_names[i]] = coords.get(coord_names[i], 0.0) + epsilon
        cm[coord_names[i]] = coords.get(coord_names[i], 0.0) - epsilon
        dX[i] = (X(cp) - X(cm)) / (2.0 * epsilon)

    for i in range(n):
        for j in range(n):
            val = 0.0
            for k in range(n):
                val += X_vals[k] * dg[k, i, j] + g[k, j] * dX[i, k] + g[i, k] * dX[j, k]
            result[i, j] = val

    return result


@dataclass
class Metric(TensorField):
    """黎曼度量：g ∈ T^{(0,2)}M，对称、正定.

    Attributes:
        signature: 度规号差 (p, q) — 微分同胚不变量（Sylvester）
        is_positive_definite: 是否正定（Riemann vs pseudo-Riemann）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Metric",
            canonical_form="g ∈ Γ(T^{(0,2)}M),  g_ab symmetric positive-definite",
            description="Riemannian metric: symmetric positive-definite (0,2) tensor field",
        )
    )
    p: int = 0
    q: int = 2
    signature: tuple[int, int] = (3, 0)
    is_positive_definite: bool = True
    components: list[list[float]] | None = None

    @property
    def function_space(self) -> str:
        return f"Γ(S²_+ T*M) — positive-definite symmetric (0,2) tensors on {self.manifold_dim}-manifold"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="signature_diffeomorphism_invariant",
                    expression=f"sign(g) = {self.signature} is invariant under diffeomorphism",
                    theorem="Sylvester's law of inertia (signature of symmetric bilinear form)",
                    affected_quantities=["signature", "metric"],
                ),
                StructuralInvariant(
                    name="volume_form",
                    expression="dV = √|det(g)| dx¹ ∧ ... ∧ dx^n",
                    theorem="Volume form from metric determinant",
                    affected_quantities=["volume", "integration"],
                ),
            ]
        )
        return invariants

    def compute_inverse(self) -> list[list[float]]:
        """计算逆度量 g^{ij}。

        利用 numpy 对度量矩阵求逆。若 components 未设置则抛出 ValueError。
        """
        if self.components is None:
            raise ValueError("度量分量 components 未设置，无法计算逆度量")
        g = np.array(self.components, dtype=float)
        g_inv = np.linalg.inv(g)
        return g_inv.tolist()  # type: ignore[no-any-return]

    def compute_volume_element(self) -> str:
        """返回体积形式表达式 √|det(g)| dx¹ ∧ ... ∧ dx^n。"""
        n = self.manifold_dim
        wedge = " ∧ ".join(f"dx^{i + 1}" for i in range(n))
        return f"√|det(g)| {wedge}"

    def compute_det(self) -> float | None:
        """计算 det(g_{ij})，若 components 未设置则返回 None。"""
        if self.components is None:
            return None
        return float(np.linalg.det(np.array(self.components, dtype=float)))

    def is_flat(self) -> bool | None:
        """判断度量是否平坦。

        对于对角常度量（对角元素全部为常数），直接返回 True。
        更一般的情形需要计算 Christoffel 符号，此处仅做简单启发式判断。
        若 components 未设置则返回 None。
        """
        if self.components is None:
            return None
        g = np.array(self.components, dtype=float)
        # 检查是否为对角矩阵且对角元素为常数
        off_diag = g - np.diag(np.diag(g))
        if np.allclose(off_diag, 0):
            # 对角矩阵，对角元素为常数 → 平坦
            return True
        # 非对角情况需要完整计算 Christoffel 符号，此处保守返回 None
        return None


@dataclass
class Connection(AbstractMathematicalStructure):
    """Koszul 联络：∇: Γ(TM) × Γ(TM) → Γ(TM).

    满足 C^∞-linear in first arg, Leibniz rule in second.

    Attributes:
        manifold_dim: 底流形维数
        is_flat: 曲率是否为零
        is_torsion_free: T(X,Y) = ∇_X Y - ∇_Y X - [X,Y] = 0
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Connection (Koszul)",
            canonical_form="∇: 𝔛(M) × 𝔛(M) → 𝔛(M),  C^∞-linear + Leibniz",
            description="Koszul connection: covariant derivative on a vector bundle",
        )
    )
    manifold_dim: int = 3
    is_flat: bool = False
    is_torsion_free: bool = False

    @property
    def function_space(self) -> str:
        return f"Affine connections on {self.manifold_dim}-manifold"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="difference_is_tensor",
                expression="∇' - ∇ is a (1,2) tensor field (Christoffel symbols are NOT tensors, but their difference is)",  # noqa: E501
                theorem="Difference of two affine connections is a tensor",
                affected_quantities=["connection", "christoffel_symbols"],
            ),
        ]


@dataclass
class LeviCivitaConnection(Connection):
    """Levi-Civita 联络：唯一的度量相容且无挠联络.

    ∇g = 0（度量相容）且 T(X,Y) = 0（无挠）。

    Christoffel 符号由度量唯一确定：
    Γ^k_{ij} = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Levi-Civita Connection",
            canonical_form="∇_k g_{ij} = 0,  T^i_{jk} = 0 → Γ^k_{ij} = ½ g^{kl}(∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})",
            description="Unique metric-compatible torsion-free connection on a Riemannian manifold",
        )
    )
    is_torsion_free: bool = True
    metric_components: list[list[float]] | None = None

    @property
    def function_space(self) -> str:
        return f"Levi-Civita connection on {self.manifold_dim}-manifold"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="fundamental_theorem_riemannian_geometry",
                    expression="∃! ∇ such that ∇g = 0 and T = 0",
                    theorem="Fundamental Theorem of Riemannian Geometry (uniqueness of Levi-Civita connection)",
                    affected_quantities=["connection", "metric", "torsion"],
                ),
                StructuralInvariant(
                    name="metric_compatibility",
                    expression="∇_X g = 0  ⇒  d/dt⟨Y, Z⟩|_{t=0} = ⟨∇_X Y, Z⟩ + ⟨Y, ∇_X Z⟩",
                    theorem="Metric compatibility preserves inner products under parallel transport",
                    affected_quantities=["metric", "parallel_transport"],
                ),
            ]
        )
        return invariants

    def compute_christoffel(self) -> list[list[list[float]]]:
        """数值计算 Christoffel 符号 Γ^k_{ij}。

        公式：Γ^k_{ij} = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})

        由于没有解析偏导数信息，当 metric_components 可用时，
        用有限差分近似偏导数（假设坐标间距 h=1e-5）。
        若 metric_components 为 None，则返回全零张量。

        注意：此方法仅使用单点度量数据，无法真正计算偏导数。
        对于需要真实 Christoffel 符号的场景，请使用 compute_christoffel_from_func。
        """
        n = self.manifold_dim
        if self.metric_components is None:
            return [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(n)]

        # 单点数据无法做有限差分，常度量情形 Christoffel 为零
        return [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(n)]

    def compute_christoffel_from_func(
        self,
        metric_func: MetricFunction,
        coords: dict[str, float],
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """使用可调用度量函数计算 Christoffel 符号。

        Args:
            metric_func: MetricFunction 实例，可在任意坐标点求值
            coords: 坐标值字典，如 {'r': 10.0, 'theta': 1.0, 'phi': 0.0}
            epsilon: 中心差分步长

        Returns:
            Γ^k_{ij}，形状 (dim, dim, dim) 的 numpy 数组，索引 [k, i, j]
        """
        return metric_func.christoffel_at(coords, epsilon=epsilon)

    def christoffel_symbol(self, i: int, j: int, k: int) -> str:
        """返回 Christoffel 符号 Γ^k_{ij} 的符号公式。"""
        return f"Γ^{k}_{{{i}{j}}} = ½ g^{{{k}l}} (∂_{i} g_{{{j}l}} + ∂_{j} g_{{{i}l}} - ∂_l g_{{{i}{j}}})"

    def parallel_transport_equation(self) -> str:
        """返回平行移动 ODE：dV^k/dt + Γ^k_{ij} (dx^i/dt) V^j = 0。"""
        return "dV^k/dt + Γ^k_{ij} (dx^i/dt) V^j = 0"


@dataclass
class Curvature(AbstractMathematicalStructure):
    """曲率：R(X,Y)Z = ∇_X ∇_Y Z - ∇_Y ∇_X Z - ∇_{[X,Y]} Z.

    Attributes:
        manifold_dim: 底流形维数
        curvature_type: "riemann", "ricci", "scalar"
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Curvature",
            canonical_form="R(X,Y)Z = ∇_X∇_Y Z - ∇_Y∇_X Z - ∇_{[X,Y]}Z",
            description="Riemann curvature: measures non-commutativity of covariant derivatives",
        )
    )
    manifold_dim: int = 3
    curvature_type: str = "riemann"

    @property
    def function_space(self) -> str:
        return f"Curvature tensor on {self.manifold_dim}-manifold"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="first_bianchi_identity",
                expression="R^a_{bcd} + R^a_{cdb} + R^a_{dbc} = 0",
                theorem="First Bianchi Identity (algebraic)",
                affected_quantities=["curvature", "torsion"],
            ),
            StructuralInvariant(
                name="second_bianchi_identity",
                expression="∇_{[a} R_{bc]de} = 0",
                theorem="Second Bianchi Identity (differential)",
                condition="self.curvature_type == 'riemann'",
                affected_quantities=["curvature", "covariant_derivative"],
            ),
        ]


@dataclass
class RiemannCurvature(Curvature):
    """Riemann 曲率张量：完整的 (1,3) 型张量.

    Symmetries:
      R_{abcd} = R_{cdab} = -R_{bacd} = -R_{abdc}
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Riemann Curvature",
            canonical_form="R^a_{bcd} = ∂_c Γ^a_{db} - ∂_d Γ^a_{cb} + Γ^a_{ce} Γ^e_{db} - Γ^a_{de} Γ^e_{cb}",
            description="Full Riemann curvature tensor of type (1,3)",
        )
    )
    curvature_type: str = "riemann"
    christoffel: list[list[list[float]]] | None = None

    @property
    def independent_components(self) -> int:
        n = self.manifold_dim
        return n * n * (n * n - 1) // 12

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="riemann_symmetry_1",
                    expression="R_{abcd} = -R_{bacd}  (antisymmetric in first pair)",
                    theorem="Skew symmetry of Riemann tensor",
                    affected_quantities=["curvature"],
                ),
                StructuralInvariant(
                    name="riemann_symmetry_2",
                    expression="R_{abcd} = R_{cdab}  (pair exchange symmetry)",
                    theorem="Block symmetry of Riemann tensor",
                    affected_quantities=["curvature"],
                ),
            ]
        )
        return invariants

    def independent_component_count(self) -> int:
        """计算 Riemann 张量的独立分量数，同时输出公式 n²(n²-1)/12。"""
        n = self.manifold_dim
        count = n * n * (n * n - 1) // 12
        formula = f"n²(n²-1)/12 = {n}²({n}²-1)/12 = {count}"
        logger.info(f"独立分量数: {formula}")
        return count

    def riemann_tensor_formula(self) -> str:
        """返回 Riemann 曲率张量的完整公式。"""
        return "R^a_{bcd} = ∂_c Γ^a_{db} - ∂_d Γ^a_{cb} + Γ^a_{ce} Γ^e_{db} - Γ^a_{de} Γ^e_{cb}"

    def compute_from_christoffel(
        self,
        gamma: list[list[list[float]]],
        dgamma: list[list[list[list[float]]]],
    ) -> list[list[list[list[float]]]]:
        """由 Christoffel 符号及其偏导数数值计算 Riemann 曲率张量。

        R^a_{bcd} = ∂_c Γ^a_{bd} - ∂_d Γ^a_{bc} + Γ^a_{ce} Γ^e_{bd} - Γ^a_{de} Γ^e_{bc}

        优先使用 Rust 加速路径，不可用时回退到纯 Python 实现。

        Args:
            gamma: Christoffel 符号 Γ^a_{bd}，形状 (n, n, n)
            dgamma: Christoffel 符号的偏导数 ∂_c Γ^a_{bd}，形状 (n, n, n, n)
        """
        n = self.manifold_dim

        # 尝试 Rust 加速路径
        try:
            from math_anything.rust_bridge import EMLAccelerator

            acc = EMLAccelerator()

            # 展平 christoffel: gamma[a][c][e] → christoffel[i*dim^2 + j*dim + k]
            # Python 索引 [upper][lower1][lower2] 与 Rust [i][j][k] 一致
            christoffel_flat: list[float] = []
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        christoffel_flat.append(gamma[i][j][k])

            # 展平 d_christoffel: dgamma[c][a][b][d] → d_christoffel[i*dim^3 + l*dim^2 + j*dim + k]
            # Rust 期望 ∂_k Γ^i_{lj}，Python 提供 ∂_c Γ^a_{bd}
            # 映射: d_christoffel[i][l][j][k] = dgamma[k][i][l][j]
            d_christoffel_flat: list[float] = [0.0] * (n * n * n * n)
            d3 = n * n * n
            d2 = n * n
            for i in range(n):
                for m in range(n):
                    for j in range(n):
                        for k in range(n):
                            d_christoffel_flat[i * d3 + m * d2 + j * n + k] = dgamma[k][i][m][j]

            riemann_flat = acc.compute_riemann_tensor(christoffel_flat, d_christoffel_flat, n)
            if riemann_flat:
                # 重塑: flat[i*dim^3 + j*dim^2 + k*dim + l] → R[i][j][k][l]
                R = [[[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            for m in range(n):
                                R[i][j][k][m] = riemann_flat[i * d3 + j * d2 + k * n + m]
                return R
        except (ValueError, TypeError, RuntimeError):
            logger.debug("Rust 加速不可用，回退到纯 Python 计算")

        # 纯 Python 回退
        R = [[[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        val = dgamma[c][a][b][d] - dgamma[d][a][b][c]
                        for e in range(n):
                            val += gamma[a][c][e] * gamma[e][b][d]
                            val -= gamma[a][d][e] * gamma[e][b][c]
                        R[a][b][c][d] = val
        return R

    def sectional_curvature(self, p: int, q: int) -> str:
        """返回截面曲率公式 K(e_p, e_q) = R_{pqpq} / (g_{pp} g_{qq} - g_{pq}²)。"""
        return f"K(e_{p}, e_{q}) = R_{{{p}{q}{p}{q}}} / (g_{{{p}{p}}} g_{{{q}{q}}} - g_{{{p}{q}}}²)"


@dataclass
class RicciCurvature(Curvature):
    """Ricci 曲率：R_{ij} = R^k_{ikj}.

    迹运算收缩第一个和第三个指标。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Ricci Curvature",
            canonical_form="R_{ij} = R^k_{ikj} = g^{kl} R_{kilj}",
            description="Ricci curvature: trace of Riemann tensor on first and third indices",
        )
    )
    curvature_type: str = "ricci"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="ricci_symmetry",
                    expression="R_{ij} = R_{ji}  (Ricci tensor is symmetric)",
                    theorem="Contracted Bianchi identity → symmetry of Ricci tensor",
                    affected_quantities=["ricci_curvature"],
                ),
            ]
        )
        return invariants

    def ricci_from_riemann(self, R: list[list[list[list[float]]]]) -> list[list[float]]:
        """由 Riemann 曲率张量收缩计算 Ricci 张量。

        R_{ij} = R^k_{ikj}，收缩第一个和第三个指标。
        """
        n = self.manifold_dim
        ricci = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += R[k][i][k][j]
                ricci[i][j] = s
        return ricci

    def einstein_tensor(
        self,
        R_ij: list[list[float]],
        R_scalar: float,
        g_ij: list[list[float]],
    ) -> list[list[float]]:
        """计算 Einstein 张量 G_{ij} = R_{ij} - ½ R g_{ij}。"""
        n = self.manifold_dim
        G = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                G[i][j] = R_ij[i][j] - 0.5 * R_scalar * g_ij[i][j]
        return G


@dataclass
class ScalarCurvature(Curvature):
    """标量曲率：R = g^{ij} R_{ij}.

    完整迹收缩。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Scalar Curvature",
            canonical_form="R = g^{ij} R_{ij} = g^{ij} g^{kl} R_{kilj}",
            description="Scalar curvature: full trace of Ricci tensor",
        )
    )
    curvature_type: str = "scalar"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="gauss_bonnet_2d",
                    expression="∫_M R dA = 4π χ(M)  (for closed 2-manifolds)",
                    theorem="Gauss-Bonnet Theorem (topological invariant from curvature integral)",
                    condition="self.manifold_dim == 2",
                    affected_quantities=["scalar_curvature", "euler_characteristic"],
                ),
                StructuralInvariant(
                    name="einstein_hilbert_action",
                    expression="S_EH = ∫ R √|g| d^n x  (variational → Einstein equations)",
                    theorem="Einstein-Hilbert action: scalar curvature as Lagrangian",
                    affected_quantities=["action", "gravity"],
                ),
            ]
        )
        return invariants

    def scalar_from_ricci(self, R_ij: list[list[float]], g_inv: list[list[float]]) -> float:
        """由 Ricci 张量和逆度量计算标量曲率 R = g^{ij} R_{ij}。"""
        n = self.manifold_dim
        s = 0.0
        for i in range(n):
            for j in range(n):
                s += g_inv[i][j] * R_ij[i][j]
        return s


@dataclass
class LieDerivative(AbstractMathematicalStructure):
    """Lie 导数：L_X T = d/dt|_{t=0} φ_t^* T.

    沿向量场 X 的流的拉回导数。

    Attributes:
        manifold_dim: 底流形维数
        vector_field_name: 生成流的向量场名称
        tensor_type: 被作用的张量类型，如 "(0,2)", "(1,1)"
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Lie Derivative",
            canonical_form="ℒ_X T = d/dt|_{t=0} φ_t^* T",
            description="Lie derivative along the flow of a vector field",
        )
    )
    manifold_dim: int = 3
    vector_field_name: str = ""
    tensor_type: str = "(0,2)"

    @property
    def function_space(self) -> str:
        return f"Lie derivative operator on {self.manifold_dim}-manifold"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="lie_commutes_with_d",
                expression="ℒ_X ∘ d = d ∘ ℒ_X  (Lie derivative commutes with exterior derivative)",
                theorem="Cartan's magic formula: ℒ_X = d∘ι_X + ι_X∘d",
                affected_quantities=["differential_form", "lie_derivative"],
            ),
            StructuralInvariant(
                name="lie_equals_lie_bracket",
                expression="ℒ_X Y = [X, Y]",
                theorem="Lie derivative of vector field equals Lie bracket",
                affected_quantities=["vector_field", "lie_bracket"],
            ),
            StructuralInvariant(
                name="leibniz_rule",
                expression="ℒ_X(T ⊗ S) = (ℒ_X T) ⊗ S + T ⊗ (ℒ_X S)",
                theorem="Lie derivative satisfies Leibniz rule (derivation)",
                affected_quantities=["tensor_field"],
            ),
        ]

    def lie_derivative_of_scalar(self, f: str, X: str) -> str:
        """标量函数的 Lie 导数：L_X f = X^i ∂_i f（方向导数）。"""
        return f"ℒ_{X} {f} = {X}^i ∂_i {f}"

    def lie_derivative_of_vector(self, Y: str, X: str) -> str:
        """向量场的 Lie 导数：L_X Y = [X, Y] = (X^j ∂_j Y^i - Y^j ∂_j X^i) ∂_i。"""
        return f"ℒ_{X} {Y} = [{X}, {Y}] = ({X}^j ∂_j {Y}^i - {Y}^j ∂_j {X}^i) ∂_i"

    def lie_derivative_of_metric(self, g: str, X: str) -> str:
        """度量的 Lie 导数：L_X g_{ij} = ∇_i X_j + ∇_j X_i。

        当 L_X g = 0 时即为 Killing 方程。
        """
        return f"ℒ_{X} {g}_{{ij}} = ∇_i {X}_j + ∇_j {X}_i  (Killing equation when ℒ_{X} {g} = 0)"

    def cartan_magic_formula(self) -> str:
        """Cartan 魔法公式：L_X = d∘ι_X + ι_X∘d。"""
        return "ℒ_X = d∘ι_X + ι_X∘d"
