"""平衡问题结构家族。

EquilibriumProblem = F(u) = 0 或 min E[u]

涵盖三大子类：
  - VariationalMinimization: δE = 0 → 结构力学的 Galerkin 形式
  - ConstrainedMinimization: Gibbs 自由能极小（热力学）
  - FixedPointProblem: SCF 不会直接解，但在迭代意义下是找不动点

结构主义视角：
  FEM 结构力学、Calphad 热力学、拓扑优化
  都是"在某个函数空间或配置空间上极小化一个泛函"的结构实例。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata
from .properties import StructuralInvariant, VariationalPrinciple, get_invariants


@dataclass
class EquilibriumProblem(AbstractMathematicalStructure):
    """平衡问题基类：F(u) = 0 或极小化泛函 E[u].

    所有静态/稳态问题的母类。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EQUILIBRIUM,
            name="Equilibrium Problem",
            canonical_form="F(u) = 0  or  δE[u] = 0",
            description="Find stationary point or minimize a functional",
        )
    )
    spatial_dim: int = 3

    @property
    def function_space(self) -> str:
        return f"H¹(Ω ⊂ ℝ^{self.spatial_dim})"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return []


@dataclass
class VariationalMinimizationProblem(EquilibriumProblem):
    """变分极小化问题：min E[u].

    弱形式：a(u, v) = ℓ(v) for all admissible v

    实例：
      - 线弹性（Abaqus, Ansys 的静态分析）
      - 热传导稳态
      - 静电学
      - 拓扑优化
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EQUILIBRIUM,
            name="Variational Minimization Problem",
            canonical_form="min E[u] = min ∫ W(∇u) dV",
            description="Minimize an energy functional; equivalent to weak form a(u,v) = l(v)",
        )
    )
    principle: str = VariationalPrinciple.MINIMUM
    convex: bool = True  # 能量泛函是否凸（保证唯一解）
    material_nonlinear: bool = False  # 材料非线性（W 非线性）
    geometric_nonlinear: bool = False  # 几何非线性（大变形）

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants: list[StructuralInvariant] = []

        if self.convex:
            invariants.append(
                StructuralInvariant(
                    name="unique_minimizer",
                    expression="u* is unique (convex energy functional)",
                    theorem="Convex analysis: strict convexity → unique minimizer",
                    affected_quantities=["solution", "displacement"],
                )
            )
        else:
            invariants.append(
                StructuralInvariant(
                    name="multiple_minimizers_possible",
                    expression="Multiple local minima may exist",
                    theorem="Non-convex optimization",
                    affected_quantities=["solution", "stability"],
                )
            )

        if not self.material_nonlinear:
            invariants.append(
                StructuralInvariant(
                    name="superposition_principle",
                    expression="u(αf₁ + βf₂) = αu(f₁) + βu(f₂)",
                    theorem="Linear elasticity → superposition",
                    affected_quantities=["solution", "displacement"],
                )
            )

        invariants.extend(get_invariants("variational"))
        return invariants


@dataclass
class ConstrainedMinimizationProblem(EquilibriumProblem):
    """约束极小化问题：min G subject to constraints.

    实例：
      - 热力学平衡（min G 受质量守恒、非负约束）
      - 接触力学（min E 受不可穿透约束）
      - 拓扑优化（min F 受体积约束）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EQUILIBRIUM,
            name="Constrained Minimization Problem",
            canonical_form="min G(x)  subject to  c(x) = 0,  h(x) ≥ 0",
            description="Minimize an objective subject to equality and inequality constraints",
        )
    )
    objective: str = ""  # "Gibbs自由能", "柔度", "质量"
    equality_constraints: list[str] = field(default_factory=list)
    inequality_constraints: list[str] = field(default_factory=list)
    method: str = "lagrange_multiplier"  # "KKT", "penalty", "barrier"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="kkt_conditions",
                expression="∇G + Σ λ_i ∇c_i + Σ μ_j ∇h_j = 0, λ_i free, μ_j ≥ 0, μ_j h_j = 0",
                theorem="Karush-Kuhn-Tucker conditions (first-order necessary)",
                condition="self.method in ('lagrange_multiplier', 'kkt')",
                affected_quantities=["lagrange_multipliers", "active_constraints"],
            ),
            StructuralInvariant(
                name="mass_conservation",
                expression="Σ n_i = N (total moles conserved)",
                theorem="Conservation of mass in closed systems",
                condition="'mass' in self.objective.lower()",
                affected_quantities=["composition", "chemical_potential"],
            ),
        ]


@dataclass
class FixedPointProblem(EquilibriumProblem):
    """不动点问题：x = T(x).

    迭代形式：x^{(k+1)} = T(x^{(k)})

    实例：
      - SCF 迭代（作为不动点问题来看）
      - 隐式时间步（每个时间步内部求解不动点）
      - Picard 迭代
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.EQUILIBRIUM,
            name="Fixed Point Problem",
            canonical_form="x = T(x)",
            description="Find a fixed point of a self-mapping",
        )
    )
    contraction: bool = False  # T 是否是压缩映射
    lipschitz_constant: float | None = None

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants: list[StructuralInvariant] = []

        if self.contraction or (self.lipschitz_constant is not None and self.lipschitz_constant < 1):
            invariants.append(
                StructuralInvariant(
                    name="unique_fixed_point",
                    expression="∃! x* : T(x*) = x* (Banach fixed point theorem)",
                    theorem="Banach Fixed Point Theorem",
                    affected_quantities=["convergence", "uniqueness"],
                )
            )
            invariants.append(
                StructuralInvariant(
                    name="linear_convergence",
                    expression="‖x^{(k+1)} - x*‖ ≤ L ‖x^{(k)} - x*‖ with L < 1",
                    theorem="Contraction mapping convergence rate",
                    affected_quantities=["convergence_rate"],
                )
            )
        else:
            invariants.append(
                StructuralInvariant(
                    name="fixed_point_existence_not_guaranteed",
                    expression="Without contraction, existence and uniqueness not guaranteed",
                    theorem="Fixed point theory (Brouwer/Schauder for compact convex sets)",
                    affected_quantities=["convergence"],
                )
            )

        return invariants


class VariationalSolver:
    """Numerical solver for variational/minimization problems.

    Solves: find u ∈ V such that a(u, v) = f(v) for all v ∈ V
    """

    def __init__(self, stiffness_assembler: Callable | None = None, load_assembler: Callable | None = None):
        """
        Args:
            stiffness_assembler: Function returning stiffness matrix K
            load_assembler: Function returning load vector f
        """
        self.stiffness_assembler = stiffness_assembler
        self.load_assembler = load_assembler

    def solve_1d_poisson(self, n_elements: int = 10, domain_length: float = 1.0, source: Callable = None) -> dict:  # type: ignore[assignment]
        """Solve 1D Poisson equation -u'' = f with u(0)=u(L)=0.

        Uses linear finite elements.
        """
        if source is None:

            def source(x):
                return 1.0

        n = n_elements + 1
        h = domain_length / n_elements
        nodes = np.linspace(0, domain_length, n)

        K = np.zeros((n, n))
        for i in range(n_elements):
            K[i, i] += 1.0 / h
            K[i, i + 1] += -1.0 / h
            K[i + 1, i] += -1.0 / h
            K[i + 1, i + 1] += 1.0 / h

        f_vec = np.zeros(n)
        for i in range(n_elements):
            f_vec[i] += h / 2 * source(nodes[i])
            f_vec[i + 1] += h / 2 * source(nodes[i + 1])

        K_red = K[1:-1, 1:-1]
        f_red = f_vec[1:-1]

        u = np.zeros(n)
        u[1:-1] = np.linalg.solve(K_red, f_red)

        return {
            "nodes": nodes.tolist(),
            "solution": u.tolist(),
            "n_elements": n_elements,
            "h": h,
        }

    def residual_norm(self, K: np.ndarray, u: np.ndarray, f: np.ndarray) -> float:
        """Compute ||Ku - f||."""
        return float(np.linalg.norm(K @ u - f))

    def energy_norm_error(self, K: np.ndarray, u_h: np.ndarray, u_exact: np.ndarray) -> float:
        """Compute energy norm error ||u - u_h||_E = sqrt((u-u_h)^T K (u-u_h))."""
        diff = u_exact - u_h
        return float(np.sqrt(max(0, diff @ K @ diff)))

    def condition_number(self, K: np.ndarray) -> float:
        """Compute condition number of stiffness matrix."""
        return float(np.linalg.cond(K))
