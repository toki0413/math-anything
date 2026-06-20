"""态射系统：数学结构之间的变换。

Morphism = 从一个结构到另一个结构的映射。

每个态射明确记录：
  - 保留了哪些结构性质（invariants_kept）
  - 丢失了哪些结构性质（invariants_lost）
  - 引入了哪些新性质（invariants_introduced）

这是 math-anything 结构主义框架的核心：
  近似不是"简化"，而是"沿态射路径投影到低维/可解子空间"。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from math_anything.utils.safe_eval import SafeEvalError, safe_eval


class MorphismCategory(StrEnum):
    """态射的数学范畴."""

    APPROXIMATION = "approximation"  # 近似：维度降低/准确度损失
    DISCRETIZATION = "discretization"  # 离散化：连续→离散
    RESTRICTION = "restriction"  # 限制：缩小定义域
    PROJECTION = "projection"  # 投影：降低自由度
    SURROGATE = "surrogate"  # 代理：用一个函数近似另一个
    EMBEDDING = "embedding"  # 嵌入：子结构→母结构
    QUOTIENT = "quotient"  # 商：同余类归并
    TRANSFORMATION = "transformation"  # 一般变换


@dataclass
class StructuralChange:
    """一个结构性质的改变记录."""

    property_name: str
    before: Any  # 变换前
    after: Any  # 变换后
    consequence: str = ""  # 后果描述


@dataclass
class Morphism(ABC):
    """Abstract morphism base class.

    Subclasses should define invariants_kept/lost/introduced
    either as @property or as regular class-level attributes.
    """

    name: str
    source_type: str
    target_type: str
    category: str = MorphismCategory.APPROXIMATION
    changes: list[StructuralChange] = field(default_factory=list)
    is_injective: bool = True
    is_surjective: bool = True
    is_isomorphism: bool = False
    kernel_description: str = ""
    condition: str = ""

    def __post_init__(self):
        if not hasattr(self, "invariants_kept"):
            self.invariants_kept = []
        if not hasattr(self, "invariants_lost"):
            self.invariants_lost = []
        if not hasattr(self, "invariants_introduced"):
            self.invariants_introduced = []

    @property
    @abstractmethod
    def mathematical_form(self) -> str:
        """态射的数学表达式."""
        ...

    @property
    def is_invertible(self) -> bool:
        """态射是否有逆（同构）."""
        return self.is_isomorphism

    def compose(self, other: "Morphism") -> "CompositeMorphism":
        """态射合成：this ∘ other = f(g(x))."""
        return CompositeMorphism(
            name=f"{self.name} ∘ {other.name}",
            source_type=other.source_type,
            target_type=self.target_type,
            first=other,  # 先应用
            second=self,  # 后应用
        )

    def apply(self, input_data: Any) -> Any:
        """Apply this morphism to input data. Override in subclasses."""
        raise NotImplementedError(f"Morphism '{self.name}' does not implement apply()")

    def apply_condition(self, params: dict[str, Any]) -> bool:
        """检查态射的适用条件."""
        if not self.condition:
            return True
        try:
            return safe_eval(self.condition, params)
        except (SafeEvalError, Exception):
            return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the morphism metadata to a dictionary."""
        return {
            "name": self.name,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "category": self.category,
            "mathematical_form": self.mathematical_form,
            "invariants_kept": self.invariants_kept,
            "invariants_lost": self.invariants_lost,
            "invariants_introduced": self.invariants_introduced,
            "kernel": self.kernel_description,
            "is_injective": self.is_injective,
            "is_surjective": self.is_surjective,
            "is_isomorphism": self.is_isomorphism,
            "condition": self.condition,
        }


@dataclass
class CompositeMorphism(Morphism):
    """两个态射的合成."""

    first: Morphism = field(default=None)
    second: Morphism = field(default=None)

    @property
    def mathematical_form(self) -> str:
        return "f ∘ g"

    @property
    def invariants_kept(self) -> list[str]:
        """合成态射保持的性质：两个都保持的."""
        if not self.first or not self.second:
            return []
        return [inv for inv in self.first.invariants_kept if inv in self.second.invariants_kept]

    @invariants_kept.setter
    def invariants_kept(self, value: list[str]) -> None:
        pass

    @property
    def invariants_lost(self) -> list[str]:
        """合成态射丢失的性质：任意一个丢失的都算."""
        if not self.first or not self.second:
            return []
        lost = set(self.first.invariants_lost)
        for inv in self.first.invariants_kept:
            if inv in self.second.invariants_lost:
                lost.add(inv)
        lost.update(self.second.invariants_lost)
        return list(lost)

    @invariants_lost.setter
    def invariants_lost(self, value: list[str]) -> None:
        pass

    @property
    def kernel_description(self) -> str:
        return "ker(g) ∪ g⁻¹(ker(f))"

    @kernel_description.setter
    def kernel_description(self, value: str) -> None:
        pass

    def apply(self, input_data: Any) -> Any:
        """Apply composite morphism: second.apply(first.apply(input))."""
        result = self.first.apply(input_data)
        return self.second.apply(result)


# ── 预定义：普适态射 ──


@dataclass
class ContinuumToDiscrete(Morphism):
    """连续→离散化态射.

    这是所有数值方法共享的元态射。
    每个具体离散化（FEM, FDM, FVM, 谱方法）是其子态射。
    """

    name: str = "continuum_to_discrete"
    source_type: str = "EvolutionProblem | EquilibriumProblem"
    target_type: str = "DiscreteAlgebraicSystem"
    category: str = MorphismCategory.DISCRETIZATION
    is_surjective: bool = False  # 离散空间是连续空间的真子集

    method: str = ""  # "fem", "fdm", "fvm", "spectral", "plane_wave"
    grid_size: float | None = None

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "conservation_laws_at_discrete_level",
            "linearity (if original is linear)",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "exact_solution",
            "infinite_dimensional_completeness",
            "frequency_components_above_nyquist",
            "pointwise_satisfaction_of_pde",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "discretization_error",
            "stability_constraint (CFL / timestep)",
            "mesh_dependency",
        ]
    )
    kernel_description: str = "Frequencies/spatial scales below the grid resolution"

    @property
    def mathematical_form(self) -> str:
        forms = {
            "fem": "u(x) ≈ Σ N_i(x) u_i  (Galerkin projection onto V_h)",
            "fdm": "∂u/∂x ≈ (u_{i+1} - u_{i-1})/(2h)",
            "fvm": "∫∂Ω F·n dS ≈ Σ F_f·n_f A_f",
            "spectral": "u(x) ≈ Σ_{|k|<N} c_k e^{ikx}",
            "plane_wave": "ψ(r) ≈ Σ_{|G|²/2 < E_cut} c_G exp(iG·r)",
        }
        return forms.get(self.method, "u(x) → u_h(x)")


@dataclass
class DimensionReductionMorphism(Morphism):
    """降维态射：降低物理维度.

    实例：
      - 3D → 2D（薄板、平面应力/应变）
      - 3D → 1D（梁、桁架）
      - 轴对称（3D → 2D r-z 平面）
    """

    name: str = "dimension_reduction"
    source_type: str = "Continuum_3D"
    target_type: str = "ReducedDimensionModel"
    category: str = MorphismCategory.RESTRICTION
    is_surjective: bool = False

    from_dim: int = 3
    to_dim: int = 2
    assumption: str = ""  # "plane_stress", "plane_strain", "axisymmetric"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "governing_equation_form",
            "conservation_laws_in_reduced_dim",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "out_of_plane_stress_components",
            "3D_surface_effects",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "plane_stress/strain_assumption",
            "reduced_integration_rule",
        ]
    )
    kernel_description: str = "Out-of-plane degrees of freedom"

    @property
    def mathematical_form(self) -> str:
        return f"R^{self.from_dim} → R^{self.to_dim}"


@dataclass
class TimeSteppingMorphism(Morphism):
    """时间步进态射：连续时间→离散时间步.

    ∂u/∂t = F(u) → u^{n+1} = G(u^n, Δt)
    """

    name: str = "time_stepping"
    source_type: str = "EvolutionProblem"
    target_type: str = "DiscreteTimeEvolution"
    category: str = MorphismCategory.DISCRETIZATION
    is_surjective: bool = False

    method: str = "explicit_euler"  # "explicit_euler", "implicit_euler", "verlet", "rk4", "bdf"
    timestep: float | None = None

    invariants_kept: list[str] = field(default_factory=list)
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "continuous_time_symmetry",
            "exact_energy_conservation (for non-symplectic methods)",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "timestep_error",
            "stability_region",
        ]
    )
    kernel_description: str = "Dynamics between timesteps"

    @property
    def mathematical_form(self) -> str:
        forms = {
            "explicit_euler": "u^{n+1} = u^n + Δt F(u^n)",
            "implicit_euler": "u^{n+1} = u^n + Δt F(u^{n+1})",
            "verlet": "r^{n+1} = 2r^n - r^{n-1} + (Δt²/m) F^n",
            "velocity_verlet": "v^{n+1/2} = v^n + (Δt/2m) F^n; r^{n+1} = r^n + Δt v^{n+1/2}",
            "rk4": "k₁ = F(u^n), k₂ = F(u^n + Δt/2 k₁), ...",
        }
        return forms.get(self.method, "u^{n+1} = G(u^n, Δt)")
