"""Abstract base class for all mathematical structures.

数学结构是第一公民。物理、化学、力学是同一组数学结构的不同实例化。

This is the core architectural inversion of math-anything:
Old: VASP → extractor → H[n]ψ=εψ  (math attached to engine)
New: NonLinearEigenvalueProblem ← instantiated by ← VASP, QE, CP2K

一个 AbstractMathematicalStructure 定义：
  - 该结构类所有实例共享的不变量（定理性质）
  - 该结构的维度空间
  - 该结构的几何空间（Hilbert 空间、Sobolev 空间等）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from math_anything._compat import StrEnum

from .properties import StructuralInvariant


class StructureFamily(StrEnum):
    """数学结构的族分类."""

    SPECTRAL = "spectral"
    EVOLUTION = "evolution"
    EQUILIBRIUM = "equilibrium"
    COUPLED = "coupled"
    ENSEMBLE = "ensemble"
    CATEGORY_THEORY = "category_theory"
    GEOMETRY = "geometry"
    ALGEBRA = "algebra"


@dataclass
class StructureMetadata:
    """结构的元信息."""

    family: StructureFamily
    name: str
    canonical_form: str
    description: str = ""
    references: list[str] = field(default_factory=list)


@dataclass
class AbstractMathematicalStructure(ABC):
    """所有数学结构类型的抽象基类.

    每个子类代表一类数学问题，具有：

    - 规范形式（canonical_form）：该结构类的数学表达
    - 结构不变量（structural_invariants）：该结构类型保证的定理性质
    - 维度空间：该结构所在的几何/函数空间

    使用示例::

        struct = NonlinearEigenvalueProblem(
            operator_type=OperatorType.SELF_ADJOINT,
            dimension=3,
        )
        # 自动推导该结构类的所有不变量
        for inv in struct.structural_invariants:
            logger.info(f"{inv.name}: {inv.expression} ({inv.theorem})")
    """

    metadata: StructureMetadata

    @property
    @abstractmethod
    def structural_invariants(self) -> list[StructuralInvariant]:
        """该结构类型保证的全部不变量（定理性质）."""
        ...

    @property
    @abstractmethod
    def function_space(self) -> str:
        """该结构所在的基本函数空间，如 L²(R³), H¹(Ω), ℝ^{3N}."""
        ...

    @property
    def family(self) -> StructureFamily:
        return self.metadata.family

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def canonical_form(self) -> str:
        return self.metadata.canonical_form

    @property
    def dimensional_rank(self) -> int:
        """该结构涉及的基础维度数量.

        大多数力学/物理结构涉及 3 个基础维度（M, L, T）。
        热力学涉及 4 个（M, L, T, Θ）。
        电磁学涉及 4 个（M, L, T, I）。
        """
        return self._compute_dimensional_rank()

    def _compute_dimensional_rank(self) -> int:
        """从结构的量纲矩阵计算独立维度数."""
        return 0  # 子类覆盖

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "name": self.name,
            "canonical_form": self.canonical_form,
            "function_space": self.function_space,
            "dimensional_rank": self.dimensional_rank,
            "structural_invariants": [inv.to_dict() for inv in self.structural_invariants],
        }
