r"""算子代数结构家族。

Operator Algebras = \*-代数、C\*-代数、von Neumann 代数、态与非交换概率。

涵盖：
  - 代数结构：\*-代数 → C\*-代数 → von Neumann 代数
  - 态与表示：GNS 构造、Gelfand-Naimark 定理
  - 模理论：Tomita-Takesaki、KMS 条件
  - 非交换概率：统一经典与量子的概率框架

本模块为 re-export 入口，实现拆分为：
  - algebras_star: StarAlgebra, CStarAlgebra
  - algebras_vonneumann: VonNeumannAlgebra, State, GNSConstruction, QuantumState, ...
"""

from __future__ import annotations

from .algebras_star import CStarAlgebra, StarAlgebra
from .algebras_vonneumann import (
    ClassicalState,
    CommutativeCase,
    GNSConstruction,
    NoncommutativeProbability,
    PureState,
    QuantumState,
    State,
    TomitaTakesakiTheory,
    VonNeumannAlgebra,
    _matrix_intersection,
)

__all__ = [
    "StarAlgebra",
    "CStarAlgebra",
    "VonNeumannAlgebra",
    "State",
    "ClassicalState",
    "QuantumState",
    "PureState",
    "GNSConstruction",
    "CommutativeCase",
    "TomitaTakesakiTheory",
    "NoncommutativeProbability",
    "_matrix_intersection",
]
