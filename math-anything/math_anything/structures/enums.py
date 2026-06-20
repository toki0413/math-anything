"""结构性质相关的枚举类型.

OperatorType, SpectrumType, SymmetryGroup, StructureDomain, VariationalPrinciple
"""

from __future__ import annotations

from enum import StrEnum


class OperatorType(StrEnum):
    """算子的数学类型."""

    SELF_ADJOINT = "self_adjoint"
    NORMAL = "normal"
    NON_SELF_ADJOINT = "non_self_adjoint"
    UNITARY = "unitary"
    POSITIVE_DEFINITE = "positive_definite"
    BOUNDED_BELOW = "bounded_below"
    COMPACT = "compact"


class SpectrumType(StrEnum):
    """谱的类型."""

    PURE_POINT = "pure_point"  # 纯点谱（束缚态）
    CONTINUOUS = "continuous"  # 连续谱（散射态）
    MIXED = "mixed"  # 混合谱
    BAND = "band"  # 能带结构（周期系统）


class SymmetryGroup(StrEnum):
    """结构的对称群."""

    TRANSLATION = "translation"
    ROTATION_SO3 = "rotation_SO3"
    REFLECTION = "reflection"
    POINT_GROUP = "point_group"
    SPACE_GROUP = "space_group"
    GAUGE_U1 = "gauge_U1"
    GAUGE_SU2 = "gauge_SU2"
    LORENTZ = "lorentz"
    GALILEAN = "galilean"
    SCALING = "scaling"
    PERMUTATION = "permutation"


class VariationalPrinciple(StrEnum):
    """变分原理类型."""

    STATIONARY = "stationary"  # δE = 0
    MINIMUM = "minimum"  # min E
    MAXIMUM = "maximum"  # max E
    HAMILTONIAN = "hamiltonian"  # Hamilton 最小作用量
    SELF_CONSISTENT = "self_consistent"  # Kohn-Sham 型
    CONSTRAINED = "constrained"  # 约束优化（如 Lagrange 乘子）


class StructureDomain(StrEnum):
    """结构所在的定义域类型."""

    CONTINUUM = "continuum"  # 连续介质 R³, Ω⊂R³
    LATTICE = "lattice"  # 晶格
    PARTICLES = "particles"  # 粒子系 R^{3N}
    MESH = "mesh"  # 离散网格
    GRAPH = "graph"  # 图结构（ML势）


__all__ = [
    "OperatorType",
    "SpectrumType",
    "SymmetryGroup",
    "VariationalPrinciple",
    "StructureDomain",
]
