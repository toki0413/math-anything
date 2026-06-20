"""Structural properties and invariants.

一个数学结构的"性质"（StructuralProperty）是该结构类型的固有属性。
一个"不变量"（StructuralInvariant）是从性质推导出的定理保证的约束。

区别：

- StructuralProperty: 描述性，如 "self_adjoint", "variational", "hamiltonian"
- StructuralInvariant: 推断性，如 "特征值均为实数"（来自自伴性），
  "能量在基态取极小"（来自变分性）

本模块为 re-export 入口，实现拆分为：
  - enums: OperatorType, SpectrumType, SymmetryGroup, StructureDomain, VariationalPrinciple
  - invariant_registry: INVARIANT_REGISTRY, query_invariants, 预定义不变量组
"""

from __future__ import annotations

from ._core import StructuralInvariant, StructuralProperty
from .enums import (
    OperatorType,
    SpectrumType,
    StructureDomain,
    SymmetryGroup,
    VariationalPrinciple,
)
from .invariant_registry import (
    CONSERVATION_LAW_INVARIANTS,
    HAMILTONIAN_INVARIANTS,
    INVARIANT_REGISTRY,
    SPECTRAL_SELF_ADJOINT_INVARIANTS,
    VARIATIONAL_INVARIANTS,
    get_invariants,
    query_invariants,
)

__all__ = [
    # Core classes
    "StructuralProperty",
    "StructuralInvariant",
    # Enums (re-exported from enums.py)
    "OperatorType",
    "SpectrumType",
    "SymmetryGroup",
    "StructureDomain",
    "VariationalPrinciple",
    # Registry (re-exported from invariant_registry.py)
    "SPECTRAL_SELF_ADJOINT_INVARIANTS",
    "VARIATIONAL_INVARIANTS",
    "HAMILTONIAN_INVARIANTS",
    "CONSERVATION_LAW_INVARIANTS",
    "INVARIANT_REGISTRY",
    "get_invariants",
    "query_invariants",
]
