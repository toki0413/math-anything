"""领域专用近似态射库。

每个近似态射明确记录：
  - 源结构 → 目标结构
  - 保持/丢失/引入的不变量
  - 适用条件
  - 数学表达式

这些不是"参数检查"，而是"结构变换路径"。

本模块为 re-export 入口，实现拆分为：
  - dft: BornOppenheimer, KohnSham, PlaneWave, SCF, XC
  - md: ClassicalLimit, ForceField
  - cfd: Incompressibility, Reynolds, TurbulenceClosure, LES
  - quantum: HartreeFock, PostHF
  - surrogate: MLSurrogate, DiffuseInterface
"""

from __future__ import annotations

from .cfd import (
    IncompressibilityMorphism,
    LESFilteringMorphism,
    ReynoldsDecompositionMorphism,
    TurbulenceModelClosureMorphism,
)
from .dft import (
    BornOppenheimerApproximation,
    ExchangeCorrelationApproximation,
    KohnShamMapping,
    PlaneWaveTruncation,
    SCFIterationMorphism,
)
from .md import (
    ClassicalLimitMorphism,
    ForceFieldMorphism,
)
from .quantum import (
    HartreeFockMorphism,
    PostHartreeFockMorphism,
)
from .surrogate import (
    DiffuseInterfaceMorphism,
    MLSurrogateMorphism,
)

__all__ = [
    # DFT
    "BornOppenheimerApproximation",
    "KohnShamMapping",
    "PlaneWaveTruncation",
    "SCFIterationMorphism",
    "ExchangeCorrelationApproximation",
    # MD
    "ClassicalLimitMorphism",
    "ForceFieldMorphism",
    # CFD
    "IncompressibilityMorphism",
    "ReynoldsDecompositionMorphism",
    "TurbulenceModelClosureMorphism",
    "LESFilteringMorphism",
    # Quantum Chemistry
    "HartreeFockMorphism",
    "PostHartreeFockMorphism",
    # Surrogate / Phase Field
    "MLSurrogateMorphism",
    "DiffuseInterfaceMorphism",
]
