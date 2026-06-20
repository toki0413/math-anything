"""维度分析模块.

提供：
  - Buckingham π 定理（尺度变换群的不变量）
  - 方程维度一致性验证
  - 面向特定领域的分析器（流体力学、量子力学）
"""

from .equation_checker import (
    EquationChecker,
    EquationDimensionalCheck,
    TermDimension,
)
from .scaling_group import (
    BASE_DIMENSIONS,
    BUILTIN_QUANTITIES,
    NAMED_PI_GROUPS,
    BuckinghamPiEngine,
    BuckinghamPiGroup,
    FluidDimensionAnalyzer,
    PhysicalQuantity,
    QMDimensionAnalyzer,
)

__all__ = [
    "BASE_DIMENSIONS",
    "BUILTIN_QUANTITIES",
    "BuckinghamPiEngine",
    "BuckinghamPiGroup",
    "PhysicalQuantity",
    "NAMED_PI_GROUPS",
    "FluidDimensionAnalyzer",
    "QMDimensionAnalyzer",
    "EquationChecker",
    "EquationDimensionalCheck",
    "TermDimension",
]
