"""约束系统 — 可学习的不变量、传播、边界演化、域学习.

核心能力:
  - LearnedInvariant: 域可学习、可弱化、可在态射链上传播的不变量
  - ConstraintPropagation: 不变量在态射链上的精确传播
  - BoundaryEvolution: 基于经验的操作边界动态调整
  - DomainLearner: 从数据中学习不变量的适用域

区别于 mat sci-agent 的 constraints/ 模块:
  mat sci-agent 的约束是"质量检查"（判断计算结果对不对）
  math-anything 的约束是"结构保证"（在什么条件下哪些数学定律成立）

两者互补:
  math-anything 提供不变量的数学推导 + 域学习
  mat sci-agent 使用 math-anything 的输出来判断计算结果
"""

from .boundary import (
    BoundaryEvolution,
    BoundaryState,
    ExecutionRecord,
    RiskItem,
)
from .domain import (
    DomainHypothesis,
    DomainLearner,
    TrainingExample,
)
from .invariant import (
    DomainCondition,
    InvariantStatus,
    LearnedInvariant,
    PropagationOutcome,
    WeakeningRule,
    from_structural_invariant,
)
from .propagation import (
    ConstraintPropagation,
    PropagationChain,
    PropagationResult,
)

__all__ = [
    "LearnedInvariant",
    "InvariantStatus",
    "PropagationOutcome",
    "WeakeningRule",
    "DomainCondition",
    "from_structural_invariant",
    "ConstraintPropagation",
    "PropagationResult",
    "PropagationChain",
    "BoundaryState",
    "BoundaryEvolution",
    "ExecutionRecord",
    "RiskItem",
    "DomainLearner",
    "DomainHypothesis",
    "TrainingExample",
]
