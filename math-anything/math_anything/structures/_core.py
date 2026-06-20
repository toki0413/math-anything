"""核心结构性质与不变量类定义.

本模块仅包含 StructuralProperty 和 StructuralInvariant 的类定义，
不含任何对 invariant_registry 的引用，避免循环导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from math_anything.utils.safe_eval import SafeEvalError, safe_eval


@dataclass(slots=True)
class StructuralProperty:
    """结构的数学性质.

    这是描述性的，不包含值判断。
    例如："H[n] 是自伴的"是一个性质；
    它隐含的不变量"特征值均为实数"是 StructuralInvariant。
    """

    name: str
    value: bool | str | float | int  # bool, str, float, int
    description: str = ""
    mathematical_context: str = ""  # 如 "谱定理", "Noether定理"

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "value": str(self.value),
            "description": self.description,
            "context": self.mathematical_context,
        }


@dataclass(slots=True)
class StructuralInvariant:
    """从结构性质推导出的不变量.

    不变量是该结构类型所有实例都必须满足的数学约束。
    它不是"建议"或"最佳实践"，而是定理的逻辑推论。

    条件不变量：仅当条件成立时才适用。
    无条件的：始终成立。
    """

    name: str
    expression: str  # "λ_i ∈ ℝ for all i"
    theorem: str  # "Spectral Theorem for Self-Adjoint Operators"
    condition: str | None = None  # 如 "operator_type == SELF_ADJOINT"
    severity: str = "theorem"  # "theorem" | "consistency" | "conservation"
    affected_quantities: list[str] = field(default_factory=list)
    proof_sketch: str = ""

    def is_active(self, properties: dict[str, bool | str | float | int]) -> bool:
        """检查该不变量在当前性质集合下是否激活."""
        if self.condition is None:
            return True
        # 简单条件求值
        try:
            return safe_eval(self.condition, properties)
        except (SafeEvalError, Exception):
            return True  # 条件无法求值时默认为激活

    def to_dict(self) -> dict[str, str | list[str] | None]:
        return {
            "name": self.name,
            "expression": self.expression,
            "theorem": self.theorem,
            "condition": self.condition,
            "severity": self.severity,
            "affected_quantities": self.affected_quantities,
        }


__all__ = ["StructuralProperty", "StructuralInvariant"]
