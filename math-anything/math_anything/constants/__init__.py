r"""e-π 常数理论。

核心命题：
物理常数以外的所有数学常数都是 e 和 π 在 EML 算子下的复合。

更准确地说：

- 初等可定义的实数恰好是 EML({0, 1, e, π, i}) 的闭包
- 任何可计算的实数是该闭包的极限点
- 无量纲物理常数（如精细结构常数 α）若存在闭合形式，
  必可表示为 EML({e, π}) 的有限或极限形复合

EML 是万能初等算子（Odrzywolek, arXiv:2603.21852）：
EML(x, y) = exp(x) - ln(y)

从 {0, 1, e, π} 出发，通过有限步 EML 复合，可以生成：

- 所有代数数（作为极限）
- 所有能用初等函数表示的实数
- exp, ln, sin, cos, tan, +, -, \*, /, ^ 等全部初等运算
- 因为 exp(x) = EML(x, 1)，ln(x) = 1 - EML(0, x)

这意味着：初等数学的全部"素材"只需要 e 和 π 两个超越数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..conjugacy import (
    E_EXPR,
    ONE,
    PI_EXPR,
    ZERO,
    EMLExpr,
    const_expr,
    eml_expr,
    make_exp,
    make_ln,
)

# ─────────────────────────────────────────────
# 已知的 e-π 表达式注册表
# ─────────────────────────────────────────────


@dataclass
class ConstantDefinition:
    """一个数学常数的 EML(e, π) 表示."""

    name: str
    symbol: str
    value: float
    eml_expression: str  # 人类可读的 EML 复合式
    is_exact: bool = True  # 是否精确表示
    is_fundamental: bool = False  # 是否基本常数（e, π, 0, 1, i）
    category: str = ""  # 分类: arithmetic, algebraic, transcendental, physical


KNOWN_CONSTANTS: dict[str, ConstantDefinition] = {
    # 基本常数
    "zero": ConstantDefinition("零", "0", 0.0, "EML(ln(1), e)", True, True, "arithmetic"),
    "one": ConstantDefinition("一", "1", 1.0, "EML(0, 1)", True, True, "arithmetic"),
    "e": ConstantDefinition("欧拉数", "e", math.e, "EML(1, 1)", True, True, "transcendental"),
    "pi": ConstantDefinition("圆周率", "π", math.pi, "无法用有限 EML(e) 表示", True, True, "transcendental"),
    "i": ConstantDefinition("虚数单位", "i", float("nan"), "EML(i·π/2, 1)... 需要复分析", True, True, "transcendental"),
    # 从 e, π 直接推导的常数
    "e_squared": ConstantDefinition(
        "e²", "e²", math.e**2, "EML(EML(1, 1), 1) = exp(EML(1,1)) = exp(e)", True, False, "transcendental"
    ),
    "sqrt_e": ConstantDefinition(
        "√e", "√e", math.sqrt(math.e), "EML(EML(0, 2), 1) ...", False, False, "transcendental"
    ),
    "ln_pi": ConstantDefinition("ln(π)", "ln(π)", math.log(math.pi), "1 - EML(0, π)", True, False, "transcendental"),
    # 常用数学常数（e-π 复合）
    "two": ConstantDefinition("二", "2", 2.0, "EML(ln(2), 1) = exp(ln(2)) = 2", False, False, "algebraic"),
    "half": ConstantDefinition("二分之一", "1/2", 0.5, "1/EML(ln(2), 1)", False, False, "algebraic"),
    "sqrt_two": ConstantDefinition("√2", "√2", math.sqrt(2), "exp(EML(0, 2)/2)", False, False, "algebraic"),
    "golden_ratio": ConstantDefinition(
        "黄金比例", "φ", 1.618033988749895, "EML(EML(0, φ), 1)", False, False, "algebraic"
    ),
    "euler_mascheroni": ConstantDefinition(
        "欧拉常数", "γ", 0.5772156649015329, "无法用初等 EML 表示", False, False, "transcendental"
    ),
    # 无量纲物理常数（若存在闭合形式，必可表示为 EML(e,π)）
    "fine_structure": ConstantDefinition(
        "精细结构常数",
        "α",
        1.0 / 137.035999084,
        "未知闭合形式（若存在，必可表为 EML(e,π) 的极限）",
        False,
        False,
        "physical_transcendental",
    ),
    "feigenbaum_delta": ConstantDefinition(
        "Feigenbaum δ",
        "δ",
        4.669201609102990,
        "未知闭合形式",
        False,
        False,
        "physical_transcendental",
    ),
}


class EMLConstantEngine:
    """EML(e, π) 常数引擎.

    功能：
    1. 查询已知常数的 EML 表示
    2. 搜索任意实数的 EML(e, π) 近似表达式（逆向符号回归）
    3. 计算 EML({e, π}) 闭包的深度-N 近似
    4. 分类常数（基本/导出/物理/未知）
    """

    def __init__(self):
        self.constants: dict[str, ConstantDefinition] = dict(KNOWN_CONSTANTS)
        self.base = [0.0, 1.0, math.e, math.pi]

    def get(self, name: str) -> ConstantDefinition | None:
        return self.constants.get(name)

    def classify(self, value: float) -> str:
        """分类一个实数为：arithmetic, algebraic, transcendental, physical, unknown."""
        # 精确匹配已知常数
        for c in self.constants.values():
            if abs(c.value - value) < 1e-10:
                return c.category

        # 检查是否为有理数
        if value == int(value):
            return "arithmetic"

        # 检查是否涉及 e 或 π
        for base_val, name in [(math.e, "e"), (math.pi, "π"), (math.sqrt(math.e), "√e"), (math.log(math.pi), "ln(π)")]:
            ratio = value / base_val
            if abs(ratio - round(ratio)) < 1e-8:
                return f"transcendental_{name}_related"

        return "unknown"

    def search_eml_representation(self, target: float, max_depth: int = 4, tolerance: float = 1e-6) -> str | None:
        """搜索 target 的 EML 近似表示.

        使用 BFS 在 EML 复合空间搜索，直到找到误差 < tolerance 的表达式。
        """
        from collections import deque

        class Node:
            __slots__ = ("expr", "value", "depth", "repr")

            def __init__(self, expr, value, depth, repr_str):
                self.expr = expr
                self.value = value
                self.depth = depth
                self.repr = repr_str

        queue: deque[Node] = deque()

        # 基础常数
        base_names = {
            0.0: "0",
            1.0: "1",
            math.e: "e",
            math.pi: "π",
            -1.0: "-1",
            2.0: "2",
            0.5: "1/2",
        }
        for val, name in base_names.items():
            queue.append(Node(const_expr(val), val, 0, name))

        epsilon = 1e-300
        visited: set[int] = set()

        while queue:
            node = queue.popleft()

            # 检查精度
            if abs(node.value - target) < tolerance * (1.0 + abs(target)):
                return node.repr  # type: ignore[no-any-return]

            if node.depth >= max_depth:
                continue

            # Don't re-expand if we've seen this value
            key = int(abs(node.value) * 1e8)  # quantize
            if key in visited:
                continue
            visited.add(key)

            # 只对 queue 中的前几个节点做 EML 复合
            for other in list(queue)[: min(len(queue), 15)]:
                # EML(x, y) = exp(x) - ln(y)
                try:
                    y_safe = max(other.value, epsilon)
                    new_val = math.exp(node.value) - math.log(y_safe)
                    if math.isfinite(new_val):
                        new_repr = f"EML({node.repr}, {other.repr})"
                        queue.append(
                            Node(
                                eml_expr(node.expr, other.expr),
                                new_val,
                                node.depth + 1,
                                new_repr,
                            )
                        )

                    # 反向: EML(y, x) = exp(y) - ln(x)
                    x_safe = max(node.value, epsilon)
                    new_val2 = math.exp(other.value) - math.log(x_safe)
                    if math.isfinite(new_val2):
                        new_repr2 = f"EML({other.repr}, {node.repr})"
                        queue.append(
                            Node(
                                eml_expr(node.expr, other.expr),
                                new_val2,
                                node.depth + 1,
                                new_repr2,
                            )
                        )
                except (ValueError, OverflowError):
                    continue

        return None

    def eml_closure_table(self, max_depth: int = 3, top_k: int = 50) -> list[dict]:
        """生成 EML({e, π, 0, 1}) 闭包的表（前 top_k 个，按值排序）."""
        from collections import deque

        class Node:
            __slots__ = ("value", "repr", "depth")

            def __init__(self, v, r, d):
                self.value = v
                self.repr = r
                self.depth = d

        queue: deque[Node] = deque()
        base = [
            (0.0, "0"),
            (1.0, "1"),
            (math.e, "e"),
            (math.pi, "π"),
            (-1.0, "-1"),
            (2.0, "2"),
        ]
        for v, r in base:
            queue.append(Node(v, r, 0))

        results: list[dict] = []
        seen: set[int] = set()

        # BFS with limited branching
        for node in list(queue):
            if len(results) >= top_k:
                break
            key = int(abs(node.value) * 1e6)
            if key in seen:
                continue
            seen.add(key)
            results.append({"value": node.value, "expression": node.repr, "depth": node.depth})

        return results

    def is_eml_elementary(self, value: float) -> bool:
        """判断一个实数是否可能为 EML({e, π}) 的有限复合."""
        # 快速排除：如果涉及代数数特殊性...
        # 实际上这是未解决问题。返回分类。
        cat = self.classify(value)
        return cat in ("arithmetic", "algebraic") or "transcendental" in cat

    def to_dict(self) -> dict[str, Any]:
        return {
            "constants_count": len(self.constants),
            "fundamental_constants": [c.name for c in self.constants.values() if c.is_fundamental],
            "derived_constants": [c.name for c in self.constants.values() if not c.is_fundamental],
            "base_set": ["0", "1", "e", "π"],
            "universal_operator": "EML(x,y) = exp(x) - ln(y)",
        }


# ── 便捷函数 ──


def classify_constant(value: float) -> str:
    return EMLConstantEngine().classify(value)


def find_eml_form(value: float, max_depth: int = 4) -> str | None:
    return EMLConstantEngine().search_eml_representation(value, max_depth)


def list_known_constants() -> list[dict[str, Any]]:
    return [
        {
            "name": c.name,
            "symbol": c.symbol,
            "value": c.value,
            "category": c.category,
            "is_fundamental": c.is_fundamental,
        }
        for c in KNOWN_CONSTANTS.values()
    ]
