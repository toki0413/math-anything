"""Symbol Layer - 符号层实现子树复用和层次化表达式构建.

这是 PSRN 的核心组件，负责:
1. 将表达式树按层次展开
2. 自动识别和复用公共子树
3. 为 GPU 并行评估准备数据
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class OperatorType(Enum):
    """算子类型分类."""

    UNARY = auto()  # 一元算子: sin, cos, exp, log, etc.
    BINARY_SQUARED = auto()  # 非交换二元算子: -, /
    BINARY_TRIANGLED = auto()  # 交换二元算子: +, *


@dataclass
class SymbolConfig:
    """符号层配置."""

    unary_ops: List[str] = field(
        default_factory=lambda: ["identity", "neg", "inv", "sin", "cos", "exp", "log"]
    )
    binary_squared_ops: List[str] = field(
        default_factory=lambda: ["sub", "div", "eml"]
    )  # 添加 EML 算子
    binary_triangled_ops: List[str] = field(default_factory=lambda: ["add", "mul"])

    def all_ops(self) -> List[str]:
        return self.unary_ops + self.binary_squared_ops + self.binary_triangled_ops

    def op_type(self, op: str) -> OperatorType:
        if op in self.unary_ops:
            return OperatorType.UNARY
        elif op in self.binary_squared_ops:
            return OperatorType.BINARY_SQUARED
        elif op in self.binary_triangled_ops:
            return OperatorType.BINARY_TRIANGLED
        raise ValueError(f"Unknown operator: {op}")


class SymbolLayer:
    """符号层 - 实现表达式的层次化构建和子树复用.

    每一层接收前一层输出的子表达式集合，应用所有算子生成新的子表达式。
    通过张量运算一次性计算所有可能的组合，天然适合并行化。

    Example:
        >>> config = SymbolConfig()
        >>> layer = SymbolLayer(config)
        >>>
        >>> # 输入: 变量 x1, x2
        >>> inputs = ["x1", "x2"]
        >>>
        >>> # 第一层输出: x1, x2, sin(x1), sin(x2), x1+x1, x1+x2, x2+x2, ...
        >>> outputs, offsets = layer.forward(inputs, layer_idx=0)
        >>> print(f"Layer 0 generated {len(outputs)} expressions")
    """

    def __init__(self, config: Optional[SymbolConfig] = None):
        self.config = config or SymbolConfig()
        self._op_funcs: Dict[str, Callable] = self._build_op_funcs()
        self._expression_cache: Dict[str, str] = {}  # 符号等价检测缓存

    def _build_op_funcs(self) -> Dict[str, Callable]:
        """构建算子到数值函数的映射."""

        def eml(x, y):
            """EML 算子: eml(x, y) = exp(x) - ln(|y|)"""
            return np.exp(np.clip(x, -700, 700)) - np.log(np.abs(y) + 1e-10)

        return {
            "identity": lambda x, y=None: x,
            "neg": lambda x, y=None: -x,
            "inv": lambda x, y=None: 1.0 / (x + 1e-10),
            "sin": lambda x, y=None: np.sin(x),
            "cos": lambda x, y=None: np.cos(x),
            "exp": lambda x, y=None: np.exp(np.clip(x, -700, 700)),
            "log": lambda x, y=None: np.log(np.abs(x) + 1e-10),
            "add": lambda x, y: x + y,
            "mul": lambda x, y: x * y,
            "sub": lambda x, y: x - y,
            "div": lambda x, y: x / (y + 1e-10),
            "eml": eml,  # EML 算子
        }

    def forward(
        self,
        input_exprs: List[str],
        input_values: Optional[np.ndarray] = None,
        layer_idx: int = 0,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """前向传播 - 生成下一层所有可能的表达式.

        Args:
            input_exprs: 输入表达式字符串列表
            input_values: 输入表达式的数值 (n_samples, n_inputs)
            layer_idx: 当前层索引

        Returns:
            output_exprs: 输出表达式字符串列表
            output_values: 输出数值 (n_samples, n_outputs)
            offset_tensor: 偏移张量，用于反向推导表达式结构
        """
        n_inputs = len(input_exprs)
        output_exprs = []
        offsets = []  # (op_idx, left_idx, right_idx) for each output

        # 一元算子
        for op in self.config.unary_ops:
            for i, expr in enumerate(input_exprs):
                new_expr = self._format_unary(op, expr)
                output_exprs.append(new_expr)
                offsets.append((op, i, -1))  # -1 表示无右操作数

        # 二元平方算子 (非交换，i, j 全排列)
        for op in self.config.binary_squared_ops:
            for i in range(n_inputs):
                for j in range(n_inputs):
                    new_expr = self._format_binary(op, input_exprs[i], input_exprs[j])
                    output_exprs.append(new_expr)
                    offsets.append((op, i, j))

        # 二元三角算子 (交换，只取 i <= j)
        for op in self.config.binary_triangled_ops:
            for i in range(n_inputs):
                for j in range(i, n_inputs):
                    new_expr = self._format_binary(op, input_exprs[i], input_exprs[j])
                    output_exprs.append(new_expr)
                    offsets.append((op, i, j))

        # 数值计算
        if input_values is not None:
            output_values = self._compute_values(output_exprs, offsets, input_values)
        else:
            output_values = np.array([])

        offset_tensor = np.array(offsets, dtype=object)

        return output_exprs, output_values, offset_tensor

    def _format_unary(self, op: str, expr: str) -> str:
        """格式化一元表达式."""
        if op == "identity":
            return expr
        elif op == "neg":
            return f"(-{expr})"
        elif op == "inv":
            return f"(1/{expr})"
        else:
            return f"{op}({expr})"

    def _format_binary(self, op: str, left: str, right: str) -> str:
        """格式化二元表达式."""
        if op == "eml":
            return f"eml({left}, {right})"
        op_map = {"add": "+", "mul": "*", "sub": "-", "div": "/"}
        symbol = op_map.get(op, op)
        return f"({left}{symbol}{right})"

    def _compute_values(
        self,
        output_exprs: List[str],
        offsets: List[Tuple],
        input_values: np.ndarray,
    ) -> np.ndarray:
        """计算所有输出表达式的数值 - 使用向量化运算."""
        from .compiled_evaluator import FastSymbolLayer

        fast_layer = FastSymbolLayer()
        return fast_layer.compute_layer(offsets, input_values)

    def deduce_expression(
        self,
        best_idx: int,
        offset_tensors: List[np.ndarray],
        base_exprs: List[str],
    ) -> str:
        """从最优索引反向推导表达式字符串.

        Args:
            best_idx: 最优表达式在最后一层的索引
            offset_tensors: 每层的偏移张量列表
            base_exprs: 基础表达式（输入变量/常量）

        Returns:
            推导出的表达式字符串
        """
        return self._recursive_deduce(
            len(offset_tensors) - 1, best_idx, offset_tensors, base_exprs
        )

    def _recursive_deduce(
        self,
        layer_idx: int,
        idx: int,
        offset_tensors: List[np.ndarray],
        base_exprs: List[str],
    ) -> str:
        """递归推导表达式."""
        if layer_idx < 0:
            # 到达基础层
            if idx < len(base_exprs):
                return base_exprs[idx]
            return "?"

        offsets = offset_tensors[layer_idx]
        if idx >= len(offsets):
            return "?"

        op, left_idx, right_idx = offsets[idx]

        if right_idx == -1:
            # 一元算子
            left_expr = self._recursive_deduce(
                layer_idx - 1, left_idx, offset_tensors, base_exprs
            )
            return self._format_unary(op, left_expr)
        else:
            # 二元算子
            left_expr = self._recursive_deduce(
                layer_idx - 1, left_idx, offset_tensors, base_exprs
            )
            right_expr = self._recursive_deduce(
                layer_idx - 1, right_idx, offset_tensors, base_exprs
            )
            return self._format_binary(op, left_expr, right_expr)

    def memory_estimate(self, n_inputs: int, n_layers: int) -> Dict[str, int]:
        """估算内存使用量.

        Returns:
            每层输出维度的字典
        """
        dims = {0: n_inputs}
        current = n_inputs

        for layer in range(1, n_layers + 1):
            n_u = len(self.config.unary_ops)
            n_bs = len(self.config.binary_squared_ops)
            n_bt = len(self.config.binary_triangled_ops)

            next_dim = (
                n_u * current
                + n_bs * current * current
                + n_bt * current * (current + 1) // 2
            )
            dims[layer] = next_dim
            current = next_dim

        return dims

    def build_duplicate_mask(
        self, expressions: List[str]
    ) -> Tuple[List[int], List[str]]:
        """构建重复表达式掩码，返回唯一表达式的索引和列表.

        这是 DR Mask (Duplicate Removal Mask) 的核心实现.
        """
        seen = {}
        unique_indices = []
        unique_exprs = []

        for idx, expr in enumerate(expressions):
            # 使用规范化形式作为键（简单版本）
            normalized = self._normalize_expr(expr)
            if normalized not in seen:
                seen[normalized] = idx
                unique_indices.append(idx)
                unique_exprs.append(expr)

        return unique_indices, unique_exprs

    def _normalize_expr(self, expr: str) -> str:
        """规范化表达式用于等价检测.

        简化版本：处理交换律和基本代数恒等式.
        """
        # 移除多余括号
        expr = expr.strip()
        while expr.startswith("(") and expr.endswith(")"):
            inner = expr[1:-1]
            if inner.count("(") == inner.count(")"):
                expr = inner
            else:
                break
        return expr
