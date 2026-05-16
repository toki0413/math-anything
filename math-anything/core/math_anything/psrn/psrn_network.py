"""PSRN (Parallel Symbolic Regression Network) - 并行符号回归网络.

核心思想：通过堆叠 SymbolLayer，一次性生成并评估所有可能的表达式树，
利用子树复用避免重复计算。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gpu_evaluator import GPUEvaluator
from .symbol_layer import SymbolConfig, SymbolLayer


@dataclass
class PSRNConfig:
    """PSRN 网络配置."""

    n_layers: int = 3                    # 符号层数量
    n_input_slots: int = 5               # 输入槽位数量（变量 + 常量）
    symbol_config: SymbolConfig = field(default_factory=SymbolConfig)
    use_gpu: Optional[bool] = None       # 是否使用 GPU
    use_dr_mask: bool = True             # 是否使用重复去除掩码
    max_constants: int = 3               # 最大常量数
    constant_range: Tuple[float, float] = (-3.0, 3.0)  # 常量采样范围


class PSRN:
    """并行符号回归网络.

    通过多层 SymbolLayer 构建表达式树，每层将浅层表达式组合为更深层的表达式。
    所有可能的组合在每一层都被并行生成和评估。

    Example:
        >>> config = PSRNConfig(n_layers=2, n_input_slots=2)
        >>> psrn = PSRN(config)
        >>>
        >>> # 输入数据
        >>> X = np.linspace(0, 1, 100).reshape(-1, 1)
        >>> y = np.sin(X).flatten()
        >>>
        >>> # 运行 PSRN
        >>> best_expr, best_mse = psrn.fit(X, y, variable_names=['x'])
        >>> print(f"Found: {best_expr}, MSE: {best_mse}")
    """

    def __init__(self, config: Optional[PSRNConfig] = None):
        self.config = config or PSRNConfig()
        self.layers: List[SymbolLayer] = [
            SymbolLayer(self.config.symbol_config)
            for _ in range(self.config.n_layers)
        ]
        self.evaluator = GPUEvaluator(use_gpu=self.config.use_gpu)

        # 存储每层的输出用于反向推导
        self._layer_outputs: List[List[str]] = []
        self._layer_values: List[np.ndarray] = []
        self._offset_tensors: List[np.ndarray] = []
        self._base_exprs: List[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None,
        token_exprs: Optional[List[str]] = None,
        token_values: Optional[np.ndarray] = None,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """运行 PSRN 搜索最优表达式.

        Args:
            X: 输入特征 (n_samples, n_features)
            y: 目标值 (n_samples,)
            variable_names: 变量名
            token_exprs: 额外的 token 表达式（来自 TokenGenerator）
            token_values: token 表达式的数值

        Returns:
            best_expr: 最优表达式字符串
            best_mse: 最优均方误差
            top_k: 前 k 个最优表达式及其误差
        """
        # 构建基础表达式集合
        base_exprs, base_values = self._build_base_expressions(
            X, variable_names, token_exprs, token_values
        )

        self._base_exprs = base_exprs
        self._layer_outputs = []
        self._layer_values = []
        self._offset_tensors = []

        # 逐层前向传播
        current_exprs = base_exprs
        current_values = base_values

        for layer_idx, layer in enumerate(self.layers):
            current_exprs, current_values, offsets = layer.forward(
                current_exprs, current_values, layer_idx
            )

            self._layer_outputs.append(current_exprs)
            self._layer_values.append(current_values)
            self._offset_tensors.append(offsets)

            # 应用 DR Mask（除最后一层外）
            if self.config.use_dr_mask and layer_idx < len(self.layers) - 1:
                unique_indices, unique_exprs = layer.build_duplicate_mask(current_exprs)
                current_exprs = unique_exprs
                current_values = current_values[:, unique_indices]
                # 注意：offsets 也需要相应调整，简化实现中省略

        # 在最后一层选择最优表达式
        best_idx, best_mse, all_mses = self._find_best_expression(
            current_values, y
        )

        # 反向推导最优表达式
        best_expr = self.layers[-1].deduce_expression(
            best_idx, self._offset_tensors, self._base_exprs
        )

        # 获取 top-k
        top_k_indices = np.argsort(all_mses)[:10]
        top_k = [
            (
                self.layers[-1].deduce_expression(
                    idx, self._offset_tensors, self._base_exprs
                ),
                all_mses[idx],
            )
            for idx in top_k_indices
        ]

        return best_expr, best_mse, top_k

    def _build_base_expressions(
        self,
        X: np.ndarray,
        variable_names: Optional[List[str]] = None,
        token_exprs: Optional[List[str]] = None,
        token_values: Optional[np.ndarray] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """构建基础表达式集合（变量 + 常量 + tokens）."""
        n_samples = X.shape[0]
        n_features = X.shape[1]

        if variable_names is None:
            variable_names = [f"x{i}" for i in range(n_features)]

        base_exprs = []
        base_values_list = []

        # 添加变量
        for i, name in enumerate(variable_names[:n_features]):
            base_exprs.append(name)
            base_values_list.append(X[:, i])

        # 添加常量
        n_constants = min(self.config.max_constants, self.config.n_input_slots - n_features)
        constants = np.linspace(
            self.config.constant_range[0],
            self.config.constant_range[1],
            n_constants,
        )
        for c in constants:
            base_exprs.append(f"{c:.4f}")
            base_values_list.append(np.full(n_samples, c))

        # 添加 token 表达式
        if token_exprs is not None and token_values is not None:
            for expr, values in zip(token_exprs, token_values.T):
                base_exprs.append(expr)
                base_values_list.append(values)

        base_values = np.column_stack(base_values_list)

        return base_exprs, base_values

    def _find_best_expression(
        self,
        values: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[int, float, np.ndarray]:
        """找到最优表达式.

        Args:
            values: 所有候选表达式的评估值 (n_samples, n_candidates)
            y: 目标值 (n_samples,)

        Returns:
            best_idx: 最优表达式索引
            best_mse: 最优 MSE
            all_mses: 所有表达式的 MSE
        """
        # 广播 y 到与 values 相同的形状
        y_broadcast = y.reshape(-1, 1)

        # 计算所有表达式的 MSE
        mses = np.mean((values - y_broadcast) ** 2, axis=0)

        # 处理 NaN
        mses = np.where(np.isnan(mses), np.inf, mses)

        best_idx = int(np.argmin(mses))
        best_mse = float(mses[best_idx])

        return best_idx, best_mse, mses

    def get_search_space_size(self) -> Dict[int, int]:
        """获取搜索空间大小统计."""
        if not self._layer_outputs:
            # 估算
            return self.layers[0].memory_estimate(
                self.config.n_input_slots, self.config.n_layers
            )

        return {
            i: len(exprs) for i, exprs in enumerate(self._layer_outputs)
        }

    def summary(self) -> str:
        """返回 PSRN 运行摘要."""
        lines = ["PSRN Summary:", "=" * 40]
        lines.append(f"Layers: {self.config.n_layers}")
        lines.append(f"Input slots: {self.config.n_input_slots}")
        lines.append(f"GPU enabled: {self.evaluator.use_gpu}")

        search_space = self.get_search_space_size()
        lines.append("\nSearch space size per layer:")
        for layer, size in search_space.items():
            lines.append(f"  Layer {layer}: {size:,} expressions")

        if self._base_exprs:
            lines.append(f"\nBase expressions: {self._base_exprs}")

        return "\n".join(lines)
