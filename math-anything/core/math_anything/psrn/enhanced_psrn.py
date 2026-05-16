"""Enhanced PSRN - 增强型并行符号回归网络.

核心改进：
1. 分层质量剪枝：每层只保留高质量候选
2. 自适应常量优化：使用梯度下降优化常量
3. 渐进式层数构建：从浅到深逐步增加复杂度
4. 多目标选择：维护 Pareto 前沿（准确性 vs 复杂度）
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pruning_strategies import LayerPruner, PruningConfig, PruningStrategy
from .psrn_network import PSRN, PSRNConfig
from .symbol_layer import SymbolConfig, SymbolLayer


@dataclass
class EnhancedPSRNConfig(PSRNConfig):
    """增强型 PSRN 配置."""

    # 分层剪枝配置
    pruning_strategy: PruningStrategy = PruningStrategy.PROGRESSIVE
    pruning_schedule: List[float] = None  # 每层的保留比例

    # 自适应常量优化
    use_adaptive_constants: bool = True
    constant_learning_rate: float = 0.1
    constant_iterations: int = 50

    # 渐进式构建
    progressive_depth: bool = True
    max_complexity: int = 10

    def __post_init__(self):
        if self.pruning_schedule is None:
            # 默认渐进收紧：0.5, 0.3, 0.2, 0.1, ...
            self.pruning_schedule = [
                max(0.05, 0.5 * (0.6**i)) for i in range(self.n_layers)
            ]


class AdaptiveConstantOptimizer:
    """自适应常量优化器 - 使用梯度下降优化表达式中的常量."""

    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 50):
        self.lr = learning_rate
        self.max_iter = max_iterations

    def optimize(
        self,
        expr_template: str,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
    ) -> Tuple[str, float]:
        """优化表达式中的常量.

        Args:
            expr_template: 包含 {c0}, {c1} 等占位符的表达式
            X: 输入数据
            y: 目标值
            variable_names: 变量名

        Returns:
            optimized_expr: 优化后的表达式
            best_mse: 最优 MSE
        """
        import re

        # 找出所有常量占位符
        placeholders = re.findall(r"\{c\d+\}", expr_template)
        if not placeholders:
            # 没有可优化的常量
            return expr_template, self._evaluate_expr(
                expr_template, X, y, variable_names
            )

        # 初始化常量值
        constants = {p: 1.0 for p in placeholders}
        best_mse = float("inf")
        best_constants = constants.copy()

        # 梯度下降优化
        for iteration in range(self.max_iter):
            # 当前表达式
            current_expr = expr_template
            for p, v in constants.items():
                current_expr = current_expr.replace(p, f"{v:.6f}")

            # 评估当前 MSE
            try:
                y_pred = self._evaluate_expr(current_expr, X, y, variable_names)
                mse = np.mean((y_pred - y) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    best_constants = constants.copy()

                # 数值梯度
                grads = {}
                for p in placeholders:
                    # 正向扰动
                    constants_plus = constants.copy()
                    constants_plus[p] += 1e-4
                    expr_plus = expr_template
                    for pp, vv in constants_plus.items():
                        expr_plus = expr_plus.replace(pp, f"{vv:.6f}")
                    try:
                        y_plus = self._evaluate_expr(expr_plus, X, y, variable_names)
                        mse_plus = np.mean((y_plus - y) ** 2)
                        grads[p] = (mse_plus - mse) / 1e-4
                    except:
                        grads[p] = 0

                # 更新常量
                for p in placeholders:
                    constants[p] -= self.lr * grads[p]

            except Exception as e:
                break

        # 构建最终表达式
        final_expr = expr_template
        for p, v in best_constants.items():
            final_expr = final_expr.replace(p, f"{v:.6f}")

        return final_expr, best_mse

    def _evaluate_expr(
        self, expr: str, X: np.ndarray, y: np.ndarray, variable_names: List[str]
    ) -> np.ndarray:
        """评估表达式."""
        # 构建局部变量
        local_vars = {
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "exp": lambda x: np.exp(np.clip(x, -700, 700)),
            "log": lambda x: np.log(np.abs(x) + 1e-10),
            "eml": lambda x, y: np.exp(np.clip(x, -700, 700))
            - np.log(np.abs(y) + 1e-10),
        }

        for i, name in enumerate(variable_names):
            local_vars[name] = X[:, i]

        return eval(expr, {"__builtins__": {}}, local_vars)


class EnhancedPSRN(PSRN):
    """增强型 PSRN - 带分层剪枝和自适应常量优化."""

    def __init__(self, config: Optional[EnhancedPSRNConfig] = None):
        self.enhanced_config = config or EnhancedPSRNConfig()
        # 调用父类初始化
        super().__init__(self.enhanced_config)

        # 初始化剪枝器
        self.pruner = LayerPruner(
            PruningConfig(
                strategy=self.enhanced_config.pruning_strategy,
                initial_keep_ratio=0.5,
                min_keep_ratio=0.05,
            )
        )

        # 初始化常量优化器
        if self.enhanced_config.use_adaptive_constants:
            self.constant_optimizer = AdaptiveConstantOptimizer(
                learning_rate=self.enhanced_config.constant_learning_rate,
                max_iterations=self.enhanced_config.constant_iterations,
            )
        else:
            self.constant_optimizer = None

        # 存储每层最优候选
        self.layer_best: List[List[Tuple[str, float]]] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None,
        token_exprs: Optional[List[str]] = None,
        token_values: Optional[np.ndarray] = None,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """运行增强型 PSRN."""
        # 构建基础表达式
        base_exprs, base_values = self._build_base_expressions(
            X, variable_names, token_exprs, token_values
        )

        self._base_exprs = base_exprs
        self._layer_outputs = []
        self._layer_values = []
        self._offset_tensors = []
        self.layer_best = []

        # 逐层前向传播 + 剪枝
        current_exprs = base_exprs
        current_values = base_values

        for layer_idx, layer in enumerate(self.layers):
            # 前向传播
            new_exprs, new_values, offsets = layer.forward(
                current_exprs, current_values, layer_idx
            )

            # 应用剪枝
            keep_ratio = self.enhanced_config.pruning_schedule[layer_idx]
            pruned_exprs, pruned_values, prune_stats = self._prune_layer(
                new_exprs, new_values, y, keep_ratio
            )

            print(
                f"  Layer {layer_idx}: {len(new_exprs)} → {len(pruned_exprs)} "
                f"({prune_stats['prune_ratio']:.1%} pruned)"
            )

            # 存储
            self._layer_outputs.append(pruned_exprs)
            self._layer_values.append(pruned_values)
            self._offset_tensors.append(offsets)

            # 记录该层最优
            mses = np.mean((pruned_values - y.reshape(-1, 1)) ** 2, axis=0)
            best_indices = np.argsort(mses)[:10]
            self.layer_best.append([(pruned_exprs[i], mses[i]) for i in best_indices])

            # 更新当前状态
            current_exprs = pruned_exprs
            current_values = pruned_values

        # 从所有层中选择最优（不只是最后一层）
        best_expr, best_mse = self._select_best_from_all_layers(X, y, variable_names)

        # 自适应常量优化
        if self.constant_optimizer and "{" in best_expr:
            best_expr, best_mse = self.constant_optimizer.optimize(
                best_expr, X, y, variable_names or ["x"]
            )

        # 构建 top-k
        top_k = self._build_top_k()

        return best_expr, best_mse, top_k

    def _prune_layer(
        self,
        expressions: List[str],
        values: np.ndarray,
        y: np.ndarray,
        keep_ratio: float,
    ) -> Tuple[List[str], np.ndarray, Dict]:
        """对一层进行剪枝."""
        # 计算分数
        mses = np.mean((values - y.reshape(-1, 1)) ** 2, axis=0)

        # 保留比例
        n_keep = max(10, int(len(expressions) * keep_ratio))

        # 按 MSE 排序
        sorted_indices = np.argsort(mses)
        keep_indices = sorted_indices[:n_keep]

        pruned_exprs = [expressions[i] for i in keep_indices]
        pruned_values = values[:, keep_indices]

        stats = {
            "before": len(expressions),
            "after": len(pruned_exprs),
            "prune_ratio": 1 - len(pruned_exprs) / len(expressions),
            "best_mse": float(mses[sorted_indices[0]]),
        }

        return pruned_exprs, pruned_values, stats

    def _select_best_from_all_layers(
        self, X: np.ndarray, y: np.ndarray, variable_names: Optional[List[str]]
    ) -> Tuple[str, float]:
        """从所有层中选择最优表达式."""
        all_candidates = []

        for layer_idx, (exprs, values) in enumerate(
            zip(self._layer_outputs, self._layer_values)
        ):
            mses = np.mean((values - y.reshape(-1, 1)) ** 2, axis=0)

            for i, (expr, mse) in enumerate(zip(exprs, mses)):
                # 复杂度惩罚
                complexity = self._estimate_complexity(expr)
                score = mse + 0.01 * complexity  # 综合分数

                all_candidates.append(
                    {
                        "expr": expr,
                        "mse": mse,
                        "complexity": complexity,
                        "score": score,
                        "layer": layer_idx,
                    }
                )

        # 选择最优
        best = min(all_candidates, key=lambda x: x["score"])
        return best["expr"], best["mse"]

    def _estimate_complexity(self, expr: str) -> int:
        """估计表达式复杂度."""
        complexity = 0
        ops = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "eml"]
        for op in ops:
            complexity += expr.count(op)
        return max(complexity, 1)

    def _build_top_k(self) -> List[Tuple[str, float]]:
        """构建 top-k 候选列表."""
        all_candidates = []

        for layer_idx, best_list in enumerate(self.layer_best):
            for expr, mse in best_list:
                all_candidates.append((expr, mse))

        # 去重并排序
        seen = set()
        unique = []
        for expr, mse in sorted(all_candidates, key=lambda x: x[1]):
            if expr not in seen:
                seen.add(expr)
                unique.append((expr, mse))

        return unique[:10]
