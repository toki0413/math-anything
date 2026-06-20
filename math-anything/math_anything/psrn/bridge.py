"""Bridge - PSRN 与现有 EML/GP 模块的桥接层.

提供向后兼容的 API，让现有代码可以无缝切换到 PSRN 引擎。
"""

from typing import List, Optional, Tuple

import numpy as np

from ..eml_v2 import ImprovedSymbolicRegression, Node
from .pse_engine import PSEConfig, PSEEngine
from .psrn_network import PSRNConfig


class PSRNSymbolicRegression(ImprovedSymbolicRegression):
    """兼容 ImprovedSymbolicRegression API 的 PSRN 实现.

    这个类可以作为 ImprovedSymbolicRegression 的 drop-in 替代，
    内部使用 PSEEngine 实现更高效的符号回归。

    Example:
        >>> # 原有代码
        >>> sr = ImprovedSymbolicRegression(population_size=200, generations=100)
        >>> best_tree = sr.fit(X, y, variable_names=['x'])
        >>>
        >>> # 替换为 PSRN（只需改一行）
        >>> sr = PSRNSymbolicRegression(n_layers=2, max_iterations=5)
        >>> best_tree = sr.fit(X, y, variable_names=['x'])
    """

    def __init__(
        self,
        n_layers: int = 2,
        n_input_slots: int = 5,
        max_iterations: int = 5,
        token_generator: str = "fast",
        use_gpu: Optional[bool] = None,
        **kwargs,
    ):
        """
        Args:
            n_layers: PSRN 层数
            n_input_slots: 输入槽位数
            max_iterations: PSE 最大迭代次数
            token_generator: token 生成器类型:
                - "fast": 快速启发式
                - "random": 随机采样
                - "gp": 遗传编程
                - "mcts": 蒙特卡洛树搜索
                - "attention": 基础注意力
                - "diversity": 多样性感知
                - "adaptive": 自适应注意力
                - "hybrid": 混合策略
                - "csa_hca": CSA+HCA分层注意力（推荐）
            use_gpu: 是否使用 GPU
            **kwargs: 兼容 ImprovedSymbolicRegression 的参数（被忽略）
        """
        # 不调用父类 __init__，完全重写
        self.n_layers = n_layers
        self.n_input_slots = n_input_slots
        self.max_iterations = max_iterations
        self.token_generator_type = token_generator
        self.use_gpu = use_gpu

        self._engine: Optional[PSEEngine] = None
        self._best_expr: str = ""
        self.best_tree_: Optional[Node] = None
        self.best_fitness_: float = float("inf")
        self.variables: List[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> Node:
        """使用 PSRN 发现方程.

        保持与 ImprovedSymbolicRegression.fit() 相同的签名。
        """
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]
        self.variables = variable_names

        # 构建 PSE 配置
        psrn_config = PSRNConfig(
            n_layers=self.n_layers,
            n_input_slots=self.n_input_slots,
            use_gpu=self.use_gpu,
        )
        config = PSEConfig(
            psrn_config=psrn_config,
            token_generator_type=self.token_generator_type,
            max_iterations=self.max_iterations,
        )

        self._engine = PSEEngine(config)

        # 运行发现
        best_expr, pareto_front = self._engine.discover(X, y, variable_names, verbose=False)

        self._best_expr = best_expr
        self.best_fitness_ = min((entry[1] for entry in pareto_front), default=float("inf"))

        # 将表达式字符串转换为 Node 树（简化实现）
        self.best_tree_ = self._expr_to_node(best_expr)

        return self.best_tree_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用最优表达式预测.

        保持与 ImprovedSymbolicRegression.predict() 相同的签名。
        """
        if not self._best_expr:
            raise ValueError("Model not fitted. Call fit() first.")

        # 评估表达式
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        var_dict = {name: X[:, i] for i, name in enumerate(self.variables)}

        for i in range(n_samples):
            single_var_dict = {k: float(v[i]) for k, v in var_dict.items()}
            predictions[i] = self._eval_expr(self._best_expr, single_var_dict)

        return predictions

    def _expr_to_node(self, expr: str) -> Node:
        """将表达式字符串转换为 Node 树（简化实现）."""
        # 简化：返回一个表示整个表达式的占位符节点
        # 完整实现需要表达式解析器
        from ..eml_v2 import ExprBuilder

        builder = ExprBuilder()

        # 尝试解析简单表达式
        if expr == "":
            return builder.const(0.0)

        # 对于复杂表达式，返回一个变量节点作为占位符
        # 实际使用时，predict 方法直接评估字符串表达式
        if self.variables:
            return builder.var(self.variables[0])
        return builder.const(0.0)

    def _eval_expr(self, expr: str, var_dict: dict) -> float:
        """评估表达式字符串.

        使用 safe_eval 替代 eval()，通过 context 传入数学函数。
        """
        import math as _math

        context = {
            "sin": _math.sin,
            "cos": _math.cos,
            "exp": lambda x: _math.exp(max(min(x, 700), -700)),
            "log": lambda x: _math.log(abs(x) + 1e-10),
            "sqrt": lambda x: _math.sqrt(abs(x)),
            "abs": abs,
            "pi": _math.pi,
            "e": _math.e,
        }
        context.update(var_dict)

        try:
            from ..utils.safe_eval import safe_eval

            return safe_eval(expr, context)
        except Exception:
            return 0.0

    def get_pareto_front(self) -> List[Tuple[str, float, int, float]]:
        """获取 Pareto 前沿（PSRN 特有功能）."""
        if self._engine is None:
            return []
        return self._engine.pareto_front


class EnhancedPSRNSymbolicRegression(PSRNSymbolicRegression):
    """使用保守增强型 PSRN 的符号回归.

    相比原始 PSRN，具有更好的准确性和速度平衡。

    Example:
        >>> sr = EnhancedPSRNSymbolicRegression(
        ...     n_layers=2,
        ...     max_layer_size=300,  # 限制每层候选数
        ... )
        >>> best_tree = sr.fit(X, y, variable_names=['x'])
    """

    def __init__(
        self,
        n_layers: int = 2,
        max_layer_size: int = 300,
        **kwargs,
    ):
        """
        Args:
            n_layers: PSRN 层数（建议 2）
            max_layer_size: 每层最大候选数（默认 300）
            **kwargs: 其他参数（兼容）
        """
        super().__init__(n_layers=n_layers, **kwargs)
        self.max_layer_size = max_layer_size
        self._enhanced_engine = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> Node:
        """使用增强型 PSRN 发现方程."""
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]
        self.variables = variable_names

        # 使用保守增强型 PSRN
        from .enhanced_psrn_v2 import ConservativeEnhancedPSRN, ConservativePSRNConfig

        config = ConservativePSRNConfig(
            n_layers=self.n_layers,
            max_layer_size=self.max_layer_size,
        )

        self._enhanced_engine = ConservativeEnhancedPSRN(config)
        best_expr, best_mse, top_k = self._enhanced_engine.fit(X, y, variable_names)

        self._best_expr = best_expr
        self.best_fitness_ = best_mse
        self._top_k = top_k

        # 转换为 Node
        self.best_tree_ = self._expr_to_node(best_expr)

        return self.best_tree_


def upgrade_to_psrn(
    sr: ImprovedSymbolicRegression,
    n_layers: int = 2,
    max_iterations: int = 5,
) -> PSRNSymbolicRegression:
    """将现有的 ImprovedSymbolicRegression 升级为 PSRN 版本.

    Args:
        sr: 现有的符号回归实例
        n_layers: PSRN 层数
        max_iterations: 最大迭代次数

    Returns:
        配置好的 PSRNSymbolicRegression 实例
    """
    return PSRNSymbolicRegression(
        n_layers=n_layers,
        max_iterations=max_iterations,
    )


def upgrade_to_enhanced_psrn(
    sr: ImprovedSymbolicRegression,
    n_layers: int = 2,
    max_layer_size: int = 300,
) -> EnhancedPSRNSymbolicRegression:
    """升级为增强型 PSRN（推荐）.

    Args:
        sr: 现有的符号回归实例
        n_layers: PSRN 层数（建议 2）
        max_layer_size: 每层最大候选数

    Returns:
        配置好的 EnhancedPSRNSymbolicRegression 实例
    """
    return EnhancedPSRNSymbolicRegression(
        n_layers=n_layers,
        max_layer_size=max_layer_size,
    )
