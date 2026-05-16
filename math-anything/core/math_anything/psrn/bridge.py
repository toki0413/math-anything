"""Bridge - PSRN 与现有 EML/GP 模块的桥接层.

提供向后兼容的 API，让现有代码可以无缝切换到 PSRN 引擎。
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from ..eml_v2 import ExprBuilder, ImprovedSymbolicRegression, Node, NodeType
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
        best_expr, pareto_front = self._engine.discover(
            X, y, variable_names, verbose=False
        )

        self._best_expr = best_expr
        self.best_fitness_ = min(
            (entry[1] for entry in pareto_front), default=float("inf")
        )

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
        """Recursively parse PSRN expression string into a Node tree.

        Handles constants, variables, binary ops (+,-,*,/,**),
        unary functions (sin,cos,exp,log,sqrt,abs), and parentheses.
        """
        builder = ExprBuilder()
        expr = expr.strip()

        if not expr:
            return builder.const(0.0)

        node = self._try_parse(expr, builder)
        if node is not None:
            return node

        return builder.const(0.0)

    @staticmethod
    def _try_parse(expr: str, builder: ExprBuilder) -> Optional[Node]:
        """Attempt recursive descent parsing of an expression string."""
        expr = expr.strip()
        if not expr:
            return None

        try:
            val = float(expr)
            return builder.const(val)
        except ValueError:
            pass

        if expr.startswith("x") and len(expr) > 1 and expr[1:].isdigit():
            return builder.var(expr)

        if expr.startswith("(") and expr.endswith(")"):
            depth = 0
            all_wrapped = True
            for i, ch in enumerate(expr):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    all_wrapped = False
                    break
            if all_wrapped:
                inner = expr[1:-1]
                result = PSRNSymbolicRegression._try_parse(inner, builder)
                if result is not None:
                    return result

        for op, node_type in [("+", NodeType.ADD), ("-", NodeType.SUB)]:
            pos = PSRNSymbolicRegression._find_outermost_op(expr, op)
            if pos is not None and pos > 0:
                left = PSRNSymbolicRegression._try_parse(expr[:pos], builder)
                right = PSRNSymbolicRegression._try_parse(expr[pos + 1 :], builder)
                if left is not None and right is not None:
                    if node_type == NodeType.ADD:
                        return builder.add(left, right)
                    else:
                        return builder.sub(left, right)

        for op, node_type in [("*", NodeType.MUL), ("/", NodeType.DIV)]:
            pos = PSRNSymbolicRegression._find_outermost_op(expr, op)
            if pos is not None and pos > 0:
                left = PSRNSymbolicRegression._try_parse(expr[:pos], builder)
                right = PSRNSymbolicRegression._try_parse(expr[pos + 1 :], builder)
                if left is not None and right is not None:
                    if node_type == NodeType.MUL:
                        return builder.mul(left, right)
                    else:
                        return builder.div(left, right)

        if "**" in expr:
            pos = expr.rfind("**")
            if pos > 0:
                base = PSRNSymbolicRegression._try_parse(expr[:pos], builder)
                exp_part = PSRNSymbolicRegression._try_parse(expr[pos + 2 :], builder)
                if base is not None and exp_part is not None:
                    return builder.pow(base, exp_part)

        for func_name, is_unary in [
            ("sin", True),
            ("cos", True),
            ("exp", True),
            ("log", True),
            ("sqrt", True),
            ("abs", True),
        ]:
            if expr.startswith(func_name + "(") and expr.endswith(")"):
                inner = expr[len(func_name) + 1 : -1]
                child = PSRNSymbolicRegression._try_parse(inner, builder)
                if child is not None:
                    if func_name == "sin":
                        return builder.sin(child)
                    elif func_name == "cos":
                        return builder.cos(child)
                    elif func_name == "sqrt":
                        return builder.sqrt(child)
                    elif func_name == "abs":
                        return builder.abs(child)
                    elif func_name == "exp":
                        return builder.pow(builder.const(math.e), child)
                    elif func_name == "log":
                        return builder.div(builder.const(1.0), child)

        return None

    @staticmethod
    def _find_outermost_op(expr: str, op: str) -> Optional[int]:
        """Find the rightmost occurrence of op outside parentheses.

        Searches right-to-left so that a-b-c parses as a-(b-c) for
        subtraction and a+b+c as a+(b+c) for addition (left-associative
        when reading right-to-left gives leftmost split).
        """
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            ch = expr[i]
            if ch == ")":
                depth += 1
            elif ch == "(":
                depth -= 1
            elif ch == op and depth == 0:
                if op in ("+", "-"):
                    if i == 0:
                        continue
                    prev = expr[i - 1]
                    if prev in ("+", "-", "*", "/", "(", "e", "E"):
                        continue
                return i
        return None

    def _eval_expr(self, expr: str, var_dict: dict) -> float:
        """评估表达式字符串."""
        safe_dict = {
            "sin": __import__("math").sin,
            "cos": __import__("math").cos,
            "exp": lambda x: __import__("math").exp(max(min(x, 700), -700)),
            "log": lambda x: __import__("math").log(abs(x) + 1e-10),
            "sqrt": lambda x: __import__("math").sqrt(abs(x)),
            "abs": abs,
            "pi": __import__("math").pi,
            "e": __import__("math").e,
        }
        safe_dict.update(var_dict)

        try:
            return eval(expr, {"__builtins__": {}}, safe_dict)
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
