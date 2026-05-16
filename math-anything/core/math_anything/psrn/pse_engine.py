"""PSE Engine - 并行符号枚举顶层引擎.

协调 PSRN 和 TokenGenerator 的迭代循环，实现论文中的完整 PSE 框架。
与现有代码的集成点：替代 ImprovedSymbolicRegression 作为高级发现引擎。
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .psrn_network import PSRN, PSRNConfig
from .token_generator import (
    FastTokenGenerator,
    GPTokenGenerator,
    MCTSTokenGenerator,
    RandomTokenGenerator,
    TokenGenerator,
)
from .attention_token_generator import (
    AttentionTokenGenerator,
    DiversityAwareAttentionGenerator,
)
from .adaptive_attention import (
    AdaptiveAttentionGenerator,
    HybridAttentionGenerator,
)
from .csa_hca_attention import CSAHCAAttentionGenerator


@dataclass
class PSEConfig:
    """PSE 引擎配置."""

    psrn_config: PSRNConfig = field(default_factory=PSRNConfig)
    # Token generator 类型：
    # - "fast": 快速启发式
    # - "random": 随机采样
    # - "gp": 遗传编程
    # - "mcts": 蒙特卡洛树搜索
    # - "attention": 基础注意力机制
    # - "diversity": 多样性感知注意力
    # - "adaptive": 自适应注意力（元学习权重）
    # - "hybrid": 混合策略
    # - "csa_hca": CSA+HCA分层注意力（推荐，借鉴DeepSeek-V4）
    token_generator_type: str = "fast"
    max_iterations: int = 10             # 最大迭代次数
    n_top_expressions: int = 5           # 保留的 top-k 表达式数
    reward_discount: float = 0.99        # 复杂度惩罚折扣因子
    early_stop_mse: float = 1e-12        # 早停 MSE 阈值
    no_improvement_limit: int = 3        # 无改进早停次数


class PSEEngine:
    """并行符号枚举引擎.

    这是 PSE 框架的顶层协调器，实现论文中的迭代循环：
    1. TokenGenerator 生成有希望的子表达式
    2. PSRN 并行评估海量候选表达式
    3. 选择最优表达式，计算奖励
    4. 反馈奖励给 TokenGenerator，更新 token 集合
    5. 更新 Pareto 前沿

    Example:
        >>> config = PSEConfig()
        >>> engine = PSEEngine(config)
        >>>
        >>> # 从数据中发现方程
        >>> x = np.linspace(0, 1, 100)
        >>> y = x**2 + np.sin(x)
        >>>
        >>> best_expr, pareto_front = engine.discover(x.reshape(-1, 1), y, ['x'])
        >>> print(f"Discovered: {best_expr}")
        >>> for expr, mse, complexity in pareto_front:
        ...     print(f"  {expr} (MSE={mse:.6f}, complexity={complexity})")
    """

    def __init__(self, config: Optional[PSEConfig] = None):
        self.config = config or PSEConfig()
        self.psrn = PSRN(self.config.psrn_config)

        # 初始化 token generator
        tg_type = self.config.token_generator_type
        if tg_type == "fast":
            self.token_generator: TokenGenerator = FastTokenGenerator()
        elif tg_type == "random":
            self.token_generator = RandomTokenGenerator()
        elif tg_type == "gp":
            self.token_generator = GPTokenGenerator()
        elif tg_type == "mcts":
            self.token_generator = MCTSTokenGenerator()
        elif tg_type == "attention":
            self.token_generator = AttentionTokenGenerator()
        elif tg_type == "diversity":
            self.token_generator = DiversityAwareAttentionGenerator()
        elif tg_type == "adaptive":
            self.token_generator = AdaptiveAttentionGenerator(
                n_tokens=20,
                exploration_ratio=0.25,
                learning_rate=0.1,
                use_eml_priority=True,
            )
        elif tg_type == "hybrid":
            self.token_generator = HybridAttentionGenerator(
                n_tokens=20,
                exploration_ratio=0.2,
            )
        elif tg_type == "csa_hca":
            self.token_generator = CSAHCAAttentionGenerator(
                n_tokens=20,
                csa_ratio=0.6,
                hca_ratio=0.4,
                adaptive_schedule=True,
            )
        else:
            raise ValueError(f"Unknown token generator: {tg_type}")

        # Pareto 前沿: (expression, mse, complexity, reward)
        self.pareto_front: List[Tuple[str, float, int, float]] = []
        self.reward_history: Dict[str, float] = {}

    def discover(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Tuple[str, List[Tuple[str, float, int, float]]]:
        """从数据中发现数学表达式.

        Args:
            X: 输入特征 (n_samples, n_features)
            y: 目标值 (n_samples,)
            variable_names: 变量名
            verbose: 是否打印进度

        Returns:
            best_expr: 最优表达式
            pareto_front: Pareto 前沿（表达式, MSE, 复杂度, 奖励）
        """
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]

        current_tokens: Optional[List[str]] = None
        token_values: Optional[np.ndarray] = None

        best_mse = float("inf")
        best_expr = ""
        no_improvement_count = 0

        for iteration in range(self.config.max_iterations):
            if verbose:
                print(f"\n{'='*50}")
                print(f"PSE Iteration {iteration + 1}/{self.config.max_iterations}")
                print(f"{'='*50}")

            # Step 1: Token Generator 生成/更新 token
            if iteration > 0:
                new_tokens, new_values = self.token_generator.generate(
                    X, y, variable_names,
                    current_tokens=current_tokens,
                    reward_history=self.reward_history,
                )
                # 合并 token
                if current_tokens is not None:
                    current_tokens = list(set(current_tokens + new_tokens))
                    # 重新评估合并后的 token
                    token_values = self._evaluate_tokens(current_tokens, X, variable_names)
                else:
                    current_tokens = new_tokens
                    token_values = new_values

            # Step 2: PSRN 并行评估
            best_expr_iter, best_mse_iter, top_k = self.psrn.fit(
                X, y,
                variable_names=variable_names,
                token_exprs=current_tokens,
                token_values=token_values,
            )

            if verbose:
                print(f"PSRN found: {best_expr_iter}")
                print(f"MSE: {best_mse_iter:.6e}")
                print(f"Search space: {self.psrn.get_search_space_size()}")

            # Step 3: 计算复杂度和奖励
            complexity = self._compute_complexity(best_expr_iter)
            reward = self._compute_reward(best_mse_iter, complexity)

            # Step 4: 更新 Pareto 前沿
            self._update_pareto_front(best_expr_iter, best_mse_iter, complexity, reward)

            # Step 5: 记录奖励历史
            self.reward_history[best_expr_iter] = reward

            # 检查是否改进
            if best_mse_iter < best_mse:
                best_mse = best_mse_iter
                best_expr = best_expr_iter
                no_improvement_count = 0

                if verbose:
                    print(f"New best! Reward: {reward:.4f}")
            else:
                no_improvement_count += 1

            # 早停检查
            if best_mse < self.config.early_stop_mse:
                if verbose:
                    print(f"Early stop: MSE {best_mse:.2e} < threshold")
                break

            if no_improvement_count >= self.config.no_improvement_limit:
                if verbose:
                    print(f"Early stop: No improvement for {no_improvement_count} iterations")
                break

        # 返回 Pareto 前沿中的最优表达式
        if self.pareto_front:
            # 按奖励排序
            sorted_front = sorted(self.pareto_front, key=lambda x: x[3], reverse=True)
            best_expr = sorted_front[0][0]

        return best_expr, self.pareto_front

    def _evaluate_tokens(
        self,
        tokens: List[str],
        X: np.ndarray,
        variable_names: List[str],
    ) -> np.ndarray:
        """评估 token 表达式的数值."""
        n_samples = X.shape[0]
        n_tokens = len(tokens)
        values = np.zeros((n_samples, n_tokens))

        var_dict = {name: X[:, i] for i, name in enumerate(variable_names)}

        for i, expr in enumerate(tokens):
            try:
                values[:, i] = self._safe_eval(expr, var_dict)
            except Exception:
                values[:, i] = 0.0

        return values

    def _safe_eval(self, expr: str, var_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """安全评估表达式."""
        safe_dict = {
            "sin": np.sin,
            "cos": np.cos,
            "exp": lambda x: np.exp(np.clip(x, -700, 700)),
            "log": lambda x: np.log(np.abs(x) + 1e-10),
            "sqrt": lambda x: np.sqrt(np.abs(x)),
            "abs": np.abs,
        }
        safe_dict.update(var_dict)

        try:
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            if isinstance(result, (int, float)):
                return np.full(list(var_dict.values())[0].shape, float(result))
            return np.asarray(result)
        except Exception:
            return np.zeros(list(var_dict.values())[0].shape)

    def _compute_complexity(self, expr: str) -> int:
        """计算表达式复杂度.

        基于运算符数量、嵌套深度等。
        """
        # 简化实现：基于字符串特征估算
        complexity = 0

        # 运算符计数
        ops = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt"]
        for op in ops:
            complexity += expr.count(op)

        # 嵌套深度
        depth = 0
        max_depth = 0
        for c in expr:
            if c == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif c == ")":
                depth -= 1

        complexity += max_depth

        return max(complexity, 1)

    def _compute_reward(self, mse: float, complexity: int) -> float:
        """计算表达式奖励.

        论文公式: r = discount^complexity / (1 + sqrt(MSE))
        """
        import math

        numerator = self.config.reward_discount ** complexity
        denominator = 1.0 + math.sqrt(mse)
        return numerator / denominator

    def _update_pareto_front(
        self,
        expr: str,
        mse: float,
        complexity: int,
        reward: float,
    ):
        """更新 Pareto 前沿.

        保留非支配解：没有其它解同时在 MSE 和复杂度上都更优。
        """
        new_entry = (expr, mse, complexity, reward)

        # 检查是否被现有解支配
        for existing in self.pareto_front:
            # 如果 existing 在 MSE 和复杂度上都更优或相等
            if existing[1] <= mse and existing[2] <= complexity:
                # 至少有一个严格更优
                if existing[1] < mse or existing[2] < complexity:
                    return  # 被支配，不加入

        # 移除被新解支配的解
        self.pareto_front = [
            e for e in self.pareto_front
            if not (mse <= e[1] and complexity <= e[2] and (mse < e[1] or complexity < e[2]))
        ]

        # 添加新解
        self.pareto_front.append(new_entry)

    def get_pareto_summary(self) -> str:
        """获取 Pareto 前沿摘要."""
        lines = ["Pareto Front:", "=" * 60]
        lines.append(f"{'Expression':<30} {'MSE':<12} {'Complexity':<10} {'Reward':<10}")
        lines.append("-" * 60)

        sorted_front = sorted(self.pareto_front, key=lambda x: x[3], reverse=True)
        for expr, mse, comp, reward in sorted_front[:10]:
            expr_short = expr[:28] + ".." if len(expr) > 30 else expr
            lines.append(f"{expr_short:<30} {mse:<12.4e} {comp:<10} {reward:<10.4f}")

        return "\n".join(lines)
