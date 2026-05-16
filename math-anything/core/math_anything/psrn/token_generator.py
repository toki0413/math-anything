"""Token Generator - 生成有希望的子表达式作为 PSRN 的输入 token.

实现多种策略:
1. Random: 直接生成随机表达式组合（最快）
2. Fast: 基于数据特征生成候选（快速且智能）
3. GP (遗传编程): 使用进化算法生成复杂子表达式（慢但可能发现复杂模式）
4. MCTS (蒙特卡洛树搜索): 使用树搜索探索表达式空间

推荐使用 Random 或 Fast，GP 太慢会抵消 PSRN 的加速优势。
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..eml_v2 import ExprBuilder, ImprovedSymbolicRegression, Node, NodeType


class TokenGenerator(ABC):
    """Token 生成器基类."""

    @abstractmethod
    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        current_tokens: Optional[List[str]] = None,
        reward_history: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """生成新的 token 集合.

        Args:
            X: 输入数据
            y: 目标值
            variable_names: 变量名
            current_tokens: 当前已有的 token
            reward_history: 历史奖励记录

        Returns:
            token_exprs: token 表达式字符串列表
            token_values: token 数值 (n_samples, n_tokens)
        """
        pass


class RandomTokenGenerator(TokenGenerator):
    """随机 Token 生成器 - 最快的选择.

    直接生成随机的变量组合和变换，不使用任何搜索。
    适合 PSRN 的并行评估优势。
    """

    def __init__(self, n_tokens: int = 10, seed: Optional[int] = None):
        self.n_tokens = n_tokens
        self.rng = random.Random(seed)

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        current_tokens: Optional[List[str]] = None,
        reward_history: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """快速生成随机 token."""
        n_samples = X.shape[0]
        n_vars = len(variable_names)

        # 预定义的简单变换模板
        templates = []

        # 1. 原始变量
        for name in variable_names:
            templates.append(name)

        # 2. 单变量变换
        for name in variable_names:
            templates.extend(
                [
                    f"sin({name})",
                    f"cos({name})",
                    f"exp({name})",
                    f"log(abs({name})+1)",
                    f"{name}*{name}",
                    f"sqrt(abs({name}))",
                ]
            )

        # 3. 两变量组合
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                templates.extend(
                    [
                        f"({variable_names[i]}+{variable_names[j]})",
                        f"({variable_names[i]}*{variable_names[j]})",
                        f"({variable_names[i]}-{variable_names[j]})",
                    ]
                )

        # 随机选择
        selected = self.rng.sample(templates, min(self.n_tokens, len(templates)))

        # 评估
        token_values = self._evaluate_expressions(selected, X, variable_names)

        return selected, token_values

    def _evaluate_expressions(
        self, expressions: List[str], X: np.ndarray, variable_names: List[str]
    ) -> np.ndarray:
        """向量化评估表达式."""
        from .compiled_evaluator import CompiledEvaluator

        evaluator = CompiledEvaluator()
        return evaluator.evaluate_batch_vec(expressions, X, variable_names)


class FastTokenGenerator(TokenGenerator):
    """快速 Token 生成器 - 基于数据特征.

    分析数据特征（相关性、范围等）生成有希望的候选表达式。
    比 Random 稍慢但更智能。
    """

    def __init__(self, n_tokens: int = 10):
        self.n_tokens = n_tokens

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        current_tokens: Optional[List[str]] = None,
        reward_history: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """基于数据特征生成 token."""
        n_samples = X.shape[0]
        n_vars = len(variable_names)

        candidates = []

        # 1. 分析每个变量与目标的相关性
        correlations = []
        for i, name in enumerate(variable_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append((name, i, abs(corr) if not np.isnan(corr) else 0))

        correlations.sort(key=lambda x: x[2], reverse=True)

        # 2. 对高相关变量生成更多变换
        for name, idx, corr in correlations[: min(3, n_vars)]:
            x_vals = X[:, idx]

            candidates.append((name, x_vals))

            # 根据数据范围选择变换
            x_range = np.max(x_vals) - np.min(x_vals)
            x_min = np.min(x_vals)

            if x_range > 0:
                # 归一化后的变换
                candidates.append((f"sin({name})", np.sin(x_vals)))
                candidates.append((f"cos({name})", np.cos(x_vals)))

            if np.all(x_vals > 0):
                candidates.append((f"log({name})", np.log(x_vals + 1e-10)))

            if np.all(np.abs(x_vals) < 10):
                candidates.append((f"exp({name})", np.exp(np.clip(x_vals, -700, 700))))

            candidates.append((f"{name}^2", x_vals**2))

        # 3. 两变量组合（优先高相关变量）
        top_vars = [c[0] for c in correlations[: min(3, n_vars)]]
        for i, name_i in enumerate(top_vars):
            for j, name_j in enumerate(top_vars):
                if i >= j:
                    continue
                idx_i = variable_names.index(name_i)
                idx_j = variable_names.index(name_j)

                candidates.append((f"({name_i}+{name_j})", X[:, idx_i] + X[:, idx_j]))
                candidates.append((f"({name_i}*{name_j})", X[:, idx_i] * X[:, idx_j]))

        # 4. 评估并选择最佳
        scored = []
        for expr, values in candidates:
            if np.std(values) > 1e-10:
                corr = np.corrcoef(values, y)[0, 1]
                score = abs(corr) if not np.isnan(corr) else 0
            else:
                score = 0
            scored.append((expr, values, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[: self.n_tokens]

        if not top:
            # 回退到原始变量
            top = [(variable_names[0], X[:, 0], 0)]

        token_exprs = [t[0] for t in top]
        token_values = np.column_stack([t[1] for t in top])

        return token_exprs, token_values


class GPTokenGenerator(TokenGenerator):
    """基于遗传编程的 Token 生成器.

    使用改进的遗传编程进化出复杂的子表达式，作为 PSRN 的输入 token。
    这是论文中默认的 token generator 实现。
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 20,
        max_depth: int = 4,
        n_tokens: int = 5,
    ):
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.n_tokens = n_tokens
        self.builder = ExprBuilder()

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        current_tokens: Optional[List[str]] = None,
        reward_history: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """使用 GP 生成优质 token."""
        # 使用现有的 ImprovedSymbolicRegression
        sr = ImprovedSymbolicRegression(
            population_size=self.population_size,
            max_depth=self.max_depth,
            generations=self.generations,
            use_standard_ops=True,
        )

        # 运行符号回归
        best_tree = sr.fit(X, y, variable_names)

        # 从种群中提取优质子树作为 token
        tokens = self._extract_subtrees(sr, X, variable_names)

        # 如果 token 不够，补充随机子树
        while len(tokens) < self.n_tokens:
            random_tree = sr._random_tree_grow(self.max_depth // 2)
            tokens.append(random_tree)

        # 限制 token 数量
        tokens = tokens[: self.n_tokens]

        # 转换为字符串和数值
        token_exprs = [tree.to_standard_form() for tree in tokens]
        token_values = self._evaluate_tokens(tokens, X, variable_names)

        return token_exprs, token_values

    def _extract_subtrees(
        self, sr: ImprovedSymbolicRegression, X: np.ndarray, variable_names: List[str]
    ) -> List[Node]:
        """从 GP 种群中提取优质子树."""
        # 简化实现：返回最佳树的子树
        if sr.best_tree_ is None:
            return []

        subtrees = []
        self._collect_subtrees(sr.best_tree_, subtrees)

        # 去重并选择有代表性的子树
        unique_subtrees = []
        seen_depths = set()

        for tree in subtrees:
            d = tree.depth()
            if d not in seen_depths and 1 <= d <= self.max_depth - 1:
                seen_depths.add(d)
                unique_subtrees.append(tree)

        return unique_subtrees

    def _collect_subtrees(self, node: Node, subtrees: List[Node]):
        """递归收集所有子树."""
        if node is None:
            return

        if node.node_type not in (NodeType.CONST, NodeType.VAR):
            subtrees.append(node)

        if node.left:
            self._collect_subtrees(node.left, subtrees)
        if node.right:
            self._collect_subtrees(node.right, subtrees)

    def _evaluate_tokens(
        self, tokens: List[Node], X: np.ndarray, variable_names: List[str]
    ) -> np.ndarray:
        """评估所有 token 的数值."""
        n_samples = X.shape[0]
        n_tokens = len(tokens)
        values = np.zeros((n_samples, n_tokens))

        for i, tree in enumerate(tokens):
            for j in range(n_samples):
                vars_dict = {
                    name: float(X[j, k]) for k, name in enumerate(variable_names)
                }
                try:
                    values[j, i] = tree.evaluate(vars_dict)
                except Exception:
                    values[j, i] = 0.0

        return values


class MCTSTokenGenerator(TokenGenerator):
    """基于蒙特卡洛树搜索的 Token 生成器.

    使用 MCTS 探索表达式空间，适合处理高维问题。
    """

    def __init__(
        self,
        n_iterations: int = 100,
        exploration_constant: float = 1.414,
        n_tokens: int = 5,
    ):
        self.n_iterations = n_iterations
        self.exploration_constant = exploration_constant
        self.n_tokens = n_tokens
        self.builder = ExprBuilder()

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        current_tokens: Optional[List[str]] = None,
        reward_history: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """使用 MCTS 生成优质 token."""
        # 简化实现：使用随机搜索 + 评估
        # 完整实现需要构建搜索树和 UCB 公式

        candidates = self._generate_candidates(X, variable_names)

        # 评估候选
        scores = []
        for expr, values in candidates:
            # 计算与目标的相关系数作为奖励
            if np.std(values) > 1e-10 and np.std(y) > 1e-10:
                corr = np.corrcoef(values, y)[0, 1]
                score = abs(corr) if not np.isnan(corr) else 0.0
            else:
                score = 0.0
            scores.append((expr, values, score))

        # 选择得分最高的 token
        scores.sort(key=lambda x: x[2], reverse=True)
        top_tokens = scores[: self.n_tokens]

        token_exprs = [t[0] for t in top_tokens]
        token_values = np.column_stack([t[1] for t in top_tokens])

        return token_exprs, token_values

    def _generate_candidates(
        self, X: np.ndarray, variable_names: List[str]
    ) -> List[Tuple[str, np.ndarray]]:
        """生成候选表达式."""
        n_samples = X.shape[0]
        candidates = []

        # 单变量变换
        for name in variable_names:
            idx = variable_names.index(name)
            x_vals = X[:, idx]

            candidates.append((name, x_vals))
            candidates.append((f"sin({name})", np.sin(x_vals)))
            candidates.append((f"cos({name})", np.cos(x_vals)))
            candidates.append((f"exp({name})", np.exp(np.clip(x_vals, -700, 700))))
            candidates.append((f"log(abs({name}))", np.log(np.abs(x_vals) + 1e-10)))
            candidates.append((f"{name}^2", x_vals**2))
            candidates.append((f"sqrt(abs({name}))", np.sqrt(np.abs(x_vals))))

        # 两变量组合
        for i, name_i in enumerate(variable_names):
            for j, name_j in enumerate(variable_names):
                if i >= j:
                    continue

                xi, xj = X[:, i], X[:, j]
                candidates.append((f"({name_i}+{name_j})", xi + xj))
                candidates.append((f"({name_i}*{name_j})", xi * xj))
                candidates.append((f"({name_i}-{name_j})", xi - xj))
                with np.errstate(divide="ignore", invalid="ignore"):
                    div = np.where(np.abs(xj) > 1e-10, xi / xj, 0.0)
                candidates.append((f"({name_i}/{name_j})", div))

        return candidates
