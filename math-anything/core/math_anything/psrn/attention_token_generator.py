"""Attention Token Generator - 基于注意力机制的 Token 生成器.

核心思想：为每个候选表达式计算"注意力分数"，只保留高分候选，
从而将搜索空间聚焦于更有希望的区域。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .token_generator import TokenGenerator


class AttentionTokenGenerator(TokenGenerator):
    """注意力 Token 生成器.

    通过计算每个候选表达式的"注意力分数"，筛选出最有希望的子集，
    大幅提升 PSRN 的搜索效率和准确性。

    注意力分数 = 相关性 * 结构合理性 * 多样性

    Example:
        >>> generator = AttentionTokenGenerator(n_tokens=20, attention_ratio=0.3)
        >>> # 只保留 top 30% 的高分候选，但确保至少有 20 个 token
    """

    def __init__(
        self,
        n_tokens: int = 20,
        attention_ratio: float = 0.3,
        correlation_weight: float = 0.4,
        complexity_weight: float = 0.3,
        diversity_weight: float = 0.2,
        constant_weight: float = 0.1,
    ):
        """
        Args:
            n_tokens: 最少保留的 token 数量
            attention_ratio: 注意力比例，只保留 top-k% 的候选
            correlation_weight: 相关性权重
            complexity_weight: 结构复杂度权重
            diversity_weight: 多样性权重
            constant_weight: 常数拟合权重
        """
        self.n_tokens = n_tokens
        self.attention_ratio = attention_ratio
        self.weights = {
            "correlation": correlation_weight,
            "complexity": complexity_weight,
            "diversity": diversity_weight,
            "constant": constant_weight,
        }

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        current_tokens: Optional[List[str]] = None,
        reward_history: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """生成带注意力筛选的 token."""
        n_samples = X.shape[0]
        n_vars = len(variable_names)

        # 步骤 1: 生成候选池（比目标多 5-10 倍）
        candidates = self._generate_candidate_pool(X, y, variable_names)

        # 步骤 2: 计算注意力分数
        scored_candidates = []
        for expr, values in candidates:
            score = self._compute_attention_score(expr, values, y, X)
            scored_candidates.append((expr, values, score))

        # 步骤 3: 按分数排序并筛选
        scored_candidates.sort(key=lambda x: x[2], reverse=True)

        # 保留策略：top attention_ratio，但至少 n_tokens 个
        n_keep = max(self.n_tokens, int(len(scored_candidates) * self.attention_ratio))
        n_keep = min(n_keep, len(scored_candidates))

        top_candidates = scored_candidates[:n_keep]

        # 步骤 4: 提取结果
        token_exprs = [c[0] for c in top_candidates]
        token_values = np.column_stack([c[1] for c in top_candidates])

        return token_exprs, token_values

    def _generate_candidate_pool(
        self, X: np.ndarray, y: np.ndarray, variable_names: List[str]
    ) -> List[Tuple[str, np.ndarray]]:
        """生成候选池（包含简单到复杂的表达式）."""
        n_samples = X.shape[0]
        n_vars = len(variable_names)
        candidates = []

        # 1. 原始变量
        for i, name in enumerate(variable_names):
            candidates.append((name, X[:, i]))

        # 2. 单变量变换（常用函数）
        for i, name in enumerate(variable_names):
            x_vals = X[:, i]
            candidates.extend([
                (f"sin({name})", np.sin(x_vals)),
                (f"cos({name})", np.cos(x_vals)),
                (f"{name}*{name}", x_vals ** 2),
                (f"sqrt(abs({name}))", np.sqrt(np.abs(x_vals))),
            ])

            # 根据数据范围添加 exp/log
            if np.all(np.abs(x_vals) < 5):
                candidates.append(
                    (f"exp({name})", np.exp(np.clip(x_vals, -700, 700)))
                )
            if np.all(x_vals > 0):
                candidates.append((f"log({name})", np.log(x_vals + 1e-10)))

        # 3. 两变量组合（只组合高相关的变量对）
        correlations = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append((i, j, abs(corr)))

        correlations.sort(key=lambda x: x[2], reverse=True)

        # 只取相关性较低的变量对（高相关的组合冗余）
        for i, j, corr in correlations[: min(3, len(correlations))]:
            if corr < 0.8:  # 避免高度冗余的组合
                name_i, name_j = variable_names[i], variable_names[j]
                candidates.extend([
                    (f"({name_i}+{name_j})", X[:, i] + X[:, j]),
                    (f"({name_i}*{name_j})", X[:, i] * X[:, j]),
                    (f"({name_i}-{name_j})", X[:, i] - X[:, j]),
                ])

        # 4. 添加 EML 组合（如果适用）
        for i in range(min(2, n_vars)):
            for j in range(min(2, n_vars)):
                name_i, name_j = variable_names[i], variable_names[j]
                eml_vals = np.exp(np.clip(X[:, i], -700, 700)) - np.log(
                    np.abs(X[:, j]) + 1e-10
                )
                candidates.append((f"eml({name_i},{name_j})", eml_vals))

        return candidates

    def _compute_attention_score(
        self,
        expr: str,
        values: np.ndarray,
        y: np.ndarray,
        X: np.ndarray,
    ) -> float:
        """计算注意力分数.

        综合评估：
        1. 与目标的相关性
        2. 结构复杂度（避免过复杂）
        3. 数值稳定性
        4. 常数拟合潜力
        """
        scores = {}

        # 1. 相关性分数（越高越好）
        if np.std(values) > 1e-10 and np.std(y) > 1e-10:
            corr = np.corrcoef(values, y)[0, 1]
            scores["correlation"] = abs(corr) if not np.isnan(corr) else 0
        else:
            scores["correlation"] = 0

        # 2. 结构复杂度分数（适中最好，避免过简单或过复杂）
        complexity = self._estimate_complexity(expr)
        # 理想复杂度在 3-8 之间
        if 3 <= complexity <= 8:
            scores["complexity"] = 1.0
        elif complexity < 3:
            scores["complexity"] = 0.7  # 偏简单，可接受
        else:
            scores["complexity"] = max(0, 1.0 - (complexity - 8) * 0.1)

        # 3. 数值稳定性（检查是否有 inf/nan）
        if np.any(np.isinf(values)) or np.any(np.isnan(values)):
            scores["stability"] = 0.0
        else:
            # 数值范围合理性
            val_range = np.max(values) - np.min(values)
            if val_range > 1e-10 and val_range < 1e6:
                scores["stability"] = 1.0
            else:
                scores["stability"] = 0.5

        # 4. 常数拟合潜力（线性拟合 R²）
        try:
            # 尝试线性拟合 y = a * values + b
            A = np.vstack([values, np.ones(len(values))]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            pred = a * values + b
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            scores["constant"] = max(0, r_squared)
        except Exception:
            scores["constant"] = 0

        # 加权求和
        total_score = (
            self.weights["correlation"] * scores.get("correlation", 0)
            + self.weights["complexity"] * scores.get("complexity", 0)
            + self.weights["diversity"] * scores.get("stability", 0)
            + self.weights["constant"] * scores.get("constant", 0)
        )

        return total_score

    def _estimate_complexity(self, expr: str) -> int:
        """估计表达式复杂度."""
        complexity = 0

        # 运算符计数
        ops = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "eml"]
        for op in ops:
            complexity += expr.count(op)

        # 括号深度
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


class DiversityAwareAttentionGenerator(AttentionTokenGenerator):
    """多样性感知的注意力生成器.

    在注意力基础上增加多样性约束，避免选中的 token 过于相似。
    """

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        current_tokens: Optional[List[str]] = None,
        reward_history: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """生成带多样性约束的 token."""
        # 先生成带分数的候选
        candidates = self._generate_candidate_pool(X, y, variable_names)
        scored_candidates = []
        for expr, values in candidates:
            score = self._compute_attention_score(expr, values, y, X)
            scored_candidates.append((expr, values, score))

        # 按分数排序
        scored_candidates.sort(key=lambda x: x[2], reverse=True)

        # 多样性筛选：避免选中的表达式过于相似
        selected = []
        selected_values = []

        for expr, values, score in scored_candidates:
            if len(selected) >= self.n_tokens:
                break

            # 检查与已选候选的相似度
            is_diverse = True
            for sel_values in selected_values:
                # 计算相关性
                if np.std(values) > 1e-10 and np.std(sel_values) > 1e-10:
                    corr = np.corrcoef(values, sel_values)[0, 1]
                    if abs(corr) > 0.95:  # 过于相似
                        is_diverse = False
                        break

            if is_diverse or len(selected) < self.n_tokens // 2:
                selected.append(expr)
                selected_values.append(values)

        if not selected:
            # 回退到简单选择
            selected = [c[0] for c in scored_candidates[: self.n_tokens]]
            selected_values = [c[1] for c in scored_candidates[: self.n_tokens]]

        token_values = np.column_stack(selected_values)
        return selected, token_values
