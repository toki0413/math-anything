"""自适应注意力机制 - 动态优化 Token 选择策略.

核心创新：
1. 元学习权重调整 - 根据历史反馈动态调整评分维度权重
2. 结构感知评分 - 不仅评分单个token，还评估其组合潜力
3. EML深度集成 - 理解eml(exp(x)-log(y))的数学本质
4. 多臂老虎机探索 - 平衡探索与利用
"""

import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .token_generator import TokenGenerator


class AdaptiveAttentionGenerator(TokenGenerator):
    """自适应注意力 Token 生成器.

    通过元学习动态调整评分策略，逐步提升token质量。

    Example:
        >>> generator = AdaptiveAttentionGenerator(
        ...     n_tokens=20,
        ...     learning_rate=0.1,
        ...     exploration_ratio=0.2
        ... )
        >>> # 每次调用都会根据历史反馈优化权重
    """

    def __init__(
        self,
        n_tokens: int = 20,
        exploration_ratio: float = 0.2,
        learning_rate: float = 0.1,
        min_pool_size: int = 100,
        use_eml_priority: bool = True,
    ):
        """
        Args:
            n_tokens: 目标token数量
            exploration_ratio: 探索比例（随机选择vs注意力选择）
            learning_rate: 权重学习率
            min_pool_size: 最小候选池大小
            use_eml_priority: 是否优先EML相关候选
        """
        self.n_tokens = n_tokens
        self.exploration_ratio = exploration_ratio
        self.learning_rate = learning_rate
        self.min_pool_size = min_pool_size
        self.use_eml_priority = use_eml_priority

        # 可学习的权重
        self.weights = {
            "correlation": 0.35,
            "structure": 0.25,
            "linearity": 0.20,
            "diversity": 0.15,
            "eml_potential": 0.05,
        }

        # 历史反馈统计
        self.feedback_history = defaultdict(list)
        self.token_success_rate = {}

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        current_tokens: Optional[List[str]] = None,
        reward_history: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """生成自适应优化的token."""

        # 1. 根据历史反馈更新权重
        if reward_history:
            self._update_weights_from_history(reward_history)

        # 2. 问题特征分析
        problem_features = self._analyze_problem(X, y, variable_names)

        # 3. 生成大规模候选池
        candidates = self._generate_enhanced_pool(
            X, y, variable_names, problem_features
        )

        # 4. 计算自适应分数
        scored_candidates = []
        for expr, values, features in candidates:
            score = self._compute_adaptive_score(
                expr, values, y, X, features, problem_features
            )
            scored_candidates.append((expr, values, score, features))

        # 5. 探索vs利用
        n_explore = int(self.n_tokens * self.exploration_ratio)
        n_exploit = self.n_tokens - n_explore

        # 利用：高分数候选
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        exploit_candidates = scored_candidates[: n_exploit * 3]  # 候选池

        # 探索：随机采样（但有一定偏向性）
        remaining = scored_candidates[n_exploit * 3 :]
        if remaining and n_explore > 0:
            probs = np.array([max(0.01, c[2]) for c in remaining])
            probs = probs / probs.sum()
            n_sample = min(n_explore * 2, len(remaining))
            indices = np.random.choice(
                len(remaining), size=n_sample, replace=False, p=probs
            )
            explore_candidates = [remaining[i] for i in indices]
        else:
            explore_candidates = []

        # 6. 多样性筛选
        selected = self._diversity_filter(
            exploit_candidates + explore_candidates, self.n_tokens
        )

        # 7. 组装结果
        token_exprs = [s[0] for s in selected]
        token_values = np.column_stack([s[1] for s in selected])

        return token_exprs, token_values

    def _analyze_problem(
        self, X: np.ndarray, y: np.ndarray, variable_names: List[str]
    ) -> Dict:
        """分析问题特征，指导候选生成."""
        features = {
            "n_vars": len(variable_names),
            "n_samples": X.shape[0],
            "y_range": (float(np.min(y)), float(np.max(y))),
            "y_std": float(np.std(y)),
        }

        # 分析目标函数的非线性程度
        y_normalized = (y - np.mean(y)) / (np.std(y) + 1e-10)
        features["y_skewness"] = float(np.mean(y_normalized**3))
        features["y_kurtosis"] = float(np.mean(y_normalized**4) - 3)

        # 变量与目标的相关性模式
        var_correlations = []
        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 1e-10:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                var_correlations.append(abs(corr) if not np.isnan(corr) else 0)
            else:
                var_correlations.append(0)
        features["var_correlations"] = var_correlations
        features["max_correlation"] = max(var_correlations) if var_correlations else 0

        # 判断可能的函数类型
        if features["max_correlation"] > 0.9:
            features["likely_pattern"] = "linear"
        elif features["y_skewness"] > 1.0:
            features["likely_pattern"] = "exponential"
        elif any(c > 0.7 for c in var_correlations):
            features["likely_pattern"] = "polynomial"
        else:
            features["likely_pattern"] = "complex"

        return features

    def _generate_enhanced_pool(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        problem_features: Dict,
    ) -> List[Tuple[str, np.ndarray, Dict]]:
        """生成增强型候选池，基于问题特征."""
        candidates = []
        n_vars = len(variable_names)

        # 1. 原始变量
        for i, name in enumerate(variable_names):
            candidates.append((name, X[:, i], {"type": "variable", "complexity": 1}))

        # 2. 基于问题特征的函数选择
        pattern = problem_features["likely_pattern"]

        # 多项式特征
        if pattern in ["polynomial", "complex", "linear"]:
            for i, name in enumerate(variable_names):
                x_vals = X[:, i]
                candidates.extend(
                    [
                        (
                            f"({name}*{name})",
                            x_vals**2,
                            {"type": "square", "complexity": 2},
                        ),
                        (
                            f"({name}*{name}*{name})",
                            x_vals**3,
                            {"type": "cubic", "complexity": 3},
                        ),
                    ]
                )

        # 三角函数（适合周期性）
        if pattern in ["complex", "polynomial"]:
            for i, name in enumerate(variable_names):
                x_vals = X[:, i]
                candidates.extend(
                    [
                        (
                            f"sin({name})",
                            np.sin(x_vals),
                            {"type": "sin", "complexity": 2},
                        ),
                        (
                            f"cos({name})",
                            np.cos(x_vals),
                            {"type": "cos", "complexity": 2},
                        ),
                    ]
                )

        # 指数/对数（适合快速增长/衰减）
        for i, name in enumerate(variable_names):
            x_vals = X[:, i]
            if np.all(x_vals < 5):  # 避免exp爆炸
                exp_vals = np.exp(np.clip(x_vals, -50, 50))
                candidates.append(
                    (f"exp({name})", exp_vals, {"type": "exp", "complexity": 2})
                )
            if np.all(x_vals > 0):
                candidates.append(
                    (
                        f"log({name})",
                        np.log(x_vals + 1e-10),
                        {"type": "log", "complexity": 2},
                    )
                )

        # 3. EML核心结构：exp(x) - log(y)
        if self.use_eml_priority and n_vars >= 2:
            for i in range(min(3, n_vars)):
                for j in range(min(3, n_vars)):
                    if i != j:
                        name_i, name_j = variable_names[i], variable_names[j]
                        # EML核心
                        eml_vals = np.exp(np.clip(X[:, i], -50, 50)) - np.log(
                            np.abs(X[:, j]) + 1e-10
                        )
                        candidates.append(
                            (
                                f"eml({name_i},{name_j})",
                                eml_vals,
                                {"type": "eml", "complexity": 4, "eml_core": True},
                            )
                        )

                        # EML变体
                        candidates.extend(
                            [
                                (
                                    f"(exp({name_i})-log({name_j}))",
                                    eml_vals,
                                    {"type": "eml_expanded", "complexity": 5},
                                ),
                            ]
                        )

        # 4. 组合特征
        if n_vars >= 2:
            # 乘积项
            for i in range(min(3, n_vars)):
                for j in range(i + 1, min(3, n_vars)):
                    name_i, name_j = variable_names[i], variable_names[j]
                    prod_vals = X[:, i] * X[:, j]
                    candidates.append(
                        (
                            f"({name_i}*{name_j})",
                            prod_vals,
                            {"type": "product", "complexity": 2},
                        )
                    )

        # 过滤无效候选
        valid_candidates = []
        for expr, values, features in candidates:
            if self._is_valid_candidate(values):
                valid_candidates.append((expr, values, features))

        return valid_candidates

    def _is_valid_candidate(self, values: np.ndarray) -> bool:
        """检查候选是否有效."""
        if np.any(np.isinf(values)) or np.any(np.isnan(values)):
            return False
        if np.std(values) < 1e-15:  # 常数
            return False
        if np.max(np.abs(values)) > 1e10:  # 数值爆炸
            return False
        return True

    def _compute_adaptive_score(
        self,
        expr: str,
        values: np.ndarray,
        y: np.ndarray,
        X: np.ndarray,
        features: Dict,
        problem_features: Dict,
    ) -> float:
        """计算自适应注意力分数."""
        scores = {}

        # 1. 相关性分数
        if np.std(values) > 1e-10 and np.std(y) > 1e-10:
            corr = np.corrcoef(values, y)[0, 1]
            corr_score = abs(corr) if not np.isnan(corr) else 0
            # 非线性相关性奖励
            if corr_score > 0.5:
                corr_score = 0.5 + 0.5 * corr_score
            scores["correlation"] = corr_score
        else:
            scores["correlation"] = 0

        # 2. 结构分数（基于问题特征）
        complexity = features.get("complexity", 3)
        if problem_features["likely_pattern"] == "linear" and complexity <= 2:
            scores["structure"] = 1.0
        elif problem_features["likely_pattern"] == "complex" and complexity >= 3:
            scores["structure"] = 0.9
        else:
            scores["structure"] = max(0, 1.0 - abs(complexity - 4) * 0.15)

        # 3. 线性拟合潜力
        try:
            A = np.vstack([values, np.ones(len(values))]).T
            coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
            if len(residuals) > 0:
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - residuals[0] / ss_tot if ss_tot > 0 else 0
                scores["linearity"] = max(0, min(1, r_squared))
            else:
                scores["linearity"] = 0
        except:
            scores["linearity"] = 0

        # 4. EML潜力（特殊奖励）
        if features.get("eml_core") or "eml" in expr:
            # 检查EML是否与目标有结构性匹配
            try:
                # EML通常适合处理乘积关系
                y_exp = np.exp(np.clip(y, -50, 50))
                if np.std(y_exp) > 1e-10:
                    corr_with_exp = np.corrcoef(values, y_exp)[0, 1]
                    if abs(corr_with_exp) > 0.5:
                        scores["eml_potential"] = abs(corr_with_exp)
                    else:
                        scores["eml_potential"] = 0.3
                else:
                    scores["eml_potential"] = 0.2
            except:
                scores["eml_potential"] = 0.1
        else:
            scores["eml_potential"] = 0.1

        # 5. 多样性分数（后续处理）
        scores["diversity"] = 0.5  # 占位，后续调整

        # 加权求和
        total = sum(self.weights.get(k, 0) * scores.get(k, 0) for k in scores)
        return total

    def _diversity_filter(self, candidates: List[Tuple], n_select: int) -> List[Tuple]:
        """多样性筛选，避免选择过于相似的候选."""
        if not candidates:
            return []

        selected = [candidates[0]]

        for cand in candidates[1:]:
            if len(selected) >= n_select:
                break

            expr, values, score, features = cand

            # 计算与已选候选的相似度
            is_diverse = True
            for sel in selected:
                sel_values = sel[1]

                # 相关性检查
                if np.std(values) > 1e-10 and np.std(sel_values) > 1e-10:
                    corr = np.corrcoef(values, sel_values)[0, 1]
                    if abs(corr) > 0.98:  # 几乎相同
                        is_diverse = False
                        break

                # 表达式结构检查
                if self._similar_structure(expr, sel[0]):
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(cand)

        # 如果不够，补充高分数候选
        if len(selected) < n_select:
            remaining = [c for c in candidates if c not in selected]
            selected.extend(remaining[: n_select - len(selected)])

        return selected[:n_select]

    def _similar_structure(self, expr1: str, expr2: str) -> bool:
        """检查表达式结构是否相似."""
        # 简化后比较
        e1 = expr1.replace(" ", "").lower()
        e2 = expr2.replace(" ", "").lower()

        # 相同变量集合
        import re

        vars1 = set(re.findall(r"[xyzt]\d?", e1))
        vars2 = set(re.findall(r"[xyzt]\d?", e2))

        if vars1 != vars2:
            return False

        # 函数调用相似性
        funcs1 = set(re.findall(r"(sin|cos|exp|log|eml)", e1))
        funcs2 = set(re.findall(r"(sin|cos|exp|log|eml)", e2))

        return funcs1 == funcs2

    def _update_weights_from_history(self, reward_history: Dict[str, float]):
        """根据历史反馈更新权重."""
        if not reward_history:
            return

        # 分析成功的token特征
        successful_tokens = [
            (token, reward) for token, reward in reward_history.items() if reward > 0.7
        ]

        if not successful_tokens:
            return

        # 统计成功token的特征
        eml_success = sum(1 for t, _ in successful_tokens if "eml" in t)
        simple_success = sum(1 for t, _ in successful_tokens if len(t) < 10)
        complex_success = sum(1 for t, _ in successful_tokens if len(t) > 20)

        total = len(successful_tokens)

        # 调整权重
        if total > 0:
            # 如果EML成功率高，提高eml_potential权重
            if eml_success / total > 0.3:
                self.weights["eml_potential"] = min(
                    0.3, self.weights["eml_potential"] + 0.05
                )

            # 根据复杂度调整structure权重
            if simple_success > complex_success:
                # 简单结构更有效，提高structure权重
                self.weights["structure"] = min(0.4, self.weights["structure"] + 0.02)

        # 记录反馈
        for token, reward in reward_history.items():
            self.feedback_history[token].append(reward)


class HybridAttentionGenerator(AdaptiveAttentionGenerator):
    """混合注意力生成器 - 结合多种策略."""

    def __init__(
        self,
        n_tokens: int = 20,
        strategy_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(n_tokens=n_tokens, **kwargs)

        # 多策略权重
        self.strategy_weights = strategy_weights or {
            "attention": 0.4,
            "evolutionary": 0.3,
            "gradient": 0.3,
        }

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        current_tokens: Optional[List[str]] = None,
        reward_history: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """混合策略生成token."""

        n_by_strategy = {
            k: int(self.n_tokens * w) for k, w in self.strategy_weights.items()
        }

        all_tokens = []
        all_values = []

        # 1. 注意力策略
        if n_by_strategy["attention"] > 0:
            # 临时调整n_tokens
            orig_n = self.n_tokens
            self.n_tokens = n_by_strategy["attention"]
            tokens, values = super().generate(
                X, y, variable_names, current_tokens, reward_history
            )
            self.n_tokens = orig_n
            all_tokens.extend(tokens)
            if len(all_values) == 0:
                all_values = values
            else:
                all_values = np.column_stack([all_values, values])

        # 2. 进化策略（保留历史优秀token）
        if n_by_strategy["evolutionary"] > 0 and reward_history:
            top_tokens = sorted(
                reward_history.items(), key=lambda x: x[1], reverse=True
            )[: n_by_strategy["evolutionary"]]

            for token, _ in top_tokens:
                if token not in all_tokens:
                    all_tokens.append(token)

        # 3. 梯度策略（基于误差方向）
        if n_by_strategy["gradient"] > 0:
            gradient_tokens = self._generate_gradient_tokens(
                X, y, variable_names, n_by_strategy["gradient"]
            )
            # 这里简化处理，实际应该计算值

        # 去重
        unique_tokens = []
        seen = set()
        for i, token in enumerate(all_tokens):
            if token not in seen:
                seen.add(token)
                unique_tokens.append((i, token))

        # 重新获取值
        final_tokens = [t for _, t in unique_tokens[: self.n_tokens]]

        # 计算值
        evaluator = self._get_evaluator()
        final_values = []
        for token in final_tokens:
            try:
                val = evaluator(token, X, variable_names)
                final_values.append(val)
            except:
                pass

        if final_values:
            return final_tokens, np.column_stack(final_values)
        else:
            # 回退
            return super().generate(
                X, y, variable_names, current_tokens, reward_history
            )

    def _get_evaluator(self) -> Callable:
        """获取表达式评估函数."""
        from .compiled_evaluator import CompiledEvaluator

        evaluator = CompiledEvaluator()
        return lambda expr, X, names: evaluator.evaluate(expr, X, names)

    def _generate_gradient_tokens(
        self, X: np.ndarray, y: np.ndarray, variable_names: List[str], n: int
    ) -> List[str]:
        """基于梯度方向生成token."""
        # 简化实现
        tokens = []
        for name in variable_names[:n]:
            tokens.extend([f"{name}", f"sin({name})", f"exp({name})"])
        return tokens[:n]
