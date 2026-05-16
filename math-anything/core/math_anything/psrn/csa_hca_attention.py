"""CSA+HCA 分层混合注意力机制.

借鉴 DeepSeek-V4 的架构思想：
- CSA 层: 粗粒度快速筛选，识别有潜力的表达式方向
- HCA 层: 精评估保留候选，深入分析组合潜力
- 分层调度: 根据迭代阶段动态调整 CSA/HCA 比例

应用到符号回归：
- CSA: 快速相关性过滤，生成大量粗候选
- HCA: 深度结构分析，精选高质量组合
"""

import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .token_generator import TokenGenerator


class ExpressionCompressor:
    """表达式压缩器 - 类比 CSA 的 KV 压缩.

    将相似表达式压缩为"语义簇"，降低评估复杂度。
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.threshold = similarity_threshold

    def compress(
        self,
        candidates: List[Tuple[str, np.ndarray]],
    ) -> List[Tuple[str, np.ndarray, List[int]]]:
        """将候选压缩为语义簇.

        Returns:
            [(representative_expr, values, original_indices), ...]
        """
        if not candidates:
            return []

        clusters = []
        assigned = set()

        for i, (expr, values) in enumerate(candidates):
            if i in assigned:
                continue

            # 创建新簇
            cluster_indices = [i]
            cluster_repr = (expr, values)

            # 寻找相似候选
            for j in range(i + 1, len(candidates)):
                if j in assigned:
                    continue

                _, other_values = candidates[j]

                # 计算相似度
                sim = self._compute_similarity(values, other_values)
                if sim > self.threshold:
                    cluster_indices.append(j)
                    assigned.add(j)

            clusters.append((cluster_repr[0], cluster_repr[1], cluster_indices))
            assigned.add(i)

        return clusters

    def _compute_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两个表达式输出的相似度."""
        if np.std(v1) < 1e-15 or np.std(v2) < 1e-15:
            return 1.0 if np.allclose(v1, v2, rtol=1e-5) else 0.0

        # 归一化后的相关性
        v1_norm = (v1 - np.mean(v1)) / (np.std(v1) + 1e-10)
        v2_norm = (v2 - np.mean(v2)) / (np.std(v2) + 1e-10)

        corr = np.corrcoef(v1_norm, v2_norm)[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0


class CSALayer:
    """压缩稀疏注意力层.

    快速粗筛，识别有潜力的表达式方向。
    类比 DeepSeek-V4 的 CSA。
    """

    def __init__(
        self,
        compression_ratio: int = 4,
        top_k_ratio: float = 0.3,
        fast_filters: Optional[List[str]] = None,
    ):
        self.compression_ratio = compression_ratio
        self.top_k_ratio = top_k_ratio
        self.fast_filters = fast_filters or ["correlation", "range", "validity"]
        self.compressor = ExpressionCompressor()

    def process(
        self,
        candidates: List[Tuple[str, np.ndarray]],
        y: np.ndarray,
        target_count: int,
    ) -> List[Tuple[str, np.ndarray, float]]:
        """CSA处理：压缩→稀疏注意力→Top-K筛选."""

        # 1. 压缩：相似表达式聚类
        clusters = self.compressor.compress(candidates)

        # 2. 稀疏注意力：快速评估每个簇
        scored_clusters = []
        for expr, values, indices in clusters:
            score = self._fast_evaluate(expr, values, y)
            scored_clusters.append((expr, values, indices, score))

        # 3. 选择 Top-K 簇
        scored_clusters.sort(key=lambda x: x[3], reverse=True)
        n_select = max(target_count, int(len(scored_clusters) * self.top_k_ratio))

        selected = []
        for expr, values, indices, score in scored_clusters[:n_select]:
            # 从簇中选择最佳代表
            selected.append((expr, values, score))

        return selected

    def _fast_evaluate(self, expr: str, values: np.ndarray, y: np.ndarray) -> float:
        """快速评估（低计算成本）."""
        scores = []

        # 相关性检查
        if "correlation" in self.fast_filters:
            if np.std(values) > 1e-10 and np.std(y) > 1e-10:
                corr = np.corrcoef(values, y)[0, 1]
                scores.append(abs(corr) if not np.isnan(corr) else 0)
            else:
                scores.append(0)

        # 数值范围检查
        if "range" in self.fast_filters:
            val_range = np.max(values) - np.min(values)
            if val_range > 1e-10 and val_range < 1e8:
                scores.append(1.0)
            else:
                scores.append(0.3)

        # 有效性检查
        if "validity" in self.fast_filters:
            if np.any(np.isinf(values)) or np.any(np.isnan(values)):
                scores.append(0.0)
            else:
                scores.append(1.0)

        return np.mean(scores) if scores else 0.5


class HCALayer:
    """重度压缩注意力层.

    深度精评估，分析表达式的组合潜力。
    类比 DeepSeek-V4 的 HCA。
    """

    def __init__(
        self,
        heavy_compression: int = 16,
        depth_metrics: Optional[List[str]] = None,
    ):
        self.heavy_compression = heavy_compression
        self.depth_metrics = depth_metrics or [
            "linear_fit",
            "nonlinear_potential",
            "eml_compatibility",
            "composability",
        ]

    def process(
        self,
        candidates: List[Tuple[str, np.ndarray, float]],  # 来自CSA的输出
        y: np.ndarray,
        X: np.ndarray,
        target_count: int,
    ) -> List[Tuple[str, np.ndarray, float]]:
        """HCA处理：深度评估→结构分析→精选."""

        # 1. 深度评估每个候选
        deeply_scored = []
        for expr, values, csa_score in candidates:
            deep_score = self._deep_evaluate(expr, values, y, X)
            # 结合CSA和HCA分数
            combined_score = 0.4 * csa_score + 0.6 * deep_score
            deeply_scored.append((expr, values, combined_score))

        # 2. 按组合分数排序
        deeply_scored.sort(key=lambda x: x[2], reverse=True)

        # 3. 多样性约束选择
        selected = self._diversity_select(deeply_scored, target_count)

        return selected

    def _deep_evaluate(
        self, expr: str, values: np.ndarray, y: np.ndarray, X: np.ndarray
    ) -> float:
        """深度评估（高计算成本，更精确）."""
        scores = {}

        # 线性拟合潜力
        if "linear_fit" in self.depth_metrics:
            try:
                A = np.vstack([values, np.ones(len(values))]).T
                coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
                if len(residuals) > 0:
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - residuals[0] / ss_tot if ss_tot > 0 else 0
                    scores["linear_fit"] = max(0, r_squared)
                else:
                    scores["linear_fit"] = 0
            except:
                scores["linear_fit"] = 0

        # 非线性潜力（高阶相关性）
        if "nonlinear_potential" in self.depth_metrics:
            try:
                # 检查与y^2, sqrt(y)等的相关性
                y_squared = y**2
                if np.std(y_squared) > 1e-10:
                    corr_sq = np.corrcoef(values, y_squared)[0, 1]
                    scores["nonlinear_potential"] = (
                        abs(corr_sq) if not np.isnan(corr_sq) else 0
                    )
                else:
                    scores["nonlinear_potential"] = 0
            except:
                scores["nonlinear_potential"] = 0

        # EML兼容性
        if "eml_compatibility" in self.depth_metrics:
            if "eml" in expr or "exp" in expr or "log" in expr:
                # EML适合处理乘积关系
                try:
                    y_exp = np.exp(np.clip(y, -50, 50))
                    if np.std(y_exp) > 1e-10:
                        corr_exp = np.corrcoef(values, y_exp)[0, 1]
                        scores["eml_compatibility"] = (
                            abs(corr_exp) if not np.isnan(corr_exp) else 0.3
                        )
                    else:
                        scores["eml_compatibility"] = 0.3
                except:
                    scores["eml_compatibility"] = 0.3
            else:
                scores["eml_compatibility"] = 0.1

        # 可组合性（与其他token组合的可能性）
        if "composability" in self.depth_metrics:
            complexity = self._estimate_complexity(expr)
            # 中等复杂度最适合组合
            if 2 <= complexity <= 6:
                scores["composability"] = 1.0
            elif complexity < 2:
                scores["composability"] = 0.7
            else:
                scores["composability"] = max(0, 1.0 - (complexity - 6) * 0.1)

        return np.mean(list(scores.values()))

    def _estimate_complexity(self, expr: str) -> int:
        """估计表达式复杂度."""
        complexity = 0
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

    def _diversity_select(
        self,
        candidates: List[Tuple[str, np.ndarray, float]],
        target_count: int,
    ) -> List[Tuple[str, np.ndarray, float]]:
        """多样性约束选择."""
        if not candidates:
            return []

        selected = [candidates[0]]

        for cand in candidates[1:]:
            if len(selected) >= target_count:
                break

            expr, values, score = cand
            is_diverse = True

            for sel in selected:
                _, sel_values, _ = sel

                # 数值相似度检查
                if np.std(values) > 1e-10 and np.std(sel_values) > 1e-10:
                    corr = np.corrcoef(values, sel_values)[0, 1]
                    if abs(corr) > 0.97:  # 过于相似
                        is_diverse = False
                        break

            if is_diverse:
                selected.append(cand)

        # 如果不够，补充高分数候选
        if len(selected) < target_count:
            remaining = [c for c in candidates if c not in selected]
            selected.extend(remaining[: target_count - len(selected)])

        return selected[:target_count]


class CSAHCAAttentionGenerator(TokenGenerator):
    """CSA+HCA 分层混合注意力 Token 生成器.

    核心思想（借鉴 DeepSeek-V4）：
    1. CSA 层：快速粗筛，识别有潜力的表达式方向
    2. HCA 层：精评估，深入分析组合潜力
    3. 分层调度：根据迭代阶段动态调整比例

    Example:
        >>> generator = CSAHCAAttentionGenerator(
        ...     n_tokens=20,
        ...     csa_ratio=0.7,  # 70% 预算给 CSA
        ...     hca_ratio=0.3,  # 30% 预算给 HCA
        ... )
    """

    def __init__(
        self,
        n_tokens: int = 20,
        csa_ratio: float = 0.6,
        hca_ratio: float = 0.4,
        adaptive_schedule: bool = True,
        min_pool_size: int = 200,
    ):
        """
        Args:
            n_tokens: 目标token数量
            csa_ratio: CSA层处理比例
            hca_ratio: HCA层处理比例
            adaptive_schedule: 是否根据迭代动态调整
            min_pool_size: 最小候选池大小
        """
        self.n_tokens = n_tokens
        self.csa_ratio = csa_ratio
        self.hca_ratio = hca_ratio
        self.adaptive_schedule = adaptive_schedule
        self.min_pool_size = min_pool_size

        # 分层组件
        self.csa_layer = CSALayer(
            compression_ratio=4,
            top_k_ratio=0.4,
        )
        self.hca_layer = HCALayer(
            heavy_compression=16,
        )

        # 迭代计数（用于自适应调度）
        self.iteration = 0

        # 历史最佳（用于进化）
        self.best_tokens_history = []

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        current_tokens: Optional[List[str]] = None,
        reward_history: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """生成 CSA+HCA 优化的 token."""

        self.iteration += 1

        # 1. 生成大规模候选池
        candidates = self._generate_candidate_pool(X, y, variable_names)

        # 2. 自适应调度：根据迭代调整 CSA/HCA 比例
        if self.adaptive_schedule:
            csa_ratio, hca_ratio = self._adaptive_schedule(self.iteration)
        else:
            csa_ratio, hca_ratio = self.csa_ratio, self.hca_ratio

        # 3. CSA 层：粗筛
        csa_target = max(self.n_tokens, int(len(candidates) * csa_ratio * 0.5))
        csa_output = self.csa_layer.process(candidates, y, csa_target)

        # 4. HCA 层：精筛
        hca_target = self.n_tokens
        hca_output = self.hca_layer.process(csa_output, y, X, hca_target)

        # 5. 保留历史优秀token（进化）
        if reward_history and len(hca_output) < self.n_tokens:
            top_historical = sorted(
                reward_history.items(), key=lambda x: x[1], reverse=True
            )[: self.n_tokens - len(hca_output)]

            for token, _ in top_historical:
                if token not in [h[0] for h in hca_output]:
                    # 需要重新计算值
                    try:
                        from .compiled_evaluator import CompiledEvaluator

                        evaluator = CompiledEvaluator()
                        values = evaluator.evaluate(token, X, variable_names)
                        hca_output.append((token, values, 0.8))  # 历史高分奖励
                    except:
                        pass

        # 6. 提取结果 - 确保所有values维度一致
        selected = hca_output[: self.n_tokens]
        token_exprs = [h[0] for h in selected]

        # 确保所有values都是一维且长度正确
        n_samples = X.shape[0]
        valid_values = []
        for h in selected:
            val = h[1]
            if val.shape != (n_samples,):
                val = np.asarray(val).reshape(-1)
                if val.shape[0] != n_samples:
                    continue  # 跳过维度不匹配的
            valid_values.append(val)

        if not valid_values:
            # 回退到简单候选
            return self._fallback_generate(X, variable_names)

        token_values = np.column_stack(valid_values)
        return token_exprs, token_values

    def _fallback_generate(
        self, X: np.ndarray, variable_names: List[str]
    ) -> Tuple[List[str], np.ndarray]:
        """回退生成器."""
        candidates = []
        for i, name in enumerate(variable_names):
            candidates.append((name, X[:, i]))
            candidates.append((f"sin({name})", np.sin(X[:, i])))
            candidates.append((f"({name}*{name})", X[:, i] ** 2))

        exprs = [c[0] for c in candidates[: self.n_tokens]]
        values = np.column_stack([c[1] for c in candidates[: self.n_tokens]])
        return exprs, values

    def _adaptive_schedule(self, iteration: int) -> Tuple[float, float]:
        """自适应调度 CSA/HCA 比例.

        早期：更多 CSA（快速探索）
        后期：更多 HCA（精细利用）
        """
        # 从 0.8/0.2 渐变到 0.4/0.6
        progress = min(1.0, iteration / 10)
        csa = 0.8 - 0.4 * progress
        hca = 0.2 + 0.4 * progress
        return csa, hca

    def _generate_candidate_pool(
        self, X: np.ndarray, y: np.ndarray, variable_names: List[str]
    ) -> List[Tuple[str, np.ndarray]]:
        """生成大规模候选池."""
        candidates = []
        n_vars = len(variable_names)

        # 基础变量
        for i, name in enumerate(variable_names):
            candidates.append((name, X[:, i]))

        # 单变量变换
        for i, name in enumerate(variable_names):
            x_vals = X[:, i]
            candidates.extend(
                [
                    (f"sin({name})", np.sin(x_vals)),
                    (f"cos({name})", np.cos(x_vals)),
                    (f"({name}*{name})", x_vals**2),
                    (f"({name}*{name}*{name})", x_vals**3),
                ]
            )

            if np.all(np.abs(x_vals) < 5):
                candidates.append((f"exp({name})", np.exp(np.clip(x_vals, -50, 50))))
            if np.all(x_vals > 0):
                candidates.append((f"log({name})", np.log(x_vals + 1e-10)))

        # EML 核心结构
        for i in range(min(3, n_vars)):
            for j in range(min(3, n_vars)):
                if i != j:
                    name_i, name_j = variable_names[i], variable_names[j]
                    eml_vals = np.exp(np.clip(X[:, i], -50, 50)) - np.log(
                        np.abs(X[:, j]) + 1e-10
                    )
                    candidates.append((f"eml({name_i},{name_j})", eml_vals))

        # 两变量组合
        if n_vars >= 2:
            for i in range(min(3, n_vars)):
                for j in range(i + 1, min(3, n_vars)):
                    name_i, name_j = variable_names[i], variable_names[j]
                    candidates.extend(
                        [
                            (f"({name_i}*{name_j})", X[:, i] * X[:, j]),
                            (f"({name_i}+{name_j})", X[:, i] + X[:, j]),
                            (f"({name_i}-{name_j})", X[:, i] - X[:, j]),
                        ]
                    )

        # 过滤无效
        valid = []
        for expr, values in candidates:
            if not (np.any(np.isinf(values)) or np.any(np.isnan(values))):
                if np.std(values) > 1e-15:
                    valid.append((expr, values))

        return valid
