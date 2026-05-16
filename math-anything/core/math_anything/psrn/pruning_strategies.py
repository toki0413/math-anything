"""Pruning Strategies - 分层剪枝策略.

核心思想：在 PSRN 的每一层动态剪枝低质量候选，
避免指数级增长的计算爆炸。

策略类型：
1. 阈值剪枝：每层只保留前 K% 候选
2. 多样性剪枝：保留不相关的多个方向
3. 潜力剪枝：基于简单指标预估最终潜力
4. 渐进收紧：随着层数增加逐渐收紧阈值
"""

from typing import Callable, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np


class PruningStrategy(Enum):
    """剪枝策略类型."""
    TOP_K = "top_k"                    # 简单 Top-K
    ADAPTIVE_THRESHOLD = "adaptive"     # 自适应阈值
    DIVERSITY_AWARE = "diversity"       # 多样性感知
    POTENTIAL_BASED = "potential"       # 潜力预估
    PROGRESSIVE = "progressive"         # 渐进收紧


@dataclass
class PruningConfig:
    """剪枝配置."""
    strategy: PruningStrategy = PruningStrategy.ADAPTIVE_THRESHOLD
    initial_keep_ratio: float = 0.3     # 第一层保留比例
    min_keep_ratio: float = 0.05        # 最小保留比例
    diversity_threshold: float = 0.95   # 相似度阈值
    potential_estimation: bool = True   # 是否预估潜力
    early_stop_layers: int = 2          # 多少层无改进后早停


class LayerPruner:
    """分层剪枝器.

    在 PSRN 每层后执行，动态控制候选数量。

    Example:
        >>> pruner = LayerPruner(PruningConfig(strategy=PruningStrategy.TOP_K))
        >>> 
        >>> # 第 1 层后：1000 候选 → 300 候选
        >>> exprs_l1, values_l1 = pruner.prune(
        ...     layer_idx=0, expressions=exprs, values=values, y=y
        ... )
        >>> 
        >>> # 第 2 层后：300 候选 → 60 候选（更严格）
        >>> exprs_l2, values_l2 = pruner.prune(
        ...     layer_idx=1, expressions=exprs_l1, values=values_l1, y=y
        ... )
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()
        self.layer_history: List[Dict] = []
        self.no_improvement_count = 0

    def prune(
        self,
        layer_idx: int,
        expressions: List[str],
        values: np.ndarray,
        y: np.ndarray,
        parent_stats: Optional[Dict] = None,
    ) -> Tuple[List[str], np.ndarray, Dict]:
        """执行剪枝.

        Returns:
            kept_exprs: 保留的表达式
            kept_values: 保留的值矩阵
            stats: 剪枝统计
        """
        n_before = len(expressions)

        # 计算基础分数
        scores = self._compute_scores(values, y)

        # 根据策略选择保留
        if self.config.strategy == PruningStrategy.TOP_K:
            kept_indices = self._top_k_prune(scores, layer_idx)
        elif self.config.strategy == PruningStrategy.ADAPTIVE_THRESHOLD:
            kept_indices = self._adaptive_prune(scores, layer_idx, parent_stats)
        elif self.config.strategy == PruningStrategy.DIVERSITY_AWARE:
            kept_indices = self._diversity_prune(expressions, values, scores)
        elif self.config.strategy == PruningStrategy.POTENTIAL_BASED:
            kept_indices = self._potential_prune(expressions, values, scores, y)
        elif self.config.strategy == PruningStrategy.PROGRESSIVE:
            kept_indices = self._progressive_prune(scores, layer_idx)
        else:
            kept_indices = list(range(len(expressions)))

        # 应用保留
        kept_exprs = [expressions[i] for i in kept_indices]
        kept_values = values[:, kept_indices]

        # 统计
        n_after = len(kept_exprs)
        stats = {
            "layer": layer_idx,
            "before": n_before,
            "after": n_after,
            "pruned": n_before - n_after,
            "prune_ratio": 1 - n_after / n_before if n_before > 0 else 0,
            "best_score": float(np.max(scores)) if len(scores) > 0 else 0,
            "mean_score": float(np.mean(scores)) if len(scores) > 0 else 0,
        }

        self.layer_history.append(stats)

        return kept_exprs, kept_values, stats

    def _compute_scores(
        self, values: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """计算每个候选的分数（综合指标）."""
        n_samples, n_candidates = values.shape
        scores = np.zeros(n_candidates)

        for j in range(n_candidates):
            val = values[:, j]

            # 1. 负 MSE（越高越好）
            mse = np.mean((val - y) ** 2)
            mse_score = 1.0 / (1.0 + mse)  # 归一化到 [0,1]

            # 2. 相关性
            if np.std(val) > 1e-10 and np.std(y) > 1e-10:
                corr = np.corrcoef(val, y)[0, 1]
                corr_score = abs(corr) if not np.isnan(corr) else 0
            else:
                corr_score = 0

            # 3. 数值稳定性
            if np.any(np.isinf(val)) or np.any(np.isnan(val)):
                stability_score = 0
            else:
                val_range = np.max(val) - np.min(val)
                stability_score = 1.0 if val_range > 1e-10 else 0.3

            # 综合分数
            scores[j] = 0.5 * mse_score + 0.3 * corr_score + 0.2 * stability_score

        return scores

    def _top_k_prune(self, scores: np.ndarray, layer_idx: int) -> List[int]:
        """简单 Top-K 剪枝."""
        keep_ratio = max(
            self.config.min_keep_ratio,
            self.config.initial_keep_ratio * (0.7 ** layer_idx)
        )
        k = max(10, int(len(scores) * keep_ratio))

        # 获取 Top-K 索引
        top_indices = np.argsort(scores)[-k:][::-1]
        return top_indices.tolist()

    def _adaptive_prune(
        self,
        scores: np.ndarray,
        layer_idx: int,
        parent_stats: Optional[Dict],
    ) -> List[int]:
        """自适应阈值剪枝.

        根据分数分布动态调整阈值，而非固定比例。
        """
        if len(scores) == 0:
            return []

        # 计算分数统计
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # 动态阈值：均值 + 0.5 倍标准差
        threshold = mean_score + 0.5 * std_score

        # 如果阈值太高（保留太少），放宽到 Top 30%
        n_above = np.sum(scores >= threshold)
        min_keep = max(10, int(len(scores) * self.config.min_keep_ratio))

        if n_above < min_keep:
            # 放宽到 Top-K
            k = max(min_keep, int(len(scores) * self.config.initial_keep_ratio))
            return np.argsort(scores)[-k:][::-1].tolist()

        # 保留高于阈值的
        return np.where(scores >= threshold)[0].tolist()

    def _diversity_prune(
        self,
        expressions: List[str],
        values: np.ndarray,
        scores: np.ndarray,
    ) -> List[int]:
        """多样性感知剪枝.

        不仅保留高分，还保留不同的"方向"。
        """
        n = len(scores)
        if n <= 10:
            return list(range(n))

        # 按分数排序
        sorted_indices = np.argsort(scores)[::-1]

        # 贪心选择
        selected = [sorted_indices[0]]

        for idx in sorted_indices[1:]:
            if len(selected) >= max(10, int(n * self.config.initial_keep_ratio)):
                break

            # 检查与已选候选的差异
            is_diverse = True
            val = values[:, idx]

            for sel_idx in selected:
                sel_val = values[:, sel_idx]

                # 相关性检查
                if np.std(val) > 1e-10 and np.std(sel_val) > 1e-10:
                    corr = np.corrcoef(val, sel_val)[0, 1]
                    if abs(corr) > self.config.diversity_threshold:
                        is_diverse = False
                        break

            if is_diverse:
                selected.append(idx)

        return selected

    def _potential_prune(
        self,
        expressions: List[str],
        values: np.ndarray,
        scores: np.ndarray,
        y: np.ndarray,
    ) -> List[int]:
        """潜力预估剪枝.

        不仅看当前分数，还预估继续组合的潜力。
        """
        n = len(scores)
        potentials = np.zeros(n)

        for j in range(n):
            val = values[:, j]

            # 基础分数
            base_score = scores[j]

            # 潜力预估：当前相关性 + 结构复杂度奖励
            complexity = self._estimate_complexity(expressions[j])

            # 适中复杂度有更高潜力
            if 2 <= complexity <= 5:
                complexity_bonus = 0.1
            else:
                complexity_bonus = 0

            # 线性拟合潜力（如果当前已经有一定相关性）
            linear_potential = 0
            if base_score > 0.3:
                try:
                    A = np.vstack([val, np.ones(len(val))]).T
                    coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    if len(residuals) > 0:
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        r_squared = 1 - residuals[0] / ss_tot if ss_tot > 0 else 0
                        linear_potential = max(0, r_squared) * 0.1
                except:
                    pass

            potentials[j] = base_score + complexity_bonus + linear_potential

        # 按潜力选择
        k = max(10, int(n * self.config.initial_keep_ratio))
        return np.argsort(potentials)[-k:][::-1].tolist()

    def _progressive_prune(
        self, scores: np.ndarray, layer_idx: int
    ) -> List[int]:
        """渐进收紧剪枝.

        随着层数增加，保留比例指数下降。
        """
        # 指数衰减：0.5, 0.25, 0.125, ...
        keep_ratio = max(
            self.config.min_keep_ratio,
            self.config.initial_keep_ratio * (0.5 ** layer_idx)
        )

        k = max(10, int(len(scores) * keep_ratio))
        return np.argsort(scores)[-k:][::-1].tolist()

    def _estimate_complexity(self, expr: str) -> int:
        """估计表达式复杂度."""
        complexity = 0
        ops = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "eml"]
        for op in ops:
            complexity += expr.count(op)
        return max(complexity, 1)

    def get_history(self) -> List[Dict]:
        """获取剪枝历史."""
        return self.layer_history


class EarlyStopController:
    """早停控制器 - 监控改进情况，提前终止无意义的搜索."""

    def __init__(
        self,
        patience: int = 2,
        min_improvement: float = 0.01,
        window_size: int = 3,
    ):
        self.patience = patience
        self.min_improvement = min_improvement
        self.window_size = window_size

        self.best_scores: List[float] = []
        self.no_improvement_count = 0

    def check_should_stop(
        self, layer_idx: int, current_best_score: float
    ) -> Tuple[bool, str]:
        """检查是否应该早停.

        Returns:
            should_stop: 是否应该停止
            reason: 原因
        """
        self.best_scores.append(current_best_score)

        if len(self.best_scores) < self.window_size:
            return False, "insufficient_history"

        # 滑动窗口平均
        recent_avg = np.mean(self.best_scores[-self.window_size:])
        previous_avg = np.mean(
            self.best_scores[-self.window_size*2:-self.window_size]
        ) if len(self.best_scores) >= self.window_size * 2 else 0

        # 改进幅度
        improvement = recent_avg - previous_avg

        if improvement < self.min_improvement:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0

        if self.no_improvement_count >= self.patience:
            return True, f"no_improvement_for_{self.patience}_layers"

        return False, "improving"


class MultiObjectivePruner:
    """多目标剪枝 - 平衡准确性与复杂度.

    维护 Pareto 前沿，保留准确且简单的表达式。
    """

    def __init__(self, complexity_weight: float = 0.1):
        self.complexity_weight = complexity_weight
        self.pareto_front: List[Tuple[str, float, int]] = []  # (expr, mse, complexity)

    def update(
        self, expressions: List[str], mses: np.ndarray, complexities: List[int]
    ):
        """更新 Pareto 前沿."""
        for expr, mse, comp in zip(expressions, mses, complexities):
            # 检查是否被现有解支配
            dominated = False
            to_remove = []

            for i, (p_expr, p_mse, p_comp) in enumerate(self.pareto_front):
                # 如果现有解更准确且更简单，当前解被支配
                if p_mse <= mse and p_comp <= comp:
                    dominated = True
                    break
                # 如果当前解更准确且更简单，支配现有解
                if mse <= p_mse and comp <= p_comp:
                    to_remove.append(i)

            if not dominated:
                # 移除被支配的解
                for i in reversed(to_remove):
                    self.pareto_front.pop(i)
                # 添加新解
                self.pareto_front.append((expr, mse, comp))

    def get_best(self, balance: str = "mse") -> Optional[Tuple[str, float, int]]:
        """获取最佳解.

        Args:
            balance: "mse" (最准确), "simple" (最简单), "balanced" (平衡)
        """
        if not self.pareto_front:
            return None

        if balance == "mse":
            return min(self.pareto_front, key=lambda x: x[1])
        elif balance == "simple":
            return min(self.pareto_front, key=lambda x: x[2])
        else:  # balanced
            # 归一化后综合
            mses = np.array([p[1] for p in self.pareto_front])
            comps = np.array([p[2] for p in self.pareto_front])

            mse_norm = (mses - mses.min()) / (mses.max() - mses.min() + 1e-10)
            comp_norm = (comps - comps.min()) / (comps.max() - comps.min() + 1e-10)

            scores = mse_norm + self.complexity_weight * comp_norm
            best_idx = np.argmin(scores)
            return self.pareto_front[best_idx]

    def get_front(self) -> List[Tuple[str, float, int]]:
        """获取完整 Pareto 前沿."""
        return sorted(self.pareto_front, key=lambda x: x[1])
