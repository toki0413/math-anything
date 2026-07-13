"""域学习器。

从成功/失败经验中学习不变量的适用域条件，
替代硬编码的 scope 标签（scope="dft", scope="md"）。

核心理念：
  不变量 "E_total < 0" 的域不是 "所有 DFT 计算"
  而是 "绑定系统"（is_bound_system=True）。
  这个域条件不是人类标注的，而是从数据中学到的。

学习算法：

  1. 收集带有成功/失败标签的参数实例
  2. 对每个不变量，找到最能区分"成立"和"不成立"的特征
  3. 生成 domain_conditions（决策树/规则集）
  4. 将域条件赋给 LearnedInvariant.domain_conditions
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .invariant import DomainCondition, InvariantStatus, LearnedInvariant


@dataclass
class TrainingExample:
    """一个训练样本：参数 + 不变量的成立状态."""

    params: dict[str, Any]
    invariant_name: str
    status: InvariantStatus
    source: str = ""  # 来源引擎或场景


@dataclass
class DomainHypothesis:
    """一个域假设：什么条件下不变量成立."""

    invariant_name: str
    conditions: list[DomainCondition]
    accuracy: float  # 在训练数据上的准确率
    coverage: float  # 覆盖的训练样本比例
    support: int  # 支持的样本数


class DomainLearner:
    """域学习器.

    从经验中学习不变量的适用域，而非预定义 scope。

    Example::

        learner = DomainLearner()

        # 收集经验
        for record in execution_history:
            learner.add_example(TrainingExample(
                params=record.params,
                invariant_name="energy_negative",
                status=InvariantStatus.SATISFIED,
            ))

        # 学习域条件
        hypothesis = learner.learn("energy_negative")
        # → DomainHypothesis(
        #     conditions=[DomainCondition("is_bound_system", "==", True, confidence=0.92)],
        #     accuracy=0.92
        #   )
    """

    def __init__(self):
        self.examples: list[TrainingExample] = []
        self._hypothesis_cache: dict[str, DomainHypothesis] = {}

    def add_example(self, example: TrainingExample) -> None:
        self.examples.append(example)
        if len(self.examples) > 10000:
            self.examples = self.examples[-5000:]  # 保留最近5000条

    def learn(
        self,
        invariant_name: str,
        min_support: int = 5,
        min_accuracy: float = 0.7,
    ) -> DomainHypothesis | None:
        """为指定的不变量学习域条件.

        Returns:
            DomainHypothesis 或 None（样本不足时）
        """

        if invariant_name in self._hypothesis_cache:
            return self._hypothesis_cache[invariant_name]

        # 收集该不变量的正例和负例
        positives = [
            e for e in self.examples if e.invariant_name == invariant_name and e.status == InvariantStatus.SATISFIED
        ]
        negatives = [
            e
            for e in self.examples
            if e.invariant_name == invariant_name and e.status in (InvariantStatus.VIOLATED, InvariantStatus.WEAKENED)
        ]

        if len(positives) < min_support or len(negatives) < min_support:
            # 样本不足 → 保持通用域（无限制）
            return None

        # 提取所有候选特征
        all_features: set[str] = set()
        for ex in positives + negatives:
            all_features.update(ex.params.keys())

        # 去掉在正例和负例中都相同的特征（无区分力）
        candidates: list[tuple[str, float]] = []
        for feature in all_features:
            pos_values = [e.params.get(feature) for e in positives if feature in e.params]
            neg_values = [e.params.get(feature) for e in negatives if feature in e.params]

            if not pos_values or not neg_values:
                continue

            # 计算区分力（简化为值域不重叠程度）
            if _is_numeric(pos_values) and _is_numeric(neg_values):
                pos_mean = sum(pos_values) / len(pos_values)  # type: ignore[arg-type]
                neg_mean = sum(neg_values) / len(neg_values)  # type: ignore[arg-type]
                pos_std = _std(pos_values, pos_mean)  # type: ignore[arg-type]
                neg_std = _std(neg_values, neg_mean)  # type: ignore[arg-type]
                separation = abs(pos_mean - neg_mean) / max(pos_std + neg_std, 0.001)
                candidates.append((feature, min(separation, 1.0)))
            else:
                # 类别特征：检查正例中出现频率
                pos_set = set(str(v) for v in pos_values)
                neg_set = set(str(v) for v in neg_values)
                overlap = len(pos_set & neg_set) / max(len(pos_set | neg_set), 1)
                if overlap < 0.5:
                    candidates.append((feature, 1.0 - overlap))

        # 选择 top-3 特征
        candidates.sort(key=lambda x: -x[1])
        top_features = candidates[:3]

        # 构建域条件
        conditions: list[DomainCondition] = []
        for feature, score in top_features:
            pos_values = [e.params[feature] for e in positives if feature in e.params]
            if not pos_values:
                continue
            if _is_numeric(pos_values):
                mean_val = sum(pos_values) / len(pos_values)  # type: ignore[arg-type]
                conditions.append(
                    DomainCondition(
                        feature=feature,
                        operator=">",
                        threshold=mean_val * 0.8,
                        confidence=min(score, 0.95),
                        learned_from=[f"{len(positives)}+{len(negatives)} examples"],
                    )
                )
            else:
                most_common = Counter(str(v) for v in pos_values).most_common(1)[0][0]
                conditions.append(
                    DomainCondition(
                        feature=feature,
                        operator="==",
                        threshold=most_common,
                        confidence=min(score, 0.95),
                        learned_from=[f"{len(positives)}+{len(negatives)} examples"],
                    )
                )

        # 准确率：正例中满足所有条件的比例
        if conditions:
            correct = 0
            for ex in positives:
                if all(_check_condition(c, ex.params) for c in conditions):
                    correct += 1
            accuracy = correct / len(positives) if positives else 0.0
        else:
            accuracy = 1.0

        hypothesis = DomainHypothesis(
            invariant_name=invariant_name,
            conditions=conditions,
            accuracy=accuracy,
            coverage=len(positives) / max(len(positives) + len(negatives), 1),
            support=len(positives),
        )

        if accuracy >= min_accuracy:
            self._hypothesis_cache[invariant_name] = hypothesis

        return hypothesis

    def apply_to_invariant(
        self,
        invariant: LearnedInvariant,
        min_support: int = 5,
    ) -> LearnedInvariant:
        """学习域条件并应用到不变量."""
        hypothesis = self.learn(invariant.name, min_support=min_support)
        if hypothesis and hypothesis.conditions:
            invariant.domain_conditions = hypothesis.conditions
            invariant.domain_confidence = hypothesis.accuracy
        return invariant

    def stats(self) -> dict[str, Any]:
        return {
            "total_examples": len(self.examples),
            "unique_invariants": len(set(e.invariant_name for e in self.examples)),
            "hypotheses_cached": len(self._hypothesis_cache),
        }


# ── helpers ──


def _is_numeric(values: list[Any]) -> bool:
    return all(isinstance(v, (int, float)) for v in values if v is not None)


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return max(variance**0.5, 1e-10)  # type: ignore[no-any-return]


def _check_condition(dc: DomainCondition, params: dict[str, Any]) -> bool:
    val = params.get(dc.feature)
    if val is None:
        return False
    if dc.operator == "==":
        return str(val) == str(dc.threshold)
    if dc.operator == "present":
        return True
    if dc.operator == "absent":
        return False
    if isinstance(val, (int, float)):
        threshold = float(dc.threshold) if isinstance(dc.threshold, str) else dc.threshold
        if dc.operator == ">":
            return float(val) > threshold
        if dc.operator == "<":
            return float(val) < threshold
        if dc.operator == ">=":
            return float(val) >= threshold
        if dc.operator == "<=":
            return float(val) <= threshold
    return True
