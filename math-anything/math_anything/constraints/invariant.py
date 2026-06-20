"""可学习的不变量。

一个 LearnedInvariant 不是硬编码的真理，而是：
  1. 域条件是可学习的（从成功/失败经验中推断）
  2. 可以是概率性的（存在置信区间）
  3. 可以被弱化而非完全抛弃（存在弱化规则链）
  4. 在态射链上传播时会保留传播记录

与静态 StructuralInvariant 的区别：
  StructuralInvariant: "H 自伴 → 特征值均为实数"（定理，100%置信）
  LearnedInvariant:   "E_total < 0"（绑定系统的经验规律，95%置信，
                       域条件是 is_bound_system = True，
                       弱化规则：如果违反，检查系统是否非绑定）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from math_anything._compat import StrEnum
from math_anything.utils.safe_eval import safe_eval

logger = logging.getLogger(__name__)


class InvariantStatus(StrEnum):
    """不变量的状态."""

    SATISFIED = "satisfied"  # 完全成立
    WEAKENED = "weakened"  # 弱化后成立（如放宽精度）
    CONDITIONAL = "conditional"  # 仅在额外条件下成立
    VIOLATED = "violated"  # 被违反
    UNKNOWN = "unknown"  # 无法判断
    INACTIVE = "inactive"  # 域条件不满足，不变量不适用


class PropagationOutcome(StrEnum):
    """不变量在态射下的传播结果."""

    PRESERVED = "preserved"  # 不变，完全传递
    WEAKENED = "weakened"  # 弱化但未消失
    CONDITIONAL = "conditional"  # 变为需要额外条件
    LOST = "lost"  # 完全丢失
    EMERGED = "emerged"  # 态射引入的新不变量


@dataclass
class WeakeningRule:
    r"""弱化规则：当不变量面临违反时，如何降级而非完全丢弃.

    Example:
        原不变量: "\|E_n+1 - E_n\| < 1e-6"
        弱化规则: "\|E_n+1 - E_n\| < 1e-4"（放宽两个数量级）
        触发条件: when iterations exceed NELM/2 without convergence
    """

    name: str
    weakened_expression: str  # 弱化后的表达式
    trigger_condition: str  # 触发条件（如 "iterations > NELM/2"）
    consequence: str  # 弱化的后果
    recovery_path: str = ""  # 如何从弱化恢复


@dataclass
class DomainCondition:
    """域的单个条件.

    不是硬编码的 scope="dft"，而是可学习的特征条件。
    """

    feature: str  # 特征名（如 "is_bound_system", "element_count > 10"）
    operator: str  # "==" | ">" | "<" | ">=" | "<=" | "in" | "derived"
    threshold: float | str  # 阈值
    confidence: float = 0.5  # 此条件本身的置信度（从经验中学习）
    learned_from: list[str] = field(default_factory=list)  # 从哪些经验中学习


@dataclass
class LearnedInvariant:
    """可学习的不变量.

    Attributes:
        name: 唯一标识符
        expression: 数学表达式
        theorem: 定理来源（若来自定理）
        domain_conditions: 可学习的域条件列表
        domain_confidence: 域整体置信度
        is_probabilistic: 是否是概率性的
        probability_threshold: 概率阈值
        weakening_rules: 弱化规则链（从强到弱）
        propagation_history: 在态射链上的传播记录
        severity: 违反的严重程度
    """

    name: str
    expression: str
    theorem: str = ""
    description: str = ""

    # 域：不是预定义的 scope，而是可学习的条件
    domain_conditions: list[DomainCondition] = field(default_factory=list)
    domain_confidence: float = 1.0
    is_probabilistic: bool = False
    probability_threshold: float = 0.95

    # 弱化
    weakening_rules: list[WeakeningRule] = field(default_factory=list)
    active_weakening: int = 0  # 当前生效的弱化级别（0=未弱化）

    # 传播
    propagation_history: list[tuple[str, str]] = field(default_factory=list)
    # [(morphism_name, outcome), ...]

    # 严重程度
    severity: str = "theorem"  # "theorem" | "consistency" | "heuristic" | "empirical"

    # 内部状态
    violation_count: int = 0
    satisfaction_count: int = 0
    effective_expression: str = ""  # 应用弱化后的有效表达式

    def __post_init__(self) -> None:
        if not self.effective_expression:
            self.effective_expression = self.expression

    def is_active(self, params: dict[str, Any]) -> bool:
        """检查域条件是否满足（不变量是否适用）."""
        if not self.domain_conditions:
            return True
        for dc in self.domain_conditions:
            if not self._check_condition(dc, params):
                return False
        return True

    def evaluate(self, params: dict[str, Any]) -> InvariantStatus:
        """评估不变量在当前参数下的状态.

        返回 SATISFIED / WEAKENED / CONDITIONAL / VIOLATED / UNKNOWN / INACTIVE.
        """
        if not self.is_active(params):
            return InvariantStatus.INACTIVE

        expr = self.effective_expression

        # 尝试数值评估
        try:
            # 构建安全的评估环境
            safe_vars = {k: v for k, v in params.items() if isinstance(v, (int, float, bool))}
            result = safe_eval(expr, safe_vars)
            if isinstance(result, bool):
                if result:
                    status = InvariantStatus.WEAKENED if self.active_weakening > 0 else InvariantStatus.SATISFIED
                    self.satisfaction_count += 1
                    return status
                else:
                    self.violation_count += 1
                    return InvariantStatus.VIOLATED
        except (ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError) as e:
            logger.debug(f"Invariant evaluation failed for '{self.name}': {e}")
            pass

        return InvariantStatus.UNKNOWN

    def weaken(self, rule_index: int | None = None) -> LearnedInvariant:
        """应用弱化规则.

        如果 rule_index 未指定，依次尝试下一个最宽松的弱化规则。
        """
        if not self.weakening_rules:
            return self

        if rule_index is None:
            rule_index = min(self.active_weakening + 1, len(self.weakening_rules) - 1)

        if rule_index >= len(self.weakening_rules):
            return self

        rule = self.weakening_rules[rule_index]
        self.effective_expression = rule.weakened_expression
        self.active_weakening = rule_index + 1
        self.propagation_history.append(("weakened", f"Applied weakening rule: {rule.name}"))
        return self

    def restore(self) -> LearnedInvariant:
        """恢复到原始表达式."""
        self.effective_expression = self.expression
        self.active_weakening = 0
        return self

    def record_propagation(self, morphism_name: str, outcome: str) -> None:
        self.propagation_history.append((morphism_name, outcome))

    def learn_domain(self, data: list[dict[str, Any]], labels: list[bool]) -> None:
        """从 label 数据中学习域条件.

        data: 参数实例列表
        labels: 每个实例中该不变量是否成立
        """
        if len(data) < 5:
            return

        # 简化：选择最能区分成立/不成立实例的特征
        from collections import Counter

        pos_features: Counter[str] = Counter()
        neg_features: Counter[str] = Counter()

        for i, params in enumerate(data):
            for key in params:
                if labels[i]:
                    pos_features[key] += 1
                else:
                    neg_features[key] += 1

        # 找到区分度最高的特征
        for feature in pos_features:
            if feature in neg_features:
                pos_count = pos_features[feature]
                neg_count = neg_features[feature]
                if pos_count > 3 * neg_count or neg_count > 3 * pos_count:
                    confidence = abs(pos_count - neg_count) / (pos_count + neg_count)
                    self.domain_conditions.append(
                        DomainCondition(
                            feature=feature,
                            operator="present" if pos_count > neg_count else "absent",
                            threshold=0.0,
                            confidence=min(confidence, 0.95),
                            learned_from=[f"trained on {len(data)} examples"],
                        )
                    )

        self.domain_confidence = 0.8

    @staticmethod
    def _check_condition(dc: DomainCondition, params: dict[str, Any]) -> bool:
        """检查单个域条件."""
        if dc.operator == "present":
            return dc.feature in params
        if dc.operator == "absent":
            return dc.feature not in params

        val = params.get(dc.feature)
        if val is None:
            return False

        threshold = dc.threshold
        if isinstance(threshold, str):
            try:
                threshold = float(threshold)
            except ValueError:
                return str(val) == threshold

        if dc.operator == "==":
            return val == threshold
        if dc.operator == ">":
            return float(val) > threshold
        if dc.operator == "<":
            return float(val) < threshold
        if dc.operator == ">=":
            return float(val) >= threshold
        if dc.operator == "<=":
            return float(val) <= threshold
        if dc.operator == "in":
            return val in (threshold if isinstance(threshold, list) else [threshold])

        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.effective_expression,
            "original_expression": self.expression,
            "theorem": self.theorem,
            "status": "weakened" if self.active_weakening > 0 else "original",
            "domain_confidence": self.domain_confidence,
            "violation_count": self.violation_count,
            "satisfaction_count": self.satisfaction_count,
            "propagation_history": self.propagation_history,
            "severity": self.severity,
        }


# ── 工厂：从 StructuralInvariant 创建 LearnedInvariant ──


def from_structural_invariant(inv) -> LearnedInvariant:
    """从静态 StructuralInvariant 创建可学习的 LearnedInvariant."""
    return LearnedInvariant(
        name=inv.name,
        expression=inv.expression,
        theorem=inv.theorem,
        severity=inv.severity,
        is_probabilistic=False,
        domain_confidence=1.0,
    )


# ── 预定义：常见的弱化规则 ──

SCF_WEAKENING_RULES = [
    WeakeningRule(
        name="relax_convergence_1e4",
        weakened_expression="abs(E_new - E_old) < 1e-4",
        trigger_condition="iterations > NELM/2",
        consequence="total energy accuracy reduced from 1e-6 to 1e-4 eV",
        recovery_path="reduce mixing parameter and retry with original tolerance",
    ),
    WeakeningRule(
        name="relax_convergence_1e2",
        weakened_expression="abs(E_new - E_old) < 1e-2",
        trigger_condition="iterations > NELM",
        consequence="significant accuracy loss; suitable only for pre-relaxation",
        recovery_path="restart with better initial wavefunction guess",
    ),
]

MD_TIMESTEP_WEAKENING = [
    WeakeningRule(
        name="halve_timestep",
        weakened_expression="dt <= 0.5 * dt_original",
        trigger_condition="energy_drift > 0.001 eV/atom/ps",
        consequence="computational cost doubles",
        recovery_path="check for high-frequency modes; consider rigid bond constraints",
    ),
]
