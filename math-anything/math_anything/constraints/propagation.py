"""约束传播：不变量在态射链上的传递。

核心操作：给定态射 f: S → T，将源结构的约束集传播到目标结构。

传播不是二元的"保持/丢失"，而是：

  PRESERVED:  不变，完全传递
  WEAKENED:   弱化（如确定性→概率性保证）
  CONDITIONAL: 变为条件（需要额外前提才成立）
  LOST:       完全丢失
  EMERGED:    态射引入了新的不变量

传播是可组合的：propagate(I, g∘f) = propagate(propagate(I, f), g)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from math_anything.morphisms import Morphism
from math_anything.rust_bridge import EMLAccelerator
from math_anything.utils.safe_eval import SafeEvalError, safe_eval

from .invariant import (
    LearnedInvariant,
    PropagationOutcome,
    WeakeningRule,
)

# 惰性初始化 Rust 加速器，避免 import 时副作用
_accel: EMLAccelerator | None = None
_logger = logging.getLogger(__name__)


def _get_accel() -> EMLAccelerator:
    """惰性获取 EMLAccelerator 单例，并在首次调用时记录后端类型."""
    global _accel
    if _accel is None:
        _accel = EMLAccelerator()
        _logger.debug(
            "Constraint propagation backend: %s",
            "Rust" if _accel.using_rust else "Python",
        )
    return _accel


def _feature_matches(feature: str, text: str) -> bool:
    """检查 feature 是否作为完整词出现在 text 中.

    用词边界正则替代裸 ``in`` 子串匹配，避免 "energy" 误匹配
    "in_energy_calc"、"in" 误匹配 "input" 这类常见 false positive。
    """
    if not feature:
        return False
    try:
        return re.search(rf"\b{re.escape(feature)}\b", text, flags=re.IGNORECASE) is not None
    except (re.error, TypeError):
        # fallback：feature 不含特殊字符时退化为大小写不敏感子串
        return feature.lower() in str(text).lower()


@dataclass
class PropagationResult:
    """单个不变量在穿过一个态射后的传播结果."""

    invariant: LearnedInvariant
    outcome: PropagationOutcome
    morphism_name: str
    source_structure: str
    target_structure: str

    # 如果丢失：为什么丢失
    loss_reason: str = ""

    # 如果弱化：弱化到什么程度
    applied_weakening: WeakeningRule | None = None

    # 如果变为条件：新条件是什么
    new_condition: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "invariant": self.invariant.name,
            "outcome": self.outcome.value,
            "morphism": self.morphism_name,
            "source": self.source_structure,
            "target": self.target_structure,
            "loss_reason": self.loss_reason,
            "new_condition": self.new_condition,
        }


@dataclass
class PropagationChain:
    """不变量在完整态射链上的累积传播.

    从 f_1 ∘ f_2 ∘ ... ∘ f_n 的首端到尾端追踪每个不变量。
    """

    invariants: list[LearnedInvariant]
    chain: list[str]  # 态射名称列表
    results: list[list[PropagationResult]] = field(default_factory=list)
    # results[i] = 第 i 个态射步骤下，当时仍存活的全部不变量的传播结果
    # （注意：每步的 current_invariants 会因 lost/introduced 而变化，
    # 因此 results[i][j] 的 j 并不全局对应同一个不变量。
    # 查询某不变量的最终状态时必须按 invariant.name 跨步骤聚合，见 final_state。）

    # 优先级：LOST > CONDITIONAL > WEAKENED > PRESERVED > EMERGED
    _OUTCOME_PRIORITY: ClassVar[list[PropagationOutcome]] = [
        PropagationOutcome.LOST,
        PropagationOutcome.CONDITIONAL,
        PropagationOutcome.WEAKENED,
        PropagationOutcome.PRESERVED,
        PropagationOutcome.EMERGED,
    ]

    @property
    def final_state(self) -> dict[str, PropagationOutcome]:
        """每个不变量在态射链终点的累积状态.

        按 invariant.name 跨所有步骤聚合结果，取优先级最高的 outcome。
        初始不变量若在某步丢失则记为 LOST；新出现的不变量也会出现在结果中。
        """
        priority_rank = {o: i for i, o in enumerate(self._OUTCOME_PRIORITY)}
        final: dict[str, PropagationOutcome] = {
            inv.name: PropagationOutcome.PRESERVED for inv in self.invariants
        }
        for step_results in self.results:
            for r in step_results:
                name = r.invariant.name
                current = final.get(name, r.outcome)
                # 取优先级更高（rank 更小）的 outcome
                if priority_rank[r.outcome] < priority_rank[current]:
                    final[name] = r.outcome
                else:
                    final[name] = current
        return final

    @property
    def preserved_invariants(self) -> list[LearnedInvariant]:
        fs = self.final_state
        return [inv for inv in self.invariants if fs.get(inv.name) == PropagationOutcome.PRESERVED]

    @property
    def weakened_invariants(self) -> list[LearnedInvariant]:
        fs = self.final_state
        return [inv for inv in self.invariants if fs.get(inv.name) == PropagationOutcome.WEAKENED]

    @property
    def lost_invariants(self) -> list[LearnedInvariant]:
        fs = self.final_state
        return [inv for inv in self.invariants if fs.get(inv.name) == PropagationOutcome.LOST]


class ConstraintPropagation:
    """约束传播引擎.

    核心不变性:

    propagate(I, g∘f) = propagate(propagate(I, f), g)

    这保证了态射链上的约束传递是良定义的。

    传播语义:

    - 如果态射 f 在其 invariants_kept 中声明了不变量 inv.name → PRESERVED
    - 如果态射 f 在其 invariants_lost 中声明了不变量 inv.name → LOST
    - 如果 inv 的域条件涉及被 f 改变的参数 → CONDITIONAL
    - 如果 inv 有弱化规则且触发条件满足 → WEAKENED
    - 新出现的不变量（f 的 invariants_introduced）→ EMERGED
    """

    def propagate_single(
        self,
        invariant: LearnedInvariant,
        morphism: Morphism,
        source: str,
        target: str,
        params: dict[str, Any] | None = None,
    ) -> PropagationResult:
        """传播单个不变量穿过单个态射.

        Args:
            invariant: 待传播的不变量
            morphism: 当前态射
            source/target: 源/目标结构标签
            params: 当前参数上下文，用于弱化规则触发条件的求值。
                若为 None，弱化规则的 trigger_condition 将无法引用任何变量，
                通常只能匹配纯常量表达式。
        """
        m = morphism
        result = PropagationResult(
            invariant=invariant,
            outcome=PropagationOutcome.PRESERVED,
            morphism_name=m.name,
            source_structure=source,
            target_structure=target,
        )

        # 检查显式声明
        if invariant.name in m.invariants_lost:
            result.outcome = PropagationOutcome.LOST
            result.loss_reason = f"Explicitly lost in morphism {m.name}"
            invariant.record_propagation(m.name, "LOST")

        elif invariant.name in m.invariants_kept:
            # 检查是否需要弱化：用真实参数上下文求值 trigger_condition
            eval_ctx = dict(params) if params else {}
            for rule in invariant.weakening_rules:
                try:
                    if safe_eval(rule.trigger_condition, eval_ctx):
                        invariant.weaken()
                        result.outcome = PropagationOutcome.WEAKENED
                        result.applied_weakening = rule
                        invariant.record_propagation(m.name, f"WEAKENED: {rule.name}")
                        break
                except (SafeEvalError, NameError, KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                    _logger.debug(
                        "Weakening rule '%s' trigger eval failed on invariant '%s': %s",
                        rule.name, invariant.name, e,
                    )
                    continue
            else:
                result.outcome = PropagationOutcome.PRESERVED
                invariant.record_propagation(m.name, "PRESERVED")

        elif invariant.name in m.invariants_introduced:
            result.outcome = PropagationOutcome.EMERGED
            invariant.record_propagation(m.name, "EMERGED")

        else:
            # 隐式：检查域条件是否被态射影响
            # 使用词边界匹配避免子串误判（如 "energy" 误匹配 "in_energy_calc"）
            if m.kernel_description and any(
                _feature_matches(dc.feature, m.kernel_description)
                for dc in invariant.domain_conditions
            ):
                result.outcome = PropagationOutcome.CONDITIONAL
                result.new_condition = f"Requires {m.kernel_description} to be negligible"
                invariant.record_propagation(m.name, "CONDITIONAL")
            else:
                result.outcome = PropagationOutcome.PRESERVED
                invariant.record_propagation(m.name, "PRESERVED")

        return result

    def propagate_chain(
        self,
        invariants: list[LearnedInvariant],
        morphisms: list[Morphism],
        source_labels: list[str],
        target_labels: list[str],
        params: dict[str, Any] | None = None,
    ) -> PropagationChain:
        """传播不变量集穿过完整的态射链.

        Args:
            invariants: 初始不变量集
            morphisms: 按序排列的态射（先应用 morphisms[0]，再 morphisms[1]，...）
            source_labels: 每个态射的源结构标签
            target_labels: 每个态射的目标结构标签
            params: 当前参数上下文，用于弱化规则触发条件的求值。

        Returns:
            PropagationChain 包含所有传播结果
        """
        chain_names = [m.name for m in morphisms]
        results: list[list[PropagationResult]] = []
        current_invariants = [inv for inv in invariants]  # 浅拷贝

        # 尝试 Rust 加速的批量传播路径
        # Rust 路径只处理简单的 kept/lost/introduced 语义，
        # 弱化规则和域条件检查仍需 Python 回退
        accel = _get_accel()
        if accel.using_rust and len(invariants) > 10 and len(morphisms) > 3:
            try:
                morphism_data = [
                    {
                        "name": m.name,
                        "kept": list(m.invariants_kept),
                        "lost": list(m.invariants_lost),
                        "introduced": list(m.invariants_introduced),
                        "kernel_desc": getattr(m, "kernel_description", "") or "",
                    }
                    for m in morphisms
                ]
                inv_names = [inv.name for inv in current_invariants]
                rust_outcomes = accel.propagate_constraints(inv_names, morphism_data)  # type: ignore[call-arg]
                _logger.debug(
                    "Constraint propagation: Rust batch path, "
                    "%d invariants x %d morphisms",
                    len(invariants), len(morphisms),
                )

                # 将 Rust 结果转换为 PropagationResult
                for i, morph in enumerate(morphisms):
                    src = source_labels[i] if i < len(source_labels) else "?"
                    dst = target_labels[i] if i < len(target_labels) else "?"
                    step_results: list[PropagationResult] = []

                    if i < len(rust_outcomes):
                        for j, inv in enumerate(current_invariants):
                            if j < len(rust_outcomes[i]):
                                outcome_str = rust_outcomes[i][j]
                                # 兼容 Rust 端可能返回的大小写/拼写差异
                                try:
                                    outcome = PropagationOutcome(outcome_str)
                                except (KeyError, ValueError):
                                    try:
                                        outcome = PropagationOutcome[outcome_str]
                                    except KeyError:
                                        _logger.warning(
                                            "Rust returned unknown outcome '%s' "
                                            "for invariant '%s' at morphism '%s'; "
                                            "defaulting to PRESERVED",
                                            outcome_str, inv.name, morph.name,
                                        )
                                        outcome = PropagationOutcome.PRESERVED
                            else:
                                outcome = PropagationOutcome.PRESERVED
                            step_results.append(
                                PropagationResult(
                                    invariant=inv,
                                    outcome=outcome,
                                    morphism_name=morph.name,
                                    source_structure=src,
                                    target_structure=dst,
                                )
                            )
                    results.append(step_results)

                # 更新 current_invariants: 丢弃丢失的，添加新出现的
                for morph in morphisms:
                    current_invariants = [inv for inv in current_invariants if inv.name not in morph.invariants_lost]
                    for new_name in morph.invariants_introduced:
                        new_inv = LearnedInvariant(
                            name=new_name,
                            expression=new_name,
                            severity="heuristic",
                        )
                        new_inv.record_propagation(morph.name, "EMERGED")
                        current_invariants.append(new_inv)

                chain = PropagationChain(
                    invariants=list(invariants),
                    chain=chain_names,
                    results=results,
                )
                return chain
            except (ValueError, TypeError, RuntimeError, KeyError) as e:
                _logger.debug("Rust batch propagation failed, falling back to Python: %s", e)

        # Python 回退路径: 逐个传播
        for i, morph in enumerate(morphisms):
            step_results: list[PropagationResult] = []  # type: ignore[no-redef]
            src = source_labels[i] if i < len(source_labels) else "?"
            dst = target_labels[i] if i < len(target_labels) else "?"

            for inv in current_invariants:
                result = self.propagate_single(inv, morph, src, dst, params=params)
                step_results.append(result)

            results.append(step_results)

            # 丢弃完全丢失的不变量
            current_invariants = [inv for inv in current_invariants if inv.name not in morph.invariants_lost]

            # 添加新出现的不变量
            for new_name in morph.invariants_introduced:
                new_inv = LearnedInvariant(
                    name=new_name,
                    expression=new_name,
                    severity="heuristic",
                )
                new_inv.record_propagation(morph.name, "EMERGED")
                current_invariants.append(new_inv)

        chain = PropagationChain(
            invariants=list(invariants),
            chain=chain_names,
            results=results,
        )
        return chain

    def compose_propagation(
        self,
        r1: PropagationResult,
        r2: PropagationResult,
    ) -> PropagationOutcome:
        """组合两个传播结果（g 的结果组合 f 的结果）.

        用于验证 propagate(I, g∘f) = propagate(propagate(I, f), g)
        """
        states = [r1.outcome, r2.outcome]
        if PropagationOutcome.LOST in states:
            return PropagationOutcome.LOST
        if PropagationOutcome.CONDITIONAL in states:
            return PropagationOutcome.CONDITIONAL
        if PropagationOutcome.WEAKENED in states:
            return PropagationOutcome.WEAKENED
        if PropagationOutcome.EMERGED in states:
            return PropagationOutcome.EMERGED
        return PropagationOutcome.PRESERVED
