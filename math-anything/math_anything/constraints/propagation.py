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
from dataclasses import dataclass, field

from math_anything.morphisms import Morphism
from math_anything.rust_bridge import EMLAccelerator
from math_anything.utils.safe_eval import SafeEvalError, safe_eval

from .invariant import (
    LearnedInvariant,
    PropagationOutcome,
    WeakeningRule,
)

_accel = EMLAccelerator()
_logger = logging.getLogger(__name__)
_logger.info(f"Constraint propagation backend: {'Rust' if _accel.using_rust else 'Python'}")


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
    # results[i][j] = 第 i 个不变量穿过第 j 个态射的结果

    @property
    def final_state(self) -> dict[str, PropagationOutcome]:
        """每个不变量的最终状态."""
        final: dict[str, PropagationOutcome] = {}
        for i, inv in enumerate(self.invariants):
            if not self.results or i >= len(self.results):
                final[inv.name] = PropagationOutcome.PRESERVED
                continue
            outcomes = [r.outcome for r in self.results[i]]
            # 优先级：LOST > CONDITIONAL > WEAKENED > PRESERVED
            for outcome in [
                PropagationOutcome.LOST,
                PropagationOutcome.CONDITIONAL,
                PropagationOutcome.WEAKENED,
                PropagationOutcome.PRESERVED,
            ]:
                if outcome in outcomes:
                    final[inv.name] = outcome
                    break
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
    ) -> PropagationResult:
        """传播单个不变量穿过单个态射."""
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
            # 检查是否需要弱化
            for rule in invariant.weakening_rules:
                try:
                    if safe_eval(rule.trigger_condition, {}):
                        invariant.weaken()
                        result.outcome = PropagationOutcome.WEAKENED
                        result.applied_weakening = rule
                        invariant.record_propagation(m.name, f"WEAKENED: {rule.name}")
                        break
                except (SafeEvalError, Exception):
                    continue
            else:
                result.outcome = PropagationOutcome.PRESERVED
                invariant.record_propagation(m.name, "PRESERVED")

        elif invariant.name in m.invariants_introduced:
            result.outcome = PropagationOutcome.EMERGED
            invariant.record_propagation(m.name, "EMERGED")

        else:
            # 隐式：检查域条件是否被态射影响
            if m.kernel_description and any(dc.feature in m.kernel_description for dc in invariant.domain_conditions):
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
    ) -> PropagationChain:
        """传播不变量集穿过完整的态射链.

        Args:
            invariants: 初始不变量集
            morphisms: 按序排列的态射（先应用 morphisms[0]，再 morphisms[1]，...）
            source_labels: 每个态射的源结构标签
            target_labels: 每个态射的目标结构标签

        Returns:
            PropagationChain 包含所有传播结果
        """
        chain_names = [m.name for m in morphisms]
        results: list[list[PropagationResult]] = []
        current_invariants = [inv for inv in invariants]  # 浅拷贝

        # 尝试 Rust 加速的批量传播路径
        # Rust 路径只处理简单的 kept/lost/introduced 语义，
        # 弱化规则和域条件检查仍需 Python 回退
        if _accel.using_rust and len(invariants) > 10 and len(morphisms) > 3:
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
                rust_outcomes = _accel.propagate_constraints(inv_names, morphism_data)  # type: ignore[call-arg]
                _logger.debug(
                    f"Constraint propagation: Rust batch path, "
                    f"{len(invariants)} invariants x {len(morphisms)} morphisms"
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
                                outcome = PropagationOutcome[outcome_str]
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
            except (ValueError, TypeError, RuntimeError):
                _logger.debug("Rust batch propagation failed, falling back to Python")

        # Python 回退路径: 逐个传播
        for i, morph in enumerate(morphisms):
            step_results: list[PropagationResult] = []  # type: ignore[no-redef]
            src = source_labels[i] if i < len(source_labels) else "?"
            dst = target_labels[i] if i < len(target_labels) else "?"

            for inv in current_invariants:
                result = self.propagate_single(inv, morph, src, dst)
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
