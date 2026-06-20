"""边界状态与演化。

操作边界不是预定义的刚性范围，而是从经验中学习的动态边界。

概念：
  Interior（场内）：所有不变量高置信度成立，系统稳定运行
  Boundary（边界）：部分不变量被弱化或处于风险中
  Exterior（场外）：关键不变量被违反，系统不可靠

边界演化：
  成功计算 → 边界向外扩张（expansion）
  失败/违反 → 边界向内收缩（contraction）
  新发现 → 边界形状调整（reshape）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .invariant import InvariantStatus, LearnedInvariant


@dataclass
class RiskItem:
    """一个风险评估项."""

    invariant: LearnedInvariant
    risk_score: float  # 0.0（安全）→ 1.0（危急）
    status: InvariantStatus
    recommendation: str = ""


@dataclass
class BoundaryState:
    """系统的当前操作边界.

    不是预定义的 scope="dft"，而是从经验学习的边界形状。

    Attributes:
        interior_invariants: 场内不变量列表（高置信度成立）
        boundary_invariants: 边界不变量（弱化或风险中）→ (invariant, risk_score)
        expansion_count: 成功次数（触发扩展）
        contraction_count: 失败/违反次数（触发收缩）
        total_experiences: 总计算经验数
    """

    interior_invariants: list[LearnedInvariant] = field(default_factory=list)
    boundary_invariants: list[tuple[LearnedInvariant, float]] = field(default_factory=list)
    exterior_invariants: list[LearnedInvariant] = field(default_factory=list)

    expansion_count: int = 0
    contraction_count: int = 0
    total_experiences: int = 0

    # 边界参数
    expansion_threshold: int = 5  # 连续成功多少次触发扩展
    contraction_threshold: int = 2  # 连续失败多少次触发收缩
    risk_escalation_rate: float = 1.5  # 风险升级速率

    def is_interior(self, params: dict[str, Any]) -> bool:
        """给定的参数是否在场内（Interior）."""
        for inv in self.interior_invariants:
            status = inv.evaluate(params)
            if status in (InvariantStatus.VIOLATED, InvariantStatus.INACTIVE):
                return False
        return True

    def is_boundary(self, params: dict[str, Any]) -> bool:
        """给定的参数是否在边界区域."""
        for inv, risk in self.boundary_invariants:
            status = inv.evaluate(params)
            if status == InvariantStatus.VIOLATED:
                return True
        return False

    def risk_assessment(self, params: dict[str, Any]) -> list[RiskItem]:
        """评估当前参数下的所有风险."""
        risks: list[RiskItem] = []

        for inv in self.interior_invariants:
            status = inv.evaluate(params)
            if status != InvariantStatus.SATISFIED:
                risk_score = 0.3 if status == InvariantStatus.WEAKENED else 0.7
                risks.append(
                    RiskItem(
                        invariant=inv,
                        risk_score=risk_score,
                        status=status,
                        recommendation=self._recommend_for(inv, status),
                    )
                )

        for inv, base_risk in self.boundary_invariants:
            status = inv.evaluate(params)
            risk_score = base_risk
            if status == InvariantStatus.VIOLATED:
                risk_score = min(1.0, base_risk * self.risk_escalation_rate)
            elif status == InvariantStatus.WEAKENED:
                risk_score = base_risk * 0.8
            risks.append(
                RiskItem(
                    invariant=inv,
                    risk_score=risk_score,
                    status=status,
                    recommendation=self._recommend_for(inv, status),
                )
            )

        for inv in self.exterior_invariants:
            status = inv.evaluate(params)
            if status == InvariantStatus.VIOLATED:
                risks.append(
                    RiskItem(
                        invariant=inv,
                        risk_score=0.95,
                        status=status,
                        recommendation=f"Critical: {inv.name} violated. Halt and investigate.",
                    )
                )

        return sorted(risks, key=lambda r: -r.risk_score)

    @staticmethod
    def _recommend_for(inv: LearnedInvariant, status: InvariantStatus) -> str:
        if status == InvariantStatus.VIOLATED and inv.weakening_rules:
            rule = inv.weakening_rules[0]
            return f"Consider: {rule.name} → {rule.consequence}"
        if status == InvariantStatus.WEAKENED:
            return f"Already weakened by rule {inv.active_weakening}. Monitor."
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "interior_count": len(self.interior_invariants),
            "boundary_count": len(self.boundary_invariants),
            "exterior_count": len(self.exterior_invariants),
            "expansion_count": self.expansion_count,
            "contraction_count": self.contraction_count,
            "total_experiences": self.total_experiences,
        }


@dataclass
class ExecutionRecord:
    """一次计算的执行记录."""

    success: bool
    params: dict[str, Any]
    invariant_results: list[tuple[str, InvariantStatus]] = field(default_factory=list)
    error_message: str = ""
    wall_time: float = 0.0
    resource_usage: dict[str, float] = field(default_factory=dict)


class BoundaryEvolution:
    """基于经验的边界演化引擎.

    原则：
      - 从不变量违反中学习边界位置
      - 成功扩大自信，失败触发收缩
      - 边界不是预定义的，而是从交互中涌现的
    """

    def __init__(self, state: BoundaryState | None = None):
        self.state = state or BoundaryState()
        self._consecutive_successes = 0
        self._consecutive_failures = 0

    def evolve(self, record: ExecutionRecord) -> BoundaryState:
        """根据执行记录演化边界."""
        self.state.total_experiences += 1

        if record.success:
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            self.state.expansion_count += 1

            # 成功：提升所有 tested invariants 的置信度
            for inv_name, status in record.invariant_results:
                for inv in self.state.interior_invariants:
                    if inv.name == inv_name and status == InvariantStatus.SATISFIED:
                        inv.satisfaction_count += 1
                        # 如果置信度超过阈值，可以扩展边界
                        if inv.satisfaction_count > self.state.expansion_threshold:
                            self._expand(inv)

            # 连续成功 → 将低风险边界不变量移入场内
            if self._consecutive_successes >= self.state.expansion_threshold:
                self._promote_boundary_invariants()

        else:
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self.state.contraction_count += 1

            # 失败：触发违规不变量的弱化
            for inv_name, status in record.invariant_results:
                if status == InvariantStatus.VIOLATED:
                    for inv in self.state.interior_invariants + [b[0] for b in self.state.boundary_invariants]:
                        if inv.name == inv_name:
                            inv.violation_count += 1
                            self._contract(inv)

            # 连续失败 → 收缩边界
            if self._consecutive_failures >= self.state.contraction_threshold:
                self._demote_interior_invariants()

        return self.state

    def _expand(self, invariant: LearnedInvariant) -> None:
        """扩展：增加不变量的域置信度."""
        invariant.domain_confidence = min(0.99, invariant.domain_confidence + 0.05)

    def _contract(self, invariant: LearnedInvariant) -> None:
        """收缩：降低不变量的域置信度，移动到边界."""
        invariant.domain_confidence = max(0.1, invariant.domain_confidence - 0.15)
        # 如果降至边界阈值以下，移入边界区域
        if invariant.domain_confidence < 0.5:
            if invariant in self.state.interior_invariants:
                self.state.interior_invariants.remove(invariant)
                self.state.boundary_invariants.append((invariant, 0.5))
            # 如果有弱化规则，尝试弱化
            if invariant.weakening_rules and invariant.active_weakening == 0:
                invariant.weaken(0)

    def _promote_boundary_invariants(self) -> None:
        """将低风险边界不变量提升回场内."""
        promoted = []
        remaining = []
        for inv, risk in self.state.boundary_invariants:
            if inv.violation_count == 0 and risk < 0.3:
                inv.domain_confidence = 0.7
                self.state.interior_invariants.append(inv)
                promoted.append(inv.name)
            else:
                remaining.append((inv, risk))
        self.state.boundary_invariants = remaining

    def _demote_interior_invariants(self) -> None:
        """将高违规的不变量降级."""
        demoted = []
        remaining = []
        for inv in self.state.interior_invariants:
            if inv.violation_count >= 3:
                self.state.boundary_invariants.append((inv, 0.5))
                demoted.append(inv.name)
            else:
                remaining.append(inv)
        self.state.interior_invariants = remaining
