"""Token 预算管理器 — 控制和优化 LLM token 消耗.

策略：
  1. Token 预算分配：每次请求分配 token 预算
  2. Prompt 分级：关键/标准/可选三级，预算不足时裁剪可选内容
  3. 响应缓存：相同 prompt 不重复调用
  4. 批量合并：多个小请求合并为一个大请求
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class PromptPriority(IntEnum):
    """Prompt 优先级."""

    CRITICAL = 0  # 必须发送（核心查询）
    STANDARD = 1  # 标准内容（上下文、示例）
    OPTIONAL = 2  # 可选内容（额外信息、格式说明）


@dataclass
class TokenUsage:
    """Token 使用记录."""

    timestamp: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    priority: PromptPriority = PromptPriority.STANDARD

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TokenBudget:
    """Token 预算."""

    total_budget: int = 100000  # 总预算
    used: int = 0
    reserved: int = 10000  # 预留紧急预算

    @property
    def available(self) -> int:
        return max(0, self.total_budget - self.used - self.reserved)

    @property
    def usage_percent(self) -> float:
        if self.total_budget == 0:
            return 0.0
        return self.used / self.total_budget * 100

    def allocate(self, tokens: int, priority: PromptPriority = PromptPriority.STANDARD) -> bool:
        """分配 token 预算.

        Returns:
            True 如果预算足够，False 如果预算不足
        """
        if priority == PromptPriority.CRITICAL:
            # 关键请求可以使用预留预算
            if self.used + tokens <= self.total_budget:
                self.used += tokens
                return True
            return False

        if tokens <= self.available:
            self.used += tokens
            return True
        return False


class TokenBudgetManager:
    """Token 预算管理器."""

    def __init__(self, daily_budget: int = 1000000):
        self._budget = TokenBudget(total_budget=daily_budget)
        self._usage_history: list[TokenUsage] = []
        self._cache_savings: int = 0

    def request_tokens(
        self, estimated_tokens: int, priority: PromptPriority = PromptPriority.STANDARD
    ) -> dict[str, Any]:
        """请求 token 分配.

        Returns:
            {"approved": bool, "allocated": int, "remaining": int, "suggestion": str}
        """
        approved = self._budget.allocate(estimated_tokens, priority)

        suggestion = ""
        if not approved:
            if priority == PromptPriority.OPTIONAL:
                suggestion = "Optional content skipped to save tokens"
            elif priority == PromptPriority.STANDARD:
                suggestion = "Consider using cache or reducing context length"
            else:
                suggestion = "Critical request: using reserved budget"

        return {
            "approved": approved,
            "allocated": estimated_tokens if approved else 0,
            "remaining": self._budget.available,
            "suggestion": suggestion,
        }

    def record_usage(self, prompt_tokens: int, completion_tokens: int, cached: bool = False) -> None:
        """记录 token 使用."""
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached=cached,
        )
        self._usage_history.append(usage)

        if cached:
            self._cache_savings += prompt_tokens + completion_tokens

    @property
    def stats(self) -> dict[str, Any]:
        """Token 使用统计."""
        total_prompt = sum(u.prompt_tokens for u in self._usage_history)
        total_completion = sum(u.completion_tokens for u in self._usage_history)
        cache_hits = sum(1 for u in self._usage_history if u.cached)
        total_requests = len(self._usage_history)

        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / total_requests if total_requests > 0 else 0.0,
            "cache_savings_tokens": self._cache_savings,
            "budget_used": self._budget.used,
            "budget_total": self._budget.total_budget,
            "budget_remaining": self._budget.available,
            "budget_usage_percent": self._budget.usage_percent,
        }

    def reset_daily(self) -> None:
        """重置每日预算."""
        self._budget = TokenBudget(total_budget=self._budget.total_budget)
        self._usage_history.clear()
        self._cache_savings = 0
