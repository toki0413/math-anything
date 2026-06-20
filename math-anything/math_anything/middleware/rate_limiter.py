"""API 限流中间件.

令牌桶算法实现，支持：
  - 按用户/IP 限流
  - 可配置的速率和容量
  - 滑动窗口统计
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class TokenBucket:
    """令牌桶."""

    capacity: int  # 桶容量
    refill_rate: float  # 每秒补充令牌数
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def consume(self, tokens: int = 1) -> bool:
        """尝试消费令牌.

        Returns:
            True 如果成功消费，False 如果令牌不足
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """补充令牌."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    @property
    def available(self) -> float:
        """当前可用令牌数."""
        self._refill()
        return self.tokens


class RateLimiter:
    """API 限流器.

    支持按客户端 ID 限流，使用令牌桶算法。
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_capacity: int = 20,
    ):
        self._rps = requests_per_second
        self._capacity = burst_capacity
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()
        self._stats = {"total_requests": 0, "rejected": 0}

    def _get_bucket(self, client_id: str) -> TokenBucket:
        """获取或创建客户端的令牌桶."""
        with self._lock:
            if client_id not in self._buckets:
                self._buckets[client_id] = TokenBucket(
                    capacity=self._capacity,
                    refill_rate=self._rps,
                )
            return self._buckets[client_id]

    def allow(self, client_id: str = "default", tokens: int = 1) -> bool:
        """检查请求是否允许.

        Args:
            client_id: 客户端标识（IP/用户ID/API key）
            tokens: 消耗的令牌数

        Returns:
            True 如果允许，False 如果被限流
        """
        self._stats["total_requests"] += 1
        bucket = self._get_bucket(client_id)
        if bucket.consume(tokens):
            return True
        self._stats["rejected"] += 1
        return False

    @property
    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "active_clients": len(self._buckets),
            "rejection_rate": (
                self._stats["rejected"] / self._stats["total_requests"] if self._stats["total_requests"] > 0 else 0.0
            ),
        }

    def cleanup(self, max_idle_seconds: float = 3600) -> int:
        """清理空闲的令牌桶."""
        now = time.time()
        to_remove = [cid for cid, bucket in self._buckets.items() if now - bucket.last_refill > max_idle_seconds]
        for cid in to_remove:
            del self._buckets[cid]
        return len(to_remove)
