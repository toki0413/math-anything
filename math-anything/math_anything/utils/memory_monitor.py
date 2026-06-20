"""内存监控器 — 追踪和优化内存占用.

特性：
  - 对象级内存追踪
  - 内存快照对比
  - 自动 GC 触发
  - 内存预算管理
"""

from __future__ import annotations

import gc
import sys
import time
import weakref
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemorySnapshot:
    """内存快照."""

    timestamp: float = field(default_factory=time.time)
    total_objects: int = 0
    total_size_mb: float = 0.0
    by_type: dict[str, int] = field(default_factory=dict)
    gc_collections: tuple[int, int, int] = (0, 0, 0)


class MemoryMonitor:
    """内存监控器."""

    def __init__(self, budget_mb: float = 512.0):
        self._budget_mb = budget_mb
        self._snapshots: list[MemorySnapshot] = []
        self._tracked_objects: dict[str, weakref.ref] = {}

    def snapshot(self) -> MemorySnapshot:
        """获取当前内存快照."""
        type_counts: dict[str, int] = {}
        total_size = 0
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            try:
                total_size += sys.getsizeof(obj)
            except (TypeError, ValueError):
                pass

        snap = MemorySnapshot(
            total_objects=len(gc.get_objects()),
            total_size_mb=total_size / (1024 * 1024),
            by_type=dict(sorted(type_counts.items(), key=lambda x: -x[1])[:20]),
            gc_collections=gc.get_count(),
        )
        self._snapshots.append(snap)
        return snap

    def check_budget(self) -> dict[str, Any]:
        """检查内存预算."""
        snap = self.snapshot()
        usage_pct = snap.total_size_mb / self._budget_mb * 100
        return {
            "used_mb": snap.total_size_mb,
            "budget_mb": self._budget_mb,
            "usage_percent": usage_pct,
            "over_budget": snap.total_size_mb > self._budget_mb,
            "gc_collections": snap.gc_collections,
            "top_types": snap.by_type,
        }

    def force_gc(self) -> int:
        """强制 GC，返回回收的对象数."""
        before = len(gc.get_objects())
        gc.collect()
        after = len(gc.get_objects())
        return before - after

    @property
    def snapshots(self) -> list[MemorySnapshot]:
        return self._snapshots
