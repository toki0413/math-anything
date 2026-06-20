"""结构化日志。

JSON 行式输出，支持 trace_id 和性能计时。
生产环境下可配置级别和输出目标。

使用:
    logger = get_logger(__name__)
    logger.extraction_start("vasp", {"ENCUT": 520})
    logger.morphism_applied("born_oppenheimer", [...], [...])
    logger.graph_query("ENCUT", 2, 45, 12.3)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

# ── Trace ID ──

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def set_trace_id(trace_id: str) -> None:
    _trace_id.set(trace_id)


def get_trace_id() -> str:
    return _trace_id.get()


# ── 自定义日志级别 ──

TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class StructuredLogger(logging.Logger):
    """输出 JSON 行式日志."""

    def __init__(self, name: str, level: int = logging.INFO):
        super().__init__(name, level)
        self._timers: dict[str, float] = {}

    # ── 计时 ──
    def start_timer(self, name: str) -> None:
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        start = self._timers.pop(name, 0.0)
        elapsed = time.perf_counter() - start
        self._emit(logging.INFO, f"Timer '{name}' completed in {elapsed:.3f}s", {})
        return elapsed

    # ── 语义方法 ──

    def extraction_start(self, engine: str, params: dict[str, Any]) -> None:
        self._emit(
            logging.INFO,
            f"Extraction started: {engine}",
            {
                "event": "extraction_start",
                "engine": engine,
                "params_count": len(params),
            },
        )

    def extraction_done(self, engine: str, structure: str, elapsed_ms: float) -> None:
        self._emit(
            logging.INFO,
            f"Extraction done: {engine} → {structure}",
            {
                "event": "extraction_done",
                "engine": engine,
                "structure": structure,
                "elapsed_ms": elapsed_ms,
            },
        )

    def morphism_applied(
        self,
        name: str,
        source: str,
        target: str,
        kept: list[str],
        lost: list[str],
    ) -> None:
        self._emit(
            logging.DEBUG,
            f"Morphism applied: {name}",
            {
                "event": "morphism_applied",
                "morphism": name,
                "source": source,
                "target": target,
                "kept": kept,
                "lost": lost,
            },
        )

    def pi_groups_computed(
        self,
        count: int,
        groups: list[str],
        elapsed_ms: float,
    ) -> None:
        self._emit(
            logging.INFO,
            f"Buckingham pi: {count} groups",
            {
                "event": "pi_groups_computed",
                "group_count": count,
                "groups": groups,
                "elapsed_ms": elapsed_ms,
            },
        )

    def graph_query(
        self,
        seed: str,
        depth: int,
        nodes_found: int,
        elapsed_ms: float,
    ) -> None:
        self._emit(
            logging.INFO,
            f"Graph query: '{seed[:40]}' → {nodes_found} nodes",
            {
                "event": "graph_query",
                "seed": seed[:100],
                "depth": depth,
                "nodes_found": nodes_found,
                "elapsed_ms": elapsed_ms,
            },
        )

    def invariant_derived(
        self,
        structure: str,
        invariant_count: int,
        derivation_source: str,
    ) -> None:
        self._emit(
            logging.DEBUG,
            f"Invariants derived: {structure}",
            {
                "event": "invariant_derived",
                "structure": structure,
                "invariant_count": invariant_count,
                "source": derivation_source,
            },
        )

    def plugin_loaded(self, name: str, version: str) -> None:
        self._emit(
            logging.INFO,
            f"Plugin loaded: {name} v{version}",
            {
                "event": "plugin_loaded",
                "plugin": name,
                "version": version,
            },
        )

    def kg_built(self, nodes: int, edges: int, source: str) -> None:
        self._emit(
            logging.INFO,
            f"KG built from {source}: {nodes}n/{edges}e",
            {
                "event": "kg_built",
                "nodes": nodes,
                "edges": edges,
                "source": source,
            },
        )

    # ── 内部 ──

    def _emit(self, level: int, msg: str, context: dict[str, Any]) -> None:
        if not self.isEnabledFor(level):
            return
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "level": logging.getLevelName(level),
            "logger": self.name,
            "trace_id": get_trace_id(),
            "message": msg,
            **context,
        }
        self.log(level, json.dumps(record, ensure_ascii=False))


# ── 全局 logger ──

_logger: StructuredLogger | None = None


def get_logger(name: str = "math_anything") -> StructuredLogger:
    global _logger
    if _logger is None:
        _logger = StructuredLogger(name)
        _logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)
    return _logger


def set_level(level: int) -> None:
    get_logger().setLevel(level)


def enable_debug() -> None:
    set_level(logging.DEBUG)


def disable() -> None:
    logging.disable(logging.CRITICAL)
