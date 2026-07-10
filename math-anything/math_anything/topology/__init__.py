"""Topology-aware loop engineering for morphism chains."""

from __future__ import annotations

from .loop import Loop, LoopType
from .loop_engine import LoopEngine

# Stub replaced in upcoming task.
class LoopClassifier:
    pass

__all__ = ["Loop", "LoopEngine", "LoopClassifier", "LoopType"]
