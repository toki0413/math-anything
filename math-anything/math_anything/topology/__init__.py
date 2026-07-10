"""Topology-aware loop engineering for morphism chains."""

from __future__ import annotations

from .classifier import LoopClassifier
from .loop import Loop, LoopType
from .loop_engine import LoopEngine

__all__ = ["Loop", "LoopEngine", "LoopClassifier", "LoopType"]
