"""Topology-aware loop engineering for morphism chains."""

from __future__ import annotations

from .classifier import LoopClassifier
from .curvature import discrete_curvature, holonomy, riemannian_curvature_bridge
from .homotopy import HomotopyWitness, are_paths_homotopic, cumulative_invariants_along_path
from .loop import Loop, LoopType
from .loop_engine import LoopEngine
from .training_curvature import OptimizationState, trajectory_curvature

__all__ = [
    "Loop",
    "LoopEngine",
    "LoopClassifier",
    "LoopType",
    "HomotopyWitness",
    "are_paths_homotopic",
    "cumulative_invariants_along_path",
    "holonomy",
    "discrete_curvature",
    "riemannian_curvature_bridge",
    "OptimizationState",
    "trajectory_curvature",
]
