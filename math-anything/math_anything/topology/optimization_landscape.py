"""Optimization-landscape homotopy for training trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from math_anything.categories.engine import CategoryEngine
from math_anything.topology.homotopy import HomotopyWitness, are_paths_homotopic
from math_anything.topology.training_curvature import TrainingResult


@dataclass
class TrainingPathMorphism:
    """One epoch-to-epoch step of a training trajectory as a morphism."""

    name: str
    source_structure: str
    target_structure: str
    invariants_kept: list[str]
    invariants_lost: list[str]

    def get_invariants_lost(self) -> list[str]:
        return self.invariants_lost


def _derive_invariants(
    prev_state: Any,
    next_state: Any,
) -> tuple[list[str], list[str]]:
    """Derive qualitative invariants from two adjacent optimization states."""
    kept: list[str] = ["parameter_space"]
    lost: list[str] = []

    if next_state.loss < prev_state.loss:
        kept.append("loss_decreased")
    else:
        lost.append("monotonic_loss_decrease")

    prev_w = np.asarray(prev_state.weights, dtype=float)
    next_w = np.asarray(next_state.weights, dtype=float)
    update_norm = float(np.linalg.norm(next_w - prev_w))
    if update_norm > 1e-12:
        kept.append("non_stationary_step")
    else:
        kept.append("stationary_step")

    return kept, lost


def build_training_path(
    result: TrainingResult,
    name_prefix: str = "run",
    source_structure: str = "params_initial",
    target_structure: str = "params_final",
) -> tuple[CategoryEngine, list[str]]:
    """Convert a TrainingResult into a CategoryEngine path.

    Returns the engine and the ordered list of morphism names. The source and
    target structures are intentionally generic so two different training runs
    can be compared as paths between the same endpoints.
    """
    engine = CategoryEngine()
    path: list[str] = []

    states = result.states
    if not states:
        return engine, path

    for i in range(len(states) - 1):
        prev_state = states[i]
        next_state = states[i + 1]
        name = f"{name_prefix}_step_{i}"
        kept, lost = _derive_invariants(prev_state, next_state)
        morphism = TrainingPathMorphism(
            name=name,
            source_structure=f"{name_prefix}_state_{i}",
            target_structure=f"{name_prefix}_state_{i + 1}",
            invariants_kept=kept,
            invariants_lost=lost,
        )
        engine.register_morphism(morphism)
        engine.link(
            name,
            morphism.source_structure,
            morphism.target_structure,
        )
        path.append(name)

    if path:
        # Anchor the path to the shared source structure without creating a
        # duplicate link (which would overwrite the original target in the
        # morphism-name-indexed map used by ``are_paths_homotopic``).
        engine.morphism_links[0].source_structure = source_structure

        terminal_name = f"{name_prefix}_terminal"
        terminal = TrainingPathMorphism(
            name=terminal_name,
            source_structure=engine.morphism_links[-1].target_structure,
            target_structure=target_structure,
            invariants_kept=[],
            invariants_lost=[],
        )
        engine.register_morphism(terminal)
        engine.link(terminal_name, terminal.source_structure, target_structure)
        path.append(terminal_name)

    return engine, path


def training_paths_homotopic(
    result_a: TrainingResult,
    result_b: TrainingResult,
) -> HomotopyWitness:
    """Check whether two training trajectories are homotopic in parameter-loss space."""
    engine_a, path_a = build_training_path(result_a, name_prefix="run_a")
    engine_b, path_b = build_training_path(result_b, name_prefix="run_b")

    merged = CategoryEngine()
    merged.morphisms.update(engine_a.morphisms)
    merged.morphism_links.extend(engine_a.morphism_links)
    merged.morphisms.update(engine_b.morphisms)
    merged.morphism_links.extend(engine_b.morphism_links)

    return are_paths_homotopic(merged, path_a, path_b)
