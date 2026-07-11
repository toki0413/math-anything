"""Curvature of optimization trajectories in parameter-loss space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OptimizationState:
    """One point in an optimization trajectory."""

    step: int
    loss: float
    weights: Any


def trajectory_curvature(states: list[OptimizationState]) -> list[float]:
    """Compute discrete curvature at each interior point of a training trajectory.

    Treats the trajectory as a polygonal curve in (weights, loss) space and
    returns the angle-change normalized curvature at each interior vertex.
    """
    if len(states) < 3:
        return []

    curvatures: list[float] = []
    for i in range(1, len(states) - 1):
        prev_w = np.asarray(states[i - 1].weights, dtype=float)
        curr_w = np.asarray(states[i].weights, dtype=float)
        next_w = np.asarray(states[i + 1].weights, dtype=float)

        v1 = np.append(curr_w - prev_w, states[i].loss - states[i - 1].loss)
        v2 = np.append(next_w - curr_w, states[i + 1].loss - states[i].loss)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            curvatures.append(0.0)
            continue

        cos_angle = float(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
        angle = np.arccos(cos_angle)
        curvatures.append(float(angle / np.pi))

    return curvatures


@dataclass
class TrainingResult:
    """Result of training a tiny network, including optimization states."""

    states: list[OptimizationState]
    final_loss: float


def _flatten_weights(network: Any) -> np.ndarray:
    """Flatten all linear-layer weights and biases into a single vector."""
    from math_anything.structures.neural_network import LinearMorphism

    parts = []
    for layer in network.layers:
        if isinstance(layer, LinearMorphism):
            parts.append(layer.weight.flatten())
            parts.append(layer.bias.flatten())
    return np.concatenate(parts) if parts else np.array([])


def train_and_capture(
    network: Any,
    dataset: list[tuple[Any, Any]],
    loss_fn: Any,
    epochs: int = 10,
    lr: float = 0.05,
) -> TrainingResult:
    """Train a tiny network and capture (weights, loss) optimization states."""
    states: list[OptimizationState] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in dataset:
            y_pred = network.forward(x)
            epoch_loss += float(loss_fn.apply((y_pred, y)))
            grads = network.backward(x, y, loss_fn)
            network.sgd_step(grads, lr)

        weights = _flatten_weights(network)
        states.append(OptimizationState(step=epoch, loss=epoch_loss, weights=weights))

    final_loss = states[-1].loss if states else float("inf")
    return TrainingResult(states=states, final_loss=final_loss)


def training_result_curvature(result: TrainingResult) -> list[float]:
    """Compute optimization-trajectory curvature from a training result."""
    return trajectory_curvature(result.states)
