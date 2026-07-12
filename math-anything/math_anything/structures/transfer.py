"""Weight-space transfer learning as a concrete natural transformation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from math_anything.structures.neural_network import LinearMorphism
from math_anything.topology.training_curvature import train_and_capture


def flatten_network_weights(network: Any) -> np.ndarray:
    """Flatten all linear-layer weights and biases into one vector."""
    parts = []
    for layer in network.layers:
        if isinstance(layer, LinearMorphism):
            parts.append(layer.weight.flatten())
            parts.append(layer.bias.flatten())
    return np.concatenate(parts) if parts else np.array([])


def set_network_weights(network: Any, weights: np.ndarray) -> None:
    """Set linear-layer weights and biases from a flattened vector."""
    weights = np.asarray(weights, dtype=float)
    offset = 0
    for layer in network.layers:
        if isinstance(layer, LinearMorphism):
            w_size = layer.weight.size
            layer.weight[:] = weights[offset : offset + w_size].reshape(layer.weight.shape)
            offset += w_size
            b_size = layer.bias.size
            layer.bias[:] = weights[offset : offset + b_size]
            offset += b_size


def transfer_weights(source_weights: np.ndarray, adapter_matrix: np.ndarray) -> np.ndarray:
    """Map a source weight vector into a target weight space."""
    return np.asarray(adapter_matrix, dtype=float) @ np.asarray(source_weights, dtype=float)


@dataclass
class WeightSpaceTransfer:
    """A linear adapter between two weight spaces."""

    source_dim: int
    target_dim: int
    matrix: np.ndarray

    def __init__(self, source_dim: int, target_dim: int, matrix: np.ndarray | None = None):
        self.source_dim = source_dim
        self.target_dim = target_dim
        if matrix is None:
            rng = np.random.default_rng(0)
            self.matrix = rng.standard_normal((target_dim, source_dim)) * 0.1
        else:
            self.matrix = np.asarray(matrix, dtype=float)
            if self.matrix.shape != (target_dim, source_dim):
                raise ValueError(
                    f"Expected adapter shape {(target_dim, source_dim)}, got {self.matrix.shape}"
                )


def transfer_learn(
    source_network: Any,
    target_network: Any,
    dataset: list[tuple[Any, Any]],
    loss_fn: Any,
    adapter_matrix: np.ndarray,
    epochs: int = 5,
    lr: float = 0.05,
) -> Any:
    """Train source, transfer weights to target via adapter, then train target."""
    # Train source
    train_and_capture(source_network, dataset, loss_fn, epochs=epochs, lr=lr)
    source_weights = flatten_network_weights(source_network)

    # Initialize target from transferred weights
    target_weights = transfer_weights(source_weights, adapter_matrix)
    set_network_weights(target_network, target_weights)

    # Train target
    return train_and_capture(target_network, dataset, loss_fn, epochs=epochs, lr=lr)
