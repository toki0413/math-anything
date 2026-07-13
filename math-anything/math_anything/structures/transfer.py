"""Weight-space transfer learning as a concrete natural transformation."""

from __future__ import annotations

from typing import Any

import numpy as np

from math_anything.structures.neural_network import LinearMorphism
from math_anything.topology.training_curvature import train_and_capture


def flatten_network_weights(network: Any) -> list[float]:
    """Flatten all linear-layer weights and biases into one serializable vector."""
    parts = []
    for layer in network.layers:
        if isinstance(layer, LinearMorphism):
            parts.append(layer.weight.flatten())
            parts.append(layer.bias.flatten())
    return np.concatenate(parts).tolist() if parts else []


def set_network_weights(network: Any, weights: list[float]) -> None:
    """Set linear-layer weights and biases from a flattened vector."""
    weights_arr = np.asarray(weights, dtype=float)
    expected = sum(layer.weight.size + layer.bias.size for layer in network.layers if isinstance(layer, LinearMorphism))
    if weights_arr.size != expected:
        raise ValueError(f"Expected {expected} weights, got {weights_arr.size}")
    offset = 0
    for layer in network.layers:
        if isinstance(layer, LinearMorphism):
            w_size = layer.weight.size
            layer.weight[:] = weights_arr[offset : offset + w_size].reshape(layer.weight.shape)
            offset += w_size
            b_size = layer.bias.size
            layer.bias[:] = weights_arr[offset : offset + b_size]
            offset += b_size


def transfer_weights(source_weights: list[float], adapter_matrix: list[list[float]]) -> list[float]:
    """Map a source weight vector into a target weight space."""
    result = np.asarray(adapter_matrix, dtype=float) @ np.asarray(source_weights, dtype=float)
    return result.tolist()  # type: ignore[no-any-return]


class WeightSpaceTransfer:
    """A linear adapter between two weight spaces."""

    def __init__(self, source_dim: int, target_dim: int, matrix: list[list[float]] | None = None):
        self.source_dim = source_dim
        self.target_dim = target_dim
        if matrix is None:
            rng = np.random.default_rng(0)
            self._matrix = rng.standard_normal((target_dim, source_dim)) * 0.1
        else:
            self._matrix = np.asarray(matrix, dtype=float)
            if self._matrix.shape != (target_dim, source_dim):
                raise ValueError(f"Expected adapter shape {(target_dim, source_dim)}, got {self._matrix.shape}")

    @property
    def matrix(self) -> list[list[float]]:
        """Serializable adapter matrix."""
        return self._matrix.tolist()  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "matrix": self.matrix,
        }


def transfer_learn(
    source_network: Any,
    target_network: Any,
    dataset: list[tuple[Any, Any]],
    loss_fn: Any,
    adapter_matrix: list[list[float]],
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
