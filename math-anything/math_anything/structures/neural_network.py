"""Neural-network layers as Bourbaki morphisms."""

from __future__ import annotations

from typing import Any

import numpy as np

from math_anything.morphisms import Morphism


class LinearMorphism(Morphism):
    """Linear layer y = W x + b as a structure-preserving transformation."""

    name: str = "linear"
    source_type: str = "VectorSpace"
    target_type: str = "VectorSpace"
    category: str = "surrogate"

    def __init__(self, name: str, input_dim: int, output_dim: int):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_type", f"R^{input_dim}")
        object.__setattr__(self, "target_type", f"R^{output_dim}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._rng = np.random.default_rng(0)
        self.weight = self._rng.standard_normal((output_dim, input_dim)) * 0.1
        self.bias = np.zeros(output_dim)

        self.invariants_kept = ["linearity", "differentiability"]
        self.invariants_lost = []
        self.invariants_introduced = ["learnable_parameters"]

    @property
    def mathematical_form(self) -> str:
        return f"y = W_{{{self.input_dim}x{self.output_dim}}} x + b"

    def apply(self, input_data: Any) -> np.ndarray:
        x = np.asarray(input_data, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(f"Expected input shape ({self.input_dim},), got {x.shape}")
        return self.weight @ x + self.bias


class ActivationMorphism(Morphism):
    """Element-wise non-linear activation as a morphism."""

    name: str = "activation"
    source_type: str = "VectorSpace"
    target_type: str = "VectorSpace"
    category: str = "surrogate"

    def __init__(self, name: str, activation: str = "relu"):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_type", "VectorSpace")
        object.__setattr__(self, "target_type", "VectorSpace")
        self.activation = activation.lower()
        if self.activation not in {"relu", "tanh", "sigmoid"}:
            raise ValueError(f"Unsupported activation: {activation}")

        self.invariants_kept = ["element_wise_structure", "differentiability"]
        self.invariants_lost = ["linearity"]
        self.invariants_introduced = ["nonlinear_representation_capacity"]

    @property
    def mathematical_form(self) -> str:
        return f"x_i = {self.activation}(x_i)"

    def apply(self, input_data: Any) -> np.ndarray:
        x = np.asarray(input_data, dtype=float)
        if self.activation == "relu":
            return np.maximum(0.0, x)
        if self.activation == "tanh":
            return np.tanh(x)
        return 1.0 / (1.0 + np.exp(-x))


class LossMorphism(Morphism):
    """Loss function comparing predictions to targets."""

    name: str = "loss"
    source_type: str = "PredictionSpace"
    target_type: str = "Scalar"
    category: str = "surrogate"

    def __init__(self, name: str, loss: str = "mse"):
        object.__setattr__(self, "name", name)
        self.loss = loss.lower()
        if self.loss not in {"mse", "mae"}:
            raise ValueError(f"Unsupported loss: {loss}")

        self.invariants_kept = ["differentiability"]
        self.invariants_lost = ["full_state_information"]
        self.invariants_introduced = ["optimization_objective"]

    @property
    def mathematical_form(self) -> str:
        if self.loss == "mse":
            return "L = (1/N) Σ (y_pred - y_true)²"
        return "L = (1/N) Σ |y_pred - y_true|"

    def apply(self, input_data: Any) -> float:
        y_pred, y_true = input_data
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        if self.loss == "mse":
            return float(np.mean((y_pred - y_true) ** 2))
        return float(np.mean(np.abs(y_pred - y_true)))
