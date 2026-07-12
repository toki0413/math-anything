"""Surrogate-model backends with a numpy fallback and optional framework stubs."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from math_anything.structures.neural_network import (
    ActivationMorphism,
    LinearMorphism,
    LossMorphism,
    SequentialNetwork,
)
from math_anything.topology.training_curvature import train_and_capture


@runtime_checkable
class SurrogateBackend(Protocol):
    """Protocol for a surrogate-model backend."""

    name: str

    def fit(self, dataset: list[tuple[Any, Any]], epochs: int = 10, lr: float = 0.05) -> None: ...

    def predict(self, x: Any) -> Any: ...

    def to_morphism_chain(self) -> list[dict[str, Any]]: ...


class NumpySurrogateBackend:
    """Numpy-only surrogate backend using SequentialNetwork."""

    name = "numpy"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 4,
        activation: str = "relu",
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self._network: SequentialNetwork | None = None
        self._build_network()

    def _build_network(self) -> None:
        self._network = SequentialNetwork(
            [
                LinearMorphism(name="linear_1", input_dim=self.input_dim, output_dim=self.hidden_dim),
                ActivationMorphism(name=f"{self.activation}_1", activation=self.activation),
                LinearMorphism(name="linear_2", input_dim=self.hidden_dim, output_dim=self.output_dim),
            ]
        )

    def fit(self, dataset: list[tuple[Any, Any]], epochs: int = 10, lr: float = 0.05) -> None:
        if self._network is None:
            self._build_network()
        loss_fn = LossMorphism(name="mse", loss="mse")
        train_and_capture(self._network, dataset, loss_fn, epochs=epochs, lr=lr)

    def predict(self, x: Any) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("Backend has not been fitted")
        return self._network.forward(np.asarray(x, dtype=float))

    def to_morphism_chain(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "data_sampling",
                "type": "restriction",
                "invariants_kept": ["empirical_risk"],
                "invariants_lost": ["true_risk"],
            },
            {
                "name": f"model_mlp_linear_{self.activation}",
                "type": "surrogate",
                "invariants_kept": ["differentiability"],
                "invariants_lost": ["true_target_function"],
            },
            {
                "name": "loss_mse",
                "type": "projection",
                "invariants_kept": ["differentiability"],
                "invariants_lost": ["full_prediction_state"],
            },
        ]


class _OptionalBackendStub:
    """Base class for optional framework backends that are not installed."""

    def __init__(self, **kwargs: Any):
        pass

    def fit(self, dataset: list[tuple[Any, Any]], epochs: int = 10, lr: float = 0.05) -> None:
        raise ImportError(
            f"The '{self.name}' backend requires the '{self._package_name}' package. "
            f"Install it with: pip install {self._package_name}"
        )

    def predict(self, x: Any) -> Any:
        raise ImportError(f"The '{self.name}' backend requires the '{self._package_name}' package.")

    def to_morphism_chain(self) -> list[dict[str, Any]]:
        return [
            {
                "name": f"model_{self.name}",
                "type": "surrogate",
                "description": f"{self.name.upper()} surrogate backend (not installed)",
                "invariants_kept": [],
                "invariants_lost": ["numpy_fallback"],
            }
        ]


class DeePMDBackend(_OptionalBackendStub):
    """DeePMD-kit backend stub."""

    name = "deepmd"
    _package_name = "deepmd-kit"


class MaceBackend(_OptionalBackendStub):
    """MACE backend stub."""

    name = "mace"
    _package_name = "mace"


class ChgnetBackend(_OptionalBackendStub):
    """CHGNet backend stub."""

    name = "chgnet"
    _package_name = "chgnet"


class BackendRegistry:
    """Registry of available surrogate backends."""

    def __init__(self) -> None:
        self._backends: dict[str, type] = {}

    def register(self, name: str, backend_cls: type) -> None:
        self._backends[name] = backend_cls

    def get(self, name: str, **params: Any) -> SurrogateBackend:
        if name not in self._backends:
            available = ", ".join(sorted(self._backends.keys()))
            raise ValueError(f"Unknown backend '{name}'. Available: {available}")
        return self._backends[name](**params)

    def list(self) -> list[str]:
        return sorted(self._backends.keys())


_REGISTRY = BackendRegistry()
_REGISTRY.register("numpy", NumpySurrogateBackend)
_REGISTRY.register("deepmd", DeePMDBackend)
_REGISTRY.register("mace", MaceBackend)
_REGISTRY.register("chgnet", ChgnetBackend)


def get_backend(name: str, **params: Any) -> SurrogateBackend:
    return _REGISTRY.get(name, **params)


def list_backends() -> list[str]:
    return _REGISTRY.list()


class SurrogateModel:
    """Facade that selects a backend by name and exposes a uniform interface."""

    def __init__(self, backend: str = "numpy", **params: Any):
        self.backend_name = backend
        self._backend = get_backend(backend, **params)

    def fit(self, dataset: list[tuple[Any, Any]], epochs: int = 10, lr: float = 0.05) -> None:
        self._backend.fit(dataset, epochs=epochs, lr=lr)

    def predict(self, x: Any) -> Any:
        return self._backend.predict(x)

    def to_morphism_chain(self) -> list[dict[str, Any]]:
        return self._backend.to_morphism_chain()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "available": self.backend_name in list_backends(),
            "morphism_chain": self.to_morphism_chain(),
        }
