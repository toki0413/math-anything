import numpy as np
import pytest

from math_anything.structures.surrogate_backend import (
    BackendRegistry,
    NumpySurrogateBackend,
    SurrogateModel,
    get_backend,
    list_backends,
)


def test_registry_lists_numpy_backend():
    assert "numpy" in list_backends()


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_backend("nonexistent")


def test_numpy_backend_trains_and_predicts():
    backend = NumpySurrogateBackend(input_dim=1, output_dim=1, hidden_dim=4)
    dataset = [(np.array([x]), np.array([2 * x + 1])) for x in [-1.0, 0.0, 1.0]]
    backend.fit(dataset, epochs=5, lr=0.05)
    pred = backend.predict(np.array([0.5]))
    assert pred.shape == (1,)
    chain = backend.to_morphism_chain()
    assert isinstance(chain, list)
    assert any("linear" in step.get("name", "") for step in chain)


def test_surrogate_model_facade_uses_numpy():
    model = SurrogateModel(backend="numpy", input_dim=1, output_dim=1, hidden_dim=4)
    dataset = [(np.array([x]), np.array([2 * x + 1])) for x in [-1.0, 0.0, 1.0]]
    model.fit(dataset, epochs=5, lr=0.05)
    pred = model.predict(np.array([0.5]))
    assert pred.shape == (1,)


def test_optional_backend_stub_raises_without_framework():
    # These frameworks are not runtime dependencies, so the stubs must raise.
    for name in ("deepmd", "mace", "chgnet"):
        backend = get_backend(name, input_dim=1, output_dim=1)
        with pytest.raises((ImportError, NotImplementedError)):
            backend.fit([], epochs=1, lr=0.05)
