# Bourbaki ML Phase 5: Real Surrogate Backend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the toy neural-network implementation (tanh/sigmoid backprop) and provide a backend adapter layer so real surrogate frameworks (DeePMD/MACE/CHGNet) can be plugged in with a numpy fallback.

**Architecture:** Extend `SequentialNetwork.backward` to support all supported activations. Add a new `math_anything/structures/surrogate_backend.py` module that defines a `SurrogateBackend` protocol, a `BackendRegistry`, a `NumpySurrogateBackend`, and optional import-guarded stubs for DeePMD/MACE/CHGNet. Provide a `SurrogateModel` facade that selects backends by name. Wire a `--backend` option into the `ml` CLI and a `backend` parameter into `analyze_ml_model`.

**Tech Stack:** Python 3.10+, `numpy`, existing `math_anything` packages. Optional frameworks are detected at runtime; the code degrades gracefully to numpy when absent.

## Global Constraints

- No new **runtime** dependencies. Optional frameworks may be imported lazily inside backend methods; missing packages must raise a clear `ImportError` or trigger a graceful numpy fallback.
- All public functions return serializable dataclasses, dicts, or primitive values.
- Any training example must finish in < 1 s.
- All code changes must be covered by unit tests; CLI/MCP changes by integration tests.
- Follow existing style: `ruff check` on touched files must stay clean (the wider repo has pre-existing violations).
- Do not break existing Phase 1–4 public APIs (`SequentialNetwork`, `train_and_capture`, `transfer_learn`, etc.).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `math_anything/structures/neural_network.py` | Extend `backward` to support tanh/sigmoid; add activation derivative helper. |
| `tests/unit/test_structures_neural_network.py` | Add tanh/sigmoid training tests. |
| `math_anything/structures/surrogate_backend.py` | `SurrogateBackend` protocol, `BackendRegistry`, `NumpySurrogateBackend`, optional stubs, `SurrogateModel` facade. |
| `tests/unit/test_structures_surrogate_backend.py` | Registry, numpy backend, unknown backend, optional stub behavior. |
| `math_anything/cli.py` | Add `--backend` argument and fallback reporting. |
| `math_anything/mcp_server.py` | Add `backend` parameter to `analyze_ml_model`. |
| `tests/integration/test_ml_cli.py` | CLI integration tests for `--backend`. |
| `tests/test_mcp_server.py` | MCP integration test for `backend` parameter. |

---

## Task 1: Complete backprop for tanh and sigmoid

**Files:**
- Modify: `math_anything/structures/neural_network.py`
- Test: `tests/unit/test_structures_neural_network.py`

**Interfaces:**
- Consumes: existing `SequentialNetwork`, `ActivationMorphism`, `LossMorphism`.
- Produces: `ActivationMorphism.derivative(pre_activation)` and `SequentialNetwork.backward` supports `relu`, `tanh`, `sigmoid`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_structures_neural_network.py`:

```python
import pytest


def test_sequential_network_training_reduces_loss_tanh():
    from math_anything.structures.neural_network import (
        ActivationMorphism,
        LinearMorphism,
        LossMorphism,
        SequentialNetwork,
    )

    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=4),
        ActivationMorphism(name="tanh_1", activation="tanh"),
        LinearMorphism(name="linear_2", input_dim=4, output_dim=1),
    ])
    loss_fn = LossMorphism(name="mse", loss="mse")

    xs = [np.array([x]) for x in [-1.0, 0.0, 1.0]]
    ys = [np.array([2 * x + 1]) for x in [-1.0, 0.0, 1.0]]

    initial_loss = None
    for epoch in range(50):
        epoch_loss = 0.0
        for x, y in zip(xs, ys):
            y_pred = net.forward(x)
            epoch_loss += loss_fn.apply((y_pred, y))
            grads = net.backward(x, y, loss_fn)
            net.sgd_step(grads, lr=0.05)
        if initial_loss is None:
            initial_loss = epoch_loss

    assert epoch_loss < initial_loss


def test_sequential_network_training_reduces_loss_sigmoid():
    from math_anything.structures.neural_network import (
        ActivationMorphism,
        LinearMorphism,
        LossMorphism,
        SequentialNetwork,
    )

    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=4),
        ActivationMorphism(name="sigmoid_1", activation="sigmoid"),
        LinearMorphism(name="linear_2", input_dim=4, output_dim=1),
    ])
    loss_fn = LossMorphism(name="mse", loss="mse")

    xs = [np.array([x]) for x in [-1.0, 0.0, 1.0]]
    ys = [np.array([2 * x + 1]) for x in [-1.0, 0.0, 1.0]]

    initial_loss = None
    for epoch in range(50):
        epoch_loss = 0.0
        for x, y in zip(xs, ys):
            y_pred = net.forward(x)
            epoch_loss += loss_fn.apply((y_pred, y))
            grads = net.backward(x, y, loss_fn)
            net.sgd_step(grads, lr=0.05)
        if initial_loss is None:
            initial_loss = epoch_loss

    assert epoch_loss < initial_loss
```

- [ ] **Step 2: Run the failing test**

```bash
cd /d/math-anything/math-anything
python -m pytest tests/unit/test_structures_neural_network.py::test_sequential_network_training_reduces_loss_tanh tests/unit/test_structures_neural_network.py::test_sequential_network_training_reduces_loss_sigmoid -v
```

Expected: tests pass for tanh (it happens to work via the fallback `h = layer.apply(h)` path) but loss reduction may fail or be unreliable for sigmoid. More importantly, we want explicit derivative support.

- [ ] **Step 3: Implement activation derivatives**

In `math_anything/structures/neural_network.py`, add a `derivative` method to `ActivationMorphism`:

```python
    def derivative(self, pre_activation: np.ndarray) -> np.ndarray:
        """Return the element-wise derivative of the activation at pre_activation."""
        x = np.asarray(pre_activation, dtype=float)
        if self.activation == "relu":
            return (x > 0).astype(float)
        if self.activation == "tanh":
            t = np.tanh(x)
            return 1.0 - t * t
        # sigmoid
        x = np.clip(x, -500.0, 500.0)
        s = 1.0 / (1.0 + np.exp(-x))
        return s * (1.0 - s)
```

Then replace the ReLU-only branch in `SequentialNetwork.backward` with:

```python
            elif isinstance(layer, ActivationMorphism):
                delta = delta * layer.derivative(pre_activation)
```

The `elif` must come before the generic `else: h = layer.apply(h)` branch, which should remain as a fallback.

Also update the docstring of `backward` from "Backprop for MSE loss through linear + relu layers" to "Backprop for MSE loss through linear + activation layers".

- [ ] **Step 4: Run the test**

```bash
python -m pytest tests/unit/test_structures_neural_network.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add math_anything/structures/neural_network.py tests/unit/test_structures_neural_network.py
git commit -m "feat(structures): support tanh and sigmoid derivatives in SequentialNetwork.backward"
```

---

## Task 2: Surrogate backend protocol, registry, numpy backend, optional stubs

**Files:**
- Create: `math_anything/structures/surrogate_backend.py`
- Test: `tests/unit/test_structures_surrogate_backend.py`

**Interfaces:**
- Consumes: `SequentialNetwork`, `LinearMorphism`, `ActivationMorphism`, `LossMorphism`, `train_and_capture`.
- Produces:
  - `SurrogateBackend` protocol
  - `BackendRegistry` with `register(name, cls)`, `get(name)`, `list()`
  - `NumpySurrogateBackend`
  - `DeePMDBackend`, `MaceBackend`, `ChgnetBackend` (import-guarded stubs)
  - `SurrogateModel` facade

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_structures_surrogate_backend.py`:

```python
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
```

- [ ] **Step 2: Run the failing test**

```bash
python -m pytest tests/unit/test_structures_surrogate_backend.py -v
```

Expected: `ModuleNotFoundError` for `math_anything.structures.surrogate_backend`.

- [ ] **Step 3: Implement the backend module**

Create `math_anything/structures/surrogate_backend.py`:

```python
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

    def fit(self, dataset: list[tuple[Any, Any]], epochs: int = 10, lr: float = 0.05) -> None:
        ...

    def predict(self, x: Any) -> Any:
        ...

    def to_morphism_chain(self) -> list[dict[str, Any]]:
        ...


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
        self._network = SequentialNetwork([
            LinearMorphism(name="linear_1", input_dim=self.input_dim, output_dim=self.hidden_dim),
            ActivationMorphism(name=f"{self.activation}_1", activation=self.activation),
            LinearMorphism(name="linear_2", input_dim=self.hidden_dim, output_dim=self.output_dim),
        ])

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
                "name": f"model_mlp_{self.activation}",
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
        raise ImportError(
            f"The '{self.name}' backend requires the '{self._package_name}' package."
        )

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

    def __init__(self):
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
```

- [ ] **Step 4: Run the test**

```bash
python -m pytest tests/unit/test_structures_surrogate_backend.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add math_anything/structures/surrogate_backend.py tests/unit/test_structures_surrogate_backend.py
git commit -m "feat(structures): add SurrogateBackend protocol, registry, numpy backend, and optional stubs"
```

---

## Task 3: Wire `--backend` into CLI and MCP

**Files:**
- Modify: `math_anything/cli.py`
- Modify: `math_anything/mcp_server.py`
- Test: `tests/integration/test_ml_cli.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `SurrogateModel`, `list_backends`, `get_backend`.
- Produces: CLI `--backend {numpy,deepmd,mace,chgnet}`; MCP `backend: str = "numpy"`; report fields `backend_requested`, `backend_used`, `backend_available`.

- [ ] **Step 1: Add the CLI argument and handler**

In `math_anything/cli.py`, add to `ml_parser` after existing arguments:

```python
    ml_parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "deepmd", "mace", "chgnet"],
        help="Surrogate backend to use (falls back to numpy if not installed)",
    )
```

In `cmd_ml`, after building the basic report (before any optional blocks), add:

```python
        from math_anything.structures.surrogate_backend import (
            SurrogateModel,
            get_backend,
            list_backends,
        )

        backend_used = args.backend
        backend_available = args.backend in list_backends()
        try:
            model = SurrogateModel(
                backend=args.backend,
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                hidden_dim=4,
            )
            dataset = [
                (np.array([x] * args.input_dim), np.array([2.0 * x + 1.0] * args.output_dim))
                for x in [-1.0, 0.0, 1.0]
            ]
            model.fit(dataset, epochs=5, lr=0.05)
            demo_pred = model.predict(np.array([0.5] * args.input_dim))
        except ImportError:
            backend_used = "numpy"
            model = SurrogateModel(
                backend="numpy",
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                hidden_dim=4,
            )
            dataset = [
                (np.array([x] * args.input_dim), np.array([2.0 * x + 1.0] * args.output_dim))
                for x in [-1.0, 0.0, 1.0]
            ]
            model.fit(dataset, epochs=5, lr=0.05)
            demo_pred = model.predict(np.array([0.5] * args.input_dim))

        report["backend_requested"] = args.backend
        report["backend_used"] = backend_used
        report["backend_available"] = backend_available
        report["surrogate_demo_prediction"] = (
            demo_pred.tolist() if hasattr(demo_pred, "tolist") else demo_pred
        )
```

- [ ] **Step 2: Add the MCP parameter and report field**

In `math_anything/mcp_server.py`, change the signature of `analyze_ml_model` to:

```python
def analyze_ml_model(
    input_dim: int = 2,
    output_dim: int = 1,
    architecture: str = "mlp",
    loss: str = "mse",
    compare_paths: bool = False,
    transfer: bool = False,
    backend: str = "numpy",
) -> str:
```

Add after the basic report building:

```python
    from math_anything.structures.surrogate_backend import (
        SurrogateModel,
        list_backends,
    )

    backend_used = backend
    backend_available = backend in list_backends()
    try:
        model = SurrogateModel(
            backend=backend,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=4,
        )
        dataset = [
            (np.array([x] * input_dim), np.array([2.0 * x + 1.0] * output_dim))
            for x in [-1.0, 0.0, 1.0]
        ]
        model.fit(dataset, epochs=5, lr=0.05)
        demo_pred = model.predict(np.array([0.5] * input_dim))
    except ImportError:
        backend_used = "numpy"
        model = SurrogateModel(
            backend="numpy",
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=4,
        )
        dataset = [
            (np.array([x] * input_dim), np.array([2.0 * x + 1.0] * output_dim))
            for x in [-1.0, 0.0, 1.0]
        ]
        model.fit(dataset, epochs=5, lr=0.05)
        demo_pred = model.predict(np.array([0.5] * input_dim))

    report["backend_requested"] = backend
    report["backend_used"] = backend_used
    report["backend_available"] = backend_available
    report["surrogate_demo_prediction"] = (
        demo_pred.tolist() if hasattr(demo_pred, "tolist") else demo_pred
    )
```

- [ ] **Step 3: Write integration tests**

Append to `tests/integration/test_ml_cli.py`:

```python
def test_cli_ml_backend_numpy_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "math_anything",
            "ml",
            "--input-dim", "1",
            "--output-dim", "1",
            "--backend", "numpy",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["backend_requested"] == "numpy"
    assert data["backend_used"] == "numpy"
    assert data["backend_available"] is True


def test_cli_ml_backend_uninstalled_falls_back_to_numpy():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "math_anything",
            "ml",
            "--input-dim", "1",
            "--output-dim", "1",
            "--backend", "deepmd",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["backend_requested"] == "deepmd"
    assert data["backend_used"] == "numpy"
    assert data["backend_available"] is True
```

Append to `tests/test_mcp_server.py`:

```python
def test_mcp_analyze_ml_model_reports_backend():
    from math_anything.mcp_server import analyze_ml_model

    raw = analyze_ml_model(
        input_dim=1,
        output_dim=1,
        architecture="mlp",
        loss="mse",
        backend="numpy",
    )
    report = json.loads(raw)
    assert report["backend_requested"] == "numpy"
    assert report["backend_used"] == "numpy"
    assert report["backend_available"] is True
    assert "surrogate_demo_prediction" in report
```

- [ ] **Step 4: Run the integration tests**

```bash
python -m pytest tests/integration/test_ml_cli.py::test_cli_ml_backend_numpy_runs tests/integration/test_ml_cli.py::test_cli_ml_backend_uninstalled_falls_back_to_numpy tests/test_mcp_server.py::test_mcp_analyze_ml_model_reports_backend -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add math_anything/cli.py math_anything/mcp_server.py tests/integration/test_ml_cli.py tests/test_mcp_server.py
git commit -m "feat(cli,mcp): add --backend parameter with numpy fallback"
```

---

## Task 4: Final verification

- [ ] **Step 1: Run the focused test suite**

```bash
cd /d/math-anything/math-anything
python -m pytest tests/unit/test_structures_neural_network.py tests/unit/test_structures_surrogate_backend.py tests/integration/test_ml_cli.py tests/test_mcp_server.py -v --tb=short --no-cov --deselect tests/test_mcp_server.py::TestDiscoverEquations::test_custom_method
```

Expected: all tests pass.

- [ ] **Step 2: Run ruff on touched files**

```bash
ruff check math_anything/structures/neural_network.py math_anything/structures/surrogate_backend.py math_anything/cli.py math_anything/mcp_server.py tests/unit/test_structures_neural_network.py tests/unit/test_structures_surrogate_backend.py tests/integration/test_ml_cli.py tests/test_mcp_server.py
```

Expected: clean.

- [ ] **Step 3: Push**

```bash
git push github main
git push origin main
```

---

## Self-Review

**Spec coverage:**
- tanh/sigmoid backprop — Task 1.
- SurrogateBackend protocol, registry, numpy backend, optional stubs — Task 2.
- CLI/MCP `--backend` integration with fallback — Task 3.
- Verification — Task 4.

**Placeholder scan:** No TBD/TODO/fill-in-details.

**Type consistency:**
- `SurrogateBackend` protocol matches `NumpySurrogateBackend` and stub classes.
- `SurrogateModel` exposes `fit/predict/to_morphism_chain/to_dict`.

**Gaps:** Optional backends are stubs only; real framework wrappers would require those packages as optional dependencies and are out of scope for this phase.
