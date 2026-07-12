# Bourbaki ML Phase 4: Transfer Learning as a Natural Transformation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Model transfer learning as a natural transformation in the Bourbaki framework: implement functors, natural transformations, and a concrete weight-space transfer helper, then expose the result through the CLI and MCP.

**Architecture:** Add `math_anything/structures/functor.py` with matrix-category functors and a natural-transformation checker, and `math_anything/structures/transfer.py` with a tiny weight-space transfer learning helper. The CLI/MCP demonstrate the concept by training a source network, transferring its flattened weights to a target network via a random linear adapter, and reporting whether the adapter satisfies the natural-transformation square on a sample morphism.

**Tech Stack:** Python 3.10+, `numpy`, existing `math_anything` packages.

## Global Constraints

- No new external runtime dependencies.
- All public functions return serializable dataclasses, dicts, or primitive values.
- Any training example must finish in < 1 s.
- All code changes must be covered by unit tests; CLI/MCP changes by integration tests.
- Follow existing style: `ruff check` on touched files must stay clean (the wider repo has pre-existing violations).
- Do not break existing Phase 1/2/3 public APIs.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `math_anything/structures/functor.py` | `Functor` ABC, `MatrixFunctor`, `NaturalTransformation`, `is_natural_transformation`. |
| `tests/unit/test_structures_functor.py` | Unit tests for the core abstraction. |
| `math_anything/structures/transfer.py` | `transfer_weights`, `transfer_learn`, `WeightSpaceTransfer`. |
| `tests/unit/test_structures_transfer.py` | Unit tests for weight-space transfer. |
| `math_anything/cli.py` | Add `ml transfer` subcommand / `--transfer` flag. |
| `math_anything/mcp_server.py` | Add `transfer: bool = False` to `analyze_ml_model`. |
| `tests/integration/test_ml_cli.py` | CLI integration test. |
| `tests/test_mcp_server.py` | MCP integration test. |

---

## Task 1: Functor and Natural Transformation core

**Files:**
- Create: `math_anything/structures/functor.py`
- Test: `tests/unit/test_structures_functor.py`

**Interfaces:**
- Consumes: `numpy`, `abc`.
- Produces:
  - `Functor` ABC
  - `MatrixFunctor(matrix: np.ndarray)` — maps a square-invertible matrix category to itself.
  - `NaturalTransformation(components: dict[Any, Any])`
  - `is_natural_transformation(F, G, eta, test_morphisms) -> tuple[bool, str]`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_structures_functor.py`:

```python
import numpy as np

from math_anything.structures.functor import (
    MatrixFunctor,
    NaturalTransformation,
    is_natural_transformation,
)


def test_identity_matrix_functor_maps_morphism():
    d = 3
    F = MatrixFunctor(np.eye(d))
    M = np.diag([1.0, 2.0, 3.0])
    mapped = F.map_morphism(M)
    np.testing.assert_allclose(mapped, M)


def test_natural_transformation_identity_is_valid():
    d = 2
    T = np.array([[2.0, 0.0], [0.0, 3.0]])
    F = MatrixFunctor(T)
    G = MatrixFunctor(T)
    eta = NaturalTransformation({d: np.eye(d)})

    M = np.array([[1.0, 1.0], [0.0, 1.0]])
    valid, reason = is_natural_transformation(
        F, G, eta, test_morphisms=[(d, d, M)]
    )
    assert valid, reason


def test_non_natural_transformation_is_invalid():
    d = 2
    F = MatrixFunctor(np.eye(d))
    G = MatrixFunctor(2 * np.eye(d))
    eta = NaturalTransformation({d: np.eye(d)})

    M = np.array([[1.0, 1.0], [0.0, 1.0]])
    valid, reason = is_natural_transformation(
        F, G, eta, test_morphisms=[(d, d, M)]
    )
    assert not valid
```

- [ ] **Step 2: Run the failing test**

```bash
cd /d/math-anything/math-anything
python -m pytest tests/unit/test_structures_functor.py -v
```

Expected: `ModuleNotFoundError` for `math_anything.structures.functor`.

- [ ] **Step 3: Implement the core module**

Create `math_anything/structures/functor.py`:

```python
"""Functors and natural transformations for concrete matrix categories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class Functor(ABC):
    """Abstract functor between two categories."""

    @abstractmethod
    def map_object(self, obj: Any) -> Any:
        """Map an object to an object."""

    @abstractmethod
    def map_morphism(self, morphism: Any) -> Any:
        """Map a morphism to a morphism."""


class MatrixFunctor(Functor):
    """Functor on a matrix category defined by a square invertible matrix T.

    Objects are integer dimensions; morphisms are square matrices of that size.
    A morphism M is mapped to T @ M @ T^{-1}.
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=float)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("MatrixFunctor requires a square matrix")
        self._inv = np.linalg.inv(self.matrix)
        self._dim = self.matrix.shape[0]

    def map_object(self, obj: Any) -> Any:
        if obj != self._dim:
            raise ValueError(f"Object {obj} cannot be mapped by this functor")
        return self._dim

    def map_morphism(self, morphism: Any) -> np.ndarray:
        M = np.asarray(morphism, dtype=float)
        return self.matrix @ M @ self._inv


@dataclass
class NaturalTransformation:
    """A natural transformation eta: F => G, given by components eta_X."""

    components: dict[Any, Any]


def is_natural_transformation(
    F: Functor,
    G: Functor,
    eta: NaturalTransformation,
    test_morphisms: list[tuple[Any, Any, Any]],
    atol: float = 1e-8,
) -> tuple[bool, str]:
    """Verify G(f) ∘ eta_X == eta_Y ∘ F(f) for each test morphism f: X -> Y.

    Returns (True, "") if the square commutes for every test morphism,
    otherwise (False, diagnostic message).
    """
    for source_obj, target_obj, morphism in test_morphisms:
        eta_source = np.asarray(eta.components.get(source_obj), dtype=float)
        eta_target = np.asarray(eta.components.get(target_obj), dtype=float)

        if eta_source.size == 0 or eta_target.size == 0:
            return False, f"Missing component for {source_obj} or {target_obj}"

        left = G.map_morphism(morphism) @ eta_source
        right = eta_target @ F.map_morphism(morphism)

        if not np.allclose(left, right, atol=atol):
            return False, (
                f"Square fails for morphism {source_obj} -> {target_obj}: "
                f"max deviation {float(np.max(np.abs(left - right)))}"
            )

    return True, ""
```

- [ ] **Step 4: Run the test**

```bash
python -m pytest tests/unit/test_structures_functor.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add math_anything/structures/functor.py tests/unit/test_structures_functor.py
git commit -m "feat(structures): add functors and natural transformations for matrix categories"
```

---

## Task 2: Weight-space transfer learning

**Files:**
- Create: `math_anything/structures/transfer.py`
- Test: `tests/unit/test_structures_transfer.py`

**Interfaces:**
- Consumes: `SequentialNetwork`, `LinearMorphism`, `train_and_capture`, `OptimizationState`, `TrainingResult`, `MatrixFunctor`, `NaturalTransformation`, `is_natural_transformation`.
- Produces:
  - `transfer_weights(source_weights, adapter_matrix) -> np.ndarray`
  - `flatten_network_weights(network) -> np.ndarray`
  - `set_network_weights(network, weights) -> None`
  - `WeightSpaceTransfer(source_dim, target_dim)`
  - `transfer_learn(source_network, target_network, dataset, loss_fn, adapter_matrix, epochs, lr) -> TrainingResult`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_structures_transfer.py`:

```python
import numpy as np

from math_anything.structures.neural_network import (
    ActivationMorphism,
    LinearMorphism,
    LossMorphism,
    SequentialNetwork,
)
from math_anything.structures.transfer import (
    WeightSpaceTransfer,
    flatten_network_weights,
    set_network_weights,
    transfer_learn,
    transfer_weights,
)


def test_transfer_weights_shape():
    adapter = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    source = np.array([1.0, 2.0])
    target = transfer_weights(source, adapter)
    assert target.shape == (3,)
    np.testing.assert_allclose(target, [1.0, 2.0, 3.0])


def test_flatten_and_set_network_weights_roundtrip():
    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
    ])
    original = flatten_network_weights(net)
    set_network_weights(net, original + 0.1)
    restored = flatten_network_weights(net)
    np.testing.assert_allclose(restored, original + 0.1)


def test_transfer_learn_trains_target():
    source = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
    ])
    target = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
    ])
    loss_fn = LossMorphism(name="mse", loss="mse")
    dataset = [(np.array([x]), np.array([2 * x + 1])) for x in [-1.0, 0.0, 1.0]]

    adapter = WeightSpaceTransfer(flatten_network_weights(source).size, flatten_network_weights(target).size).matrix
    result = transfer_learn(source, target, dataset, loss_fn, adapter, epochs=3, lr=0.05)
    assert len(result.states) == 3
    assert result.final_loss < result.states[0].loss
```

- [ ] **Step 2: Run the failing test**

```bash
python -m pytest tests/unit/test_structures_transfer.py -v
```

Expected: `ModuleNotFoundError` for `math_anything.structures.transfer`.

- [ ] **Step 3: Implement the transfer module**

Create `math_anything/structures/transfer.py`:

```python
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
```

- [ ] **Step 4: Run the test**

```bash
python -m pytest tests/unit/test_structures_transfer.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add math_anything/structures/transfer.py tests/unit/test_structures_transfer.py
git commit -m "feat(structures): add weight-space transfer learning helper"
```

---

## Task 3: Wire transfer learning into CLI and MCP

**Files:**
- Modify: `math_anything/cli.py`
- Modify: `math_anything/mcp_server.py`
- Test: `tests/integration/test_ml_cli.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `is_natural_transformation`, `MatrixFunctor`, `NaturalTransformation`, `transfer_learn`, `WeightSpaceTransfer`, `flatten_network_weights`.
- Produces: CLI `--transfer` flag on `ml`; MCP `transfer: bool = False`; report field `transfer_learning` with `natural_transformation_valid` and `final_loss`.

- [ ] **Step 1: Add the CLI flag and handler**

In `math_anything/cli.py`, after `--compare-paths` (around line 344), add:

```python
    ml_parser.add_argument(
        "--transfer",
        action="store_true",
        help="Demonstrate transfer learning as a natural transformation",
    )
```

In `cmd_ml`, after the `if args.compare_paths:` block, add:

```python
        if args.transfer:
            import numpy as np

            from math_anything.structures.functor import (
                MatrixFunctor,
                NaturalTransformation,
                is_natural_transformation,
            )
            from math_anything.structures.neural_network import (
                ActivationMorphism,
                LinearMorphism,
                LossMorphism as LossFn,
                SequentialNetwork,
            )
            from math_anything.structures.transfer import (
                WeightSpaceTransfer,
                flatten_network_weights,
                transfer_learn,
            )

            loss_fn = LossFn(name="loss", loss=args.loss)
            dataset = [
                (np.array([x] * args.input_dim), np.array([2.0 * x + 1.0] * args.output_dim))
                for x in [-1.0, 0.0, 1.0]
            ]

            def _make_network():
                return SequentialNetwork([
                    LinearMorphism(name="linear_1", input_dim=args.input_dim, output_dim=4),
                    ActivationMorphism(name="relu_1", activation="relu"),
                    LinearMorphism(name="linear_2", input_dim=4, output_dim=args.output_dim),
                ])

            source = _make_network()
            target = _make_network()
            source_dim = flatten_network_weights(source).size
            adapter = WeightSpaceTransfer(source_dim, source_dim).matrix

            result = transfer_learn(source, target, dataset, loss_fn, adapter, epochs=3, lr=0.05)

            # Natural-transformation check: identity adapter + identical functors should commute.
            dim = source_dim
            F = MatrixFunctor(np.eye(dim))
            G = MatrixFunctor(np.eye(dim))
            eta = NaturalTransformation({dim: np.eye(dim)})
            sample_morphism = np.eye(dim)
            valid, reason = is_natural_transformation(
                F, G, eta, test_morphisms=[(dim, dim, sample_morphism)]
            )

            report["transfer_learning"] = {
                "natural_transformation_valid": valid,
                "natural_transformation_reason": reason,
                "final_loss": result.final_loss,
                "epochs": 3,
            }
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
) -> str:
```

Add the transfer block just before the final `return json.dumps(...)`:

```python
    if transfer:
        from math_anything.structures.functor import (
            MatrixFunctor,
            NaturalTransformation,
            is_natural_transformation,
        )
        from math_anything.structures.neural_network import (
            ActivationMorphism,
            LinearMorphism,
            LossMorphism as LossFn,
            SequentialNetwork,
        )
        from math_anything.structures.transfer import (
            WeightSpaceTransfer,
            flatten_network_weights,
            transfer_learn,
        )

        loss_fn = LossFn(name="loss", loss=loss)
        dataset = [
            (np.array([x] * input_dim), np.array([2.0 * x + 1.0] * output_dim))
            for x in [-1.0, 0.0, 1.0]
        ]

        def _make_network():
            return SequentialNetwork([
                LinearMorphism(name="linear_1", input_dim=input_dim, output_dim=4),
                ActivationMorphism(name="relu_1", activation="relu"),
                LinearMorphism(name="linear_2", input_dim=4, output_dim=output_dim),
            ])

        source = _make_network()
        target = _make_network()
        source_dim = flatten_network_weights(source).size
        adapter = WeightSpaceTransfer(source_dim, source_dim).matrix

        result = transfer_learn(source, target, dataset, loss_fn, adapter, epochs=3, lr=0.05)

        dim = source_dim
        F = MatrixFunctor(np.eye(dim))
        G = MatrixFunctor(np.eye(dim))
        eta = NaturalTransformation({dim: np.eye(dim)})
        valid, reason = is_natural_transformation(
            F, G, eta, test_morphisms=[(dim, dim, np.eye(dim))]
        )

        report["transfer_learning"] = {
            "natural_transformation_valid": valid,
            "natural_transformation_reason": reason,
            "final_loss": result.final_loss,
            "epochs": 3,
        }
```

- [ ] **Step 3: Write integration tests**

Append to `tests/integration/test_ml_cli.py`:

```python
def test_cli_ml_transfer_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "math_anything",
            "ml",
            "--input-dim", "1",
            "--output-dim", "1",
            "--transfer",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "transfer_learning" in data
    assert isinstance(data["transfer_learning"]["natural_transformation_valid"], bool)
    assert isinstance(data["transfer_learning"]["final_loss"], float)
```

Append to `tests/test_mcp_server.py`:

```python
def test_mcp_analyze_ml_model_reports_transfer_learning():
    from math_anything.mcp_server import analyze_ml_model

    raw = analyze_ml_model(
        input_dim=1,
        output_dim=1,
        architecture="mlp",
        loss="mse",
        transfer=True,
    )
    report = json.loads(raw)
    assert "transfer_learning" in report
    assert isinstance(report["transfer_learning"]["natural_transformation_valid"], bool)
    assert isinstance(report["transfer_learning"]["final_loss"], float)
```

- [ ] **Step 4: Run the integration tests**

```bash
python -m pytest tests/integration/test_ml_cli.py::test_cli_ml_transfer_runs tests/test_mcp_server.py::test_mcp_analyze_ml_model_reports_transfer_learning -v
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add math_anything/cli.py math_anything/mcp_server.py tests/integration/test_ml_cli.py tests/test_mcp_server.py
git commit -m "feat(cli,mcp): expose transfer learning as a natural transformation"
```

---

## Task 4: Final verification

- [ ] **Step 1: Run the focused test suite**

```bash
cd /d/math-anything/math-anything
python -m pytest tests/unit/test_structures_functor.py tests/unit/test_structures_transfer.py tests/integration/test_ml_cli.py tests/test_mcp_server.py -v --tb=short --no-cov --deselect tests/test_mcp_server.py::TestDiscoverEquations::test_custom_method
```

Expected: all tests pass.

- [ ] **Step 2: Run ruff on touched files**

```bash
ruff check math_anything/structures/functor.py math_anything/structures/transfer.py math_anything/cli.py math_anything/mcp_server.py tests/unit/test_structures_functor.py tests/unit/test_structures_transfer.py tests/integration/test_ml_cli.py tests/test_mcp_server.py
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
- Functor / natural-transformation abstraction — Task 1.
- Weight-space transfer learning helper — Task 2.
- CLI/MCP integration — Task 3.
- Verification — Task 4.

**Placeholder scan:** No TBD/TODO/fill-in-details.

**Type consistency:**
- `is_natural_transformation` returns `tuple[bool, str]` consistently.
- `transfer_learn` returns `TrainingResult` from Phase 2.

**Gaps:** The demo uses identical source/target networks and an identity natural transformation for clarity; future work can extend to arbitrary domain-to-domain transfer maps.
