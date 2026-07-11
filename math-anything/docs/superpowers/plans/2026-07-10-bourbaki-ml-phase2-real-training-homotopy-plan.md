# Bourbaki ML Phase 2: Real Training Loop & Cross-Domain Homotopy

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the supervised-learning domain compute a real end-to-end training loop (forward, backward, SGD) and compare the resulting ML surrogate path against the DFT path using the existing homotopy engine.

**Architecture:** Build a tiny numpy MLP (`SequentialNetwork`) from the existing `LinearMorphism`/`ActivationMorphism`. Run SGD on a toy dataset, capture `(weights, loss)` states, and feed them into `trajectory_curvature`. Then build a `CategoryEngine` containing both the DFT morphism chain and the ML surrogate chain and ask `are_paths_homotopic` whether they preserve the same invariants. Expose the result through the existing `ml` CLI and `analyze_ml_model` MCP tool.

**Tech Stack:** Python 3.10+, `numpy`, existing `math_anything` packages. No new runtime dependencies.

## Global Constraints

- No new external runtime dependencies.
- All public functions return serializable dataclasses, dicts, or primitive values.
- Keep the NN small enough to run in unit tests in under 1 second.
- All code changes must be covered by unit tests; CLI/MCP changes by integration tests.
- Follow existing style: `ruff check math_anything/ tests/` must stay clean.
- Do not break existing Phase 1/2 APIs.

---

## Task 1: Composable numpy MLP with SGD

**Files:**
- Modify: `math_anything/structures/neural_network.py`
- Test: `tests/unit/test_structures_neural_network.py`

**Interfaces:**
- Consumes: `LinearMorphism`, `ActivationMorphism`.
- Produces:
  - `SequentialNetwork(layers: list[Morphism])`
  - `SequentialNetwork.forward(x) -> np.ndarray`
  - `SequentialNetwork.backward(x, y_true, loss_fn) -> dict[str, np.ndarray]`
  - `SequentialNetwork.sgd_step(gradients, lr) -> None`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_structures_neural_network.py`:

```python
def test_sequential_network_forward_shape():
    from math_anything.structures.neural_network import SequentialNetwork

    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=2, output_dim=3),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=3, output_dim=1),
    ])
    out = net.forward(np.array([1.0, -1.0]))
    assert out.shape == (1,)


def test_sequential_network_training_reduces_loss():
    from math_anything.structures.neural_network import SequentialNetwork

    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=4),
        ActivationMorphism(name="relu_1", activation="relu"),
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

- [ ] **Step 2: Run to confirm failure**

```bash
cd /d/math-anything/math-anything
python -m pytest tests/unit/test_structures_neural_network.py::test_sequential_network_forward_shape tests/unit/test_structures_neural_network.py::test_sequential_network_training_reduces_loss -v
```

Expected: `AttributeError: module 'math_anything.structures.neural_network' has no attribute 'SequentialNetwork'`.

- [ ] **Step 3: Implement `SequentialNetwork`**

Append to `math_anything/structures/neural_network.py`:

```python
class SequentialNetwork:
    """A tiny sequential MLP built from LinearMorphism and ActivationMorphism.

    Supports forward evaluation, manual backprop for linear+relu, and SGD updates.
    This is intentionally minimal: no new runtime dependencies, deterministic
    seed-0 initialization, and fast enough for unit tests.
    """

    def __init__(self, layers: list[Morphism]):
        self.layers = layers
        self._cache: list[tuple[np.ndarray, np.ndarray]] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass, caching pre- and post-activations for backward."""
        self._cache = []
        h = np.asarray(x, dtype=float)
        for layer in self.layers:
            if isinstance(layer, LinearMorphism):
                z = layer.weight @ h + layer.bias
                self._cache.append((h, z))
                h = z
            elif isinstance(layer, ActivationMorphism):
                h = layer.apply(h)
                self._cache.append((None, h))
            else:
                h = layer.apply(h)
                self._cache.append((None, h))
        return h

    def backward(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        loss_fn: LossMorphism,
    ) -> dict[str, np.ndarray]:
        """Backprop for MSE loss through linear + relu layers."""
        y_pred = self.forward(x)
        y_true = np.asarray(y_true, dtype=float)
        delta = 2 * (y_pred - y_true) / max(y_true.size, 1)

        grads: dict[str, np.ndarray] = {}
        for idx in reversed(range(len(self.layers))):
            layer = self.layers[idx]
            prev_input, pre_activation = self._cache[idx]
            if isinstance(layer, ActivationMorphism) and layer.activation == "relu":
                delta = delta * (pre_activation > 0).astype(float)
            elif isinstance(layer, LinearMorphism):
                grads[f"{layer.name}_weight"] = np.outer(delta, prev_input)
                grads[f"{layer.name}_bias"] = delta
                delta = layer.weight.T @ delta
        return grads

    def sgd_step(self, gradients: dict[str, np.ndarray], lr: float) -> None:
        """Apply gradients to linear-layer weights and biases."""
        for layer in self.layers:
            if isinstance(layer, LinearMorphism):
                w_key = f"{layer.name}_weight"
                b_key = f"{layer.name}_bias"
                if w_key in gradients:
                    layer.weight -= lr * gradients[w_key]
                if b_key in gradients:
                    layer.bias -= lr * gradients[b_key]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/unit/test_structures_neural_network.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add math_anything/structures/neural_network.py tests/unit/test_structures_neural_network.py
git commit -m "feat(structures): add SequentialNetwork with real SGD training"
```

---

## Task 2: Capture real training states and compute curvature

**Files:**
- Modify: `math_anything/topology/training_curvature.py`
- Test: `tests/unit/test_topology_training_curvature.py`

**Interfaces:**
- Consumes: `SequentialNetwork`, `OptimizationState`, `trajectory_curvature`.
- Produces:
  - `TrainingResult` dataclass
  - `train_and_capture(network, dataset, loss_fn, epochs, lr) -> TrainingResult`
  - `training_result_curvature(result: TrainingResult) -> list[float]`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_topology_training_curvature.py`:

```python
def test_train_and_capture_produces_curvature():
    from math_anything.structures.neural_network import (
        ActivationMorphism,
        LinearMorphism,
        LossMorphism,
        SequentialNetwork,
    )
    from math_anything.topology.training_curvature import (
        train_and_capture,
        training_result_curvature,
    )

    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
    ])
    loss_fn = LossMorphism(name="mse", loss="mse")
    dataset = [(np.array([x]), np.array([2 * x + 1])) for x in [-1.0, 0.0, 1.0]]

    result = train_and_capture(net, dataset, loss_fn, epochs=10, lr=0.05)
    assert len(result.states) > 0
    curvatures = training_result_curvature(result)
    assert isinstance(curvatures, list)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/unit/test_topology_training_curvature.py::test_train_and_capture_produces_curvature -v
```

Expected: ImportError.

- [ ] **Step 3: Implement training capture helpers**

Append to `math_anything/topology/training_curvature.py`:

```python
from dataclasses import dataclass
from typing import Any

import numpy as np


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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/unit/test_topology_training_curvature.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add math_anything/topology/training_curvature.py tests/unit/test_topology_training_curvature.py
git commit -m "feat(topology): capture real SGD training states and compute curvature"
```

---

## Task 3: Cross-domain homotopy and MCP/CLI update

**Files:**
- Create: `math_anything/topology/cross_domain.py`
- Modify: `math_anything/mcp_server.py`
- Modify: `math_anything/cli.py`
- Test: `tests/unit/test_topology_cross_domain.py`, `tests/integration/test_ml_cli.py`

**Interfaces:**
- Consumes: `CategoryEngine`, `are_paths_homotopic`, `DFTDomain`, `SupervisedLearningDomain`, `train_and_capture`, `SequentialNetwork`.
- Produces:
  - `cross_domain_homotopy(domain_a_name, params_a, domain_b_name, params_b) -> HomotopyWitness`
  - MCP `analyze_ml_model` reports `dft_homotopy`
  - CLI `ml --compare-with dft`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_topology_cross_domain.py`:

```python
from math_anything.topology.cross_domain import cross_domain_homotopy


def test_cross_domain_homotopy_returns_witness():
    witness = cross_domain_homotopy(
        "dft",
        {"n_electrons": 2, "ecutwfc": 50.0},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1, "architecture": "mlp"},
    )
    assert witness.source == "ManyBodySchrodinger"
    assert "energy_conservation" in witness.shared_invariants or not witness.equivalent
```

Append to `tests/integration/test_ml_cli.py`:

```python
def test_cli_ml_compare_with_dft():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "math_anything",
            "ml",
            "--input-dim",
            "2",
            "--output-dim",
            "1",
            "--compare-with",
            "dft",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert "dft_homotopy" in report
    assert "equivalent" in report["dft_homotopy"]
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/unit/test_topology_cross_domain.py tests/integration/test_ml_cli.py::test_cli_ml_compare_with_dft -v
```

Expected: ImportError / assertion failures.

- [ ] **Step 3: Implement cross-domain homotopy helper**

Create `math_anything/topology/cross_domain.py`:

```python
"""Cross-domain homotopy checking for Bourbaki domains."""

from __future__ import annotations

from typing import Any

from math_anything.categories.engine import CategoryEngine
from math_anything.domains import DOMAIN_REGISTRY
from math_anything.topology.homotopy import HomotopyWitness, are_paths_homotopic


def cross_domain_homotopy(
    domain_a_name: str,
    params_a: dict[str, Any],
    domain_b_name: str,
    params_b: dict[str, Any],
) -> HomotopyWitness:
    """Check whether two domain instantiation paths are homotopic.

    Builds a merged CategoryEngine from both domain morphism chains and compares
    the cumulative invariants along each path.
    """
    if domain_a_name not in DOMAIN_REGISTRY or domain_b_name not in DOMAIN_REGISTRY:
        available = sorted(DOMAIN_REGISTRY.keys())
        raise KeyError(f"Unknown domain. Available: {available}")

    dom_a = DOMAIN_REGISTRY[domain_a_name](params_a)
    dom_b = DOMAIN_REGISTRY[domain_b_name](params_b)

    chain_a = dom_a.build_morphism_chain()
    chain_b = dom_b.build_morphism_chain()

    ce = CategoryEngine()
    path_a: list[str] = []
    path_b: list[str] = []

    prev_a = "ManyBodySchrodinger" if domain_a_name == "dft" else f"{domain_a_name}_start"
    for i, step in enumerate(chain_a):
        name = f"a_{step['name']}"
        ce.register_morphism(type("M", (), {
            "name": name,
            "invariants_kept": step.get("invariants_kept", []),
            "invariants_lost": step.get("invariants_lost", []),
        })())
        target = f"a_state_{i}"
        ce.link(name, prev_a, target)
        path_a.append(name)
        prev_a = target

    prev_b = f"{domain_b_name}_start"
    for i, step in enumerate(chain_b):
        name = f"b_{step['name']}"
        ce.register_morphism(type("M", (), {
            "name": name,
            "invariants_kept": step.get("invariants_kept", []),
            "invariants_lost": step.get("invariants_lost", []),
        })())
        target = f"b_state_{i}"
        ce.link(name, prev_b, target)
        path_b.append(name)
        prev_b = target

    return are_paths_homotopic(ce, path_a, path_b)
```

- [ ] **Step 4: Update MCP tool**

Modify `analyze_ml_model` in `math_anything/mcp_server.py` to include:

```python
from math_anything.topology.cross_domain import cross_domain_homotopy

homotopy_witness = cross_domain_homotopy(
    "dft",
    {"n_electrons": 2},
    "supervised_learning",
    {"input_dim": input_dim, "output_dim": output_dim, "architecture": architecture},
)

report["dft_homotopy"] = {
    "equivalent": homotopy_witness.equivalent,
    "shared_invariants": homotopy_witness.shared_invariants,
    "confidence": homotopy_witness.confidence,
}
```

- [ ] **Step 5: Update CLI subcommand**

Add `--compare-with` argument to the `ml` parser:

```python
ml_parser.add_argument("--compare-with", type=str, default=None, help="Cross-domain homotopy comparison target")
```

In `cmd_ml`, after building the ML analysis, if `args.compare_with` is set:

```python
from math_anything.topology.cross_domain import cross_domain_homotopy

witness = cross_domain_homotopy(
    args.compare_with,
    {},
    "supervised_learning",
    {
        "input_dim": args.input_dim,
        "output_dim": args.output_dim,
        "architecture": args.architecture,
        "loss": args.loss,
    },
)
report["dft_homotopy"] = {
    "equivalent": witness.equivalent,
    "shared_invariants": witness.shared_invariants,
    "confidence": witness.confidence,
}
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/unit/test_topology_cross_domain.py tests/integration/test_ml_cli.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add math_anything/topology/cross_domain.py math_anything/mcp_server.py math_anything/cli.py tests/unit/test_topology_cross_domain.py tests/integration/test_ml_cli.py
git commit -m "feat(topology,ml): add cross-domain homotopy and wire into CLI/MCP"
```

---

## Self-Review

**Spec coverage:**
- Task 1: Real numpy MLP with SGD — covered.
- Task 2: Training-state capture and curvature — covered.
- Task 3: Cross-domain homotopy and MCP/CLI — covered.

**Placeholder scan:** No TBD/TODO.

**Type consistency:**
- `SequentialNetwork.forward/backward/sgd_step` operate on `np.ndarray`.
- `train_and_capture` returns `TrainingResult`.
- `cross_domain_homotopy` returns `HomotopyWitness`.

**Gaps:**
- This plan does not integrate real surrogate frameworks (DeePMD/MACE/CHGNet); it uses toy numpy networks.
- Backprop is limited to ReLU + MSE; tanh/sigmoid are not yet supported in `backward`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-10-bourbaki-ml-phase2-real-training-homotopy-plan.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
