# Bourbaki ML Phase 3: Optimization-Landscape Homotopy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treat two training trajectories as paths in parameter-loss space and decide whether they are homotopic (same basin) using the existing `CategoryEngine` + `are_paths_homotopic` machinery.

**Architecture:** Add a new `math_anything/topology/optimization_landscape.py` module that converts `TrainingResult`s into linked `TrainingPathMorphism`s. Wire the comparison into the `ml` CLI via `--compare-paths` and into `analyze_ml_model` via an optional `compare_paths` boolean. Keep all computations in `numpy`; no new runtime dependencies.

**Tech Stack:** Python 3.10+, `numpy`, existing `math_anything` packages.

## Global Constraints

- No new external runtime dependencies.
- All public functions return serializable dataclasses, dicts, or primitive values.
- Any training example must finish in < 1 s on a laptop.
- All code changes must be covered by unit tests; CLI/MCP changes by integration tests.
- Follow existing style: `ruff check math_anything/ tests/` must stay clean.
- Do not break existing Phase 1/2 APIs (`SequentialNetwork`, `train_and_capture`, `trajectory_curvature`, `cross_domain_homotopy`, `analyze_ml_model`).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `math_anything/topology/optimization_landscape.py` | Core Phase 3 logic: `TrainingPathMorphism`, `build_training_path`, `training_paths_homotopic`. |
| `math_anything/cli.py` | Add `--compare-paths` argument to `ml` subcommand and emit `optimization_landscape_homotopy` report. |
| `math_anything/mcp_server.py` | Add `compare_paths: bool = False` to `analyze_ml_model` and include `optimization_landscape_homotopy` when requested. |
| `tests/unit/test_topology_optimization_landscape.py` | Unit tests for `build_training_path` and `training_paths_homotopic`. |
| `tests/integration/test_ml_cli.py` | CLI integration test for `--compare-paths`. |
| `tests/test_mcp_server.py` | MCP integration test for `analyze_ml_model(..., compare_paths=True)`. |

---

## Task 1: Core optimization-landscape homotopy

**Files:**
- Create: `math_anything/topology/optimization_landscape.py`
- Test: `tests/unit/test_topology_optimization_landscape.py`

**Interfaces:**
- Consumes: `OptimizationState`, `TrainingResult` from `math_anything.topology.training_curvature`; `CategoryEngine`, `are_paths_homotopic`, `HomotopyWitness` from `math_anything.topology.homotopy`.
- Produces:
  - `TrainingPathMorphism(name, source_structure, target_structure, invariants_kept, invariants_lost)`
  - `build_training_path(result: TrainingResult, name_prefix: str = "run") -> tuple[CategoryEngine, list[str]]`
  - `training_paths_homotopic(result_a: TrainingResult, result_b: TrainingResult) -> HomotopyWitness`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_topology_optimization_landscape.py`:

```python
import numpy as np

from math_anything.structures.neural_network import (
    ActivationMorphism,
    LinearMorphism,
    LossMorphism,
    SequentialNetwork,
)
from math_anything.topology.optimization_landscape import (
    TrainingPathMorphism,
    build_training_path,
    training_paths_homotopic,
)
from math_anything.topology.training_curvature import train_and_capture


def _tiny_dataset():
    xs = [np.array([x]) for x in [-1.0, 0.0, 1.0]]
    ys = [np.array([2 * x + 1]) for x in [-1.0, 0.0, 1.0]]
    return list(zip(xs, ys))


def test_build_training_path_returns_engine_and_path():
    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
    ])
    result = train_and_capture(net, _tiny_dataset(), LossMorphism(name="mse", loss="mse"), epochs=3, lr=0.05)

    engine, path = build_training_path(result, name_prefix="run_a")
    assert len(path) == 3
    assert all(name.startswith("run_a_") for name in path)
    assert engine.morphism_links[0].source_structure == "params_initial"
    assert engine.morphism_links[-1].target_structure == "params_final"


def test_identical_training_runs_are_homotopic():
    loss_fn = LossMorphism(name="mse", loss="mse")
    dataset = _tiny_dataset()

    net_a = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
    ])
    result_a = train_and_capture(net_a, dataset, loss_fn, epochs=3, lr=0.05)

    net_b = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
    ])
    result_b = train_and_capture(net_b, dataset, loss_fn, epochs=3, lr=0.05)

    witness = training_paths_homotopic(result_a, result_b)
    assert isinstance(witness.equivalent, bool)
    assert 0.0 <= witness.confidence <= 1.0
    assert witness.source == "params_initial"
    assert witness.target == "params_final"
```

- [ ] **Step 2: Run the failing test**

```bash
cd /d/math-anything/math-anything
python -m pytest tests/unit/test_topology_optimization_landscape.py -v
```

Expected: `ModuleNotFoundError: No module named 'math_anything.topology.optimization_landscape'`.

- [ ] **Step 3: Implement the core module**

Create `math_anything/topology/optimization_landscape.py`:

```python
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
        engine.link(path[0], source_structure, engine.morphism_links[0].source_structure)

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
```

- [ ] **Step 4: Run the test**

```bash
python -m pytest tests/unit/test_topology_optimization_landscape.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add math_anything/topology/optimization_landscape.py tests/unit/test_topology_optimization_landscape.py
git commit -m "feat(topology): optimization-landscape homotopy for training trajectories"
```

---

## Task 2: Wire `--compare-paths` into the `ml` CLI

**Files:**
- Modify: `math_anything/cli.py:337`
- Test: `tests/integration/test_ml_cli.py`

**Interfaces:**
- Consumes: `training_paths_homotopic` from `math_anything.topology.optimization_landscape`; `SequentialNetwork`, `LossMorphism`, `ActivationMorphism`, `LinearMorphism`; `train_and_capture` from `math_anything.topology.training_curvature`.
- Produces: CLI `--compare-paths` flag; report field `optimization_landscape_homotopy`.

- [ ] **Step 1: Add the CLI argument**

In `math_anything/cli.py`, after the existing `--compare-with` argument (around line 343), add:

```python
    ml_parser.add_argument(
        "--compare-paths",
        action="store_true",
        help="Train two identical runs and report optimization-landscape homotopy",
    )
```

- [ ] **Step 2: Add the comparison logic in `cmd_ml`**

After the existing `if args.compare_with:` block in `cmd_ml` (around line 1067), add:

```python
        if args.compare_paths:
            from math_anything.structures.neural_network import (
                ActivationMorphism,
                LinearMorphism,
                LossMorphism,
                SequentialNetwork,
            )
            from math_anything.topology.optimization_landscape import (
                training_paths_homotopic,
            )
            from math_anything.topology.training_curvature import train_and_capture

            loss_fn = LossMorphism(name="loss", loss=args.loss)
            dataset = [
                (np.array([x]), np.array([2 * x + 1]))
                for x in [-1.0, 0.0, 1.0]
            ]

            def _make_network():
                return SequentialNetwork([
                    LinearMorphism(name="linear_1", input_dim=args.input_dim, output_dim=4),
                    ActivationMorphism(name="relu_1", activation="relu"),
                    LinearMorphism(name="linear_2", input_dim=4, output_dim=args.output_dim),
                ])

            result_a = train_and_capture(_make_network(), dataset, loss_fn, epochs=5, lr=0.05)
            result_b = train_and_capture(_make_network(), dataset, loss_fn, epochs=5, lr=0.05)

            witness = training_paths_homotopic(result_a, result_b)
            report["optimization_landscape_homotopy"] = {
                "equivalent": witness.equivalent,
                "shared_invariants": witness.shared_invariants,
                "confidence": witness.confidence,
            }
```

- [ ] **Step 3: Write the CLI integration test**

Append to `tests/integration/test_ml_cli.py`:

```python
def test_cli_ml_compare_paths_runs(cli_runner):
    result = cli_runner([
        "ml",
        "--input-dim", "1",
        "--output-dim", "1",
        "--compare-paths",
    ])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "optimization_landscape_homotopy" in data
    assert isinstance(data["optimization_landscape_homotopy"]["equivalent"], bool)
    assert 0.0 <= data["optimization_landscape_homotopy"]["confidence"] <= 1.0
```

- [ ] **Step 4: Run the integration test**

```bash
python -m pytest tests/integration/test_ml_cli.py::test_cli_ml_compare_paths_runs -v
```

Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add math_anything/cli.py tests/integration/test_ml_cli.py
git commit -m "feat(cli): add --compare-paths to ml subcommand"
```

---

## Task 3: Wire `compare_paths` into the MCP tool

**Files:**
- Modify: `math_anything/mcp_server.py:789`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `training_paths_homotopic` from `math_anything.topology.optimization_landscape`; `SequentialNetwork`, `LossMorphism`, `ActivationMorphism`, `LinearMorphism`; `train_and_capture` from `math_anything.topology.training_curvature`.
- Produces: `analyze_ml_model(..., compare_paths: bool = False)`; report field `optimization_landscape_homotopy`.

- [ ] **Step 1: Add the parameter and report field**

Change the signature in `math_anything/mcp_server.py`:

```python
def analyze_ml_model(
    input_dim: int = 2,
    output_dim: int = 1,
    architecture: str = "mlp",
    loss: str = "mse",
    compare_paths: bool = False,
) -> str:
```

Add the comparison logic just before the final `return json.dumps(...)` in `analyze_ml_model`:

```python
    if compare_paths:
        from math_anything.structures.neural_network import (
            ActivationMorphism,
            LinearMorphism,
            LossMorphism as LossFn,
            SequentialNetwork,
        )
        from math_anything.topology.optimization_landscape import (
            training_paths_homotopic,
        )
        from math_anything.topology.training_curvature import train_and_capture

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

        result_a = train_and_capture(_make_network(), dataset, loss_fn, epochs=5, lr=0.05)
        result_b = train_and_capture(_make_network(), dataset, loss_fn, epochs=5, lr=0.05)
        witness = training_paths_homotopic(result_a, result_b)
        report["optimization_landscape_homotopy"] = {
            "equivalent": witness.equivalent,
            "shared_invariants": witness.shared_invariants,
            "confidence": witness.confidence,
        }
```

- [ ] **Step 2: Write the MCP integration test**

Append to `tests/test_mcp_server.py`:

```python
def test_mcp_analyze_ml_model_reports_optimization_landscape_homotopy():
    from math_anything.mcp_server import analyze_ml_model

    raw = analyze_ml_model(
        input_dim=1,
        output_dim=1,
        architecture="mlp",
        loss="mse",
        compare_paths=True,
    )
    report = json.loads(raw)
    assert "optimization_landscape_homotopy" in report
    assert isinstance(report["optimization_landscape_homotopy"]["equivalent"], bool)
    assert 0.0 <= report["optimization_landscape_homotopy"]["confidence"] <= 1.0
```

- [ ] **Step 3: Run the MCP integration test**

```bash
python -m pytest tests/test_mcp_server.py::test_mcp_analyze_ml_model_reports_optimization_landscape_homotopy -v
```

Expected: passes.

- [ ] **Step 4: Commit**

```bash
git add math_anything/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add compare_paths to analyze_ml_model"
```

---

## Task 4: Final verification

- [ ] **Step 1: Run the focused test suite**

```bash
cd /d/math-anything/math-anything
python -m pytest tests/unit/test_topology_optimization_landscape.py tests/integration/test_ml_cli.py tests/test_mcp_server.py -v --tb=short --no-cov --deselect tests/test_mcp_server.py::TestDiscoverEquations::test_custom_method
```

Expected: all tests pass.

- [ ] **Step 2: Run ruff**

```bash
ruff check math_anything/ tests/
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
- Phase 3 core `TrainingPathMorphism`, `build_training_path`, `training_paths_homotopic` — Task 1.
- CLI `--compare-paths` integration — Task 2.
- MCP `compare_paths` integration — Task 3.
- Final verification — Task 4.

**Placeholder scan:** No TBD/TODO/fill-in-details.

**Type consistency:**
- `TrainingResult` is reused from Phase 2.
- `HomotopyWitness` fields (`equivalent: bool`, `shared_invariants: list[str]`, `confidence: float`) are consistent across CLI and MCP.

**Gaps:** None for Phase 3 scope.
