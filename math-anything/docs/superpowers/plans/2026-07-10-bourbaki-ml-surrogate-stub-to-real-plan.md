# Bourbaki × ML Surrogate: STUB → REAL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Bourbaki computationally useful for machine learning by turning neural-network layers into morphisms, optimization trajectories into curvature objects, and ML surrogates into a first-class domain.

**Architecture:** Reuse the existing three-layer Bourbaki pattern. Layer 2 gets concrete NN morphisms (`LinearMorphism`, `ActivationMorphism`, `LossMorphism`) with real `apply()` methods. Layer 1 topology/curvature gets an optimization-trajectory curvature helper. Layer 3 gets a `SupervisedLearningDomain` registered alongside DFT/MD/CFD/FEM. Everything is exposed through CLI and MCP without new runtime dependencies.

**Tech Stack:** Python 3.10+, `numpy`, existing `math_anything` packages. No new runtime dependencies (no PyTorch/TensorFlow).

## Global Constraints

- No new external runtime dependencies.
- All public functions return serializable dataclasses, dicts, or primitive values.
- All NN morphisms must work with plain `numpy.ndarray` inputs.
- All code changes must be covered by unit tests; CLI/MCP changes by integration tests.
- Follow existing style: `ruff check math_anything/ tests/` must stay clean.
- Preserve backward compatibility where possible; breaking CLI changes must be documented in the task.

---

## Task 1: Neural-network layer morphisms

**Files:**
- Create: `math_anything/structures/neural_network.py`
- Modify: `math_anything/structures/__init__.py`
- Test: `tests/unit/test_structures_neural_network.py`

**Interfaces:**
- Consumes: `math_anything.morphisms.Morphism`, `numpy`.
- Produces:
  - `LinearMorphism(name, input_dim, output_dim)` with `apply(x) -> np.ndarray`
  - `ActivationMorphism(name, activation="relu")` with `apply(x) -> np.ndarray`
  - `LossMorphism(name, loss="mse")` with `apply((y_pred, y_true)) -> float`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from math_anything.structures.neural_network import (
    ActivationMorphism,
    LinearMorphism,
    LossMorphism,
)


def test_linear_morphism_shape():
    m = LinearMorphism(name="linear_1", input_dim=3, output_dim=2)
    x = np.array([1.0, 2.0, 3.0])
    y = m.apply(x)
    assert y.shape == (2,)


def test_activation_relu():
    m = ActivationMorphism(name="relu_1", activation="relu")
    x = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(m.apply(x), [0.0, 0.0, 2.0])


def test_loss_mse():
    m = LossMorphism(name="mse_loss", loss="mse")
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 2.5, 2.5])
    loss = m.apply((y_pred, y_true))
    assert pytest.approx(loss) == ((0.5**2 + 0.5**2 + 0.5**2) / 3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /d/math-anything/math-anything
python -m pytest tests/unit/test_structures_neural_network.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/unit/test_structures_neural_network.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Export from structures package**

Modify `math_anything/structures/__init__.py` to add:

```python
from math_anything.structures.neural_network import (
    ActivationMorphism,
    LinearMorphism,
    LossMorphism,
)

__all__ = [
    # existing entries ...
    "LinearMorphism",
    "ActivationMorphism",
    "LossMorphism",
]
```

- [ ] **Step 6: Commit**

```bash
git add math_anything/structures/neural_network.py math_anything/structures/__init__.py tests/unit/test_structures_neural_network.py
git commit -m "feat(structures): add neural-network layer morphisms"
```

---

## Task 2: Optimization-trajectory curvature

**Files:**
- Create: `math_anything/topology/training_curvature.py`
- Modify: `math_anything/topology/__init__.py`
- Test: `tests/unit/test_topology_training_curvature.py`

**Interfaces:**
- Consumes: `math_anything.topology.curvature.discrete_curvature`, `math_anything.topology.loop.Loop`.
- Produces:
  - `OptimizationState` dataclass
  - `trajectory_curvature(states: list[OptimizationState]) -> list[float]`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from math_anything.topology.training_curvature import (
    OptimizationState,
    trajectory_curvature,
)


def test_straight_trajectory_has_zero_curvature():
    states = [
        OptimizationState(step=0, loss=1.0, weights=np.array([0.0])),
        OptimizationState(step=1, loss=0.5, weights=np.array([1.0])),
        OptimizationState(step=2, loss=0.0, weights=np.array([2.0])),
    ]
    curvatures = trajectory_curvature(states)
    assert all(abs(c) < 1e-6 for c in curvatures)


def test_curved_trajectory_has_nonzero_curvature():
    states = [
        OptimizationState(step=0, loss=1.0, weights=np.array([0.0])),
        OptimizationState(step=1, loss=0.3, weights=np.array([1.0])),
        OptimizationState(step=2, loss=0.5, weights=np.array([1.5])),
        OptimizationState(step=3, loss=0.1, weights=np.array([2.0])),
    ]
    curvatures = trajectory_curvature(states)
    assert any(abs(c) > 1e-3 for c in curvatures)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/unit/test_topology_training_curvature.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement training curvature**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/unit/test_topology_training_curvature.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Export**

Modify `math_anything/topology/__init__.py`:

```python
from math_anything.topology.training_curvature import (
    OptimizationState,
    trajectory_curvature,
)

__all__ = [
    # existing entries ...
    "OptimizationState",
    "trajectory_curvature",
]
```

- [ ] **Step 6: Commit**

```bash
git add math_anything/topology/training_curvature.py math_anything/topology/__init__.py tests/unit/test_topology_training_curvature.py
git commit -m "feat(topology): add optimization-trajectory curvature"
```

---

## Task 3: Supervised-learning domain

**Files:**
- Create: `math_anything/domains/ml.py`
- Modify: `math_anything/domains/__init__.py`
- Test: `tests/unit/test_domains_ml.py`

**Interfaces:**
- Consumes: `math_anything.domains.base.Domain`, `math_anything.structures.neural_network.*`, `math_anything.morphisms.surrogate.MLSurrogateMorphism`.
- Produces: `SupervisedLearningDomain` registered as `"supervised_learning"`.

- [ ] **Step 1: Write the failing test**

```python
from math_anything.domains import DOMAIN_REGISTRY


def test_supervised_learning_domain_registered():
    assert "supervised_learning" in DOMAIN_REGISTRY


def test_supervised_learning_analysis():
    domain = DOMAIN_REGISTRY["supervised_learning"]({
        "input_dim": 2,
        "output_dim": 1,
        "architecture": "mlp",
    })
    analysis = domain.analyze()
    assert analysis.domain_name == "supervised_learning"
    assert "function_approximation" in analysis.conservation_field.get("equation_type", "")
    assert len(analysis.morphism_chain) > 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/unit/test_domains_ml.py -v
```

Expected: KeyError or assertion failure.

- [ ] **Step 3: Implement ML domain**

```python
"""Machine-learning domain as a Bourbaki instantiation."""

from __future__ import annotations

from typing import Any

from math_anything.domains import register_domain
from math_anything.domains.base import Domain


@register_domain("supervised_learning")
class SupervisedLearningDomain(Domain):
    """Supervised learning as a morphism chain over function spaces."""

    name = "supervised_learning"
    description = "Supervised learning — function approximation from data"
    equation_type = "function_approximation"
    default_params = {
        "input_dim": 2,
        "output_dim": 1,
        "architecture": "mlp",
        "loss": "mse",
        "activation": "relu",
    }

    def build_conservation_field(self) -> dict[str, Any]:
        return {
            "equation_type": "function_approximation",
            "conservation_laws": [
                "expected_risk_minimization",
                "gradient_flow",
            ],
            "symmetries": ["data_permutation_invariance"],
            "eigenvalues": [],
        }

    def build_morphism_chain(self) -> list[dict[str, Any]]:
        chain = [
            {
                "name": "data_sampling",
                "type": "restriction",
                "description": "Replace true distribution with finite dataset",
                "invariants_kept": ["empirical_risk"],
                "invariants_lost": ["true_risk", "population_distribution"],
                "invariants_introduced": ["finite_sample_noise", "generalization_gap"],
            },
            {
                "name": "feature_map",
                "type": "embedding",
                "description": "Embed raw inputs into representation space",
                "invariants_kept": ["input_topology"],
                "invariants_lost": ["raw_feature_semantics"],
                "invariants_introduced": ["learned_representation"],
            },
            {
                "name": f"model_{self.params.get('architecture', 'mlp')}",
                "type": "surrogate",
                "description": "Parametric function family approximating the target",
                "invariants_kept": ["differentiability"],
                "invariants_lost": ["true_target_function"],
                "invariants_introduced": ["approximation_error", "optimization_landscape"],
            },
            {
                "name": f"loss_{self.params.get('loss', 'mse')}",
                "type": "projection",
                "description": "Project predictions and targets onto scalar objective",
                "invariants_kept": ["differentiability"],
                "invariants_lost": ["full_prediction_state"],
                "invariants_introduced": ["gradient_direction"],
            },
            {
                "name": "optimizer_step",
                "type": "transformation",
                "description": "Update parameters along gradient direction",
                "invariants_kept": ["parameter_space"],
                "invariants_lost": ["exact_minimum"],
                "invariants_introduced": ["learning_rate_dependence", "convergence_dynamics"],
            },
        ]
        return chain
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/unit/test_domains_ml.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Ensure domain import**

`math_anything/domains/__init__.py` already imports domain modules; add at the bottom if not present:

```python
from math_anything.domains.ml import SupervisedLearningDomain
```

- [ ] **Step 6: Commit**

```bash
git add math_anything/domains/ml.py math_anything/domains/__init__.py tests/unit/test_domains_ml.py
git commit -m "feat(domains): add supervised learning domain"
```

---

## Task 4: MCP tool and CLI exposure

**Files:**
- Modify: `math_anything/mcp_server.py`
- Modify: `math_anything/cli.py`
- Test: `tests/test_mcp_server.py` (append to existing), `tests/integration/test_ml_cli.py`

**Interfaces:**
- Consumes: `SupervisedLearningDomain`, `trajectory_curvature`, `LinearMorphism`/`ActivationMorphism`/`LossMorphism`.
- Produces:
  - MCP tool `analyze_ml_model`
  - CLI subcommand `ml`

- [ ] **Step 1: Write the failing MCP test**

Append to `tests/test_mcp_server.py` in the appropriate location:

```python
def test_mcp_analyze_ml_model_tool_exists():
    tool_names = [t.name for t in mcp._tools]
    assert "analyze_ml_model" in tool_names


def test_mcp_analyze_ml_model_runs():
    result = analyze_ml_model(
        input_dim=2,
        output_dim=1,
        architecture="mlp",
    )
    report = json.loads(result)
    assert report["domain"] == "supervised_learning"
    assert "morphism_chain" in report
```

- [ ] **Step 2: Write the failing CLI integration test**

Create `tests/integration/test_ml_cli.py`:

```python
import json
import subprocess
import sys
from pathlib import Path


def test_cli_ml_subcommand_runs():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything", "ml", "--input-dim", "2", "--output-dim", "1"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["domain"] == "supervised_learning"


def test_cli_ml_mermaid_output():
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
            "--visualize",
            "mermaid",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "graph LR" in result.stdout
```

- [ ] **Step 3: Run to confirm failures**

```bash
python -m pytest tests/test_mcp_server.py::test_mcp_analyze_ml_model_tool_exists tests/test_mcp_server.py::test_mcp_analyze_ml_model_runs tests/integration/test_ml_cli.py -v
```

Expected: failures for missing tool/command.

- [ ] **Step 4: Add MCP tool**

In `math_anything/mcp_server.py`, add:

```python
@mcp.tool()
def analyze_ml_model(
    input_dim: int = 2,
    output_dim: int = 1,
    architecture: str = "mlp",
    loss: str = "mse",
) -> str:
    """Analyze a supervised-learning model as a morphism chain.

    Reveals which mathematical properties are preserved, lost, and introduced
    when approximating a target function with a neural network.
    """
    from math_anything.domains import DOMAIN_REGISTRY
    from math_anything.structures.neural_network import (
        ActivationMorphism,
        LinearMorphism,
        LossMorphism,
    )
    from math_anything.topology.training_curvature import (
        OptimizationState,
        trajectory_curvature,
    )

    domain = DOMAIN_REGISTRY["supervised_learning"]({
        "input_dim": input_dim,
        "output_dim": output_dim,
        "architecture": architecture,
        "loss": loss,
    })
    analysis = domain.analyze()

    # Demonstrate forward pass through a tiny network
    linear = LinearMorphism(name="linear_1", input_dim=input_dim, output_dim=output_dim)
    activation = ActivationMorphism(name="relu_1", activation="relu")
    loss_fn = LossMorphism(name="loss", loss=loss)

    x = [1.0] * input_dim
    y_pred = activation.apply(linear.apply(x))
    y_true = [0.0] * output_dim
    demo_loss = loss_fn.apply((y_pred, y_true))

    # Dummy optimization trajectory for curvature illustration
    states = [
        OptimizationState(step=0, loss=demo_loss * 1.5, weights=[0.0]),
        OptimizationState(step=1, loss=demo_loss, weights=[0.5]),
        OptimizationState(step=2, loss=demo_loss * 0.5, weights=[1.0]),
    ]
    curvatures = trajectory_curvature(states)

    report = {
        "domain": analysis.domain_name,
        "architecture": architecture,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "preserved": analysis.preserved,
        "lost": analysis.lost,
        "emerged": analysis.emerged,
        "morphism_chain": analysis.morphism_chain,
        "demo_forward_pass": {
            "input": x,
            "predicted": y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred,
            "loss": demo_loss,
        },
        "optimization_curvature": curvatures,
    }
    return json.dumps(report, indent=2, ensure_ascii=False, default=str)
```

- [ ] **Step 5: Add CLI subcommand**

In `math_anything/cli.py`, add parser and handler:

```python
    ml_parser = subparsers.add_parser(
        "ml",
        help="Analyze a supervised-learning model as a morphism chain",
    )
    ml_parser.add_argument("--input-dim", type=int, default=2)
    ml_parser.add_argument("--output-dim", type=int, default=1)
    ml_parser.add_argument("--architecture", type=str, default="mlp")
    ml_parser.add_argument("--loss", type=str, default="mse")
    ml_parser.add_argument("--visualize", choices=["mermaid"], default=None)
```

Implement `cmd_ml`:

```python
def cmd_ml(args: argparse.Namespace) -> int:
    """Analyze a supervised-learning model."""
    import json

    from math_anything.domains import DOMAIN_REGISTRY
    from math_anything.topology.visualization import to_mermaid

    try:
        domain = DOMAIN_REGISTRY["supervised_learning"]({
            "input_dim": args.input_dim,
            "output_dim": args.output_dim,
            "architecture": args.architecture,
            "loss": args.loss,
        })
        analysis = domain.analyze()

        report = {
            "domain": analysis.domain_name,
            "architecture": args.architecture,
            "input_dim": args.input_dim,
            "output_dim": args.output_dim,
            "preserved": analysis.preserved,
            "lost": analysis.lost,
            "emerged": analysis.emerged,
            "morphism_chain": analysis.morphism_chain,
        }

        if args.visualize == "mermaid":
            from math_anything.categories.engine import CategoryEngine

            ce = CategoryEngine()
            for step in analysis.morphism_chain:
                ce.register_morphism(type("M", (), {
                    "name": step["name"],
                    "source_type": "MLState",
                    "target_type": "MLState",
                })())
            prev = "Input"
            for step in analysis.morphism_chain:
                ce.link(step["name"], prev, step["name"])
                prev = step["name"]
            output = to_mermaid(ce)
        else:
            output = json.dumps(report, indent=2, ensure_ascii=False)

        from math_anything.utils.terminal import safe_print
        safe_print(output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
```

Add dispatch mapping for `"ml"`.

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_mcp_server.py::test_mcp_analyze_ml_model_tool_exists tests/test_mcp_server.py::test_mcp_analyze_ml_model_runs tests/integration/test_ml_cli.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add math_anything/mcp_server.py math_anything/cli.py tests/test_mcp_server.py tests/integration/test_ml_cli.py
git commit -m "feat(ml): expose supervised-learning domain via MCP and CLI"
```

---

## Self-Review

**Spec coverage:**
- Task 1: NN layer morphisms with real `apply()` — covered.
- Task 2: Optimization-trajectory curvature — covered.
- Task 3: Supervised-learning domain registered — covered.
- Task 4: MCP tool + CLI subcommand — covered.

**Placeholder scan:** No TBD/TODO. Every step includes concrete code and commands.

**Type consistency:**
- `LinearMorphism.apply` returns `np.ndarray`.
- `LossMorphism.apply` returns `float`.
- `trajectory_curvature` returns `list[float]`.
- `SupervisedLearningDomain.analyze()` returns `DomainAnalysis`.
- `analyze_ml_model` returns JSON string.

**Gaps:**
- This plan does not train real neural networks; it uses small numpy demos.
- No integration with actual surrogate frameworks (DeePMD, MACE, CHGNet) — that would require optional dependencies and is out of scope.
- The CLI `ml --visualize mermaid` builds a minimal category graph only for display.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-10-bourbaki-ml-surrogate-stub-to-real-plan.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
