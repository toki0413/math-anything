# Bourbaki ML Phases 3–5 Design

> **For agentic workers:** This is a multi-phase spec. Each phase can be implemented as a separate plan via `superpowers:writing-plans`. The phases are ordered by dependency: Phase 3 → Phase 4 → Phase 5.

## Goal

Extend the Bourbaki treatment of machine learning from "a single training trajectory" (Phase 2) to:

1. **Phase 3 — Optimization-landscape homotopy:** treat two training trajectories as paths in parameter-loss space and decide whether they live in the same basin / are homotopic.
2. **Phase 4 — Transfer learning as a natural transformation:** model "pre-train on source task, fine-tune on target task" as a natural transformation between two functors.
3. **Phase 5 — Real surrogate backbone integration:** complete the toy network (tanh/sigmoid backprop) and provide an adapter layer for DeePMD/MACE/CHGNet with a numpy fallback.

All work stays inside the existing three-layer architecture (Foundation → Structures → Domains), uses only `numpy` for the core, and preserves the STUB-to-REAL mandate.

---

## Global Constraints

- No new **runtime** dependencies. Optional frameworks (DeePMD-kit, MACE, CHGNet) may be detected at import time; the code must degrade gracefully to the numpy backend when absent.
- All public functions return serializable dataclasses, dicts, or primitive values.
- Any training example must finish in < 1 s on a laptop so it can run in unit tests.
- `ruff check math_anything/ tests/` must remain clean.
- Do not break Phase 1/2 public APIs (`SequentialNetwork`, `train_and_capture`, `trajectory_curvature`, `cross_domain_homotopy`, `analyze_ml_model`).

---

## Phase 3: Optimization-Landscape Homotopy

### Motivation

Phase 2 captures one training trajectory. In practice we care whether two trajectories (different seeds, learning rates, architectures) end up in the **same basin** of the loss landscape. Topology gives us a natural vocabulary: two paths are homotopic if they can be continuously deformed into one another without crossing a loss barrier. Computationally we approximate this by checking whether the two trajectories share the same source/target structures and preserve the same qualitative invariants.

### Components

#### 1. `TrainingPathMorphism`

A small morphism-like object representing one epoch (or a sub-sequence of epochs) of a training trajectory.

```python
@dataclass
class TrainingPathMorphism:
    name: str
    source_structure: str      # e.g. "params_epoch_0"
    target_structure: str      # e.g. "params_epoch_1"
    invariants_kept: list[str]
    invariants_lost: list[str]
```

Invariants are derived from the two endpoint `OptimizationState`s, for example:

- `loss_decreased` if `loss_next < loss_prev`
- `gradient_norm_small` if parameter update norm is below a threshold
- `monotonic_near_minimum` if curvature stays low

#### 2. `build_training_path(...)`

Convert a `TrainingResult` into a list of `TrainingPathMorphism`s suitable for `are_paths_homotopic`.

```python
def build_training_path(
    result: TrainingResult,
    name_prefix: str = "run",
) -> tuple[CategoryEngine, list[str]]:
    ...
```

The engine registers one morphism per epoch step and links them into a chain.

#### 3. `training_paths_homotopic(result_a, result_b) -> HomotopyWitness`

Builds two paths in a merged `CategoryEngine` and delegates to `are_paths_homotopic`. The source is the initial parameter structure, the target is the final parameter structure.

### CLI / MCP additions

- CLI: `bourbaki ml train --compare-paths 2 --lr 0.05 0.01` trains two networks and reports homotopy.
- MCP `analyze_ml_model`: add `optimization_landscape_homotopy` field when `compare_paths` is requested.

### Testing

- Two runs with **identical seed/hyperparameters** must be homotopic (`equivalent=True`).
- Two runs with **different initialization** on a convex toy problem should still be homotopic; on a non-convex problem they may not be.
- Deterministic seeds, < 1 s runtime.

---

## Phase 4: Transfer Learning as a Natural Transformation

### Motivation

Transfer learning is usually described procedurally. In the Bourbaki view it is a **natural transformation** between two functors:

- Functor `F`: source task → model parameters learned on the source.
- Functor `G`: target task → model parameters learned on the target.
- The natural transformation `η` maps a source-trained model to a useful initialization for the target task, and the square must commute for every object in the shared category.

This lets us ask formal questions: what invariants are preserved by `η`? What is lost? When is transfer valid?

### Components

#### 1. `Functor` ABC

```python
class Functor(ABC):
    @abstractmethod
    def map_object(self, obj: str) -> str: ...

    @abstractmethod
    def map_morphism(self, morphism: Any) -> Any: ...
```

#### 2. `TransferFunctor`

A concrete functor from a source domain chain to a target supervised-learning model state. It maps each source-domain structure to a target parameter structure and each source morphism to a model-building step.

```python
class TransferFunctor(Functor):
    def __init__(self, source_domain: str, target_params: dict[str, Any]):
        ...
```

#### 3. `NaturalTransformation` + `is_natural_transformation`

```python
@dataclass
class NaturalTransformation:
    components: dict[str, Any]   # object -> morphism


def is_natural_transformation(
    F: Functor,
    G: Functor,
    eta: NaturalTransformation,
    test_objects: list[str],
) -> bool:
    ...
```

The check verifies `G(f) ∘ η_X = η_Y ∘ F(f)` for a small set of test morphisms `f: X → Y`.

#### 4. `transfer_learn(...)` helper

Given a source-domain chain and a target dataset, produce a `TrainingResult` initialized via the natural transformation.

### CLI / MCP additions

- CLI: `bourbaki ml transfer --source-domain dft --target-task ...`
- MCP `analyze_ml_model`: optional `transfer` argument; report `natural_transformation_valid` and `preserved_invariants`.

### Testing

- Construct a trivial linear natural transformation where the square commutes; assert `is_natural_transformation(...)` is `True`.
- Construct a non-commuting transformation and assert `False`.
- Verify that transfer from `supervised_learning` to itself is the identity natural transformation.

---

## Phase 5: Real Surrogate Backbone Integration

### Motivation

The Phase 2 network is intentionally a toy. Phase 5 makes it production-shaped without adding runtime dependencies:

1. Complete backprop for all supported activations.
2. Define a backend adapter so real surrogate packages can be plugged in; ship a numpy fallback.

### Components

#### 1. Complete backprop

Extend `SequentialNetwork.backward` to handle:

- `tanh`: derivative `1 - tanh(x)^2`
- `sigmoid`: derivative `sigmoid(x) * (1 - sigmoid(x))`

Add a private `_activation_derivative(activation, pre_activation)` helper. Keep MSE as the default loss; if demand appears later, add cross-entropy in a follow-up.

#### 2. `SurrogateBackend` protocol

```python
class SurrogateBackend(Protocol):
    name: str

    def fit(self, dataset: list[tuple[np.ndarray, np.ndarray]]) -> None: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...
    def to_morphism_chain(self) -> list[dict[str, Any]]: ...
```

#### 3. Backends

- `NumpySurrogateBackend`: wraps `SequentialNetwork`; always available.
- `DeePMDBackend`, `MaceBackend`, `ChgnetBackend`: metadata-only stubs. They import the real frameworks inside `fit()` and raise a clear `ImportError` if missing. This keeps the registry useful without forcing installs.

#### 4. `SurrogateModel` facade

```python
class SurrogateModel:
    def __init__(self, backend: str = "numpy", **params):
        ...
```

The facade selects the backend by name, exposes `fit/predict/chain`, and can be used from the `supervised_learning` domain.

### CLI / MCP additions

- CLI: `bourbaki ml train --backend numpy|deepmd|mace|chgnet`
- MCP `analyze_ml_model`: optional `backend` parameter; report `backend_used` and `backend_available`.

### Testing

- Unit test `SequentialNetwork` on `tanh` and `sigmoid` reduces loss.
- Unit test `SurrogateModel("numpy", ...)` trains and predicts.
- Unit test unknown backend raises `ValueError`.
- Unit test `DeePMDBackend` raises informative `ImportError` when `deepmd` is absent.

---

## Architecture Overview

```text
Foundation
  └── categories/engine.py          (CategoryEngine, Functor)
  └── topology/homotopy.py          (are_paths_homotopic)

Structures
  └── structures/neural_network.py  (SequentialNetwork, extended backward)
  └── topology/training_curvature.py (OptimizationState, TrainingResult)
  └── topology/optimization_landscape.py  (NEW: TrainingPathMorphism, build_training_path)
  └── structures/functor.py         (NEW: Functor, TransferFunctor, NaturalTransformation)
  └── structures/surrogate_backend.py (NEW: SurrogateBackend, NumpySurrogateBackend, optional stubs)

Domains
  └── domains/ml.py                 (supervised_learning uses SurrogateModel, reports transfer/homotopy)

Interfaces
  └── cli.py                        (ml train --compare-paths, ml transfer, --backend)
  └── mcp_server.py                 (analyze_ml_model extended fields)
```

---

## Data Flow

### Phase 3

```text
Network A  →  train_and_capture  →  build_training_path  →
Network B  →  train_and_capture  →  build_training_path  →  CategoryEngine
                                                       →  are_paths_homotopic
                                                       →  HomotopyWitness
```

### Phase 4

```text
source_domain_chain  →  TransferFunctor F
                                      ↘
                                       NaturalTransformation η  →  is_natural_transformation
                                      ↗
target_domain_chain  →  TransferFunctor G
```

### Phase 5

```text
user selects backend  →  SurrogateModel
                              ↓
                   NumpySurrogateBackend  (or optional real framework)
                              ↓
                      fit / predict / chain
```

---

## Error Handling

- Unknown backend: raise `ValueError` with list of registered backends.
- Missing optional framework: raise `ImportError` with install instructions; CLI exits with code 1 and a concise message.
- Empty training result: return `HomotopyWitness(equivalent=False, confidence=0.0)` rather than crashing.
- Natural-transformation square failure: return `False` plus the first failing object/morphism for diagnostics.

---

## Testing Strategy

| Layer | Test file |
|-------|-----------|
| Phase 3 core | `tests/unit/test_topology_optimization_landscape.py` |
| Phase 4 core | `tests/unit/test_structures_functor.py` |
| Phase 5 core | `tests/unit/test_structures_neural_network_extended.py` (or extend existing) |
| Backend registry | `tests/unit/test_structures_surrogate_backend.py` |
| CLI integration | `tests/integration/test_ml_cli.py` |
| MCP integration | `tests/test_mcp_server.py` |

All new tests must run under the existing `pytest` command and remain < 1 s individually.

---

## Risks & Gaps

- **Convex toy data** may make every training path homotopic, which is correct but not visually exciting. We can add a tiny non-convex problem (e.g. XOR with two minima) for Phase 3 demonstration.
- **Natural transformation** is a fairly abstract concept; the implementation must stay concrete and testable with linear maps.
- **Optional backends** are stubs unless the user installs the real packages. We will not ship wrappers for full training pipelines because that would explode scope.

---

## Milestones

1. **Phase 3** — `optimization_landscape_homotopy` works end-to-end from CLI and MCP.
2. **Phase 4** — `TransferFunctor` and `is_natural_transformation` are tested and wired into `analyze_ml_model`.
3. **Phase 5** — `tanh`/`sigmoid` backprop passes; backend registry + numpy facade passes; optional stubs raise clear errors.

---

## Approval

This design is ready for implementation planning. Each phase should get its own `docs/superpowers/plans/YYYY-MM-DD-...-plan.md` produced by `superpowers:writing-plans`.
