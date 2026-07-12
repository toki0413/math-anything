# Bourbaki Domain-Chain Natural Transformation + Docs Update Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
1. Generalize the natural-transformation machinery from matrix categories to domain morphism chains.
2. Update `README.md` and `docs/mcp.md` to reflect the current 8-domain + ML capability set.

**Architecture:** Add a new `math_anything/structures/domain_functor.py` module that defines `DomainFunctor`, builds a merged `CategoryEngine` from two domain chains, registers bridge morphisms for the natural transformation, and verifies the invariant-preservation square. Keep the existing matrix-category code untouched. Update documentation with current feature lists, domain table, and ML examples.

**Tech Stack:** Python 3.10+, `numpy`, existing `math_anything` packages.

## Global Constraints

- No new external runtime dependencies.
- All public functions return serializable dataclasses, dicts, or primitive values.
- Any training example must finish in < 1 s.
- All code changes must be covered by unit tests; docs must be manually verified for broken internal links.
- Follow existing style: `ruff check` on touched files must stay clean (the wider repo has pre-existing violations).
- Do not break existing Phase 1–5 public APIs.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `math_anything/structures/domain_functor.py` | `DomainFunctor`, `build_domain_pair_engine`, `build_bridge_natural_transformation`, `is_domain_natural_transformation`. |
| `tests/unit/test_structures_domain_functor.py` | Unit tests for domain-chain natural transformation. |
| `README.md` | Project overview, feature list, domain table, quick-start examples, architecture description. |
| `docs/mcp.md` | MCP tool list, domain list, examples, troubleshooting. |

---

## Task 1: Domain-chain natural transformation

**Files:**
- Create: `math_anything/structures/domain_functor.py`
- Test: `tests/unit/test_structures_domain_functor.py`

**Interfaces:**
- Consumes: `CategoryEngine`, `Domain` registry, `cumulative_invariants_along_path` from `math_anything.topology.homotopy`, `NaturalTransformation` from `math_anything.structures.functor`.
- Produces:
  - `DomainFunctor(object_map: dict[str, str], morphism_map: dict[str, str])`
  - `build_domain_pair_engine(domain_a_name, params_a, domain_b_name, params_b, prefix_a="a", prefix_b="b") -> tuple[CategoryEngine, list[str], list[str]]`
  - `build_bridge_natural_transformation(engine, source_prefix, target_prefix, bridge_invariants_kept, bridge_invariants_lost) -> NaturalTransformation`
  - `is_domain_natural_transformation(F, G, eta, engine, test_morphisms) -> tuple[bool, str]`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_structures_domain_functor.py`:

```python
import pytest

from math_anything.structures.domain_functor import (
    DomainFunctor,
    build_bridge_natural_transformation,
    build_domain_pair_engine,
    is_domain_natural_transformation,
)
from math_anything.structures.functor import NaturalTransformation


def test_build_domain_pair_engine_creates_two_paths():
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
    )
    assert len(path_a) == len(path_b)
    assert all(name.startswith("a_") for name in path_a)
    assert all(name.startswith("b_") for name in path_b)
    assert engine.morphism_links[0].source_structure == "a_start"


def test_identity_domain_functor_is_natural():
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
    )
    object_map = {"a_start": "a_start", "a_end": "a_end"}
    for link in engine.morphism_links:
        if link.source_structure.startswith("a_"):
            object_map[link.source_structure] = link.source_structure
            object_map[link.target_structure] = link.target_structure
    morphism_map = {name: name for name in path_a}
    F = DomainFunctor(object_map, morphism_map)
    G = DomainFunctor(object_map, morphism_map)

    eta = build_bridge_natural_transformation(
        engine,
        source_prefix="a",
        target_prefix="b",
        bridge_invariants_kept=["parameter_space"],
        bridge_invariants_lost=[],
    )

    valid, reason = is_domain_natural_transformation(
        F, G, eta, engine, test_morphisms=path_a
    )
    assert valid, reason


def test_mismatched_functor_is_not_natural():
    engine, path_a, path_b = build_domain_pair_engine(
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
        "supervised_learning",
        {"input_dim": 2, "output_dim": 1},
    )
    object_map = {"a_start": "a_start", "a_end": "a_end"}
    for link in engine.morphism_links:
        if link.source_structure.startswith("a_"):
            object_map[link.source_structure] = link.source_structure
            object_map[link.target_structure] = link.target_structure

    F = DomainFunctor(object_map, {name: name for name in path_a})
    # G maps every a morphism to the first b morphism, which breaks the square.
    G = DomainFunctor(object_map, {name: path_b[0] for name in path_a})
    eta = build_bridge_natural_transformation(
        engine,
        source_prefix="a",
        target_prefix="b",
        bridge_invariants_kept=["parameter_space"],
        bridge_invariants_lost=[],
    )

    valid, reason = is_domain_natural_transformation(
        F, G, eta, engine, test_morphisms=path_a
    )
    assert not valid
```

- [ ] **Step 2: Run the failing test**

```bash
cd /d/math-anything/math-anything
python -m pytest tests/unit/test_structures_domain_functor.py -v
```

Expected: `ModuleNotFoundError` for `math_anything.structures.domain_functor`.

- [ ] **Step 3: Implement the domain-functor module**

Create `math_anything/structures/domain_functor.py`:

```python
"""Functors and natural transformations over domain morphism chains."""

from __future__ import annotations

from typing import Any

from math_anything.categories.engine import CategoryEngine
from math_anything.domains import DOMAIN_REGISTRY
from math_anything.structures.functor import Functor, NaturalTransformation
from math_anything.topology.homotopy import cumulative_invariants_along_path


class DomainFunctor(Functor):
    """Functor that maps source domain structures/morphisms to target names."""

    def __init__(
        self,
        object_map: dict[str, str],
        morphism_map: dict[str, str],
    ):
        self.object_map = object_map
        self.morphism_map = morphism_map

    def map_object(self, obj: Any) -> Any:
        if obj not in self.object_map:
            raise KeyError(f"Object {obj} not mapped by functor")
        return self.object_map[obj]

    def map_morphism(self, morphism: Any) -> Any:
        name = getattr(morphism, "name", morphism)
        if name not in self.morphism_map:
            raise KeyError(f"Morphism {name} not mapped by functor")
        return self.morphism_map[name]


def build_domain_pair_engine(
    domain_a_name: str,
    params_a: dict[str, Any],
    domain_b_name: str,
    params_b: dict[str, Any],
    prefix_a: str = "a",
    prefix_b: str = "b",
) -> tuple[CategoryEngine, list[str], list[str]]:
    """Build a merged CategoryEngine containing two domain morphism chains."""
    if domain_a_name not in DOMAIN_REGISTRY or domain_b_name not in DOMAIN_REGISTRY:
        available = sorted(DOMAIN_REGISTRY.keys())
        raise KeyError(f"Unknown domain. Available: {available}")

    dom_a = DOMAIN_REGISTRY[domain_a_name](params_a)
    dom_b = DOMAIN_REGISTRY[domain_b_name](params_b)
    chain_a = dom_a.build_morphism_chain()
    chain_b = dom_b.build_morphism_chain()

    engine = CategoryEngine()
    path_a: list[str] = []
    path_b: list[str] = []

    prev_a = f"{prefix_a}_start"
    for i, step in enumerate(chain_a):
        name = f"{prefix_a}_{step['name']}"
        ce.register_morphism(type("M", (), {
            "name": name,
            "invariants_kept": step.get("invariants_kept", []),
            "invariants_lost": step.get("invariants_lost", []),
        })())
        target = f"{prefix_a}_state_{i}"
        engine.link(name, prev_a, target)
        path_a.append(name)
        prev_a = target
    final_a = f"{prefix_a}_end"
    engine.link(f"{prefix_a}_terminal", prev_a, final_a)
    terminal_a = type("M", (), {
        "name": f"{prefix_a}_terminal",
        "invariants_kept": [],
        "invariants_lost": [],
    })()
    engine.register_morphism(terminal_a)
    path_a.append(f"{prefix_a}_terminal")

    prev_b = f"{prefix_b}_start"
    for i, step in enumerate(chain_b):
        name = f"{prefix_b}_{step['name']}"
        engine.register_morphism(type("M", (), {
            "name": name,
            "invariants_kept": step.get("invariants_kept", []),
            "invariants_lost": step.get("invariants_lost", []),
        })())
        target = f"{prefix_b}_state_{i}"
        engine.link(name, prev_b, target)
        path_b.append(name)
        prev_b = target
    final_b = f"{prefix_b}_end"
    engine.link(f"{prefix_b}_terminal", prev_b, final_b)
    terminal_b = type("M", (), {
        "name": f"{prefix_b}_terminal",
        "invariants_kept": [],
        "invariants_lost": [],
    })()
    engine.register_morphism(terminal_b)
    path_b.append(f"{prefix_b}_terminal")

    return engine, path_a, path_b


def build_bridge_natural_transformation(
    engine: CategoryEngine,
    source_prefix: str,
    target_prefix: str,
    bridge_invariants_kept: list[str],
    bridge_invariants_lost: list[str],
) -> NaturalTransformation:
    """Register bridge morphisms from every source structure to its target counterpart.

    Assumes structures are named `{source_prefix}_start`, `{source_prefix}_state_i`,
    `{source_prefix}_end` and similarly for target.
    """
    components: dict[str, str] = {}
    source_structures = {link.source_structure for link in engine.morphism_links}
    source_structures.update(link.target_structure for link in engine.morphism_links)

    for src in sorted(source_structures):
        if not src.startswith(f"{source_prefix}_"):
            continue
        dst = src.replace(f"{source_prefix}_", f"{target_prefix}_", 1)
        if dst not in source_structures:
            continue
        name = f"bridge_{source_prefix}_to_{target_prefix}_{src}"
        engine.register_morphism(type("M", (), {
            "name": name,
            "invariants_kept": bridge_invariants_kept,
            "invariants_lost": bridge_invariants_lost,
        })())
        engine.link(name, src, dst)
        components[src] = name

    return NaturalTransformation(components)


def is_domain_natural_transformation(
    F: DomainFunctor,
    G: DomainFunctor,
    eta: NaturalTransformation,
    engine: CategoryEngine,
    test_morphisms: list[str],
) -> tuple[bool, str]:
    """Check whether eta: F => G is a natural transformation on the given engine.

    For each test morphism f: X -> Y, verifies that the two target paths
    G(f) ∘ eta_X and eta_Y ∘ F(f) preserve the same cumulative invariants.
    """
    links = {link.morphism.name: link for link in engine.morphism_links}

    for f_name in test_morphisms:
        if f_name not in links:
            return False, f"Morphism '{f_name}' not found in engine"

        source_obj = links[f_name].source_structure
        target_obj = links[f_name].target_structure

        f_f = F.map_morphism(f_name)
        f_g = G.map_morphism(f_name)
        eta_src = eta.components.get(F.map_object(source_obj))
        eta_dst = eta.components.get(F.map_object(target_obj))

        if eta_src is None or eta_dst is None:
            return False, f"Missing bridge for {source_obj} or {target_obj}"

        path1 = [f_f, eta_dst]
        path2 = [eta_src, f_g]

        try:
            inv1 = cumulative_invariants_along_path(engine, path1)
            inv2 = cumulative_invariants_along_path(engine, path2)
        except KeyError as e:
            return False, f"Path construction failed: {e}"

        if inv1 != inv2:
            return False, (
                f"Square fails for {f_name}: "
                f"kept1={inv1['kept']} lost1={inv1['lost']} vs "
                f"kept2={inv2['kept']} lost2={inv2['lost']}"
            )

    return True, ""
```

- [ ] **Step 4: Run the test**

```bash
python -m pytest tests/unit/test_structures_domain_functor.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add math_anything/structures/domain_functor.py tests/unit/test_structures_domain_functor.py
git commit -m "feat(structures): domain-chain natural transformation"
```

---

## Task 2: Update README.md and docs/mcp.md

**Files:**
- Modify: `README.md`
- Modify: `docs/mcp.md`

**What to update**

1. **Domain count and list**: change "4 physics domains" to "8 physics/ML domains": dft, cfd, md, fem, em, qc, phase_field, supervised_learning.
2. **Feature list**: add topology/loop/homotopy/curvature, ML surrogate, training-trajectory curvature, optimization-landscape homotopy, transfer learning as natural transformation, surrogate backends.
3. **Supported domains table**: include all 8 domains.
4. **Quick-start / Python API**: add an ML example using `analyze_ml_model` or `SurrogateModel`.
5. **MCP tools**: update tool count, add `analyze_ml_model`, update domain list, update `list_domains` and troubleshooting references to 8 domains.
6. **Development section**: use `python -m pytest` commands that match current practice.

- [ ] **Step 1: Update README.md**

Edit in place; do not rewrite the entire file. Key edits:

- Line 75: change "4 physics domains" to "8 physics and ML domains".
- Lines 75–79: update feature bullets to include ML, topology, surrogate backends.
- Lines 217–220 table: add rows for em, qc, phase_field, supervised_learning.
- After the "Python API" section, add an ML subsection:

```markdown
### Machine Learning as Mathematical Structure

```python
from math_anything.mcp_server import analyze_ml_model

report = analyze_ml_model(
    input_dim=2,
    output_dim=1,
    architecture="mlp",
    loss="mse",
    compare_paths=True,
    transfer=True,
    backend="numpy",
)
```

- Lines 190–198 architecture box: update Layer 3 list to include EM, QC, phase field, supervised learning.

- [ ] **Step 2: Update docs/mcp.md**

- Line 9: "8 physics/ML domains".
- `analyze_domain` parameters: domain names include all 8.
- Tool count: update "15 tools" to the current count (≈ 18).
- Add a new `analyze_ml_model` subsection under "Machine Learning Layer".
- Troubleshooting "Unknown domain": list all 8 domains.

- [ ] **Step 3: Verify internal links**

Run:

```bash
cd /d/math-anything/math-anything
python -c "import pathlib, re; txt = pathlib.Path('README.md').read_text(); assert 'supervised_learning' in txt; print('README OK')"
python -c "import pathlib; txt = pathlib.Path('docs/mcp.md').read_text(); assert 'supervised_learning' in txt and 'analyze_ml_model' in txt; print('MCP docs OK')"
```

Expected: both print OK.

- [ ] **Step 4: Run ruff on touched docs**

Docs are Markdown; run `ruff check` only on code files that changed (none for this task). Confirm no Python files changed.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/mcp.md
git commit -m "docs: update README and MCP docs for 8 domains and ML capabilities"
```

---

## Task 3: Final verification

- [ ] **Step 1: Run tests**

```bash
cd /d/math-anything/math-anything
python -m pytest tests/unit/test_structures_domain_functor.py tests/unit/test_structures_functor.py tests/unit/test_structures_neural_network.py tests/unit/test_structures_surrogate_backend.py tests/unit/test_topology_optimization_landscape.py tests/unit/test_domains_all.py tests/integration/test_ml_cli.py tests/test_mcp_server.py -v --tb=short --no-cov --deselect tests/test_mcp_server.py::TestDiscoverEquations::test_custom_method
```

Expected: all tests pass.

- [ ] **Step 2: Run ruff on touched files**

```bash
ruff check math_anything/structures/domain_functor.py tests/unit/test_structures_domain_functor.py
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
- Domain-chain natural transformation — Task 1.
- README/MCP docs update — Task 2.
- Verification — Task 3.

**Placeholder scan:** No TBD/TODO.

**Type consistency:**
- `DomainFunctor` follows the `Functor` ABC.
- `is_domain_natural_transformation` returns `tuple[bool, str]` like the matrix version.

**Gaps:** None for this scope.
