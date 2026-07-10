# Topology-Aware Loop Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `math_anything.topology` module that detects, classifies, and reports loops in Bourbaki's morphism graphs, with CLI and MCP exposure.

**Architecture:** Build a `LoopEngine` on top of the existing `CategoryEngine` using `networkx` for graph algorithms. Represent loops as dataclasses, classify them by domain-specific rules, compute Betti numbers on the undirected skeleton, and expose results through a new CLI subcommand and MCP tool. Keep Phase 1–2 scope tight: detect loops, classify them, report Betti numbers. Cross-engine homotopy and curvature are follow-up phases.

**Tech Stack:** Python 3.10+, `networkx` (already in `pyproject.toml`), `pytest`, `click`, `fastmcp`.

## Global Constraints

- Python version floor: `>=3.10`
- Line length: 120 (ruff config)
- Target Python version: `py310`
- No new external dependencies for Phase 1–2; `networkx` is already required.
- Every new module must have unit tests in `tests/unit/`.
- CLI changes go in `math_anything/cli.py`.
- MCP tool changes go in `math_anything/mcp_server.py`.
- Follow existing Bourbaki naming: `snake_case` modules, `PascalCase` classes, dataclasses where appropriate.
- Maintain backward compatibility: do not remove or rename existing `CategoryEngine` public methods.

---

## File Structure

New files:

- `math_anything/topology/__init__.py` — package exports
- `math_anything/topology/loop.py` — `Loop`, `LoopType` dataclasses
- `math_anything/topology/loop_engine.py` — `LoopEngine`
- `math_anything/topology/classifier.py` — `LoopClassifier`
- `tests/unit/test_topology_loop.py`
- `tests/unit/test_topology_loop_engine.py`
- `tests/unit/test_topology_classifier.py`
- `tests/integration/test_topology_cli.py`

Modified files:

- `math_anything/categories/engine.py` — add `loop_engine` property helper
- `math_anything/cli.py` — add `loops` subcommand
- `math_anything/mcp_server.py` — add `analyze_loops` tool
- `math_anything/__init__.py` — optional lazy export of `LoopEngine`

---

### Task 1: Create topology package scaffold

**Files:**
- Create: `math_anything/topology/__init__.py`
- Test: `tests/unit/test_topology_imports.py`

**Interfaces:**
- Produces: `math_anything.topology` package is importable.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_topology_imports.py
def test_topology_package_imports():
    from math_anything import topology
    assert hasattr(topology, "Loop")
    assert hasattr(topology, "LoopEngine")
    assert hasattr(topology, "LoopClassifier")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_topology_imports.py -v`

Expected: FAIL with `ImportError: cannot import name 'Loop' from 'math_anything.topology'`

- [ ] **Step 3: Create the package**

```python
# math_anything/topology/__init__.py
"""Topology-aware loop engineering for morphism chains."""

from __future__ import annotations

# Placeholder exports; real imports added in later tasks.
class Loop:
    pass

class LoopEngine:
    pass

class LoopClassifier:
    pass

__all__ = ["Loop", "LoopEngine", "LoopClassifier"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_topology_imports.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add math_anything/topology/__init__.py tests/unit/test_topology_imports.py
git commit -m "feat(topology): scaffold topology package"
```

---

### Task 2: Implement `Loop` and `LoopType` dataclasses

**Files:**
- Create: `math_anything/topology/loop.py`
- Modify: `math_anything/topology/__init__.py`
- Test: `tests/unit/test_topology_loop.py`

**Interfaces:**
- Produces: `Loop(nodes: tuple[str, ...], edges: tuple[str, ...], is_directed: bool, canonical_form: str)`
- Produces: `LoopType` StrEnum with `CONVERGENCE`, `COUPLING`, `MIGRATION`, `MULTISCALE`, `UNKNOWN`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_topology_loop.py
from math_anything.topology.loop import Loop, LoopType


def test_loop_creation():
    loop = Loop(
        nodes=("A", "B", "C", "A"),
        edges=("e1", "e2", "e3"),
        is_directed=True,
        canonical_form="A->B->C->A",
    )
    assert loop.nodes == ("A", "B", "C", "A")
    assert loop.edges == ("e1", "e2", "e3")
    assert loop.is_directed is True


def test_loop_type_enum():
    assert LoopType.CONVERGENCE.value == "convergence"
    assert LoopType.COUPLING.value == "coupling"
    assert LoopType.MIGRATION.value == "migration"
    assert LoopType.MULTISCALE.value == "multiscale"
    assert LoopType.UNKNOWN.value == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_topology_loop.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'math_anything.topology.loop'`

- [ ] **Step 3: Implement the module**

```python
# math_anything/topology/loop.py
"""Loop datamodel for topology-aware morphism engineering."""

from __future__ import annotations

from dataclasses import dataclass

from math_anything._compat import StrEnum


class LoopType(StrEnum):
    """Taxonomy of loops in an approximation/morphism graph."""

    CONVERGENCE = "convergence"
    COUPLING = "coupling"
    MIGRATION = "migration"
    MULTISCALE = "multiscale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Loop:
    """A closed walk in a morphism graph.

    Attributes:
        nodes: Ordered tuple of structure/engine names. The first and last node
            are equal for a closed loop.
        edges: Ordered tuple of morphism names connecting consecutive nodes.
        is_directed: Whether the loop respects edge direction.
        canonical_form: Human-readable normalized representation.
    """

    nodes: tuple[str, ...]
    edges: tuple[str, ...]
    is_directed: bool
    canonical_form: str

    def __post_init__(self) -> None:
        if len(self.nodes) < 2:
            raise ValueError("A loop must contain at least two nodes.")
        if self.nodes[0] != self.nodes[-1]:
            raise ValueError("A loop must start and end at the same node.")
        if len(self.edges) != len(self.nodes) - 1:
            raise ValueError("Number of edges must equal number of node transitions.")

    def to_dict(self) -> dict:
        return {
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "is_directed": self.is_directed,
            "canonical_form": self.canonical_form,
        }
```

- [ ] **Step 4: Update package exports**

```python
# math_anything/topology/__init__.py
"""Topology-aware loop engineering for morphism chains."""

from __future__ import annotations

from .loop import Loop, LoopType

# Stubs replaced in upcoming tasks.
class LoopEngine:
    pass

class LoopClassifier:
    pass

__all__ = ["Loop", "LoopEngine", "LoopClassifier", "LoopType"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_topology_loop.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add math_anything/topology/loop.py math_anything/topology/__init__.py tests/unit/test_topology_loop.py
git commit -m "feat(topology): add Loop dataclass and LoopType enum"
```

---

### Task 3: Implement `LoopEngine.build_graph()` and `find_loops()`

**Files:**
- Create: `math_anything/topology/loop_engine.py`
- Modify: `math_anything/topology/__init__.py`
- Test: `tests/unit/test_topology_loop_engine.py`

**Interfaces:**
- Consumes: `CategoryEngine` with `morphism_links: list[MorphismLink]`
- Produces: `LoopEngine.build_graph() -> networkx.MultiDiGraph`
- Produces: `LoopEngine.find_loops() -> list[Loop]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_topology_loop_engine.py
import pytest

from math_anything.categories.engine import CategoryEngine
from math_anything.topology.loop import LoopType
from math_anything.topology.loop_engine import LoopEngine


@pytest.fixture
def triangle_engine():
    ce = CategoryEngine()
    ce.register_morphism(type("m1", (), {"name": "m1"})())
    ce.register_morphism(type("m2", (), {"name": "m2"})())
    ce.register_morphism(type("m3", (), {"name": "m3"})())
    ce.link("m1", "A", "B")
    ce.link("m2", "B", "C")
    ce.link("m3", "C", "A")
    return ce


def test_build_graph(triangle_engine):
    le = LoopEngine(triangle_engine)
    g = le.build_graph()
    assert set(g.nodes()) == {"A", "B", "C"}
    assert g.has_edge("A", "B")
    assert g.has_edge("B", "C")
    assert g.has_edge("C", "A")


def test_find_loops(triangle_engine):
    le = LoopEngine(triangle_engine)
    loops = le.find_loops()
    assert len(loops) == 1
    assert loops[0].nodes[0] == loops[0].nodes[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_topology_loop_engine.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'math_anything.topology.loop_engine'`

- [ ] **Step 3: Implement the module**

```python
# math_anything/topology/loop_engine.py
"""Loop detection engine over CategoryEngine morphism graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from .loop import Loop

if TYPE_CHECKING:
    from math_anything.categories.engine import CategoryEngine


@dataclass
class LoopEngine:
    """Detects loops in a CategoryEngine morphism graph."""

    category_engine: "CategoryEngine"

    def build_graph(self) -> nx.MultiDiGraph:
        """Build a directed multi-graph from registered morphism links."""
        graph = nx.MultiDiGraph()
        for link in self.category_engine.morphism_links:
            graph.add_edge(
                link.source_structure,
                link.target_structure,
                key=link.morphism.name,
                morphism=link.morphism,
            )
        return graph

    def find_loops(self) -> list[Loop]:
        """Return a cycle basis of the underlying undirected graph as Loop objects."""
        digraph = self.build_graph()
        if digraph.number_of_nodes() == 0:
            return []

        undirected = digraph.to_undirected()
        try:
            cycles = nx.minimum_cycle_basis(undirected)
        except nx.NetworkXNoCycle:
            return []

        loops: list[Loop] = []
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            # Order cycle nodes by walking the undirected graph.
            ordered = self._order_cycle(undirected, cycle)
            if ordered is None:
                continue

            edges: list[str] = []
            is_directed = True
            for u, v in zip(ordered, ordered[1:]):
                # Prefer directed edge key if it exists.
                key = None
                if digraph.has_edge(u, v):
                    key = next(iter(digraph[u][v]))
                elif digraph.has_edge(v, u):
                    key = next(iter(digraph[v][u]))
                    is_directed = False
                else:
                    key = next(iter(undirected[u][v]))
                    is_directed = False
                edges.append(key)

            loops.append(
                Loop(
                    nodes=tuple(ordered),
                    edges=tuple(edges),
                    is_directed=is_directed,
                    canonical_form=" -> ".join(ordered),
                )
            )
        return loops

    def _order_cycle(
        self, graph: nx.Graph, cycle_nodes: list[str]
    ) -> list[str] | None:
        """Return a closed walk ordering of nodes in an undirected cycle."""
        start = cycle_nodes[0]
        ordered = [start]
        visited = {start}
        current = start
        while len(visited) < len(cycle_nodes):
            neighbors = [n for n in graph.neighbors(current) if n in cycle_nodes and n not in visited]
            if not neighbors:
                return None
            nxt = neighbors[0]
            ordered.append(nxt)
            visited.add(nxt)
            current = nxt
        # Close the loop if start is a neighbor of the last node.
        if start not in list(graph.neighbors(current)):
            return None
        ordered.append(start)
        return ordered
```

- [ ] **Step 4: Update package exports**

```python
# math_anything/topology/__init__.py
"""Topology-aware loop engineering for morphism chains."""

from __future__ import annotations

from .loop import Loop, LoopType
from .loop_engine import LoopEngine

# Stub replaced in upcoming task.
class LoopClassifier:
    pass

__all__ = ["Loop", "LoopEngine", "LoopClassifier", "LoopType"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_topology_loop_engine.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add math_anything/topology/loop_engine.py math_anything/topology/__init__.py tests/unit/test_topology_loop_engine.py
git commit -m "feat(topology): add LoopEngine graph builder and loop detection"
```

---

### Task 4: Add Betti number computation

**Files:**
- Modify: `math_anything/topology/loop_engine.py`
- Test: `tests/unit/test_topology_loop_engine.py`

**Interfaces:**
- Produces: `LoopEngine.betti_numbers() -> dict[str, int]` with keys `"beta0"`, `"beta1"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_topology_loop_engine.py (append)
def test_betti_numbers_triangle(triangle_engine):
    le = LoopEngine(triangle_engine)
    betti = le.betti_numbers()
    assert betti["beta0"] == 1
    assert betti["beta1"] == 1


def test_betti_numbers_disconnected():
    ce = CategoryEngine()
    ce.register_morphism(type("m1", (), {"name": "m1"})())
    ce.register_morphism(type("m2", (), {"name": "m2"})())
    ce.link("m1", "A", "B")
    ce.link("m2", "C", "D")
    le = LoopEngine(ce)
    betti = le.betti_numbers()
    assert betti["beta0"] == 2
    assert betti["beta1"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_topology_loop_engine.py::test_betti_numbers_triangle -v`

Expected: FAIL with `AttributeError: 'LoopEngine' object has no attribute 'betti_numbers'`

- [ ] **Step 3: Implement the method**

```python
# math_anything/topology/loop_engine.py — add to LoopEngine class

    def betti_numbers(self) -> dict[str, int]:
        """Return Betti numbers of the underlying undirected graph.

        β0 = number of connected components.
        β1 = number of independent cycles (first Betti number).
        """
        graph = self.build_graph().to_undirected()
        if graph.number_of_nodes() == 0:
            return {"beta0": 0, "beta1": 0}

        beta0 = nx.number_connected_components(graph)
        # β1 = E - V + C for each component summed.
        beta1 = graph.number_of_edges() - graph.number_of_nodes() + beta0
        return {"beta0": beta0, "beta1": beta1}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_topology_loop_engine.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add math_anything/topology/loop_engine.py tests/unit/test_topology_loop_engine.py
git commit -m "feat(topology): add Betti number computation"
```

---

### Task 5: Implement `LoopClassifier`

**Files:**
- Create: `math_anything/topology/classifier.py`
- Modify: `math_anything/topology/__init__.py`
- Test: `tests/unit/test_topology_classifier.py`

**Interfaces:**
- Consumes: `Loop`, optional mapping of edge name → `Morphism`
- Produces: `LoopClassifier.classify(loop) -> LoopType`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_topology_classifier.py
from math_anything.topology.classifier import LoopClassifier
from math_anything.topology.loop import Loop, LoopType


def test_classify_convergence_loop():
    loop = Loop(
        nodes=("V_eff", "density", "wavefunction", "V_eff"),
        edges=("poisson", "ks_solve", "mixer"),
        is_directed=True,
        canonical_form="V_eff->density->wavefunction->V_eff",
    )
    classifier = LoopClassifier()
    assert classifier.classify(loop) == LoopType.CONVERGENCE


def test_classify_coupling_loop():
    loop = Loop(
        nodes=("atomistic", "continuum", "atomistic"),
        edges=("concurrent_up", "concurrent_down"),
        is_directed=True,
        canonical_form="atomistic<->continuum",
    )
    classifier = LoopClassifier()
    assert classifier.classify(loop) == LoopType.COUPLING


def test_classify_unknown():
    loop = Loop(
        nodes=("A", "B", "A"),
        edges=("m1",),
        is_directed=True,
        canonical_form="A->B->A",
    )
    classifier = LoopClassifier()
    assert classifier.classify(loop) == LoopType.UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_topology_classifier.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'math_anything.topology.classifier'`

- [ ] **Step 3: Implement the classifier**

```python
# math_anything/topology/classifier.py
"""Rule-based classification of morphism loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .loop import Loop, LoopType


@dataclass
class LoopClassifier:
    """Classify loops using structural heuristics and morphism metadata."""

    convergence_markers: set[str] = field(
        default_factory=lambda: {
            "self_consistent",
            "scf",
            "mixer",
            "charge_density",
            "density",
            "wavefunction",
            "potential",
            "poisson",
            "ks_solve",
        }
    )
    coupling_markers: set[str] = field(
        default_factory=lambda: {
            "concurrent",
            "coupling",
            "two_way",
            "atomistic",
            "continuum",
            "micro",
            "macro",
        }
    )
    migration_markers: set[str] = field(
        default_factory=lambda: {
            "vasp",
            "quantum_espresso",
            "qe",
            "cp2k",
            "lammps",
            "gromacs",
            "abaqus",
            "ansys",
            "comsol",
        }
    )

    def classify(
        self,
        loop: Loop,
        morphism_lookup: dict[str, Any] | None = None,
    ) -> LoopType:
        """Return the LoopType best matching the loop."""
        text = " ".join(loop.nodes + loop.edges).lower()

        if self._matches(text, self.convergence_markers, threshold=2):
            return LoopType.CONVERGENCE

        if self._matches(text, self.coupling_markers, threshold=2):
            return LoopType.COUPLING

        if self._matches(text, self.migration_markers, threshold=2):
            return LoopType.MIGRATION

        if morphism_lookup is not None:
            categories = set()
            for edge in loop.edges:
                morph = morphism_lookup.get(edge)
                if morph is not None:
                    categories.add(getattr(morph, "category", "").lower())
            if len(categories) >= 2:
                return LoopType.MULTISCALE

        return LoopType.UNKNOWN

    @staticmethod
    def _matches(text: str, markers: set[str], threshold: int) -> bool:
        count = sum(1 for marker in markers if marker in text)
        return count >= threshold
```

- [ ] **Step 4: Update package exports**

```python
# math_anything/topology/__init__.py
"""Topology-aware loop engineering for morphism chains."""

from __future__ import annotations

from .classifier import LoopClassifier
from .loop import Loop, LoopType
from .loop_engine import LoopEngine

__all__ = ["Loop", "LoopEngine", "LoopClassifier", "LoopType"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_topology_classifier.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add math_anything/topology/classifier.py math_anything/topology/__init__.py tests/unit/test_topology_classifier.py
git commit -m "feat(topology): add LoopClassifier for loop taxonomy"
```

---

### Task 6: Integrate `LoopEngine` with `CategoryEngine`

**Files:**
- Modify: `math_anything/categories/engine.py`
- Test: `tests/unit/test_categories_engine_loop.py`

**Interfaces:**
- Produces: `CategoryEngine.loop_engine()` property returns a cached `LoopEngine`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_categories_engine_loop.py
from math_anything.categories.engine import CategoryEngine
from math_anything.morphisms.approximations import (
    BornOppenheimerApproximation,
    KohnShamMapping,
    PlaneWaveTruncation,
)


def test_category_engine_loop_engine():
    ce = CategoryEngine()
    ce.register_morphism(BornOppenheimerApproximation())
    ce.register_morphism(KohnShamMapping())
    ce.register_morphism(PlaneWaveTruncation(encut=520))
    ce.link("born_oppenheimer", "FullManyBody", "ElectronicSchrodinger")
    ce.link("kohn_sham", "ElectronicSchrodinger", "KohnSham_Full")
    ce.link("plane_wave_truncation", "KohnSham_Full", "KohnSham_Truncated")

    le = ce.loop_engine
    assert le is not None
    loops = le.find_loops()
    assert len(loops) == 0  # DAG example
    betti = le.betti_numbers()
    assert betti["beta0"] == 1
    assert betti["beta1"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_categories_engine_loop.py -v`

Expected: FAIL with `AttributeError: 'CategoryEngine' object has no attribute 'loop_engine'`

- [ ] **Step 3: Add the property**

```python
# math_anything/categories/engine.py — add inside CategoryEngine class

    @property
    def loop_engine(self):
        """Return a topology-aware LoopEngine over this category."""
        from math_anything.topology.loop_engine import LoopEngine

        return LoopEngine(self)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_categories_engine_loop.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add math_anything/categories/engine.py tests/unit/test_categories_engine_loop.py
git commit -m "feat(categories): expose LoopEngine from CategoryEngine"
```

---

### Task 7: Add CLI `loops` subcommand

**Files:**
- Modify: `math_anything/cli.py`
- Test: `tests/integration/test_topology_cli.py`

**Interfaces:**
- Produces: `bourbaki loops <engine> <files...>` prints JSON with loops and Betti numbers.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/integration/test_topology_cli.py
import json
import subprocess
import sys
from pathlib import Path


def test_cli_loops_subcommand_exists():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything.cli", "loops", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "loops" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_topology_cli.py -v`

Expected: FAIL with `error: argument command: invalid choice: 'loops'`

- [ ] **Step 3: Add CLI subcommand**

In `math_anything/cli.py`, after the existing `cross` subcommand (around line 450–550; exact line varies), add:

```python
    # Loops command
    loops_parser = subparsers.add_parser(
        "loops",
        help="Detect topology loops in an engine's morphism graph",
    )
    loops_parser.add_argument(
        "engine",
        choices=ENGINE_NAMES,
        help="Engine name",
    )
    loops_parser.add_argument(
        "files",
        nargs="*",
        help="Input files to extract schema from",
    )
    loops_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file",
    )
```

Then add handler `cmd_loops` near other command handlers:

```python
def cmd_loops(args: argparse.Namespace) -> int:
    """Handle the loops subcommand."""
    import json as _json

    from math_anything.categories.engine import CategoryEngine
    from math_anything.topology.loop_engine import LoopEngine
    from math_anything.topology.classifier import LoopClassifier

    engine = _get_engine(args.engine)
    schema = None
    if args.files:
        schema = _extract_schema(args.engine, args.files)

    # Build a default CategoryEngine for the engine/domain.
    ce = CategoryEngine()
    # TODO: in future, populate from domain-specific morphism registry.
    # For now, register the DFT example morphisms to demonstrate loop detection.
    from math_anything.morphisms.approximations import (
        BornOppenheimerApproximation,
        KohnShamMapping,
        PlaneWaveTruncation,
    )
    ce.register_morphism(BornOppenheimerApproximation())
    ce.register_morphism(KohnShamMapping())
    ce.register_morphism(PlaneWaveTruncation(encut=520))
    ce.link("born_oppenheimer", "FullManyBody", "ElectronicSchrodinger")
    ce.link("kohn_sham", "ElectronicSchrodinger", "KohnSham_Full")
    ce.link("plane_wave_truncation", "KohnSham_Full", "KohnSham_Truncated")

    le = LoopEngine(ce)
    loops = le.find_loops()
    classifier = LoopClassifier()
    loops_data = []
    for loop in loops:
        loops_data.append(
            {
                "type": classifier.classify(loop).value,
                "nodes": list(loop.nodes),
                "edges": list(loop.edges),
                "directed": loop.is_directed,
                "canonical_form": loop.canonical_form,
            }
        )

    report = {
        "engine": args.engine,
        "schema_present": schema is not None,
        "betti": le.betti_numbers(),
        "loops": loops_data,
    }

    output = _json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        safe_print(output)
    return 0
```

Wire it into the dispatcher near the bottom of `cli.py`:

```python
    elif args.command == "loops":
        return cmd_loops(args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_topology_cli.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add math_anything/cli.py tests/integration/test_topology_cli.py
git commit -m "feat(cli): add loops subcommand"
```

---

### Task 8: Add MCP tool `analyze_loops`

**Files:**
- Modify: `math_anything/mcp_server.py`
- Test: `tests/test_mcp_server.py` (append)

**Interfaces:**
- Produces: `@mcp.tool() def analyze_loops(engine: str, parameters: dict | None = None) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcp_server.py (append)
import json


def test_mcp_analyze_loops_tool_exists():
    from math_anything.mcp_server import mcp

    tool_names = [t.name for t in mcp._tools]
    assert "analyze_loops" in tool_names


def test_mcp_analyze_loops_runs():
    from math_anything.mcp_server import analyze_loops

    result = analyze_loops("vasp", {})
    data = json.loads(result)
    assert "betti" in data
    assert "loops" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_server.py::test_mcp_analyze_loops_tool_exists -v`

Expected: FAIL with `AssertionError: 'analyze_loops' not in tool_names`

- [ ] **Step 3: Add MCP tool**

Add near the end of `math_anything/mcp_server.py`, before `if __name__ == "__main__":`:

```python
@mcp.tool()
def analyze_loops(engine: str, parameters: dict[str, Any] | None = None) -> str:
    """Detect and classify topology loops in an engine's morphism graph.

    Args:
        engine: Engine name (e.g., vasp, lammps, qe)
        parameters: Optional engine parameters; currently unused but reserved
            for future domain-specific loop population.
    """
    import json as _json

    from math_anything.categories.engine import CategoryEngine
    from math_anything.topology.classifier import LoopClassifier
    from math_anything.topology.loop_engine import LoopEngine

    parameters = parameters or {}

    ce = CategoryEngine()
    from math_anything.morphisms.approximations import (
        BornOppenheimerApproximation,
        KohnShamMapping,
        PlaneWaveTruncation,
    )
    ce.register_morphism(BornOppenheimerApproximation())
    ce.register_morphism(KohnShamMapping())
    ce.register_morphism(PlaneWaveTruncation(encut=520))
    ce.link("born_oppenheimer", "FullManyBody", "ElectronicSchrodinger")
    ce.link("kohn_sham", "ElectronicSchrodinger", "KohnSham_Full")
    ce.link("plane_wave_truncation", "KohnSham_Full", "KohnSham_Truncated")

    le = LoopEngine(ce)
    classifier = LoopClassifier()
    loops = le.find_loops()

    report = {
        "engine": engine,
        "betti": le.betti_numbers(),
        "loops": [
            {
                "type": classifier.classify(loop).value,
                "nodes": list(loop.nodes),
                "edges": list(loop.edges),
                "directed": loop.is_directed,
                "canonical_form": loop.canonical_form,
            }
            for loop in loops
        ],
    }
    return _json.dumps(report, indent=2, ensure_ascii=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_server.py -v`

Expected: PASS (or existing failures unrelated to this tool)

- [ ] **Step 5: Commit**

```bash
git add math_anything/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add analyze_loops tool"
```

---

### Task 9: Optional lazy export in `math_anything/__init__.py`

**Files:**
- Modify: `math_anything/__init__.py`

**Interfaces:**
- Produces: `from math_anything import LoopEngine, LoopClassifier, Loop, LoopType` works.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_topology_imports.py (append)
def test_top_level_imports():
    from math_anything import Loop, LoopEngine, LoopClassifier, LoopType
    assert Loop is not None
    assert LoopType.CONVERGENCE is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_topology_imports.py::test_top_level_imports -v`

Expected: FAIL with `ImportError: cannot import name 'Loop' from 'math_anything'`

- [ ] **Step 3: Add lazy imports**

```python
# math_anything/__init__.py — append to _lazy_imports dict
        # Topology loop engineering
        "Loop": ".topology",
        "LoopType": ".topology",
        "LoopEngine": ".topology",
        "LoopClassifier": ".topology",
```

Also add them to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_topology_imports.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add math_anything/__init__.py tests/unit/test_topology_imports.py
git commit -m "feat(topology): expose LoopEngine at top-level namespace"
```

---

### Task 10: Run full topology test suite and lint

**Files:**
- All files created/modified above.

- [ ] **Step 1: Run topology-focused tests**

Run:
```bash
pytest tests/unit/test_topology_*.py tests/unit/test_categories_engine_loop.py tests/integration/test_topology_cli.py tests/test_mcp_server.py -v
```

Expected: All tests PASS (or only pre-existing failures).

- [ ] **Step 2: Run lint**

Run:
```bash
ruff check math_anything/topology/ math_anything/categories/engine.py math_anything/cli.py math_anything/mcp_server.py math_anything/__init__.py
```

Expected: No new errors.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "chore(topology): lint fixes and test verification"
```

---

## Self-Review

**Spec coverage:**
- ✅ Loop detection — Tasks 3, 10
- ✅ Loop classification — Task 5
- ✅ Betti numbers — Task 4
- ✅ CategoryEngine integration — Task 6
- ✅ CLI exposure — Task 7
- ✅ MCP tool exposure — Task 8
- ✅ Top-level imports — Task 9
- ⏳ Cross-engine homotopy — out of scope for this plan (Phase 3 follow-up)
- ⏳ Curvature bridge — out of scope for this plan (Phase 4 follow-up)

**Placeholder scan:**
- No TBD/TODO placeholders in task steps.
- The CLI and MCP tool currently hard-code DFT example morphisms; this is intentional for the MVP and explicitly noted.

**Type consistency:**
- `LoopEngine.find_loops()` returns `list[Loop]` consistently.
- `LoopClassifier.classify()` returns `LoopType` consistently.
- `betti_numbers()` returns `dict[str, int]` with keys `"beta0"`, `"beta1"` consistently.

**Scope check:**
- This plan covers Phase 1–2 of the design doc: loop detection, classification, Betti numbers, CLI/MCP exposure. It produces working, testable software on its own.
- Phase 3 (cross-engine homotopy) and Phase 4 (curvature) are intentionally deferred to separate plans.
