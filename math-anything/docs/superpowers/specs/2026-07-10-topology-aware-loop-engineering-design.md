# Design: Topology-Aware Loop Engineering for Cross-Engine Morphism Chains

**Date:** 2026-07-10  
**Project:** Bourbaki / math_anything (`D:/math-anything/math-anything`)  
**Status:** Draft pending review  

---

## 1. Overview

Bourbaki already models computational physics engines as morphisms over mathematical structures, with each morphism recording `invariants_kept`, `invariants_lost`, and `invariants_introduced`. However, the current `CategoryEngine` treats morphism collections as a DAG: it can find chains and cumulative loss, but it cannot reason about loops, feedback cycles, or the topological shape of the approximation space.

This design introduces **topology-aware loop engineering**: we augment Bourbaki's morphism graph with topological/geometric primitives (manifolds, homotopy classes, Betti numbers) so that the system can:

1. Detect and classify feedback loops in approximation chains (e.g., SCF cycles, multi-scale coupling loops).
2. Measure whether two cross-engine migration paths are topologically equivalent.
3. Surface topological obstructions that prevent a consistent cross-engine parameter mapping.
4. Provide LLM/MCP tools with rigorous, computable statements about approximation structure.

This is not a full symbolic-regression feature on its own; it is the **structural backbone** that makes physics-constrained symbolic regression and cross-engine verification more rigorous.

---

## 2. Problem Statement

### 2.1 Current limitations

- `CategoryEngine.get_morphism_chain()` uses BFS and assumes a DAG; it fails or silently ignores cycles.
- `CompositeMorphism` only composes two morphisms linearly; there is no model for closed loops.
- Cross-engine mapping in `core/cross_engine.py` is descriptive (`CouplingInterface`, `CouplingType`) but not computable: it cannot answer "Is this coupling loop consistent?"
- `structures/geometry_riemannian.py` computes Christoffel/Riemann/Ricci tensors numerically, but it is not connected to the morphism engine.

### 2.2 Research question

> Can we equip Bourbaki's morphism engine with lightweight algebraic topology so that approximation chains, cross-engine mappings, and multi-scale coupling loops become objects with computable invariants?

---

## 3. Goals and Non-Goals

### Goals

- Add a `LoopEngine` module that operates on the existing `CategoryEngine` graph.
- Represent morphism graphs as 1-dimensional simplicial complexes and compute:
  - Connected components
  - Fundamental cycles / loop basis
  - Betti numbers β₀ and β₁
- Connect loop detection to physical meaning:
  - SCF loop → self-consistency cycle
  - Micro↔macro coupling loop → two-way multiscale feedback
  - Cross-engine path pair → homotopy equivalence check
- Expose results through the existing MCP server and CLI (`analyze_morphism_chain`, `cross`, `validate`).
- Reuse existing `geometry_riemannian.py` for a optional "curvature of approximation space" metric.

### Non-Goals

- We do **not** aim for a full homotopy type theory prover (that work lives in `type_theory/hott.py`).
- We do **not** rewrite existing engine extractors.
- We do **not** implement persistent homology on simulation trajectories (out of scope; may be a follow-up).
- We do **not** replace the current DAG-style chain queries; we extend them.

---

## 4. Core Concepts

| Concept | Existing representation | New representation |
|---|---|---|
| Engine | `EnginePlugin` | Node in a structure graph |
| Approximation | `Morphism` / `CompositeMorphism` | Directed edge / path |
| Chain | `CategoryEngine.get_morphism_chain()` | Path in graph |
| Loop | Not represented | Cycle in graph; element of cycle basis |
| Cross-engine path equivalence | String comparison | Homotopy / loop contraction test |
| Multi-scale coupling | `CouplingInterface` | Edge in coupled simplicial complex |

### 4.1 Loop taxonomy

| Loop type | Example | Topological interpretation |
|---|---|---|
| **Convergence loop** | SCF: `V_eff → n → ψ → V_eff` | Contractible loop that should collapse to a fixed point |
| **Coupling loop** | MD↔FEM concurrent coupling | Non-contractible loop representing two-way information flow |
| **Migration loop** | VASP→QE→VASP round-trip | Test of cross-engine semantic equivalence; contractible if identity |
| **Approximation loop** | DFT→MD→FEM→DFT (multiscale chain) | May enclose a "hole" in approximation space, indicating missing physics |

---

## 5. Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  Consumers: CLI, MCP tools, Python API                          │
├─────────────────────────────────────────────────────────────────┤
│  LoopEngine                                                     │
│  ├── build_graph(morphism_links) → nx.MultiDiGraph              │
│  ├── find_loops() → list[Loop]                                  │
│  ├── classify_loop(loop) → LoopType                             │
│  ├── are_paths_homotopic(p1, p2) → bool                         │
│  ├── betti_numbers() → {β0, β1}                                 │
│  └── curvature_along(path, metric_fn) → scalar                  │
├─────────────────────────────────────────────────────────────────┤
│  CategoryEngine (existing)                                      │
│  └── morphisms, morphism_links, structures                      │
├─────────────────────────────────────────────────────────────────┤
│  Geometry bridge (new)                                          │
│  └── Graph → SimplicialComplex → topology computations          │
├─────────────────────────────────────────────────────────────────┤
│  Existing primitives                                            │
│  ├── Morphism / CompositeMorphism                               │
│  ├── ConservationMatrixField                                    │
│  └── geometry_riemannian.MetricFunction                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Key Components

### 6.1 `math_anything/topology/__init__.py`

New package. Files:

- `loop_engine.py` — main API
- `loop.py` — `Loop` dataclass
- `simplicial.py` — `ApproximationComplex` wrapper around `networkx`/`gudhi` (optional)
- `curvature.py` — bridge to `geometry_riemannian.MetricFunction`
- `classifier.py` — `LoopClassifier`

### 6.2 `LoopEngine` class

```python
@dataclass
class LoopEngine:
    category_engine: CategoryEngine

    def build_graph(self) -> nx.MultiDiGraph:
        ...

    def find_loops(self) -> list[Loop]:
        """Return a cycle basis for the underlying undirected graph."""
        ...

    def classify_loop(self, loop: Loop) -> LoopType:
        """Classify based on node/edge labels and known patterns."""
        ...

    def are_paths_homotopic(
        self,
        path_a: list[str],
        path_b: list[str],
    ) -> bool:
        """True if path_a and path_b form a contractible loop."""
        ...

    def betti_numbers(self) -> dict[str, int]:
        """β0 (components) and β1 (independent loops)."""
        ...

    def curvature_penalty(self, path: list[str], metric_fn) -> float:
        """Numerical curvature integral along a path using existing Riemann code."""
        ...
```

### 6.3 `Loop` dataclass

```python
@dataclass(frozen=True)
class Loop:
    nodes: tuple[str, ...]
    edges: tuple[str, ...]  # morphism names
    is_directed: bool
    canonical_form: str
```

### 6.4 `LoopClassifier`

Rules-based classifier using labels from `MorphismCategory` and domain tags:

| Pattern | Classification |
|---|---|
| Same structure appears twice with `DISCRETIZATION` edges in between | `CONVERGENCE_LOOP` |
| Two different scales connected by `CONCURRENT` coupling edges | `COUPLING_LOOP` |
| Path A and path B share start/end engine | `MIGRATION_LOOP` |
| Mix of `APPROXIMATION`, `DISCRETIZATION`, `RESTRICTION` across multiple domains | `MULTISCALE_LOOP` |

### 6.5 Curvature bridge

Reuse `geometry_riemannian.MetricFunction`:

- Build a discrete metric on the graph where edge weight = information loss (e.g., number of lost invariants).
- Embed the graph in a coordinate space (MDS or spring layout).
- Use `MetricFunction.riemann_at()` to compute scalar curvature at sample points.
- Report "high curvature" regions as places where approximation choices are non-trivial.

---

## 7. Data Flow

### 7.1 Loop detection flow

```text
User calls: math-anything loops --engine vasp --input INCAR

1. Extract schema via existing VASP extractor.
2. Build CategoryEngine with domain-specific morphisms.
3. LoopEngine.build_graph() creates MultiDiGraph.
4. LoopEngine.find_loops() returns cycle basis.
5. LoopClassifier tags each loop.
6. Output JSON:
   {
     "loops": [
       {
         "type": "convergence_loop",
         "nodes": ["V_eff", "charge_density", "wavefunction", "V_eff"],
         "edges": ["poisson", "ks_solve", "mixer_update"],
         "contractible": true,
         "physical_meaning": "SCF self-consistency cycle"
       }
     ],
     "betti": {"β0": 1, "β1": 2}
   }
```

### 7.2 Cross-engine homotopy flow

```text
User calls: math-anything cross vasp_schema.json qe --check-homotopy

1. Extract both schemas.
2. Find all paths from VASP canonical structure to QE canonical structure.
3. LoopEngine.are_paths_homotopic(path_a, path_b) checks if their concatenation is contractible.
4. If not contractible, report topological obstruction.
```

---

## 8. Integration with Existing Code

| Existing file | Integration point |
|---|---|
| `math_anything/categories/engine.py` | Add `register_loop_engine()` and `get_loops()` helpers |
| `math_anything/morphisms/__init__.py` | Ensure all morphisms have stable `name` and `category` |
| `math_anything/core/cross_engine.py` | Use `LoopEngine` to validate `CouplingInterface` consistency |
| `math_anything/mcp_server.py` | Add tool `analyze_loops(engine, parameters)` |
| `math_anything/cli.py` | Add subcommand `loops` and `--check-homotopy` flag to `cross` |
| `math_anything/structures/geometry_riemannian.py` | Import via `topology/curvature.py` |

---

## 9. Testing Strategy

1. **Unit tests** in `tests/unit/test_topology_loop_engine.py`:
   - Triangle graph has β₁ = 1.
   - Square with one diagonal has β₁ = 2.
   - Two parallel paths between engines are homotopic if they form a contractible quadrilateral.

2. **Domain tests** in `tests/unit/test_topology_domains.py`:
   - VASP SCF loop is classified as `CONVERGENCE_LOOP`.
   - LAMMPS↔Abaqus coupling is classified as `COUPLING_LOOP`.

3. **Integration tests** in `tests/integration/test_topology_cli.py`:
   - `math-anything loops vasp INCAR` returns valid JSON.
   - `math-anything cross ... --check-homotopy` detects non-contractible migration paths.

4. **Rust bridge**: no new Rust code initially; curvature computation can remain Python-only for the MVP.

---

## 10. Phases / Milestones

### Phase 1: Loop detection (MVP)

- Create `math_anything/topology/` package.
- Implement `LoopEngine.build_graph()` and `find_loops()` using `networkx`.
- Add `Loop` dataclass and basic classifier.
- Add MCP tool `analyze_loops` and CLI subcommand `loops`.
- Tests for cycle basis and Betti numbers.

### Phase 2: Classification & semantics

- Implement `LoopClassifier` with domain-specific rules.
- Connect to `ConservationMatrixField` to check if a loop preserves conservation laws.
- Add `betti_numbers()` and reports.

### Phase 3: Cross-engine homotopy

- Implement `are_paths_homotopic()`.
- Integrate into `core/cross_engine.py` and CLI `cross --check-homotopy`.
- Build a small benchmark: VASP↔QE↔CP2K canonical parameter round-trips.

### Phase 4: Curvature & visualization

- Build `topology/curvature.py` bridge to `geometry_riemannian.py`.
- Add `curvature_penalty()` and visualization output (Mermaid/Graphviz).
- Optional: expose via MCP.

---

## 11. Risks and Open Questions

1. **Directed vs. undirected cycles**: Many physical loops are directed (SCF). We will represent directed loops but compute the cycle basis on the undirected skeleton, then annotate direction.
2. **Graph size**: With 19 engines and many morphisms, the graph may become dense. We will scope the graph per-domain or per-query.
3. **Topological correctness**: Betti numbers on a manually curated graph are only as good as the curation. We will document this limitation and provide an `explain()` method.
4. **Dependency**: `networkx` is already a dependency; no new external packages are required for Phase 1–3. Phase 4 may optionally use `gudhi` if we move to higher-dimensional simplicial complexes.

---

## 12. Success Criteria

- `LoopEngine` can detect and classify at least three real loops: VASP SCF, MD↔FEM coupling, VASP↔QE migration.
- MCP server exposes `analyze_loops` and returns structured JSON.
- CLI `math-anything loops` works on a VASP INCAR/POSCAR example.
- Test coverage for `math_anything/topology/` ≥ 80%.

---

## 13. Related Work in Codebase

- `math_anything/categories/engine.py`
- `math_anything/morphisms/__init__.py`
- `math_anything/structures/geometry_riemannian.py`
- `math_anything/core/cross_engine.py`
- `math_anything/mcp_server.py`
- `math_anything/cli.py`
