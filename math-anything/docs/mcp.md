# Bourbaki MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes Bourbaki's mathematical structure analysis and verification capabilities to LLMs through standardized tool interfaces.

Built on [FastMCP](https://gofastmcp.com) SDK v3.2+.

## What is Bourbaki MCP?

Bourbaki MCP analyzes, compares, validates, and discovers mathematical structures across **8 physics/ML domains** (DFT, CFD, MD, FEM, EM, QC, phase field, supervised learning) and a wide range of computational science engines. It provides:

- **Domain analysis** — conservation fields, morphism chains, and constraint propagation for each physics/ML discipline
- **Cross-domain comparison** — what invariants are shared vs. domain-specific
- **Structure verification** — multi-layer verification pipeline via `verify_structure`
- **Cross-engine comparison** — semantic diff of mathematical schemas
- **Constraint analysis** — invariant detection and conservation law verification
- **Dimensional analysis** — Buckingham π theorem and consistency checking
- **Equation discovery** — symbolic regression from data
- **Riemann geometry** — metric tensors, curvature, Christoffel symbols
- **Morphism chain analysis** — trace how approximations preserve/lose structural invariants

## Installation

### pip

```bash
pip install mcp fastmcp
cd /path/to/math-anything
```

### uv

```bash
uv pip install mcp fastmcp
cd /path/to/math-anything
```

### From source

```bash
git clone https://github.com/toki0413/math-anything.git
cd math-anything
pip install -e ".[mcp]"
```

## Configuration

### Claude Desktop

Add to `~/AppData/Roaming/Claude/claude_desktop_config.json` (Windows) or `~/.config/Claude/claude_desktop_config.json` (macOS/Linux):

```json
{
  "mcpServers": {
    "bourbaki": {
      "command": "python",
      "args": ["-m", "server.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/math-anything"
      }
    }
  }
}
```

For **uv** users:

```json
{
  "mcpServers": {
    "bourbaki": {
      "command": "uv",
      "args": ["run", "--with", "mcp", "python", "-m", "server.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/math-anything"
      }
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "bourbaki": {
      "command": "python",
      "args": ["-m", "server.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/math-anything"
      }
    }
  }
}
```

### Windsurf

Add to `~/.windsurf/mcp.json`:

```json
{
  "mcpServers": {
    "bourbaki": {
      "command": "python",
      "args": ["-m", "server.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/math-anything"
      }
    }
  }
}
```

## Tools

### Domain Instantiation Layer (Layer 3)

#### 1. `analyze_domain`

Analyze a physics/ML domain: conservation field + morphism chain + constraint propagation.

**Parameters:**
- `domain` (string, required) — Domain name: `dft`, `cfd`, `md`, `fem`, `em`, `qc`, `phase_field`, `supervised_learning`
- `parameters` (object, optional) — Domain-specific parameters

**Example:**
```json
{
  "domain": "dft",
  "parameters": {"xc_functional": "PBE", "ecutwfc": 50.0}
}
```

**Returns:** Domain analysis with preserved/weakened/lost/emerged invariants, eigenvalues, CFL conditions, and morphism chain details.

#### 2. `compare_domains`

Compare two physics/ML domains — same conservation field, different morphism chains.

**Parameters:**
- `domain_a` (string, required) — First domain name
- `params_a` (object, optional) — First domain parameters
- `domain_b` (string, optional) — Second domain name. Defaults to `""`, which falls back to `md`
- `params_b` (object, optional) — Second domain parameters

**Example:**
```json
{
  "domain_a": "dft",
  "params_a": {"xc_functional": "PBE"},
  "domain_b": "md",
  "params_b": {"force_field": "Lennard-Jones"}
}
```

**Returns:** Comparison with common preserved invariants, domain-specific invariants, and what each domain loses.

#### 3. `list_domains`

List all available physics/ML domains with their morphism chain descriptions.

**Parameters:** None

**Returns:** Domain list with names, descriptions, equation types, and morphism chain lengths.

### Conservation & Structures

#### 4. `build_conservation_field`

Build conservation matrix field for a given equation type.

**Parameters:**
- `equation_type` (string, required) — One of: `navier_stokes`, `euler`, `schrodinger`, `maxwell`, `elasticity`, `mhd`, `heat`, `dirac`, `einstein_field`, `klein_gordon`, `wave`, `kohn_sham`, `boltzmann`, `shallow_water`, `schrodinger_nonlinear`, `vlasov`, `hartree_fock`, `advection_diffusion`
- `parameters` (object, optional) — Equation-specific parameters (e.g., `{"mu": 0.01, "hbar": 1.0}`)

**Returns:** Conservation laws with Noether correspondence and symmetry mappings.

#### 5. `analyze_morphism_chain`

Analyze a chain of mathematical morphisms.

**Parameters:**
- `domain` (string, required) — Domain name (dft, cfd, md, fem, em, qc, phase_field, supervised_learning)
- `chain` (array of strings, optional) — Morphism chain to trace
- `parameters` (object, optional) — Domain parameters

**Returns:** Step-by-step analysis with invariants kept, lost, and introduced at each stage, plus cumulative summary.

#### 6. `solve_numerical`

Unified numerical solver for mathematical structures.

**Parameters:**
- `solver_type` (string, required) — One of: `symplectic`, `eigenvalue`, `scf`, `conservation`, `variational`, `continuum`
- `parameters` (object, required) — Solver-specific parameters

**Returns:** Numerical solution for the selected solver type.

### Foundation Layer

#### 7. `dimensional_analyze`

Buckingham π theorem dimensional analysis and symbolic dimensional checking.

**Parameters:**
- `schema` (object, optional) — Mathematical schema for context
- `quantities` (array, optional) — List of quantity dicts with `name`, `symbol`, `dimension`
- `expression_lhs` (string, optional) — Left-hand side of equation for dimensional checking
- `expression_rhs` (string, optional) — Right-hand side of equation for dimensional checking

**Returns:** Dimensional consistency check, dimensionless π groups, and/or expression dimensional comparison.

#### 8. `verify_structure`

Verify mathematical structure through multi-layer verification pipeline.

**Parameters:**
- `schema` (object, required) — Mathematical schema to verify
- `layers` (array of strings, optional) — Subset of layers: `symbolic`, `type_system`, `logic`, `llm_semantic`, `lean4_formal`

**Returns:** Verification report checking symbolic validation, type system, logic consistency, and optional LLM/Lean4 layers.

#### 9. `discover_equations`

Discover governing equations from data using symbolic regression.

**Parameters:**
- `variable_names` (string, required) — Comma-separated variable names
- `method` (string, optional) — Discovery method: `sindyc` (default), `genetic`
- `max_complexity` (integer, optional) — Maximum equation complexity (default: 10)

**Returns:** Symbolic regression configuration with candidate function library.

### Engine Adapter

#### 10. `translate_engine_params`

Translate engine-specific parameters to domain parameters.

**Parameters:**
- `engine` (string, required) — Simulation engine name
- `parameters` (object, required) — Engine-specific parameters

**Returns:** Domain-agnostic parameter mapping.

### Topology Layer

#### 11. `analyze_loops`

Detect and classify topology loops in a demonstration morphism graph.

**Parameters:**
- `engine` (string, required) — Engine name (e.g., `vasp`, `lammps`, `qe`)
- `parameters` (object, optional) — Engine parameters; reserved for future domain-specific loop population

**Returns:** Loop classification, Betti numbers, curvature map, and Mermaid visualization.

#### 12. `compute_riemann_geometry`

Compute Riemannian geometric structures from a metric specification.

**Parameters:**
- `metric` (array of arrays, required) — Metric tensor g_{ij} (dim x dim)
- `christoffel` (array of arrays of arrays, required) — Christoffel symbols Gamma^k_{ij} (dim x dim x dim)
- `dim` (integer, required) — Manifold dimension

**Returns:** Riemann tensor, Ricci tensor, scalar curvature.

### ML Surrogate Layer

#### 13. `analyze_ml_model`

Analyze a supervised-learning model as a morphism chain.

**Parameters:**
- `input_dim` (integer, optional) — Input dimension (default: 2)
- `output_dim` (integer, optional) — Output dimension (default: 1)
- `architecture` (string, optional) — Network architecture (default: `mlp`)
- `loss` (string, optional) — Loss function (default: `mse`)
- `compare_paths` (boolean, optional) — Compute optimization-landscape homotopy between two training paths (default: false)
- `transfer` (boolean, optional) — Verify transfer learning as a natural transformation (default: false)
- `backend` (string, optional) — Surrogate backend to demo: `numpy`, `deepmd`, `mace`, `chgnet` (default: `numpy`). If an optional backend is not installed, the demo gracefully falls back to `numpy`.

**Example:**
```json
{
  "input_dim": 2,
  "output_dim": 1,
  "architecture": "mlp",
  "loss": "mse",
  "compare_paths": true,
  "transfer": true,
  "backend": "numpy"
}
```

**Returns:** Domain analysis of supervised learning as a morphism chain, demo forward pass, optimization curvature, cross-domain homotopy witness, surrogate backend demo, and optional training-path homotopy and transfer-learning natural-transformation checks.

## Resources

### `bourbaki://engines`

List all supported simulation engines with metadata and category information.

### `bourbaki://conservation-laws/{equation_type}`

Conservation laws for a given equation type.

### `bourbaki://domains/{domain_name}`

Domain configuration details: description, equation type, default parameters, and full morphism chain.

**Example:** `bourbaki://domains/dft` returns the DFT domain's morphism chain (Born-Oppenheimer → Kohn-Sham → XC approximation → Pseudopotential → Basis set → k-point sampling).

### `bourbaki://version`

Server version information including SDK version, engine count, and tool count.

## Prompts

### `analyze_simulation`

Analyze the mathematical structure of a simulation setup.

**Arguments:**
- `engine` — Simulation engine name
- `description` — Natural language description of the simulation

### `compare_simulations`

Compare mathematical structures between two simulation engines.

**Arguments:**
- `engine_a` — First simulation engine
- `engine_b` — Second simulation engine

### `compare_approaches`

Compare two physics/ML domains at the structural level — what invariants are shared vs. domain-specific.

**Arguments:**
- `domain_a` — First domain name (default: `dft`)
- `domain_b` — Second domain name (default: `md`)

### `discover_from_data`

Discover governing equations from simulation data.

**Arguments:**
- `variable_names` — Comma-separated variable names from the dataset

## SSE Transport Setup

For remote or networked deployments, Bourbaki MCP supports SSE transport:

```bash
# Start SSE server on default port 8080
python -m server.mcp_server --transport sse

# Custom port
python -m server.mcp_server --transport sse --port 3000
```

Configure clients to connect to `http://localhost:8080/sse` (or your custom port).

For Claude Desktop with SSE:

```json
{
  "mcpServers": {
    "bourbaki": {
      "url": "http://localhost:8080/sse"
    }
  }
}
```

## Troubleshooting

### Server not starting

1. **Import errors:** Ensure `PYTHONPATH` includes the `math-anything` root directory
2. **FastMCP not found:** Install with `pip install fastmcp`
3. **Module not found:** Run from the `math-anything` directory, not `server/`

### Tool returns error

1. **"Unknown domain":** Available domains are `dft`, `cfd`, `md`, `fem`, `em`, `qc`, `phase_field`, `supervised_learning`. Use `list_domains` to confirm.
2. **"Unknown engine":** Only 7 engines have primary extractors (vasp, lammps, abaqus, ansys, comsol, gromacs, multiwfn). Other engines are listed but extraction is not yet implemented.
3. **JSON serialization error:** The extractor may return a non-serializable object — this is a bug, please report.

### Connection issues with Claude Desktop

1. **Check config path:** Windows: `%APPDATA%\Claude\claude_desktop_config.json`
2. **Restart Claude Desktop** after config changes
3. **Check PYTHONPATH** — must point to the `math-anything` root, not the `server/` subdirectory
4. **Verify Python** — ensure the same Python environment is used (check with `which python`)

### Running tests

```bash
cd math-anything
python -m pytest tests/test_mcp_server.py -v
```
