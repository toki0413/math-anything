# Bourbaki MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes Bourbaki's mathematical structure extraction capabilities to LLMs through standardized tool interfaces.

Built on [FastMCP](https://gofastmcp.com) SDK v3.2+.

## What is Bourbaki MCP?

Bourbaki MCP extracts, compares, validates, and discovers mathematical structures across **4 physics domains** (DFT, CFD, MD, FEM) and **19 computational science engines**. It provides:

- **Domain analysis** — conservation fields, morphism chains, and constraint propagation for each physics discipline
- **Cross-domain comparison** — what invariants are shared vs. domain-specific
- **Structure extraction** — canonical forms, variable dependencies, discretization, solution strategies
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

Analyze a physics domain: conservation field + morphism chain + constraint propagation.

**Parameters:**
- `domain` (string, required) — Domain name: `dft`, `cfd`, `md`, `fem`
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

Compare two physics domains — same conservation field, different morphism chains.

**Parameters:**
- `domain_a` (string, required) — First domain name
- `params_a` (object, optional) — First domain parameters
- `domain_b` (string, required) — Second domain name
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

List all available physics domains with their morphism chain descriptions.

**Parameters:** None

**Returns:** Domain list with names, descriptions, equation types, and morphism chain lengths.

### Structure Extraction

#### 4. `extract_mathematical_structure`

Extract mathematical structure from simulation parameters.

**Parameters:**
- `engine` (string, required) — Simulation engine name
- `parameters` (object, required) — Engine-specific parameters

**Example:**
```json
{
  "engine": "vasp",
  "parameters": {"encut": 520, "ediff": 1e-6, "sigma": 0.05}
}
```

**Returns:** Mathematical structure with canonical forms, variable dependencies, discretization scheme, solution strategy, approximations, and mathematical decoding.

#### 5. `compare_calculations`

Compare two mathematical schemas and report differences.

**Parameters:**
- `schema_a` (object, required) — First EnhancedMathSchema
- `schema_b` (object, required) — Second EnhancedMathSchema
- `critical_only` (boolean, optional) — Only report critical changes

**Returns:** Diff report with categorized changes (critical, warning, info).

#### 6. `validate_constraints`

Validate symbolic constraints in a mathematical schema.

**Parameters:**
- `schema` (object, required) — EnhancedMathSchema

**Returns:** Validation report with constraint expressions and statuses.

#### 7. `list_supported_engines`

List all 19 supported simulation engines organized by category.

**Parameters:** None

**Returns:** Engine list with categories:
- **Quantum Mechanics:** vasp, qe, cp2k, gaussian, gamess, nwchem, multiwfn
- **Molecular Dynamics:** lammps, gromacs, liggghts
- **Continuum Mechanics:** abaqus, ansys, comsol, solidworks
- **CFD:** openfoam, fluent, su2
- **Uncertainty Quantification:** dakota
- **Multiscale:** voxel

### Conservation & Constraints

#### 8. `analyze_constraints`

Analyze mathematical invariants and constraints for a calculation setup.

**Parameters:**
- `engine` (string, required) — Simulation engine name
- `parameters` (object, required) — Engine-specific parameters
- `morphism_chain` (array of strings, optional) — Morphism chain for propagation analysis

**Returns:** Invariant analysis with conservation laws and constraint propagation.

#### 9. `build_conservation_field`

Build conservation matrix field for a given equation type.

**Parameters:**
- `equation_type` (string, required) — One of: `navier_stokes`, `euler`, `schrodinger`, `maxwell`, `elasticity`, `mhd`, `heat`, `dirac`, `einstein_field`, `klein_gordon`, `wave`, `kohn_sham`, `boltzmann`, `shallow_water`, `schrodinger_nonlinear`, `vlasov`, `hartree_fock`, `advection_diffusion`

**Returns:** Conservation laws with Noether correspondence and symmetry mappings.

#### 10. `dimensional_analyze`

Buckingham π theorem dimensional analysis.

**Parameters:**
- `schema` (object, required) — Schema with `canonical_form` key
- `quantities` (array, optional) — List of quantity dicts with `name`, `symbol`, `dimension`

**Returns:** Dimensional consistency check and dimensionless π groups.

### Verification & Discovery

#### 11. `verify_mathematical_structure`

Verify mathematical structure consistency and completeness.

**Parameters:**
- `schema` (object, required) — EnhancedMathSchema

**Returns:** Verification report checking governing equations, boundary conditions, conservation properties, and discretization.

#### 12. `discover_equations`

Discover governing equations from data using symbolic regression.

**Parameters:**
- `variable_names` (string, required) — Comma-separated variable names
- `method` (string, optional) — Discovery method: `sindyc`, `genetic`
- `max_complexity` (integer, optional) — Maximum equation complexity (default: 10)

**Returns:** Symbolic regression configuration with candidate function library.

#### 13. `query_knowledge_base`

Query the mathematical knowledge base for related concepts.

**Parameters:**
- `query` (string, required) — Search query
- `sources` (array of strings, optional) — Sources to search: `arxiv`, `wikidata`, `nist`
- `max_results` (integer, optional) — Maximum results (default: 5)

**Returns:** Search results from arXiv, Wikidata, and NIST Interatomic Potentials.

### Morphisms & Geometry

#### 14. `analyze_morphism_chain`

Analyze a chain of mathematical morphisms.

**Parameters:**
- `engine` (string, required) — Starting engine
- `parameters` (object, required) — Engine parameters
- `chain` (array of strings, optional) — Morphism chain to trace

**Returns:** Step-by-step analysis with invariants kept, lost, and introduced at each stage, plus cumulative summary.

#### 15. `compute_riemann_geometry`

Compute Riemannian geometric structures from a metric specification.

**Parameters:**
- `metric` (array of arrays, required) — Metric tensor g_{ij} (dim x dim)
- `christoffel` (array of arrays of arrays, required) — Christoffel symbols Gamma^k_{ij} (dim x dim x dim)
- `dim` (integer, required) — Manifold dimension

**Returns:** Riemann tensor, Ricci tensor, scalar curvature.

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

Compare two physics domains at the structural level — what invariants are shared vs. domain-specific.

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

1. **"Unknown domain":** Available domains are `dft`, `cfd`, `md`, `fem`. Use `list_domains` to confirm.
2. **"Unknown engine":** Only 7 engines have primary extractors (vasp, lammps, abaqus, ansys, comsol, gromacs, multiwfn). Other engines are listed but extraction is not yet implemented.
3. **JSON serialization error:** The extractor may return a non-serializable object — this is a bug, please report.
4. **Knowledge base unavailable:** Set API keys for arXiv/Wikidata/NIST access. The server gracefully degrades to offline mode.

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
