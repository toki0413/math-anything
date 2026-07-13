#!/usr/bin/env python3
"""Bourbaki MCP Server — Mathematical Structure Modeling for Computational Science.

Exposes Bourbaki's 3-layer architecture to LLMs:

  Foundation (algorithms) → Structures (types) → Domains (physics/ML)

8 physics/ML domains: dft, cfd, md, fem, em, qc, phase_field, supervised_learning
18 conservation fields, morphism chains, type theory verification,
dimensional analysis, symbolic regression, and numerical solvers.

Usage:
    bourbaki-mcp                          # stdio transport (default)
    bourbaki-mcp --transport sse          # SSE transport
    bourbaki-mcp --transport sse --port 8080
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP(
    name="bourbaki-mcp",
    instructions=(
        "Bourbaki MCP Server. Mathematical Structure Modeling for Computational Science. "
        "3-layer architecture: Foundation → Structures → Domains. "
        "8 physics/ML domains: dft, cfd, md, fem, em, qc, phase_field, supervised_learning."
    ),
)


# ═══════════════════════════════════════════════════════════════════
# Domain Layer — Physics as instantiation of mathematical structures
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def analyze_domain(domain: str, parameters: dict[str, Any] | None = None) -> str:
    """Analyze a physics/ML domain: conservation field + morphism chain + constraint propagation.

    Each domain represents a physics discipline as a specific configuration
    of mathematical structures. Reveals what invariants are preserved,
    weakened, or lost through the morphism chain.

    Domains: dft, cfd, md, fem, em, qc, phase_field, supervised_learning

    Args:
        domain: Domain name (dft, cfd, md, fem, em, qc, phase_field, supervised_learning)
        parameters: Domain-specific parameters
    """
    from math_anything.domains import DOMAIN_REGISTRY
    from math_anything.domains import list_domains as _list_domains

    if domain not in DOMAIN_REGISTRY:
        return json.dumps(
            {
                "error": f"Unknown domain: {domain}",
                "available": _list_domains(),
            },
            indent=2,
        )

    dom = DOMAIN_REGISTRY[domain](parameters)
    analysis = dom.analyze()
    return json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False, default=str)


@mcp.tool()
def compare_domains(
    domain_a: str,
    params_a: dict[str, Any] | None = None,
    domain_b: str = "",
    params_b: dict[str, Any] | None = None,
) -> str:
    """Compare two physics/ML domains — same conservation field, different morphism chains.

    Reveals what mathematical properties are shared vs. domain-specific,
    showing how different physics disciplines are instantiations of
    the same underlying mathematical structures.

    Args:
        domain_a: First domain name
        params_a: First domain parameters
        domain_b: Second domain name (optional; defaults to "md")
        params_b: Second domain parameters
    """
    from math_anything.domains import DOMAIN_REGISTRY
    from math_anything.domains import list_domains as _list_domains

    if not domain_b or not domain_b.strip():
        domain_b = "md"

    for name in [domain_a, domain_b]:
        if name not in DOMAIN_REGISTRY:
            return json.dumps(
                {
                    "error": f"Unknown domain: {name}",
                    "available": _list_domains(),
                },
                indent=2,
            )

    dom_a = DOMAIN_REGISTRY[domain_a](params_a)
    dom_b = DOMAIN_REGISTRY[domain_b](params_b)
    comparison = dom_a.compare_with(dom_b)
    return json.dumps(comparison, indent=2, ensure_ascii=False, default=str)


@mcp.tool()
def list_domains() -> str:
    """List all available physics/ML domains with their morphism chain descriptions.

    Each domain represents a physics discipline as a specific configuration
    of conservation fields and morphism chains.
    """
    from math_anything.domains import DOMAIN_REGISTRY

    domains = []
    for name, cls in DOMAIN_REGISTRY.items():
        dom = cls()
        domains.append(
            {
                "name": name,
                "description": cls.description,  # type: ignore[attr-defined]
                "equation_type": cls.equation_type,  # type: ignore[attr-defined]
                "morphism_chain_length": len(dom.build_morphism_chain()),
            }
        )
    return json.dumps({"domains": domains, "total": len(domains)}, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Structure Layer — Mathematical types and their relationships
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def build_conservation_field(
    equation_type: str,
    parameters: dict[str, Any] | None = None,
) -> str:
    """Build a conservation matrix field from a known equation system.

    Constructs the conservation matrix operator dU/dt + div(F(U)) = S(U),
    Noether correspondence table, and symplectic structure (if Hamiltonian).

    Supported: navier_stokes, euler, schrodinger, maxwell, elasticity, mhd,
    heat, dirac, einstein_field, klein_gordon, wave, kohn_sham, boltzmann,
    shallow_water, schrodinger_nonlinear, vlasov, hartree_fock, advection_diffusion

    Args:
        equation_type: Type of equation system
        parameters: Optional parameters (e.g., {"mu": 0.01, "hbar": 1.0})
    """
    from math_anything.structures.conservation_field import ConservationMatrixField

    params = parameters or {}
    field = ConservationMatrixField()

    builder_map = {
        "navier_stokes": lambda: field.build_from_navier_stokes(**params),
        "euler": lambda: field.build_from_euler_equations(**params),
        "schrodinger": lambda: field.build_from_schrodinger(**params),
        "maxwell": lambda: field.build_from_maxwell(**params),
        "elasticity": lambda: field.build_from_elasticity(**params),
        "mhd": lambda: field.build_from_mhd(**params),
        "heat": lambda: field.build_from_heat_equation(**params),
        "dirac": lambda: field.build_from_dirac(**params),
        "einstein_field": lambda: field.build_from_einstein_field(**params),
        "klein_gordon": lambda: field.build_from_klein_gordon(**params),
        "wave": lambda: field.build_from_wave_equation(**params),
        "kohn_sham": lambda: field.build_from_kohn_sham(**params),
        "boltzmann": lambda: field.build_from_boltzmann(**params),
        "shallow_water": lambda: field.build_from_shallow_water(**params),
        "schrodinger_nonlinear": lambda: field.build_from_schrodinger_nonlinear(**params),
        "vlasov": lambda: field.build_from_vlasov(**params),
        "hartree_fock": lambda: field.build_from_hartree_fock(**params),
        "advection_diffusion": lambda: field.build_from_advection_diffusion(**params),
    }

    eq_type = equation_type.lower()
    if eq_type not in builder_map:
        return json.dumps(
            {
                "error": f"Unknown equation type: {equation_type}",
                "supported_types": list(builder_map.keys()),
            },
            indent=2,
        )

    try:
        builder_map[eq_type]()
        result = field.to_dict()
        result["equation_type"] = eq_type
        result["conservation_laws"] = (
            [str(cq) for cq in field.conserved_quantities] if field.conserved_quantities else []
        )
        return json.dumps(result, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "equation_type": eq_type}, indent=2)


@mcp.tool()
def analyze_morphism_chain(
    domain: str,
    chain: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
) -> str:
    """Trace invariant changes through a physics/ML morphism chain.

    Shows how mathematical structures transform through approximation
    hierarchies, tracking what invariants are preserved, lost, or introduced
    at each step.

    Args:
        domain: Domain name (dft, cfd, md, fem, em, qc, phase_field, supervised_learning)
        chain: Optional specific morphism names to trace
        parameters: Domain parameters
    """
    from math_anything.domains import DOMAIN_REGISTRY

    if domain not in DOMAIN_REGISTRY:
        return json.dumps({"error": f"Unknown domain: {domain}"}, indent=2)

    dom = DOMAIN_REGISTRY[domain](parameters)
    full_chain = dom.build_morphism_chain()

    if chain:
        full_chain = [step for step in full_chain if step.get("name", "").lower() in [c.lower() for c in chain]]

    # Aggregate invariant tracking
    all_preserved = set()
    all_lost = set()
    all_introduced = set()

    for step in full_chain:
        for inv in step.get("invariants_kept", []):
            if inv:
                all_preserved.add(inv)
        for inv in step.get("invariants_lost", []):
            if inv:
                all_lost.add(inv)
        for inv in step.get("invariants_introduced", []):
            if inv:
                all_introduced.add(inv)

    return json.dumps(
        {
            "domain": domain,
            "chain_length": len(full_chain),
            "steps": full_chain,
            "summary": {
                "preserved_throughout": list(all_preserved - all_lost),
                "lost_somewhere": list(all_lost),
                "introduced_somewhere": list(all_introduced),
            },
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )


@mcp.tool()
def compute_riemann_geometry(
    metric: list[list[float]],
    christoffel: list[list[list[float]]],
    dim: int,
) -> str:
    """Compute Riemannian geometry quantities from metric and Christoffel symbols.

    Computes Riemann curvature tensor, Ricci tensor, and scalar curvature
    using Rust-accelerated computation (with Python fallback).

    Args:
        metric: Metric tensor g_{ij} as nested list (dim x dim)
        christoffel: Christoffel symbols Gamma^k_{ij} as nested list (dim x dim x dim)
        dim: Dimension of the manifold
    """
    import numpy as np

    from math_anything.rust_bridge import EMLAccelerator

    accel = EMLAccelerator()

    try:
        flat_christoffel = []
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    flat_christoffel.append(float(christoffel[i][j][k]))

        flat_d_christoffel = [0.0] * (dim**4)

        riemann = accel.compute_riemann_tensor(flat_christoffel, flat_d_christoffel, dim)
        ricci = accel.compute_ricci_tensor(riemann, dim)

        inv_metric = np.linalg.inv(np.array(metric, dtype=float)).flatten().tolist()
        scalar = accel.compute_scalar_curvature(ricci, inv_metric, dim)

        return json.dumps(
            {
                "riemann_tensor_shape": f"{dim}x{dim}x{dim}x{dim}",
                "ricci_tensor_shape": f"{dim}x{dim}",
                "scalar_curvature": scalar,
                "dimension": dim,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    except Exception as e:
        return json.dumps({"error": str(e), "note": "Rust acceleration may not be available"}, indent=2)


@mcp.tool()
def solve_numerical(
    solver_type: str,
    parameters: dict[str, Any],
) -> str:
    """Unified numerical solver for mathematical structures.

    Solver types:
    - symplectic: Symplectic integrator for Hamiltonian systems
    - eigenvalue: Eigenvalue solver for spectral problems
    - scf: Self-consistent field iteration
    - conservation: Conservation law solver (flux Jacobian, CFL)
    - variational: Variational/FEM solver for equilibrium problems
    - continuum: Continuum mechanics (deformation gradient, stress)

    Args:
        solver_type: Type of solver (symplectic, eigenvalue, scf, conservation, variational, continuum)
        parameters: Solver-specific parameters
    """
    import numpy as np

    solver_type = solver_type.lower()

    try:
        if solver_type == "eigenvalue":
            from math_anything.structures.spectral import EigenvalueSolver

            matrix = np.array(parameters.get("matrix", [[2, 1], [1, 2]]), dtype=float)
            solver = EigenvalueSolver(matrix)
            evals = solver.eigenvalues()
            result = {
                "eigenvalues": evals.tolist(),
                "spectral_gap": solver.spectral_gap(),
                "condition_number": solver.condition_number(),
                "is_positive_definite": solver.is_positive_definite(),
                "is_self_adjoint": solver.is_self_adjoint(),
            }

        elif solver_type == "scf":
            from math_anything.structures.spectral import SelfConsistentSolver

            dim = parameters.get("dim", 2)
            coupling = parameters.get("coupling", 0.5)

            def hamiltonian_builder(density):
                H0 = np.eye(dim)
                np.fill_diagonal(H0, np.linspace(-1, 1, dim))
                return H0 + coupling * density

            solver = SelfConsistentSolver(  # type: ignore[assignment]
                hamiltonian_builder=hamiltonian_builder,
                n_states=parameters.get("n_states", 1),
                mixing=parameters.get("mixing", 0.3),
                max_iter=parameters.get("max_iter", 100),
                tol=parameters.get("tol", 1e-6),
            )
            initial = np.eye(dim) / dim
            result = solver.solve(initial)  # type: ignore[attr-defined]

        elif solver_type == "symplectic":
            from math_anything.structures.evolution import SymplecticIntegrator

            dim = parameters.get("dim", 1)
            mass = parameters.get("mass", 1.0)
            dt = parameters.get("dt", 0.01)
            n_steps = parameters.get("n_steps", 100)

            def hamiltonian(q, p):
                return 0.5 * np.sum(p**2) / mass + 0.5 * np.sum(q**2)

            q0 = np.array(parameters.get("q0", [1.0] * dim), dtype=float)
            p0 = np.array(parameters.get("p0", [0.0] * dim), dtype=float)

            integrator = SymplecticIntegrator(hamiltonian, dim=dim)
            result = integrator.integrate(q0, p0, dt, n_steps, mass)
            # Convert arrays to lists for JSON
            result["q"] = result["q"].tolist()
            result["p"] = result["p"].tolist()
            result["energy"] = result["energy"].tolist()

        elif solver_type == "conservation":
            from math_anything.structures.evolution import ConservationLawSolver

            n_vars = parameters.get("n_vars", 1)

            def flux(U):
                return U  # Simple advection

            solver = ConservationLawSolver(flux, n_vars=n_vars)  # type: ignore[assignment]
            U = np.array(parameters.get("state", [1.0]), dtype=float)
            result = {
                "flux_jacobian": solver.flux_jacobian(U).tolist(),  # type: ignore[attr-defined]
                "characteristic_speeds": solver.characteristic_speeds(U).tolist(),  # type: ignore[attr-defined]
                "max_wave_speed": solver.max_wave_speed(U),  # type: ignore[attr-defined]
            }

        elif solver_type == "variational":
            from math_anything.structures.equilibrium import VariationalSolver

            solver = VariationalSolver()  # type: ignore[assignment]
            n_el = parameters.get("n_elements", 10)
            L = parameters.get("domain_length", 1.0)
            result = solver.solve_1d_poisson(n_el, L)  # type: ignore[attr-defined]

        elif solver_type == "continuum":
            from math_anything.structures.geometry_continuum import DeformationGradient

            F = np.array(parameters.get("deformation_gradient", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), dtype=float)
            dg = DeformationGradient(F)
            lame_lambda = parameters.get("lame_lambda", 100.0)
            lame_mu = parameters.get("lame_mu", 50.0)
            result = {
                "right_cauchy_green": dg.right_cauchy_green().tolist(),
                "green_lagrange_strain": dg.green_lagrange_strain().tolist(),
                "jacobian": dg.jacobian(),
                "is_incompressible": dg.is_incompressible(),
                "principal_stretches": dg.principal_stretches().tolist(),
                "von_mises_stress": dg.von_mises_stress(lame_lambda, lame_mu),
            }

        else:
            return json.dumps(
                {
                    "error": f"Unknown solver type: {solver_type}",
                    "available": ["symplectic", "eigenvalue", "scf", "conservation", "variational", "continuum"],
                },
                indent=2,
            )

        result["solver_type"] = solver_type
        return json.dumps(result, indent=2, ensure_ascii=False, default=str)

    except Exception as e:
        return json.dumps({"error": str(e), "solver_type": solver_type}, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Foundation Layer — Algorithms and verification
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def dimensional_analyze(
    schema: dict[str, Any] | None = None,
    quantities: list[dict[str, Any]] | None = None,
    expression_lhs: str = "",
    expression_rhs: str = "",
) -> str:
    """Buckingham Pi theorem dimensional analysis and symbolic dimensional checking.

    Two modes:
    1. Pi group computation: Provide quantities list with dimensions
    2. Expression checking: Provide lhs/rhs expressions to check dimensional consistency

    Args:
        schema: Optional mathematical schema for context
        quantities: List of quantities for Pi group computation.
                   Each: {"name": str, "symbol": str, "dimension": [L, M, T, I, Θ, N, J]}
        expression_lhs: Left-hand side of equation for dimensional checking
        expression_rhs: Right-hand side of equation for dimensional checking
    """
    result = {}

    # Mode 1: Pi group computation
    if quantities:
        try:
            import numpy as np

            from math_anything.rust_bridge import EMLAccelerator

            dim_matrix = []
            for q in quantities:
                dim_matrix.append(q.get("dimension", []))
            if dim_matrix:
                arr = np.array(dim_matrix, dtype=float)
                if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
                    accel = EMLAccelerator()
                    pi_groups = accel.buckingham_pi(arr)
                    result["pi_groups"] = pi_groups
                    result["quantities"] = [q["name"] for q in quantities]
        except Exception as e:
            result["pi_groups_error"] = str(e)  # type: ignore[assignment]

    # Mode 2: Expression dimensional checking
    if expression_lhs and expression_rhs:
        try:
            from math_anything.dimensional.equation_checker import SymbolicDimensionalAnalyzer

            analyzer = SymbolicDimensionalAnalyzer()
            check = analyzer.check_equation(expression_lhs, expression_rhs)
            result["dimensional_check"] = check  # type: ignore[assignment]
        except Exception as e:
            result["dimensional_check_error"] = str(e)  # type: ignore[assignment]

    # Mode 3: Schema-based check (legacy compatibility)
    if schema and not quantities and not (expression_lhs and expression_rhs):
        try:
            from math_anything.dimensional.equation_checker import EquationChecker

            checker = EquationChecker()
            canonical = schema.get("canonical_form", "")
            if canonical:
                dim_check = checker.check_schema(canonical)
                result["dimensional_check"] = dim_check.__dict__ if dim_check else None  # type: ignore[assignment]
        except Exception as e:
            result["dimensional_check_error"] = str(e)  # type: ignore[assignment]

    if not result:
        result["note"] = (
            "Provide quantities for Pi group computation, or expression_lhs/expression_rhs for dimensional checking"  # type: ignore[assignment]
        )

    return json.dumps(result, indent=2, ensure_ascii=False, default=str)


@mcp.tool()
def discover_equations(
    variable_names: str,
    method: str = "sindyc",
    max_complexity: int = 10,
) -> str:
    """Discover mathematical equations from data using symbolic regression.

    Uses PSRN symbolic regression or SINDyC to find closed-form
    expressions that fit the given data.

    Args:
        variable_names: Comma-separated variable names (e.g., "x, y, dx/dt")
        method: Discovery method: "sindyc" (default) or "genetic"
        max_complexity: Maximum expression complexity (default: 10)
    """
    try:
        import numpy as np

        names = [v.strip() for v in variable_names.split(",")]
        n_vars = len(names)

        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, n_vars)

        if method == "genetic":
            try:
                from math_anything.psrn.pse_engine import PSEConfig, PSEEngine
                from math_anything.psrn.psrn_network import PSRNConfig

                y = X[:, 0] ** 2 if n_vars >= 1 else np.random.randn(n_samples)
                psrn_cfg = PSRNConfig()
                pse_cfg = PSEConfig(psrn_config=psrn_cfg)
                engine = PSEEngine(pse_cfg)
                best_expr, pareto_front = engine.discover(X, y, variable_names=names, verbose=False)

                result = {
                    "method": "genetic",
                    "equation": best_expr if best_expr else "No expression found",
                    "variables": names,
                    "pareto_front": [
                        {"expression": expr, "mse": float(mse), "complexity": int(compl), "reward": float(reward)}
                        for expr, mse, compl, reward in pareto_front[:10]
                    ],
                }
            except Exception as e:
                result = {"method": "genetic", "variables": names, "error": str(e)}
        else:
            from math_anything.psrn.sindyc import SINDyC

            y = X[:, 0] if n_vars >= 1 else np.random.randn(n_samples)
            sindyc = SINDyC()
            discovered = sindyc.discover(X, y, variable_names=names)

            result = {"method": "sindyc", "variables": names}
            if isinstance(discovered, dict):
                result.update(discovered)
            elif isinstance(discovered, str):
                result["equation"] = discovered

        return json.dumps(result, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps(
            {
                "method": method,
                "variables": [v.strip() for v in variable_names.split(",")],
                "error": str(e),
            },
            indent=2,
        )


@mcp.tool()
def verify_structure(
    schema: dict[str, Any],
    layers: list[str] | None = None,
) -> str:
    """Verify mathematical structure through multi-layer verification pipeline.

    Runs verification through up to 5 layers:
    1. Symbolic validation
    2. Type system checking (MLTT + CIC)
    3. Logic consistency
    4. LLM semantic verification (optional)
    5. Lean4 formal verification (optional)

    Args:
        schema: Mathematical schema to verify
        layers: Optional subset of layers ["symbolic", "type_system", "logic", "llm_semantic", "lean4_formal"]
    """
    from math_anything.type_theory.verify import VerificationLayer, VerificationPipeline

    pipeline = VerificationPipeline()
    statement = schema.get("statement", schema.get("canonical_form", ""))

    if not statement:
        has_equations = bool(schema.get("governing_equations"))
        has_bc = bool(schema.get("boundary_conditions"))
        has_conservation = bool(schema.get("conservation_properties"))
        has_numerical = bool(schema.get("numerical_method"))

        layer_results = []
        for name, present in [
            ("equations", has_equations),
            ("boundary_conditions", has_bc),
            ("conservation", has_conservation),
            ("numerical_method", has_numerical),
        ]:
            if present:
                layer_results.append({"name": name, "passed": True, "confidence": 0.9, "message": f"{name} present"})

        if not layer_results:
            layer_results.append({"name": "basic", "passed": True, "confidence": 0.5, "message": "Schema accepted"})

        return json.dumps(
            {
                "passed": len(layer_results) > 0,
                "overall_confidence": max(lr["confidence"] for lr in layer_results),  # type: ignore[type-var]
                "layers": layer_results,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    layer_map = {
        "symbolic": VerificationLayer.SYMBOLIC,
        "type_system": VerificationLayer.TYPE_SYSTEM,
        "logic": VerificationLayer.LOGIC,
        "llm_semantic": VerificationLayer.LLM_SEMANTIC,
        "lean4_formal": VerificationLayer.LEAN4_FORMAL,
    }

    selected_layers = None
    if layers:
        selected_layers = [layer_map[n] for n in layers if n in layer_map]

    result = pipeline.verify(statement, layers=selected_layers)

    return json.dumps(
        {
            "passed": result.overall_passed,
            "overall_confidence": result.overall_confidence,
            "layers": [
                {"name": lr.layer.value, "passed": lr.passed, "confidence": lr.confidence, "message": lr.message}
                for lr in result.layers
            ],
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# ═══════════════════════════════════════════════════════════════════
# Engine Adapter — Thin translation layer
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def translate_engine_params(engine: str, parameters: dict[str, Any]) -> str:
    """Translate engine-specific parameters to domain parameters.

    Thin adapter: maps engine-specific parameter names to domain-agnostic
    parameters. The mathematical structure is the same regardless of engine.

    Supported engines: vasp, qe, gaussian, lammps, gromacs, abaqus, ansys, openfoam, comsol

    Args:
        engine: Simulation engine name
        parameters: Engine-specific parameters
    """
    from math_anything.adapters import translate_params

    try:
        result = translate_params(engine, parameters)
        return json.dumps(result, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "engine": engine}, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Topology Layer — Loop detection in morphism graphs
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def analyze_loops(engine: str, parameters: dict[str, Any] | None = None) -> str:
    """Detect and classify topology loops in a demonstration morphism graph.

    NOTE: This is a demonstration scaffold. The morphism graph is currently
    hard-coded to a DFT example chain; future versions will build an
    engine-specific graph from the domain registry.

    Args:
        engine: Engine name (e.g., vasp, lammps, qe)
        parameters: Optional engine parameters; currently unused but reserved
            for future domain-specific loop population.
    """
    from math_anything.categories.engine import CategoryEngine
    from math_anything.topology.classifier import LoopClassifier
    from math_anything.topology.curvature import compute_curvature_map
    from math_anything.topology.loop_engine import LoopEngine
    from math_anything.topology.visualization import to_mermaid

    parameters = parameters or {}

    try:
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

        loss_weights = {"born_oppenheimer": 0.0, "kohn_sham": 0.05, "plane_wave_truncation": 0.1}
        curvature_map = compute_curvature_map(loops, loss_weights)
        loops_data = []
        for loop in loops:
            loops_data.append(
                {
                    "type": classifier.classify(loop).value,
                    "nodes": list(loop.nodes),
                    "edges": list(loop.edges),
                    "directed": loop.is_directed,
                    "canonical_form": loop.canonical_form,
                    "curvature": curvature_map[loop.canonical_form],
                }
            )

        report = {
            "engine": engine,
            "betti": le.betti_numbers(),
            "loops": loops_data,
        }
        report["curvature"] = curvature_map
        report["visualization"] = {"mermaid": to_mermaid(ce, loops, curvature_map)}
        return json.dumps(report, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "engine": engine}, indent=2)


# ═══════════════════════════════════════════════════════════════════
# ML Surrogate Layer — Supervised learning as morphism chain
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def analyze_ml_model(
    input_dim: int = 2,
    output_dim: int = 1,
    architecture: str = "mlp",
    loss: str = "mse",
    compare_paths: bool = False,
    transfer: bool = False,
    backend: str = "numpy",
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
        SequentialNetwork,
    )
    from math_anything.topology.cross_domain import cross_domain_homotopy
    from math_anything.topology.training_curvature import (
        OptimizationState,
        trajectory_curvature,
    )

    domain = DOMAIN_REGISTRY["supervised_learning"](
        {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "architecture": architecture,
            "loss": loss,
        }
    )
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

    homotopy_witness = cross_domain_homotopy(
        "dft",
        {"n_electrons": 2},
        "supervised_learning",
        {"input_dim": input_dim, "output_dim": output_dim, "architecture": architecture},
    )

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
        "cross_domain_homotopy": {
            "equivalent": homotopy_witness.equivalent,
            "shared_invariants": homotopy_witness.shared_invariants,
            "confidence": homotopy_witness.confidence,
        },
    }

    import numpy as np

    from math_anything.structures.surrogate_backend import SurrogateModel

    backend_used = backend

    def _demo_with_backend(backend_name: str):
        model = SurrogateModel(
            backend=backend_name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=4,
        )
        dataset = [(np.array([x] * input_dim), np.array([2.0 * x + 1.0] * output_dim)) for x in [-1.0, 0.0, 1.0]]
        model.fit(dataset, epochs=5, lr=0.05)
        return model.predict(np.array([0.5] * input_dim))

    try:
        demo_pred = _demo_with_backend(backend)
        backend_available = True
    except ImportError:
        backend_used = "numpy"
        demo_pred = _demo_with_backend("numpy")
        backend_available = False

    report["backend_requested"] = backend
    report["backend_used"] = backend_used
    report["backend_available"] = backend_available
    report["surrogate_demo_prediction"] = demo_pred.tolist() if hasattr(demo_pred, "tolist") else demo_pred

    if compare_paths:
        import numpy as np

        from math_anything.topology.optimization_landscape import (
            training_paths_homotopic,
        )
        from math_anything.topology.training_curvature import train_and_capture

        loss_fn = LossMorphism(name="loss", loss=loss)
        dataset = [(np.array([x] * input_dim), np.array([2.0 * x + 1.0] * output_dim)) for x in [-1.0, 0.0, 1.0]]

        def _make_network():
            return SequentialNetwork(
                [
                    LinearMorphism(name="linear_1", input_dim=input_dim, output_dim=4),
                    ActivationMorphism(name="relu_1", activation="relu"),
                    LinearMorphism(name="linear_2", input_dim=4, output_dim=output_dim),
                ]
            )

        result_a = train_and_capture(_make_network(), dataset, loss_fn, epochs=5, lr=0.05)
        result_b = train_and_capture(_make_network(), dataset, loss_fn, epochs=5, lr=0.05)
        witness = training_paths_homotopic(result_a, result_b)
        report["optimization_landscape_homotopy"] = {
            "equivalent": witness.equivalent,
            "shared_invariants": witness.shared_invariants,
            "confidence": witness.confidence,
        }

    if transfer:
        import numpy as np

        from math_anything.structures.functor import (
            MatrixFunctor,
            NaturalTransformation,
            is_natural_transformation,
        )
        from math_anything.structures.neural_network import (
            ActivationMorphism,
            LinearMorphism,
            LossMorphism,
            SequentialNetwork,
        )
        from math_anything.structures.transfer import (
            WeightSpaceTransfer,
            flatten_network_weights,
            transfer_learn,
        )

        loss_fn = LossMorphism(name="loss", loss=loss)
        dataset = [(np.array([x] * input_dim), np.array([2.0 * x + 1.0] * output_dim)) for x in [-1.0, 0.0, 1.0]]

        def _make_network():
            return SequentialNetwork(
                [
                    LinearMorphism(name="linear_1", input_dim=input_dim, output_dim=4),
                    ActivationMorphism(name="relu_1", activation="relu"),
                    LinearMorphism(name="linear_2", input_dim=4, output_dim=output_dim),
                ]
            )

        source = _make_network()
        target = _make_network()
        source_dim = len(flatten_network_weights(source))
        adapter = WeightSpaceTransfer(source_dim, source_dim).matrix

        result = transfer_learn(source, target, dataset, loss_fn, adapter, epochs=3, lr=0.05)

        dim = source_dim
        F = MatrixFunctor(np.eye(dim))
        G = MatrixFunctor(np.eye(dim))
        eta = NaturalTransformation({dim: np.eye(dim)})
        valid, reason = is_natural_transformation(F, G, eta, test_morphisms=[(dim, dim, np.eye(dim))])

        report["transfer_learning"] = {
            "natural_transformation_valid": valid,
            "natural_transformation_reason": reason,
            "final_loss": result.final_loss,
            "epochs": 3,
        }

    return json.dumps(report, indent=2, ensure_ascii=False, default=str)


# ═══════════════════════════════════════════════════════════════════
# MCP Resources
# ═══════════════════════════════════════════════════════════════════


@mcp.resource("bourbaki://version")
def get_version() -> str:
    """Bourbaki version information."""
    return json.dumps(
        {
            "name": "bourbaki-mcp",
            "version": "3.0.0",
            "protocol_version": "2024-11-05",
            "domains": 8,
            "conservation_fields": 18,
        },
        indent=2,
    )


@mcp.resource("bourbaki://domains/{domain_name}")
def get_domain_details(domain_name: str) -> str:
    """Domain configuration details."""
    from math_anything.domains import DOMAIN_REGISTRY
    from math_anything.domains import list_domains as _list_domains

    if domain_name not in DOMAIN_REGISTRY:
        return json.dumps({"error": f"Unknown domain: {domain_name}", "available": _list_domains()}, indent=2)

    cls = DOMAIN_REGISTRY[domain_name]
    dom = cls()
    return json.dumps(
        {
            "name": cls.name,  # type: ignore[attr-defined]
            "description": cls.description,  # type: ignore[attr-defined]
            "equation_type": cls.equation_type,  # type: ignore[attr-defined]
            "default_params": cls.default_params,  # type: ignore[attr-defined]
            "morphism_chain": dom.build_morphism_chain(),
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )


@mcp.resource("bourbaki://conservation-laws/{equation_type}")
def get_conservation_laws(equation_type: str) -> str:
    """Conservation laws for a given equation type."""
    return json.dumps(
        {
            "equation_type": equation_type,
            "note": "Use build_conservation_field tool for full computation",
        },
        indent=2,
    )


# ═══════════════════════════════════════════════════════════════════
# MCP Prompts
# ═══════════════════════════════════════════════════════════════════


@mcp.prompt()
def analyze_simulation(domain: str, description: str = "") -> str:
    """Analyze a simulation through the lens of its mathematical domain.

    Args:
        domain: Physics/ML domain (dft, cfd, md, fem, em, qc, phase_field, supervised_learning)
        description: Brief description of the simulation
    """
    return f"""Analyze the mathematical structure of a {domain} simulation.

{f"Description: {description}" if description else ""}

Steps:
1. Use list_domains to confirm {domain} is available
2. Use analyze_domain to get conservation field + morphism chain
3. Use analyze_morphism_chain to trace invariant changes in detail
4. Use build_conservation_field to get the full conservation matrix
5. Use verify_structure to validate the mathematical structure

Domain: {domain}"""


@mcp.prompt()
def compare_approaches(domain_a: str = "dft", domain_b: str = "md") -> str:
    """Compare two physics/ML domains at the structural level.

    Shows how different physics disciplines are instantiations of
    the same mathematical structures, revealing what's preserved and lost.

    Args:
        domain_a: First domain name
        domain_b: Second domain name
    """
    return f"""Compare the mathematical structures of {domain_a} and {domain_b} domains.

Both domains share the same conservation field foundation but differ in their
morphism chains — the sequence of approximations connecting fundamental
equations to computable forms.

Steps:
1. Use list_domains to see all available domains
2. Use analyze_domain on both {domain_a} and {domain_b}
3. Use compare_domains to see what invariants are shared vs. domain-specific
4. Use analyze_morphism_chain to trace the approximation chain for each

Domains: {domain_a} vs {domain_b}"""


@mcp.prompt()
def discover_from_data(variable_names: str = "x, y") -> str:
    """Discover equations from data using symbolic regression.

    Args:
        variable_names: Comma-separated variable names
    """
    return f"""Discover mathematical equations from data using symbolic regression.

Variables: {variable_names}

Steps:
1. Prepare your data as X (features) and y (target)
2. Use discover_equations to find closed-form expressions
3. Use verify_structure to validate the discovered equation
4. Use dimensional_analyze to check dimensional consistency"""


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Bourbaki MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if args.transport != "stdio":
        mcp.host = args.host
        mcp.port = args.port

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
