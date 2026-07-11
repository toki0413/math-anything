"""Bourbaki MCP Server Tests — Protocol compliance and tool validation."""

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "server"))
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMCPToolRegistration(unittest.TestCase):
    """Test that all MCP tools are properly registered."""

    def setUp(self):
        from math_anything.mcp_server import mcp
        self.mcp = mcp

    def test_server_name(self):
        self.assertEqual(self.mcp.name, "bourbaki-mcp")

    def test_tool_count(self):
        """Should have 13 tools registered."""
        tools = asyncio.run(self.mcp.list_tools())
        self.assertEqual(len(tools), 13)

    def test_required_tools_exist(self):
        """Check all required tools are registered."""
        tools = asyncio.run(self.mcp.list_tools())
        tool_names = [t.name for t in tools]
        required = [
            # Domain Layer
            "analyze_domain",
            "compare_domains",
            "list_domains",
            # Structure Layer
            "build_conservation_field",
            "analyze_morphism_chain",
            "compute_riemann_geometry",
            "solve_numerical",
            # Foundation Layer
            "dimensional_analyze",
            "discover_equations",
            "verify_structure",
            # Engine Adapter
            "translate_engine_params",
            # Topology Layer
            "analyze_loops",
            # ML surrogate Layer
            "analyze_ml_model",
        ]
        for name in required:
            self.assertIn(name, tool_names, f"Tool '{name}' not registered")


class TestConservationField(unittest.TestCase):
    """Test the build_conservation_field tool."""

    def test_navier_stokes(self):
        from math_anything.mcp_server import build_conservation_field
        result = json.loads(build_conservation_field("navier_stokes"))
        self.assertIsInstance(result, dict)
        self.assertEqual(result["equation_type"], "navier_stokes")
        self.assertIn("conservation_laws", result)

    def test_unknown_equation(self):
        from math_anything.mcp_server import build_conservation_field
        result = json.loads(build_conservation_field("unknown_equation"))
        self.assertIn("error", result)
        self.assertIn("supported_types", result)

    def test_supported_equation_types(self):
        from math_anything.mcp_server import build_conservation_field
        result = json.loads(build_conservation_field("unknown"))
        supported = result.get("supported_types", [])
        expected = ["navier_stokes", "euler", "schrodinger", "maxwell", "elasticity",
                    "mhd", "heat", "dirac", "einstein_field"]
        for eq in expected:
            self.assertIn(eq, supported)


class TestDimensionalAnalysis(unittest.TestCase):
    """Test the dimensional_analyze tool."""

    def test_basic_analysis(self):
        from math_anything.mcp_server import dimensional_analyze
        result = json.loads(dimensional_analyze(schema={"canonical_form": "F = ma"}))
        self.assertIsInstance(result, dict)

    def test_with_quantities(self):
        from math_anything.mcp_server import dimensional_analyze
        result = json.loads(dimensional_analyze(
            schema={"canonical_form": "F = ma"},
            quantities=[
                {"name": "force", "symbol": "F", "dimension": [1, 1, -2]},
                {"name": "mass", "symbol": "m", "dimension": [0, 1, 0]},
                {"name": "acceleration", "symbol": "a", "dimension": [1, 0, -2]},
            ],
        ))
        self.assertIsInstance(result, dict)


class TestVerifyStructure(unittest.TestCase):
    """Test the verify_structure tool."""

    def test_verify_basic(self):
        from math_anything.mcp_server import verify_structure
        result = json.loads(verify_structure({"problem_type": "test"}))
        self.assertIn("passed", result)

    def test_verify_complete_schema(self):
        from math_anything.mcp_server import verify_structure
        schema = {
            "governing_equations": [{"id": "eq1"}],
            "boundary_conditions": [{"id": "bc1"}],
            "conservation_properties": {"energy": True},
            "numerical_method": {"discretization": "fem"},
        }
        result = json.loads(verify_structure(schema))
        self.assertTrue(result["passed"])


class TestTranslateEngineParams(unittest.TestCase):
    """Test the translate_engine_params tool."""

    def test_translate_vasp(self):
        from math_anything.mcp_server import translate_engine_params
        result = json.loads(translate_engine_params("vasp", {"ENCUT": 500, "EDIFF": 1e-6, "ISPIN": 2}))
        self.assertEqual(result["engine"], "vasp")
        self.assertEqual(result["domain"], "dft")
        self.assertEqual(result["domain_params"]["ecutwfc"], 500)
        self.assertEqual(result["domain_params"]["scf_tol"], 1e-6)
        self.assertEqual(result["domain_params"]["n_spin"], 2)

    def test_translate_lammps(self):
        from math_anything.mcp_server import translate_engine_params
        result = json.loads(translate_engine_params("lammps", {"timestep": 0.001, "temperature": 300}))
        self.assertEqual(result["engine"], "lammps")
        self.assertEqual(result["domain"], "md")
        self.assertEqual(result["domain_params"]["dt"], 0.001)

    def test_translate_unknown_engine(self):
        from math_anything.mcp_server import translate_engine_params
        result = json.loads(translate_engine_params("unknown_engine", {"param": 1}))
        self.assertEqual(result["engine"], "unknown_engine")
        self.assertEqual(result["domain"], "unknown")


class TestMCPResources(unittest.TestCase):
    """Test MCP resource handlers."""

    def test_version_resource(self):
        from math_anything.mcp_server import get_version
        result = json.loads(get_version())
        self.assertEqual(result["name"], "bourbaki-mcp")
        self.assertIn("domains", result)
        self.assertIn("conservation_fields", result)

    def test_domain_details_resource(self):
        from math_anything.mcp_server import get_domain_details
        result = json.loads(get_domain_details("dft"))
        self.assertIn("name", result)
        self.assertIn("equation_type", result)

    def test_conservation_laws_resource(self):
        from math_anything.mcp_server import get_conservation_laws
        result = json.loads(get_conservation_laws("navier_stokes"))
        self.assertIn("equation_type", result)


class TestMCPPrompts(unittest.TestCase):
    """Test MCP prompt templates."""

    def test_analyze_simulation_prompt(self):
        from math_anything.mcp_server import analyze_simulation
        prompt = analyze_simulation("dft", "DFT calculation of silicon")
        self.assertIn("dft", prompt)
        self.assertIn("analyze_domain", prompt)

    def test_compare_approaches_prompt(self):
        from math_anything.mcp_server import compare_approaches
        prompt = compare_approaches("dft", "md")
        self.assertIn("dft", prompt)
        self.assertIn("md", prompt)

    def test_discover_from_data_prompt(self):
        from math_anything.mcp_server import discover_from_data
        prompt = discover_from_data("x, y, z")
        self.assertIn("discover_equations", prompt)


class TestDiscoverEquations(unittest.TestCase):
    """Test the discover_equations tool."""

    def test_basic_discovery(self):
        from math_anything.mcp_server import discover_equations
        result = json.loads(discover_equations("x, y, dx/dt"))
        self.assertIn("method", result)
        self.assertIn("variables", result)

    def test_custom_method(self):
        from math_anything.mcp_server import discover_equations
        result = json.loads(discover_equations("x, y", method="genetic"))
        self.assertIn("method", result)


class TestComputeRiemannGeometry(unittest.TestCase):
    """Test the compute_riemann_geometry tool."""

    def test_euclidean(self):
        from math_anything.mcp_server import compute_riemann_geometry
        # Euclidean metric in 3D: identity matrix, zero Christoffel symbols
        dim = 3
        metric = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
        christoffel = [[[0.0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
        result = json.loads(compute_riemann_geometry(metric, christoffel, dim))
        self.assertIn("scalar_curvature", result)
        # Flat space should have zero (or near-zero) curvature
        self.assertAlmostEqual(result["scalar_curvature"], 0.0, places=5)

    def test_schwarzschild(self):
        from math_anything.mcp_server import compute_riemann_geometry
        # Simplified test: 2D with non-trivial metric
        dim = 2
        metric = [[2.0, 0.0], [0.0, 3.0]]
        christoffel = [[[0.0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
        result = json.loads(compute_riemann_geometry(metric, christoffel, dim))
        self.assertIn("scalar_curvature", result)


class TestAnalyzeMorphismChain(unittest.TestCase):
    """Test the analyze_morphism_chain tool."""

    def test_dft_chain(self):
        from math_anything.mcp_server import analyze_morphism_chain
        result = json.loads(analyze_morphism_chain("dft"))
        self.assertIn("domain", result)
        self.assertIn("steps", result)
        self.assertIn("summary", result)

    def test_unknown_domain(self):
        from math_anything.mcp_server import analyze_morphism_chain
        result = json.loads(analyze_morphism_chain("nonexistent"))
        self.assertIn("error", result)


class TestDomainTools(unittest.TestCase):
    """Test the Domain Instantiation Layer tools."""

    def test_list_domains(self):
        from math_anything.mcp_server import list_domains
        result = json.loads(list_domains())
        self.assertIn("domains", result)
        self.assertGreaterEqual(result["total"], 7)
        domain_names = [d["name"] for d in result["domains"]]
        self.assertIn("dft", domain_names)
        self.assertIn("cfd", domain_names)
        self.assertIn("md", domain_names)
        self.assertIn("fem", domain_names)
        self.assertIn("em", domain_names)
        self.assertIn("qc", domain_names)
        self.assertIn("phase_field", domain_names)

    def test_analyze_dft_domain(self):
        from math_anything.mcp_server import analyze_domain
        result = json.loads(analyze_domain("dft", {"n_electrons": 10}))
        self.assertIn("domain_name", result)
        self.assertIn("preserved", result)
        self.assertIn("lost", result)
        self.assertEqual(result["domain_name"], "dft")

    def test_analyze_cfd_domain(self):
        from math_anything.mcp_server import analyze_domain
        result = json.loads(analyze_domain("cfd", {"regime": "incompressible"}))
        self.assertIn("domain_name", result)
        self.assertEqual(result["domain_name"], "cfd")

    def test_analyze_md_domain(self):
        from math_anything.mcp_server import analyze_domain
        result = json.loads(analyze_domain("md", {"n_atoms": 1000}))
        self.assertIn("domain_name", result)
        self.assertIn("preserved", result)

    def test_analyze_fem_domain(self):
        from math_anything.mcp_server import analyze_domain
        result = json.loads(analyze_domain("fem", {"basis_degree": 2}))
        self.assertIn("domain_name", result)

    def test_analyze_unknown_domain(self):
        from math_anything.mcp_server import analyze_domain
        result = json.loads(analyze_domain("nonexistent"))
        self.assertIn("error", result)

    def test_compare_domains(self):
        from math_anything.mcp_server import compare_domains
        result = json.loads(compare_domains("dft", None, "md", None))
        self.assertIn("domain_a", result)
        self.assertIn("domain_b", result)
        self.assertIn("common_preserved", result)
        self.assertEqual(result["domain_a"], "dft")
        self.assertEqual(result["domain_b"], "md")


class TestSolveNumerical(unittest.TestCase):
    """Test the solve_numerical tool."""

    def test_eigenvalue_solver(self):
        from math_anything.mcp_server import solve_numerical
        result = json.loads(solve_numerical("eigenvalue", {"matrix": [[2, 1], [1, 2]]}))
        self.assertIn("eigenvalues", result)
        self.assertEqual(result["solver_type"], "eigenvalue")

    def test_unknown_solver(self):
        from math_anything.mcp_server import solve_numerical
        result = json.loads(solve_numerical("nonexistent", {}))
        self.assertIn("error", result)
        self.assertIn("available", result)


class TestAdaptersModule(unittest.TestCase):
    """Test the adapters module directly."""

    def test_translate_vasp(self):
        from math_anything.adapters import translate_params
        result = translate_params("vasp", {"ENCUT": 500, "EDIFF": 1e-6, "ISPIN": 2})
        self.assertEqual(result["domain"], "dft")
        self.assertEqual(result["domain_params"]["ecutwfc"], 500)

    def test_translate_qe(self):
        from math_anything.adapters import translate_params
        result = translate_params("qe", {"ecutwfc": 60, "conv_thr": 1e-8})
        self.assertEqual(result["domain"], "dft")
        self.assertEqual(result["domain_params"]["ecutwfc"], 60)
        self.assertEqual(result["domain_params"]["scf_tol"], 1e-8)

    def test_translate_lammps(self):
        from math_anything.adapters import translate_params
        result = translate_params("lammps", {"timestep": 0.001, "run": 10000})
        self.assertEqual(result["domain"], "md")
        self.assertEqual(result["domain_params"]["dt"], 0.001)
        self.assertEqual(result["domain_params"]["n_steps"], 10000)

    def test_translate_gromacs(self):
        from math_anything.adapters import translate_params
        result = translate_params("gromacs", {"dt": 0.002, "nsteps": 50000})
        self.assertEqual(result["domain"], "md")
        self.assertEqual(result["domain_params"]["n_steps"], 50000)

    def test_translate_abaqus(self):
        from math_anything.adapters import translate_params
        result = translate_params("abaqus", {"NLGEOM": 1, "ELEMENT_TYPE": "C3D8R"})
        self.assertEqual(result["domain"], "fem")
        self.assertTrue(result["domain_params"]["geometric_nonlinear"])

    def test_translate_openfoam(self):
        from math_anything.adapters import translate_params
        result = translate_params("openfoam", {"deltaT": 0.001, "endTime": 10.0})
        self.assertEqual(result["domain"], "cfd")
        self.assertEqual(result["domain_params"]["dt"], 0.001)

    def test_translate_unknown(self):
        from math_anything.adapters import translate_params
        result = translate_params("some_unknown_engine", {"param": 42})
        self.assertEqual(result["domain"], "unknown")
        self.assertEqual(result["domain_params"]["param"], 42)

    def test_list_supported_engines(self):
        from math_anything.adapters import list_supported_engines
        engines = list_supported_engines()
        self.assertIn("vasp", engines)
        self.assertIn("lammps", engines)
        self.assertIn("abaqus", engines)
        self.assertIn("openfoam", engines)

    def test_list_all_engines(self):
        from math_anything.adapters import list_all_engines
        engines = list_all_engines()
        self.assertGreater(len(engines), 6)


def test_mcp_analyze_loops_tool_exists():
    from math_anything.mcp_server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool_names = [t.name for t in tools]
    assert "analyze_loops" in tool_names


def test_mcp_analyze_loops_runs():
    from math_anything.mcp_server import analyze_loops

    result = analyze_loops("vasp", {})
    data = json.loads(result)
    assert data["engine"] == "vasp"
    assert "betti" in data
    assert "beta0" in data["betti"]
    assert "beta1" in data["betti"]
    assert "loops" in data
    assert isinstance(data["loops"], list)


def test_mcp_analyze_ml_model_tool_exists():
    from math_anything.mcp_server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool_names = [t.name for t in tools]
    assert "analyze_ml_model" in tool_names


def test_mcp_analyze_ml_model_runs():
    from math_anything.mcp_server import analyze_ml_model

    result = analyze_ml_model(
        input_dim=2,
        output_dim=1,
        architecture="mlp",
    )
    report = json.loads(result)
    assert report["domain"] == "supervised_learning"
    assert "morphism_chain" in report


def test_mcp_analyze_ml_model_reports_cross_domain_homotopy():
    from math_anything.mcp_server import analyze_ml_model

    result = analyze_ml_model(
        input_dim=2,
        output_dim=1,
        architecture="mlp",
    )
    report = json.loads(result)
    assert "cross_domain_homotopy" in report
    homotopy = report["cross_domain_homotopy"]
    assert isinstance(homotopy["equivalent"], bool)
    assert isinstance(homotopy["shared_invariants"], list)
    assert isinstance(homotopy["confidence"], float)
    assert 0.0 <= homotopy["confidence"] <= 1.0


if __name__ == "__main__":
    unittest.main()
