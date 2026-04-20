"""Tests for Math Anything MCP Server."""

import json
import sys
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from mcp_server import MCPServer, extract_mathematical_structure, list_supported_engines


class TestMCPTools(unittest.TestCase):
    """Test MCP tool functions."""

    def test_list_supported_engines(self):
        """Test listing supported engines."""
        engines = list_supported_engines()
        self.assertIn("vasp", engines)
        self.assertIn("lammps", engines)
        self.assertIn("abaqus", engines)
        self.assertEqual(len(engines), 7)

    def test_extract_vasp_structure(self):
        """Test extracting VASP mathematical structure."""
        params = {"encut": 520, "ediff": 1e-6, "sigma": 0.05}
        result = extract_mathematical_structure("vasp", params)

        self.assertIn("mathematical_structure", result)
        self.assertIn("variable_dependencies", result)
        self.assertIn("discretization_scheme", result)
        self.assertIn("solution_strategy", result)
        self.assertIn("approximations", result)
        self.assertIn("mathematical_decoding", result)

        ms = result["mathematical_structure"]
        self.assertEqual(ms["problem_type"], "nonlinear_eigenvalue")
        self.assertEqual(ms["canonical_form"], "H[n]ψ = εψ")

    def test_extract_lammps_structure(self):
        """Test extracting LAMMPS mathematical structure."""
        params = {"timestep": 0.001, "temperature": 300}
        result = extract_mathematical_structure("lammps", params)

        ms = result["mathematical_structure"]
        self.assertEqual(ms["problem_type"], "initial_value_ode")
        self.assertIn("d", ms["canonical_form"])
        self.assertIn("F", ms["canonical_form"])

    def test_extract_unknown_engine(self):
        """Test error on unknown engine."""
        with self.assertRaises(ValueError) as ctx:
            extract_mathematical_structure("unknown", {})
        self.assertIn("Unknown engine", str(ctx.exception))


class TestMCPServerProtocol(unittest.TestCase):
    """Test MCP server protocol handling."""

    def setUp(self):
        self.server = MCPServer()

    def test_initialize(self):
        """Test initialize response."""
        result = self.server.handle_initialize({})
        self.assertEqual(result["protocolVersion"], "2024-11-05")
        self.assertEqual(result["serverInfo"]["name"], "math-anything-mcp")

    def test_tools_list(self):
        """Test tools/list response."""
        result = self.server.handle_tools_list({})
        tools = result["tools"]
        self.assertEqual(len(tools), 4)

        names = [t["name"] for t in tools]
        self.assertIn("extract_mathematical_structure", names)
        self.assertIn("compare_calculations", names)
        self.assertIn("validate_constraints", names)
        self.assertIn("list_supported_engines", names)

    def test_tools_call_extract(self):
        """Test calling extract tool."""
        result = self.server.handle_tools_call({
            "name": "extract_mathematical_structure",
            "arguments": {
                "engine": "vasp",
                "parameters": {"encut": 520},
            },
        })

        self.assertNotIn("isError", result)
        content = json.loads(result["content"][0]["text"])
        self.assertIn("mathematical_structure", content)

    def test_tools_call_list_engines(self):
        """Test calling list engines tool."""
        result = self.server.handle_tools_call({
            "name": "list_supported_engines",
            "arguments": {},
        })

        content = json.loads(result["content"][0]["text"])
        self.assertIn("vasp", content)
        self.assertIn("lammps", content)

    def test_tools_call_unknown(self):
        """Test calling unknown tool."""
        result = self.server.handle_tools_call({
            "name": "unknown_tool",
            "arguments": {},
        })

        self.assertTrue(result["isError"])
        self.assertIn("Unknown tool", result["content"][0]["text"])


class TestMCPServerStdio(unittest.TestCase):
    """Test MCP server stdio transport."""

    def setUp(self):
        self.server = MCPServer()

    def test_stdio_initialize(self):
        """Test stdio initialize request/response."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }

        old_stdin = sys.stdin
        old_stdout = sys.stdout

        try:
            sys.stdin = StringIO(json.dumps(request) + "\n")
            sys.stdout = StringIO()

            # Run one iteration
            line = sys.stdin.readline()
            req = json.loads(line)
            result = self.server.handle_initialize(req.get("params", {}))
            response = {
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": result,
            }
            print(json.dumps(response), flush=True)

            sys.stdout.seek(0)
            output = sys.stdout.read().strip()
            response = json.loads(output)

            self.assertEqual(response["id"], 1)
            self.assertEqual(response["result"]["serverInfo"]["name"], "math-anything-mcp")

        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout


if __name__ == "__main__":
    unittest.main()
