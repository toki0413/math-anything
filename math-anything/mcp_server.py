#!/usr/bin/env python3
"""Math Anything MCP Server.

A Model Context Protocol server that exposes Math Anything's mathematical
structure extraction capabilities to LLMs through standardized tool interfaces.

Usage:
    python mcp_server.py                    # Run with stdio transport
    python mcp_server.py --transport sse    # Run with SSE transport
    python mcp_server.py --port 8080        # Custom port for SSE

Design Principles:
    - Zero intrusion: Only reads files, never modifies
    - Zero judgment: Reports mathematical structures, not opinions
    - Mathematical precision: Expresses structures in canonical forms
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

from math_anything.schemas import (
    AbaqusMathematicalPrecisionExtractor,
    AnsysMathematicalPrecisionExtractor,
    ComsolMathematicalPrecisionExtractor,
    EnhancedMathSchema,
    GromacsMathematicalPrecisionExtractor,
    LammpsMathematicalPrecisionExtractor,
    MultiwfnMathematicalPrecisionExtractor,
    VaspMathematicalPrecisionExtractor,
)
from math_anything.utils.math_diff import MathDiffer

# Registry of available extractors
EXTRACTORS = {
    "vasp": VaspMathematicalPrecisionExtractor,
    "lammps": LammpsMathematicalPrecisionExtractor,
    "abaqus": AbaqusMathematicalPrecisionExtractor,
    "ansys": AnsysMathematicalPrecisionExtractor,
    "comsol": ComsolMathematicalPrecisionExtractor,
    "gromacs": GromacsMathematicalPrecisionExtractor,
    "multiwfn": MultiwfnMathematicalPrecisionExtractor,
}


def extract_mathematical_structure(
    engine: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract mathematical structure from simulation parameters.

    Args:
        engine: Simulation engine name (vasp, lammps, abaqus, ansys,
                comsol, gromacs, multiwfn)
        parameters: Engine-specific parameters dict

    Returns:
        EnhancedMathSchema as dict with mathematical structure,
        variable dependencies, discretization, solution strategy,
        approximations, and mathematical decoding.
    """
    engine = engine.lower()
    if engine not in EXTRACTORS:
        available = ", ".join(EXTRACTORS.keys())
        raise ValueError(f"Unknown engine '{engine}'. Available: {available}")

    extractor_class = EXTRACTORS[engine]
    extractor = extractor_class()
    schema = extractor.extract(parameters)
    return schema.to_dict()


def compare_calculations(
    schema_a: Dict[str, Any],
    schema_b: Dict[str, Any],
    critical_only: bool = False,
) -> Dict[str, Any]:
    """Compare two mathematical schemas and report differences.

    Args:
        schema_a: First EnhancedMathSchema as dict
        schema_b: Second EnhancedMathSchema as dict
        critical_only: If True, only report critical changes

    Returns:
        Diff report with categorized changes (critical, warning, info)
    """
    differ = MathDiffer()
    report = differ.compare_dicts(schema_a, schema_b)

    result = {
        "has_changes": report.has_changes,
        "has_critical_changes": report.has_critical_changes,
        "summary": report.summary,
    }

    if critical_only:
        result["changes"] = [
            c.to_dict() if hasattr(c, "to_dict") else c for c in report.critical_changes
        ]
    else:
        result["changes"] = [
            c.to_dict() if hasattr(c, "to_dict") else c for c in report.all_changes
        ]

    return result


def validate_constraints(
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate symbolic constraints in a mathematical schema.

    Args:
        schema: EnhancedMathSchema as dict

    Returns:
        Validation report with satisfied and violated constraints.
    """
    constraints = schema.get("symbolic_constraints", [])
    results = []

    for constraint in constraints:
        expr = constraint.get("expression", "")
        desc = constraint.get("description", "")
        results.append(
            {
                "expression": expr,
                "description": desc,
                "status": "unknown",
                "note": "Constraint validation requires runtime parameter values",
            }
        )

    return {
        "total_constraints": len(results),
        "constraints": results,
        "note": "Zero-judgment validation: reports what constraints exist, "
        "not whether they are satisfied",
    }


def list_supported_engines() -> List[str]:
    """List all supported simulation engines."""
    return list(EXTRACTORS.keys())


# MCP Protocol Implementation


class MCPServer:
    """Minimal MCP server implementation over stdio."""

    def __init__(self):
        self.tools = {
            "extract_mathematical_structure": {
                "description": "Extract mathematical structure from simulation parameters",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "engine": {
                            "type": "string",
                            "enum": list(EXTRACTORS.keys()),
                            "description": "Simulation engine name",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Engine-specific parameters",
                        },
                    },
                    "required": ["engine", "parameters"],
                },
                "handler": extract_mathematical_structure,
            },
            "compare_calculations": {
                "description": "Compare two mathematical schemas",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "schema_a": {"type": "object"},
                        "schema_b": {"type": "object"},
                        "critical_only": {
                            "type": "boolean",
                            "default": False,
                        },
                    },
                    "required": ["schema_a", "schema_b"],
                },
                "handler": compare_calculations,
            },
            "validate_constraints": {
                "description": "Validate symbolic constraints in a schema",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "schema": {"type": "object"},
                    },
                    "required": ["schema"],
                },
                "handler": validate_constraints,
            },
            "list_supported_engines": {
                "description": "List all supported simulation engines",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
                "handler": list_supported_engines,
            },
        }

    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": "math-anything-mcp",
                "version": "0.1.0",
            },
        }

    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tools/list request."""
        tools = []
        for name, tool in self.tools.items():
            tools.append(
                {
                    "name": name,
                    "description": tool["description"],
                    "inputSchema": tool["input_schema"],
                }
            )
        return {"tools": tools}

    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tools/call request."""
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        if name not in self.tools:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Unknown tool: {name}",
                    }
                ],
                "isError": True,
            }

        try:
            handler = self.tools[name]["handler"]
            result = handler(**arguments)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2, ensure_ascii=False),
                    }
                ],
            }
        except Exception as e:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {str(e)}\n{traceback.format_exc()}",
                    }
                ],
                "isError": True,
            }

    def run_stdio(self):
        """Run server over stdio (MCP standard transport)."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                request = json.loads(line)
                method = request.get("method", "")
                params = request.get("params", {})
                req_id = request.get("id")

                if method == "initialize":
                    result = self.handle_initialize(params)
                elif method == "tools/list":
                    result = self.handle_tools_list(params)
                elif method == "tools/call":
                    result = self.handle_tools_call(params)
                else:
                    result = {"error": f"Unknown method: {method}"}

                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": result,
                }
                print(json.dumps(response), flush=True)

            except json.JSONDecodeError:
                continue
            except Exception as e:
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": str(e)},
                }
                print(json.dumps(response), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Math Anything MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE transport (default: 8080)",
    )
    args = parser.parse_args()

    server = MCPServer()

    if args.transport == "stdio":
        server.run_stdio()
    else:
        print("SSE transport not yet implemented. Use --transport stdio")
        sys.exit(1)


if __name__ == "__main__":
    main()
