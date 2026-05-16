"""ToolRegistry - centralized tool registration and discovery.

Manages all registered MathTool instances and provides:
- Tool lookup by name
- LLM function-calling format tool definitions
- Tool listing with JSON schemas
- Auto-discovery of analysis tools (SymmetryAnalyzer, TDAAnalyzer, etc.)
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from .tool_system import MathTool

logger = logging.getLogger(__name__)

_ANALYSIS_TOOL_SPECS = [
    {
        "module": "math_anything.tools.symmetry",
        "class_name": "SymmetryAnalyzer",
        "tool_name": "symmetry_analysis",
        "description": "Group theory analysis for crystal structures: space group detection, irreducible representations, character tables, selection rules",
    },
    {
        "module": "math_anything.tools.tda",
        "class_name": "TDAAnalyzer",
        "tool_name": "tda_analysis",
        "description": "Topological data analysis: persistent homology, Betti numbers, persistence entropy",
    },
    {
        "module": "math_anything.tools.spectral",
        "class_name": "SpectralAnalyzer",
        "tool_name": "spectral_analysis",
        "description": "Spectral analysis: density of states, band gap detection, topological invariants",
    },
    {
        "module": "math_anything.tools.dynamics",
        "class_name": "DynamicsAnalyzer",
        "tool_name": "dynamics_analysis",
        "description": "Dynamical systems analysis: Lyapunov exponents, chaos detection, DMD",
    },
    {
        "module": "math_anything.tools.sindy",
        "class_name": "SINDyDiscoverer",
        "tool_name": "sindy_discovery",
        "description": "Sparse identification of nonlinear dynamics: ODE/PDE equation discovery from time series",
    },
    {
        "module": "math_anything.tools.ml_potential",
        "class_name": "MLPotentialAnalyzer",
        "tool_name": "ml_potential_analysis",
        "description": "Mathematical structure analysis of ML interatomic potentials (DeepMD, MACE, NequIP)",
    },
    {
        "module": "math_anything.tools.langlands",
        "class_name": "LanglandsAnalyzer",
        "tool_name": "langlands_analysis",
        "description": "Langlands Program computations: Galois groups, L-functions, representation theory for materials",
    },
    {
        "module": "math_anything.tools.viz",
        "class_name": "InteractiveVisualizer",
        "tool_name": "interactive_viz",
        "description": "Interactive scientific visualization: DOS, persistence diagrams, phase portraits, 3D manifolds",
    },
]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, MathTool] = {}
        self._analysis_tools: dict[str, Any] = {}

    def register(self, tool: MathTool) -> None:
        self._tools[tool.name] = tool

    def register_analysis_tool(self, name: str, instance: Any, description: str = "") -> None:
        self._analysis_tools[name] = {
            "instance": instance,
            "description": description,
            "class_name": type(instance).__name__,
            "module": type(instance).__module__,
        }

    def get(self, name: str) -> MathTool | None:
        return self._tools.get(name)

    def get_analysis_tool(self, name: str) -> Any | None:
        entry = self._analysis_tools.get(name)
        return entry["instance"] if entry else None

    def list_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def list_analysis_tool_names(self) -> list[str]:
        return list(self._analysis_tools.keys())

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.get_json_schema(),
            }
            for t in self._tools.values()
        ]

    def list_analysis_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": info["description"],
                "class_name": info["class_name"],
                "module": info["module"],
            }
            for name, info in self._analysis_tools.items()
        ]

    def get_tool_definitions_for_llm(self) -> list[dict[str, Any]]:
        return [t.get_tool_definition_for_llm() for t in self._tools.values()]

    def all_tools(self) -> list[MathTool]:
        return list(self._tools.values())

    def auto_discover_analysis_tools(self) -> list[str]:
        """Auto-discover and register analysis tools from known modules.

        Returns list of successfully registered tool names.
        """
        registered = []
        for spec in _ANALYSIS_TOOL_SPECS:
            try:
                mod = importlib.import_module(spec["module"])
                cls = getattr(mod, spec["class_name"])
                instance = cls()
                self.register_analysis_tool(
                    spec["tool_name"],
                    instance,
                    spec["description"],
                )
                registered.append(spec["tool_name"])
            except Exception as e:
                logger.debug(
                    "Skipping analysis tool %s: %s", spec["tool_name"], e
                )
        return registered


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    from .tools import register_all_tools
    register_all_tools(registry)
    registry.auto_discover_analysis_tools()
    return registry
