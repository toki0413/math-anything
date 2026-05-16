"""Math tools package - register all tools into a ToolRegistry."""

from __future__ import annotations

from ..tool_registry import ToolRegistry


def register_all_tools(registry: ToolRegistry) -> None:
    from .compare_tool import CompareTool
    from .crossval_tool import CrossValidateTool
    from .dual_tool import DualPerspectiveTool
    from .emergence_tool import EmergenceTool
    from .extract_tool import ExtractTool
    from .geometry_tool import GeometryTool
    from .proposition_tool import PropositionTool
    from .validate_tool import ValidateTool
    from .verify_tool import VerifyTool

    for tool in [
        ExtractTool,
        ValidateTool,
        CompareTool,
        VerifyTool,
        PropositionTool,
        CrossValidateTool,
        DualPerspectiveTool,
        EmergenceTool,
        GeometryTool,
    ]:
        registry.register(tool)


from .dynamics import DynamicsAnalysisResult, DynamicsAnalyzer
from .langlands import LanglandsAnalyzer, LanglandsResult
from .ml_potential import MLPotentialAnalyzer, MLPotentialResult
from .sindy import SINDyDiscoverer, SINDyResult
from .spectral import SpectralAnalysisResult, SpectralAnalyzer
from .symmetry import SymmetryAnalysisResult, SymmetryAnalyzer
from .tda import TDAAnalyzer, TopologyResult
from .viz import InteractiveVisualizer

__all__ = [
    "register_all_tools",
    "SymmetryAnalyzer",
    "SymmetryAnalysisResult",
    "TDAAnalyzer",
    "TopologyResult",
    "SpectralAnalyzer",
    "SpectralAnalysisResult",
    "DynamicsAnalyzer",
    "DynamicsAnalysisResult",
    "InteractiveVisualizer",
    "MLPotentialAnalyzer",
    "MLPotentialResult",
    "LanglandsAnalyzer",
    "LanglandsResult",
    "SINDyDiscoverer",
    "SINDyResult",
]
