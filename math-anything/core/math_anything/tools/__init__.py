"""Math tools package - register all tools into a ToolRegistry."""

from __future__ import annotations

from ..tool_registry import ToolRegistry


def register_all_tools(registry: ToolRegistry) -> None:
    from .extract_tool import ExtractTool
    from .validate_tool import ValidateTool
    from .compare_tool import CompareTool
    from .verify_tool import VerifyTool
    from .proposition_tool import PropositionTool
    from .crossval_tool import CrossValidateTool
    from .dual_tool import DualPerspectiveTool
    from .emergence_tool import EmergenceTool
    from .geometry_tool import GeometryTool

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


from .symmetry import SymmetryAnalyzer, SymmetryAnalysisResult
from .tda import TDAAnalyzer, TopologyResult
from .spectral import SpectralAnalyzer, SpectralAnalysisResult
from .dynamics import DynamicsAnalyzer, DynamicsAnalysisResult
from .viz import InteractiveVisualizer
from .ml_potential import MLPotentialAnalyzer, MLPotentialResult
from .langlands import LanglandsAnalyzer, LanglandsResult
from .sindy import SINDyDiscoverer, SINDyResult

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
