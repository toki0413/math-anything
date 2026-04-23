"""Tiered analysis system for Math-Anything.

Provides 5 levels of analysis complexity:
- Level 1: Basic - Quick screening
- Level 2: Enhanced - Detailed parameters
- Level 3: Professional - Topology analysis
- Level 4: Advanced - Geometric methods
- Level 5: Complete - Five-layer unified framework
"""

from .tier_recommender import TierRecommendation, TierRecommender
from .tiered_analyzer import AnalysisTier, TieredAnalyzer
from .tiered_schema import (
    ComplexityScore,
    FileAnalysis,
    ResourceRequirements,
    TieredAnalysisResult,
)

__all__ = [
    "TieredAnalyzer",
    "AnalysisTier",
    "TierRecommender",
    "TierRecommendation",
    "TieredAnalysisResult",
    "FileAnalysis",
    "ComplexityScore",
    "ResourceRequirements",
]
