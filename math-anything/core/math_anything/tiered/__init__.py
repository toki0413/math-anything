"""Tiered analysis system for Math-Anything.

Provides 5 levels of analysis complexity:
- Level 1: Basic - Quick screening
- Level 2: Enhanced - Detailed parameters
- Level 3: Professional - Topology analysis + Symbolic Regression
- Level 4: Advanced - Geometric methods + Symbolic-Geometric integration
- Level 5: Complete - Five-layer unified framework with Symbolic-Geometric synthesis

Integration with Symbolic Regression:
- Level 3+: Uses EnhancedPSRN to discover mathematical relationships
- Level 4+: Combines symbolic expressions with geometric analysis
- Level 5: Unified symbolic-geometric representation
"""

# Symbolic Regression Integration
from .symbolic_regression_integration import (
    IntegratedTieredAnalyzer,
    TieredSymbolicRegressionAnalyzer,
    tiered_symbolic_regression_analysis,
)
from .tier_recommender import TierRecommendation, TierRecommender
from .tiered_analyzer import AnalysisTier, TieredAnalyzer
from .tiered_schema import (
    ComplexityScore,
    FileAnalysis,
    ResourceRequirements,
    TieredAnalysisResult,
)

__all__ = [
    # Core Tiered Analysis
    "TieredAnalyzer",
    "AnalysisTier",
    "TierRecommender",
    "TierRecommendation",
    "TieredAnalysisResult",
    "FileAnalysis",
    "ComplexityScore",
    "ResourceRequirements",
    # Symbolic Regression Integration
    "TieredSymbolicRegressionAnalyzer",
    "IntegratedTieredAnalyzer",
    "tiered_symbolic_regression_analysis",
]
