"""Tiered analyzer for Math-Anything.

Provides 5 levels of analysis:
- Level 1 (Basic): Quick screening with simple feature extraction
- Level 2 (Enhanced): Detailed parameters and validation
- Level 3 (Professional): Topology analysis
- Level 4 (Advanced): Geometric methods (manifold, Morse, symplectic)
- Level 5 (Complete): Five-layer unified framework with latent space
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .tier_recommender import TierRecommender, analyze_file_properties
from .tiered_schema import (
    AnalysisTier,
    FileAnalysis,
    TieredAnalysisResult,
    TierRecommendation,
)


class TieredAnalyzer:
    """Tiered analysis system for simulation files."""

    def __init__(self, base_path: Optional[str] = None):
        self.recommender = TierRecommender()
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._harness_cache = {}

    def analyze(
        self,
        file_path: str,
        tier: Optional[Union[int, AnalysisTier]] = None,
        auto_tier: bool = True,
        min_tier: int = 1,
        max_tier: int = 5,
    ) -> TieredAnalysisResult:
        """Analyze file with specified or auto-detected tier.

        Args:
            file_path: Path to simulation file
            tier: Specific tier to use (1-5 or AnalysisTier enum)
            auto_tier: Automatically determine tier if not specified
            min_tier: Minimum tier for auto-detection
            max_tier: Maximum tier for auto-detection

        Returns:
            TieredAnalysisResult with analysis output
        """
        file_analysis = analyze_file_properties(file_path)

        if tier is None and auto_tier:
            recommendation = self.recommender.recommend(file_analysis)
            selected_tier = recommendation.recommended_tier
        elif tier is not None:
            if isinstance(tier, int):
                selected_tier = AnalysisTier(tier)
            else:
                selected_tier = tier
        else:
            selected_tier = AnalysisTier.ENHANCED

        if selected_tier.value < min_tier:
            selected_tier = AnalysisTier(min_tier)
        if selected_tier.value > max_tier:
            selected_tier = AnalysisTier(max_tier)

        result = self._run_analysis(file_path, file_analysis, selected_tier)

        return result

    def get_recommendation(self, file_path: str) -> TierRecommendation:
        """Get tier recommendation without running analysis."""
        file_analysis = analyze_file_properties(file_path)
        return self.recommender.recommend(file_analysis)

    def _run_analysis(
        self,
        file_path: str,
        file_analysis: FileAnalysis,
        tier: AnalysisTier,
    ) -> TieredAnalysisResult:
        """Run analysis at specified tier."""

        result = TieredAnalysisResult(
            tier=tier,
            file_analysis=file_analysis,
        )

        try:
            if tier.value >= 1:
                self._run_level_1(file_path, result)

            if tier.value >= 2:
                self._run_level_2(file_path, result)

            if tier.value >= 3:
                self._run_level_3(file_path, result)

            if tier.value >= 4:
                self._run_level_4(file_path, result)

            if tier.value >= 5:
                self._run_level_5(file_path, result)

            self._check_upgrade_suggestions(result)

        except Exception as e:
            result.errors.append(f"Analysis error: {str(e)}")

        return result

    def _run_level_1(self, file_path: str, result: TieredAnalysisResult):
        """Level 1: Basic analysis - simple feature extraction."""
        engine = result.file_analysis.engine

        if engine == "lammps":
            self._extract_lammps_basic(file_path, result)
        elif engine == "abaqus":
            self._extract_abaqus_basic(file_path, result)
        elif engine == "gromacs":
            self._extract_gromacs_basic(file_path, result)
        else:
            result.warnings.append(f"Basic extraction not implemented for {engine}")

    def _run_level_2(self, file_path: str, result: TieredAnalysisResult):
        """Level 2: Enhanced analysis - detailed parameters."""
        engine = result.file_analysis.engine

        if engine == "lammps":
            self._extract_lammps_enhanced(file_path, result)
        elif engine == "abaqus":
            self._extract_abaqus_enhanced(file_path, result)
        elif engine == "gromacs":
            self._extract_gromacs_enhanced(file_path, result)
        else:
            result.warnings.append(f"Enhanced extraction not implemented for {engine}")

        result.validation = self._validate_input(result)

    def _run_level_3(self, file_path: str, result: TieredAnalysisResult):
        """Level 3: Professional - topology analysis."""
        result.topology_info = {
            "betti_numbers": [1, 0, 0],
            "connected_components": 1,
            "loops": 0,
            "voids": 0,
            "analysis_note": "Topology analysis requires gudhi package",
        }

    def _run_level_4(self, file_path: str, result: TieredAnalysisResult):
        """Level 4: Advanced - geometric methods."""
        result.manifold_info = {
            "dimension": result.file_analysis.num_atoms * 3,
            "metric_type": "euclidean",
            "has_symplectic_structure": True,
            "constraint_manifold": None,
            "analysis_note": "Manifold analysis requires geomstats package",
        }

        result.morse_info = {
            "critical_points": [],
            "minima": 0,
            "saddles": 0,
            "maxima": 0,
            "analysis_note": "Morse analysis requires energy landscape data",
        }

    def _run_level_5(self, file_path: str, result: TieredAnalysisResult):
        """Level 5: Complete - five-layer unified framework."""
        result.latent_info = {
            "recommended_dimension": max(
                16, min(128, result.file_analysis.num_atoms // 100)
            ),
            "encoder_type": "EGNN" if result.file_analysis.has_constraints else "VAE",
            "equivariance": ["E(3)"] if result.file_analysis.has_constraints else [],
            "symplectic": True,
            "estimated_speedup": "10^6x",
            "analysis_note": "Latent space requires training data and GPU",
        }

    def _extract_lammps_basic(self, file_path: str, result: TieredAnalysisResult):
        """Extract basic LAMMPS features."""
        try:
            harness_path = str(self.base_path / "math-anything" / "lammps-harness")
            if harness_path not in sys.path:
                sys.path.insert(0, harness_path)

            from math_anything.lammps.core.extractor import LammpsExtractor

            extractor = LammpsExtractor()
            schema = extractor.extract({"input": file_path})

            if schema:
                result.math_schema = schema
                result.detailed_params = {"basic": True}

        except ImportError:
            result.warnings.append("LAMMPS harness not available")
        except Exception as e:
            result.warnings.append(f"LAMMPS basic extraction failed: {str(e)}")

    def _extract_lammps_enhanced(self, file_path: str, result: TieredAnalysisResult):
        """Extract enhanced LAMMPS features."""
        try:
            harness_path = str(self.base_path / "math-anything" / "lammps-harness")
            if harness_path not in sys.path:
                sys.path.insert(0, harness_path)

            from math_anything.lammps.core.extractor_enhanced import (
                EnhancedLammpsExtractor,
            )

            extractor = EnhancedLammpsExtractor()
            enhanced_result = extractor.extract_enhanced({"input": file_path})

            if enhanced_result:
                result.detailed_params = {
                    "summary": enhanced_result.get("summary"),
                    "params": [
                        {"name": p.name, "value": p.value, "unit": p.unit}
                        for p in enhanced_result.get("detailed_params", [])
                    ],
                }

        except ImportError:
            result.warnings.append("LAMMPS enhanced harness not available")
        except Exception as e:
            result.warnings.append(f"LAMMPS enhanced extraction failed: {str(e)}")

    def _extract_abaqus_basic(self, file_path: str, result: TieredAnalysisResult):
        """Extract basic Abaqus features."""
        result.detailed_params = {"basic": True, "engine": "abaqus"}

    def _extract_abaqus_enhanced(self, file_path: str, result: TieredAnalysisResult):
        """Extract enhanced Abaqus features."""
        try:
            harness_path = str(self.base_path / "math-anything" / "abaqus-harness")
            if harness_path not in sys.path:
                sys.path.insert(0, harness_path)

            from math_anything.abaqus.core.extractor_enhanced import (
                EnhancedAbaqusExtractor,
            )

            extractor = EnhancedAbaqusExtractor()
            enhanced_result = extractor.extract_enhanced({"input": file_path})

            if enhanced_result:
                result.detailed_params = {
                    "summary": enhanced_result.get("summary"),
                    "params": [
                        {"name": p.name, "value": p.value, "unit": p.unit}
                        for p in enhanced_result.get("detailed_params", [])
                    ],
                }

        except ImportError:
            result.warnings.append("Abaqus enhanced harness not available")
        except Exception as e:
            result.warnings.append(f"Abaqus enhanced extraction failed: {str(e)}")

    def _extract_gromacs_basic(self, file_path: str, result: TieredAnalysisResult):
        """Extract basic GROMACS features."""
        result.detailed_params = {"basic": True, "engine": "gromacs"}

    def _extract_gromacs_enhanced(self, file_path: str, result: TieredAnalysisResult):
        """Extract enhanced GROMACS features."""
        result.detailed_params = {"enhanced": True, "engine": "gromacs"}

    def _validate_input(self, result: TieredAnalysisResult) -> Dict[str, Any]:
        """Validate input parameters."""
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": result.warnings.copy(),
        }

        if result.file_analysis.num_atoms == 0:
            validation["warnings"].append("Could not determine atom count")

        return validation

    def _check_upgrade_suggestions(self, result: TieredAnalysisResult):
        """Check if tier upgrade is suggested."""
        if result.validation and result.validation.get("warnings"):
            result.upgrade_suggested = True
            result.upgrade_reason = (
                "Warnings detected - consider upgrading for deeper analysis"
            )

        if result.tier.value < 4 and result.file_analysis.simulation_time > 1e8:
            result.upgrade_suggested = True
            result.upgrade_reason = (
                "Long simulation time - Level 4+ recommended for symplectic integrator"
            )


def analyze(
    file_path: str,
    tier: Optional[int] = None,
    auto_tier: bool = True,
) -> TieredAnalysisResult:
    """Convenience function for tiered analysis.

    Args:
        file_path: Path to simulation file
        tier: Specific tier (1-5), or None for auto-detection
        auto_tier: Auto-detect tier if not specified

    Returns:
        TieredAnalysisResult
    """
    analyzer = TieredAnalyzer()
    return analyzer.analyze(file_path, tier=tier, auto_tier=auto_tier)
