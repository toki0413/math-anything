"""Tests for tiered analysis system."""

import os
import tempfile
from pathlib import Path

import pytest
from math_anything.tiered import (
    AnalysisTier,
    ComplexityScore,
    FileAnalysis,
    ResourceRequirements,
    TieredAnalysisResult,
    TieredAnalyzer,
    TierRecommendation,
    TierRecommender,
)
from math_anything.tiered.tier_recommender import analyze_file_properties


class TestTieredSchema:
    """Test tiered schema data structures."""

    def test_analysis_tier_enum(self):
        assert AnalysisTier.BASIC.value == 1
        assert AnalysisTier.ENHANCED.value == 2
        assert AnalysisTier.PROFESSIONAL.value == 3
        assert AnalysisTier.ADVANCED.value == 4
        assert AnalysisTier.COMPLETE.value == 5

    def test_complexity_score(self):
        score = ComplexityScore(
            total=50.0,
            system_size=20.0,
            time_scale=15.0,
            constraints=10.0,
            data_availability=5.0,
        )
        assert score.total == 50.0
        assert "total=50.0" in str(score)

    def test_resource_requirements(self):
        resources = ResourceRequirements(
            cpu_time_seconds=10.0,
            memory_gb=2.0,
            gpu_required=True,
            gpu_memory_gb=4.0,
        )
        assert resources.cpu_time_seconds == 10.0
        assert resources.gpu_required is True
        assert "GPU=4.0GB" in str(resources)

    def test_file_analysis(self):
        analysis = FileAnalysis(
            file_path="test.lmp",
            file_type="lammps",
            engine="lammps",
            num_atoms=1000,
            simulation_time=1e7,
            has_constraints=True,
        )
        assert analysis.num_atoms == 1000
        assert analysis.has_constraints is True
        d = analysis.to_dict()
        assert d["num_atoms"] == 1000


class TestTierRecommender:
    """Test tier recommendation system."""

    def test_small_system_recommendation(self):
        analysis = FileAnalysis(
            file_path="test.lmp",
            num_atoms=50,
            simulation_time=1e5,
            has_constraints=False,
        )
        recommender = TierRecommender()
        recommendation = recommender.recommend(analysis)

        assert recommendation.recommended_tier in [
            AnalysisTier.BASIC,
            AnalysisTier.ENHANCED,
        ]
        assert len(recommendation.reasons) > 0
        assert recommendation.complexity_score.total < 30

    def test_large_system_recommendation(self):
        analysis = FileAnalysis(
            file_path="test.lmp",
            num_atoms=15000,
            simulation_time=1e9,
            has_constraints=True,
            has_training_data=True,
            training_data_size=5000,
        )
        recommender = TierRecommender()
        recommendation = recommender.recommend(analysis)

        assert recommendation.recommended_tier.value >= 4
        assert recommendation.complexity_score.total > 60

    def test_complexity_calculation(self):
        recommender = TierRecommender()

        analysis1 = FileAnalysis(num_atoms=100, simulation_time=1e6)
        score1 = recommender._calculate_complexity(analysis1)
        assert score1.total < 30

        analysis2 = FileAnalysis(
            num_atoms=5000,
            simulation_time=1e8,
            has_constraints=True,
        )
        score2 = recommender._calculate_complexity(analysis2)
        assert score2.total > score1.total

    def test_resource_estimation(self):
        recommender = TierRecommender()

        resources1 = recommender._estimate_resources(
            AnalysisTier.BASIC, FileAnalysis(num_atoms=100)
        )
        assert resources1.cpu_time_seconds < 5
        assert resources1.gpu_required is False

        resources5 = recommender._estimate_resources(
            AnalysisTier.COMPLETE, FileAnalysis(num_atoms=10000)
        )
        assert resources5.gpu_required is True
        assert "e3nn" in resources5.additional_packages


class TestTieredAnalyzer:
    """Test tiered analyzer."""

    def test_analyzer_initialization(self):
        analyzer = TieredAnalyzer()
        assert analyzer.recommender is not None

    def test_basic_analysis(self):
        analyzer = TieredAnalyzer()

        result = analyzer.analyze(
            "nonexistent.lmp",
            tier=AnalysisTier.BASIC,
            auto_tier=False,
        )

        assert result.tier == AnalysisTier.BASIC
        assert result.file_analysis is not None

    def test_tier_constraint(self):
        analyzer = TieredAnalyzer()

        result = analyzer.analyze(
            "test.lmp",
            tier=1,
            min_tier=2,
            max_tier=4,
        )

        assert result.tier.value >= 2
        assert result.tier.value <= 4

    def test_result_structure(self):
        analyzer = TieredAnalyzer()
        result = analyzer.analyze("test.lmp", tier=2)

        assert isinstance(result, TieredAnalysisResult)
        assert result.tier == AnalysisTier.ENHANCED
        assert isinstance(result.file_analysis, FileAnalysis)
        assert isinstance(result.warnings, list)
        assert isinstance(result.errors, list)


class TestFileAnalysis:
    """Test file property analysis."""

    def test_lammps_file_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lmp_path = os.path.join(tmpdir, "test_tier.lmp")
            with open(lmp_path, "w") as f:
                f.write("""
# LAMMPS input
units real
atom_style full
read_data data.file

# 1000 atoms
fix 1 all nvt temp 300 300 100
run 100000
""")

            analysis = analyze_file_properties(lmp_path)
            assert analysis.engine == "lammps"
            assert analysis.file_type == "lammps"

    def test_abaqus_file_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inp_path = os.path.join(tmpdir, "test_tier.inp")
            with open(inp_path, "w") as f:
                f.write("""
*Heading
*Preprint, echo=0, model=0, history=0, contact=0
*Part
*End Part
""")

            analysis = analyze_file_properties(inp_path)
            assert analysis.engine == "abaqus"


class TestIntegration:
    """Integration tests for tiered system."""

    def test_full_workflow(self):
        analyzer = TieredAnalyzer()

        with tempfile.TemporaryDirectory() as tmpdir:
            lmp_path = os.path.join(tmpdir, "integration_test.lmp")
            with open(lmp_path, "w") as f:
                f.write("""
units real
atom_style full
dimension 3
boundary p p p
timestep 0.5
run 50000
""")

            recommendation = analyzer.get_recommendation(lmp_path)
            assert isinstance(recommendation, TierRecommendation)

            result = analyzer.analyze(lmp_path, auto_tier=True)
            assert isinstance(result, TieredAnalysisResult)

            result_dict = result.to_dict()
            assert "tier" in result_dict
            assert "file_analysis" in result_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
