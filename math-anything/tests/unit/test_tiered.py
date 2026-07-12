"""Tests for tiered analysis module — schema, recommender, and analyzer."""

import pytest

from math_anything.tiered.tier_recommender import TierRecommender, analyze_file_properties
from math_anything.tiered.tiered_schema import (
    AnalysisTier,
    ComplexityScore,
    FileAnalysis,
    ResourceRequirements,
    TieredAnalysisResult,
    TierRecommendation,
)

# ── AnalysisTier ──


class TestAnalysisTier:
    def test_values(self):
        assert AnalysisTier.BASIC.value == 1
        assert AnalysisTier.ENHANCED.value == 2
        assert AnalysisTier.PROFESSIONAL.value == 3
        assert AnalysisTier.ADVANCED.value == 4
        assert AnalysisTier.COMPLETE.value == 5

    def test_from_int(self):
        assert AnalysisTier(1) == AnalysisTier.BASIC
        assert AnalysisTier(5) == AnalysisTier.COMPLETE


# ── ComplexityScore ──


class TestComplexityScore:
    def test_defaults(self):
        cs = ComplexityScore()
        assert cs.total == 0.0
        assert cs.system_size == 0.0

    def test_str(self):
        cs = ComplexityScore(total=50.0, system_size=30.0, time_scale=20.0)
        s = str(cs)
        assert "50.0" in s
        assert "30.0" in s


# ── ResourceRequirements ──


class TestResourceRequirements:
    def test_defaults(self):
        r = ResourceRequirements()
        assert r.cpu_time_seconds == 1.0
        assert r.gpu_required is False

    def test_str_no_gpu(self):
        r = ResourceRequirements(cpu_time_seconds=10.0, memory_gb=4.0)
        s = str(r)
        assert "10.0s" in s
        assert "4.0GB" in s

    def test_str_with_gpu(self):
        r = ResourceRequirements(gpu_required=True, gpu_memory_gb=8.0)
        s = str(r)
        assert "GPU=8.0GB" in s


# ── FileAnalysis ──


class TestFileAnalysis:
    def test_defaults(self):
        fa = FileAnalysis()
        assert fa.file_path == ""
        assert fa.num_atoms == 0

    def test_to_dict(self):
        fa = FileAnalysis(file_path="test.in", engine="vasp", num_atoms=100)
        d = fa.to_dict()
        assert d["file_path"] == "test.in"
        assert d["num_atoms"] == 100


# ── TierRecommendation ──


class TestTierRecommendation:
    def test_to_dict(self):
        rec = TierRecommendation(
            recommended_tier=AnalysisTier.ENHANCED,
            suitable_tiers=[AnalysisTier.BASIC, AnalysisTier.ENHANCED],
            reasons=["small system"],
            complexity_score=ComplexityScore(total=20.0),
            estimated_time="1s",
            required_resources=ResourceRequirements(),
        )
        d = rec.to_dict()
        assert d["recommended_tier"] == 2
        assert d["complexity_score"] == 20.0
        assert len(d["suitable_tiers"]) == 2


# ── TieredAnalysisResult ──


class TestTieredAnalysisResult:
    def test_to_dict_minimal(self):
        r = TieredAnalysisResult(
            tier=AnalysisTier.BASIC,
            file_analysis=FileAnalysis(),
        )
        d = r.to_dict()
        assert d["tier"] == 1
        assert "warnings" in d

    def test_to_dict_with_details(self):
        r = TieredAnalysisResult(
            tier=AnalysisTier.PROFESSIONAL,
            file_analysis=FileAnalysis(file_path="test.in"),
            detailed_params={"encut": 500},
            topology_info={"betti": [1, 0, 0]},
            warnings=["check this"],
        )
        d = r.to_dict()
        assert d["detailed_params"]["encut"] == 500
        assert d["topology_info"]["betti"] == [1, 0, 0]
        assert "check this" in d["warnings"]


# ── TierRecommender ──


class TestTierRecommender:
    def test_small_system_basic(self):
        rec = TierRecommender()
        fa = FileAnalysis(num_atoms=10, simulation_time=1000)
        result = rec.recommend(fa)
        assert result.recommended_tier in (AnalysisTier.BASIC, AnalysisTier.ENHANCED)

    def test_large_system_higher_tier(self):
        rec = TierRecommender()
        fa = FileAnalysis(num_atoms=50000, simulation_time=1e9, has_constraints=True)
        result = rec.recommend(fa)
        assert result.recommended_tier.value >= AnalysisTier.PROFESSIONAL.value

    def test_constraints_increase_complexity(self):
        rec = TierRecommender()
        fa_no = FileAnalysis(num_atoms=100, has_constraints=False)
        fa_yes = FileAnalysis(num_atoms=100, has_constraints=True)
        r_no = rec.recommend(fa_no)
        r_yes = rec.recommend(fa_yes)
        assert r_yes.complexity_score.total >= r_no.complexity_score.total

    def test_training_data_increases_complexity(self):
        rec = TierRecommender()
        fa = FileAnalysis(num_atoms=100, has_training_data=True, training_data_size=1000)
        result = rec.recommend(fa)
        assert result.complexity_score.data_availability > 0

    def test_reasons_not_empty(self):
        rec = TierRecommender()
        fa = FileAnalysis(num_atoms=100)
        result = rec.recommend(fa)
        assert len(result.reasons) > 0

    def test_suitable_tiers_not_empty(self):
        rec = TierRecommender()
        fa = FileAnalysis(num_atoms=100)
        result = rec.recommend(fa)
        assert len(result.suitable_tiers) > 0

    def test_resources_estimated(self):
        rec = TierRecommender()
        fa = FileAnalysis(num_atoms=100)
        result = rec.recommend(fa)
        assert result.required_resources.cpu_time_seconds > 0

    def test_upgrade_triggers(self):
        rec = TierRecommender()
        fa = FileAnalysis(num_atoms=100)
        result = rec.recommend(fa)
        assert isinstance(result.upgrade_triggers, list)


# ── analyze_file_properties ──


class TestAnalyzeFileProperties:
    def test_nonexistent_file(self):
        fa = analyze_file_properties("/nonexistent/path/INCAR")
        assert isinstance(fa, FileAnalysis)

    def test_returns_file_analysis(self):
        fa = analyze_file_properties("some_file.in")
        assert hasattr(fa, "file_path")
        assert hasattr(fa, "engine")
