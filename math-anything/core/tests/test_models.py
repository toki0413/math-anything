"""Tests for Pydantic models."""

import pytest
from math_anything.models import (
    AnalysisTierEnum,
    ComplexityScore,
    Constraint,
    EngineType,
    ExtractionRequest,
    ExtractionResultModel,
    FileExtractionRequest,
    MathematicalStructure,
    ProblemType,
    ResourceRequirements,
    TieredAnalysisRequest,
    TierRecommendation,
)
from pydantic import ValidationError


class TestExtractionRequest:
    """Test extraction request model."""

    def test_valid_request(self):
        """Test valid extraction request."""
        req = ExtractionRequest(engine="vasp", params={"ENCUT": 520})
        assert req.engine == EngineType.VASP
        assert req.params == {"ENCUT": 520}
        assert req.validate_params is True

    def test_default_validate_params(self):
        """Test default validation flag."""
        req = ExtractionRequest(engine="lammps")
        assert req.validate_params is True

    def test_empty_params(self):
        """Test extraction with empty params."""
        req = ExtractionRequest(engine="vasp")
        assert req.params == {}

    def test_engine_case_normalization(self):
        """Test engine name case normalization."""
        req = ExtractionRequest(engine="VASP")
        assert req.engine == EngineType.VASP
        req2 = ExtractionRequest(engine="Lammps")
        assert req2.engine == EngineType.LAMMPS

    def test_invalid_engine(self):
        """Test invalid engine validation."""
        with pytest.raises(ValidationError):
            ExtractionRequest(engine="invalid_engine")


class TestFileExtractionRequest:
    """Test file extraction request model."""

    def test_valid_single_file(self):
        """Test valid single file request."""
        req = FileExtractionRequest(engine="vasp", filepath="INCAR")
        assert req.engine == EngineType.VASP
        assert req.filepath == "INCAR"

    def test_valid_multiple_files(self):
        """Test valid multiple files request."""
        files = {"INCAR": "path/to/INCAR", "POSCAR": "path/to/POSCAR"}
        req = FileExtractionRequest(engine="vasp", filepath=files)
        assert req.filepath == files


class TestMathematicalStructure:
    """Test mathematical structure model."""

    def test_creation(self):
        """Test basic creation."""
        struct = MathematicalStructure(
            problem_type=ProblemType.NONLINEAR_EIGENVALUE,
            canonical_form="H[n]ψ = εψ",
        )
        assert struct.problem_type == ProblemType.NONLINEAR_EIGENVALUE
        assert struct.canonical_form == "H[n]ψ = εψ"

    def test_default_dependencies(self):
        """Test default empty dependencies."""
        struct = MathematicalStructure()
        assert struct.variable_dependencies == {}


class TestConstraint:
    """Test constraint model."""

    def test_valid_constraint(self):
        """Test valid constraint."""
        cons = Constraint(expression="ENCUT > 0", satisfied=True)
        assert cons.expression == "ENCUT > 0"
        assert cons.satisfied is True
        assert cons.description is None

    def test_with_description(self):
        """Test constraint with description."""
        cons = Constraint(
            expression="ENCUT > 0",
            satisfied=True,
            description="Energy cutoff must be positive",
        )
        assert cons.description == "Energy cutoff must be positive"


class TestExtractionResultModel:
    """Test extraction result model."""

    def test_creation(self):
        """Test basic creation."""
        result = ExtractionResultModel(engine="vasp", success=True)
        assert result.engine == "vasp"
        assert result.success is True

    def test_default_fields(self):
        """Test default empty fields."""
        result = ExtractionResultModel(engine="vasp", success=True)
        assert result.constraints == []
        assert result.approximations == []
        assert result.errors == []
        assert result.warnings == []


class TestTieredAnalysisRequest:
    """Test tiered analysis request model."""

    def test_valid_request(self):
        """Test valid request."""
        req = TieredAnalysisRequest(filepath="test.lmp")
        assert req.filepath == "test.lmp"
        assert req.auto_tier is True
        assert req.min_tier == 1
        assert req.max_tier == 5

    def test_custom_tiers(self):
        """Test custom tier range."""
        req = TieredAnalysisRequest(
            filepath="test.lmp",
            min_tier=2,
            max_tier=4,
            auto_tier=False,
        )
        assert req.min_tier == 2
        assert req.max_tier == 4
        assert req.auto_tier is False

    def test_invalid_min_tier(self):
        """Test invalid min tier."""
        with pytest.raises(ValidationError):
            TieredAnalysisRequest(filepath="test.lmp", min_tier=0)

    def test_invalid_max_tier(self):
        """Test invalid max tier."""
        with pytest.raises(ValidationError):
            TieredAnalysisRequest(filepath="test.lmp", max_tier=6)

    def test_max_less_than_min(self):
        """Test max_tier < min_tier validation."""
        with pytest.raises(ValidationError) as exc_info:
            TieredAnalysisRequest(filepath="test.lmp", min_tier=4, max_tier=2)
        assert "max_tier must be >= min_tier" in str(exc_info.value)

    def test_specific_tier(self):
        """Test with specific tier."""
        req = TieredAnalysisRequest(
            filepath="test.lmp",
            tier=AnalysisTierEnum.PROFESSIONAL,
        )
        assert req.tier == AnalysisTierEnum.PROFESSIONAL


class TestComplexityScore:
    """Test complexity score model."""

    def test_valid_score(self):
        """Test valid complexity score."""
        score = ComplexityScore(
            total=50.0,
            system_size=20.0,
            time_scale=15.0,
            constraints=10.0,
            data_availability=5.0,
        )
        assert score.total == 50.0
        assert score.system_size == 20.0

    def test_total_range_validation(self):
        """Test total score range validation."""
        with pytest.raises(ValidationError):
            ComplexityScore(total=101.0)
        with pytest.raises(ValidationError):
            ComplexityScore(total=-1.0)

    def test_component_ranges(self):
        """Test component score ranges."""
        with pytest.raises(ValidationError):
            ComplexityScore(system_size=51.0)
        with pytest.raises(ValidationError):
            ComplexityScore(time_scale=31.0)


class TestResourceRequirements:
    """Test resource requirements model."""

    def test_default_values(self):
        """Test default resource values."""
        res = ResourceRequirements()
        assert res.cpu_time_seconds == 1.0
        assert res.memory_gb == 1.0
        assert res.gpu_required is False
        assert res.gpu_memory_gb == 0.0

    def test_gpu_requirements(self):
        """Test GPU requirements."""
        res = ResourceRequirements(
            gpu_required=True,
            gpu_memory_gb=8.0,
            cpu_time_seconds=3600.0,
        )
        assert res.gpu_required is True
        assert res.gpu_memory_gb == 8.0
        assert res.cpu_time_seconds == 3600.0

    def test_positive_constraints(self):
        """Test positive value constraints."""
        with pytest.raises(ValidationError):
            ResourceRequirements(cpu_time_seconds=0)
        with pytest.raises(ValidationError):
            ResourceRequirements(memory_gb=-1)
        with pytest.raises(ValidationError):
            ResourceRequirements(gpu_memory_gb=-1)


class TestTierRecommendation:
    """Test tier recommendation model."""

    def test_creation(self):
        """Test basic creation."""
        rec = TierRecommendation(
            recommended_tier=AnalysisTierEnum.ADVANCED,
            suitable_tiers=[AnalysisTierEnum.PROFESSIONAL, AnalysisTierEnum.ADVANCED],
        )
        assert rec.recommended_tier == AnalysisTierEnum.ADVANCED
        assert len(rec.suitable_tiers) == 2

    def test_default_values(self):
        """Test default values."""
        rec = TierRecommendation(
            recommended_tier=AnalysisTierEnum.BASIC,
            suitable_tiers=[AnalysisTierEnum.BASIC],
        )
        assert rec.reasons == []
        assert rec.estimated_time == "< 1 second"


class TestEnums:
    """Test enum types."""

    def test_analysis_tier_enum(self):
        """Test analysis tier enum values."""
        assert AnalysisTierEnum.BASIC.value == "basic"
        assert AnalysisTierEnum.ENHANCED.value == "enhanced"
        assert AnalysisTierEnum.PROFESSIONAL.value == "professional"
        assert AnalysisTierEnum.ADVANCED.value == "advanced"
        assert AnalysisTierEnum.COMPLETE.value == "complete"

    def test_engine_type_enum(self):
        """Test engine type enum values."""
        assert EngineType.VASP.value == "vasp"
        assert EngineType.LAMMPS.value == "lammps"
        assert EngineType.ABAQUS.value == "abaqus"

    def test_problem_type_enum(self):
        """Test problem type enum values."""
        assert ProblemType.NONLINEAR_EIGENVALUE.value == "nonlinear_eigenvalue"
        assert ProblemType.INITIAL_VALUE_ODE.value == "initial_value_ode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
