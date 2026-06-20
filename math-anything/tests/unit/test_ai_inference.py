"""Unit tests for ai/inference.py — IntelligentInferenceEngine, PhysicsKnowledgeBase."""

import pytest

from math_anything.ai.inference import (
    InferenceConfidence,
    InferenceResult,
    IntelligentInferenceEngine,
    PhysicsKnowledgeBase,
)


# ── InferenceConfidence enum ──

class TestInferenceConfidence:
    def test_confidence_values(self):
        assert InferenceConfidence.HIGH.value == "high"
        assert InferenceConfidence.MEDIUM.value == "medium"
        assert InferenceConfidence.LOW.value == "low"
        assert InferenceConfidence.SPECULATIVE.value == "speculative"

    def test_confidence_count(self):
        assert len(list(InferenceConfidence)) == 4


# ── InferenceResult ──

class TestInferenceResult:
    def test_result_creation_minimal(self):
        r = InferenceResult(
            parameter="timestep",
            inferred_value=1.0,
            confidence=InferenceConfidence.MEDIUM,
            reasoning="typical value",
            source="经验规律",
        )
        assert r.parameter == "timestep"
        assert r.inferred_value == 1.0
        assert r.confidence == InferenceConfidence.MEDIUM
        assert r.alternatives == []

    def test_result_creation_with_alternatives(self):
        r = InferenceResult(
            parameter="x",
            inferred_value=1,
            confidence=InferenceConfidence.HIGH,
            reasoning="law",
            source="物理定律",
            alternatives=[(2, InferenceConfidence.LOW)],
        )
        assert len(r.alternatives) == 1
        assert r.alternatives[0][0] == 2


# ── PhysicsKnowledgeBase ──

class TestPhysicsKnowledgeBaseParameterRanges:
    def test_has_vasp_params(self):
        assert "encut" in PhysicsKnowledgeBase.PARAMETER_RANGES
        assert "ediff" in PhysicsKnowledgeBase.PARAMETER_RANGES
        assert "kpoints_grid" in PhysicsKnowledgeBase.PARAMETER_RANGES

    def test_has_lammps_params(self):
        assert "timestep" in PhysicsKnowledgeBase.PARAMETER_RANGES
        assert "temperature" in PhysicsKnowledgeBase.PARAMETER_RANGES
        assert "pressure" in PhysicsKnowledgeBase.PARAMETER_RANGES

    def test_has_fem_params(self):
        assert "youngs_modulus" in PhysicsKnowledgeBase.PARAMETER_RANGES
        assert "poisson_ratio" in PhysicsKnowledgeBase.PARAMETER_RANGES
        assert "density" in PhysicsKnowledgeBase.PARAMETER_RANGES

    def test_has_cfd_params(self):
        assert "reynolds_number" in PhysicsKnowledgeBase.PARAMETER_RANGES
        assert "viscosity" in PhysicsKnowledgeBase.PARAMETER_RANGES

    def test_has_gaussian_params(self):
        assert "basis_set_quality" in PhysicsKnowledgeBase.PARAMETER_RANGES
        assert "method_quality" in PhysicsKnowledgeBase.PARAMETER_RANGES

    def test_param_range_has_min_max(self):
        info = PhysicsKnowledgeBase.PARAMETER_RANGES["encut"]
        assert "min" in info
        assert "max" in info
        assert "typical" in info

    def test_param_range_has_unit(self):
        info = PhysicsKnowledgeBase.PARAMETER_RANGES["encut"]
        assert info["unit"] == "eV"


class TestPhysicsKnowledgeBaseConstraints:
    def test_has_poisson_constraint(self):
        assert "poisson_ratio_range" in PhysicsKnowledgeBase.PHYSICS_CONSTRAINTS

    def test_has_bulk_modulus_constraint(self):
        assert "bulk_modulus_positive" in PhysicsKnowledgeBase.PHYSICS_CONSTRAINTS

    def test_has_speed_of_sound(self):
        assert "speed_of_sound" in PhysicsKnowledgeBase.PHYSICS_CONSTRAINTS

    def test_has_cfl_condition(self):
        assert "cfl_condition" in PhysicsKnowledgeBase.PHYSICS_CONSTRAINTS

    def test_has_energy_conservation(self):
        assert "energy_conservation" in PhysicsKnowledgeBase.PHYSICS_CONSTRAINTS


class TestPhysicsKnowledgeBaseCrossEngineMap:
    def test_has_vasp_to_qe_mapping(self):
        assert ("vasp", "quantum_espresso") in PhysicsKnowledgeBase.CROSS_ENGINE_MAP

    def test_has_lammps_to_openfoam_mapping(self):
        assert ("lammps", "openfoam") in PhysicsKnowledgeBase.CROSS_ENGINE_MAP

    def test_has_abaqus_to_ansys_mapping(self):
        assert ("abaqus", "ansys") in PhysicsKnowledgeBase.CROSS_ENGINE_MAP

    def test_vasp_qe_encut_mapping(self):
        mapping = PhysicsKnowledgeBase.CROSS_ENGINE_MAP[("vasp", "quantum_espresso")]
        assert mapping["encut"] == "ecutwfc"


class TestPhysicsKnowledgeBaseInferParameter:
    def test_infer_typical_value(self):
        result = PhysicsKnowledgeBase.infer_parameter(
            "encut", {}, "vasp"
        )
        assert result is not None
        assert result.inferred_value == 520
        assert result.confidence == InferenceConfidence.MEDIUM
        assert result.source == "经验规律"

    def test_infer_timestep_typical(self):
        result = PhysicsKnowledgeBase.infer_parameter(
            "timestep", {}, "lammps"
        )
        assert result is not None
        assert result.inferred_value == 1.0

    def test_infer_timestep_from_reynolds(self):
        # timestep has a typical value in PARAMETER_RANGES, so strategy 1
        # (typical value) is applied before strategy 2 (CFL condition).
        # Verify the typical value is returned.
        result = PhysicsKnowledgeBase.infer_parameter(
            "timestep", {"reynolds_number": 1000}, "openfoam"
        )
        assert result is not None
        assert result.confidence == InferenceConfidence.MEDIUM
        assert result.inferred_value == 1.0  # typical value

    def test_infer_speed_of_sound(self):
        result = PhysicsKnowledgeBase.infer_parameter(
            "speed_of_sound",
            {"youngs_modulus": 200, "density": 7.8},
            "abaqus",
        )
        assert result is not None
        assert result.confidence == InferenceConfidence.HIGH
        assert result.inferred_value > 0

    def test_infer_cross_engine_migration(self):
        # VASP encut -> QE ecutwfc
        result = PhysicsKnowledgeBase.infer_parameter(
            "ecutwfc", {"encut": 520}, "quantum_espresso"
        )
        assert result is not None
        assert result.confidence == InferenceConfidence.LOW
        assert result.inferred_value == 520
        assert result.source == "跨引擎迁移"

    def test_infer_unknown_parameter_returns_none(self):
        result = PhysicsKnowledgeBase.infer_parameter(
            "totally_unknown_param", {}, "vasp"
        )
        assert result is None

    def test_infer_no_typical_value(self):
        # basis_set_quality has 'values' but no 'typical'
        result = PhysicsKnowledgeBase.infer_parameter(
            "basis_set_quality", {}, "gaussian"
        )
        # No 'typical' key, so strategy 1 fails; no other strategy applies
        assert result is None


# ── IntelligentInferenceEngine ──

class TestIntelligentInferenceEngineCreation:
    def test_creates_with_kb(self):
        engine = IntelligentInferenceEngine()
        assert engine._kb is not None

    def test_creates_with_empty_history(self):
        engine = IntelligentInferenceEngine()
        assert engine.inference_history == []


class TestInferMissingParameters:
    def test_infer_no_missing(self):
        engine = IntelligentInferenceEngine()
        results = engine.infer_missing_parameters(
            "vasp",
            {"encut": 520, "ediff": 1e-6},
            ["encut", "ediff"],
        )
        assert results == {}

    def test_infer_missing_encut(self):
        engine = IntelligentInferenceEngine()
        results = engine.infer_missing_parameters(
            "vasp",
            {},
            ["encut"],
        )
        assert "encut" in results
        assert results["encut"].inferred_value == 520

    def test_infer_multiple_missing(self):
        engine = IntelligentInferenceEngine()
        results = engine.infer_missing_parameters(
            "vasp",
            {},
            ["encut", "ediff"],
        )
        assert "encut" in results
        assert "ediff" in results

    def test_infer_appends_to_history(self):
        engine = IntelligentInferenceEngine()
        engine.infer_missing_parameters("vasp", {}, ["encut"])
        assert len(engine.inference_history) == 1

    def test_infer_unknown_param_skipped(self):
        engine = IntelligentInferenceEngine()
        results = engine.infer_missing_parameters(
            "vasp", {}, ["totally_unknown"]
        )
        assert results == {}

    def test_inference_history_returns_copy(self):
        engine = IntelligentInferenceEngine()
        engine.infer_missing_parameters("vasp", {}, ["encut"])
        history = engine.inference_history
        history.clear()
        # Original should be unchanged
        assert len(engine.inference_history) == 1


class TestCheckPhysicsConsistency:
    def test_no_issues_with_valid_params(self):
        engine = IntelligentInferenceEngine()
        issues = engine.check_physics_consistency({
            "poisson_ratio": 0.3,
            "youngs_modulus": 200,
            "density": 7.8,
            "temperature": 300,
        })
        assert issues == []

    def test_poisson_ratio_too_high(self):
        engine = IntelligentInferenceEngine()
        issues = engine.check_physics_consistency({"poisson_ratio": 0.6})
        assert len(issues) == 1
        assert "Poisson" in issues[0]

    def test_poisson_ratio_negative(self):
        engine = IntelligentInferenceEngine()
        issues = engine.check_physics_consistency({"poisson_ratio": -0.1})
        assert len(issues) == 1

    def test_youngs_modulus_zero(self):
        engine = IntelligentInferenceEngine()
        issues = engine.check_physics_consistency({"youngs_modulus": 0})
        assert len(issues) == 1
        assert "杨氏模量" in issues[0]

    def test_youngs_modulus_negative(self):
        engine = IntelligentInferenceEngine()
        issues = engine.check_physics_consistency({"youngs_modulus": -10})
        assert len(issues) == 1

    def test_density_zero(self):
        engine = IntelligentInferenceEngine()
        issues = engine.check_physics_consistency({"density": 0})
        assert len(issues) == 1
        assert "密度" in issues[0]

    def test_temperature_negative(self):
        engine = IntelligentInferenceEngine()
        issues = engine.check_physics_consistency({"temperature": -5})
        assert len(issues) == 1
        assert "温度" in issues[0]

    def test_cfl_violation(self):
        engine = IntelligentInferenceEngine()
        issues = engine.check_physics_consistency({
            "timestep": 10.0,
            "mesh_size": 0.1,
            "speed_of_sound": 1.0,
        })
        assert len(issues) >= 1
        assert any("CFL" in i for i in issues)

    def test_empty_params_no_issues(self):
        engine = IntelligentInferenceEngine()
        issues = engine.check_physics_consistency({})
        assert issues == []


class TestSuggestImprovements:
    def test_no_suggestions_for_good_params(self):
        engine = IntelligentInferenceEngine()
        suggestions = engine.suggest_improvements(
            "vasp", {"encut": 520, "timestep": 1.0}
        )
        assert suggestions == []

    def test_low_encut_suggestion(self):
        engine = IntelligentInferenceEngine()
        suggestions = engine.suggest_improvements("vasp", {"encut": 300})
        assert len(suggestions) == 1
        assert "ENCUT" in suggestions[0]

    def test_large_timestep_suggestion(self):
        engine = IntelligentInferenceEngine()
        suggestions = engine.suggest_improvements("lammps", {"timestep": 5.0})
        assert len(suggestions) == 1
        assert "timestep" in suggestions[0]

    def test_high_reynolds_suggestion(self):
        engine = IntelligentInferenceEngine()
        suggestions = engine.suggest_improvements(
            "openfoam", {"reynolds_number": 1e7}
        )
        assert len(suggestions) == 1
        assert "Re" in suggestions[0]

    def test_low_basis_set_suggestion(self):
        engine = IntelligentInferenceEngine()
        suggestions = engine.suggest_improvements(
            "gaussian", {"basis": "sto-3g"}
        )
        assert len(suggestions) == 1
        assert "基组" in suggestions[0]

    def test_empty_params_no_suggestions(self):
        engine = IntelligentInferenceEngine()
        suggestions = engine.suggest_improvements("vasp", {})
        assert suggestions == []


class TestCrossEngineTranslate:
    def test_translate_vasp_to_qe(self):
        engine = IntelligentInferenceEngine()
        results = engine.cross_engine_translate(
            "vasp", "quantum_espresso", {"encut": 520, "ediff": 1e-6}
        )
        assert "ecutwfc" in results
        assert "conv_thr" in results
        assert results["ecutwfc"].inferred_value == 520
        assert results["ecutwfc"].confidence == InferenceConfidence.LOW

    def test_translate_lammps_to_openfoam(self):
        engine = IntelligentInferenceEngine()
        results = engine.cross_engine_translate(
            "lammps", "openfoam", {"timestep": 1.0, "temperature": 300}
        )
        assert "deltaT" in results
        assert "T" in results

    def test_translate_abaqus_to_ansys(self):
        engine = IntelligentInferenceEngine()
        results = engine.cross_engine_translate(
            "abaqus", "ansys", {"youngs_modulus": 200, "poisson_ratio": 0.3, "density": 7.8}
        )
        assert "EX" in results
        assert "PRXY" in results
        assert "DENS" in results

    def test_translate_unknown_pair(self):
        engine = IntelligentInferenceEngine()
        results = engine.cross_engine_translate(
            "unknown1", "unknown2", {"x": 1}
        )
        assert results == {}

    def test_translate_partial_params(self):
        engine = IntelligentInferenceEngine()
        # Only encut provided, ediff missing
        results = engine.cross_engine_translate(
            "vasp", "quantum_espresso", {"encut": 520}
        )
        assert "ecutwfc" in results
        assert "conv_thr" not in results
