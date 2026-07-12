"""Unit tests for ai/nlp_interface.py — NaturalLanguageInterface, NLParseResult."""

import pytest

from math_anything.ai.nlp_interface import (
    _ANALYSIS_KEYWORDS,
    _ENGINE_KEYWORDS,
    _MATERIAL_KEYWORDS,
    NaturalLanguageInterface,
    NLParseResult,
)

# ── NLParseResult ──


class TestNLParseResult:
    def test_result_defaults(self):
        r = NLParseResult(raw_query="test")
        assert r.raw_query == "test"
        assert r.detected_engine is None
        assert r.detected_material is None
        assert r.detected_analysis is None
        assert r.extracted_parameters == {}
        assert r.confidence == 0.0
        assert r.alternatives == []
        assert r.clarification_questions == []

    def test_result_with_values(self):
        r = NLParseResult(
            raw_query="test",
            detected_engine="vasp",
            detected_material="silicon",
            detected_analysis="band_structure",
            extracted_parameters={"encut": 520},
            confidence=0.8,
        )
        assert r.detected_engine == "vasp"
        assert r.confidence == 0.8


# ── Keyword mappings ──


class TestKeywordMappings:
    def test_engine_keywords_has_vasp(self):
        assert "vasp" in _ENGINE_KEYWORDS
        assert "dft" in _ENGINE_KEYWORDS["vasp"]

    def test_engine_keywords_has_lammps(self):
        assert "lammps" in _ENGINE_KEYWORDS
        assert "md" in _ENGINE_KEYWORDS["lammps"]

    def test_engine_keywords_has_abaqus(self):
        assert "abaqus" in _ENGINE_KEYWORDS
        assert "fem" in _ENGINE_KEYWORDS["abaqus"]

    def test_engine_keywords_has_openfoam(self):
        assert "openfoam" in _ENGINE_KEYWORDS
        assert "cfd" in _ENGINE_KEYWORDS["openfoam"]

    def test_material_keywords_has_steel(self):
        assert "steel" in _MATERIAL_KEYWORDS
        assert "youngs_modulus" in _MATERIAL_KEYWORDS["steel"]

    def test_material_keywords_has_water(self):
        assert "water" in _MATERIAL_KEYWORDS
        assert "density" in _MATERIAL_KEYWORDS["water"]

    def test_analysis_keywords_has_stress(self):
        assert "应力" in _ANALYSIS_KEYWORDS
        assert _ANALYSIS_KEYWORDS["应力"] == "static_structural"

    def test_analysis_keywords_has_band(self):
        assert "能带" in _ANALYSIS_KEYWORDS
        assert _ANALYSIS_KEYWORDS["能带"] == "band_structure"


# ── NaturalLanguageInterface creation ──


class TestNaturalLanguageInterfaceCreation:
    def test_creates_with_keywords(self):
        nlp = NaturalLanguageInterface()
        assert nlp._engine_keywords is not None
        assert nlp._material_keywords is not None
        assert nlp._analysis_keywords is not None

    def test_engine_keywords_not_empty(self):
        nlp = NaturalLanguageInterface()
        assert len(nlp._engine_keywords) > 0

    def test_material_keywords_not_empty(self):
        nlp = NaturalLanguageInterface()
        assert len(nlp._material_keywords) > 0


# ── parse() — engine detection ──


class TestParseEngineDetection:
    def test_parse_vasp_keyword(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("calculate the band structure of silicon using DFT")
        assert result.detected_engine == "vasp"

    def test_parse_lammps_keyword(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("run molecular dynamics simulation with NVT ensemble")
        assert result.detected_engine == "lammps"

    def test_parse_abaqus_keyword(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("finite element analysis of stress in a beam")
        assert result.detected_engine == "abaqus"

    def test_parse_openfoam_keyword(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("CFD simulation of turbulent flow")
        assert result.detected_engine == "openfoam"

    def test_parse_no_engine_detected(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("hello world")
        assert result.detected_engine is None

    def test_parse_confidence_increases_with_keywords(self):
        nlp = NaturalLanguageInterface()
        single = nlp.parse("DFT calculation")
        multi = nlp.parse("DFT band structure calculation for semiconductor silicon")
        assert multi.confidence >= single.confidence

    def test_parse_alternatives_populated(self):
        nlp = NaturalLanguageInterface()
        # Both VASP and QE match "dft" and "能带"
        result = nlp.parse("DFT 能带 calculation")
        assert isinstance(result.alternatives, list)

    def test_parse_case_insensitive(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("DFT CALCULATION")
        assert result.detected_engine is not None


# ── parse() — material detection ──


class TestParseMaterialDetection:
    def test_parse_steel_material(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("stress analysis of steel beam")
        assert result.detected_material == "steel"
        assert result.extracted_parameters.get("youngs_modulus") == 200

    def test_parse_aluminum_material(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("aluminum component analysis")
        assert result.detected_material == "aluminum"
        assert result.extracted_parameters.get("youngs_modulus") == 70

    def test_parse_silicon_material(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("silicon semiconductor band structure")
        assert result.detected_material == "silicon"

    def test_parse_water_material(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("water flow simulation")
        assert result.detected_material == "water"
        assert result.extracted_parameters.get("density") == 1.0

    def test_parse_no_material_detected(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("run a simulation")
        assert result.detected_material is None


# ── parse() — analysis type detection ──


class TestParseAnalysisDetection:
    def test_parse_stress_analysis(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("应力分析 of a beam")
        assert result.detected_analysis == "static_structural"

    def test_parse_band_structure(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("能带 structure calculation")
        assert result.detected_analysis == "band_structure"

    def test_parse_md_analysis(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("分子动力学 simulation")
        assert result.detected_analysis == "md"

    def test_parse_no_analysis_detected(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("hello world")
        assert result.detected_analysis is None


# ── parse() — number extraction ──


class TestParseNumberExtraction:
    def test_extract_load_newton(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("apply 1000N load")
        assert result.extracted_parameters.get("load") == 1000.0

    def test_extract_temperature_kelvin(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("heat to 300K")
        assert result.extracted_parameters.get("temperature") == 300.0

    def test_extract_pressure_mpa(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("apply 1.5MPa pressure")
        assert result.extracted_parameters.get("pressure") == 1.5

    def test_extract_encut_ev(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("set ENCUT to 520eV")
        assert result.extracted_parameters.get("encut") == 520.0

    def test_extract_time_ps(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("timestep 1.5ps")
        # ps converted to fs
        assert result.extracted_parameters.get("timestep") == 1500.0

    def test_extract_reynolds_number(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("Re=1000 flow")
        assert result.extracted_parameters.get("reynolds_number") == 1000.0

    def test_extract_no_numbers(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("no numbers here")
        # Only number params should be absent
        assert "load" not in result.extracted_parameters
        assert "temperature" not in result.extracted_parameters


# ── parse() — clarification questions ──


class TestParseClarification:
    def test_no_engine_adds_clarification(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("hello world")
        assert len(result.clarification_questions) >= 1
        assert any("引擎" in q for q in result.clarification_questions)

    def test_no_analysis_adds_clarification(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("hello world")
        assert any("分析" in q for q in result.clarification_questions)

    def test_abaqus_without_material_adds_clarification(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("finite element stress analysis")
        # engine=abaqus detected, but no material
        assert any("材料" in q for q in result.clarification_questions)

    def test_complete_query_no_clarifications(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("应力 analysis of steel beam with 1000N load")
        # All fields detected, no clarifications needed
        assert len(result.clarification_questions) == 0


# ── generate_prompt() ──


class TestGeneratePrompt:
    def test_generate_prompt_basic(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("stress analysis of steel beam")
        prompt = nlp.generate_prompt(result)
        assert "Engine:" in prompt
        assert "Material:" in prompt
        assert "Analysis:" in prompt
        assert "Parameters:" in prompt

    def test_generate_prompt_with_clarifications(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("hello world")
        prompt = nlp.generate_prompt(result)
        assert "Clarifications needed:" in prompt

    def test_generate_prompt_contains_engine(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("DFT band structure of silicon")
        prompt = nlp.generate_prompt(result)
        assert "vasp" in prompt

    def test_generate_prompt_contains_material(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("stress of steel beam")
        prompt = nlp.generate_prompt(result)
        assert "steel" in prompt

    def test_generate_prompt_unknown_engine(self):
        nlp = NaturalLanguageInterface()
        result = nlp.parse("hello world")
        prompt = nlp.generate_prompt(result)
        assert "unknown" in prompt


# ── _extract_numbers() ──


class TestExtractNumbers:
    def test_extract_load_with_space(self):
        nlp = NaturalLanguageInterface()
        params = nlp._extract_numbers("apply 500 N force")
        assert params.get("load") == 500.0

    def test_extract_load_newton_word(self):
        nlp = NaturalLanguageInterface()
        params = nlp._extract_numbers("apply 500 Newton")
        assert params.get("load") == 500.0

    def test_extract_temperature_with_space(self):
        nlp = NaturalLanguageInterface()
        params = nlp._extract_numbers("heat to 250 K")
        assert params.get("temperature") == 250.0

    def test_extract_no_match(self):
        nlp = NaturalLanguageInterface()
        params = nlp._extract_numbers("no numbers here")
        assert params == {}

    def test_extract_multiple_numbers(self):
        nlp = NaturalLanguageInterface()
        params = nlp._extract_numbers("1000N load at 300K with 520eV")
        assert params.get("load") == 1000.0
        assert params.get("temperature") == 300.0
        assert params.get("encut") == 520.0
