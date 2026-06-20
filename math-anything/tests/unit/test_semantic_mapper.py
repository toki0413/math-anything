"""Unit tests for codegen/semantic_mapper.py — semantic mapping to math concepts."""

import pytest

from math_anything.codegen.semantic_mapper import (
    MathMapping,
    SemanticMapper,
    quick_map,
)


# ── MathMapping dataclass ──

class TestMathMapping:
    def test_creation(self):
        m = MathMapping(
            command_name="nvt",
            math_type="governing_equation",
            mathematical_form="dT/dt = ...",
            variables=["T"],
            confidence=0.8,
            description="thermostat",
        )
        assert m.command_name == "nvt"
        assert m.math_type == "governing_equation"
        assert m.mathematical_form == "dT/dt = ..."
        assert m.variables == ["T"]
        assert m.confidence == 0.8
        assert m.description == "thermostat"

    def test_with_empty_variables(self):
        m = MathMapping(
            command_name="x",
            math_type="unknown",
            mathematical_form="",
            variables=[],
            confidence=0.0,
            description="",
        )
        assert m.variables == []
        assert m.confidence == 0.0


# ── SemanticMapper: creation ──

class TestSemanticMapperCreation:
    def test_creates_empty(self):
        m = SemanticMapper()
        assert m.mappings == []

    def test_has_equation_keywords(self):
        m = SemanticMapper()
        assert "nvt" in m.EQUATION_KEYWORDS
        assert "npt" in m.EQUATION_KEYWORDS
        assert "minimize" in m.EQUATION_KEYWORDS
        assert "heat" in m.EQUATION_KEYWORDS
        assert "deform" in m.EQUATION_KEYWORDS

    def test_has_boundary_keywords(self):
        m = SemanticMapper()
        assert "fix" in m.BOUNDARY_KEYWORDS
        assert "velocity" in m.BOUNDARY_KEYWORDS
        assert "force" in m.BOUNDARY_KEYWORDS
        assert "temperature" in m.BOUNDARY_KEYWORDS
        assert "wall" in m.BOUNDARY_KEYWORDS
        assert "periodic" in m.BOUNDARY_KEYWORDS

    def test_has_numerical_keywords(self):
        m = SemanticMapper()
        assert "verlet" in m.NUMERICAL_KEYWORDS
        assert "leapfrog" in m.NUMERICAL_KEYWORDS
        assert "rk4" in m.NUMERICAL_KEYWORDS
        assert "cg" in m.NUMERICAL_KEYWORDS
        assert "pcg" in m.NUMERICAL_KEYWORDS
        assert "gmres" in m.NUMERICAL_KEYWORDS

    def test_has_physical_parameters(self):
        m = SemanticMapper()
        assert "temp" in m.PHYSICAL_PARAMETERS
        assert "pressure" in m.PHYSICAL_PARAMETERS
        assert "volume" in m.PHYSICAL_PARAMETERS
        assert "density" in m.PHYSICAL_PARAMETERS
        assert "mass" in m.PHYSICAL_PARAMETERS
        assert "charge" in m.PHYSICAL_PARAMETERS

    def test_equation_keyword_has_form(self):
        m = SemanticMapper()
        assert "form" in m.EQUATION_KEYWORDS["nvt"]
        assert "type" in m.EQUATION_KEYWORDS["nvt"]
        assert "description" in m.EQUATION_KEYWORDS["nvt"]

    def test_boundary_keyword_has_type(self):
        m = SemanticMapper()
        assert "type" in m.BOUNDARY_KEYWORDS["fix"]
        assert "field" in m.BOUNDARY_KEYWORDS["fix"]

    def test_numerical_keyword_has_method(self):
        m = SemanticMapper()
        assert "method" in m.NUMERICAL_KEYWORDS["verlet"]
        assert "order" in m.NUMERICAL_KEYWORDS["verlet"]
        assert "symplectic" in m.NUMERICAL_KEYWORDS["verlet"]

    def test_physical_parameter_has_symbol(self):
        m = SemanticMapper()
        assert "symbol" in m.PHYSICAL_PARAMETERS["temp"]
        assert "unit" in m.PHYSICAL_PARAMETERS["temp"]
        assert "physical" in m.PHYSICAL_PARAMETERS["temp"]


# ── SemanticMapper: map_commands ──

class TestMapCommands:
    def test_empty_commands(self):
        m = SemanticMapper()
        result = m.map_commands([])
        assert result["equations"] == []
        assert result["boundary_conditions"] == []
        assert result["numerical_methods"] == []
        assert result["variables"] == []
        assert result["total_mappings"] == 0

    def test_nvt_command_maps_to_equation(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "nvt", "description": "thermostat"}])
        assert len(result["equations"]) >= 1
        eq = result["equations"][0]
        assert eq["source_command"] == "nvt"
        assert "thermostat" in eq["name"].lower() or "thermostat" in eq["type"]

    def test_npt_command_maps_to_equation(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "npt", "description": "barostat"}])
        assert len(result["equations"]) >= 1
        assert "barostat" in result["equations"][0]["type"]

    def test_minimize_command_maps_to_equation(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "minimize", "description": "energy min"}])
        assert len(result["equations"]) >= 1
        assert "energy_minimization" in result["equations"][0]["type"]

    def test_heat_command_maps_to_equation(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "heat", "description": "heat transfer"}])
        assert len(result["equations"]) >= 1
        assert "heat_equation" in result["equations"][0]["type"]

    def test_deform_command_maps_to_equation(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "deform", "description": "deformation"}])
        assert len(result["equations"]) >= 1
        assert "mechanics" in result["equations"][0]["type"]

    def test_equation_confidence_higher_for_name_match(self):
        m = SemanticMapper()
        # Keyword in name -> confidence 0.8
        result_name = m.map_commands([{"name": "nvt", "description": ""}])
        # Keyword in description only -> confidence 0.5
        m2 = SemanticMapper()
        result_desc = m2.map_commands([{"name": "cmd", "description": "nvt thermostat"}])
        if result_name["equations"] and result_desc["equations"]:
            assert result_name["equations"][0]["confidence"] >= result_desc["equations"][0]["confidence"]

    def test_fix_command_maps_to_boundary(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "fix", "description": "boundary"}])
        assert len(result["boundary_conditions"]) >= 1
        bc = result["boundary_conditions"][0]
        assert bc["type"] == "boundary_condition"
        assert bc["field"] == "displacement"

    def test_velocity_command_maps_to_boundary(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "velocity", "description": "set velocity"}])
        assert len(result["boundary_conditions"]) >= 1
        assert result["boundary_conditions"][0]["field"] == "velocity"

    def test_verlet_command_maps_to_numerical(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "verlet", "description": "integrator"}])
        assert len(result["numerical_methods"]) >= 1
        nm = result["numerical_methods"][0]
        assert nm["method"] == "velocity_verlet"
        assert nm["order"] == 2
        assert nm["symplectic"] is True

    def test_rk4_command_maps_to_numerical(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "rk4", "description": "integrator"}])
        assert len(result["numerical_methods"]) >= 1
        assert result["numerical_methods"][0]["method"] == "runge_kutta_4"
        assert result["numerical_methods"][0]["order"] == 4
        assert result["numerical_methods"][0]["symplectic"] is False

    def test_cg_command_maps_to_numerical(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "cg", "description": "optimizer"}])
        assert len(result["numerical_methods"]) >= 1
        assert result["numerical_methods"][0]["method"] == "conjugate_gradient"

    def test_temp_parameter_maps_to_variable(self):
        m = SemanticMapper()
        result = m.map_commands(
            commands=[],
            parameters=[{"name": "temp"}],
        )
        assert len(result["variables"]) >= 1
        v = result["variables"][0]
        assert v["symbol"] == "T"
        assert v["unit"] == "K"
        assert v["physical_quantity"] == "temperature"

    def test_pressure_parameter_maps_to_variable(self):
        m = SemanticMapper()
        result = m.map_commands(
            commands=[],
            parameters=[{"name": "pressure"}],
        )
        assert len(result["variables"]) >= 1
        assert result["variables"][0]["symbol"] == "P"

    def test_multiple_parameters(self):
        m = SemanticMapper()
        result = m.map_commands(
            commands=[],
            parameters=[{"name": "temp"}, {"name": "pressure"}, {"name": "volume"}],
        )
        assert len(result["variables"]) >= 3

    def test_unknown_parameter_not_mapped(self):
        m = SemanticMapper()
        result = m.map_commands(
            commands=[],
            parameters=[{"name": "unknown_param"}],
        )
        assert result["variables"] == []

    def test_explicit_equations_added(self):
        m = SemanticMapper()
        result = m.map_commands(
            commands=[],
            equations=[{"form": "E = mc^2", "type": "energy"}],
        )
        assert len(result["equations"]) >= 1
        eq = result["equations"][0]
        assert eq["form"] == "E = mc^2"
        assert eq["confidence"] == 0.9

    def test_explicit_equation_with_source(self):
        m = SemanticMapper()
        result = m.map_commands(
            commands=[],
            equations=[{"form": "F = ma", "type": "newton", "source_file": "physics.py"}],
        )
        assert len(result["equations"]) >= 1
        assert result["equations"][0]["source"] == "physics.py"

    def test_returns_total_mappings(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "nvt", "description": "thermostat"}])
        assert "total_mappings" in result
        assert result["total_mappings"] >= 1

    def test_result_has_all_keys(self):
        m = SemanticMapper()
        result = m.map_commands([])
        assert "equations" in result
        assert "boundary_conditions" in result
        assert "numerical_methods" in result
        assert "variables" in result
        assert "total_mappings" in result

    def test_description_keyword_match(self):
        m = SemanticMapper()
        # "nvt" in description, not in name
        result = m.map_commands([{"name": "cmd", "description": "use nvt thermostat"}])
        assert len(result["equations"]) >= 1

    def test_case_insensitive_match(self):
        m = SemanticMapper()
        # Names are lowercased before matching
        result = m.map_commands([{"name": "NVT", "description": "thermostat"}])
        assert len(result["equations"]) >= 1


# ── SemanticMapper: _extract_variables ──

class TestExtractVariables:
    def test_extracts_variables(self):
        m = SemanticMapper()
        vars = m._extract_variables("dT/dt = (T_target - T) / tau_T")
        assert isinstance(vars, list)
        assert len(vars) > 0

    def test_excludes_math_functions(self):
        m = SemanticMapper()
        vars = m._extract_variables("sin(x) + cos(y) + exp(z)")
        assert "sin" not in vars
        assert "cos" not in vars
        assert "exp" not in vars

    def test_excludes_differentiation_symbols(self):
        m = SemanticMapper()
        vars = m._extract_variables("dT/dt = 0")
        # "d", "dt", "dx" are excluded
        assert "d" not in vars
        assert "dt" not in vars

    def test_excludes_pi_and_e(self):
        m = SemanticMapper()
        vars = m._extract_variables("pi + e + x")
        assert "pi" not in vars
        assert "e" not in vars
        assert "x" in vars

    def test_returns_unique(self):
        m = SemanticMapper()
        vars = m._extract_variables("alpha alpha alpha")
        # set() deduplicates
        assert vars.count("alpha") <= 1

    def test_empty_expression(self):
        m = SemanticMapper()
        vars = m._extract_variables("")
        assert vars == []

    def test_numbers_not_extracted(self):
        m = SemanticMapper()
        vars = m._extract_variables("123 456 789")
        assert vars == []


# ── SemanticMapper: suggest_math_semantics ──

class TestSuggestMathSemantics:
    def test_suggest_nvt(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("nvt")
        assert suggestion is not None
        assert suggestion["type"] == "governing_equation"
        assert "mathematical_form" in suggestion
        assert suggestion["confidence"] == 0.9

    def test_suggest_npt(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("npt")
        assert suggestion is not None
        assert suggestion["type"] == "governing_equation"

    def test_suggest_minimize(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("minimize")
        assert suggestion is not None
        assert "minimization" in suggestion["name"].lower() or "minimization" in suggestion["mathematical_form"].lower()

    def test_suggest_fix_pattern(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("fix_wall")
        assert suggestion is not None
        assert suggestion["type"] == "boundary_condition"
        assert suggestion["confidence"] == 0.7

    def test_suggest_compute_pattern(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("compute_temp")
        assert suggestion is not None
        assert suggestion["type"] == "derived_quantity"
        assert suggestion["confidence"] == 0.6

    def test_suggest_unknown_returns_none(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("zzz_unknown")
        assert suggestion is None

    def test_suggest_case_insensitive(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("NVT")
        assert suggestion is not None

    def test_suggest_with_context(self):
        m = SemanticMapper()
        # context is accepted but not heavily used
        suggestion = m.suggest_math_semantics("nvt", context="thermostat")
        assert suggestion is not None

    def test_suggest_fix_in_name(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("my_fix_command")
        assert suggestion is not None
        assert suggestion["type"] == "boundary_condition"

    def test_suggest_compute_in_name(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("my_compute_command")
        assert suggestion is not None
        assert suggestion["type"] == "derived_quantity"


# ── SemanticMapper: build_computational_graph ──

class TestBuildComputationalGraph:
    def test_empty_commands(self):
        m = SemanticMapper()
        graph = m.build_computational_graph([])
        assert graph["nodes"] == []
        assert graph["edges"] == []

    def test_single_command(self):
        m = SemanticMapper()
        graph = m.build_computational_graph([{"name": "nvt"}])
        assert len(graph["nodes"]) == 1
        assert graph["edges"] == []
        assert graph["nodes"][0]["id"] == "node_0"
        assert graph["nodes"][0]["type"] == "nvt"

    def test_multiple_commands_sequential_edges(self):
        m = SemanticMapper()
        graph = m.build_computational_graph([
            {"name": "nvt"},
            {"name": "minimize"},
            {"name": "run"},
        ])
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2  # sequential links
        # Check edge structure
        for edge in graph["edges"]:
            assert "from" in edge
            assert "to" in edge
            assert edge["dependency"] == "sequence"

    def test_node_has_math_semantics(self):
        m = SemanticMapper()
        graph = m.build_computational_graph([{"name": "nvt"}])
        assert "math_semantics" in graph["nodes"][0]

    def test_node_math_semantics_populated_for_known(self):
        m = SemanticMapper()
        graph = m.build_computational_graph([{"name": "nvt"}])
        # nvt is a known keyword, so semantics should be populated
        assert graph["nodes"][0]["math_semantics"] != {}

    def test_node_math_semantics_empty_for_unknown(self):
        m = SemanticMapper()
        graph = m.build_computational_graph([{"name": "zzz_unknown"}])
        # Unknown command -> empty semantics
        assert graph["nodes"][0]["math_semantics"] == {}

    def test_node_ids_increment(self):
        m = SemanticMapper()
        graph = m.build_computational_graph([
            {"name": "a"},
            {"name": "b"},
            {"name": "c"},
        ])
        ids = [n["id"] for n in graph["nodes"]]
        assert ids == ["node_0", "node_1", "node_2"]

    def test_edges_link_consecutive_nodes(self):
        m = SemanticMapper()
        graph = m.build_computational_graph([
            {"name": "a"},
            {"name": "b"},
        ])
        edge = graph["edges"][0]
        assert edge["from"] == "node_0"
        assert edge["to"] == "node_1"

    def test_result_has_nodes_and_edges(self):
        m = SemanticMapper()
        graph = m.build_computational_graph([{"name": "x"}])
        assert "nodes" in graph
        assert "edges" in graph


# ── quick_map ──

class TestQuickMap:
    def test_quick_map_returns_dict(self):
        result = quick_map([{"name": "nvt", "description": "thermostat"}])
        assert isinstance(result, dict)
        assert "equations" in result

    def test_quick_map_empty(self):
        result = quick_map([])
        assert result["equations"] == []
        assert result["boundary_conditions"] == []
        assert result["numerical_methods"] == []
        assert result["variables"] == []

    def test_quick_map_finds_equation(self):
        result = quick_map([{"name": "nvt", "description": "thermostat"}])
        assert len(result["equations"]) >= 1

    def test_quick_map_with_parameters(self):
        result = quick_map([{"name": "x"}])
        # quick_map only passes commands, no parameters
        assert "variables" in result
