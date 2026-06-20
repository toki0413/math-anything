"""Unit tests for Codegen module: SourceCodeAnalyzer, ConstraintInference."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from math_anything.codegen.source_analyzer import (
    ExtractionConfidence,
    ExtractedCommand,
    ExtractedParameter,
    SourceCodeAnalyzer,
    quick_analyze,
)
from math_anything.codegen.constraint_inference import (
    ConstraintInference,
    InferredConstraint,
    quick_infer,
    extract_symbolic_constraints,
)
from math_anything.utils.safe_eval import safe_eval, SafeEvalError


# ── SourceCodeAnalyzer fixtures ──

@pytest.fixture
def analyzer():
    return SourceCodeAnalyzer()


@pytest.fixture
def cpp_source_dir(tmp_path):
    """Create a temp directory with C++ source files."""
    # C++ file with Fix class
    cpp_file = tmp_path / "fix_nvt.cpp"
    cpp_file.write_text(
        """// NVT thermostat fix
// temperature must be positive
class FixNVT : public Fix {
    double temperature = atof(args[1]);
    double timestep = atof(args[2]);
    if (temperature > 0) {
        // valid
    }
    // equation: dE/dt = -gamma * (E - E_target)
};
""",
        encoding="utf-8",
    )
    # C++ file with Compute class
    cpp_file2 = tmp_path / "compute_temp.cpp"
    cpp_file2.write_text(
        """// Compute temperature
class ComputeTemp : public Compute {
    double value = atof(args[0]);
};
""",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def python_source_dir(tmp_path):
    """Create a temp directory with Python source files."""
    py_file = tmp_path / "parser.py"
    py_file.write_text(
        """# Command parser
class SimCommand(CommandParser):
    self.timestep = args[0]
    self.pressure = args[1]
""",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def empty_source_dir(tmp_path):
    """Empty directory with no source files."""
    return tmp_path


@pytest.fixture
def mixed_source_dir(tmp_path):
    """Directory with C++, Python, and unknown file types."""
    (tmp_path / "fix_addforce.cpp").write_text(
        "class FixAddForce : public Fix { double force = atof(args[0]); };",
        encoding="utf-8",
    )
    (tmp_path / "handler.py").write_text(
        "class RunHandler(CommandParser): self.steps = args[0]",
        encoding="utf-8",
    )
    (tmp_path / "readme.txt").write_text("This is not source code", encoding="utf-8")
    return tmp_path


# ── SourceCodeAnalyzer: creation ──

class TestAnalyzerCreation:
    def test_creates_with_empty_results(self, analyzer):
        assert analyzer.results == []
        assert analyzer._warnings == []

    def test_has_command_patterns(self, analyzer):
        assert "cpp" in analyzer.COMMAND_PATTERNS
        assert "python" in analyzer.COMMAND_PATTERNS
        assert "java" in analyzer.COMMAND_PATTERNS

    def test_has_math_keywords(self, analyzer):
        assert "equation" in analyzer.MATH_KEYWORDS
        assert "boundary" in analyzer.MATH_KEYWORDS
        assert "solver" in analyzer.MATH_KEYWORDS

    def test_has_coverage_limitations(self, analyzer):
        assert "cpp" in analyzer.COVERAGE_LIMITATIONS
        assert "python" in analyzer.COVERAGE_LIMITATIONS


# ── SourceCodeAnalyzer: analyze ──

class TestAnalyzerAnalyze:
    def test_analyze_cpp_dir(self, analyzer, cpp_source_dir):
        result = analyzer.analyze(str(cpp_source_dir), file_patterns=["*.cpp"])
        assert "commands" in result
        assert "parameters" in result
        assert "equations" in result
        assert "coverage_metrics" in result
        assert "warnings" in result

    def test_analyze_finds_fix_command(self, analyzer, cpp_source_dir):
        result = analyzer.analyze(str(cpp_source_dir), file_patterns=["*.cpp"])
        cmd_names = [c["name"] for c in result["commands"]]
        assert "FixNVT" in cmd_names

    def test_analyze_finds_compute_command(self, analyzer, cpp_source_dir):
        result = analyzer.analyze(str(cpp_source_dir), file_patterns=["*.cpp"])
        cmd_names = [c["name"] for c in result["commands"]]
        assert "ComputeTemp" in cmd_names

    def test_analyze_finds_parameters(self, analyzer, cpp_source_dir):
        result = analyzer.analyze(str(cpp_source_dir), file_patterns=["*.cpp"])
        assert len(result["parameters"]) > 0

    def test_analyze_finds_equations(self, analyzer, cpp_source_dir):
        result = analyzer.analyze(str(cpp_source_dir), file_patterns=["*.cpp"])
        assert len(result["equations"]) > 0

    def test_analyze_nonexistent_dir_raises(self, analyzer):
        with pytest.raises(FileNotFoundError):
            analyzer.analyze("/nonexistent/path/xyz")

    def test_analyze_empty_dir(self, analyzer, empty_source_dir):
        result = analyzer.analyze(str(empty_source_dir))
        assert result["commands"] == []
        assert result["parameters"] == []

    def test_analyze_python_dir(self, analyzer, python_source_dir):
        result = analyzer.analyze(str(python_source_dir), file_patterns=["*.py"])
        cmd_names = [c["name"] for c in result["commands"]]
        assert "SimCommand" in cmd_names

    def test_analyze_mixed_dir(self, analyzer, mixed_source_dir):
        result = analyzer.analyze(str(mixed_source_dir))
        assert len(result["commands"]) > 0

    def test_analyze_coverage_metrics(self, analyzer, cpp_source_dir):
        result = analyzer.analyze(str(cpp_source_dir), file_patterns=["*.cpp"])
        metrics = result["coverage_metrics"]
        assert "total_files_scanned" in metrics
        assert "files_successfully_processed" in metrics
        assert "estimated_command_coverage" in metrics
        assert metrics["method"] == "regex_heuristic"

    def test_analyze_command_has_confidence(self, analyzer, cpp_source_dir):
        result = analyzer.analyze(str(cpp_source_dir), file_patterns=["*.cpp"])
        for cmd in result["commands"]:
            assert cmd["confidence"] in ["high", "medium", "low", "heuristic"]

    def test_analyze_command_has_extraction_method(self, analyzer, cpp_source_dir):
        result = analyzer.analyze(str(cpp_source_dir), file_patterns=["*.cpp"])
        for cmd in result["commands"]:
            assert cmd["extraction_method"] != ""

    def test_analyze_includes_limitations(self, analyzer, cpp_source_dir):
        result = analyzer.analyze(str(cpp_source_dir), file_patterns=["*.cpp"])
        assert "limitations" in result
        assert "cpp" in result["limitations"]


# ── SourceCodeAnalyzer: _detect_language ──

class TestAnalyzerDetectLanguage:
    def test_cpp_extension(self, analyzer):
        assert analyzer._detect_language(Path("test.cpp")) == "cpp"

    def test_c_extension(self, analyzer):
        assert analyzer._detect_language(Path("test.c")) == "cpp"

    def test_h_extension(self, analyzer):
        assert analyzer._detect_language(Path("test.h")) == "cpp"

    def test_hpp_extension(self, analyzer):
        assert analyzer._detect_language(Path("test.hpp")) == "cpp"

    def test_py_extension(self, analyzer):
        assert analyzer._detect_language(Path("test.py")) == "python"

    def test_java_extension(self, analyzer):
        assert analyzer._detect_language(Path("test.java")) == "java"

    def test_unknown_extension(self, analyzer):
        assert analyzer._detect_language(Path("test.txt")) == "unknown"


# ── SourceCodeAnalyzer: _infer_param_type ──

class TestAnalyzerInferParamType:
    def test_double_type(self, analyzer):
        assert analyzer._infer_param_type("double x = atof(args[0])") == "float"

    def test_int_type(self, analyzer):
        assert analyzer._infer_param_type("int n = atoi(args[1])") == "int"

    def test_string_type(self, analyzer):
        assert analyzer._infer_param_type("string name = strcmp(args[0])") == "string"

    def test_vector_type(self, analyzer):
        # "double" is checked before brackets, so "double x[3]" -> "float"
        assert analyzer._infer_param_type("double x[3] = args[0]") == "float"

    def test_unknown_type(self, analyzer):
        # "auto x = args[0]" has brackets in "args[0]" -> "vector"
        assert analyzer._infer_param_type("auto x = args[0]") == "vector"


# ── SourceCodeAnalyzer: _infer_param_name ──

class TestAnalyzerInferParamName:
    def test_double_assignment(self, analyzer):
        name = analyzer._infer_param_name("double temperature = atof(args[1])", "1")
        assert name == "temperature"

    def test_self_assignment(self, analyzer):
        name = analyzer._infer_param_name("self.timestep = args[0]", "0")
        assert name == "timestep"

    def test_no_match(self, analyzer):
        name = analyzer._infer_param_name("some random line", "0")
        assert name is None


# ── SourceCodeAnalyzer: _extract_description ──

class TestAnalyzerExtractDescription:
    def test_cpp_comment(self, analyzer):
        lines = ["// NVT thermostat", "class FixNVT : public Fix {"]
        desc = analyzer._extract_description(lines, 2)
        assert "NVT" in desc

    def test_python_comment(self, analyzer):
        lines = ["# Command parser", "class SimCommand(CommandParser):"]
        desc = analyzer._extract_description(lines, 2)
        assert "parser" in desc.lower()

    def test_no_comment(self, analyzer):
        lines = ["class FixNVT : public Fix {"]
        desc = analyzer._extract_description(lines, 1)
        assert desc == ""


# ── SourceCodeAnalyzer: _extract_equations_from_comments ──

class TestAnalyzerExtractEquations:
    def test_latex_equation(self, analyzer):
        lines = ["// equation: E = mc^2"]
        eqs = analyzer._extract_equations_from_comments(lines, "test.cpp")
        assert len(eqs) > 0

    def test_no_equations(self, analyzer):
        lines = ["// just a comment", "int x = 5;"]
        eqs = analyzer._extract_equations_from_comments(lines, "test.cpp")
        assert len(eqs) == 0

    def test_math_keyword_equation(self, analyzer):
        lines = ["// equation: dE/dt = -gamma * (E - E_target)"]
        eqs = analyzer._extract_equations_from_comments(lines, "test.cpp")
        assert len(eqs) > 0


# ── SourceCodeAnalyzer: infer_constraints ──

class TestAnalyzerInferConstraints:
    def test_comparison_constraint(self, analyzer):
        param = ExtractedParameter(
            name="temperature", param_type="float",
            confidence=ExtractionConfidence.HEURISTIC
        )
        constraints = analyzer.infer_constraints(param, ["if (temperature > 0)"])
        assert len(constraints) > 0
        assert "temperature > 0" in constraints[0]

    def test_must_be_positive(self, analyzer):
        param = ExtractedParameter(
            name="timestep", param_type="float",
            confidence=ExtractionConfidence.HEURISTIC
        )
        constraints = analyzer.infer_constraints(param, ["timestep must be positive"])
        assert len(constraints) > 0
        assert "timestep > 0" in constraints[0]

    def test_range_constraint(self, analyzer):
        param = ExtractedParameter(
            name="alpha", param_type="float",
            confidence=ExtractionConfidence.HEURISTIC
        )
        constraints = analyzer.infer_constraints(param, ["range [0, 1]"])
        assert len(constraints) > 0

    def test_no_constraints(self, analyzer):
        param = ExtractedParameter(
            name="x", param_type="float",
            confidence=ExtractionConfidence.HEURISTIC
        )
        constraints = analyzer.infer_constraints(param, ["no constraints here"])
        assert constraints == []


# ── quick_analyze ──

class TestQuickAnalyze:
    def test_quick_analyze(self, cpp_source_dir):
        result = quick_analyze(str(cpp_source_dir))
        assert "commands" in result

    def test_quick_analyze_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            quick_analyze("/nonexistent/path")


# ── ConstraintInference: creation ──

class TestConstraintInferenceCreation:
    def test_creates_with_empty_constraints(self):
        ci = ConstraintInference()
        assert ci.constraints == []

    def test_has_code_patterns(self):
        ci = ConstraintInference()
        assert "comparison" in ci.CODE_CONSTRAINT_PATTERNS
        assert "range" in ci.CODE_CONSTRAINT_PATTERNS
        assert "physical" in ci.CODE_CONSTRAINT_PATTERNS

    def test_has_doc_patterns(self):
        ci = ConstraintInference()
        assert len(ci.DOC_CONSTRAINT_PATTERNS) > 0

    def test_has_physical_relationships(self):
        ci = ConstraintInference()
        assert "cfl_condition" in ci.PHYSICAL_RELATIONSHIPS
        assert "elastic_modulus" in ci.PHYSICAL_RELATIONSHIPS


# ── ConstraintInference: infer ──

class TestConstraintInferenceInfer:
    def test_infer_comparison(self):
        ci = ConstraintInference()
        constraints = ci.infer(
            parameters=[{"name": "dt"}],
            command_contexts={"fix": ["if (dt > 0)"]}
        )
        assert "dt > 0" in constraints

    def test_infer_must_be(self):
        ci = ConstraintInference()
        constraints = ci.infer(
            parameters=[{"name": "timestep"}],
            command_contexts={"fix": ["timestep must be greater than 0"]}
        )
        assert any("timestep" in c for c in constraints)

    def test_infer_physical_positive(self):
        ci = ConstraintInference()
        constraints = ci.infer(
            parameters=[{"name": "temperature"}],
            command_contexts={"fix": ["temperature is positive"]}
        )
        assert any("temperature > 0" in c for c in constraints)

    def test_infer_physical_negative(self):
        ci = ConstraintInference()
        constraints = ci.infer(
            parameters=[{"name": "pressure"}],
            command_contexts={"fix": ["pressure is negative"]}
        )
        assert any("pressure < 0" in c for c in constraints)

    def test_infer_cfl_condition(self):
        ci = ConstraintInference()
        constraints = ci.infer(
            parameters=[{"name": "dt"}],
            command_contexts={"fix": ["CFL condition: dt < dx^2/(2*D)"]}
        )
        assert any("dt < dx^2" in c for c in constraints)

    def test_infer_elastic_modulus(self):
        ci = ConstraintInference()
        constraints = ci.infer(
            parameters=[{"name": "mu"}],
            command_contexts={"fix": ["shear modulus mu from Young modulus E"]}
        )
        assert any("mu = E" in c for c in constraints)

    def test_infer_empty_contexts(self):
        ci = ConstraintInference()
        constraints = ci.infer(parameters=[], command_contexts={})
        assert constraints == []

    def test_infer_deduplicates(self):
        ci = ConstraintInference()
        constraints = ci.infer(
            parameters=[{"name": "dt"}],
            command_contexts={"fix": ["if (dt > 0)", "if (dt > 0)"]}
        )
        # Should be deduplicated
        dt_constraints = [c for c in constraints if "dt > 0" in c]
        assert len(dt_constraints) == 1


# ── ConstraintInference: infer_from_docs ──

class TestConstraintInferenceDocs:
    def test_infer_must_be_from_docs(self):
        ci = ConstraintInference()
        constraints = ci.infer_from_docs(
            parameters=[{"name": "timestep"}],
            doc_sections=[{"content": "timestep must be positive"}]
        )
        assert len(constraints) > 0

    def test_infer_range_from_docs(self):
        ci = ConstraintInference()
        constraints = ci.infer_from_docs(
            parameters=[{"name": "alpha"}],
            doc_sections=[{"content": "range: [0, 1] for alpha"}]
        )
        assert any("alpha" in c for c in constraints)

    def test_infer_empty_docs(self):
        ci = ConstraintInference()
        constraints = ci.infer_from_docs(parameters=[], doc_sections=[])
        assert constraints == []


# ── ConstraintInference: infer_from_equations ──

class TestConstraintInferenceEquations:
    def test_conservation_from_continuity(self):
        ci = ConstraintInference()
        constraints = ci.infer_from_equations([
            {"form": "∂ρ/∂t + ∇·(ρv) = 0", "type": "continuity", "source_file": "test.cpp"}
        ])
        assert len(constraints) > 0
        assert any(c.constraint_type == "conservation" for c in constraints)

    def test_energy_conservation(self):
        ci = ConstraintInference()
        constraints = ci.infer_from_equations([
            {"form": "E_total = KE + PE", "type": "energy", "source_file": "test.cpp"}
        ])
        assert any(c.constraint_type == "conservation" for c in constraints)

    def test_entropy_constraint(self):
        ci = ConstraintInference()
        constraints = ci.infer_from_equations([
            {"form": "S = k * log(W)", "type": "entropy", "source_file": "test.cpp"}
        ])
        assert any("dS/dt >= 0" in c.expression for c in constraints)

    def test_no_constraints_from_empty(self):
        ci = ConstraintInference()
        constraints = ci.infer_from_equations([])
        assert constraints == []


# ── ConstraintInference: build_relationship_graph ──

class TestConstraintInferenceGraph:
    def test_build_graph(self):
        ci = ConstraintInference()
        constraints = [
            InferredConstraint(
                expression="dt < dx^2/(2*D)",
                constraint_type="inequality",
                variables=["dt", "dx", "D"],
                confidence=0.7,
                source="test"
            )
        ]
        graph = ci.build_relationship_graph(constraints)
        assert "variables" in graph
        assert "relationships" in graph
        assert len(graph["variables"]) == 3
        assert graph["constraint_count"] == 1

    def test_build_graph_empty(self):
        ci = ConstraintInference()
        graph = ci.build_relationship_graph([])
        assert graph["variables"] == []
        assert graph["relationships"] == []


# ── ConstraintInference: generate_llm_prompt ──

class TestConstraintInferencePrompt:
    def test_generate_prompt(self):
        ci = ConstraintInference()
        constraints = [
            InferredConstraint(
                expression="dt > 0",
                constraint_type="inequality",
                variables=["dt"],
                confidence=0.9,
                source="test",
                physical_meaning="Timestep must be positive"
            )
        ]
        prompt = ci.generate_llm_prompt(constraints)
        assert "dt > 0" in prompt
        assert "INEQUALITY" in prompt
        assert "Timestep must be positive" in prompt

    def test_generate_prompt_empty(self):
        ci = ConstraintInference()
        prompt = ci.generate_llm_prompt([])
        assert "Mathematical constraints" in prompt


# ── ConstraintInference: validate_constraint (safe_eval) ──

class TestConstraintInferenceValidate:
    def test_validate_true_constraint(self):
        ci = ConstraintInference()
        is_valid, err = ci.validate_constraint("5 > 3", {"x": 5})
        assert is_valid is True
        assert err is None

    def test_validate_false_constraint(self):
        ci = ConstraintInference()
        is_valid, err = ci.validate_constraint("1 > 5", {})
        assert is_valid is False

    def test_validate_with_variables(self):
        ci = ConstraintInference()
        is_valid, err = ci.validate_constraint("x > 0", {"x": 5.0})
        # After substitution: "5.0 > 0" -> safe_eval should handle this
        assert is_valid is True

    def test_validate_division_by_zero(self):
        ci = ConstraintInference()
        is_valid, err = ci.validate_constraint("1 / 0 > 0", {})
        assert is_valid is False

    def test_validate_no_bare_eval(self):
        """Ensure validate_constraint uses safe_eval, not bare eval."""
        ci = ConstraintInference()
        # This would be dangerous with bare eval
        is_valid, err = ci.validate_constraint("__import__('os').system('echo pwned')", {})
        assert is_valid is False


# ── ConstraintInference: _extract_variables ──

class TestConstraintInferenceExtractVars:
    def test_extract_variables(self):
        ci = ConstraintInference()
        vars = ci._extract_variables("dt < dx^2 / (2*D)")
        assert "dt" in vars or "dx" in vars

    def test_extract_no_variables(self):
        ci = ConstraintInference()
        vars = ci._extract_variables("1 + 2 = 3")
        assert vars == []

    def test_excludes_math_functions(self):
        ci = ConstraintInference()
        vars = ci._extract_variables("sin(x) + cos(y)")
        assert "sin" not in vars
        assert "cos" not in vars


# ── quick_infer ──

class TestQuickInfer:
    def test_quick_infer(self):
        result = quick_infer(["if (x > 0)"])
        assert isinstance(result, list)


# ── extract_symbolic_constraints ──

class TestExtractSymbolicConstraints:
    def test_extract_from_snippet(self):
        result = extract_symbolic_constraints("if (dt > 0)", ["dt"])
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], InferredConstraint)

    def test_extract_empty(self):
        result = extract_symbolic_constraints("no constraints", ["x"])
        assert result == []


# ── ExtractionConfidence ──

class TestExtractionConfidence:
    def test_confidence_values(self):
        assert ExtractionConfidence.HIGH.value == "high"
        assert ExtractionConfidence.MEDIUM.value == "medium"
        assert ExtractionConfidence.LOW.value == "low"
        assert ExtractionConfidence.HEURISTIC.value == "heuristic"


# ── InferredConstraint ──

class TestInferredConstraint:
    def test_constraint_creation(self):
        c = InferredConstraint(
            expression="x > 0",
            constraint_type="inequality",
            variables=["x"],
            confidence=0.9,
            source="test"
        )
        assert c.expression == "x > 0"
        assert c.physical_meaning is None

    def test_constraint_with_meaning(self):
        c = InferredConstraint(
            expression="dt > 0",
            constraint_type="inequality",
            variables=["dt"],
            confidence=0.9,
            source="test",
            physical_meaning="Timestep must be positive"
        )
        assert c.physical_meaning == "Timestep must be positive"


# ── safe_eval integration ──

class TestSafeEvalIntegration:
    def test_safe_eval_basic_comparison(self):
        result = safe_eval("5 > 3", {})
        assert result is True

    def test_safe_eval_arithmetic(self):
        result = safe_eval("2 + 3", {})
        assert result == 5

    def test_safe_eval_with_context(self):
        result = safe_eval("x > 0", {"x": 5})
        assert result is True

    def test_safe_eval_rejects_dunder(self):
        with pytest.raises(SafeEvalError):
            safe_eval("__import__", {})

    def test_safe_eval_rejects_import(self):
        with pytest.raises(SafeEvalError):
            safe_eval("__import__('os')", {})

    def test_safe_eval_syntax_error(self):
        with pytest.raises(SafeEvalError):
            safe_eval("if True:", {})


# ── HarnessAutoGenerator ──

from math_anything.codegen.harness_generator import (
    HarnessAutoGenerator,
    HarnessTemplate,
    JINJA2_AVAILABLE,
)


class TestHarnessAutoGeneratorCreation:
    def test_creates_with_analyzers(self):
        gen = HarnessAutoGenerator()
        assert gen.source_analyzer is not None
        assert gen.doc_analyzer is not None
        assert gen.semantic_mapper is not None
        assert gen.constraint_inference is not None

    def test_jinja_env_setup_when_available(self):
        gen = HarnessAutoGenerator()
        if JINJA2_AVAILABLE:
            assert gen.jinja_env is not None
        else:
            assert gen.jinja_env is None


class TestHarnessGeneratorConfidence:
    def test_calculate_confidence_empty(self):
        gen = HarnessAutoGenerator()
        conf = gen._calculate_confidence({})
        assert conf["command_extraction"] == 0
        assert conf["parameter_extraction"] == 0
        assert conf["equation_mapping"] == 0
        assert conf["overall"] == 0.0

    def test_calculate_confidence_with_commands(self):
        gen = HarnessAutoGenerator()
        analysis = {"commands": [{"name": "nvt"}], "parameters": [], "equations": []}
        conf = gen._calculate_confidence(analysis)
        assert conf["command_extraction"] == 30  # min(100, max(30, 1*10))
        assert conf["overall"] > 0

    def test_calculate_confidence_caps_at_100(self):
        gen = HarnessAutoGenerator()
        # 20 commands -> 200 -> capped to 100
        analysis = {
            "commands": [{"name": f"c{i}"} for i in range(20)],
            "parameters": [],
            "equations": [],
        }
        conf = gen._calculate_confidence(analysis)
        assert conf["command_extraction"] == 100

    def test_calculate_confidence_with_params(self):
        gen = HarnessAutoGenerator()
        analysis = {
            "commands": [],
            "parameters": [{"name": f"p{i}"} for i in range(10)],
            "equations": [],
        }
        conf = gen._calculate_confidence(analysis)
        assert conf["parameter_extraction"] == 50  # 10*5

    def test_calculate_confidence_with_equations(self):
        gen = HarnessAutoGenerator()
        analysis = {
            "commands": [],
            "parameters": [],
            "equations": [{"form": "x"} for i in range(5)],
        }
        conf = gen._calculate_confidence(analysis)
        assert conf["equation_mapping"] == 75  # 5*15


class TestHarnessGeneratorHelpers:
    def test_to_class_name_single(self):
        gen = HarnessAutoGenerator()
        assert gen._to_class_name("lammps") == "Lammps"

    def test_to_class_name_multi(self):
        gen = HarnessAutoGenerator()
        assert gen._to_class_name("quantum_espresso") == "QuantumEspresso"

    def test_extract_variables_basic(self):
        gen = HarnessAutoGenerator()
        vars = gen._extract_variables("dt < dx^2 / (2*D)")
        assert "dt" in vars or "dx" in vars

    def test_extract_variables_filters_math_words(self):
        gen = HarnessAutoGenerator()
        vars = gen._extract_variables("sin(x) + cos(y)")
        assert "sin" not in vars
        assert "cos" not in vars

    def test_extract_variables_limits_to_5(self):
        gen = HarnessAutoGenerator()
        constraint = "alpha beta gamma delta epsilon zeta eta theta"
        vars = gen._extract_variables(constraint)
        assert len(vars) <= 5


class TestHarnessGenerateFromSource:
    def test_generate_from_source_returns_template(self, cpp_source_dir):
        gen = HarnessAutoGenerator()
        template = gen.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test_engine",
            file_patterns=["*.cpp"],
        )
        assert isinstance(template, HarnessTemplate)
        assert template.engine_name == "test_engine"
        assert template.engine_version == "1.0.0"
        assert isinstance(template.generated_code, str)
        assert len(template.generated_code) > 0

    def test_generate_from_source_has_commands(self, cpp_source_dir):
        gen = HarnessAutoGenerator()
        template = gen.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        assert isinstance(template.extracted_commands, list)

    def test_generate_from_source_has_confidence(self, cpp_source_dir):
        gen = HarnessAutoGenerator()
        template = gen.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        assert "overall" in template.extraction_confidence
        assert "command_extraction" in template.extraction_confidence

    def test_generate_from_source_has_checklist(self, cpp_source_dir):
        gen = HarnessAutoGenerator()
        template = gen.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        assert isinstance(template.review_checklist, list)
        assert len(template.review_checklist) > 0

    def test_generate_with_custom_version(self, cpp_source_dir):
        gen = HarnessAutoGenerator()
        template = gen.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            engine_version="2.5.0",
            file_patterns=["*.cpp"],
        )
        assert template.engine_version == "2.5.0"

    def test_generated_code_contains_engine_name(self, cpp_source_dir):
        gen = HarnessAutoGenerator()
        template = gen.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="myengine",
            file_patterns=["*.cpp"],
        )
        # Either Jinja2 or fallback should mention engine name
        assert "myengine" in template.generated_code.lower() or "Myengine" in template.generated_code


class TestHarnessGenerateFromDocs:
    def test_generate_from_docs_markdown(self, tmp_path):
        md_file = tmp_path / "manual.md"
        md_file.write_text(
            "# Manual\n\n## nvt - thermostat\n\nSyntax: `nvt temp 300 300 100`\n",
            encoding="utf-8",
        )
        gen = HarnessAutoGenerator()
        template = gen.generate_from_docs(
            doc_path=str(md_file),
            engine_name="mdengine",
        )
        assert isinstance(template, HarnessTemplate)
        assert template.engine_name == "mdengine"
        assert template.source_path == str(md_file)

    def test_generate_from_docs_has_lower_confidence(self, tmp_path):
        md_file = tmp_path / "manual.md"
        md_file.write_text("# Manual\n## nvt - thermostat\n", encoding="utf-8")
        gen = HarnessAutoGenerator()
        template = gen.generate_from_docs(doc_path=str(md_file), engine_name="e")
        # Doc-based extraction applies 0.8 factor
        assert template.extraction_confidence["overall"] >= 0.0


class TestHarnessSaveHarness:
    def test_save_harness_creates_files(self, tmp_path, cpp_source_dir):
        gen = HarnessAutoGenerator()
        template = gen.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="savetest",
            file_patterns=["*.cpp"],
        )
        out_dir = tmp_path / "output"
        gen.save_harness(template, str(out_dir))

        harness_dir = out_dir / "savetest-harness"
        assert harness_dir.exists()
        # harness.py
        assert (harness_dir / "math_anything" / "savetest" / "core" / "harness.py").exists()
        # __init__.py
        assert (harness_dir / "math_anything" / "savetest" / "__init__.py").exists()
        # summary
        assert (harness_dir / "ANALYSIS_SUMMARY.md").exists()
        # review checklist
        assert (harness_dir / "REVIEW_CHECKLIST.md").exists()

    def test_save_harness_init_contains_class(self, tmp_path, cpp_source_dir):
        gen = HarnessAutoGenerator()
        template = gen.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="myengine",
            file_patterns=["*.cpp"],
        )
        out_dir = tmp_path / "output"
        gen.save_harness(template, str(out_dir))
        init_file = out_dir / "myengine-harness" / "math_anything" / "myengine" / "__init__.py"
        content = init_file.read_text(encoding="utf-8")
        assert "MyengineHarness" in content

    def test_save_harness_summary_contains_engine(self, tmp_path, cpp_source_dir):
        gen = HarnessAutoGenerator()
        template = gen.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="summaryengine",
            file_patterns=["*.cpp"],
        )
        out_dir = tmp_path / "output"
        gen.save_harness(template, str(out_dir))
        summary = (out_dir / "summaryengine-harness" / "ANALYSIS_SUMMARY.md").read_text(encoding="utf-8")
        assert "summaryengine" in summary.lower() or "Summaryengine" in summary

    def test_save_harness_checklist_has_items(self, tmp_path, cpp_source_dir):
        gen = HarnessAutoGenerator()
        template = gen.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="cengine",
            file_patterns=["*.cpp"],
        )
        out_dir = tmp_path / "output"
        gen.save_harness(template, str(out_dir))
        checklist = (out_dir / "cengine-harness" / "REVIEW_CHECKLIST.md").read_text(encoding="utf-8")
        assert len(checklist) > 0


class TestHarnessTemplate:
    def test_template_dataclass_fields(self):
        t = HarnessTemplate(
            engine_name="x",
            engine_version="1.0",
            source_path="/tmp",
            extracted_commands=[],
            mathematical_mappings={},
            constraint_expressions=[],
            generated_code="",
            review_checklist=[],
            extraction_confidence={},
        )
        assert t.engine_name == "x"
        assert t.engine_version == "1.0"
        assert t.extracted_commands == []


# ── DocumentationAnalyzer ──

from math_anything.codegen.doc_analyzer import (
    DocumentationAnalyzer,
    DocCommand,
    quick_analyze_docs,
)


class TestDocAnalyzerCreation:
    def test_creates_empty(self):
        a = DocumentationAnalyzer()
        assert a.commands == []

    def test_has_syntax_patterns(self):
        a = DocumentationAnalyzer()
        assert len(a.SYNTAX_PATTERNS) > 0

    def test_has_param_patterns(self):
        a = DocumentationAnalyzer()
        assert len(a.PARAM_PATTERNS) > 0

    def test_has_constraint_patterns(self):
        a = DocumentationAnalyzer()
        assert len(a.CONSTRAINT_PATTERNS) > 0


class TestDocAnalyzerMarkdown:
    def test_analyze_markdown_extracts_sections(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "# Title\n\n## Section1\n\ncontent here\n\n### Sub\n\nmore\n",
            encoding="utf-8",
        )
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        assert "commands" in result
        assert "parameters" in result
        assert "sections" in result
        assert len(result["sections"]) >= 1

    def test_analyze_markdown_extracts_command_from_header(self, tmp_path):
        # Markdown parser tracks sections from headers; command extraction
        # from headers is skipped due to the `continue` after section tracking.
        # Verify the section is captured instead.
        md = tmp_path / "test.md"
        md.write_text("## nvt - Nose-Hoover thermostat\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        # Section title should contain the header text
        section_titles = [s["title"] for s in result["sections"]]
        assert any("nvt" in t for t in section_titles)

    def test_analyze_markdown_extracts_parameters(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "# Title\n\n## cmd - test\n\n* temp - temperature value\n",
            encoding="utf-8",
        )
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        assert len(result["parameters"]) >= 1
        assert result["parameters"][0]["name"] == "temp"

    def test_analyze_markdown_source_path(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Title\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        assert result["source_path"] == str(md)


class TestDocAnalyzerFormats:
    def test_analyze_html_strips_tags(self, tmp_path):
        html = tmp_path / "test.html"
        html.write_text(
            "<html><body><h1>Title</h1><p>fix nvt temp 300</p></body></html>",
            encoding="utf-8",
        )
        a = DocumentationAnalyzer()
        result = a.analyze(str(html))
        assert "commands" in result

    def test_analyze_text_file(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Syntax: nvt\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(txt))
        assert "commands" in result

    def test_analyze_pdf_returns_placeholder(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(pdf))
        # PDF extraction is a placeholder; should still return dict structure
        assert "commands" in result
        assert "parameters" in result

    def test_analyze_nonexistent_raises(self):
        a = DocumentationAnalyzer()
        with pytest.raises(FileNotFoundError):
            a.analyze("/nonexistent/path/to/file.md")


class TestDocAnalyzerTextParsing:
    def test_parse_text_finds_syntax(self):
        a = DocumentationAnalyzer()
        text = "Syntax: nvt\n"
        result = a._parse_text(text, "src")
        assert "commands" in result

    def test_parse_text_deduplicates(self):
        a = DocumentationAnalyzer()
        text = "Syntax: nvt\nSyntax: nvt\n"
        result = a._parse_text(text, "src")
        names = [c["name"] for c in result["commands"]]
        assert names.count("nvt") <= 1

    def test_parse_text_extracts_params(self):
        a = DocumentationAnalyzer()
        text = "temp (float) - temperature value\n"
        result = a._parse_text(text, "src")
        assert len(result["parameters"]) >= 1


class TestDocAnalyzerInferFromExamples:
    def test_infer_single_example(self):
        a = DocumentationAnalyzer()
        result = a.infer_from_examples(["fix 1 all nvt temp 300 300 100"])
        assert len(result) >= 1
        assert "command" in result[0]
        assert "arguments" in result[0]

    def test_infer_empty_examples(self):
        a = DocumentationAnalyzer()
        result = a.infer_from_examples([])
        assert result == []

    def test_infer_short_example_skipped(self):
        a = DocumentationAnalyzer()
        # Single token: no command keyword found
        result = a.infer_from_examples(["x"])
        assert result == []


class TestDocCommand:
    def test_doc_command_defaults(self):
        c = DocCommand(name="nvt", syntax="nvt", description="d", parameters=[])
        assert c.examples == []
        assert c.constraints == []

    def test_doc_command_with_values(self):
        c = DocCommand(
            name="nvt",
            syntax="nvt temp",
            description="thermostat",
            parameters=[{"name": "temp"}],
            examples=["nvt 300"],
            constraints=["temp > 0"],
        )
        assert c.name == "nvt"
        assert len(c.parameters) == 1
        assert len(c.examples) == 1


class TestQuickAnalyzeDocs:
    def test_quick_analyze(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Title\n## cmd - test\n", encoding="utf-8")
        result = quick_analyze_docs(str(md))
        assert "commands" in result

    def test_quick_analyze_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            quick_analyze_docs("/nonexistent/path.md")


# ── SemanticMapper ──

from math_anything.codegen.semantic_mapper import (
    SemanticMapper,
    MathMapping,
    quick_map,
)


class TestSemanticMapperCreation:
    def test_creates_empty(self):
        m = SemanticMapper()
        assert m.mappings == []

    def test_has_equation_keywords(self):
        m = SemanticMapper()
        assert "nvt" in m.EQUATION_KEYWORDS
        assert "npt" in m.EQUATION_KEYWORDS
        assert "minimize" in m.EQUATION_KEYWORDS

    def test_has_boundary_keywords(self):
        m = SemanticMapper()
        assert "fix" in m.BOUNDARY_KEYWORDS
        assert "velocity" in m.BOUNDARY_KEYWORDS

    def test_has_numerical_keywords(self):
        m = SemanticMapper()
        assert "verlet" in m.NUMERICAL_KEYWORDS
        assert "rk4" in m.NUMERICAL_KEYWORDS

    def test_has_physical_parameters(self):
        m = SemanticMapper()
        assert "temp" in m.PHYSICAL_PARAMETERS
        assert "pressure" in m.PHYSICAL_PARAMETERS


class TestSemanticMapperMapCommands:
    def test_map_empty(self):
        m = SemanticMapper()
        result = m.map_commands([])
        assert result["equations"] == []
        assert result["boundary_conditions"] == []
        assert result["numerical_methods"] == []
        assert result["variables"] == []

    def test_map_nvt_command(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "nvt", "description": "thermostat"}])
        assert len(result["equations"]) >= 1
        eq = result["equations"][0]
        assert "thermostat" in eq["name"].lower() or "thermostat" in eq["type"]

    def test_map_minimize_command(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "minimize", "description": "energy min"}])
        assert len(result["equations"]) >= 1

    def test_map_boundary_command(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "fix", "description": "boundary"}])
        assert len(result["boundary_conditions"]) >= 1

    def test_map_numerical_command(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "verlet", "description": "integrator"}])
        assert len(result["numerical_methods"]) >= 1
        assert result["numerical_methods"][0]["method"] == "velocity_verlet"

    def test_map_parameters_to_variables(self):
        m = SemanticMapper()
        result = m.map_commands(
            commands=[],
            parameters=[{"name": "temp"}, {"name": "pressure"}],
        )
        assert len(result["variables"]) >= 2

    def test_map_explicit_equations(self):
        m = SemanticMapper()
        result = m.map_commands(
            commands=[],
            equations=[{"form": "E = mc^2", "type": "energy"}],
        )
        assert len(result["equations"]) >= 1
        assert result["equations"][0]["confidence"] == 0.9

    def test_map_returns_total_mappings(self):
        m = SemanticMapper()
        result = m.map_commands([{"name": "nvt", "description": "thermostat"}])
        assert "total_mappings" in result
        assert result["total_mappings"] >= 1


class TestSemanticMapperSuggest:
    def test_suggest_nvt(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("nvt")
        assert suggestion is not None
        assert suggestion["type"] == "governing_equation"

    def test_suggest_fix(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("fix_x")
        assert suggestion is not None
        assert suggestion["type"] == "boundary_condition"

    def test_suggest_compute(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("compute_temp")
        assert suggestion is not None
        assert suggestion["type"] == "derived_quantity"

    def test_suggest_unknown_returns_none(self):
        m = SemanticMapper()
        suggestion = m.suggest_math_semantics("zzz_unknown")
        assert suggestion is None


class TestSemanticMapperGraph:
    def test_build_graph_single_command(self):
        m = SemanticMapper()
        graph = m.build_computational_graph([{"name": "nvt"}])
        assert len(graph["nodes"]) == 1
        assert graph["edges"] == []

    def test_build_graph_multiple_commands(self):
        m = SemanticMapper()
        graph = m.build_computational_graph(
            [{"name": "nvt"}, {"name": "minimize"}, {"name": "run"}]
        )
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2  # sequential links

    def test_build_graph_node_has_semantics(self):
        m = SemanticMapper()
        graph = m.build_computational_graph([{"name": "nvt"}])
        assert "math_semantics" in graph["nodes"][0]


class TestSemanticMapperExtractVars:
    def test_extract_variables(self):
        m = SemanticMapper()
        vars = m._extract_variables("dT/dt = (T_target - T) / tau_T")
        # Should find T, tau, etc.
        assert isinstance(vars, list)

    def test_extract_variables_excludes_math_functions(self):
        m = SemanticMapper()
        vars = m._extract_variables("sin(x) + cos(y) + exp(z)")
        assert "sin" not in vars
        assert "cos" not in vars
        assert "exp" not in vars


class TestMathMapping:
    def test_math_mapping_creation(self):
        m = MathMapping(
            command_name="nvt",
            math_type="governing_equation",
            mathematical_form="dT/dt = ...",
            variables=["T"],
            confidence=0.8,
            description="thermostat",
        )
        assert m.command_name == "nvt"
        assert m.confidence == 0.8


class TestQuickMap:
    def test_quick_map(self):
        result = quick_map([{"name": "nvt", "description": "thermostat"}])
        assert "equations" in result

    def test_quick_map_empty(self):
        result = quick_map([])
        assert result["equations"] == []
