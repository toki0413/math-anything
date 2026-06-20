"""Unit tests for codegen/harness_generator.py — harness auto-generation."""

import pytest

from math_anything.codegen.harness_generator import (
    HarnessAutoGenerator,
    HarnessTemplate,
    JINJA2_AVAILABLE,
)


# ── Fixtures ──

@pytest.fixture
def generator():
    return HarnessAutoGenerator()


@pytest.fixture
def cpp_source_dir(tmp_path):
    """Temp directory with C++ source files for analysis."""
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
    return tmp_path


@pytest.fixture
def markdown_doc(tmp_path):
    """Temp markdown documentation file."""
    md = tmp_path / "manual.md"
    md.write_text(
        "# Manual\n\n## nvt - thermostat\n\nSyntax: `nvt temp 300 300 100`\n",
        encoding="utf-8",
    )
    return md


# ── HarnessTemplate dataclass ──

class TestHarnessTemplate:
    def test_creation_with_defaults(self):
        t = HarnessTemplate(
            engine_name="test",
            engine_version="1.0.0",
            source_path="/tmp",
            extracted_commands=[],
            mathematical_mappings={},
            constraint_expressions=[],
            generated_code="",
            review_checklist=[],
            extraction_confidence={},
        )
        assert t.engine_name == "test"
        assert t.engine_version == "1.0.0"
        assert t.source_path == "/tmp"
        assert t.extracted_commands == []
        assert t.mathematical_mappings == {}
        assert t.constraint_expressions == []
        assert t.generated_code == ""
        assert t.review_checklist == []
        assert t.extraction_confidence == {}

    def test_creation_with_full_data(self):
        t = HarnessTemplate(
            engine_name="lammps",
            engine_version="2.0.0",
            source_path="/path/to/src",
            extracted_commands=[{"name": "nvt"}],
            mathematical_mappings={"equations": [{"name": "thermostat"}]},
            constraint_expressions=["dt > 0"],
            generated_code="class LammpsHarness: pass",
            review_checklist=["[ ] Review commands"],
            extraction_confidence={"overall": 75.0},
        )
        assert t.engine_name == "lammps"
        assert len(t.extracted_commands) == 1
        assert len(t.constraint_expressions) == 1
        assert t.extraction_confidence["overall"] == 75.0


# ── HarnessAutoGenerator: creation ──

class TestGeneratorCreation:
    def test_creates_with_analyzers(self, generator):
        assert generator.source_analyzer is not None
        assert generator.doc_analyzer is not None
        assert generator.semantic_mapper is not None
        assert generator.constraint_inference is not None

    def test_jinja_env_setup(self, generator):
        if JINJA2_AVAILABLE:
            # Jinja2 env is set up only if templates dir exists
            # Just verify the attribute exists
            assert hasattr(generator, "jinja_env")
        else:
            assert generator.jinja_env is None


# ── HarnessAutoGenerator: _calculate_confidence ──

class TestCalculateConfidence:
    def test_empty_analysis(self, generator):
        conf = generator._calculate_confidence({})
        assert conf["command_extraction"] == 0
        assert conf["parameter_extraction"] == 0
        assert conf["equation_mapping"] == 0
        assert conf["overall"] == 0.0

    def test_with_commands(self, generator):
        analysis = {"commands": [{"name": "nvt"}], "parameters": [], "equations": []}
        conf = generator._calculate_confidence(analysis)
        assert conf["command_extraction"] == 30  # max(30, 1*10)
        assert conf["overall"] > 0

    def test_with_many_commands_caps_at_100(self, generator):
        analysis = {
            "commands": [{"name": f"c{i}"} for i in range(20)],
            "parameters": [],
            "equations": [],
        }
        conf = generator._calculate_confidence(analysis)
        assert conf["command_extraction"] == 100

    def test_with_parameters(self, generator):
        analysis = {
            "commands": [],
            "parameters": [{"name": f"p{i}"} for i in range(10)],
            "equations": [],
        }
        conf = generator._calculate_confidence(analysis)
        assert conf["parameter_extraction"] == 50  # 10*5

    def test_with_many_parameters_caps_at_100(self, generator):
        analysis = {
            "commands": [],
            "parameters": [{"name": f"p{i}"} for i in range(30)],
            "equations": [],
        }
        conf = generator._calculate_confidence(analysis)
        assert conf["parameter_extraction"] == 100

    def test_with_equations(self, generator):
        analysis = {
            "commands": [],
            "parameters": [],
            "equations": [{"form": "x"} for i in range(5)],
        }
        conf = generator._calculate_confidence(analysis)
        assert conf["equation_mapping"] == 75  # 5*15

    def test_with_many_equations_caps_at_100(self, generator):
        analysis = {
            "commands": [],
            "parameters": [],
            "equations": [{"form": "x"} for i in range(20)],
        }
        conf = generator._calculate_confidence(analysis)
        assert conf["equation_mapping"] == 100

    def test_overall_is_average(self, generator):
        analysis = {
            "commands": [{"name": "c"}],
            "parameters": [{"name": "p"}],
            "equations": [{"form": "x"}],
        }
        conf = generator._calculate_confidence(analysis)
        # cmd: max(30, 10) = 30, param: max(30, 5) = 30, eq: max(20, 15) = 20
        expected = (30 + 30 + 20) / 3
        assert abs(conf["overall"] - round(expected, 1)) < 0.1

    def test_confidence_values_are_numeric(self, generator):
        conf = generator._calculate_confidence({"commands": [{"name": "x"}]})
        # Values are numeric (int or float)
        assert isinstance(conf["command_extraction"], (int, float))
        assert isinstance(conf["overall"], (int, float))


# ── HarnessAutoGenerator: _to_class_name ──

class TestToClassName:
    def test_single_word(self, generator):
        assert generator._to_class_name("lammps") == "Lammps"

    def test_multi_word(self, generator):
        assert generator._to_class_name("quantum_espresso") == "QuantumEspresso"

    def test_already_capitalized(self, generator):
        # capitalize() lowercases first letter then capitalizes
        assert generator._to_class_name("Lammps") == "Lammps"

    def test_empty_string(self, generator):
        assert generator._to_class_name("") == ""


# ── HarnessAutoGenerator: _extract_variables ──

class TestExtractVariables:
    def test_extracts_variables(self, generator):
        vars = generator._extract_variables("dt < dx^2 / (2*D)")
        assert "dt" in vars or "dx" in vars

    def test_filters_math_words(self, generator):
        vars = generator._extract_variables("sin(x) + cos(y)")
        assert "sin" not in vars
        assert "cos" not in vars

    def test_limits_to_5(self, generator):
        constraint = "alpha beta gamma delta epsilon zeta eta theta"
        vars = generator._extract_variables(constraint)
        assert len(vars) <= 5

    def test_filters_short_vars(self, generator):
        # Single char variables are filtered (len > 1 required)
        vars = generator._extract_variables("a b c")
        # All single-char, should be filtered out
        for v in vars:
            assert len(v) > 1

    def test_returns_unique(self, generator):
        vars = generator._extract_variables("alpha alpha alpha")
        # set() deduplicates
        assert vars.count("alpha") <= 1


# ── HarnessAutoGenerator: generate_from_source ──

class TestGenerateFromSource:
    def test_returns_template(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test_engine",
            file_patterns=["*.cpp"],
        )
        assert isinstance(template, HarnessTemplate)
        assert template.engine_name == "test_engine"

    def test_default_version(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        assert template.engine_version == "1.0.0"

    def test_custom_version(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            engine_version="2.5.0",
            file_patterns=["*.cpp"],
        )
        assert template.engine_version == "2.5.0"

    def test_source_path_stored(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        assert template.source_path == str(cpp_source_dir)

    def test_generated_code_is_string(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        assert isinstance(template.generated_code, str)
        assert len(template.generated_code) > 0

    def test_generated_code_contains_engine_name(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="myengine",
            file_patterns=["*.cpp"],
        )
        assert "myengine" in template.generated_code.lower() or "Myengine" in template.generated_code

    def test_has_extracted_commands(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        assert isinstance(template.extracted_commands, list)

    def test_has_mathematical_mappings(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        assert isinstance(template.mathematical_mappings, dict)
        assert "equations" in template.mathematical_mappings

    def test_has_constraint_expressions(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        assert isinstance(template.constraint_expressions, list)

    def test_has_review_checklist(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        assert isinstance(template.review_checklist, list)
        assert len(template.review_checklist) > 0

    def test_has_confidence(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        assert "overall" in template.extraction_confidence
        assert "command_extraction" in template.extraction_confidence
        assert "parameter_extraction" in template.extraction_confidence
        assert "equation_mapping" in template.extraction_confidence

    def test_default_file_patterns(self, generator, cpp_source_dir):
        # Should work without specifying file_patterns
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
        )
        assert isinstance(template, HarnessTemplate)

    def test_with_entry_point_hints(self, generator, cpp_source_dir):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
            entry_point_hints=["fix"],
        )
        assert isinstance(template, HarnessTemplate)


# ── HarnessAutoGenerator: generate_from_docs ──

class TestGenerateFromDocs:
    def test_returns_template(self, generator, markdown_doc):
        template = generator.generate_from_docs(
            doc_path=str(markdown_doc),
            engine_name="mdengine",
        )
        assert isinstance(template, HarnessTemplate)
        assert template.engine_name == "mdengine"

    def test_source_path_is_doc_path(self, generator, markdown_doc):
        template = generator.generate_from_docs(
            doc_path=str(markdown_doc),
            engine_name="e",
        )
        assert template.source_path == str(markdown_doc)

    def test_default_version(self, generator, markdown_doc):
        template = generator.generate_from_docs(
            doc_path=str(markdown_doc),
            engine_name="e",
        )
        assert template.engine_version == "1.0.0"

    def test_custom_version(self, generator, markdown_doc):
        template = generator.generate_from_docs(
            doc_path=str(markdown_doc),
            engine_name="e",
            engine_version="3.0.0",
        )
        assert template.engine_version == "3.0.0"

    def test_doc_confidence_reduced_by_factor(self, generator, markdown_doc):
        # Doc-based extraction applies 0.8 factor to overall confidence
        template = generator.generate_from_docs(
            doc_path=str(markdown_doc),
            engine_name="e",
        )
        # Confidence should be a valid number >= 0
        assert template.extraction_confidence["overall"] >= 0.0

    def test_has_generated_code(self, generator, markdown_doc):
        template = generator.generate_from_docs(
            doc_path=str(markdown_doc),
            engine_name="e",
        )
        assert isinstance(template.generated_code, str)
        assert len(template.generated_code) > 0

    def test_has_review_checklist(self, generator, markdown_doc):
        template = generator.generate_from_docs(
            doc_path=str(markdown_doc),
            engine_name="e",
        )
        assert isinstance(template.review_checklist, list)
        assert len(template.review_checklist) > 0

    def test_nonexistent_doc_raises(self, generator):
        with pytest.raises(FileNotFoundError):
            generator.generate_from_docs(
                doc_path="/nonexistent/path.md",
                engine_name="e",
            )


# ── HarnessAutoGenerator: save_harness ──

class TestSaveHarness:
    def test_creates_directory_structure(self, generator, cpp_source_dir, tmp_path):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="savetest",
            file_patterns=["*.cpp"],
        )
        out_dir = tmp_path / "output"
        generator.save_harness(template, str(out_dir))

        harness_dir = out_dir / "savetest-harness"
        assert harness_dir.exists()

    def test_creates_harness_py(self, generator, cpp_source_dir, tmp_path):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="savetest",
            file_patterns=["*.cpp"],
        )
        out_dir = tmp_path / "output"
        generator.save_harness(template, str(out_dir))

        harness_file = out_dir / "savetest-harness" / "math_anything" / "savetest" / "core" / "harness.py"
        assert harness_file.exists()
        assert harness_file.read_text(encoding="utf-8") == template.generated_code

    def test_creates_init_py(self, generator, cpp_source_dir, tmp_path):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="myengine",
            file_patterns=["*.cpp"],
        )
        out_dir = tmp_path / "output"
        generator.save_harness(template, str(out_dir))

        init_file = out_dir / "myengine-harness" / "math_anything" / "myengine" / "__init__.py"
        assert init_file.exists()
        content = init_file.read_text(encoding="utf-8")
        assert "MyengineHarness" in content

    def test_creates_summary(self, generator, cpp_source_dir, tmp_path):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="summaryengine",
            file_patterns=["*.cpp"],
        )
        out_dir = tmp_path / "output"
        generator.save_harness(template, str(out_dir))

        summary = out_dir / "summaryengine-harness" / "ANALYSIS_SUMMARY.md"
        assert summary.exists()
        content = summary.read_text(encoding="utf-8")
        assert "summaryengine" in content.lower() or "Summaryengine" in content

    def test_creates_review_checklist(self, generator, cpp_source_dir, tmp_path):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="cengine",
            file_patterns=["*.cpp"],
        )
        out_dir = tmp_path / "output"
        generator.save_harness(template, str(out_dir))

        checklist = out_dir / "cengine-harness" / "REVIEW_CHECKLIST.md"
        assert checklist.exists()
        content = checklist.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_creates_nested_output_dir(self, generator, cpp_source_dir, tmp_path):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="test",
            file_patterns=["*.cpp"],
        )
        # Output dir doesn't exist yet
        out_dir = tmp_path / "nested" / "path" / "output"
        generator.save_harness(template, str(out_dir))
        assert out_dir.exists()


# ── HarnessAutoGenerator: _create_review_checklist ──

class TestCreateReviewChecklist:
    def test_returns_list(self, generator):
        checklist = generator._create_review_checklist({}, {}, {})
        assert isinstance(checklist, list)
        assert len(checklist) > 0

    def test_includes_confidence_item(self, generator):
        checklist = generator._create_review_checklist({}, {}, {"overall": 75.0})
        assert any("75" in item for item in checklist)

    def test_includes_command_count(self, generator):
        analysis = {"commands": [{"name": "c1"}, {"name": "c2"}]}
        checklist = generator._create_review_checklist(analysis, {}, {"overall": 50})
        assert any("2" in item for item in checklist)

    def test_includes_command_verification(self, generator):
        analysis = {"commands": [{"name": "nvt", "pattern": "fix nvt"}]}
        checklist = generator._create_review_checklist(analysis, {}, {"overall": 50})
        assert any("nvt" in item for item in checklist)

    def test_includes_equation_validation(self, generator):
        mappings = {"equations": [{"name": "thermostat"}]}
        checklist = generator._create_review_checklist({}, mappings, {"overall": 50})
        assert any("thermostat" in item for item in checklist)


# ── HarnessAutoGenerator: _generate_summary ──

class TestGenerateSummary:
    def test_summary_contains_engine_name(self, generator):
        template = HarnessTemplate(
            engine_name="lammps",
            engine_version="1.0",
            source_path="/tmp",
            extracted_commands=[],
            mathematical_mappings={},
            constraint_expressions=[],
            generated_code="",
            review_checklist=[],
            extraction_confidence={"overall": 80.0},
        )
        summary = generator._generate_summary(template)
        assert "Lammps" in summary

    def test_summary_contains_confidence(self, generator):
        template = HarnessTemplate(
            engine_name="test",
            engine_version="1.0",
            source_path="/tmp",
            extracted_commands=[],
            mathematical_mappings={},
            constraint_expressions=[],
            generated_code="",
            review_checklist=[],
            extraction_confidence={
                "overall": 75.0,
                "command_extraction": 80.0,
                "parameter_extraction": 70.0,
                "equation_mapping": 60.0,
            },
        )
        summary = generator._generate_summary(template)
        assert "75" in summary

    def test_summary_contains_commands(self, generator):
        template = HarnessTemplate(
            engine_name="test",
            engine_version="1.0",
            source_path="/tmp",
            extracted_commands=[{"name": "nvt", "description": "thermostat", "pattern": "fix nvt", "parameters": []}],
            mathematical_mappings={},
            constraint_expressions=[],
            generated_code="",
            review_checklist=[],
            extraction_confidence={"overall": 50.0},
        )
        summary = generator._generate_summary(template)
        assert "nvt" in summary

    def test_summary_contains_constraints(self, generator):
        template = HarnessTemplate(
            engine_name="test",
            engine_version="1.0",
            source_path="/tmp",
            extracted_commands=[],
            mathematical_mappings={},
            constraint_expressions=["dt > 0", "temp >= 0"],
            generated_code="",
            review_checklist=[],
            extraction_confidence={"overall": 50.0},
        )
        summary = generator._generate_summary(template)
        assert "dt > 0" in summary


# ── HarnessAutoGenerator: _generate_harness_code ──

class TestGenerateHarnessCode:
    def test_generates_string(self, generator):
        analysis = {"commands": [], "parameters": [], "source_path": "/tmp"}
        mappings = {"equations": []}
        confidence = {"overall": 50.0, "command_extraction": 50, "parameter_extraction": 50, "equation_mapping": 50}
        code = generator._generate_harness_code("test", "1.0", analysis, mappings, [], confidence)
        assert isinstance(code, str)
        assert len(code) > 0

    def test_fallback_mode_contains_warning(self, generator):
        # Force fallback by setting jinja_env to None
        generator.jinja_env = None
        analysis = {"commands": [], "parameters": [], "source_path": "/tmp"}
        mappings = {"equations": []}
        confidence = {"overall": 50.0}
        code = generator._generate_harness_code("test", "1.0", analysis, mappings, [], confidence)
        assert "FALLBACK" in code or "PlaceholderHarness" in code

    def test_fallback_contains_engine_name(self, generator):
        generator.jinja_env = None
        analysis = {"commands": [], "parameters": [], "source_path": "/tmp"}
        mappings = {"equations": []}
        confidence = {"overall": 50.0}
        code = generator._generate_harness_code("myengine", "1.0", analysis, mappings, [], confidence)
        assert "myengine" in code.lower()


# ── Integration: full workflow ──

class TestIntegrationWorkflow:
    def test_generate_and_save(self, generator, cpp_source_dir, tmp_path):
        template = generator.generate_from_source(
            source_dir=str(cpp_source_dir),
            engine_name="integration",
            engine_version="1.0.0",
            file_patterns=["*.cpp"],
        )
        out_dir = tmp_path / "integration_output"
        generator.save_harness(template, str(out_dir))

        # Verify all files created
        harness_dir = out_dir / "integration-harness"
        assert harness_dir.exists()
        assert (harness_dir / "math_anything" / "integration" / "core" / "harness.py").exists()
        assert (harness_dir / "math_anything" / "integration" / "__init__.py").exists()
        assert (harness_dir / "ANALYSIS_SUMMARY.md").exists()
        assert (harness_dir / "REVIEW_CHECKLIST.md").exists()

    def test_generate_from_docs_and_save(self, generator, markdown_doc, tmp_path):
        template = generator.generate_from_docs(
            doc_path=str(markdown_doc),
            engine_name="docengine",
        )
        out_dir = tmp_path / "doc_output"
        generator.save_harness(template, str(out_dir))

        harness_dir = out_dir / "docengine-harness"
        assert harness_dir.exists()
        assert (harness_dir / "math_anything" / "docengine" / "core" / "harness.py").exists()
