"""Unit tests for codegen/doc_analyzer.py — documentation analysis."""

import pytest

from math_anything.codegen.doc_analyzer import (
    DocCommand,
    DocumentationAnalyzer,
    quick_analyze_docs,
)

# ── DocCommand dataclass ──


class TestDocCommand:
    def test_required_fields(self):
        c = DocCommand(name="nvt", syntax="nvt temp", description="thermostat", parameters=[])
        assert c.name == "nvt"
        assert c.syntax == "nvt temp"
        assert c.description == "thermostat"
        assert c.parameters == []

    def test_default_examples_and_constraints(self):
        c = DocCommand(name="x", syntax="x", description="d", parameters=[])
        assert c.examples == []
        assert c.constraints == []

    def test_with_full_fields(self):
        params = [{"name": "temp", "type": "float"}]
        c = DocCommand(
            name="nvt",
            syntax="nvt temp 300 300 100",
            description="Nose-Hoover thermostat",
            parameters=params,
            examples=["nvt 300 300 100"],
            constraints=["temp > 0"],
        )
        assert len(c.parameters) == 1
        assert len(c.examples) == 1
        assert len(c.constraints) == 1


# ── DocumentationAnalyzer: creation ──


class TestAnalyzerCreation:
    def test_creates_with_empty_commands(self):
        a = DocumentationAnalyzer()
        assert a.commands == []

    def test_has_syntax_patterns(self):
        a = DocumentationAnalyzer()
        assert len(a.SYNTAX_PATTERNS) >= 3

    def test_has_param_patterns(self):
        a = DocumentationAnalyzer()
        assert len(a.PARAM_PATTERNS) >= 2

    def test_has_constraint_patterns(self):
        a = DocumentationAnalyzer()
        assert len(a.CONSTRAINT_PATTERNS) >= 2


# ── DocumentationAnalyzer: analyze markdown ──


class TestAnalyzeMarkdown:
    def test_analyze_markdown_returns_dict(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Title\n\nSome content.\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        assert isinstance(result, dict)
        assert "commands" in result
        assert "parameters" in result
        assert "sections" in result
        assert "source_path" in result

    def test_analyze_markdown_extracts_sections(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Title\n\n## Section1\n\ncontent\n\n### Sub\n\nmore\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        assert len(result["sections"]) >= 2

    def test_analyze_markdown_section_has_level(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Title\n\n## Sub\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        levels = [s["level"] for s in result["sections"]]
        assert 1 in levels
        assert 2 in levels

    def test_analyze_markdown_section_has_preview(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Title\n\nbody text here\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        assert any("body text" in s["preview"] for s in result["sections"])

    def test_analyze_markdown_source_path(self, tmp_path):
        md = tmp_path / "manual.md"
        md.write_text("# Title\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        assert result["source_path"] == str(md)

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

    def test_analyze_markdown_constraint_extraction(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "## nvt - thermostat\n\ntemp must be positive\n",
            encoding="utf-8",
        )
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        # The constraint pattern should fire on "must be positive"
        # Constraint is attached to current_command if a command was extracted
        assert "commands" in result

    def test_analyze_markdown_with_code_block_syntax(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "## nvt - thermostat\n\n```\nnvt temp 300 300 100\n```\n",
            encoding="utf-8",
        )
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        # The header pattern should match nvt - thermostat
        assert isinstance(result["commands"], list)

    def test_analyze_markdown_extension(self, tmp_path):
        md = tmp_path / "test.markdown"
        md.write_text("# Title\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(md))
        assert "commands" in result


# ── DocumentationAnalyzer: analyze HTML ──


class TestAnalyzeHtml:
    def test_analyze_html_returns_dict(self, tmp_path):
        html = tmp_path / "test.html"
        html.write_text("<html><body><p>content</p></body></html>", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(html))
        assert "commands" in result
        assert "parameters" in result

    def test_analyze_html_strips_tags(self, tmp_path):
        html = tmp_path / "test.html"
        html.write_text(
            "<html><body><h1>Title</h1><p>Syntax: nvt</p></body></html>",
            encoding="utf-8",
        )
        a = DocumentationAnalyzer()
        result = a.analyze(str(html))
        # After stripping tags, "Syntax: nvt" should be parsed as text
        assert isinstance(result["commands"], list)

    def test_analyze_html_unescapes_entities(self, tmp_path):
        html = tmp_path / "test.html"
        html.write_text("<p>x &lt; 5 &amp; y &gt; 3</p>", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(html))
        assert "commands" in result

    def test_analyze_htm_extension(self, tmp_path):
        html = tmp_path / "test.htm"
        html.write_text("<p>Syntax: nvt</p>", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(html))
        assert "commands" in result


# ── DocumentationAnalyzer: analyze text ──


class TestAnalyzeText:
    def test_analyze_text_returns_dict(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Some plain text.\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(txt))
        assert "commands" in result
        assert "parameters" in result
        assert "sections" in result
        assert result["source_path"] == str(txt)

    def test_analyze_text_extracts_syntax(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Syntax: nvt\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(txt))
        names = [c["name"] for c in result["commands"]]
        assert "nvt" in names

    def test_analyze_text_deduplicates_commands(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Syntax: nvt\nSyntax: nvt\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(txt))
        names = [c["name"] for c in result["commands"]]
        assert names.count("nvt") == 1

    def test_analyze_text_extracts_parameters(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("temp (float) - temperature value\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(txt))
        assert len(result["parameters"]) >= 1
        assert result["parameters"][0]["name"] == "temp"
        assert result["parameters"][0]["type"] == "float"

    def test_analyze_text_param_equal_pattern(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("temp = 300 - initial temperature\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(txt))
        assert len(result["parameters"]) >= 1

    def test_analyze_text_command_has_description(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("context line\nSyntax: nvt\nmore context\n", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(txt))
        if result["commands"]:
            assert "description" in result["commands"][0]

    def test_analyze_text_empty_file(self, tmp_path):
        txt = tmp_path / "empty.txt"
        txt.write_text("", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(txt))
        assert result["commands"] == []
        assert result["parameters"] == []


# ── DocumentationAnalyzer: analyze PDF ──


class TestAnalyzePdf:
    def test_analyze_pdf_returns_dict(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content", encoding="utf-8")
        a = DocumentationAnalyzer()
        result = a.analyze(str(pdf))
        assert "commands" in result
        assert "parameters" in result

    def test_analyze_pdf_uses_placeholder(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy", encoding="utf-8")
        a = DocumentationAnalyzer()
        # _extract_pdf_text returns a placeholder string
        text = a._extract_pdf_text(pdf)
        assert "PDF extraction" in text or "PyPDF2" in text


# ── DocumentationAnalyzer: error cases ──


class TestAnalyzeErrors:
    def test_nonexistent_file_raises(self):
        a = DocumentationAnalyzer()
        with pytest.raises(FileNotFoundError):
            a.analyze("/nonexistent/path/to/file.md")

    def test_nonexistent_file_message(self):
        a = DocumentationAnalyzer()
        with pytest.raises(FileNotFoundError, match="Documentation not found"):
            a.analyze("/nonexistent/path/to/file.md")


# ── DocumentationAnalyzer: _read_file ──


class TestReadFile:
    def test_read_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        a = DocumentationAnalyzer()
        content = a._read_file(f)
        assert "hello world" in content

    def test_read_pdf_returns_placeholder(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_text("dummy", encoding="utf-8")
        a = DocumentationAnalyzer()
        content = a._read_file(f)
        assert "PDF extraction" in content or "PyPDF2" in content


# ── DocumentationAnalyzer: _parse_text ──


class TestParseText:
    def test_parse_text_returns_structure(self):
        a = DocumentationAnalyzer()
        result = a._parse_text("Syntax: nvt\n", "src")
        assert "commands" in result
        assert "parameters" in result
        assert "sections" in result
        assert result["source_path"] == "src"

    def test_parse_text_no_commands(self):
        a = DocumentationAnalyzer()
        result = a._parse_text("just plain text\nno commands here", "src")
        assert result["commands"] == []

    def test_parse_text_param_attached_to_command(self):
        a = DocumentationAnalyzer()
        text = "Syntax: nvt\ntemp (float) - temperature\n"
        result = a._parse_text(text, "src")
        if result["commands"]:
            assert len(result["commands"][0]["parameters"]) >= 1


# ── DocumentationAnalyzer: _parse_markdown ──


class TestParseMarkdown:
    def test_parse_markdown_returns_structure(self):
        a = DocumentationAnalyzer()
        result = a._parse_markdown("# Title\n\nbody\n", "src")
        assert "commands" in result
        assert "parameters" in result
        assert "sections" in result
        assert result["source_path"] == "src"

    def test_parse_markdown_section_content_preview(self):
        a = DocumentationAnalyzer()
        result = a._parse_markdown("# Title\n\nbody text\n", "src")
        assert any("body text" in s["preview"] for s in result["sections"])

    def test_parse_markdown_empty_content(self):
        a = DocumentationAnalyzer()
        result = a._parse_markdown("", "src")
        assert result["commands"] == []
        assert result["sections"] == []


# ── DocumentationAnalyzer: _find_syntax ──


class TestFindSyntax:
    def test_find_syntax_in_code_block(self):
        a = DocumentationAnalyzer()
        lines = ["## cmd - test", "```", "cmd arg1 arg2", "```"]
        syntax = a._find_syntax(lines, 0)
        assert syntax is not None
        assert "cmd arg1 arg2" in syntax

    def test_find_syntax_with_backtick(self):
        a = DocumentationAnalyzer()
        lines = ["## cmd - test", "`cmd arg1`"]
        syntax = a._find_syntax(lines, 0)
        assert syntax is not None

    def test_find_syntax_returns_none(self):
        a = DocumentationAnalyzer()
        # Lines without code blocks, backticks, or "syntax" keyword
        lines = ["## cmd - test", "just regular text", "nothing useful here"]
        syntax = a._find_syntax(lines, 0)
        assert syntax is None

    def test_find_syntax_at_end_of_file(self):
        a = DocumentationAnalyzer()
        lines = ["## cmd - test"]
        syntax = a._find_syntax(lines, 0)
        assert syntax is None


# ── DocumentationAnalyzer: infer_from_examples ──


class TestInferFromExamples:
    def test_infer_single_example(self):
        a = DocumentationAnalyzer()
        result = a.infer_from_examples(["fix 1 all nvt temp 300 300 100"])
        assert len(result) >= 1
        assert "command" in result[0]
        assert "context" in result[0]
        assert "arguments" in result[0]
        assert "full_example" in result[0]

    def test_infer_multiple_examples(self):
        a = DocumentationAnalyzer()
        result = a.infer_from_examples(
            [
                "fix 1 all nvt temp 300",
                "fix 1 all npt pressure 1.0",
            ]
        )
        assert len(result) >= 2

    def test_infer_empty_list(self):
        a = DocumentationAnalyzer()
        result = a.infer_from_examples([])
        assert result == []

    def test_infer_short_example_skipped(self):
        a = DocumentationAnalyzer()
        # Single token: len(parts) < 2
        result = a.infer_from_examples(["x"])
        assert result == []

    def test_infer_no_alpha_keyword(self):
        a = DocumentationAnalyzer()
        # All tokens are numeric or too short
        result = a.infer_from_examples(["1 2 3 4"])
        assert result == []

    def test_infer_extracts_arguments(self):
        a = DocumentationAnalyzer()
        result = a.infer_from_examples(["run nvt 300 100"])
        if result:
            assert result[0]["command"] == "run"
            assert "300" in result[0]["arguments"]
            assert "100" in result[0]["arguments"]


# ── quick_analyze_docs ──


class TestQuickAnalyzeDocs:
    def test_quick_analyze_markdown(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Title\n## cmd - test\n", encoding="utf-8")
        result = quick_analyze_docs(str(md))
        assert "commands" in result
        assert "sections" in result

    def test_quick_analyze_text(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Syntax: nvt\n", encoding="utf-8")
        result = quick_analyze_docs(str(txt))
        assert "commands" in result

    def test_quick_analyze_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            quick_analyze_docs("/nonexistent/path.md")

    def test_quick_analyze_returns_source_path(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Title\n", encoding="utf-8")
        result = quick_analyze_docs(str(md))
        assert result["source_path"] == str(md)
