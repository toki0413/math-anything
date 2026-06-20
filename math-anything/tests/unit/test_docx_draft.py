"""Unit tests for draft/docx_draft.py — Markdown to DOCX conversion."""

from pathlib import Path

import pytest

# Skip all tests if python-docx is not installed
docx = pytest.importorskip("docx")

from math_anything.draft.docx_draft import markdown_to_docx


# ── markdown_to_docx: basic ──

class TestMarkdownToDocxBasic:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "output.docx"
        markdown_to_docx("# Title\n\nSome content.\n", str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_title(self, tmp_path):
        out = tmp_path / "output.docx"
        markdown_to_docx("content", str(out), title="My Custom Title")
        # Read back the document
        doc = docx.Document(str(out))
        # First paragraph should be the title
        title_para = doc.paragraphs[0]
        assert "My Custom Title" in title_para.text

    def test_default_title(self, tmp_path):
        out = tmp_path / "output.docx"
        markdown_to_docx("content", str(out))
        doc = docx.Document(str(out))
        title_para = doc.paragraphs[0]
        assert "Computational Methodology" in title_para.text

    def test_empty_markdown(self, tmp_path):
        out = tmp_path / "output.docx"
        markdown_to_docx("", str(out))
        assert out.exists()


# ── markdown_to_docx: headings ──

class TestMarkdownToDocxHeadings:
    def test_h1_heading(self, tmp_path):
        out = tmp_path / "output.docx"
        markdown_to_docx("# Section Title\n", str(out))
        doc = docx.Document(str(out))
        texts = [p.text for p in doc.paragraphs]
        assert "Section Title" in texts

    def test_h2_heading(self, tmp_path):
        out = tmp_path / "output.docx"
        markdown_to_docx("## Subsection\n", str(out))
        doc = docx.Document(str(out))
        texts = [p.text for p in doc.paragraphs]
        assert "Subsection" in texts

    def test_h3_heading(self, tmp_path):
        out = tmp_path / "output.docx"
        markdown_to_docx("### Subsubsection\n", str(out))
        doc = docx.Document(str(out))
        texts = [p.text for p in doc.paragraphs]
        assert "Subsubsection" in texts

    def test_multiple_headings(self, tmp_path):
        out = tmp_path / "output.docx"
        md = "# Title\n\n## Section 1\n\n### Sub\n\n## Section 2\n"
        markdown_to_docx(md, str(out))
        doc = docx.Document(str(out))
        texts = [p.text for p in doc.paragraphs]
        assert "Title" in texts
        assert "Section 1" in texts
        assert "Sub" in texts
        assert "Section 2" in texts


# ── markdown_to_docx: lists ──

class TestMarkdownToDocxLists:
    def test_bullet_list(self, tmp_path):
        out = tmp_path / "output.docx"
        md = "- Item 1\n- Item 2\n- Item 3\n"
        markdown_to_docx(md, str(out))
        doc = docx.Document(str(out))
        texts = [p.text for p in doc.paragraphs]
        assert "Item 1" in texts
        assert "Item 2" in texts
        assert "Item 3" in texts

    def test_numbered_list(self, tmp_path):
        out = tmp_path / "output.docx"
        md = "1. First\n2. Second\n3. Third\n"
        markdown_to_docx(md, str(out))
        doc = docx.Document(str(out))
        texts = [p.text for p in doc.paragraphs]
        assert "First" in texts
        assert "Second" in texts
        assert "Third" in texts


# ── markdown_to_docx: paragraphs ──

class TestMarkdownToDocxParagraphs:
    def test_simple_paragraph(self, tmp_path):
        out = tmp_path / "output.docx"
        markdown_to_docx("This is a paragraph.\n", str(out))
        doc = docx.Document(str(out))
        texts = [p.text for p in doc.paragraphs]
        assert "This is a paragraph." in texts

    def test_multi_line_paragraph(self, tmp_path):
        out = tmp_path / "output.docx"
        md = "Line one.\nLine two.\nLine three.\n"
        markdown_to_docx(md, str(out))
        doc = docx.Document(str(out))
        # Multi-line paragraphs should be joined
        full_text = " ".join(p.text for p in doc.paragraphs)
        assert "Line one" in full_text
        assert "Line two" in full_text

    def test_blank_line_separates_paragraphs(self, tmp_path):
        out = tmp_path / "output.docx"
        md = "First paragraph.\n\nSecond paragraph.\n"
        markdown_to_docx(md, str(out))
        doc = docx.Document(str(out))
        texts = [p.text for p in doc.paragraphs]
        assert "First paragraph." in texts
        assert "Second paragraph." in texts


# ── markdown_to_docx: code blocks ──

class TestMarkdownToDocxCodeBlocks:
    def test_code_block(self, tmp_path):
        out = tmp_path / "output.docx"
        md = "```\nprint('hello')\nx = 42\n```\n"
        markdown_to_docx(md, str(out))
        doc = docx.Document(str(out))
        full_text = "\n".join(p.text for p in doc.paragraphs)
        assert "print" in full_text or "hello" in full_text

    def test_code_block_with_surrounding_text(self, tmp_path):
        out = tmp_path / "output.docx"
        md = "Before code.\n\n```\ncode here\n```\n\nAfter code.\n"
        markdown_to_docx(md, str(out))
        doc = docx.Document(str(out))
        full_text = " ".join(p.text for p in doc.paragraphs)
        assert "Before code" in full_text
        assert "After code" in full_text


# ── markdown_to_docx: complex documents ──

class TestMarkdownToDocxComplex:
    def test_full_document(self, tmp_path):
        out = tmp_path / "output.docx"
        md = """# Introduction

This is the introduction.

## Methodology

The methodology follows these steps:

1. Setup the simulation
2. Run the calculation
3. Analyze results

### Code Example

```
result = run_simulation()
```

## Results

- Energy converged
- Forces below threshold
"""
        markdown_to_docx(md, str(out))
        doc = docx.Document(str(out))
        full_text = " ".join(p.text for p in doc.paragraphs)
        assert "Introduction" in full_text
        assert "Methodology" in full_text
        assert "Setup the simulation" in full_text
        assert "Energy converged" in full_text

    def test_output_is_valid_docx(self, tmp_path):
        out = tmp_path / "output.docx"
        markdown_to_docx("# Test\n\nContent\n", str(out))
        # Should be able to re-open the document
        doc = docx.Document(str(out))
        assert len(doc.paragraphs) > 0
