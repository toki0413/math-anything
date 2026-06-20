"""Convert Markdown methodology draft to Jupyter Notebook (.ipynb)."""

import json
from typing import Any, List

import nbformat
from nbformat.v4 import new_markdown_cell, new_notebook


def markdown_to_notebook(md_text: str, output_path: str, title: str = "Computational Methodology") -> None:
    """Convert a Markdown methodology draft to a .ipynb notebook.

    Args:
        md_text: Markdown-formatted methodology text.
        output_path: Path to write the .ipynb file.
        title: Notebook title.
    """
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.x",
    }

    # Title cell
    cells: List[Any] = [new_markdown_cell(f"# {title}\n")]

    # Split by headers to create logical cells
    current_lines: List[str] = []
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") or stripped.startswith("## "):
            if current_lines:
                cells.append(new_markdown_cell("\n".join(current_lines).strip()))
                current_lines = []
        current_lines.append(line)

    if current_lines:
        cells.append(new_markdown_cell("\n".join(current_lines).strip()))

    nb.cells = cells

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nbformat.from_dict(nb), f, indent=2, ensure_ascii=False)
