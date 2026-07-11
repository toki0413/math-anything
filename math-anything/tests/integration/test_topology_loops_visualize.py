import subprocess
import sys
from pathlib import Path


def test_cli_loops_mermaid_output():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything", "loops", "--engine", "vasp", "--visualize", "mermaid"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "graph LR" in result.stdout


def test_cli_loops_graphviz_output():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything", "loops", "--engine", "vasp", "--visualize", "graphviz"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "digraph" in result.stdout
