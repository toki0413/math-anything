import subprocess
import sys
from pathlib import Path


def test_cli_loops_mermaid_output():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything.cli", "loops", "vasp", "--visualize", "mermaid"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "graph LR" in result.stdout
