import json
import subprocess
import sys
from pathlib import Path


def test_cli_homotopy_vasp_vs_qe():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "math_anything",
            "homotopy",
            "vasp",
            "qe",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["engine_a"] == "vasp"
    assert report["engine_b"] == "qe"
    assert "witness" in report
    assert isinstance(report["witness"]["equivalent"], bool)
