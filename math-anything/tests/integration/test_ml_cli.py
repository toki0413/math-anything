import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_ml_subcommand_runs():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything", "ml", "--input-dim", "2", "--output-dim", "1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["domain"] == "supervised_learning"


def test_cli_ml_mermaid_output():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "math_anything",
            "ml",
            "--input-dim",
            "2",
            "--output-dim",
            "1",
            "--visualize",
            "mermaid",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "graph LR" in result.stdout
    assert "data_sampling" in result.stdout


def test_cli_ml_compare_with_dft():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "math_anything",
            "ml",
            "--input-dim",
            "2",
            "--output-dim",
            "1",
            "--compare-with",
            "dft",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert "cross_domain_homotopy" in report
    assert "equivalent" in report["cross_domain_homotopy"]
