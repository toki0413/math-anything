import json
import subprocess
import sys
from pathlib import Path


def test_cli_loops_subcommand_exists():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything", "loops", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "loops" in result.stdout.lower()


def test_cli_loops_vasp_executes():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything", "loops", "--engine", "vasp"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["engine"] == "vasp"
    assert "betti" in data
    assert "loops" in data


def test_cli_loops_rejects_unsupported_engine():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything", "loops", "--engine", "lammps"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "not yet supported" in result.stdout


def test_cli_loops_writes_relative_output():
    out_file = Path("test_loops_output.json")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "math_anything", "loops", "--engine", "vasp", "--output", str(out_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["engine"] == "vasp"
    finally:
        if out_file.exists():
            out_file.unlink()


def test_cli_loops_rejects_absolute_output():
    absolute_path = "/tmp/loops_report.json" if sys.platform != "win32" else "C:\\loops_report.json"
    result = subprocess.run(
        [sys.executable, "-m", "math_anything", "loops", "--engine", "vasp", "--output", absolute_path],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "working directory" in result.stdout


def test_cli_loops_accepts_positional_engine():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything", "loops", "vasp"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["engine"] == "vasp"


def test_cli_loops_flag_wins_over_positional_engine():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything", "loops", "vasp", "--engine", "qe"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["engine"] == "qe"


def test_cli_loops_errors_without_engine():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything", "loops"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "engine" in result.stderr.lower()
