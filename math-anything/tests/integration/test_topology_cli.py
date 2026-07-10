import json
import subprocess
import sys


def test_cli_loops_subcommand_exists():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything.cli", "loops", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "loops" in result.stdout.lower()


def test_cli_loops_vasp_executes():
    result = subprocess.run(
        [sys.executable, "-m", "math_anything.cli", "loops", "vasp"],
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
        [sys.executable, "-m", "math_anything.cli", "loops", "lammps"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "not yet supported" in result.stdout
