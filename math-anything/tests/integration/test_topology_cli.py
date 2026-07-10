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
