import json
import subprocess
import sys
from pathlib import Path


def _run_homotopy(*extra_args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "math_anything",
            "homotopy",
            *extra_args,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )


def test_cli_homotopy_vasp_vs_qe():
    result = _run_homotopy("vasp", "qe")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["engine_a"] == "vasp"
    assert report["engine_b"] == "qe"
    assert "witness" in report
    assert isinstance(report["witness"]["equivalent"], bool)


def test_cli_homotopy_params_affect_equivalence():
    same = _run_homotopy("vasp", "qe", "--param-a", "520", "--param-b", "520")
    assert same.returncode == 0, same.stderr
    report_same = json.loads(same.stdout)
    assert report_same["witness"]["equivalent"] is True

    different = _run_homotopy("vasp", "qe", "--param-a", "300", "--param-b", "520")
    assert different.returncode == 0, different.stderr
    report_diff = json.loads(different.stdout)
    assert report_diff["cutoff_a_eV"] == 300.0
    assert report_diff["cutoff_b_eV"] == 520.0
    assert report_diff["witness"]["equivalent"] is False


def test_cli_homotopy_writes_relative_output():
    out_file = Path("test_homotopy_output.json")
    try:
        result = _run_homotopy("vasp", "qe", "--output", str(out_file))
        assert result.returncode == 0, result.stderr
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["engine_a"] == "vasp"
        assert data["engine_b"] == "qe"
    finally:
        if out_file.exists():
            out_file.unlink()


def test_cli_homotopy_rejects_absolute_output():
    absolute_path = (
        "/tmp/homotopy_report.json" if sys.platform != "win32" else "C:\\homotopy_report.json"
    )
    result = _run_homotopy("vasp", "qe", "--output", absolute_path)
    assert result.returncode == 1
    assert "working directory" in result.stdout


def test_cli_homotopy_param_a_binds_to_engine_a_for_qe():
    """--param-a must apply to engine_a regardless of engine name."""
    result = _run_homotopy("qe", "vasp", "--param-a", "300")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["engine_a"] == "qe"
    assert report["engine_b"] == "vasp"
    assert report["cutoff_a_eV"] == 300.0
    assert report["cutoff_b_eV"] == 520.0
    assert report["witness"]["equivalent"] is False
