"""Integration tests for the MathAnything API.

Tests the high-level API: extract(), parameter validation/coercion,
error handling, and discover().
"""

import pytest

from math_anything.api import ENGINE_EXTRACTORS, ExtractionResult, MathAnything
from math_anything.exceptions import (
    ExtractionFileNotFoundError,
    MathAnythingError,
    UnsupportedEngineError,
)


# ── Fixtures ──


@pytest.fixture
def api():
    return MathAnything()


@pytest.fixture
def vasp_params():
    return {"ENCUT": 520, "EDIFF": 1e-6, "SIGMA": 0.05}


@pytest.fixture
def lammps_params():
    return {"ensemble": "NVE", "n_atoms": 1000}


@pytest.fixture
def abaqus_params():
    return {"analysis_type": "static"}


@pytest.fixture
def ansys_params():
    return {"E": 210e9, "nu": 0.3, "analysis_type": "static"}


# ── Supported engines ──


class TestSupportedEngines:
    def test_list_contains_core_engines(self, api):
        engines = api.supported_engines
        for name in ("vasp", "lammps", "abaqus", "ansys"):
            assert name in engines

    def test_list_is_copy(self, api):
        engines = api.supported_engines
        engines.append("bogus")
        assert "bogus" not in api.supported_engines


# ── extract() with different engines ──


class TestExtract:
    def test_vasp_extract_success(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        assert result.success
        assert result.engine == "vasp"
        assert isinstance(result.schema, dict)

    def test_lammps_extract_success(self, api, lammps_params):
        result = api.extract("lammps", lammps_params)
        assert result.success
        assert result.engine == "lammps"

    def test_abaqus_extract_success(self, api, abaqus_params):
        result = api.extract("abaqus", abaqus_params)
        assert result.success
        assert result.engine == "abaqus"

    def test_ansys_extract_success(self, api, ansys_params):
        result = api.extract("ansys", ansys_params)
        assert result.success
        assert result.engine == "ansys"

    def test_extract_result_has_mathematical_structure(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        assert "mathematical_structure" in result.schema
        ms = result.schema["mathematical_structure"]
        assert ms["problem_type"] == "nonlinear_eigenvalue"
        assert "H[n]" in ms["canonical_form"]

    def test_extract_result_type(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        assert isinstance(result, ExtractionResult)

    def test_extract_case_insensitive(self, api, vasp_params):
        result = api.extract("VASP", vasp_params)
        assert result.success
        assert result.engine == "vasp"

    def test_extract_comsol(self, api):
        result = api.extract("comsol", {"analysis_type": "static"})
        assert result.success
        assert result.engine == "comsol"

    def test_extract_gromacs(self, api):
        result = api.extract("gromacs", {"ensemble": "NVT"})
        assert result.success
        assert result.engine == "gromacs"


# ── Parameter validation & coercion ──


class TestParamValidation:
    def test_kpoints_bare_list_coerced(self, api):
        params = {"ENCUT": 520, "kpoints": [4, 4, 4]}
        result = api.extract("vasp", params)
        assert result.success
        # The coerced params should have kpoints as {"grid": [4,4,4]}
        assert result.files["params"]["kpoints"] == {"grid": [4, 4, 4]}

    def test_kpoints_dict_passes_through(self, api):
        params = {"ENCUT": 520, "kpoints": {"grid": [4, 4, 4]}}
        result = api.extract("vasp", params)
        assert result.success
        assert result.files["params"]["kpoints"] == {"grid": [4, 4, 4]}

    def test_lattice_bare_list_coerced(self, api):
        params = {
            "ENCUT": 520,
            "lattice": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
        }
        result = api.extract("vasp", params)
        assert result.success
        coerced_lattice = result.files["params"]["lattice"]
        assert "vectors" in coerced_lattice

    def test_params_not_dict_raises(self, api):
        with pytest.raises(MathAnythingError):
            api.extract("vasp", "not_a_dict")

    def test_params_list_raises(self, api):
        with pytest.raises(MathAnythingError, match="Expected dict"):
            api.extract("vasp", [520, 0.05])

    def test_params_tuple_raises(self, api):
        with pytest.raises(MathAnythingError, match="Expected dict"):
            api.extract("vasp", (520, 0.05))

    def test_scalar_params_unchanged(self, api):
        params = {"ENCUT": 520, "EDIFF": 1e-6}
        result = api.extract("vasp", params)
        assert result.success
        assert result.files["params"]["ENCUT"] == 520
        assert result.files["params"]["EDIFF"] == 1e-6


# ── Error handling ──


class TestErrorHandling:
    def test_unsupported_engine_raises(self, api):
        with pytest.raises(UnsupportedEngineError):
            api.extract("nonexistent_engine", {"ENCUT": 520})

    def test_unsupported_engine_message(self, api):
        with pytest.raises(UnsupportedEngineError, match="not supported"):
            api.extract("foo", {})

    def test_extract_file_not_found(self, api):
        with pytest.raises(ExtractionFileNotFoundError):
            api.extract_file("vasp", "/nonexistent/path/INCAR")

    def test_extract_file_invalid_type(self, api):
        with pytest.raises(MathAnythingError, match="Expected str, Path, or dict"):
            api.extract_file("vasp", 12345)

    def test_extract_file_list_raises(self, api):
        with pytest.raises(MathAnythingError, match="Expected str, Path, or dict"):
            api.extract_file("vasp", ["INCAR", "POSCAR"])


# ── ExtractionResult ──


class TestExtractionResult:
    def test_summary_success(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        summary = result.summary()
        assert "VASP" in summary
        assert "Success" in summary

    def test_schema_attribute_access(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        # ExtractionResult.__getattr__ delegates to schema keys
        assert result.mathematical_structure is not None

    def test_schema_attribute_missing_raises(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        with pytest.raises(AttributeError, match="no attribute"):
            _ = result.totally_bogus_key_xyz


# ── discover() ──


class TestDiscover:
    def test_discover_simple_linear(self, api):
        import numpy as np

        x = np.linspace(0, 10, 50)
        y = 2 * x + 1
        result = api.discover(x.reshape(-1, 1), y, ["x"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_discover_returns_string(self, api):
        import numpy as np

        x = np.linspace(0, 5, 30)
        y = np.sin(x)
        result = api.discover(x.reshape(-1, 1), y, ["x"])
        assert isinstance(result, str)
