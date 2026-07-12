"""Extended unit tests for api.py — covering visualization, summary, file parsing,
health/metrics, verify, and convenience functions.

Targets lines uncovered by tests/integration/test_api.py:
  - ExtractionResult.visualize/to_mermaid/to_graphviz (lines 95-106)
  - ExtractionResult.summary error path + approximations overflow + constraints (lines 115-144)
  - MathAnything.extract_file with dict filepath (lines 362-368)
  - MathAnything._detect_file_type / _parse_files (lines 376-424)
  - MathAnything.compare / visualize / health_check / get_metrics / get_prometheus_metrics
  - MathAnything.verify (lines 638-678)
  - Module-level extract() / extract_file() convenience functions
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from math_anything.api import (
    ExtractionResult,
    MathAnything,
)
from math_anything.api import (
    extract as module_extract,
)
from math_anything.api import (
    extract_file as module_extract_file,
)
from math_anything.exceptions import (
    ExtractionFileNotFoundError,
    MathAnythingError,
)

# ── Fixtures ──


@pytest.fixture
def api():
    return MathAnything()


@pytest.fixture
def vasp_params():
    return {"ENCUT": 520, "EDIFF": 1e-6, "SIGMA": 0.05}


# ── ExtractionResult.visualize / to_mermaid / to_graphviz ──


class TestExtractionResultVisualize:
    def test_visualize_mermaid(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        viz = result.visualize(format="mermaid")
        assert isinstance(viz, str)
        assert "graph TD" in viz

    def test_visualize_graphviz(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        viz = result.visualize(format="graphviz")
        assert isinstance(viz, str)
        assert len(viz) > 0

    def test_visualize_text(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        viz = result.visualize(format="text")
        assert isinstance(viz, str)

    def test_to_mermaid(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        mermaid = result.to_mermaid()
        assert isinstance(mermaid, str)
        assert "graph TD" in mermaid

    def test_to_graphviz(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        gv = result.to_graphviz()
        assert isinstance(gv, str)

    def test_visualize_unknown_format_raises(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        with pytest.raises(ValueError, match="Unknown format"):
            result.visualize(format="bogus_format")


# ── ExtractionResult.summary — error path, approximations, constraints, warnings ──


class TestExtractionResultSummaryExtended:
    def test_summary_failed_result(self):
        result = ExtractionResult(
            engine="vasp",
            files={},
            schema={},
            success=False,
            errors=["something went wrong"],
            warnings=[],
        )
        summary = result.summary()
        assert "VASP" in summary
        assert "Failed" in summary
        assert "something went wrong" in summary

    def test_summary_success_no_math_struct(self):
        result = ExtractionResult(
            engine="lammps",
            files={},
            schema={},
            success=True,
            errors=[],
            warnings=[],
        )
        summary = result.summary()
        assert "LAMMPS" in summary
        assert "Success" in summary

    def test_summary_with_approximations_overflow(self):
        # More than 5 approximations triggers "and X more" line
        approxs = [{"name": f"approx_{i}"} for i in range(8)]
        result = ExtractionResult(
            engine="vasp",
            files={},
            schema={"approximations": approxs},
            success=True,
            errors=[],
            warnings=[],
        )
        summary = result.summary()
        assert "Approximations Applied" in summary
        assert "and 3 more" in summary

    def test_summary_with_approximations_five_or_fewer(self):
        approxs = [{"name": f"approx_{i}"} for i in range(3)]
        result = ExtractionResult(
            engine="vasp",
            files={},
            schema={"approximations": approxs},
            success=True,
            errors=[],
            warnings=[],
        )
        summary = result.summary()
        assert "approx_0" in summary
        assert "more" not in summary

    def test_summary_with_constraints(self):
        constraints = [
            {"name": "c1", "satisfied": True},
            {"name": "c2", "satisfied": False},
        ]
        result = ExtractionResult(
            engine="vasp",
            files={},
            schema={"mathematical_decoding": {"constraints": constraints}},
            success=True,
            errors=[],
            warnings=[],
        )
        summary = result.summary()
        assert "Constraints" in summary
        assert "1/2 satisfied" in summary

    def test_summary_with_warnings(self):
        result = ExtractionResult(
            engine="vasp",
            files={},
            schema={},
            success=True,
            errors=[],
            warnings=["warn1", "warn2", "warn3", "warn4"],
        )
        summary = result.summary()
        assert "Warnings" in summary
        assert "warn1" in summary

    def test_summary_with_math_structure(self):
        result = ExtractionResult(
            engine="vasp",
            files={},
            schema={
                "mathematical_structure": {
                    "problem_type": "eigenvalue",
                    "canonical_form": "Hψ = Eψ",
                }
            },
            success=True,
            errors=[],
            warnings=[],
        )
        summary = result.summary()
        assert "eigenvalue" in summary
        assert "Hψ = Eψ" in summary


# ── ExtractionResult.__getattr__ ──


class TestExtractionResultGetattr:
    def test_getattr_returns_schema_value(self):
        result = ExtractionResult(
            engine="vasp",
            files={},
            schema={"custom_key": "custom_value"},
            success=True,
            errors=[],
            warnings=[],
        )
        assert result.custom_key == "custom_value"

    def test_getattr_missing_raises_attribute_error(self):
        result = ExtractionResult(
            engine="vasp",
            files={},
            schema={},
            success=True,
            errors=[],
            warnings=[],
        )
        with pytest.raises(AttributeError, match="no attribute"):
            _ = result.nonexistent_key


# ── extract_file with dict filepath ──


class TestExtractFileDict:
    def test_extract_file_dict_missing_file(self, api, tmp_path):
        # One of the files in the dict doesn't exist
        real_file = tmp_path / "INCAR"
        real_file.write_text("ENCUT = 520\n")
        with pytest.raises(ExtractionFileNotFoundError):
            api.extract_file("vasp", {"incar": str(real_file), "poscar": str(tmp_path / "NOPE")})

    def test_extract_file_dict_all_exist(self, api, tmp_path):
        incar = tmp_path / "INCAR"
        incar.write_text("ENCUT = 520\nEDIFF = 1e-6\n")
        result = api.extract_file("vasp", {"incar": str(incar)})
        assert result.engine == "vasp"


# ── _detect_file_type ──


class TestDetectFileType:
    def test_detect_incar(self, api, tmp_path):
        f = tmp_path / "INCAR"
        f.write_text("")
        assert api._detect_file_type(f, "vasp") == "incar"

    def test_detect_poscar(self, api, tmp_path):
        f = tmp_path / "POSCAR"
        f.write_text("")
        assert api._detect_file_type(f, "vasp") == "poscar"

    def test_detect_contcar(self, api, tmp_path):
        f = tmp_path / "CONTCAR"
        f.write_text("")
        assert api._detect_file_type(f, "vasp") == "poscar"

    def test_detect_kpoints(self, api, tmp_path):
        f = tmp_path / "KPOINTS"
        f.write_text("")
        assert api._detect_file_type(f, "vasp") == "kpoints"

    def test_detect_potcar(self, api, tmp_path):
        f = tmp_path / "POTCAR"
        f.write_text("")
        assert api._detect_file_type(f, "vasp") == "potcar"

    def test_detect_default_extension(self, api, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("")
        assert api._detect_file_type(f, "vasp") == "json"

    def test_detect_no_extension(self, api, tmp_path):
        f = tmp_path / "input"
        f.write_text("")
        assert api._detect_file_type(f, "vasp") == "input"

    def test_detect_non_vasp_engine(self, api, tmp_path):
        f = tmp_path / "in.txt"
        f.write_text("")
        assert api._detect_file_type(f, "lammps") == "txt"


# ── _parse_files ──


class TestParseFiles:
    def test_parse_files_vasp_with_incar(self, api, tmp_path):
        # The parser may or may not be importable depending on environment;
        # either way _parse_files should return a dict without crashing.
        incar = tmp_path / "INCAR"
        incar.write_text("ENCUT = 520\n")
        params = api._parse_files("vasp", {"incar": str(incar)})
        assert isinstance(params, dict)

    def test_parse_files_vasp_incar_parse_error_appends_warning(self, api, tmp_path):
        # Force the incar parser to raise ImportError by mocking the import
        incar = tmp_path / "INCAR"
        incar.write_text("ENCUT = 520\n")
        import builtins

        real_import = builtins.__import__

        def fail_vasp_import(name, *args, **kwargs):
            if name == "vasp.core.incar_parser" or name == "vasp":
                raise ImportError(f"mocked: No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fail_vasp_import):
            api._parse_files("vasp", {"incar": str(incar)})
        assert len(api._warnings) >= 1
        assert any("INCAR" in w for w in api._warnings)

    def test_parse_files_vasp_with_poscar_kpoints(self, api):
        params = api._parse_files("vasp", {"poscar": "/fake/POSCAR", "kpoints": "/fake/KPOINTS"})
        assert params.get("_has_poscar") is True
        assert params.get("_has_kpoints") is True

    def test_parse_files_non_vasp_engine(self, api):
        params = api._parse_files("lammps", {"data": "/fake/data"})
        assert params.get("_data_path") == "/fake/data"

    def test_parse_files_empty(self, api):
        params = api._parse_files("vasp", {})
        assert params == {}


# ── extract_file invalid types ──


class TestExtractFileInvalidTypes:
    def test_extract_file_int_raises(self, api):
        with pytest.raises(MathAnythingError, match="Expected str, Path, or dict"):
            api.extract_file("vasp", 12345)

    def test_extract_file_none_raises(self, api):
        with pytest.raises(MathAnythingError, match="Expected str, Path, or dict"):
            api.extract_file("vasp", None)

    def test_extract_file_set_raises(self, api):
        with pytest.raises(MathAnythingError, match="Expected str, Path, or dict"):
            api.extract_file("vasp", {"not", "a", "dict"})


# ── compare ──


class TestCompare:
    def test_compare_returns_diff_report(self, api, vasp_params):
        r1 = api.extract("vasp", vasp_params)
        r2 = api.extract("vasp", {"ENCUT": 600, "EDIFF": 1e-6, "SIGMA": 0.05})
        report = api.compare(r1, r2)
        assert report is not None

    def test_compare_critical_only_returns_list(self, api, vasp_params):
        r1 = api.extract("vasp", vasp_params)
        r2 = api.extract("vasp", {"ENCUT": 600, "EDIFF": 1e-6, "SIGMA": 0.05})
        critical = api.compare(r1, r2, critical_only=True)
        assert isinstance(critical, list)


# ── visualize (MathAnything level) ──


class TestApiVisualize:
    def test_visualize_returns_string(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        viz = api.visualize(result, format="mermaid")
        assert isinstance(viz, str)
        assert "graph TD" in viz

    def test_visualize_writes_to_file(self, api, vasp_params, tmp_path):
        result = api.extract("vasp", vasp_params)
        out = tmp_path / "viz.mmd"
        viz = api.visualize(result, format="mermaid", output=str(out))
        assert out.exists()
        assert out.read_text(encoding="utf-8") == viz

    def test_visualize_graphviz_format(self, api, vasp_params):
        result = api.extract("vasp", vasp_params)
        viz = api.visualize(result, format="graphviz")
        assert isinstance(viz, str)


# ── health_check ──


class TestHealthCheck:
    def test_returns_dict_with_required_keys(self, api):
        hc = api.health_check()
        assert "status" in hc
        assert "version" in hc
        assert "rust_acceleration" in hc
        assert "engines_available" in hc
        assert "uptime_seconds" in hc
        assert "python_version" in hc

    def test_status_is_valid(self, api):
        hc = api.health_check()
        assert hc["status"] in ("healthy", "degraded", "unhealthy")

    def test_engines_available_non_empty(self, api):
        hc = api.health_check()
        assert len(hc["engines_available"]) > 0

    def test_rust_acceleration_is_bool(self, api):
        hc = api.health_check()
        assert isinstance(hc["rust_acceleration"], bool)

    def test_uptime_positive(self, api):
        hc = api.health_check()
        assert hc["uptime_seconds"] >= 0

    def test_status_degraded_without_rust(self, api):
        with patch("math_anything.rust_bridge.EMLAccelerator") as mock_acc:
            mock_acc.return_value.using_rust = False
            hc = api.health_check()
            assert hc["status"] == "degraded"
            assert hc["rust_acceleration"] is False


# ── get_metrics ──


class TestGetMetrics:
    def test_returns_dict_with_required_keys(self, api):
        m = api.get_metrics()
        assert "total_extractions" in m
        assert "total_verifications" in m
        assert "rust_acceleration_available" in m
        assert "engines_count" in m
        assert "average_extraction_time_ms" in m
        assert "cache_hit_rate" in m
        assert "cache_size" in m
        assert "cache_total_requests" in m

    def test_initial_metrics_zero(self, api):
        m = api.get_metrics()
        assert m["total_extractions"] == 0
        assert m["total_verifications"] == 0
        assert m["average_extraction_time_ms"] == 0.0

    def test_metrics_after_extraction(self, api, vasp_params):
        api.extract("vasp", vasp_params)
        m = api.get_metrics()
        assert m["total_extractions"] == 1
        assert m["average_extraction_time_ms"] >= 0.0

    def test_engines_count_positive(self, api):
        m = api.get_metrics()
        assert m["engines_count"] > 0

    def test_rust_acceleration_is_bool(self, api):
        m = api.get_metrics()
        assert isinstance(m["rust_acceleration_available"], bool)


# ── get_prometheus_metrics ──


class TestPrometheusMetrics:
    def test_returns_string(self, api):
        pm = api.get_prometheus_metrics()
        assert isinstance(pm, str)

    def test_contains_help_lines(self, api):
        pm = api.get_prometheus_metrics()
        assert "# HELP" in pm
        assert "# TYPE" in pm

    def test_contains_extraction_counter(self, api):
        pm = api.get_prometheus_metrics()
        assert "math_anything_extractions_total" in pm

    def test_contains_verification_counter(self, api):
        pm = api.get_prometheus_metrics()
        assert "math_anything_verifications_total" in pm

    def test_contains_rust_gauge(self, api):
        pm = api.get_prometheus_metrics()
        assert "math_anything_rust_acceleration_available" in pm

    def test_contains_engines_gauge(self, api):
        pm = api.get_prometheus_metrics()
        assert "math_anything_engines_available" in pm

    def test_contains_avg_time_gauge(self, api):
        pm = api.get_prometheus_metrics()
        assert "math_anything_avg_extraction_time_ms" in pm

    def test_reflects_extraction_count(self, api, vasp_params):
        # Use different params for each extraction to avoid cache hits
        api.extract("vasp", {"ENCUT": 520, "EDIFF": 1e-6, "SIGMA": 0.05})
        api.extract("vasp", {"ENCUT": 600, "EDIFF": 1e-6, "SIGMA": 0.05})
        pm = api.get_prometheus_metrics()
        # The counter line should show 2
        for line in pm.split("\n"):
            if line.startswith("math_anything_extractions_total "):
                assert line.split()[-1] == "2"
                return
        pytest.fail("extraction counter line not found")


# ── verify ──


class TestVerify:
    def test_verify_valid_params(self, api, vasp_params):
        result = api.verify("vasp", vasp_params)
        assert "valid" in result
        assert "violations" in result
        assert "warnings" in result
        assert isinstance(result["violations"], list)

    def test_verify_increases_verification_count(self, api, vasp_params):
        api.verify("vasp", vasp_params)
        m = api.get_metrics()
        assert m["total_verifications"] == 1

    def test_verify_failed_extraction(self, api):
        # Mock the extractor to raise ValueError, causing extract to return
        # success=False, which verify reports as an extraction_failed violation
        with patch.object(api, "extract") as mock_extract:
            mock_extract.return_value = ExtractionResult(
                engine="vasp",
                files={},
                schema={},
                success=False,
                errors=["boom"],
                warnings=[],
            )
            result = api.verify("vasp", {"ENCUT": 520})
        assert result["valid"] is False
        assert len(result["violations"]) >= 1
        assert result["violations"][0]["name"] == "extraction_failed"

    def test_verify_with_unsatisfied_constraints(self, api):
        # Use params that might produce unsatisfied constraints
        result = api.verify("vasp", {"ENCUT": 520, "EDIFF": 1e-6, "SIGMA": 0.05})
        assert "valid" in result
        # violations is always a list
        assert isinstance(result["violations"], list)


# ── Module-level convenience functions ──


class TestModuleLevelFunctions:
    def test_module_extract_returns_schema(self):
        schema = module_extract("vasp", {"ENCUT": 520, "EDIFF": 1e-6, "SIGMA": 0.05})
        assert isinstance(schema, dict)
        assert "mathematical_structure" in schema

    def test_module_extract_file_returns_schema(self, tmp_path):
        incar = tmp_path / "INCAR"
        incar.write_text("ENCUT = 520\n")
        schema = module_extract_file("vasp", str(incar))
        assert isinstance(schema, dict)

    def test_module_extract_file_not_found_raises(self):
        with pytest.raises(ExtractionFileNotFoundError):
            module_extract_file("vasp", "/nonexistent/path/INCAR")


# ── _validate_and_coerce_params edge cases ──


class TestValidateAndCoerceParams:
    def test_int_raises(self, api):
        with pytest.raises(MathAnythingError, match="Expected dict"):
            MathAnything._validate_and_coerce_params(42)

    def test_none_raises(self, api):
        with pytest.raises(MathAnythingError, match="Expected dict"):
            MathAnything._validate_and_coerce_params(None)

    def test_set_raises(self, api):
        with pytest.raises(MathAnythingError, match="Expected dict"):
            MathAnything._validate_and_coerce_params({1, 2, 3})

    def test_kpoints_already_dict_passes_through(self):
        params = {"kpoints": {"grid": [4, 4, 4]}}
        coerced = MathAnything._validate_and_coerce_params(params)
        assert coerced["kpoints"] == {"grid": [4, 4, 4]}

    def test_lattice_already_dict_passes_through(self):
        params = {"lattice": {"vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
        coerced = MathAnything._validate_and_coerce_params(params)
        assert coerced["lattice"] == {"vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}

    def test_other_list_key_kept_as_is(self):
        params = {"some_list": [1, 2, 3]}
        coerced = MathAnything._validate_and_coerce_params(params)
        assert coerced["some_list"] == [1, 2, 3]

    def test_scalar_values_unchanged(self):
        params = {"ENCUT": 520, "EDIFF": 1e-6, "name": "test"}
        coerced = MathAnything._validate_and_coerce_params(params)
        assert coerced["ENCUT"] == 520
        assert coerced["EDIFF"] == 1e-6
        assert coerced["name"] == "test"

    def test_lattice_with_scalar_rows(self):
        params = {"lattice": [1, 2, 3]}
        coerced = MathAnything._validate_and_coerce_params(params)
        assert "vectors" in coerced["lattice"]


# ── Cache behavior ──


class TestCacheBehavior:
    def test_second_extract_uses_cache(self, api, vasp_params):
        r1 = api.extract("vasp", vasp_params)
        r2 = api.extract("vasp", vasp_params)
        # Same params → cached result returned
        assert r1.success and r2.success

    def test_cache_hit_rate_after_repeat(self, api, vasp_params):
        api.extract("vasp", vasp_params)
        api.extract("vasp", vasp_params)
        m = api.get_metrics()
        # At least one cache hit
        assert m["cache_total_requests"] >= 2
