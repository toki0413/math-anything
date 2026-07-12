"""Tests for engine adapters — parameter translation and registry."""

import pytest

from math_anything.adapters import (
    ENGINE_DOMAIN_MAP,
    list_all_engines,
    list_supported_engines,
    translate_params,
)


class TestVaspTranslation:
    def test_basic_translation(self):
        result = translate_params("vasp", {"ENCUT": 500, "EDIFF": 1e-6, "ISPIN": 2})
        assert result["domain"] == "dft"
        assert result["domain_params"]["ecutwfc"] == 500
        assert result["domain_params"]["scf_tol"] == 1e-6
        assert result["domain_params"]["n_spin"] == 2

    def test_smearing_params(self):
        result = translate_params("vasp", {"ISMEAR": 1, "SIGMA": 0.2})
        assert result["domain_params"]["smearing_type"] == 1
        assert result["domain_params"]["smearing_width"] == 0.2

    def test_algorithm_mapping(self):
        result = translate_params("vasp", {"ALGO": "Normal"})
        assert result["domain_params"]["algorithm"] == "davidson"

    def test_dft_u(self):
        result = translate_params("vasp", {"LDAU": True})
        assert result["domain_params"]["dft_u"] is True

    def test_original_params_preserved(self):
        params = {"ENCUT": 500}
        result = translate_params("vasp", params)
        assert result["original_params"] == params


class TestQETranslation:
    def test_basic_translation(self):
        result = translate_params("qe", {"ecutwfc": 60, "conv_thr": 1e-8})
        assert result["domain"] == "dft"
        assert result["domain_params"]["ecutwfc"] == 60
        assert result["domain_params"]["scf_tol"] == 1e-8

    def test_ecutrho(self):
        result = translate_params("qe", {"ecutrho": 240})
        assert result["domain_params"]["ecutrho"] == 240


class TestLammpsTranslation:
    def test_basic_translation(self):
        result = translate_params("lammps", {"timestep": 0.001, "pair_style": "lj/cut"})
        assert result["domain"] == "md"
        assert result["domain_params"]["dt"] == 0.001
        assert result["domain_params"]["force_field"] == "lj/cut"

    def test_ensemble_npt(self):
        result = translate_params("lammps", {"fix_npt": True})
        assert result["domain_params"]["ensemble"] == "NPT"

    def test_ensemble_nve(self):
        result = translate_params("lammps", {"fix_nve": True})
        assert result["domain_params"]["ensemble"] == "NVE"


class TestGromacsTranslation:
    def test_basic_translation(self):
        result = translate_params("gromacs", {"dt": 0.002, "nsteps": 100000})
        assert result["domain"] == "md"
        assert result["domain_params"]["dt"] == 0.002
        assert result["domain_params"]["n_steps"] == 100000

    def test_thermostat_barostat(self):
        result = translate_params("gromacs", {"tcoupl": "V-rescale", "pcoupl": "Parrinello-Rahman"})
        assert result["domain_params"]["thermostat"] == "V-rescale"
        assert result["domain_params"]["barostat"] == "Parrinello-Rahman"


class TestAbaqusTranslation:
    def test_nlgeom(self):
        result = translate_params("abaqus", {"NLGEOM": True})
        assert result["domain"] == "fem"
        assert result["domain_params"]["geometric_nonlinear"] is True

    def test_element_type(self):
        result = translate_params("abaqus", {"ELEMENT_TYPE": "C3D8R"})
        assert result["domain_params"]["element_type"] == "C3D8R"


class TestOpenFOAMTranslation:
    def test_basic_translation(self):
        result = translate_params("openfoam", {"deltaT": 0.001, "endTime": 10.0})
        assert result["domain"] == "cfd"
        assert result["domain_params"]["dt"] == 0.001
        assert result["domain_params"]["end_time"] == 10.0

    def test_turbulence_model(self):
        result = translate_params("openfoam", {"turbulenceModel": "kEpsilon"})
        assert result["domain_params"]["turbulence_model"] == "kEpsilon"


class TestUnknownEngine:
    def test_unknown_engine_returns_unknown_domain(self):
        result = translate_params("unknown_engine", {"param": 1})
        assert result["domain"] == "unknown"

    def test_unknown_engine_passes_through(self):
        result = translate_params("unknown_engine", {"param": 1})
        assert result["domain_params"]["param"] == 1

    def test_case_insensitive(self):
        result = translate_params("VASP", {"ENCUT": 500})
        assert result["domain"] == "dft"


class TestEngineRegistry:
    def test_list_supported(self):
        engines = list_supported_engines()
        assert "vasp" in engines
        assert "lammps" in engines
        assert "gromacs" in engines
        assert "abaqus" in engines
        assert "openfoam" in engines
        assert "qe" in engines

    def test_list_all(self):
        engines = list_all_engines()
        assert len(engines) > 10
        # Should include both supported and unsupported engines
        assert "vasp" in engines
        assert "cst" in engines

    def test_domain_map_completeness(self):
        """Every supported engine should be in the domain map."""
        supported = list_supported_engines()
        for engine in supported:
            assert engine in ENGINE_DOMAIN_MAP
