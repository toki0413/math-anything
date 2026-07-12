"""Tests for insight engines — VASP, LAMMPS, Abaqus, and base classes."""

import pytest

from math_anything.insight.abaqus_insight import AbaqusInsightEngine
from math_anything.insight.base import (
    GenericInsightEngine,
    InsightBlock,
    InsightEngine,
    explain_schema,
    get_insight_engine,
    register_insight_engine,
)
from math_anything.insight.lammps_insight import LammpsInsightEngine
from math_anything.insight.vasp_insight import VaspInsightEngine
from math_anything.schemas import MathSchema


def _make_schema(raw: dict) -> MathSchema:
    s = MathSchema()
    s.raw_symbols = raw
    return s


# ── InsightBlock ──


class TestInsightBlock:
    def test_to_text_info(self):
        b = InsightBlock(title="Test", content="hello", level="info")
        text = b.to_text()
        assert "[*]" in text
        assert "Test" in text

    def test_to_text_math(self):
        b = InsightBlock(title="Eq", content="E=mc^2", level="math")
        text = b.to_text()
        assert "[MATH]" in text

    def test_to_text_warning(self):
        b = InsightBlock(title="Warn", content="bad", level="warning")
        text = b.to_text()
        assert "[WARN]" in text

    def test_to_text_tip(self):
        b = InsightBlock(title="Tip", content="try this", level="tip")
        text = b.to_text()
        assert "[TIP]" in text

    def test_to_dict(self):
        b = InsightBlock(title="T", content="C", level="info", params=["ENCUT"])
        d = b.to_dict()
        assert d["title"] == "T"
        assert d["params"] == ["ENCUT"]

    def test_default_params(self):
        b = InsightBlock(title="T", content="C")
        assert b.params == []


# ── GenericInsightEngine ──


class TestGenericInsightEngine:
    def test_engine_name(self):
        e = GenericInsightEngine()
        assert e.engine_name == "generic"

    def test_generate_empty_schema(self):
        e = GenericInsightEngine()
        blocks = e.generate(MathSchema())
        assert isinstance(blocks, list)

    def test_generate_with_equations(self):
        from math_anything.schemas import GoverningEquation

        s = MathSchema()
        s.mathematical_model.governing_equations.append(
            GoverningEquation(
                id="eq1",
                type="PDE",
                name="Laplace",
                mathematical_form="nabla^2 u = 0",
            )
        )
        blocks = GenericInsightEngine().generate(s)
        assert any("Governing" in b.title for b in blocks)

    def test_explain_text(self):
        s = MathSchema()
        text = GenericInsightEngine().explain(s, fmt="text")
        assert "Mathematical Insight" in text

    def test_explain_json(self):
        s = MathSchema()
        j = GenericInsightEngine().explain(s, fmt="json")
        assert isinstance(j, str)
        assert "[" in j


# ── Registry ──


class TestInsightRegistry:
    def test_get_unknown_returns_generic(self):
        e = get_insight_engine("nonexistent")
        assert isinstance(e, GenericInsightEngine)

    def test_register_and_get(self):
        e = GenericInsightEngine()
        register_insight_engine(e)
        got = get_insight_engine("generic")
        assert isinstance(got, GenericInsightEngine)

    def test_explain_schema_convenience(self):
        text = explain_schema(MathSchema(), "nonexistent")
        assert isinstance(text, str)


# ── VaspInsightEngine ──


class TestVaspInsightEngine:
    def test_engine_name(self):
        assert VaspInsightEngine().engine_name == "vasp"

    def test_generate_empty_schema(self):
        blocks = VaspInsightEngine().generate(MathSchema())
        assert len(blocks) > 0

    def test_generate_with_encut(self):
        schema = _make_schema({"incar": {"ENCUT": 500, "ISPIN": 2}})
        blocks = VaspInsightEngine().generate(schema)
        assert any("ENCUT" in b.title for b in blocks)

    def test_generate_spin_polarized(self):
        schema = _make_schema({"incar": {"ISPIN": 2}})
        blocks = VaspInsightEngine().generate(schema)
        problem = [b for b in blocks if "Problem" in b.title]
        assert len(problem) > 0
        assert "spin-polarized" in problem[0].content

    def test_generate_spin_restricted(self):
        schema = _make_schema({"incar": {"ISPIN": 1}})
        blocks = VaspInsightEngine().generate(schema)
        problem = [b for b in blocks if "Problem" in b.title]
        assert "spin-restricted" in problem[0].content

    def test_convergence_analysis(self):
        schema = _make_schema({"incar": {"EDIFF": 1e-6, "NELM": 60}})
        blocks = VaspInsightEngine().generate(schema)
        assert any("SCF" in b.title or "Convergence" in b.title for b in blocks)

    def test_consistency_warnings_sigma(self):
        schema = _make_schema({"incar": {"ISMEAR": 0, "SIGMA": 0.5}})
        blocks = VaspInsightEngine().generate(schema)
        warns = [b for b in blocks if b.level == "warning"]
        assert len(warns) > 0

    def test_sampling_insight_with_kpoints(self):
        schema = _make_schema(
            {
                "kpoints": {"mesh": {"subdivisions": [4, 4, 4], "mode": "Gamma"}},
            }
        )
        blocks = VaspInsightEngine().generate(schema)
        assert any("k-Point" in b.title or "Sampling" in b.title for b in blocks)

    def test_wavelength_estimate(self):
        wl = VaspInsightEngine._wavelength_estimate(500.0)
        assert float(wl) > 0

    def test_explain_text_format(self):
        schema = _make_schema({"incar": {"ENCUT": 400}})
        text = VaspInsightEngine().explain(schema, fmt="text")
        assert "Insight" in text

    def test_explain_json_format(self):
        schema = _make_schema({"incar": {"ENCUT": 400}})
        j = VaspInsightEngine().explain(schema, fmt="json")
        assert "[" in j


# ── LammpsInsightEngine ──


class TestLammpsInsightEngine:
    def test_engine_name(self):
        assert LammpsInsightEngine().engine_name == "lammps"

    def test_generate_empty_schema(self):
        blocks = LammpsInsightEngine().generate(MathSchema())
        assert len(blocks) > 0

    def test_generate_with_nve(self):
        schema = _make_schema(
            {
                "fixes": {"f1": {"style": "nve"}},
                "timestep": 1.0,
                "units": "metal",
                "pair_style": "eam/alloy",
                "boundary": "p p p",
            }
        )
        blocks = LammpsInsightEngine().generate(schema)
        ensemble = [b for b in blocks if "Ensemble" in b.title or "Hamiltonian" in b.title]
        assert len(ensemble) > 0
        assert "NVE" in ensemble[0].content

    def test_generate_with_nvt(self):
        schema = _make_schema(
            {
                "fixes": {"f1": {"style": "nvt"}},
                "timestep": 1.0,
                "units": "metal",
                "pair_style": "lj/cut",
                "boundary": "p p p",
            }
        )
        blocks = LammpsInsightEngine().generate(schema)
        ensemble = [b for b in blocks if "Ensemble" in b.title]
        assert "NVT" in ensemble[0].content

    def test_stability_insight_large_timestep(self):
        schema = _make_schema(
            {
                "timestep": 5.0,
                "units": "metal",
                "pair_style": "eam/alloy",
            }
        )
        blocks = LammpsInsightEngine().generate(schema)
        stability = [b for b in blocks if "Stability" in b.title]
        assert len(stability) > 0

    def test_boundary_insight(self):
        _make_schema({"boundary": "p f s"})
        LammpsInsightEngine().generate(MathSchema())
        # Even empty schema should produce boundary insight
        blocks2 = LammpsInsightEngine().generate(_make_schema({"boundary": "p p p"}))
        assert any("Boundary" in b.title for b in blocks2)

    def test_potential_warnings_lj(self):
        schema = _make_schema(
            {
                "pair_style": "lj/cut",
                "timestep": 1.0,
                "units": "metal",
            }
        )
        blocks = LammpsInsightEngine().generate(schema)
        tips = [b for b in blocks if "Potential" in b.title]
        assert len(tips) > 0

    def test_langevin_ensemble(self):
        schema = _make_schema(
            {
                "fixes": {"f1": {"style": "langevin"}},
                "timestep": 0.5,
                "units": "real",
                "pair_style": "lj/cut",
            }
        )
        blocks = LammpsInsightEngine().generate(schema)
        ensemble = [b for b in blocks if "Ensemble" in b.title]
        assert "Langevin" in ensemble[0].content


# ── AbaqusInsightEngine ──


class TestAbaqusInsightEngine:
    def test_engine_name(self):
        assert AbaqusInsightEngine().engine_name == "abaqus"

    def test_generate_empty_schema(self):
        blocks = AbaqusInsightEngine().generate(MathSchema())
        assert isinstance(blocks, list)
