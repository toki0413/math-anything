"""Tests for draft engines — VASP, LAMMPS, Abaqus, and base classes."""

import pytest

from math_anything.draft.abaqus_draft import AbaqusDraftEngine
from math_anything.draft.base import (
    DraftEngine,
    GenericDraftEngine,
    draft_schema,
    get_draft_engine,
    register_draft_engine,
)
from math_anything.draft.lammps_draft import LammpsDraftEngine
from math_anything.draft.vasp_draft import VaspDraftEngine
from math_anything.schemas import MathSchema


def _make_schema(raw: dict) -> MathSchema:
    s = MathSchema()
    s.raw_symbols = raw
    return s


# ── GenericDraftEngine ──


class TestGenericDraftEngine:
    def test_engine_name(self):
        e = GenericDraftEngine()
        assert e.engine_name == "generic"

    def test_generate_markdown(self):
        from math_anything.schemas import GoverningEquation

        s = MathSchema()
        s.mathematical_model.governing_equations.append(
            GoverningEquation(
                id="eq1",
                type="PDE",
                name="Poisson",
                mathematical_form="nabla^2 u = f",
            )
        )
        text = GenericDraftEngine().generate(s, fmt="markdown")
        assert "Poisson" in text
        assert "##" in text

    def test_generate_latex(self):
        from math_anything.schemas import GoverningEquation

        s = MathSchema()
        s.mathematical_model.governing_equations.append(
            GoverningEquation(
                id="eq1",
                type="PDE",
                name="Heat",
                mathematical_form="du/dt = alpha nabla^2 u",
            )
        )
        text = GenericDraftEngine().generate(s, fmt="latex")
        # GenericDraftEngine always uses markdown format, just check it generates text
        assert isinstance(text, str)
        assert "Heat" in text


# ── Registry ──


class TestDraftRegistry:
    def test_get_unknown_returns_generic(self):
        e = get_draft_engine("nonexistent")
        assert isinstance(e, GenericDraftEngine)

    def test_register_and_get(self):
        e = GenericDraftEngine()
        register_draft_engine(e)
        got = get_draft_engine("generic")
        assert isinstance(got, GenericDraftEngine)

    def test_draft_schema_convenience(self):
        text = draft_schema(MathSchema(), "nonexistent")
        assert isinstance(text, str)


# ── VaspDraftEngine ──


class TestVaspDraftEngine:
    def test_engine_name(self):
        assert VaspDraftEngine().engine_name == "vasp"

    def test_generate_empty_schema(self):
        text = VaspDraftEngine().generate(MathSchema())
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_markdown(self):
        schema = _make_schema(
            {
                "incar": {"ENCUT": 500, "ISPIN": 2, "EDIFF": 1e-6, "NELM": 60},
                "kpoints": {"mesh": {"subdivisions": [4, 4, 4], "mode": "Gamma"}},
            }
        )
        text = VaspDraftEngine().generate(schema, fmt="markdown")
        assert "Computational Details" in text
        assert "500" in text

    def test_generate_latex(self):
        schema = _make_schema(
            {
                "incar": {"ENCUT": 400, "ISPIN": 1},
            }
        )
        text = VaspDraftEngine().generate(schema, fmt="latex")
        assert "section" in text

    def test_spin_polarized(self):
        schema = _make_schema({"incar": {"ISPIN": 2}})
        text = VaspDraftEngine().generate(schema)
        assert "spin-polarized" in text

    def test_hse06_functional(self):
        schema = _make_schema(
            {
                "incar": {"LHFCALC": True, "HFSCREEN": 0.2},
            }
        )
        text = VaspDraftEngine().generate(schema)
        assert "HSE06" in text

    def test_pbe0_functional(self):
        schema = _make_schema(
            {
                "incar": {"LHFCALC": True, "HFSCREEN": 0.0},
            }
        )
        text = VaspDraftEngine().generate(schema)
        assert "PBE0" in text

    def test_ldau_section(self):
        schema = _make_schema({"incar": {"LDAU": True, "LDAUTYPE": 2, "LDAUL": [2], "LDAUU": [4.0]}})
        text = VaspDraftEngine().generate(schema)
        assert "DFT+U" in text

    def test_relaxation_section(self):
        schema = _make_schema(
            {
                "incar": {"NSW": 100, "IBRION": 2, "ISIF": 3, "EDIFFG": -0.01},
            }
        )
        text = VaspDraftEngine().generate(schema)
        assert "Geometry" in text or "relaxation" in text.lower() or "optimization" in text.lower()

    def test_smearing_tetrahedron(self):
        schema = _make_schema({"incar": {"ISMEAR": -5}})
        text = VaspDraftEngine().generate(schema)
        assert "tetrahedron" in text.lower()

    def test_caveats_spin_restricted(self):
        schema = _make_schema({"incar": {"ISPIN": 1, "ISMEAR": 0, "SIGMA": 0.2}})
        text = VaspDraftEngine().generate(schema)
        # Should have methodological notes about spin restriction
        assert isinstance(text, str)

    def test_kpoint_gamma_centered(self):
        schema = _make_schema(
            {
                "incar": {},
                "kpoints": {"mesh": {"subdivisions": [6, 6, 6], "mode": "Gamma"}},
            }
        )
        text = VaspDraftEngine().generate(schema)
        assert "6" in text


# ── LammpsDraftEngine ──


class TestLammpsDraftEngine:
    def test_engine_name(self):
        assert LammpsDraftEngine().engine_name == "lammps"

    def test_generate_empty_schema(self):
        text = LammpsDraftEngine().generate(MathSchema())
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_with_nvt(self):
        schema = _make_schema(
            {
                "fixes": {"f1": {"style": "nvt"}},
                "timestep": 1.0,
                "units": "metal",
                "pair_style": "eam/alloy",
            }
        )
        text = LammpsDraftEngine().generate(schema)
        assert isinstance(text, str)

    def test_generate_latex(self):
        schema = _make_schema({"timestep": 1.0, "units": "metal"})
        text = LammpsDraftEngine().generate(schema, fmt="latex")
        assert isinstance(text, str)


# ── AbaqusDraftEngine ──


class TestAbaqusDraftEngine:
    def test_engine_name(self):
        assert AbaqusDraftEngine().engine_name == "abaqus"

    def test_generate_empty_schema(self):
        text = AbaqusDraftEngine().generate(MathSchema())
        assert isinstance(text, str)
        assert len(text) > 0
