"""Pytest configuration and shared fixtures."""

import math
import sys
from pathlib import Path

import pytest

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "engines"))


# Pre-load schemas to resolve circular import in math_anything.structures
# before any engine extractor tries to import math_anything.schemas.
@pytest.fixture(scope="session", autouse=True)
def _preload_schemas():
    try:
        import math_anything.schemas.math_schema  # noqa: F401
    except ImportError:
        pass


# ── Structure fixtures ──

@pytest.fixture
def self_consistent_structure():
    from math_anything.structures import SelfConsistentProblem
    return SelfConsistentProblem(
        operator_type="self_adjoint",
        variational=True,
        bounded_below=True,
        nonlinearity_source="density_dependent",
    )


@pytest.fixture
def navier_stokes_structure():
    from math_anything.structures import NavierStokesProblem
    return NavierStokesProblem(
        regime="incompressible",
        reynolds_number=1000.0,
        has_diffusion=True,
    )


@pytest.fixture
def hamiltonian_structure():
    from math_anything.structures import HamiltonianSystem
    return HamiltonianSystem(
        phase_space_dim=3000,
        symplectic=True,
        reversible=True,
    )


@pytest.fixture
def variational_structure():
    from math_anything.structures import VariationalMinimizationProblem
    return VariationalMinimizationProblem(
        convex=True,
    )


# ── Morphism fixtures ──

@pytest.fixture
def born_oppenheimer():
    from math_anything.morphisms.approximations import BornOppenheimerApproximation
    return BornOppenheimerApproximation()


@pytest.fixture
def plane_wave_truncation():
    from math_anything.morphisms.approximations import PlaneWaveTruncation
    return PlaneWaveTruncation(encut=520)


@pytest.fixture
def kohn_sham_mapping():
    from math_anything.morphisms.approximations import KohnShamMapping
    return KohnShamMapping()


# ── Category engine fixture ──

@pytest.fixture
def category_engine():
    from math_anything.categories.engine import CategoryEngine
    from math_anything.morphisms.approximations import (
        BornOppenheimerApproximation,
        KohnShamMapping,
        PlaneWaveTruncation,
    )
    ce = CategoryEngine()
    ce.register_morphism(BornOppenheimerApproximation())
    ce.register_morphism(KohnShamMapping())
    ce.register_morphism(PlaneWaveTruncation())
    ce.link("born_oppenheimer", "FullManyBody", "ElectronicSchrodinger")
    ce.link("kohn_sham", "ElectronicSchrodinger", "KohnSham_Full")
    ce.link("plane_wave_truncation", "KohnSham_Full", "KohnSham_Truncated")
    return ce


# ── Dimensional fixtures ──

@pytest.fixture
def buckingham_engine():
    from math_anything.dimensional.scaling_group import BuckinghamPiEngine
    return BuckinghamPiEngine()


@pytest.fixture
def fluid_analyzer():
    from math_anything.dimensional.scaling_group import FluidDimensionAnalyzer
    return FluidDimensionAnalyzer()


# ── EML fixtures ──

@pytest.fixture
def eml_conjugacy_engine():
    from math_anything.conjugacy import EMLConjugacyEngine
    return EMLConjugacyEngine()


@pytest.fixture
def eml_constant_engine():
    from math_anything.constants import EMLConstantEngine
    return EMLConstantEngine()


# ── Knowledge graph fixture ──

@pytest.fixture
def temp_kg(tmp_path):
    from math_anything.categories.graph import MathKnowledgeGraph
    kg = MathKnowledgeGraph(tmp_path)
    yield kg
    # cleanup
    if kg.path.exists():
        kg.path.unlink()


# ── Plugin fixture ──

@pytest.fixture
def plugin_registry():
    from math_anything.plugin import PluginRegistry
    return PluginRegistry()
