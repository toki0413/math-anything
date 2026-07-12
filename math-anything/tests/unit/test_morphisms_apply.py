"""Tests for morphism apply() methods — verifying state transformations."""

import pytest

from math_anything.morphisms.cfd import (
    IncompressibilityMorphism,
    LESFilteringMorphism,
    ReynoldsDecompositionMorphism,
    TurbulenceModelClosureMorphism,
)
from math_anything.morphisms.dft import (
    BornOppenheimerApproximation,
    ExchangeCorrelationApproximation,
    KohnShamMapping,
    PlaneWaveTruncation,
    SCFIterationMorphism,
)
from math_anything.morphisms.md import (
    ClassicalLimitMorphism,
    ForceFieldMorphism,
)
from math_anything.morphisms.quantum import (
    HartreeFockMorphism,
    PostHartreeFockMorphism,
)
from math_anything.morphisms.surrogate import (
    DiffuseInterfaceMorphism,
    MLSurrogateMorphism,
)
from math_anything.morphisms.symmetry import (
    BlochTheoremMorphism,
    SymmetryReductionMorphism,
)


class TestDFTMorphismApply:
    def test_born_oppenheimer_apply(self):
        bo = BornOppenheimerApproximation()
        result = bo.apply({"n_electrons": 10})
        assert result["adiabatic_approximation"] is True
        assert result["nuclear_quantum_effects"] is False
        assert result["vibronic_coupling"] is False

    def test_kohn_sham_apply(self):
        ks = KohnShamMapping()
        result = ks.apply({"n_electrons": 10})
        assert result["self_consistent"] is True
        assert result["explicit_correlation"] is False
        assert result["n_orbitals"] == 5  # 10 electrons -> 5 orbitals

    def test_kohn_sham_apply_odd_electrons(self):
        ks = KohnShamMapping()
        result = ks.apply({"n_electrons": 5})
        assert result["n_orbitals"] == 3  # 5 electrons -> 3 orbitals

    def test_plane_wave_apply(self):
        pw = PlaneWaveTruncation()
        result = pw.apply({"ecutwfc": 50.0})
        assert result["basis_completeness"] is False
        assert result["ecutwfc"] == 50.0
        assert "n_pw" in result
        assert result["n_pw"] > 0

    def test_scf_iteration_apply(self):
        scf = SCFIterationMorphism()
        state = {"scf_iteration": 0, "density_change": 1.0}
        result = scf.apply(state)
        assert result["scf_iteration"] == 1
        assert result["density_change"] < 1.0

    def test_scf_iteration_converges(self):
        scf = SCFIterationMorphism()
        state = {"scf_iteration": 0, "density_change": 1.0}
        for _ in range(30):
            state = scf.apply(state)
        assert state["converged"] is True

    def test_xc_apply(self):
        xc = ExchangeCorrelationApproximation()
        result = xc.apply({"xc_functional": "PBE"})
        assert result["exact_xc"] is False
        assert result["xc_functional"] == "PBE"


class TestCFDMorphismApply:
    def test_incompressibility_apply(self):
        m = IncompressibilityMorphism()
        result = m.apply({})
        assert result["compressible"] is False
        assert result["divergence_free"] is True
        assert result["acoustic_waves"] is False

    def test_reynolds_apply(self):
        m = ReynoldsDecompositionMorphism()
        result = m.apply({})
        assert result["reynolds_decomposed"] is True
        assert result["mean_flow"] is True

    def test_turbulence_closure_apply(self):
        m = TurbulenceModelClosureMorphism()
        result = m.apply({"turbulence_model": "RANS"})
        assert result["turbulence_modeled"] is True
        assert result["model_type"] == "RANS"

    def test_les_filtering_apply(self):
        m = LESFilteringMorphism()
        result = m.apply({"filter_width": 0.05})
        assert result["filtered"] is True
        assert result["filter_width"] == 0.05


class TestMDMorphismApply:
    def test_classical_limit_apply(self):
        m = ClassicalLimitMorphism()
        result = m.apply({})
        assert result["quantum_effects"] is False
        assert result["classical_mechanics"] is True

    def test_force_field_apply(self):
        m = ForceFieldMorphism()
        result = m.apply({"force_field": "LJ"})
        assert result["ab_initio"] is False
        assert result["force_field"] == "LJ"

    def test_force_field_apply_default(self):
        m = ForceFieldMorphism()
        result = m.apply({})
        assert result["force_field"] == "Lennard-Jones"


class TestQuantumMorphismApply:
    def test_hartree_fock_apply(self):
        hf = HartreeFockMorphism()
        result = hf.apply({})
        assert isinstance(result, dict)

    def test_post_hartree_fock_apply(self):
        phf = PostHartreeFockMorphism()
        result = phf.apply({})
        assert isinstance(result, dict)


class TestSurrogateMorphismApply:
    def test_ml_surrogate_apply(self):
        m = MLSurrogateMorphism()
        result = m.apply({})
        assert isinstance(result, dict)

    def test_diffuse_interface_apply(self):
        m = DiffuseInterfaceMorphism()
        result = m.apply({})
        assert isinstance(result, dict)


class TestSymmetryMorphismApply:
    def test_symmetry_reduction_apply(self):
        m = SymmetryReductionMorphism()
        result = m.apply({})
        assert isinstance(result, dict)

    def test_bloch_theorem_apply(self):
        m = BlochTheoremMorphism()
        result = m.apply({})
        assert isinstance(result, dict)


class TestMorphismChainApply:
    def test_dft_chain_sequential(self):
        """Apply the full DFT morphism chain sequentially."""
        state = {"n_electrons": 10}

        bo = BornOppenheimerApproximation()
        state = bo.apply(state)
        assert state["adiabatic_approximation"] is True

        ks = KohnShamMapping()
        state = ks.apply(state)
        assert state["self_consistent"] is True

        pw = PlaneWaveTruncation()
        state = pw.apply(state)
        assert state["basis_completeness"] is False

        xc = ExchangeCorrelationApproximation()
        state = xc.apply(state)
        assert state["exact_xc"] is False

    def test_cfd_chain_sequential(self):
        """Apply the CFD morphism chain sequentially."""
        state = {}

        incomp = IncompressibilityMorphism()
        state = incomp.apply(state)
        assert state["divergence_free"] is True

        reynolds = ReynoldsDecompositionMorphism()
        state = reynolds.apply(state)
        assert state["reynolds_decomposed"] is True

    def test_md_chain_sequential(self):
        """Apply the MD morphism chain sequentially."""
        state = {}

        classical = ClassicalLimitMorphism()
        state = classical.apply(state)
        assert state["classical_mechanics"] is True

        ff = ForceFieldMorphism()
        state = ff.apply(state)
        assert state["ab_initio"] is False
