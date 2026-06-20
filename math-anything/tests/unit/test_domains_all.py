"""Tests for all 7 physics domains — registry, analysis, morphism chains, conservation fields."""

import pytest

from math_anything.domains import DOMAIN_REGISTRY, list_domains, get_domain
from math_anything.domains.base import Domain, DomainAnalysis


class TestDomainRegistry:

    def test_seven_domains_registered(self):
        domains = list_domains()
        assert len(domains) == 7
        for expected in ["cfd", "dft", "em", "fem", "md", "phase_field", "qc"]:
            assert expected in domains

    def test_get_domain_valid(self):
        cls = get_domain("dft")
        assert cls is not None

    def test_get_domain_invalid(self):
        with pytest.raises(KeyError, match="not found"):
            get_domain("nonexistent_domain")


class TestAllDomainsAnalysis:

    @pytest.mark.parametrize("name", list_domains())
    def test_each_domain_analyzable(self, name):
        cls = DOMAIN_REGISTRY[name]
        dom = cls()
        analysis = dom.analyze()
        assert isinstance(analysis, DomainAnalysis)
        assert analysis.domain_name == dom.name

    @pytest.mark.parametrize("name", list_domains())
    def test_each_domain_has_morphism_chain(self, name):
        cls = DOMAIN_REGISTRY[name]
        dom = cls()
        chain = dom.build_morphism_chain()
        assert len(chain) > 0, f"Domain {name} has empty morphism chain"

    @pytest.mark.parametrize("name", list_domains())
    def test_each_domain_has_conservation_field(self, name):
        cls = DOMAIN_REGISTRY[name]
        dom = cls()
        cf = dom.build_conservation_field()
        assert isinstance(cf, dict)
        assert "conservation_laws" in cf
        assert len(cf["conservation_laws"]) > 0

    @pytest.mark.parametrize("name", list_domains())
    def test_each_domain_analysis_has_summary(self, name):
        cls = DOMAIN_REGISTRY[name]
        dom = cls()
        analysis = dom.analyze()
        summary = analysis.summary()
        assert isinstance(summary, str)
        assert name in summary or dom.name in summary

    @pytest.mark.parametrize("name", list_domains())
    def test_each_domain_analysis_to_dict(self, name):
        cls = DOMAIN_REGISTRY[name]
        dom = cls()
        analysis = dom.analyze()
        d = analysis.to_dict()
        assert "domain_name" in d
        assert "preserved" in d
        assert "lost" in d
        assert "emerged" in d


class TestDomainComparison:

    def test_dft_vs_md(self):
        dft = DOMAIN_REGISTRY["dft"]()
        md = DOMAIN_REGISTRY["md"]()
        comparison = dft.compare_with(md)
        assert isinstance(comparison, dict)
        assert "common_preserved" in comparison
        assert "only_in_a" in comparison
        assert "only_in_b" in comparison

    def test_cfd_vs_fem(self):
        cfd = DOMAIN_REGISTRY["cfd"]()
        fem = DOMAIN_REGISTRY["fem"]()
        comparison = cfd.compare_with(fem)
        assert isinstance(comparison, dict)

    def test_same_domain_comparison(self):
        dft1 = DOMAIN_REGISTRY["dft"]()
        dft2 = DOMAIN_REGISTRY["dft"]()
        comparison = dft1.compare_with(dft2)
        # Same domain should have no unique invariants
        assert comparison["only_in_a"] == []
        assert comparison["only_in_b"] == []
        assert comparison["a_lost"] == comparison["b_lost"]


class TestEMDomain:

    def test_fdtd_method(self):
        em = DOMAIN_REGISTRY["em"]({"method": "FDTD", "n_cells": 100})
        chain = em.build_morphism_chain()
        assert len(chain) > 0
        # Should contain spatial discretization step
        names = [s["name"] for s in chain]
        assert any("fdtd" in n.lower() for n in names)

    def test_pml_boundary(self):
        em = DOMAIN_REGISTRY["em"]({"method": "FDTD", "pml_layers": 10})
        chain = em.build_morphism_chain()
        has_pml = any("pml" in step.get("name", "").lower() for step in chain)
        assert has_pml

    def test_no_pml_when_zero(self):
        em = DOMAIN_REGISTRY["em"]({"method": "FDTD", "pml_layers": 0})
        chain = em.build_morphism_chain()
        has_pml = any("pml" in step.get("name", "").lower() for step in chain)
        assert not has_pml

    def test_cfl_condition(self):
        em = DOMAIN_REGISTRY["em"]()
        cf = em.build_conservation_field()
        assert "cfl_condition" in cf
        assert cf["cfl_condition"]["max_dt"] > 0


class TestQCDomain:

    def test_default_hf_method(self):
        qc = DOMAIN_REGISTRY["qc"]()
        chain = qc.build_morphism_chain()
        names = [s["name"] for s in chain]
        assert any("hf" in n.lower() for n in names)

    def test_ccsd_method(self):
        qc = DOMAIN_REGISTRY["qc"]({"method": "CCSD(T)", "basis_set": "cc-pVTZ"})
        chain = qc.build_morphism_chain()
        names = [s["name"] for s in chain]
        assert any("ccsd" in n.lower() for n in names)

    def test_dft_method(self):
        qc = DOMAIN_REGISTRY["qc"]({"method": "DFT"})
        chain = qc.build_morphism_chain()
        names = [s["name"] for s in chain]
        assert any("dft" in n.lower() for n in names)

    def test_relativistic(self):
        qc = DOMAIN_REGISTRY["qc"]({"relativistic": True})
        chain = qc.build_morphism_chain()
        names = [s["name"] for s in chain]
        assert any("relativistic" in n.lower() for n in names)


class TestPhaseFieldDomain:

    def test_cahn_hilliard(self):
        pf = DOMAIN_REGISTRY["phase_field"]({"model": "Cahn-Hilliard"})
        cf = pf.build_conservation_field()
        assert "mass_conservation" in cf["conservation_laws"]

    def test_allen_cahn(self):
        pf = DOMAIN_REGISTRY["phase_field"]({"model": "Allen-Cahn"})
        cf = pf.build_conservation_field()
        assert "mass_conservation" not in cf["conservation_laws"]
        assert "free_energy_dissipation" in cf["conservation_laws"]

    def test_cahn_hilliard_cfl(self):
        pf = DOMAIN_REGISTRY["phase_field"]({"model": "Cahn-Hilliard"})
        cf = pf.build_conservation_field()
        assert "cfl_condition" in cf
        assert cf["cfl_condition"]["type"] == "4th_order_diffusion"

    def test_allen_cahn_cfl(self):
        pf = DOMAIN_REGISTRY["phase_field"]({"model": "Allen-Cahn"})
        cf = pf.build_conservation_field()
        assert cf["cfl_condition"]["type"] == "2nd_order_diffusion"

    def test_anisotropy(self):
        pf = DOMAIN_REGISTRY["phase_field"]({"anisotropy": True})
        chain = pf.build_morphism_chain()
        names = [s["name"] for s in chain]
        assert any("anisotrop" in n.lower() for n in names)

    def test_coupled_mechanics(self):
        pf = DOMAIN_REGISTRY["phase_field"]({"coupled_mechanics": True})
        chain = pf.build_morphism_chain()
        names = [s["name"] for s in chain]
        assert any("coupled" in n.lower() for n in names)


class TestDomainWhatIsLost:

    @pytest.mark.parametrize("name", list_domains())
    def test_what_is_lost_returns_list(self, name):
        cls = DOMAIN_REGISTRY[name]
        dom = cls()
        lost = dom.what_is_lost()
        assert isinstance(lost, list)

    @pytest.mark.parametrize("name", list_domains())
    def test_what_is_kept_returns_list(self, name):
        cls = DOMAIN_REGISTRY[name]
        dom = cls()
        kept = dom.what_is_kept()
        assert isinstance(kept, list)
