"""Unit tests for ConservationMatrixField and related types."""

import pytest

from math_anything.structures.conservation_field import (
    ConservationMatrixField,
    FieldConservedQuantity,
    NoetherCurrent,
    NOETHER_CORRESPONDENCE,
    TIME_TRANSLATION_CONSERVATION,
)
from math_anything.structures.enums import SymmetryGroup
from math_anything.structures._core import StructuralInvariant


# ── Build method names for parametrized tests ──

BUILD_METHODS = [
    "build_from_euler_equations",
    "build_from_navier_stokes",
    "build_from_schrodinger",
    "build_from_maxwell",
    "build_from_elasticity",
    "build_from_heat_equation",
    "build_from_advection_diffusion",
    "build_from_mhd",
    "build_from_kohn_sham",
    "build_from_boltzmann",
    "build_from_shallow_water",
    "build_from_wave_equation",
    "build_from_dirac",
    "build_from_klein_gordon",
    "build_from_einstein_field",
    "build_from_schrodinger_nonlinear",
    "build_from_vlasov",
    "build_from_hartree_fock",
]


class TestConservationMatrixField:

    @pytest.mark.parametrize("method_name", BUILD_METHODS)
    def test_build_method_returns_self(self, method_name):
        field = ConservationMatrixField()
        result = getattr(field, method_name)()
        assert result is field

    @pytest.mark.parametrize("method_name", BUILD_METHODS)
    def test_build_method_populates_conserved_quantities(self, method_name):
        field = ConservationMatrixField()
        getattr(field, method_name)()
        assert len(field.conserved_quantities) > 0

    @pytest.mark.parametrize("method_name", BUILD_METHODS)
    def test_build_method_populates_noether_currents(self, method_name):
        field = ConservationMatrixField()
        getattr(field, method_name)()
        assert len(field.noether_currents) > 0

    def test_n_conserved_property(self):
        field = ConservationMatrixField()
        assert field.n_conserved == 0
        field.build_from_euler_equations()
        assert field.n_conserved == 3

    def test_noether_map_euler(self):
        field = ConservationMatrixField()
        field.build_from_euler_equations()
        nmap = field.noether_map
        assert SymmetryGroup.GAUGE_U1 in nmap
        assert nmap[SymmetryGroup.GAUGE_U1].name == "mass"
        assert SymmetryGroup.TRANSLATION in nmap
        assert nmap[SymmetryGroup.TRANSLATION].name == "momentum"

    def test_noether_map_empty_when_no_symmetry(self):
        field = ConservationMatrixField()
        field.build_from_heat_equation()
        nmap = field.noether_map
        # heat equation conserved quantity has symmetry=None
        assert len(nmap) == 0 or all(k is not None for k in nmap)

    def test_is_hamiltonian_schrodinger(self):
        field = ConservationMatrixField()
        field.build_from_schrodinger()
        assert field.is_hamiltonian is True

    def test_is_hamiltonian_nls(self):
        field = ConservationMatrixField()
        field.build_from_schrodinger_nonlinear()
        assert field.is_hamiltonian is True

    def test_is_hamiltonian_wave(self):
        field = ConservationMatrixField()
        field.build_from_wave_equation()
        assert field.is_hamiltonian is True

    def test_is_not_hamiltonian_euler(self):
        field = ConservationMatrixField()
        field.build_from_euler_equations()
        assert field.is_hamiltonian is False

    def test_is_not_hamiltonian_heat(self):
        field = ConservationMatrixField()
        field.build_from_heat_equation()
        assert field.is_hamiltonian is False

    def test_structural_invariants_returns_proper_type(self):
        field = ConservationMatrixField()
        field.build_from_euler_equations()
        invariants = field.structural_invariants()
        assert len(invariants) > 0
        for inv in invariants:
            assert isinstance(inv, StructuralInvariant)
            assert inv.name
            assert inv.expression

    def test_structural_invariants_hamiltonian_adds_symplectic(self):
        field = ConservationMatrixField()
        field.build_from_schrodinger()
        invariants = field.structural_invariants()
        names = [inv.name for inv in invariants]
        assert "symplectic_structure" in names

    def test_structural_invariants_non_hamiltonian_no_symplectic(self):
        field = ConservationMatrixField()
        field.build_from_euler_equations()
        invariants = field.structural_invariants()
        names = [inv.name for inv in invariants]
        assert "symplectic_structure" not in names

    def test_to_dict_serialization(self):
        field = ConservationMatrixField()
        field.build_from_euler_equations()
        d = field.to_dict()
        assert "n_conserved" in d
        assert d["n_conserved"] == 3
        assert "conserved_quantities" in d
        assert len(d["conserved_quantities"]) == 3
        assert "is_hamiltonian" in d
        assert "noether_map" in d
        assert "noether_currents" in d
        assert "hamiltonian" in d

    def test_to_dict_hamiltonian_field(self):
        field = ConservationMatrixField()
        field.build_from_schrodinger()
        d = field.to_dict()
        assert d["is_hamiltonian"] is True
        assert d["hamiltonian"] != ""

    def test_is_hyperbolic_default(self):
        field = ConservationMatrixField()
        assert field.is_hyperbolic is False

    def test_characteristic_speeds_default(self):
        field = ConservationMatrixField()
        assert field.characteristic_speeds is None


class TestFieldConservedQuantity:

    def test_creation(self):
        q = FieldConservedQuantity(
            name="mass",
            symbol="ρ",
            expression="dρ/dt + div(ρu) = 0",
            symmetry=SymmetryGroup.GAUGE_U1,
            spatial_dim=3,
        )
        assert q.name == "mass"
        assert q.symbol == "ρ"
        assert q.symmetry == SymmetryGroup.GAUGE_U1
        assert q.spatial_dim == 3

    def test_default_spatial_dim(self):
        q = FieldConservedQuantity(name="energy", symbol="E")
        assert q.spatial_dim == 3

    def test_default_symmetry_is_none(self):
        q = FieldConservedQuantity(name="energy", symbol="E")
        assert q.symmetry is None

    def test_default_expression_is_empty(self):
        q = FieldConservedQuantity(name="energy", symbol="E")
        assert q.expression == ""


class TestNoetherCurrent:

    def test_creation(self):
        nc = NoetherCurrent(
            name="mass_current",
            symmetry=SymmetryGroup.GAUGE_U1,
            current_components=["ρ", "ρu"],
            continuity_equation="dρ/dt + div(ρu) = 0",
        )
        assert nc.name == "mass_current"
        assert nc.symmetry == SymmetryGroup.GAUGE_U1
        assert len(nc.current_components) == 2
        assert nc.continuity_equation != ""

    def test_default_components_empty(self):
        nc = NoetherCurrent(
            name="test_current",
            symmetry=SymmetryGroup.TRANSLATION,
        )
        assert nc.current_components == []
        assert nc.continuity_equation == ""


class TestNoetherCorrespondence:

    def test_mapping_exists(self):
        assert isinstance(NOETHER_CORRESPONDENCE, dict)
        assert len(NOETHER_CORRESPONDENCE) > 0

    def test_translation_maps_to_momentum(self):
        assert NOETHER_CORRESPONDENCE[SymmetryGroup.TRANSLATION] == "momentum_conservation"

    def test_rotation_maps_to_angular_momentum(self):
        assert NOETHER_CORRESPONDENCE[SymmetryGroup.ROTATION_SO3] == "angular_momentum_conservation"

    def test_gauge_u1_maps_to_charge(self):
        assert NOETHER_CORRESPONDENCE[SymmetryGroup.GAUGE_U1] == "charge_conservation"

    def test_lorentz_maps_to_four_momentum(self):
        assert NOETHER_CORRESPONDENCE[SymmetryGroup.LORENTZ] == "four_momentum_conservation"

    def test_time_translation_constant(self):
        assert TIME_TRANSLATION_CONSERVATION == "energy_conservation"

    def test_all_keys_are_symmetry_groups(self):
        for key in NOETHER_CORRESPONDENCE:
            assert isinstance(key, SymmetryGroup)

    def test_all_values_are_strings(self):
        for val in NOETHER_CORRESPONDENCE.values():
            assert isinstance(val, str)
