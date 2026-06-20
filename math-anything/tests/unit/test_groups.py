"""Tests for group representation theory — CharacterTable, CG coefficients, selection rules."""

import numpy as np
import pytest

from math_anything.structures.groups import (
    BandStructureAnalysis,
    CharacterTable,
    ClebschGordanCoefficients,
    character_table_c2v,
    character_table_d2h,
    character_table_oh,
    character_table_td,
)


class TestCharacterTableC2v:
    @pytest.fixture()
    def ct(self):
        return character_table_c2v()

    def test_group_name(self, ct):
        assert ct.group_name == "C2v"

    def test_group_order(self, ct):
        assert ct.group_order == 4

    def test_irrep_count(self, ct):
        assert len(ct.irreps) == 4

    def test_orthogonality(self, ct):
        result = ct.verify_orthogonality()
        for key, val in result.items():
            assert val, f"Orthogonality failed for {key}"

    def test_decompose_a1(self, ct):
        chars = np.array([1, 1, 1, 1])
        decomp = ct.decompose_representation(chars)
        assert decomp.get("A1", 0) == 1

    def test_selection_rules_allowed(self, ct):
        # A1 ⊗ B1 ⊗ B1: product chars = 1*1*1, 1*(-1)*(-1), 1*1*1, 1*(-1)*(-1) = [1,1,1,1] → A1
        result = ct.selection_rules("A1", "B1", "B1")
        assert result

    def test_selection_rules_forbidden(self, ct):
        # A1 → A2 via B1: A1⊗B1⊗A2 chars = 1*1*1, 1*(-1)*1, 1*1*(-1), 1*(-1)*(-1) = [1,-1,-1,1] → B2
        # B2 does not contain A1 → forbidden
        result = ct.selection_rules("A1", "B1", "A2")
        assert not result


class TestCharacterTableD2h:
    @pytest.fixture()
    def ct(self):
        return character_table_d2h()

    def test_group_order(self, ct):
        assert ct.group_order == 8

    def test_orthogonality(self, ct):
        result = ct.verify_orthogonality()
        for key, val in result.items():
            assert val, f"Orthogonality failed for {key}"

    def test_decompose_ag(self, ct):
        chars = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        decomp = ct.decompose_representation(chars)
        assert decomp.get("Ag", 0) == 1


class TestCharacterTableOh:
    @pytest.fixture()
    def ct(self):
        return character_table_oh()

    def test_group_order(self, ct):
        assert ct.group_order == 48

    def test_orthogonality(self, ct):
        result = ct.verify_orthogonality()
        for key, val in result.items():
            assert val, f"Orthogonality failed for {key}"

    def test_eg_degeneracy(self, ct):
        assert ct.degeneracy("Eg") == 2

    def test_t1u_degeneracy(self, ct):
        assert ct.degeneracy("T1u") == 3


class TestCharacterTableTd:
    @pytest.fixture()
    def ct(self):
        return character_table_td()

    def test_group_order(self, ct):
        assert ct.group_order == 24

    def test_orthogonality(self, ct):
        result = ct.verify_orthogonality()
        for key, val in result.items():
            assert val, f"Orthogonality failed for {key}"


class TestClebschGordan:
    def test_c2v_product(self):
        ct = character_table_c2v()
        cg = ClebschGordanCoefficients(ct)
        result = cg.compute("A1", "B1")
        assert result.get("B1", 0) == 1

    def test_c2v_coupling_coefficient(self):
        ct = character_table_c2v()
        cg = ClebschGordanCoefficients(ct)
        coeff = cg.coupling_coefficient("A1", "B1", "B1")
        assert coeff == 1.0

    def test_c2v_zero_coefficient(self):
        ct = character_table_c2v()
        cg = ClebschGordanCoefficients(ct)
        coeff = cg.coupling_coefficient("A1", "A2", "B1")
        assert coeff == 0.0


class TestBandStructure:
    def test_degeneracy(self):
        ct = character_table_oh()
        bs = BandStructureAnalysis(ct)
        assert bs.degeneracy("A1g") == 1
        assert bs.degeneracy("Eg") == 2
        assert bs.degeneracy("T1u") == 3

    def test_band_crossing(self):
        ct = character_table_oh()
        bs = BandStructureAnalysis(ct)
        assert bs.band_crossing_allowed("Eg", "T1u")
        assert not bs.band_crossing_allowed("Eg", "Eg")
