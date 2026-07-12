"""Tests for UnivalenceChecker and h-level computation."""

import pytest

from math_anything.type_theory.hott import UnivalenceChecker


class TestUnivalenceChecker:
    @pytest.fixture()
    def checker(self):
        return UnivalenceChecker()

    def test_identity_equivalence(self, checker):
        """f(x)=x, g(x)=x is an equivalence."""
        result = checker.verify_equivalence(
            f=lambda x: x,
            g=lambda x: x,
            A_elements=[1, 2, 3],
            B_elements=[1, 2, 3],
            name="identity",
        )
        assert result["is_equivalence"]
        assert result["left_inverse"]
        assert result["right_inverse"]

    def test_negation_equivalence(self, checker):
        """f(x)=-x, g(x)=-x is an equivalence on integers."""
        result = checker.verify_equivalence(
            f=lambda x: -x,
            g=lambda x: -x,
            A_elements=[-2, -1, 0, 1, 2],
            B_elements=[-2, -1, 0, 1, 2],
            name="negation",
        )
        assert result["is_equivalence"]

    def test_non_equivalence(self, checker):
        """f(x)=2x is not an equivalence on {1,2,3}→{1,2,3} (not surjective)."""
        result = checker.verify_equivalence(
            f=lambda x: 2 * x,
            g=lambda x: x // 2,
            A_elements=[1, 2, 3],
            B_elements=[1, 2, 3],
            name="doubling",
        )
        assert not result["is_equivalence"]

    def test_univalence_instance(self, checker):
        """After verifying an equivalence, assert Univalence instance."""
        checker.verify_equivalence(
            f=lambda x: x,
            g=lambda x: x,
            A_elements=[1, 2],
            B_elements=[1, 2],
            name="test_equiv",
        )
        result = checker.univalence_instance("test_equiv")
        assert result["univalence_asserted"]

    def test_univalence_not_verified(self, checker):
        """Univalence instance for unverified equivalence should error."""
        result = checker.univalence_instance("nonexistent")
        assert "error" in result

    def test_h_level_computation(self, checker):
        """Compute h-levels for simple types."""
        result = checker.h_level_computation(
            {
                "Unit": [()],
                "Bool": [True, False],
                "Empty": [],
            }
        )
        assert result["Unit"] == 0  # contractible
        assert result["Bool"] == 2  # set
        assert result["Empty"] == -1  # empty
