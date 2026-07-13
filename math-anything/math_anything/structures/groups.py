"""Group Representation Theory structure family.

GroupRepresentation = D: G → GL(V)

The representation theory of groups maps abstract algebraic symmetries
to concrete linear operators on vector spaces.

Covers:
  - Finite group representations (characters, irreps, projection operators)
  - Space group representations (crystallography, band theory)
  - Wigner-Eckart theorem (selection rules, Clebsch-Gordan decomposition)
  - Band compatibility relations (topological quantum chemistry)

In the math-anything framework, group representations are structural
invariants that classify all possible symmetry-adapted solution spaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata
from .properties import StructuralInvariant


@dataclass
class GroupRepresentation(AbstractMathematicalStructure):
    """Base class for group-theoretic structures.

    D: G → GL(V) maps group elements to invertible linear transformations
    on a representation space V.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Group Representation",
            canonical_form="D: G → GL(V)",
            description="Linear representation of a group on a vector space",
        )
    )
    group_name: str = ""
    group_order: int = 0
    representation_dim: int = 0
    field: str = "complex"

    @property
    def function_space(self) -> str:
        return f"V ≅ ℂ^{self.representation_dim}" if self.representation_dim > 0 else "V (representation space)"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="maschke_complete_reducibility",
                expression="V = ⊕_i V_i (decomposes into irreps for finite groups)",
                theorem="Maschke's Theorem (complete reducibility)",
                condition="self.field == 'complex'",
                affected_quantities=["representation", "decomposition"],
            ),
        ]


@dataclass
class FiniteGroupRepresentation(GroupRepresentation):
    """Full representation data for a finite group.

    Complete data: group order, conjugacy classes, character table,
    and the list of irreducible representations.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Finite Group Representation",
            canonical_form="D: G → GL(V),  G finite",
            description="Complete representation theory data for a finite group",
        )
    )
    conjugacy_classes: int = 0
    character_table: list[list[complex]] = field(default_factory=list)
    irreducible_representations: list[str] = field(default_factory=list)
    class_representatives: list[str] = field(default_factory=list)
    irrep_dimensions: list[int] = field(default_factory=list)
    class_sizes: list[int] = field(default_factory=list)

    def character_array(self) -> np.ndarray:
        """Return the character table as a numpy array (n_irreps x n_classes)."""
        if not self.character_table:
            return np.array([], dtype=complex)
        return np.array(self.character_table, dtype=complex)

    def verify_orthogonality(self) -> dict[str, bool]:
        """Verify the great orthogonality theorem: sum_g chi_i(g)* chi_j(g) = |G| delta_ij."""
        if not self.character_table or self.group_order == 0:
            return {}
        chars = self.character_array()
        sizes = np.array(self.class_sizes if self.class_sizes else [1] * chars.shape[1])
        n_irreps = chars.shape[0]
        results = {}
        for i in range(n_irreps):
            for j in range(i, n_irreps):
                dot = np.sum(sizes * chars[i] * np.conj(chars[j]))
                expected = self.group_order if i == j else 0
                results[f"({self.irreducible_representations[i]},{self.irreducible_representations[j]})"] = (
                    abs(dot - expected) < 1e-10
                )
        return results

    def decompose_representation(self, characters: np.ndarray) -> dict[str, int]:
        """Decompose a representation into irreps using character projection.

        Returns the multiplicity of each irrep.
        """
        if not self.character_table or self.group_order == 0:
            return {}
        chars = self.character_array()
        sizes = np.array(self.class_sizes if self.class_sizes else [1] * chars.shape[1])
        multiplicities = {}
        for i, irrep_name in enumerate(self.irreducible_representations):
            a_i = np.sum(sizes * np.conj(chars[i]) * characters) / self.group_order
            mult = int(round(a_i.real))
            if mult > 0:
                multiplicities[irrep_name] = mult
        return multiplicities

    def verify_burnside(self) -> bool:
        """Verify Burnside's theorem: sum of squared irrep dimensions equals group order."""
        if not self.irrep_dimensions or self.group_order == 0:
            return False
        return sum(d * d for d in self.irrep_dimensions) == self.group_order

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="orthogonality_of_characters",
                    expression="(1/abs:Gabs:) Σ_g χ^a(g)* χ^b(g) = δ_{a,b}",
                    theorem="First Orthogonality Theorem for Characters",
                    affected_quantities=["character_table", "irreps"],
                    proof_sketch="Schur's lemma applied to character inner products",
                ),
                StructuralInvariant(
                    name="orthogonality_of_classes",
                    expression="(1/abs:Gabs:) Σ_a χ^a(C_i)* χ^a(C_j) = (1/abs:C_iabs:) δ_{i,j}",
                    theorem="Second Orthogonality Theorem for Characters",
                    affected_quantities=["character_table", "conjugacy_classes"],
                ),
                StructuralInvariant(
                    name="number_of_irreps_equals_conjugacy_classes",
                    expression=f"n_irrep = {self.conjugacy_classes} (number of irreducible representations = number of conjugacy classes)",  # noqa: E501
                    theorem="Representation Theory of Finite Groups",
                    condition="self.conjugacy_classes > 0",
                    affected_quantities=["irreps", "conjugacy_classes"],
                ),
                StructuralInvariant(
                    name="sum_of_squares_equals_order",
                    expression=f"Σ_i (d_i)² = {self.group_order} (sum of squared dimensions equals group order)",
                    theorem="Burnside's Theorem (dimension sum rule)",
                    condition="self.group_order > 0 and len(self.irrep_dimensions) > 0",
                    affected_quantities=["irrep_dimensions", "group_order"],
                ),
                StructuralInvariant(
                    name="character_inner_product_multiplicity",
                    expression="c_a = (1/abs:Gabs:) Σ_g χ(g)* χ^a(g) = multiplicity of Γ_a in the representation",
                    theorem="Character orthogonality and decomposition",
                    affected_quantities=["character_table", "multiplicity"],
                ),
            ]
        )
        return invariants


@dataclass
class IrreducibleRepresentation(GroupRepresentation):
    """A single irreducible representation (irrep) of a group.

    Γ^(a): G → GL(V_a), with character χ^a(g) = Tr(D^a(g)).

    The fundamental building blocks of all representations.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Irreducible Representation",
            canonical_form="Γ^(a): G → GL(V_a),  dim(V_a) = d_a",
            description="Irreducible representation — building block of all reps",
        )
    )
    dimension: int = 1
    character: list[complex] = field(default_factory=list)
    label: str = ""
    basis_functions: list[str] = field(default_factory=list)

    def character_array(self) -> np.ndarray:
        """Return the character vector as a numpy array."""
        if not self.character:
            return np.array([], dtype=complex)
        return np.array(self.character, dtype=complex)

    def inner_product(
        self, other: "IrreducibleRepresentation", class_sizes: list[int] | None = None, group_order: int = 1
    ) -> float:
        """Compute the character inner product <chi_a, chi_b> = (1/|G|) sum_g chi_a(g)* chi_b(g).

        For irreps this should be delta_ab.
        """
        if not self.character or not other.character:
            return 0.0
        chi_a = self.character_array()
        chi_b = other.character_array()
        sizes = np.array(class_sizes if class_sizes else [1] * len(chi_a))
        return float(np.sum(sizes * np.conj(chi_a) * chi_b).real / group_order)

    def is_irreducible(self, class_sizes: list[int] | None = None, group_order: int = 1) -> bool:
        """Check if this representation is irreducible via <chi, chi> = 1."""
        return abs(self.inner_product(self, class_sizes, group_order) - 1.0) < 1e-10

    @property
    def projection_operator_formula(self) -> str:
        return "P^(Γ)_{ij} = (d_Γ/abs:Gabs:) Σ_g D^Γ(g^{-1})_{ji} g"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="schur_lemma",
                    expression="Any G-equivariant map between irreps is either 0 or scalar multiple of identity",
                    theorem="Schur's Lemma",
                    affected_quantities=["intertwiner", "multiplicity"],
                ),
                StructuralInvariant(
                    name="projection_operator",
                    expression=self.projection_operator_formula,
                    theorem="Group algebra projection operator (Wigner)",
                    affected_quantities=["basis_functions", "symmetry_adaptation"],
                ),
                StructuralInvariant(
                    name="character_orthonormality",
                    expression="⟨χ^a, χ^b⟩ = δ_{a,b} (characters form orthonormal basis of class functions)",
                    theorem="First Orthogonality Theorem",
                    affected_quantities=["character", "class_functions"],
                ),
            ]
        )
        return invariants


@dataclass
class DirectProductDecomposition(GroupRepresentation):
    """Clebsch-Gordan decomposition of tensor products of irreps.

    Γ_a ⊗ Γ_b = ⊕_i c_i Γ_i

    Coefficients: c_i = (1/abs:Gabs:) Σ_g χ^a(g)* χ^b(g)* χ^i(g)
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Direct Product Decomposition",
            canonical_form="Γ_a ⊗ Γ_b = ⊕_i c_i Γ_i",
            description="Clebsch-Gordan decomposition of tensor product representations",
        )
    )
    irrep_a: str = ""
    irrep_b: str = ""
    clebsch_gordan_coefficients: dict[str, int] = field(default_factory=dict)

    def verify_dimension_conservation(self, irrep_dimensions: dict[str, int]) -> bool:
        """Verify d_a * d_b = sum_i c_i * d_i (total dimension conserved)."""
        if not self.clebsch_gordan_coefficients or not irrep_dimensions:
            return False
        d_a = irrep_dimensions.get(self.irrep_a, 0)
        d_b = irrep_dimensions.get(self.irrep_b, 0)
        lhs = d_a * d_b
        rhs = sum(c * irrep_dimensions.get(irrep, 0) for irrep, c in self.clebsch_gordan_coefficients.items())
        return lhs == rhs

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="clebsch_gordan_series",
                    expression=f"c_i = (1/abs:Gabs:) Σ_g χ^{self.irrep_a}(g)* χ^{self.irrep_b}(g)* χ^i(g)",
                    theorem="Character formula for tensor product decomposition",
                    affected_quantities=["character_table", "multiplicity"],
                ),
                StructuralInvariant(
                    name="dimension_consistency",
                    expression="d_a · d_b = Σ_i c_i d_i (total dimension conserved)",
                    theorem="Dimension sum rule for tensor products",
                    affected_quantities=["irrep_dimensions", "multiplicity"],
                ),
                StructuralInvariant(
                    name="identity_in_product_implies_equivalence",
                    expression="Γ_a ≅ Γ_b* (contragredient) iff identity Γ_0 appears in Γ_a ⊗ Γ_b",
                    theorem="Schur orthogonality and representation equivalence",
                    affected_quantities=["irrep", "identity_representation"],
                ),
            ]
        )
        return invariants


@dataclass
class WignerEckartTheorem(GroupRepresentation):
    """Wigner-Eckart theorem: factorization of matrix elements.

    ⟨Γ',γ'abs:T^(Γ)_qabs:Γ,γ⟩ = ⟨Γ,γ;Γ,qabs:Γ',γ'⟩ · ⟨Γ'∥T^(Γ)∥Γ⟩

    Separates matrix elements into:
    - Clebsch-Gordan coefficient (geometry, symmetry)
    - Reduced matrix element (physics, dynamics)
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Wigner-Eckart Theorem",
            canonical_form="⟨Γ',γ'abs:T^(Γ)_qabs:Γ,γ⟩ = ⟨Γ,γ;Γ,qabs:Γ',γ'⟩ · ⟨Γ'∥T^(Γ)∥Γ⟩",
            description="Factorization of matrix elements into geometric and dynamic parts",
        )
    )
    operator_irrep: str = ""
    initial_irrep: str = ""
    final_irrep: str = ""
    operator_component: str = "q"
    reduced_matrix_element: float = 0.0

    def matrix_element(self, cg_coefficient: float) -> complex:
        """Compute the full matrix element: <f|T|O|i> = CG * reduced_matrix_element."""
        return cg_coefficient * self.reduced_matrix_element

    def is_transition_allowed(self, char_table: "CharacterTable | None" = None) -> bool:
        """Check if the transition is allowed by group theory.

        If a CharacterTable is provided, uses selection rules.
        Otherwise falls back to the stored is_allowed flag.
        """
        if char_table is not None:
            try:
                return char_table.selection_rules(self.initial_irrep, self.operator_irrep, self.final_irrep)
            except (ValueError, IndexError):
                return False
        return self.reduced_matrix_element != 0.0

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="wigner_eckart_factorization",
                    expression="⟨Γ',γ'abs:T^(Γ)_qabs:Γ,γ⟩ = ⟨Γ,γ;Γ,qabs:Γ',γ'⟩ · ⟨Γ'∥T^(Γ)∥Γ⟩",
                    theorem="Wigner-Eckart Theorem",
                    affected_quantities=["matrix_element", "clebsch_gordan", "reduced_matrix_element"],
                    proof_sketch="T^(Γ)_q abs:Γ,γ⟩ transforms as Γ ⊗ Γ' under G; project onto ⟨Γ',γ'abs:",
                ),
                StructuralInvariant(
                    name="selection_rule_from_cg",
                    expression="⟨Γ',γ'abs:T^(Γ)_qabs:Γ,γ⟩ = 0 if Γ' ∉ Γ ⊗ Γ_q (Clebsch-Gordan zero)",
                    theorem="Group-theoretic selection rules via Clebsch-Gordan",
                    condition="self.final_irrep and self.initial_irrep and self.operator_irrep",
                    affected_quantities=["transition_amplitude", "selection_rule"],
                ),
            ]
        )
        return invariants


@dataclass
class SpaceGroupRepresentation(GroupRepresentation):
    """Representation of a crystallographic space group (No. 1-230).

    Space group irreps are labeled by:
    - k-vector in the Brillouin zone
    - Small representation of the little group G_k
    - Induced to full star representation
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Space Group Representation",
            canonical_form="Ind_{G_k}^{G} Γ_k (induced representation from little group)",
            description="Representation of a 3D crystallographic space group at wavevector k",
        )
    )
    space_group_number: int = 1
    bravais_lattice: str = ""
    k_vector: tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    little_group: str = ""
    small_representation: str = ""
    star_of_k: list[tuple[float, float, float]] = field(default_factory=list)
    full_star_dimension: int = 0

    def star_degeneracy(self) -> int:
        """Return the number of k-vectors in the star (time-reversal included)."""
        return len(self.star_of_k) if self.star_of_k else 1

    def bloch_phase(self, lattice_vector: tuple[float, float, float]) -> complex:
        """Compute the Bloch phase exp(ik·R) for a lattice translation R."""
        k = np.array(self.k_vector)
        R = np.array(lattice_vector)
        return np.exp(1j * np.dot(k, R))  # type: ignore[no-any-return]

    def brillouin_zone_fractional(self, k_frac: tuple[float, float, float]) -> np.ndarray:
        """Convert fractional k-vector to Cartesian coordinates (cubic lattice assumed)."""
        return np.array(k_frac, dtype=float)

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="bloch_theorem",
                    expression="ψ_k(r + R) = exp(ik·R) ψ_k(r) (periodic part times phase)",
                    theorem="Bloch's Theorem for periodic potentials",
                    affected_quantities=["wavefunction", "crystal_momentum"],
                ),
                StructuralInvariant(
                    name="little_group_induced_representation",
                    expression="Full space group irrep = Ind_{G_k}^G (small representation of G_k)",
                    theorem="Induced representation (Frobenius reciprocity)",
                    affected_quantities=["band_structure", "irrep_label"],
                ),
                StructuralInvariant(
                    name="star_degeneracy",
                    expression=f"Full irrep dimension = abs:star(k)abs: · dim(small rep) = {self.full_star_dimension}",
                    theorem="Star of k-vector gives band degeneracy at general points",
                    condition="self.full_star_dimension > 0",
                    affected_quantities=["band_degeneracy", "star_of_k"],
                ),
                StructuralInvariant(
                    name="frobenius_reciprocity",
                    expression="Multiplicity of Γ in Ind_H^G ρ = multiplicity of ρ in Res_H^G Γ",
                    theorem="Frobenius Reciprocity Theorem",
                    affected_quantities=["induced_representation", "restricted_representation"],
                ),
            ]
        )
        return invariants


@dataclass
class BandCompatibility(GroupRepresentation):
    """Band compatibility relations at high-symmetry points.

    When moving from a high-symmetry k-point to a lower-symmetry line/plane,
    irreps must decompose compatibly. This constrains band connectivity.

    Essential for topological quantum chemistry and band topology analysis.
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Band Compatibility",
            canonical_form="Γ^{high} ↓ G_{low} = ⊕_i n_i Γ^{low}_i (subduced representation decomposition)",
            description="Compatibility relations for band connectivity at high-symmetry points",
        )
    )
    high_symmetry_point: str = ""
    low_symmetry_direction: str = ""
    high_symmetry_irrep: str = ""
    compatible_irreps: list[str] = field(default_factory=list)
    band_connectivity_graph: dict[str, list[str]] = field(default_factory=list)  # type: ignore[arg-type]

    def subduction_multiplicities(
        self, char_table_high: "CharacterTable | None" = None, char_table_low: "CharacterTable | None" = None
    ) -> dict[str, int]:
        """Compute subduction multiplicities for the high-symmetry irrep going to lower symmetry.

        Returns which low-symmetry irreps appear and with what multiplicity.
        """
        if char_table_low is not None:
            try:
                i_high = char_table_high.irreps.index(self.high_symmetry_irrep)  # type: ignore[union-attr]
                high_chars = char_table_high.characters[i_high]  # type: ignore[union-attr]
                return char_table_low.decompose_representation(high_chars)
            except (ValueError, AttributeError):
                pass
        # Fallback: each compatible irrep appears once
        return {irrep: 1 for irrep in self.compatible_irreps}

    def band_crossing_allowed(self, irrep1: str, irrep2: str) -> bool:
        """Check if two bands with different irreps can cross (allowed if different irreps)."""
        return irrep1 != irrep2

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="subduction_criterion",
                    expression="Γ_high ↓ G_low = ⊕_i n_i Γ^i_low (branching rule)",
                    theorem="Subduced representation decomposition",
                    affected_quantities=["band_connectivity", "irrep"],
                ),
                StructuralInvariant(
                    name="compatibility_relation",
                    expression=f"Irrep {self.high_symmetry_irrep} at {self.high_symmetry_point} branches to {self.compatible_irreps} along {self.low_symmetry_direction}",  # noqa: E501
                    theorem="Compatibility relations for space group irreps (Bouckaert-Smoluchowski-Wigner)",
                    condition="self.high_symmetry_point and self.low_symmetry_direction",
                    affected_quantities=["band_structure", "irrep_label"],
                ),
                StructuralInvariant(
                    name="band_connectivity_law",
                    expression="Bands cannot cross if they belong to the same irrep at all k along a line",
                    theorem="Non-crossing rule (Wigner-von Neumann)",
                    affected_quantities=["band_crossing", "irrep"],
                ),
            ]
        )
        return invariants


@dataclass
class SelectionRules(GroupRepresentation):
    """Selection rules from representation theory.

    A transition ⟨ψ_fabs:Ôabs:ψ_i⟩ is allowed only if:
    Γ_f ⊗ Γ_Ô ⊗ Γ_i contains the identity representation.

    This determines:
    - IR/Raman activity of vibrational modes
    - Optical transition rules (dipole, quadrupole)
    - Neutron scattering cross-sections
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Selection Rules",
            canonical_form="⟨ψ_fabs:Ôabs:ψ_i⟩ ≠ 0 only if Γ_f* ⊗ Γ_Ô ⊗ Γ_i ⊇ Γ_1",
            description="Group-theoretic conditions for non-zero transition matrix elements",
        )
    )
    transition_type: str = ""
    operator_irrep_label: str = ""
    initial_state_irrep: str = ""
    final_state_irrep: str = ""
    is_allowed: bool = True

    def check_selection_rule(self, char_table: "CharacterTable | None" = None) -> bool:
        """Numerically check if the transition is allowed.

        Uses character table orthogonality if available, otherwise returns stored flag.
        """
        if char_table is not None:
            try:
                return char_table.selection_rules(
                    self.initial_state_irrep,
                    self.operator_irrep_label,
                    self.final_state_irrep,
                )
            except (ValueError, IndexError):
                return False
        return self.is_allowed

    def multiplicity_in_product(self, char_table: "CharacterTable") -> int:
        """Compute the multiplicity of the totally symmetric irrep in Gamma_f* ⊗ Gamma_O ⊗ Gamma_i."""
        try:
            i_init = char_table.irreps.index(self.initial_state_irrep)
            i_op = char_table.irreps.index(self.operator_irrep_label)
            i_fin = char_table.irreps.index(self.final_state_irrep)
            product_chars = (
                np.conj(char_table.characters[i_fin]) * char_table.characters[i_op] * char_table.characters[i_init]
            )
            decomp = char_table.decompose_representation(product_chars)
            return decomp.get(char_table.irreps[0], 0)
        except (ValueError, IndexError):
            return 0

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="selection_rule_criterion",
                    expression="⟨ψ_fabs:Ôabs:ψ_i⟩ ≠ 0 iff Γ_f* ⊗ Γ_Ô ⊗ Γ_i contains Γ_1 (totally symmetric)",
                    theorem="Group-theoretic selection rules",
                    affected_quantities=["transition_amplitude", "irrep"],
                ),
                StructuralInvariant(
                    name="infrared_activity_rule",
                    expression="IR active iff irrep of mode ⊂ Γ_{translation} (x, y, z)",
                    theorem="IR spectroscopy selection rules",
                    condition="self.transition_type == 'dipole'",
                    affected_quantities=["infrared_intensity", "vibrational_mode"],
                ),
                StructuralInvariant(
                    name="raman_activity_rule",
                    expression="Raman active iff irrep of mode ⊂ Γ_{quadratic} (x², xy, xz, etc.)",
                    theorem="Raman spectroscopy selection rules",
                    condition="self.transition_type == 'raman'",
                    affected_quantities=["raman_intensity", "vibrational_mode"],
                ),
                StructuralInvariant(
                    name="laporte_rule",
                    expression="For centrosymmetric systems: only g↔u transitions allowed (parity change)",
                    theorem="Laporte selection rule (inversion symmetry)",
                    affected_quantities=["parity", "transition_amplitude"],
                ),
            ]
        )
        return invariants


# ──────────────────────────────────────────────────────────────────────────────
# Computational group representation theory classes
# ──────────────────────────────────────────────────────────────────────────────


class CharacterTable:
    """Character table for a finite group — the foundation of representation theory.

    Stores the full character table and provides methods for orthogonality
    verification, representation decomposition, direct product decomposition,
    and selection rule computation.
    """

    def __init__(
        self, group_name: str, classes: list[str], irreps: list[str], characters: np.ndarray, class_sizes: list[int]
    ):
        self.group_name = group_name
        self.classes = classes
        self.irreps = irreps
        self.characters = np.asarray(characters, dtype=float)
        self.class_sizes = class_sizes
        self.group_order = sum(class_sizes)

    def verify_orthogonality(self) -> dict[str, bool]:
        """Verify the great orthogonality theorem: sum_g chi_i(g)* chi_j(g) = |G| delta_ij."""
        n_irreps = len(self.irreps)
        sizes = np.array(self.class_sizes, dtype=float)
        results = {}
        for i in range(n_irreps):
            for j in range(i, n_irreps):
                dot = np.sum(sizes * self.characters[i] * np.conj(self.characters[j]))
                expected = self.group_order if i == j else 0
                results[f"({self.irreps[i]},{self.irreps[j]})"] = abs(dot - expected) < 1e-10
        return results

    def decompose_representation(self, characters: np.ndarray) -> dict[str, int]:
        """Decompose a representation into irreps using character projection.

        Returns the multiplicity of each irrep.
        """
        sizes = np.array(self.class_sizes, dtype=float)
        multiplicities = {}
        for i, irrep_name in enumerate(self.irreps):
            a_i = np.sum(sizes * np.conj(self.characters[i]) * characters) / self.group_order
            mult = int(round(a_i.real))
            if mult > 0:
                multiplicities[irrep_name] = mult
        return multiplicities

    def direct_product(self, other: "CharacterTable") -> dict[str, int]:
        """Compute the direct product decomposition of all irrep pairs.

        Returns which irreps appear in each product.
        """
        if self.group_name != other.group_name:
            return {"error": "Direct product requires same group"}  # type: ignore[dict-item]
        results = {}
        for i, irrep_i in enumerate(self.irreps):
            for j, irrep_j in enumerate(self.irreps):
                product_chars = self.characters[i] * self.characters[j]
                decomp = self.decompose_representation(product_chars)
                results[f"{irrep_i}⊗{irrep_j}"] = decomp
        return results  # type: ignore[return-value]

    def selection_rules(self, irrep_initial: str, irrep_operator: str, irrep_final: str) -> bool:
        """Check if a transition is allowed by group theory.

        A transition <psi_f|O|psi_i> is allowed if the product
        Gamma_f ⊗ Gamma_O ⊗ Gamma_i contains the totally symmetric
        representation (first irrep, usually A1).
        """
        i_init = self.irreps.index(irrep_initial)
        i_op = self.irreps.index(irrep_operator)
        i_fin = self.irreps.index(irrep_final)

        product_chars = self.characters[i_fin] * self.characters[i_op] * self.characters[i_init]
        decomp = self.decompose_representation(product_chars)

        a1_name = self.irreps[0]
        return decomp.get(a1_name, 0) > 0

    def degeneracy(self, irrep_name: str) -> int:
        """Return the degeneracy (dimension) of an irrep.

        The character of the identity element equals the irrep dimension.
        """
        i = self.irreps.index(irrep_name)
        return int(self.characters[i, 0])


class ClebschGordanCoefficients:
    """Clebsch-Gordan coefficients for coupling representations.

    Computes which irreps appear in the tensor product of two irreps
    and the coupling multiplicities.
    """

    def __init__(self, char_table: CharacterTable):
        self.char_table = char_table

    def compute(self, irrep1: str, irrep2: str) -> dict[str, int]:
        """Compute CG decomposition: which irreps appear in irrep1 ⊗ irrep2."""
        i1 = self.char_table.irreps.index(irrep1)
        i2 = self.char_table.irreps.index(irrep2)
        product_chars = self.char_table.characters[i1] * self.char_table.characters[i2]
        return self.char_table.decompose_representation(product_chars)

    def coupling_coefficient(self, irrep1: str, irrep2: str, irrep3: str) -> float:
        """Compute the coupling coefficient (multiplicity) of irrep3 in irrep1 ⊗ irrep2."""
        decomp = self.compute(irrep1, irrep2)
        return float(decomp.get(irrep3, 0))


class BandStructureAnalysis:
    """Analyze band structure using group theory.

    Provides methods for compatibility relations, degeneracy computation,
    and band crossing analysis.
    """

    def __init__(self, char_table: CharacterTable):
        self.char_table = char_table

    def compatibility_relations(self, high_sym_irrep: str, subgroup_table: CharacterTable) -> dict[str, int]:
        """Determine which irreps of a subgroup are compatible with a high-symmetry point irrep.

        Essential for understanding band connectivity.
        """
        i_high = self.char_table.irreps.index(high_sym_irrep)
        high_chars = self.char_table.characters[i_high]
        return subgroup_table.decompose_representation(high_chars)

    def degeneracy(self, irrep_name: str) -> int:
        """Return the degeneracy (dimension) of an irrep."""
        return self.char_table.degeneracy(irrep_name)

    def band_crossing_allowed(self, irrep1: str, irrep2: str) -> bool:
        """Check if two bands with different irreps can cross (allowed if different irreps)."""
        return irrep1 != irrep2


# ──────────────────────────────────────────────────────────────────────────────
# Predefined character tables for common point groups
# ──────────────────────────────────────────────────────────────────────────────


def character_table_c2v() -> CharacterTable:
    """C2v point group character table."""
    return CharacterTable(
        group_name="C2v",
        classes=["E", "C2", "σv", "σv'"],
        irreps=["A1", "A2", "B1", "B2"],
        characters=np.array(
            [
                [1, 1, 1, 1],  # A1
                [1, 1, -1, -1],  # A2
                [1, -1, 1, -1],  # B1
                [1, -1, -1, 1],  # B2
            ],
            dtype=float,
        ),
        class_sizes=[1, 1, 1, 1],
    )


def character_table_d2h() -> CharacterTable:
    """D2h point group character table."""
    return CharacterTable(
        group_name="D2h",
        classes=["E", "C2z", "C2y", "C2x", "i", "σxy", "σxz", "σyz"],
        irreps=["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"],
        characters=np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],  # Ag
                [1, 1, -1, -1, 1, 1, -1, -1],  # B1g
                [1, -1, 1, -1, 1, -1, 1, -1],  # B2g
                [1, -1, -1, 1, 1, -1, -1, 1],  # B3g
                [1, 1, 1, 1, -1, -1, -1, -1],  # Au
                [1, 1, -1, -1, -1, -1, 1, 1],  # B1u
                [1, -1, 1, -1, -1, 1, -1, 1],  # B2u
                [1, -1, -1, 1, -1, 1, 1, -1],  # B3u
            ],
            dtype=float,
        ),
        class_sizes=[1, 1, 1, 1, 1, 1, 1, 1],
    )


def character_table_oh() -> CharacterTable:
    """Oh point group character table (cubic)."""
    return CharacterTable(
        group_name="Oh",
        classes=["E", "8C3", "6C2", "6C4", "3C2'", "i", "6S4", "8S6", "3σh", "6σd"],
        irreps=["A1g", "A2g", "Eg", "T1g", "T2g", "A1u", "A2u", "Eu", "T1u", "T2u"],
        characters=np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # A1g
                [1, 1, -1, -1, 1, 1, -1, 1, 1, -1],  # A2g
                [2, -1, 0, 0, 2, 2, 0, -1, 2, 0],  # Eg
                [3, 0, -1, 1, -1, 3, 1, 0, -1, -1],  # T1g
                [3, 0, 1, -1, -1, 3, -1, 0, -1, 1],  # T2g
                [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],  # A1u
                [1, 1, -1, -1, 1, -1, 1, -1, -1, 1],  # A2u
                [2, -1, 0, 0, 2, -2, 0, 1, -2, 0],  # Eu
                [3, 0, -1, 1, -1, -3, -1, 0, 1, 1],  # T1u
                [3, 0, 1, -1, -1, -3, 1, 0, 1, -1],  # T2u
            ],
            dtype=float,
        ),
        class_sizes=[1, 8, 6, 6, 3, 1, 6, 8, 3, 6],
    )


def character_table_td() -> CharacterTable:
    """Td point group character table (tetrahedral)."""
    return CharacterTable(
        group_name="Td",
        classes=["E", "8C3", "3C2", "6S4", "6σd"],
        irreps=["A1", "A2", "E", "T1", "T2"],
        characters=np.array(
            [
                [1, 1, 1, 1, 1],  # A1
                [1, 1, 1, -1, -1],  # A2
                [2, -1, 2, 0, 0],  # E
                [3, 0, -1, 1, -1],  # T1
                [3, 0, -1, -1, 1],  # T2
            ],
            dtype=float,
        ),
        class_sizes=[1, 8, 3, 6, 6],
    )
