"""Symmetry-related morphisms for physics and materials science.

Symmetry morphisms describe how a physical system is transformed under
group actions: reduction by irreducible representations, Bloch theorem
for periodic systems, projection onto specific irreps, and selection
rules for transitions.

Each morphism tracks what structure is preserved, lost, and introduced
when exploiting symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import Morphism, MorphismCategory


@dataclass
class SymmetryReductionMorphism(Morphism):
    """Reduction of a full problem to an irreducible subspace.

    Given a symmetry group G acting on a Hilbert space H, the space
    decomposes into a direct sum of isotypic components:
      H = ⨁_{Γ ∈ Irr(G)} H_Γ

    Each H_Γ is the subspace transforming according to the irreducible
    representation Γ. The Hamiltonian (or other operator) becomes
    block-diagonal in this basis:
      H = diag(H_{Γ₁}, H_{Γ₂}, ...)

    Computational savings: eigenvalue problem of size N reduces to
    problems of size d_Γ·N/|G|, with cost proportional to
    Σ (d_Γ·N/|G|)² ≈ N²/|G| (for large |G|).

    Invariants:
      - Energy spectrum within each irrep is preserved
      - Block-diagonal structure of the Hamiltonian
    """

    name: str = "symmetry_reduction"
    source_type: str = "FullHilbertSpace"
    target_type: str = "IrreducibleSubspace"
    category: str = MorphismCategory.PROJECTION

    group: str = ""
    irrep: str = ""
    irrep_dimension: int = 1
    full_dimension: int = 1000
    is_block_diagonalized: bool = True

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "energy_spectrum_within_each_irrep",
            "eigenvalue_accuracy_for_target_irrep",
            "symmetry_labels_of_states",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "coupling_between_different_irreps",
            "full_Hamiltonian_dense_representation",
            "off_diagonal_blocks_in_original_basis",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "block_diagonal_hamiltonian",
            "computational_savings ~ (dim/irrep_dim)^2",
            "symmetry_adapted_basis",
        ]
    )
    kernel_description: str = (
        "Subspace orthogonal to the target irrep; modes transforming under different irreducible representations"
    )

    @property
    def mathematical_form(self) -> str:
        factor = (self.full_dimension / max(self.irrep_dimension, 1)) ** 2
        return (
            f"P^Gamma: H -> H_Gamma = (d_Gamma/|G|) sum_{{g in G}} chi^Gamma(g)* g\n"
            f"H_Gamma = P^Gamma H P^Gamma (block-diagonal form)\n"
            f"dim(H_Gamma)/dim(H) = {self.irrep_dimension}/{self.full_dimension} "
            f"(speedup ~ {factor:.0f}x)"
        )

    def apply(self, state: dict) -> dict:
        """Apply symmetry reduction: project onto the target irreducible subspace."""
        result = dict(state)
        result["full_space"] = False
        result["irrep"] = state.get("irrep", self.irrep)
        result["block_diagonal"] = True
        return result


@dataclass
class BlochTheoremMorphism(Morphism):
    """Bloch theorem: reduction of periodic system to k-point sampling.

    For a system with lattice periodicity, the wavefunction can be
    written as a Bloch function:
      ψ_{n,k}(r) = e^{ik·r} u_{n,k}(r)
    where u_{n,k} has the periodicity of the lattice.

    The infinite periodic system reduces to solving within a single
    unit cell for each k-point in the Brillouin zone. Practical
    calculations sample a finite set of k-points.

    Invariants:
      - Ground state energy (with sufficient k-point density)
      - Charge density (with sufficient k-point density)
    Lost:
      - Continuous band dispersion (sampled at discrete k-points)
    """

    name: str = "bloch_theorem"
    source_type: str = "InfinitePeriodicSystem"
    target_type: str = "UnitCell_kPointSampling"
    category: str = MorphismCategory.PROJECTION

    n_kpoints: int = 1
    kpoint_mesh: str = "1x1x1"  # e.g., "8x8x8"
    is_gamma_only: bool = False
    has_time_reversal_symmetry: bool = True

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "ground_state_energy (converged w.r.t. k-points)",
            "charge_density (converged w.r.t. k-points)",
            "lattice_periodicity",
            "band_eigenvalues_at_sampled_k_points",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "continuous_band_dispersion",
            "full_k_dependence_of_observables",
            "precise_band_extrema (if not at sampled k-points)",
            "van_Hove_singularities (require dense k-sampling)",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "k_point_discretization_error",
            "finite_k_point_set_truncation",
            "band_interpolation_uncertainty",
        ]
    )
    kernel_description: str = "Unsolved k-points: Brillouin zone modes not included in the finite k-point mesh"

    @property
    def mathematical_form(self) -> str:
        t_rev = "T-symmetric " if self.has_time_reversal_symmetry else ""
        return (
            f"psi_{{n,k}}(r+R) = psi_{{n,k}}(r) e^{{ik·R}}  (Bloch theorem)\n"
            f"k in {self.kpoint_mesh} {t_rev}({self.n_kpoints} k-points)\n"
            f"[-½∇² + V_eff(r)] u_{{n,k}}(r) = ε_{{n,k}} u_{{n,k}}(r)"
        )

    def apply(self, state: dict) -> dict:
        """Apply Bloch theorem: reduce infinite periodic system to unit-cell k-point sampling."""
        result = dict(state)
        result["infinite_system"] = False
        result["n_kpoints"] = state.get("n_kpoints", self.n_kpoints)
        result["unit_cell_calculation"] = True
        return result


@dataclass
class ProjectionOperatorMorphism(Morphism):
    """Projection onto a specific irreducible representation.

    The projection operator P^Γ projects the full Hilbert space onto
    the subspace transforming as irrep Γ:

      P^Γ = (d_Γ / |G|) Σ_{g∈G} χ^Γ(g)* g

    where d_Γ = dim(Γ) is the dimension of the irrep, |G| is the group
    order, and χ^Γ(g) is the character of group element g in irrep Γ.

    For constructing symmetry-adapted basis functions:
      φ^Γ = P^Γ φ  (from an arbitrary starting function φ)

    The projector is idempotent (P^Γ P^Γ = P^Γ) and Hermitian
    (P^Γ)† = P^Γ for unitary representations.
    """

    name: str = "projection_operator"
    source_type: str = "FullFunctionSpace"
    target_type: str = "IrrepSubspace_Gamma"
    category: str = MorphismCategory.PROJECTION

    group: str = ""
    irrep: str = ""
    irrep_dimension: int = 1
    group_order: int = 1
    is_orthogonal_projection: bool = True

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "symmetry_character_of_irrep_Gamma",
            "inner_product_structure_within_irrep",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "components_belonging_to_other_irreps",
            "full_function_basis",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "symmetry_adapted_basis_vector",
            "block_diagonal_representation_of_operators",
            "selection_rules_on_matrix_elements",
        ]
    )
    kernel_description: str = "⊕_{Γ' ≠ Γ} H_{Γ'}: subspace of all other irreducible representations"

    @property
    def mathematical_form(self) -> str:
        return (
            f"P^Γ = ({self.irrep_dimension}/{self.group_order}) "
            f"Σ_{{g∈{self.group}}} χ^Γ(g)* g\n"
            f"P^Γ H P^Γ = H_Γ  (block corresponding to irrep {self.irrep})\n"
            f"P^Γ · P^{self.irrep} = P^{self.irrep}  (idempotent)"
        )

    def apply(self, state: dict) -> dict:
        """Apply projection operator: project state onto the target irrep subspace."""
        result = dict(state)
        result["projected"] = True
        result["target_irrep"] = state.get("irrep", self.irrep)
        result["orthogonal_projection"] = self.is_orthogonal_projection
        return result


@dataclass
class SelectionRuleMorphism(Morphism):
    """Selection rule: determining whether a transition is allowed by symmetry.

    The matrix element ⟨f|O|i⟩ for a transition between initial state
    |i⟩ (in irrep Γ_i) and final state |f⟩ (in irrep Γ_f) induced by
    operator O (in irrep Γ_O) is non-zero only if:

      Γ_f ⊗ Γ_O ⊗ Γ_i contains the totally symmetric irrep A_1

    i.e., the direct product of the three representations must contain
    the identity representation.

    Equivalent formulation via the Wigner-Eckart theorem:
      ⟨Γ_f, m_f| O^{Γ_O}_{m_O} |Γ_i, m_i⟩
      = ⟨Γ_i m_i; Γ_O m_O | Γ_f m_f⟩ · ⟨Γ_f || O || Γ_i⟩
    where the first factor is a Clebsch-Gordan coefficient and the
    second is the reduced matrix element (independent of m).

    This morphism filters transitions: allowed → pass through zero
    kernel; forbidden → mapped to zero (kernel).
    """

    name: str = "selection_rule"
    source_type: str = "AllTransitions"
    target_type: str = "SymmetryAllowedTransitions"
    category: str = MorphismCategory.RESTRICTION

    initial_irrep: str = ""
    operator_irrep: str = ""
    final_irrep: str = ""
    is_allowed: bool = True
    transition_type: str = ""  # "electric_dipole", "magnetic_dipole", "raman", "vibrational"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "total_symmetry_of_the_system",
            "allowed_transition_intensities",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "forbidden_transition_matrix_elements",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "symmetry_forbidden_transitions_identically_zero",
            "wigner_eckart_factorization",
        ]
    )
    kernel_description: str = "Forbidden transitions: those where A_1 ∉ Γ_f ⊗ Γ_O ⊗ Γ_i"

    @property
    def mathematical_form(self) -> str:
        operator_str = f" ({self.transition_type})" if self.transition_type else ""
        if self.is_allowed:
            rule = "A_1 ∈ Γ_f ⊗ Γ_O ⊗ Γ_i → ⟨f|O|i⟩ ≠ 0 (allowed)"
        else:
            rule = "A_1 ∉ Γ_f ⊗ Γ_O ⊗ Γ_i → ⟨f|O|i⟩ = 0 (forbidden)"
        return (
            f"Γ_f ⊗ Γ_O ⊗ Γ_i ⊇ A_1 ?\n"
            f"  Γ_i = {self.initial_irrep},  "
            f"Γ_O = {self.operator_irrep}{operator_str},  "
            f"Γ_f = {self.final_irrep}\n"
            f"  Result: {rule}"
        )

    def apply(self, state: dict) -> dict:
        """Apply selection rule: filter transitions by symmetry-allowed irreps."""
        result = dict(state)
        result["selection_rule_applied"] = True
        result["transition_allowed"] = self.is_allowed
        result["initial_irrep"] = state.get("initial_irrep", self.initial_irrep)
        result["final_irrep"] = state.get("final_irrep", self.final_irrep)
        return result
