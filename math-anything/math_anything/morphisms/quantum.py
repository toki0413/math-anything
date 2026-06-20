"""量子化学态射链.

Hartree-Fock 态射 → 后 HF 关联修正
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import Morphism


@dataclass
class HartreeFockMorphism(Morphism):
    """Hartree-Fock 态射：全电子 Schrödinger → 单行列式.

    变分极小化 ⟨Ψ|Ĥ|Ψ⟩，限制 Ψ 为单个 Slater 行列式。
    """

    name: str = "hartree_fock"
    source_type: str = "ExactElectronicSchrodinger"
    target_type: str = "HartreeFockEquations"
    condition: str = ""

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "antisymmetry (Pauli principle)",
            "variational_upper_bound",
            "total_energy_as_expectation_value",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "electron_correlation_energy",
            "dispersion_interactions",
            "bond_breaking_accuracy",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "fock_operator_invariance",
            "brillouin_theorem",
            "size_consistency_issue (for restricted HF)",
        ]
    )
    kernel_description: str = "Correlated electron motion (beyond mean field)"

    @property
    def mathematical_form(self) -> str:
        return "F ψ_i = ε_i ψ_i,  F = h + Σ_j (J_j - K_j)"

    def apply(self, state: dict) -> dict:
        """Apply Hartree-Fock approximation: replace correlated motion with mean field."""
        result = dict(state)
        result["correlation"] = "mean_field"
        result["method"] = "HF"
        return result


@dataclass
class PostHartreeFockMorphism(Morphism):
    """后 HF 关联修正态射.

    HF 参考态 → 考虑关联的方法（MP2, CCSD, CI, ...）
    各种方法构成一个态射层次：精度递增，计算成本指数递增。
    """

    name: str = "post_hf_correlation"
    source_type: str = "HartreeFock"
    target_type: str = "CorrelatedWavefunction"
    condition: str = ""

    method: str = "CCSD_T"  # "MP2", "CCSD", "CCSD(T)", "CISD", "FCI"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "Hartree_Fock_reference_quality",
            "variational_upper_bound (CI methods)",
        ]
    )

    @property
    def invariants_lost(self) -> list[str]:
        lost = []
        if "mp2" in self.method.lower():
            lost.append("variational_bound (MP2 is perturbative)")
        if "cisd" in self.method.lower():
            lost.append("size_consistency (truncated CI)")
        return lost

    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "correlation_energy",
            "method_specific_truncation_error",
        ]
    )
    kernel_description: str = "Higher-order excitations not included in the method"

    @property
    def mathematical_form(self) -> str:
        forms = {
            "MP2": "E_corr^(2) = Σ (|⟨ij||ab⟩|²)/(ε_i+ε_j-ε_a-ε_b)",
            "CCSD": "Ψ = exp(T_1 + T_2) Φ_0",
            "CCSD_T": "E = E_CCSD + E_(T) (perturbative triples)",
            "FCI": "Ψ = Σ c_I Φ_I (all determinants in basis)",
        }
        return forms.get(self.method, f"Ψ_corr = f(Φ_HF; {self.method})")

    def apply(self, state: dict) -> dict:
        """Apply post-HF correlation correction on top of the HF reference."""
        result = dict(state)
        result["correlation"] = state.get("post_hf_method", "MP2")
        result["method"] = state.get("post_hf_method", "MP2")
        return result


__all__ = [
    "HartreeFockMorphism",
    "PostHartreeFockMorphism",
]
