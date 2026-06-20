"""DFT / 电子结构态射链.

Born-Oppenheimer 近似 → Kohn-Sham 映射 → 平面波截断 → SCF 迭代 → 交换关联近似
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import Morphism


@dataclass
class BornOppenheimerApproximation(Morphism):
    """Born-Oppenheimer 近似：解耦电子和核运动.

    全量子多体 Schrödinger → 电子 Schrödinger（核作为固定参数）
    """

    name: str = "born_oppenheimer"
    source_type: str = "FullQuantumManyBody"
    target_type: str = "ElectronicSchrodingerWithParametricNuclei"
    condition: str = "nuclear_mass >> electron_mass"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "electronic_structure",
            "potential_energy_surface_concept",
            "chemical_bonding",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "nuclear_quantum_effects",
            "nonadiabatic_coupling",
            "vibronic_coupling",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "parametric_dependence_on_nuclear_coordinates",
            "independent_electronic_and_nuclear_energy_scales",
        ]
    )
    kernel_description: str = "Correlated electron-nuclear motion (non-BO coupling terms)"

    @property
    def mathematical_form(self) -> str:
        return "Ĥ_e(R) ψ_e(r;R) = E_e(R) ψ_e(r;R)\nĤ_n(R) χ(R) = [T_n + E_e(R)] χ(R) = E_total χ(R)"

    def apply(self, state: dict) -> dict:
        """Apply Born-Oppenheimer approximation: separate electronic and nuclear DOF."""
        result = dict(state)
        result["nuclear_quantum_effects"] = False
        result["vibronic_coupling"] = False
        result["adiabatic_approximation"] = True
        result["electronic_nuclear_separated"] = True
        return result


@dataclass
class KohnShamMapping(Morphism):
    """Kohn-Sham 映射：相互作用多电子 → 非相互作用辅助系统.

    全电子 Schrödinger → Kohn-Sham 轨道方程
    """

    name: str = "kohn_sham"
    source_type: str = "InteractingManyElectron"
    target_type: str = "NonInteractingKS_Orbitals"
    condition: str = "v-representability of the density"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "electron_density_n(r)",
            "total_energy",
            "ionization_potential (exact KS)",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "many_body_wavefunction",
            "explicit_electron_correlation",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "exchange_correlation_functional_uncertainty",
            "self_interaction_error (for approximate xc)",
            "single_determinant_representation",
        ]
    )
    kernel_description: str = "Exchange-correlation hole (all many-body correlation compressed into E_xc[n])"

    @property
    def mathematical_form(self) -> str:
        return "[-½∇² + V_ext(r) + V_H[n](r) + V_xc[n](r)] ψ_i(r) = ε_i ψ_i(r)\nn(r) = Σ f_i |ψ_i(r)|²"

    def apply(self, state: dict) -> dict:
        """Apply Kohn-Sham mapping: replace many-body with non-interacting particles."""
        result = dict(state)
        result["explicit_correlation"] = False
        result["self_consistent"] = True
        result["n_orbitals"] = state.get("n_electrons", 1) // 2 + state.get("n_electrons", 1) % 2
        return result


@dataclass
class PlaneWaveTruncation(Morphism):
    """平面波截断：无限维 Hilbert 空间 → 有限维子空间.

    L²(ℝ³) → span{exp(iG·r) : |G|²/2 < E_cut}
    """

    name: str = "plane_wave_truncation"
    source_type: str = "KohnSham_Full"
    target_type: str = "KohnSham_Truncated"
    condition: str = "E_cut > 0"

    encut: float = 520  # eV
    is_orthogonal_projection: bool = True  # 截断是正交投影

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "self_adjointness (orthogonal projection preserves it)",
            "variational_principle (Rayleigh-Ritz)",
            "low_frequency_physics",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "basis_completeness",
            "high_frequency_components",
        ]
    )

    def get_invariants_lost(self) -> list[str]:
        """考虑 ENCUT 值的条件丢失."""
        lost = list(self.invariants_lost)
        if self.encut < 400:
            lost.append("variational_upper_bound_strictness - Pulay stress")
        return lost

    @property
    def invariants_introduced(self) -> list[str]:
        return [
            "truncation_error proportional to exp(-alpha E_cut)",
            "basis_set_superposition_error for small cells",
        ]

    kernel_description: str = "{|G|^2/2 > E_cut} - truncated Fourier subspace"

    @property
    def mathematical_form(self) -> str:
        return "psi(r) = sum_{|G|^2/2 < " + str(self.encut) + " eV} c_G exp(iG*r)"

    @property
    def _invariants_introduced(self) -> list[str]:
        return self.invariants_introduced

    # HACK: provide setters so dataclass __init__ doesn't fail
    @invariants_introduced.setter
    def invariants_introduced(self, value: list[str]) -> None:
        pass

    def apply(self, state: dict) -> dict:
        """Apply plane wave basis truncation."""
        result = dict(state)
        ecut = state.get("ecutwfc", 50.0)
        result["basis_completeness"] = False
        result["ecutwfc"] = ecut
        result["n_pw"] = int(4 * 3.14159 * (2 * ecut) ** 1.5 / 3)
        return result


@dataclass
class SCFIterationMorphism(Morphism):
    """SCF 迭代态射：从不精确密度到精确密度的不动点迭代.

    n^{(k)} → build H[n^{(k)}] → solve → n^{(k+1)} → check convergence
    """

    name: str = "scf_iteration"
    source_type: str = "KohnSham_Setup"
    target_type: str = "KohnSham_Converged"
    condition: str = "mixing_parameters_allow_convergence"

    mixing_scheme: str = "linear"  # "linear", "kerker", "pulay", "broyden"
    max_iterations: int = 60
    convergence_threshold: float = 1e-6

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "Kohn_Sham_equations_form",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "global_convergence_guarantee",
            "monotonic_energy_decrease",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "convergence_criterion_dependence",
            "mixing_parameter_sensitivity",
            "charge_sloshing_risk",
        ]
    )
    kernel_description: str = "Non-convergent initial guesses (Banach fixed point needs contraction)"

    @property
    def mathematical_form(self) -> str:
        return (
            f"n^{{(k+1)}} = (1-α) n^{{(k)}} + α F[n^{{(k)}}]\n"
            f"terminate when |E^{{(k+1)}} - E^{{(k)}}| < {self.convergence_threshold}"
        )

    def apply(self, state: dict) -> dict:
        """Apply one SCF iteration step."""
        result = dict(state)
        result["scf_iteration"] = state.get("scf_iteration", 0) + 1
        prev_change = state.get("density_change", 1.0)
        result["density_change"] = prev_change * 0.5
        result["converged"] = result["density_change"] < 1e-6
        return result


@dataclass
class ExchangeCorrelationApproximation(Morphism):
    """交换关联泛函近似.

    E_xc[exact] → E_xc[approximate] (LDA, GGA, meta-GGA, hybrid, ...)
    """

    name: str = "xc_approximation"
    source_type: str = "ExactXCFunctional"
    target_type: str = "ApproximateXCFunctional"
    condition: str = ""

    functional: str = "PBE"  # "LDA", "PBE", "SCAN", "HSE06", "PBE0"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "electron_density_as_basic_variable",
            "Kohn_Sham_framework",
            "scaling_relations (for exact constraints satisfied by functional)",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "exact_exchange_correlation",
            "derivative_discontinuity",
            "self_interaction_cancellation (for semi-local functionals)",
            "long-range_correlation (for semi-local functionals)",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "functional_specific_errors",
            "band_gap_underestimation (semi-local)",
        ]
    )
    kernel_description: str = "Exact xc hole shape (approximated by model hole)"

    @property
    def mathematical_form(self) -> str:
        return f"E_xc[n] ≈ E_xc^{self.functional}[n]"

    def apply(self, state: dict) -> dict:
        """Apply XC approximation."""
        result = dict(state)
        result["exact_xc"] = False
        result["xc_functional"] = state.get("xc_functional", "PBE")
        return result


__all__ = [
    "BornOppenheimerApproximation",
    "KohnShamMapping",
    "PlaneWaveTruncation",
    "SCFIterationMorphism",
    "ExchangeCorrelationApproximation",
]
