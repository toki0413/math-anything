"""MD / 经典力学态射链.

经典力学极限 → 力场/经验势近似
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import Morphism


@dataclass
class ClassicalLimitMorphism(Morphism):
    """经典力学极限：量子 → 经典.

    Schrödinger 方程 → Newton 方程（ħ → 0, kT >> ħω）
    """

    name: str = "classical_limit"
    source_type: str = "QuantumDynamics"
    target_type: str = "ClassicalDynamics"
    condition: str = "kT >> hbar * omega_characteristic"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "total_energy",
            "center_of_mass_motion",
            "translational_degrees_of_freedom",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "zero_point_energy",
            "tunneling",
            "quantum_interference",
            "discrete_energy_levels",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "continuous_phase_space",
            "deterministic_trajectories",
        ]
    )
    kernel_description: str = "Quantum coherence and wave nature"

    @property
    def mathematical_form(self) -> str:
        return "m d²r/dt² = F(r)  ←  lim_{ħ→0} iħ ∂ψ/∂t = Ĥψ"

    def apply(self, state: dict) -> dict:
        """Apply classical limit: disable quantum effects and enable Newtonian dynamics."""
        result = dict(state)
        result["quantum_effects"] = False
        result["classical_mechanics"] = True
        return result


@dataclass
class ForceFieldMorphism(Morphism):
    """力场/经验势近似：第一性原理相互作用 → 经验解析函数.

    E_QM(r₁...r_N) → E_FF(r₁...r_N) = Σ_bonds + Σ_angles + Σ_torsions + Σ_nonbonded
    """

    name: str = "force_field"
    source_type: str = "AbInitioPotential"
    target_type: str = "EmpiricalForceField"
    condition: str = "force_field_parameters_available"

    force_field: str = "LJ"  # "LJ", "EAM", "ReaxFF", "COMPASS", "CHARMM"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "coarse_energy_landscape",
            "equilibrium_structure (if well-parameterized)",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "electronic_structure_information",
            "bond_breaking_accuracy (unless reactive FF)",
            "charge_transfer",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "empirical_parameter_uncertainty",
            "functional_form_rigidity",
            "cutoff_artifacts",
        ]
    )
    kernel_description: str = "Electronic degrees of freedom"

    @property
    def mathematical_form(self) -> str:
        return (
            "E = Σ k_b(r - r₀)²  (bonds)\n"
            "  + Σ k_θ(θ - θ₀)²  (angles)\n"
            "  + Σ V_n[1 + cos(nφ - δ)]  (torsions)\n"
            "  + Σ 4ε[(σ/r)¹² - (σ/r)⁶]  (LJ nonbonded)\n"
            "  + Σ qᵢqⱼ/(4πε₀r)  (Coulomb)"
        )

    def apply(self, state: dict) -> dict:
        """Apply force field approximation: replace ab-initio potential with empirical form."""
        result = dict(state)
        result["ab_initio"] = False
        result["force_field"] = state.get("force_field", "Lennard-Jones")
        return result


__all__ = [
    "ClassicalLimitMorphism",
    "ForceFieldMorphism",
]
