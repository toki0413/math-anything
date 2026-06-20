"""CFD / Navier-Stokes 态射链.

不可压近似 → Reynolds 分解 → 湍流模型封闭 → LES 滤波
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import Morphism


@dataclass
class IncompressibilityMorphism(Morphism):
    """不可压近似：压缩 NS → 不可压 NS.

    条件：Ma << 1（低马赫数）
    代价：丢失声波传播和热力学压力-密度耦合
    """

    name: str = "incompressibility"
    source_type: str = "CompressibleNavierStokes"
    target_type: str = "IncompressibleNavierStokes"
    condition: str = "Ma < 0.3"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "momentum_conservation",
            "viscous_dissipation",
            "vorticity_dynamics",
            "mass_conservation (incompressible form)",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "acoustic_waves",
            "density_variations",
            "thermodynamic_pressure_density_coupling",
            "energy_equation (decoupled from momentum)",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "pressure_as_Lagrange_multiplier",
            "elliptic_pressure_Poisson_equation",
            "divergence_free_constraint",
        ]
    )
    kernel_description: str = "Compressibility modes (sound waves)"

    @property
    def mathematical_form(self) -> str:
        return "∇·u = 0\n∂u/∂t + u·∇u = -∇p/ρ + ν∇²u + f"

    def apply(self, state: dict) -> dict:
        """Apply incompressibility approximation: enforce divergence-free velocity."""
        result = dict(state)
        result["compressible"] = False
        result["divergence_free"] = True
        result["acoustic_waves"] = False
        return result


@dataclass
class ReynoldsDecompositionMorphism(Morphism):
    """Reynolds 分解：全尺度 NS → RANS.

    u = ū + u'（平均+涨落），对 NS 取系综平均
    结果：出现了未知的 Reynold 应力张量 τ_ij = -⟨u_i' u_j'⟩
    """

    name: str = "reynolds_decomposition"
    source_type: str = "FullNavierStokes"
    target_type: str = "ReynoldsAveragedNavierStokes"
    condition: str = "flow_is_turbulent AND ensemble_average_is_meaningful"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "mean_flow_quantities",
            "mass_conservation_at_mean_level",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "turbulent_fluctuations",
            "instantaneous_flow_field",
            "deterministic_description",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "reynolds_stress_tensor (requires closure)",
            "turbulence_model_uncertainty",
            "closure_problem",
        ]
    )
    kernel_description: str = "Turbulent fluctuation field u'(x,t)"

    @property
    def mathematical_form(self) -> str:
        return "∂ū/∂t + ū·∇ū = -∇p̄/ρ + ν∇²ū - ∇·⟨u'⊗u'⟩\nwhere ⟨u'⊗u'⟩ = -ν_t (∇ū + ∇ū^T) + (2/3)kI  (Boussinesq)"

    def apply(self, state: dict) -> dict:
        """Apply Reynolds decomposition: split flow into mean and fluctuations."""
        result = dict(state)
        result["reynolds_decomposed"] = True
        result["mean_flow"] = True
        result["fluctuations"] = True
        return result


@dataclass
class TurbulenceModelClosureMorphism(Morphism):
    """湍流模型封闭态射：选择具体的封闭方式.

    RANS 方程组需要封闭 Reynold 应力。这就是"湍流模型"的本质。
    每个封闭方式是一个子态射，有不同的保真度和适用范围。
    """

    name: str = "turbulence_closure"
    source_type: str = "ReynoldsAveragedNavierStokes_Open"
    target_type: str = "ReynoldsAveragedNavierStokes_Closed"
    condition: str = ""

    model: str = "k_epsilon"  # "k_epsilon", "k_omega_sst", "rsm", "spalart_allmaras"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "mean_flow_equations_form",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "exact_reynolds_stress",
            "anisotropy_of_near_wall_turbulence (for isotropic eddy-viscosity models)",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "model_specific_calibration_constants",
            "wall_function_incompatibility (for high-Re models)",
        ]
    )
    kernel_description: str = (
        "Exact relationship between Reynolds stress and mean flow (replaced by eddy viscosity hypothesis)"
    )

    @property
    def mathematical_form(self) -> str:
        forms = {
            "k_epsilon": "ν_t = C_μ k²/ε,  with transport eqs for k and ε",
            "k_omega_sst": "ν_t = k/ω,  blended k-ω near wall + k-ε far field",
            "spalart_allmaras": "ν_t = ν̃ f_v1,  one transport eq for ν̃",
        }
        return forms.get(self.model, f"ν_t = f(mean_flow; {self.model})")

    def apply(self, state: dict) -> dict:
        """Apply turbulence model closure: select and activate a closure model."""
        result = dict(state)
        result["turbulence_modeled"] = True
        result["model_type"] = state.get("turbulence_model", "RANS")
        return result


@dataclass
class LESFilteringMorphism(Morphism):
    """大涡模拟滤波：NS → 空间滤波 NS.

    ũ(x) = ∫ G(x-y) u(y) dy
    解析大尺度，model 亚格子。
    """

    name: str = "les_filtering"
    source_type: str = "FullNavierStokes"
    target_type: str = "FilteredNavierStokes"
    condition: str = "filter_width >> Kolmogorov_scale"

    filter_type: str = "implicit_grid"  # "gaussian", "box", "spectral_cutoff"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "large_scale_structures",
            "energy_cascade_down_to_filter_scale",
            "transient_flow_features",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "subgrid_scales",
            "dissipation_range",
            "kolmogorov_scale_physics",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "subgrid_stress_tensor",
            "filter_width_dependency",
            "commutation_error (if non-uniform filter)",
        ]
    )
    kernel_description: str = "Subgrid scales < Δx"

    @property
    def mathematical_form(self) -> str:
        return "∂ũ/∂t + ũ·∇ũ = -∇p̃/ρ + ν∇²ũ - ∇·τ_SGS\nτ_SGS ≈ -2 ν_SGS S̃_ij  (Smagorinsky)  or dynamic / WALE / etc."

    def apply(self, state: dict) -> dict:
        """Apply LES spatial filtering: separate resolved and subgrid scales."""
        result = dict(state)
        result["filtered"] = True
        result["filter_width"] = state.get("filter_width", 0.01)
        return result


__all__ = [
    "IncompressibilityMorphism",
    "ReynoldsDecompositionMorphism",
    "TurbulenceModelClosureMorphism",
    "LESFilteringMorphism",
]
