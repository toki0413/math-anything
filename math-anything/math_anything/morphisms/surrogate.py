"""ML 代理与相场态射.

ML 代理态射：用神经网络替换物理模型
扩散界面态射：尖锐界面 → 扩散界面
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import Morphism


@dataclass
class MLSurrogateMorphism(Morphism):
    """ML 代理态射：用神经网络替换物理模型.

    E_DFT(R) → E_ML(R; θ)

    关键概念：这不是传统近似（physically motivated），
    而是代理建模（data-driven surrogate）。
    """

    name: str = "ml_surrogate"
    source_type: str = "AbInitioPotentialEnergySurface"
    target_type: str = "MLSurrogateModel"
    category: str = "surrogate"
    is_surjective: bool = False

    architecture: str = "GAP"  # "GAP", "SNAP", "DeePMD", "MACE", "CHGNet", "NequIP"

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "total_energy",
            "forces (via autodiff)",
            "stress (if trained)",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "electronic_structure",
            "charge_density",
            "band_structure",
            "exact_long_range_behavior",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "training_data_bias",
            "extrapolation_risk",
            "model_uncertainty",
            "smoothness_constraints",
        ]
    )
    kernel_description: str = "Electronic degrees of freedom (replaced by learned functional form)"

    @property
    def mathematical_form(self) -> str:
        return f"E(R) ≈ f_θ({{descriptors(R_i)}})\nF_i = -∂E/∂R_i (via autodiff)\nArchitecture: {self.architecture}"

    def apply(self, state: dict) -> dict:
        """Apply ML surrogate: replace ab-initio evaluation with a trained model."""
        result = dict(state)
        result["ab_initio"] = False
        result["surrogate_model"] = state.get("architecture", self.architecture)
        result["requires_training"] = True
        return result


@dataclass
class DiffuseInterfaceMorphism(Morphism):
    """扩散界面态射：尖锐界面 → 扩散界面.

    用连续场 φ(x) 描述界面，代替 Γ 的显式追踪。
    """

    name: str = "diffuse_interface"
    source_type: str = "SharpInterfaceProblem"
    target_type: str = "PhaseFieldProblem"
    condition: str = "interface_width << domain_size"

    interface_width: float = 1e-9  # 界面宽度参数

    invariants_kept: list[str] = field(
        default_factory=lambda: [
            "bulk_free_energies",
            "total_mass (conserved order parameters)",
            "curvature_driven_interface_motion",
        ]
    )
    invariants_lost: list[str] = field(
        default_factory=lambda: [
            "sharp_interface_position",
            "exact_gibbs_thomson_condition",
        ]
    )
    invariants_introduced: list[str] = field(
        default_factory=lambda: [
            "interface_width_parameter",
            "asymptotic_limit_to_sharp_interface (ε→0)",
            "gradient_energy_penalty",
        ]
    )
    kernel_description: str = "Sharp interface Γ (replaced by smooth transition)"

    @property
    def mathematical_form(self) -> str:
        return "F[φ] = ∫ [½κ|∇φ|² + f(φ)] dV\n∂φ/∂t = -L δF/δφ  (Allen-Cahn)\n∂φ/∂t = ∇·(M∇ δF/δφ)  (Cahn-Hilliard)"

    def apply(self, state: dict) -> dict:
        """Apply diffuse-interface mapping: replace sharp interface with smooth phase field."""
        result = dict(state)
        result["sharp_interface"] = False
        result["interface_width"] = state.get("interface_width", self.interface_width)
        result["diffuse"] = True
        return result


__all__ = [
    "MLSurrogateMorphism",
    "DiffuseInterfaceMorphism",
]
