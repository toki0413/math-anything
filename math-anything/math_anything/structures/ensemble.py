"""统计系综结构家族。

EnsembleProblem = 在概率测度空间上的采样/推断。

统计系综是"给确定性结构加一层概率"：
  - 平衡态系综：在相空间上按分布采样（MD 的 NVT/NPT）
  - 动力学随机过程：Fokker-Planck 或 Master Equation
  - 贝叶斯推断：用数据更新先验 → 后验（UQ, 逆问题）
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata
from .properties import StructuralInvariant


@dataclass
class EnsembleProblem(AbstractMathematicalStructure):
    """统计系综基类：在测度空间上的问题.

    基本设定：
      有一个确定性结构 S（如 HamiltonianSystem），
      在 S 的状态空间上定义一个概率测度 P，
      系综问题是关于 P 的问题（采样、期望、推断）。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ENSEMBLE,
            name="Statistical Ensemble Problem",
            canonical_form="⟨A⟩ = ∫ A(x) P(x) dx",
            description="Expected values and sampling over a probability measure on state space",
        )
    )
    underlying_structure: str = ""  # 底层确定性结构类型
    state_space_dim: int = 0

    @property
    def function_space(self) -> str:
        return "L¹(Γ, P) — integrable functions on phase space"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return []


@dataclass
class EquilibriumEnsemble(EnsembleProblem):
    """平衡态系综：配分函数 + 正则/微正则/巨正则.

    实例：
      - 正则系综（NVT）：P(x) ∝ exp(-H(x)/k_B T)
      - 微正则（NVE）：P(x) ∝ δ(H(x)-E)
      - 巨正则（μVT）：P(x,N) ∝ exp(-(H(x)-μN)/k_B T)
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ENSEMBLE,
            name="Equilibrium Ensemble",
            canonical_form="P(x) ∝ exp(-β H(x))  (canonical)",
            description="Statistical equilibrium: Boltzmann-Gibbs distribution over microstates",
        )
    )
    ensemble_type: str = "canonical"  # "microcanonical", "canonical", "grand_canonical"
    temperature: float | None = None
    partition_function: str = "Z = ∫ exp(-βH) dΓ"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = [
            StructuralInvariant(
                name="detailed_balance",
                expression="P(i→j) exp(-β E_i) = P(j→i) exp(-β E_j)",
                theorem="Detailed balance → stationary distribution",
                affected_quantities=["transition_probabilities", "equilibrium"],
            ),
            StructuralInvariant(
                name="fluctuation_dissipation",
                expression="⟨δA²⟩ ∝ k_B T χ_A  (fluctuation-dissipation theorem)",
                theorem="Fluctuation-Dissipation Theorem (linear response)",
                affected_quantities=["fluctuations", "susceptibility"],
            ),
        ]

        if self.ensemble_type == "microcanonical":
            invariants.append(
                StructuralInvariant(
                    name="energy_shell",
                    expression="H(x) = E (constant energy hypersurface)",
                    theorem="Microcanonical ensemble: uniform measure on energy shell",
                    affected_quantities=["energy", "entropy"],
                )
            )

        return invariants


@dataclass
class KineticProcess(EnsembleProblem):
    """动力学随机过程：Master Equation / Fokker-Planck.

    实例：
      - 化学反应速率理论
      - 扩散过程
      - 相变动力学（成核与生长）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ENSEMBLE,
            name="Kinetic Process",
            canonical_form="∂P/∂t = L P  (master equation)",
            description="Time evolution of a probability distribution",
        )
    )
    process_type: str = "markov"  # "markov", "non_markov"
    transition_rates: str = ""

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="probability_conservation",
                expression="d/dt ∫ P(x,t) dx = 0",
                theorem="Kolmogorov forward equation conserves probability",
                affected_quantities=["probability", "normalization"],
            ),
            StructuralInvariant(
                name="entropy_production",
                expression="dS/dt ≥ 0 (H-theorem for Markov processes)",
                theorem="H-theorem / Second Law of Thermodynamics",
                affected_quantities=["entropy"],
            ),
        ]


@dataclass
class BayesianInferenceProblem(EnsembleProblem):
    """贝叶斯推断：先验 + 似然 → 后验.

    P(θ|D) ∝ P(D|θ) × P(θ)

    实例：
      - 参数标定/逆问题（UQ）
      - 模型选择（贝叶斯因子）
      - 实验设计

    关键：底层结构是"参数→可观测量"的映射，
    概率是叠加在该映射上的不确定性。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ENSEMBLE,
            name="Bayesian Inference Problem",
            canonical_form="P(θ|D) ∝ P(D|θ) P(θ)",
            description="Update prior beliefs about parameters using observed data",
        )
    )
    parameters: list[str] = field(default_factory=list)
    observables: list[str] = field(default_factory=list)
    prior_type: str = ""

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="bayes_theorem",
                expression="P(θ|D) = P(D|θ) P(θ) / P(D)",
                theorem="Bayes Theorem (conditional probability)",
                affected_quantities=["posterior", "evidence"],
            ),
            StructuralInvariant(
                name="information_gain",
                expression="D_KL(P_post || P_prior) ≥ 0",
                theorem="Kullback-Leibler divergence: data always informative (or neutral)",
                affected_quantities=["information", "uncertainty_reduction"],
            ),
            StructuralInvariant(
                name="consistency",
                expression="P_post → δ(θ - θ_true) as N → ∞ (under regularity)",
                theorem="Bernstein-von Mises theorem (Bayesian consistency)",
                affected_quantities=["posterior_concentration"],
            ),
        ]
