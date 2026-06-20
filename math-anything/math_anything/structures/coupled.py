"""耦合系统结构家族。

CoupledSystem = 两个或多个结构通过界面交换信息。

耦合是计算材料科学中最复杂也最有价值的数学结构。
数学上，耦合 = 将两个独立结构通过一个耦合态射连接为新的整体结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata
from .properties import StructuralInvariant


@dataclass
class CoupledSystem(AbstractMathematicalStructure):
    """耦合系统基类：多个数学结构的联合.

    属性：
      - subsystems: 参与耦合的子结构
      - coupling_interface: 子结构之间交换的物理量
      - coupling_type: 耦合的数学类型
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.COUPLED,
            name="Coupled System",
            canonical_form="{L₁(u₁, u₂) = f₁, L₂(u₁, u₂) = f₂}",
            description="Multiple mathematical structures coupled through shared interfaces",
        )
    )
    subsystems: list[str] = field(default_factory=list)
    coupling_quantity: str = ""
    coupling_dimension: int = 3

    @property
    def function_space(self) -> str:
        return "Product of component function spaces"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return []


@dataclass
class SequentialCoupling(CoupledSystem):
    """单向耦合：结构 A 的输出 → 结构 B 的输入.

    A → B，无反馈。

    实例：
      - DFT → 力场拟合 → MD（电子结构→经典原子论）
      - 电磁场 → 热源（焦耳热）
      - 湍流 RANS → 声学类比
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.COUPLED,
            name="Sequential Coupling",
            canonical_form="A → B (one-way, no feedback)",
            description="Unidirectional coupling: output of A feeds into B",
        )
    )

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="causality_preserved",
                expression="B depends on A, A is independent of B",
                theorem="Sequential coupling → DAG of dependencies",
                affected_quantities=["order", "dependencies"],
            ),
            StructuralInvariant(
                name="information_loss",
                expression="Information from A is projected onto input space of B",
                theorem="Information theory: dimension reduction",
                affected_quantities=["coupling_accuracy"],
            ),
        ]


@dataclass
class BidirectionalCoupling(CoupledSystem):
    """双向耦合：结构 A 和 B 互相反馈.

    A ⇌ B。

    实例：
      - 流固耦合（FSI）：流体压力⇌固体位移
      - 热-力耦合：温度⇌热应力
      - 电磁-力耦合：Lorentz力⇌变形
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.COUPLED,
            name="Bidirectional Coupling",
            canonical_form="A ⇌ B (two-way, with mutual feedback)",
            description="Bidirectional coupling where both systems influence each other",
        )
    )

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="coupled_fixed_point",
                expression="(u_A, u_B) is a fixed point of the coupled system",
                theorem="Coupled fixed point problem → staggered or monolithic solver",
                affected_quantities=["convergence", "coupling_iterations"],
            ),
            StructuralInvariant(
                name="partitioned_consistency",
                expression="Interface conditions must be satisfied simultaneously",
                theorem="Coupled problem consistency conditions",
                affected_quantities=["interface", "traction", "velocity"],
            ),
        ]


@dataclass
class MultiscaleCoupling(CoupledSystem):
    """多尺度耦合：微观→介观→宏观的信息传递.

    实例：
      - 量子/原子论/连续介质（QM/MM/FEM）
      - 位错动力学→晶体塑性
      - 分子动力学→连续介质（AtC coupling）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.COUPLED,
            name="Multiscale Coupling",
            canonical_form="Micro → Meso → Macro (hierarchical information transfer)",
            description="Coupling across spatial/temporal scales with scale separation",
        )
    )
    scales: list[str] = field(default_factory=list)  # ["quantum", "atomistic", "continuum"]
    scale_separation: bool = True

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = [
            StructuralInvariant(
                name="scale_separation",
                expression="L_micro ≪ L_meso ≪ L_macro (if scale separation holds)",
                theorem="Asymptotic analysis: scale separation → homogenization",
                affected_quantities=["effective_properties", "homogenization_error"],
            ),
            StructuralInvariant(
                name="hill_mandel_condition",
                expression="⟨σ:ε⟩ = ⟨σ⟩:⟨ε⟩ (energy consistency across scales)",
                theorem="Hill-Mandel macrohomogeneity condition",
                condition="self.scale_separation",
                affected_quantities=["stress", "strain", "energy_density"],
            ),
        ]

        if not self.scale_separation:
            invariants.append(
                StructuralInvariant(
                    name="concurrent_coupling_required",
                    expression="No scale separation → concurrent coupling needed",
                    theorem="Multiscale methods: overlapping domains or handshake regions",
                    affected_quantities=["coupling_method"],
                )
            )

        return invariants


@dataclass
class MultiphysicsCoupling(CoupledSystem):
    """多物理场耦合：同一时空尺度上不同物理场的相互作用.

    实例：
      - COMSOL 的任意多物理场
      - 热-电-力耦合（热电材料）
      - 流-热-化耦合（反应流）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.COUPLED,
            name="Multiphysics Coupling",
            canonical_form="{L_i(u₁, ..., u_n) = f_i} for i = 1...n",
            description="Multiple physics interacting at the same scale",
        )
    )
    physics_modules: list[str] = field(default_factory=list)  # ["solid", "heat", "EM"]
    fully_coupled: bool = True  # 全耦合 vs 分离求解

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="onsager_reciprocity",
                expression="L_ij = L_ji (cross-coupling coefficients symmetric)",
                theorem="Onsager reciprocal relations",
                condition="len(self.physics_modules) >= 2",
                affected_quantities=["cross_coupling", "transport_coefficients"],
            ),
            StructuralInvariant(
                name="block_structure",
                expression="System matrix has block structure K_{ij} for physics i,j",
                theorem="Multiphysics discretization: block preconditioners",
                affected_quantities=["stiffness_matrix", "preconditioner"],
            ),
        ]
