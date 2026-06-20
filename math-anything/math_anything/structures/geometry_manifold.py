"""光滑流形与切丛结构。

微分拓扑基础：流形、切丛、向量场、张量场、余向量场。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata
from .properties import StructuralInvariant


@dataclass
class SmoothManifold(AbstractMathematicalStructure):
    """光滑流形：局部同胚于 ℝ^dim 的 Hausdorff 第二可数空间.

    Attributes:
        dim: 拓扑维度（拓扑不变量）
        orientable: 是否可定向（微分同胚不变量）
        differentiable_structure: 微分结构类型（如 "C∞", "C^k"）
        atlas: 图册 {(U_α, φ_α)}，通常以名称列表表示
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Smooth Manifold",
            canonical_form="M: locally ℝ^dim with C^∞ transition maps",
            description="Smooth manifold: locally Euclidean Hausdorff second-countable topological space",
        )
    )
    dim: int = 3
    orientable: bool = False
    differentiable_structure: str = "C^∞"
    atlas: list[str] = field(default_factory=list)

    @property
    def function_space(self) -> str:
        return f"C^∞(M) — smooth functions on {self.dim}-manifold"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="dimension_topological_invariant",
                expression="dim M is invariant under homeomorphism",
                theorem="Invariance of Domain (Brouwer)",
                affected_quantities=["dimension", "topology"],
            ),
            StructuralInvariant(
                name="orientability_diffeomorphism_invariant",
                expression="M orientable ⇒ any diffeomorphic image is orientable",
                theorem="Orientability is a diffeomorphism invariant (preserved under pushforward of volume form)",
                affected_quantities=["orientation", "volume_form"],
            ),
        ]


@dataclass
class TangentBundle(AbstractMathematicalStructure):
    """切丛：π: TM → M，纤维 = ℝ^{dim M}.

    Attributes:
        base_dim: 底流形的维数
        fiber_dim: 纤维维数（通常等于 base_dim）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Tangent Bundle",
            canonical_form="π: TM → M,  fiber = ℝ^{dim M}",
            description="Vector bundle of tangent vectors over a smooth manifold",
        )
    )
    base_dim: int = 3
    fiber_dim: int = 3

    @property
    def total_dimension(self) -> int:
        return 2 * self.base_dim

    @property
    def function_space(self) -> str:
        return f"Γ(TM) — sections of tangent bundle over {self.base_dim}-manifold"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="dim_TM_equals_2_dim_M",
                expression=f"dim(TM) = 2 dim(M) = {2 * self.base_dim}",
                theorem="Dimension of tangent bundle (total space)",
                affected_quantities=["dimension", "tangent_space"],
            ),
            StructuralInvariant(
                name="local_trivialization",
                expression="∀p ∈ M, ∃U ∋ p: π^{-1}(U) ≅ U × ℝ^{dim M}",
                theorem="Local triviality of vector bundles",
                affected_quantities=["chart", "trivialization"],
            ),
        ]


@dataclass
class VectorField(AbstractMathematicalStructure):
    """向量场：X: M → TM 满足 π∘X = id_M.

    Attributes:
        manifold_dim: 底流形维数
        is_complete: 是否完备（积分曲线全局存在）
        is_hamiltonian: 是否从 Hamilton 函数生成
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Vector Field",
            canonical_form="X: M → TM,  π∘X = id_M",
            description="Section of the tangent bundle — infinitesimal generator of diffeomorphisms",
        )
    )
    manifold_dim: int = 3
    is_complete: bool = False
    is_hamiltonian: bool = False

    @property
    def function_space(self) -> str:
        return f"𝔛(M) — Lie algebra of vector fields on {self.manifold_dim}-manifold"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="lie_bracket_antisymmetry",
                expression="[X, Y] = -[Y, X]",
                theorem="Lie bracket algebra axioms",
                affected_quantities=["vector_field", "lie_bracket"],
            ),
            StructuralInvariant(
                name="jacobi_identity",
                expression="[X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0",
                theorem="Jacobi identity (Lie algebra axiom)",
                affected_quantities=["vector_field", "lie_bracket"],
            ),
        ]


@dataclass
class CovectorField(AbstractMathematicalStructure):
    """余向量场（1-形式）：ω_x ∈ T^*_x M.

    Attributes:
        manifold_dim: 底流形维数
        is_exact: ω = df 是否为恰当形式
        is_closed: dω = 0 是否为闭形式
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Covector Field (1-form)",
            canonical_form="ω: M → T*M,  ω_x ∈ T*_x M",
            description="Differential 1-form — pullback under smooth maps",
        )
    )
    manifold_dim: int = 3
    is_exact: bool = False
    is_closed: bool = False

    @property
    def function_space(self) -> str:
        return f"Ω¹(M) — 1-forms on {self.manifold_dim}-manifold"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="exterior_derivative_nilpotent",
                expression="d² = d∘d = 0",
                theorem="Exterior derivative is a differential (cohomology)",
                affected_quantities=["differential_form", "exterior_derivative"],
            ),
            StructuralInvariant(
                name="poincare_lemma",
                expression="dω = 0 ⇒ locally ω = df (closed ⇒ locally exact)",
                theorem="Poincaré Lemma (on contractible domains)",
                affected_quantities=["closed_form", "exact_form"],
            ),
        ]


@dataclass
class TensorField(AbstractMathematicalStructure):
    """张量场：multilinear map (T^*)^p × T^q → ℝ.

    Attributes:
        p: 反变阶数（上指标数）
        q: 协变阶数（下指标数）
        manifold_dim: 底流形维数
        total_components: p+q 阶张量的独立分量数 = dim^{p+q}
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Tensor Field",
            canonical_form="T^{(p,q)}: (T*M)^p × (TM)^q → C^∞(M) multilinear",
            description="Multilinear tensor field of type (p,q) over a smooth manifold",
        )
    )
    p: int = 1
    q: int = 0
    manifold_dim: int = 3

    @property
    def total_components(self) -> int:
        return self.manifold_dim ** (self.p + self.q)

    @property
    def function_space(self) -> str:
        return f"Γ(T^{{({self.p},{self.q})}}M) — tensor fields on {self.manifold_dim}-manifold"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="tensor_transformation_law",
                expression="T'^{a...}_{b...} = (∂x'^a/∂x^c)...(∂x^d/∂x'^b)... T^{c...}_{d...}",
                theorem="Tensor transformation law under coordinate change",
                affected_quantities=["components", "coordinate_system"],
            ),
        ]
